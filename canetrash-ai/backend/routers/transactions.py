"""
transactions.py
================
CRUD + workflow API for truck inspection transactions.

Endpoints:
    POST   /transactions/                  - create a pending transaction (ANPR intake)
    GET    /transactions/                  - list transactions (paginated, filterable)
    GET    /transactions/{transaction_id}  - fetch a single transaction
    POST   /transactions/{transaction_id}/scan  - run the full AI scan pipeline
    DELETE /transactions/{transaction_id}  - remove a transaction (dev/test utility)
"""

from __future__ import annotations

import datetime as dt
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import Settings, get_settings
from backend.database import get_db
from backend.models import Transaction, TransactionStatus
from backend.schemas import (
    TransactionCreate,
    TransactionListResponse,
    TransactionRead,
    TransactionScanTrigger,
)
from backend.services.pipeline_runner import PipelineRunner

logger = logging.getLogger("canetrash.backend.routers.transactions")

router = APIRouter(prefix="/transactions", tags=["transactions"])

# The pipeline owns AI engine instances (potentially model weights), so it is
# constructed once per process and reused across requests.
_pipeline_runner: PipelineRunner | None = None


def get_pipeline_runner(settings: Settings = Depends(get_settings)) -> PipelineRunner:
    global _pipeline_runner
    if _pipeline_runner is None:
        _pipeline_runner = PipelineRunner(settings)
    return _pipeline_runner


def _generate_transaction_code() -> str:
    now = dt.datetime.now(dt.timezone.utc)
    return f"TRX-{now.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"


@router.post("/", response_model=TransactionRead, status_code=status.HTTP_201_CREATED)
async def create_transaction(
    payload: TransactionCreate, db: AsyncSession = Depends(get_db)
) -> Transaction:
    """Register a new truck arrival (post-ANPR) as a PENDING transaction."""
    transaction = Transaction(
        id=str(uuid.uuid4()),
        transaction_code=_generate_transaction_code(),
        supplier_id=payload.supplier_id,
        plate_number=payload.plate_number,
        base_price_per_ton=payload.base_price_per_ton,
        status=TransactionStatus.PENDING,
    )
    db.add(transaction)
    await db.flush()
    await db.refresh(transaction)
    return transaction


@router.get("/", response_model=TransactionListResponse)
async def list_transactions(
    db: AsyncSession = Depends(get_db),
    status_filter: TransactionStatus | None = Query(default=None, alias="status"),
    plate_number: str | None = Query(default=None),
    limit: int = Query(default=25, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> TransactionListResponse:
    """List transactions, most recent first, with optional status/plate filters."""
    query = select(Transaction)
    count_query = select(func.count()).select_from(Transaction)

    if status_filter is not None:
        query = query.where(Transaction.status == status_filter)
        count_query = count_query.where(Transaction.status == status_filter)
    if plate_number:
        query = query.where(Transaction.plate_number.ilike(f"%{plate_number}%"))
        count_query = count_query.where(Transaction.plate_number.ilike(f"%{plate_number}%"))

    query = query.order_by(Transaction.created_at.desc()).limit(limit).offset(offset)

    total = (await db.execute(count_query)).scalar_one()
    rows = (await db.execute(query)).scalars().all()

    return TransactionListResponse(total=total, items=list(rows))


@router.get("/{transaction_id}", response_model=TransactionRead)
async def get_transaction(transaction_id: str, db: AsyncSession = Depends(get_db)) -> Transaction:
    transaction = await db.get(Transaction, transaction_id)
    if transaction is None:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return transaction


@router.post("/{transaction_id}/scan", response_model=TransactionRead)
async def trigger_scan(
    transaction_id: str,
    payload: TransactionScanTrigger,
    db: AsyncSession = Depends(get_db),
    runner: PipelineRunner = Depends(get_pipeline_runner),
) -> Transaction:
    """
    Runs the full 15-second gantry scan pipeline (segmentation + LiDAR fusion
    + SLM refaction decision) for a PENDING transaction and persists results.
    """
    transaction = await db.get(Transaction, transaction_id)
    if transaction is None:
        raise HTTPException(status_code=404, detail="Transaction not found")

    if transaction.status not in (TransactionStatus.PENDING, TransactionStatus.ERROR):
        raise HTTPException(
            status_code=409,
            detail=f"Transaction is in status '{transaction.status.value}' and cannot be re-scanned.",
        )

    try:
        transaction = await runner.run_full_scan(
            db=db,
            transaction=transaction,
            gross_weight_kg_override=payload.gross_weight_kg,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Scan pipeline raised an unhandled exception")
        raise HTTPException(status_code=500, detail=f"Scan pipeline failed: {exc}") from exc

    await db.refresh(transaction)
    return transaction


@router.delete("/{transaction_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_transaction(transaction_id: str, db: AsyncSession = Depends(get_db)) -> None:
    """Development/testing utility -- not typically exposed in production."""
    transaction = await db.get(Transaction, transaction_id)
    if transaction is None:
        raise HTTPException(status_code=404, detail="Transaction not found")
    await db.delete(transaction)
    await db.flush()
