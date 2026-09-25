"""
analytics.py
============
Mill-wide quality statistics and KPI summaries for the operator dashboard
and public transparency screen.
"""

from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_db
from backend.models import Transaction, TransactionStatus
from backend.schemas import AnalyticsResponse, KPISummary, TrashClassAverage

router = APIRouter(prefix="/analytics", tags=["analytics"])

TRASH_CLASSES = ["dry_leaf", "green_top", "soil_dirt", "rotten_stalk"]


@router.get("/summary", response_model=AnalyticsResponse)
async def analytics_summary(
    db: AsyncSession = Depends(get_db),
    since_hours: int = Query(default=24, ge=1, le=24 * 90),
) -> AnalyticsResponse:
    """
    Returns aggregate KPIs (throughput, average trash %, rejection rate,
    total refaction value) and average per-class trash breakdown over the
    requested trailing time window.
    """
    window_start = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=since_hours)

    base_query = select(Transaction).where(Transaction.created_at >= window_start)
    rows = (await db.execute(base_query)).scalars().all()

    total_transactions = len(rows)
    total_gross = sum(t.gross_weight_kg or 0.0 for t in rows)
    total_net = sum(t.net_payable_weight_kg or 0.0 for t in rows)
    total_deduction_amount = sum(t.price_deduction_amount or 0.0 for t in rows)

    trash_pcts = [t.trash_mass_equivalent_pct for t in rows if t.trash_mass_equivalent_pct is not None]
    avg_trash_pct = round(sum(trash_pcts) / len(trash_pcts), 2) if trash_pcts else 0.0

    rejected = sum(1 for t in rows if t.status == TransactionStatus.REJECTED)
    manual_review = sum(
        1 for t in rows if t.sop_actions and "ROUTE_MANUAL_REINSPECTION" in t.sop_actions
    )

    kpi = KPISummary(
        total_transactions=total_transactions,
        total_gross_weight_kg=round(total_gross, 2),
        total_net_payable_weight_kg=round(total_net, 2),
        average_trash_pct=avg_trash_pct,
        rejected_loads=rejected,
        manual_review_loads=manual_review,
        total_price_deduction_amount=round(total_deduction_amount, 2),
    )

    class_totals: dict[str, list[float]] = {cls: [] for cls in TRASH_CLASSES}
    for t in rows:
        if not t.class_breakdown:
            continue
        for cls in TRASH_CLASSES:
            if cls in t.class_breakdown:
                class_totals[cls].append(t.class_breakdown[cls])

    breakdown = [
        TrashClassAverage(
            class_name=cls,
            average_area_ratio=round(sum(vals) / len(vals), 4) if vals else 0.0,
        )
        for cls, vals in class_totals.items()
    ]

    return AnalyticsResponse(
        kpi=kpi,
        trash_class_breakdown=breakdown,
        window_start=window_start,
        window_end=dt.datetime.now(dt.timezone.utc),
    )


@router.get("/throughput")
async def throughput_stats(db: AsyncSession = Depends(get_db)) -> dict:
    """Simple hourly throughput count for the current day (dashboard sparkline)."""
    today_start = dt.datetime.now(dt.timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    query = select(Transaction.created_at).where(Transaction.created_at >= today_start)
    rows = (await db.execute(query)).scalars().all()

    hourly_counts = [0] * 24
    for created_at in rows:
        hourly_counts[created_at.hour] += 1

    return {"date": today_start.date().isoformat(), "hourly_counts": hourly_counts}
