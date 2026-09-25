"""
models.py
=========
SQLAlchemy ORM models for CaneTrash-AI.

Tables:
    suppliers     - registered cane suppliers / farmer groups
    transactions  - one row per truck inspection pass (the core record)
    quality_logs  - immutable per-scan AI output snapshot (audit trail)
"""

from __future__ import annotations

import datetime as dt
import enum
import uuid

from sqlalchemy import JSON, DateTime, Enum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base


def _uuid_str() -> str:
    return str(uuid.uuid4())


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class TransactionStatus(str, enum.Enum):
    PENDING = "pending"
    SCANNING = "scanning"
    PROCESSING = "processing"
    COMPLETED = "completed"
    REJECTED = "rejected"
    ERROR = "error"


class Supplier(Base):
    __tablename__ = "suppliers"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    supplier_code: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    region: Mapped[str | None] = mapped_column(String(128), nullable=True)
    contact_phone: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    transactions: Mapped[list["Transaction"]] = relationship(
        back_populates="supplier", cascade="all, delete-orphan"
    )


class Transaction(Base):
    """One full truck inspection: ANPR + weighbridge + scan + AI decision."""

    __tablename__ = "transactions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    transaction_code: Mapped[str] = mapped_column(String(64), unique=True, index=True)

    supplier_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("suppliers.id"), nullable=True
    )
    supplier: Mapped[Supplier | None] = relationship(back_populates="transactions")

    plate_number: Mapped[str] = mapped_column(String(32), index=True)
    anpr_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    gross_weight_kg: Mapped[float] = mapped_column(Float, default=0.0)
    tare_weight_kg: Mapped[float | None] = mapped_column(Float, nullable=True)

    trash_mass_equivalent_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    trash_volume_pct: Mapped[float | None] = mapped_column(Float, nullable=True)

    deduction_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    deducted_weight_kg: Mapped[float | None] = mapped_column(Float, nullable=True)
    net_payable_weight_kg: Mapped[float | None] = mapped_column(Float, nullable=True)
    base_price_per_ton: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_deduction_amount: Mapped[float | None] = mapped_column(Float, nullable=True)
    final_payable_amount: Mapped[float | None] = mapped_column(Float, nullable=True)

    sop_actions: Mapped[list | None] = mapped_column(JSON, nullable=True)
    class_breakdown: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    audit_summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    status: Mapped[TransactionStatus] = mapped_column(
        Enum(TransactionStatus), default=TransactionStatus.PENDING, index=True
    )

    slip_printed: Mapped[bool] = mapped_column(default=False)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )
    completed_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    quality_logs: Mapped[list["QualityLog"]] = relationship(
        back_populates="transaction", cascade="all, delete-orphan"
    )


class QualityLog(Base):
    """
    Immutable audit-trail record of a single AI processing step
    (segmentation frame, lidar fusion, or SLM decision) tied to a transaction.
    """

    __tablename__ = "quality_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    transaction_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("transactions.id"), index=True
    )
    transaction: Mapped[Transaction] = relationship(back_populates="quality_logs")

    stage: Mapped[str] = mapped_column(String(32))  # "segmentation" | "lidar" | "slm_agent"
    backend_used: Mapped[str] = mapped_column(String(32))  # "synthetic" | "yolov10" | ...
    payload: Mapped[dict] = mapped_column(JSON)
    processing_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
