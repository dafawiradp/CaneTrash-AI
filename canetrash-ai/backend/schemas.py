"""
schemas.py
==========
Pydantic v2 request/response schemas for the FastAPI routers.
"""

from __future__ import annotations

import datetime as dt
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class TransactionStatusSchema(str, Enum):
    PENDING = "pending"
    SCANNING = "scanning"
    PROCESSING = "processing"
    COMPLETED = "completed"
    REJECTED = "rejected"
    ERROR = "error"


# --------------------------------------------------------------------------- #
# Supplier
# --------------------------------------------------------------------------- #
class SupplierCreate(BaseModel):
    name: str
    supplier_code: str
    region: str | None = None
    contact_phone: str | None = None


class SupplierRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    supplier_code: str
    region: str | None
    contact_phone: str | None
    created_at: dt.datetime


# --------------------------------------------------------------------------- #
# Transaction
# --------------------------------------------------------------------------- #
class TransactionCreate(BaseModel):
    plate_number: str = Field(..., description="Truck license plate, e.g. from ANPR")
    supplier_id: str | None = None
    base_price_per_ton: float | None = Field(
        default=None, description="Overrides default mill tariff if provided"
    )


class TransactionScanTrigger(BaseModel):
    """Payload to manually trigger the scan pipeline for a pending transaction."""

    gross_weight_kg: float | None = Field(
        default=None,
        description="If omitted, the weighbridge listener/simulator value is used.",
    )


class TransactionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    transaction_code: str
    supplier_id: str | None
    plate_number: str
    anpr_confidence: float | None
    gross_weight_kg: float
    tare_weight_kg: float | None
    trash_mass_equivalent_pct: float | None
    trash_volume_pct: float | None
    deduction_pct: float | None
    deducted_weight_kg: float | None
    net_payable_weight_kg: float | None
    base_price_per_ton: float | None
    price_deduction_amount: float | None
    final_payable_amount: float | None
    sop_actions: list[str] | None
    class_breakdown: dict[str, float] | None
    audit_summary: str | None
    status: TransactionStatusSchema
    slip_printed: bool
    created_at: dt.datetime
    updated_at: dt.datetime
    completed_at: dt.datetime | None


class TransactionListResponse(BaseModel):
    total: int
    items: list[TransactionRead]


# --------------------------------------------------------------------------- #
# Analytics
# --------------------------------------------------------------------------- #
class KPISummary(BaseModel):
    total_transactions: int
    total_gross_weight_kg: float
    total_net_payable_weight_kg: float
    average_trash_pct: float
    rejected_loads: int
    manual_review_loads: int
    total_price_deduction_amount: float


class TrashClassAverage(BaseModel):
    class_name: str
    average_area_ratio: float


class AnalyticsResponse(BaseModel):
    kpi: KPISummary
    trash_class_breakdown: list[TrashClassAverage]
    window_start: dt.datetime | None
    window_end: dt.datetime | None


# --------------------------------------------------------------------------- #
# System health
# --------------------------------------------------------------------------- #
class ComponentHealth(BaseModel):
    component: str
    status: str  # "ok" | "degraded" | "down"
    backend: str | None = None
    detail: str | None = None


class SystemHealthResponse(BaseModel):
    overall_status: str
    checked_at: dt.datetime
    components: list[ComponentHealth]
