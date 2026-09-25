"""
pipeline_runner.py
===================
Orchestrates the end-to-end ~15-second gantry scan workflow for a single
truck transaction:

    1. Read gross weight from the weighbridge (real or simulated).
    2. Capture N overhead RGB frames + a LiDAR point cloud from the gantry
       (real or simulated via hardware_sim.mock_gantry_cameras).
    3. Run 2D instance segmentation on each frame (ai_core.segmentation_engine).
    4. Fuse per-frame segmentation with the LiDAR point cloud to get a
       volumetric / mass-equivalent trash percentage (ai_core.lidar_3d_engine).
    5. Run the SLM agent to compute refaction deduction + SOP actions
       (ai_core.slm_agent).
    6. Persist a Transaction row + QualityLog audit entries.
    7. Optionally trigger the thermal slip printer.

This module is transport-agnostic: it can be invoked directly from the
routers, from a background task queue, or from a CLI script.
"""

from __future__ import annotations

import logging
import time
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from ai_core.lidar_3d_engine import Lidar3DEngine
from ai_core.segmentation_engine import SegmentationEngine, SegmentationResult
from ai_core.slm_agent import SLMAgent
from backend.config import Settings
from backend.models import QualityLog, Transaction, TransactionStatus
from backend.services.weighbridge_listener import WeighbridgeListener

logger = logging.getLogger("canetrash.backend.pipeline_runner")

FRAMES_PER_SCAN = 5


class PipelineRunner:
    """
    Stateless-per-call orchestrator. Instantiate once (it owns the AI engines,
    which may load model weights) and reuse across requests.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.segmentation_engine = SegmentationEngine(
            weights_path=settings.segmentation_weights_path,
            confidence_threshold=settings.segmentation_confidence_threshold,
            force_synthetic=settings.force_synthetic_ai,
        )
        self.lidar_engine = Lidar3DEngine(
            voxel_size_m=settings.lidar_voxel_size_m,
            force_synthetic=settings.force_synthetic_ai,
        )
        self.slm_agent = SLMAgent(
            ollama_host=settings.ollama_host,
            ollama_model=settings.ollama_model,
            vllm_base_url=settings.vllm_base_url,
        )
        self.weighbridge = WeighbridgeListener(settings)

        # Camera / LiDAR gantry: always uses the hardware_sim module for
        # frame/point-cloud generation in phase 1 (real camera SDK integration
        # is a hardware-specific follow-up).
        from hardware_sim.mock_gantry_cameras import MockGantryCameras

        self.gantry = MockGantryCameras()

    # --------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------- #
    async def run_full_scan(
        self,
        db: AsyncSession,
        transaction: Transaction,
        gross_weight_kg_override: float | None = None,
    ) -> Transaction:
        """
        Executes the full scan workflow for an existing PENDING transaction
        and updates it in place. Returns the updated, persisted transaction.
        """
        pipeline_start = time.perf_counter()
        logger.info("Starting scan pipeline for transaction %s", transaction.transaction_code)

        transaction.status = TransactionStatus.SCANNING
        db.add(transaction)
        await db.flush()

        try:
            # --- 1. Weighbridge ---
            gross_weight_kg = gross_weight_kg_override or self.weighbridge.read_weight_kg(
                truck_hint=transaction.plate_number
            )
            transaction.gross_weight_kg = gross_weight_kg

            # --- 2 & 3. Gantry frames + segmentation ---
            transaction.status = TransactionStatus.PROCESSING
            db.add(transaction)
            await db.flush()

            frame_results: list[SegmentationResult] = []
            for i in range(FRAMES_PER_SCAN):
                frame_id = f"{transaction.transaction_code}_frame_{i:02d}"
                frame_bytes = self.gantry.capture_rgb_frame(frame_id)
                seg_result = self.segmentation_engine.infer(frame_bytes, frame_id)
                frame_results.append(seg_result)

                db.add(
                    QualityLog(
                        id=str(uuid.uuid4()),
                        transaction_id=transaction.id,
                        stage="segmentation",
                        backend_used=seg_result.backend,
                        payload=seg_result.to_dict(),
                        processing_ms=seg_result.inference_ms,
                    )
                )

            averaged_breakdown = self._average_class_breakdown(frame_results)

            # --- 4. LiDAR volumetric fusion ---
            point_cloud = self.gantry.capture_point_cloud(transaction.transaction_code)
            vol_result = self.lidar_engine.fuse(
                point_cloud=point_cloud,
                class_area_ratios=averaged_breakdown,
                scan_id=f"{transaction.transaction_code}_lidar",
            )
            db.add(
                QualityLog(
                    id=str(uuid.uuid4()),
                    transaction_id=transaction.id,
                    stage="lidar",
                    backend_used=vol_result.backend,
                    payload=vol_result.to_dict(),
                    processing_ms=vol_result.processing_ms,
                )
            )

            transaction.trash_volume_pct = vol_result.trash_volume_pct
            transaction.trash_mass_equivalent_pct = vol_result.trash_mass_equivalent_pct
            transaction.class_breakdown = averaged_breakdown

            # --- 5. SLM agent decision ---
            base_price = transaction.base_price_per_ton or self.settings.default_base_price_per_ton
            decision = self.slm_agent.evaluate(
                transaction_id=transaction.transaction_code,
                trash_mass_equivalent_pct=vol_result.trash_mass_equivalent_pct,
                gross_weight_kg=gross_weight_kg,
                base_price_per_ton=base_price,
                class_breakdown=averaged_breakdown,
            )
            db.add(
                QualityLog(
                    id=str(uuid.uuid4()),
                    transaction_id=transaction.id,
                    stage="slm_agent",
                    backend_used=decision.summary_backend,
                    payload=decision.to_dict(),
                    processing_ms=None,
                )
            )

            transaction.base_price_per_ton = base_price
            transaction.deduction_pct = decision.deduction_pct
            transaction.deducted_weight_kg = decision.deducted_weight_kg
            transaction.net_payable_weight_kg = decision.net_payable_weight_kg
            transaction.price_deduction_amount = decision.price_deduction_amount
            transaction.final_payable_amount = decision.final_payable_amount
            transaction.sop_actions = decision.sop_actions
            transaction.audit_summary = decision.audit_summary

            # --- 6. Finalize status ---
            import datetime as dt

            if "REJECT_LOAD" in decision.sop_actions:
                transaction.status = TransactionStatus.REJECTED
            else:
                transaction.status = TransactionStatus.COMPLETED
            transaction.completed_at = dt.datetime.now(dt.timezone.utc)

            db.add(transaction)
            await db.flush()

            # --- 7. Slip printer (best-effort, never fails the pipeline) ---
            try:
                from hardware_sim.mock_slip_printer import MockSlipPrinter

                printer = MockSlipPrinter(device=self.settings.slip_printer_device)
                printer.print_slip(transaction_to_slip_dict(transaction))
                transaction.slip_printed = True
                db.add(transaction)
                await db.flush()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Slip printing failed (non-fatal): %s", exc)

            elapsed = time.perf_counter() - pipeline_start
            logger.info(
                "Pipeline completed for %s in %.2fs (status=%s, trash=%.2f%%)",
                transaction.transaction_code,
                elapsed,
                transaction.status.value,
                transaction.trash_mass_equivalent_pct or 0.0,
            )
            return transaction

        except Exception as exc:  # noqa: BLE001
            logger.exception("Pipeline failed for transaction %s", transaction.transaction_code)
            transaction.status = TransactionStatus.ERROR
            transaction.audit_summary = f"Pipeline error: {exc}"
            db.add(transaction)
            await db.flush()
            raise

    # --------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------- #
    @staticmethod
    def _average_class_breakdown(
        results: list[SegmentationResult],
    ) -> dict[str, float]:
        if not results:
            return {}
        totals: dict[str, float] = {}
        for r in results:
            for cls, ratio in r.class_breakdown().items():
                totals[cls] = totals.get(cls, 0.0) + ratio
        n = len(results)
        return {cls: round(v / n, 4) for cls, v in totals.items()}


def transaction_to_slip_dict(transaction: Transaction) -> dict:
    """Flatten a Transaction ORM object into the dict shape expected by the slip printer."""
    return {
        "transaction_code": transaction.transaction_code,
        "plate_number": transaction.plate_number,
        "gross_weight_kg": transaction.gross_weight_kg,
        "trash_mass_equivalent_pct": transaction.trash_mass_equivalent_pct,
        "deduction_pct": transaction.deduction_pct,
        "net_payable_weight_kg": transaction.net_payable_weight_kg,
        "final_payable_amount": transaction.final_payable_amount,
        "sop_actions": transaction.sop_actions,
        "status": transaction.status.value
        if hasattr(transaction.status, "value")
        else transaction.status,
    }
