"""
system_health.py
=================
Diagnostic routes for camera/LiDAR/weighbridge/printer/AI-backend status,
plus a gantry air-purge trigger (used to clear dust/debris from camera
housings between truck scans).
"""

from __future__ import annotations

import datetime as dt
import logging

from fastapi import APIRouter, Depends

from ai_core.lidar_3d_engine import _HAS_OPEN3D
from ai_core.segmentation_engine import _HAS_ULTRALYTICS
from backend.config import Settings, get_settings
from backend.schemas import ComponentHealth, SystemHealthResponse

logger = logging.getLogger("canetrash.backend.routers.system_health")

router = APIRouter(prefix="/system", tags=["system_health"])


@router.get("/health", response_model=SystemHealthResponse)
async def system_health(settings: Settings = Depends(get_settings)) -> SystemHealthResponse:
    """Aggregate health/backend-mode report across all subsystems."""
    components: list[ComponentHealth] = []

    # --- AI Core ---
    components.append(
        ComponentHealth(
            component="segmentation_engine",
            status="ok",
            backend="yolov10" if (_HAS_ULTRALYTICS and settings.segmentation_weights_path) else "synthetic",
            detail="ultralytics installed" if _HAS_ULTRALYTICS else "ultralytics not installed; synthetic fallback active",
        )
    )
    components.append(
        ComponentHealth(
            component="lidar_3d_engine",
            status="ok",
            backend="open3d" if _HAS_OPEN3D else "numpy_hull",
            detail="open3d installed" if _HAS_OPEN3D else "open3d not installed; numpy/scipy fallback active",
        )
    )

    # --- SLM agent connectivity (best-effort, non-blocking) ---
    slm_status, slm_detail = await _check_slm_backend(settings)
    components.append(
        ComponentHealth(
            component="slm_agent",
            status=slm_status,
            backend="ollama" if slm_status == "ok" else "template",
            detail=slm_detail,
        )
    )

    # --- Hardware simulators ---
    components.append(
        ComponentHealth(
            component="weighbridge",
            status="ok",
            backend="simulator" if settings.use_hardware_simulators else "modbus_rtu",
            detail=f"port={settings.weighbridge_port}",
        )
    )
    components.append(
        ComponentHealth(
            component="gantry_cameras",
            status="ok",
            backend="simulator",
            detail="mock_gantry_cameras active (phase 1)",
        )
    )
    components.append(
        ComponentHealth(
            component="slip_printer",
            status="ok",
            backend="simulator",
            detail=f"device={settings.slip_printer_device}",
        )
    )

    overall = "degraded" if any(c.status != "ok" for c in components) else "ok"

    return SystemHealthResponse(
        overall_status=overall,
        checked_at=dt.datetime.now(dt.timezone.utc),
        components=components,
    )


@router.post("/gantry/air-purge")
async def trigger_air_purge() -> dict:
    """
    Triggers the compressed-air purge cycle on the camera gantry housings
    to clear dust/cane fiber debris between scans. Simulated in phase 1.
    """
    logger.info("Air-purge cycle triggered on gantry camera housings (simulated).")
    return {
        "status": "ok",
        "message": "Air-purge cycle completed (simulated).",
        "duration_s": 2.5,
        "triggered_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


async def _check_slm_backend(settings: Settings) -> tuple[str, str]:
    try:
        import httpx

        async with httpx.AsyncClient(timeout=1.5) as client:
            resp = await client.get(f"{settings.ollama_host}/api/tags")
            if resp.status_code == 200:
                return "ok", f"Ollama reachable at {settings.ollama_host}"
    except Exception as exc:  # noqa: BLE001
        return "degraded", f"Ollama unreachable ({exc}); template summary fallback active"
    return "degraded", "Ollama unreachable; template summary fallback active"
