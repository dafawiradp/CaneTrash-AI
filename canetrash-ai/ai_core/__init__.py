"""
ai_core
=======
Computer-vision and reasoning core for CaneTrash-AI.

Modules:
    segmentation_engine  - 2D instance segmentation (YOLOv10 / SAM2) w/ synthetic fallback
    lidar_3d_engine       - 3D LiDAR volumetric fusion & density estimation
    slm_agent             - Local SLM (Ollama/vLLM) reasoning agent for refaction & SOP

All modules are designed to run WITHOUT live GPU/model weights or physical
sensors present. When real backends (ultralytics, open3d, ollama) are not
installed or reachable, each module transparently falls back to a
deterministic synthetic implementation so the rest of the stack (API,
dashboard, pipeline) can be developed and tested end-to-end.
"""

from .segmentation_engine import SegmentationEngine, SegmentationResult
from .lidar_3d_engine import Lidar3DEngine, VolumetricResult
from .slm_agent import SLMAgent, RefactionDecision

__all__ = [
    "SegmentationEngine",
    "SegmentationResult",
    "Lidar3DEngine",
    "VolumetricResult",
    "SLMAgent",
    "RefactionDecision",
]

__version__ = "0.1.0-phase1"
