"""
segmentation_engine.py
=======================
2D Instance Segmentation wrapper for the Overhead RGB Gantry Cameras.

Production backend: YOLOv10 (ultralytics) fine-tuned checkpoint, or SAM2
(segment-anything-2) for promptable mask refinement.

This module NEVER hard-fails if `ultralytics` / model weights are absent.
It probes for a real backend at import time and otherwise falls back to a
deterministic synthetic segmentation generator so the rest of the system
(pipeline_runner, API, dashboard) can run fully offline / CI-safe.

Classes detected (fixed label set for CaneTrash-AI):
    0 - clean_cane
    1 - dry_leaf
    2 - green_top
    3 - soil_dirt
    4 - rotten_stalk
"""

from __future__ import annotations

import io
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("canetrash.ai_core.segmentation")

CLASS_NAMES: list[str] = [
    "clean_cane",
    "dry_leaf",
    "green_top",
    "soil_dirt",
    "rotten_stalk",
]

TRASH_CLASSES = {"dry_leaf", "green_top", "soil_dirt", "rotten_stalk"}

# --------------------------------------------------------------------------- #
# Optional real backend detection
# --------------------------------------------------------------------------- #
try:
    from ultralytics import YOLO  # type: ignore

    _HAS_ULTRALYTICS = True
except ImportError:  # pragma: no cover - environment dependent
    _HAS_ULTRALYTICS = False


@dataclass
class Detection:
    """A single instance-segmentation detection."""

    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    mask_area_px: int
    is_trash: bool = field(init=False)

    def __post_init__(self) -> None:
        self.is_trash = self.class_name in TRASH_CLASSES

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox_xyxy": [round(v, 2) for v in self.bbox_xyxy],
            "mask_area_px": self.mask_area_px,
            "is_trash": self.is_trash,
        }


@dataclass
class SegmentationResult:
    """Aggregate segmentation output for a single frame."""

    frame_id: str
    detections: list[Detection]
    frame_width: int
    frame_height: int
    inference_ms: float
    backend: str  # "yolov10", "sam2", or "synthetic"

    @property
    def total_mask_area_px(self) -> int:
        return sum(d.mask_area_px for d in self.detections)

    @property
    def trash_mask_area_px(self) -> int:
        return sum(d.mask_area_px for d in self.detections if d.is_trash)

    @property
    def trash_area_ratio(self) -> float:
        """Trash pixel area / total detected mask area (0.0 - 1.0)."""
        if self.total_mask_area_px == 0:
            return 0.0
        return round(self.trash_mask_area_px / self.total_mask_area_px, 4)

    def class_breakdown(self) -> dict[str, float]:
        """Percentage of total mask area occupied by each class."""
        total = self.total_mask_area_px or 1
        breakdown: dict[str, int] = {name: 0 for name in CLASS_NAMES}
        for d in self.detections:
            breakdown[d.class_name] += d.mask_area_px
        return {k: round(v / total, 4) for k, v in breakdown.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "backend": self.backend,
            "inference_ms": round(self.inference_ms, 2),
            "detections": [d.to_dict() for d in self.detections],
            "trash_area_ratio": self.trash_area_ratio,
            "class_breakdown": self.class_breakdown(),
        }


class SegmentationEngine:
    """
    Instance segmentation engine for a single overhead camera frame.

    Usage:
        engine = SegmentationEngine(weights_path="models/canetrash_yolov10.pt")
        result = engine.infer(frame_bytes, frame_id="cam1_0001")

    If `weights_path` is None, missing, or ultralytics is not installed,
    the engine automatically operates in synthetic mode, producing
    plausible-looking detections seeded by frame_id for reproducibility.
    """

    def __init__(
        self,
        weights_path: str | None = None,
        confidence_threshold: float = 0.35,
        device: str = "cpu",
        force_synthetic: bool = False,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._model: Any = None
        self.backend = "synthetic"

        if force_synthetic:
            logger.info("SegmentationEngine started in forced-synthetic mode.")
            return

        if _HAS_ULTRALYTICS and weights_path and Path(weights_path).exists():
            try:
                self._model = YOLO(weights_path)
                self.backend = "yolov10"
                logger.info("Loaded YOLOv10 weights from %s", weights_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to load real segmentation model (%s). "
                    "Falling back to synthetic backend.",
                    exc,
                )
                self._model = None
                self.backend = "synthetic"
        else:
            logger.info(
                "No usable YOLOv10 weights found (ultralytics_installed=%s, "
                "weights_path=%s). Using synthetic segmentation backend.",
                _HAS_ULTRALYTICS,
                weights_path,
            )

    # --------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------- #
    def infer(self, frame: bytes | Any, frame_id: str) -> SegmentationResult:
        """
        Run instance segmentation on a single frame.

        Args:
            frame: raw image bytes (jpg/png) or an already-decoded array/object.
                   Ignored in synthetic mode beyond size hinting.
            frame_id: unique identifier for the frame (used for logging and,
                      in synthetic mode, as a deterministic RNG seed).

        Returns:
            SegmentationResult
        """
        start = time.perf_counter()

        if self._model is not None:
            try:
                result = self._infer_real(frame, frame_id)
                result.inference_ms = (time.perf_counter() - start) * 1000
                return result
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Real segmentation inference failed for frame %s (%s); "
                    "falling back to synthetic result.",
                    frame_id,
                    exc,
                )

        result = self._infer_synthetic(frame, frame_id)
        result.inference_ms = (time.perf_counter() - start) * 1000
        return result

    # --------------------------------------------------------------- #
    # Real backend
    # --------------------------------------------------------------- #
    def _infer_real(self, frame: bytes | Any, frame_id: str) -> SegmentationResult:
        image_source: Any
        if isinstance(frame, (bytes, bytearray)):
            image_source = io.BytesIO(frame)
        else:
            image_source = frame

        preds = self._model.predict(
            source=image_source,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )
        pred = preds[0]
        h, w = pred.orig_shape
        detections: list[Detection] = []

        boxes = getattr(pred, "boxes", None)
        masks = getattr(pred, "masks", None)

        if boxes is not None:
            for i, box in enumerate(boxes):
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                xyxy = tuple(float(v) for v in box.xyxy[0].tolist())
                if masks is not None and i < len(masks.data):
                    mask_area = int(masks.data[i].sum().item())
                else:
                    mask_area = int((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]))
                class_name = (
                    CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
                )
                detections.append(
                    Detection(
                        class_id=cls_id,
                        class_name=class_name,
                        confidence=conf,
                        bbox_xyxy=xyxy,
                        mask_area_px=mask_area,
                    )
                )

        return SegmentationResult(
            frame_id=frame_id,
            detections=detections,
            frame_width=w,
            frame_height=h,
            inference_ms=0.0,
            backend=self.backend,
        )

    # --------------------------------------------------------------- #
    # Synthetic fallback
    # --------------------------------------------------------------- #
    def _infer_synthetic(self, frame: bytes | Any, frame_id: str) -> SegmentationResult:
        """
        Generates deterministic, plausible synthetic detections so that
        downstream volumetric fusion / SLM / API / dashboard logic can be
        fully exercised without real hardware or trained weights.
        """
        rng = random.Random(frame_id)
        frame_w, frame_h = 1920, 1080

        num_detections = rng.randint(6, 14)
        detections: list[Detection] = []

        # Bias toward clean_cane dominating the load, trash classes minority.
        weighted_classes = (
            ["clean_cane"] * 6
            + ["dry_leaf"] * 3
            + ["green_top"] * 2
            + ["soil_dirt"] * 2
            + ["rotten_stalk"] * 1
        )

        for i in range(num_detections):
            class_name = rng.choice(weighted_classes)
            class_id = CLASS_NAMES.index(class_name)
            x1 = rng.uniform(0, frame_w * 0.8)
            y1 = rng.uniform(0, frame_h * 0.8)
            bw = rng.uniform(60, 320)
            bh = rng.uniform(60, 320)
            x2 = min(frame_w, x1 + bw)
            y2 = min(frame_h, y1 + bh)
            mask_area = int((x2 - x1) * (y2 - y1) * rng.uniform(0.45, 0.85))
            confidence = rng.uniform(self.confidence_threshold, 0.98)

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox_xyxy=(x1, y1, x2, y2),
                    mask_area_px=mask_area,
                )
            )

        return SegmentationResult(
            frame_id=frame_id,
            detections=detections,
            frame_width=frame_w,
            frame_height=frame_h,
            inference_ms=0.0,
            backend="synthetic",
        )


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    logging.basicConfig(level=logging.INFO)
    engine = SegmentationEngine(weights_path=None)
    res = engine.infer(frame=b"", frame_id="demo_frame_001")
    import json

    print(json.dumps(res.to_dict(), indent=2))
