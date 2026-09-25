"""
lidar_3d_engine.py
===================
3D Volumetric Fusion & Density Estimation.

Fuses:
  - Raw LiDAR point-cloud volume estimate of the truck cargo (Open3D voxel
    carving when available, NumPy convex-hull approximation otherwise).
  - 2D segmentation results (per-class pixel area ratios) from
    segmentation_engine.SegmentationResult.

...to estimate the TOTAL CARGO TRASH PERCENTAGE by weight-equivalent volume,
using class-specific density coefficients (dry leaves are much less dense
than clean cane stalks, soil/dirt is denser, etc).

Falls back cleanly to NumPy-only geometry when `open3d` is not installed.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger("canetrash.ai_core.lidar3d")

try:
    import open3d as o3d  # type: ignore

    _HAS_OPEN3D = True
except ImportError:  # pragma: no cover - environment dependent
    _HAS_OPEN3D = False

# Relative density coefficients vs. clean_cane == 1.0.
# Used to convert 2D pixel-area class ratios into volumetric / mass-equivalent
# trash fraction, since a leaf occupies visual area disproportionate to its mass.
DENSITY_COEFFICIENTS: dict[str, float] = {
    "clean_cane": 1.00,
    "dry_leaf": 0.18,
    "green_top": 0.35,
    "soil_dirt": 1.40,
    "rotten_stalk": 0.85,
}


@dataclass
class VolumetricResult:
    """Output of the 3D LiDAR + 2D-class density fusion step."""

    scan_id: str
    total_volume_m3: float
    point_count: int
    backend: str  # "open3d", "numpy_hull", or "synthetic"
    class_area_ratios: dict[str, float]
    class_volume_estimate_m3: dict[str, float]
    trash_volume_pct: float          # raw volumetric trash %, unweighted density
    trash_mass_equivalent_pct: float  # density-weighted -> closer to refaction basis
    processing_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "scan_id": self.scan_id,
            "backend": self.backend,
            "total_volume_m3": round(self.total_volume_m3, 4),
            "point_count": self.point_count,
            "class_area_ratios": {k: round(v, 4) for k, v in self.class_area_ratios.items()},
            "class_volume_estimate_m3": {
                k: round(v, 4) for k, v in self.class_volume_estimate_m3.items()
            },
            "trash_volume_pct": round(self.trash_volume_pct, 2),
            "trash_mass_equivalent_pct": round(self.trash_mass_equivalent_pct, 2),
            "processing_ms": round(self.processing_ms, 2),
        }


class Lidar3DEngine:
    """
    Volumetric fusion engine.

    Usage:
        engine = Lidar3DEngine()
        vol_result = engine.fuse(
            point_cloud=raw_points_or_None,
            class_area_ratios=seg_result.class_breakdown(),
            scan_id="truck_0001_scan",
        )
    """

    TRASH_CLASSES = {"dry_leaf", "green_top", "soil_dirt", "rotten_stalk"}

    def __init__(self, voxel_size_m: float = 0.05, force_synthetic: bool = False) -> None:
        self.voxel_size_m = voxel_size_m
        self.force_synthetic = force_synthetic
        self.backend_available = "open3d" if _HAS_OPEN3D else "numpy_hull"
        logger.info(
            "Lidar3DEngine initialized (open3d_available=%s, voxel_size=%.3fm)",
            _HAS_OPEN3D,
            voxel_size_m,
        )

    # --------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------- #
    def fuse(
        self,
        point_cloud: "np.ndarray | None",
        class_area_ratios: dict[str, float],
        scan_id: str,
    ) -> VolumetricResult:
        """
        Args:
            point_cloud: (N, 3) float array of XYZ points in meters, relative
                to the gantry scan origin. If None, a synthetic cargo point
                cloud is generated.
            class_area_ratios: per-class fraction of total 2D mask area, as
                produced by SegmentationResult.class_breakdown().
            scan_id: unique identifier for this LiDAR scan / truck pass.

        Returns:
            VolumetricResult
        """
        import time

        start = time.perf_counter()

        if point_cloud is None or self.force_synthetic:
            point_cloud, backend_note = self._synthetic_point_cloud(scan_id)
        else:
            backend_note = None

        if _HAS_OPEN3D and not self.force_synthetic and backend_note is None:
            volume_m3, point_count, backend = self._volume_open3d(point_cloud)
        else:
            volume_m3, point_count, backend = self._volume_numpy(point_cloud)
            if backend_note:
                backend = "synthetic"

        class_volume = {
            cls: round(volume_m3 * ratio, 4) for cls, ratio in class_area_ratios.items()
        }

        trash_volume_m3 = sum(
            v for cls, v in class_volume.items() if cls in self.TRASH_CLASSES
        )
        trash_volume_pct = (trash_volume_m3 / volume_m3 * 100) if volume_m3 > 0 else 0.0

        # Density-weighted mass-equivalent trash percentage (used for refaction).
        weighted_total = sum(
            class_volume.get(cls, 0.0) * DENSITY_COEFFICIENTS.get(cls, 1.0)
            for cls in DENSITY_COEFFICIENTS
        )
        weighted_trash = sum(
            class_volume.get(cls, 0.0) * DENSITY_COEFFICIENTS.get(cls, 1.0)
            for cls in self.TRASH_CLASSES
        )
        trash_mass_pct = (weighted_trash / weighted_total * 100) if weighted_total > 0 else 0.0

        result = VolumetricResult(
            scan_id=scan_id,
            total_volume_m3=volume_m3,
            point_count=point_count,
            backend=backend,
            class_area_ratios=class_area_ratios,
            class_volume_estimate_m3=class_volume,
            trash_volume_pct=trash_volume_pct,
            trash_mass_equivalent_pct=trash_mass_pct,
        )
        result.processing_ms = (time.perf_counter() - start) * 1000
        return result

    # --------------------------------------------------------------- #
    # Volume estimation backends
    # --------------------------------------------------------------- #
    def _volume_open3d(self, points: np.ndarray) -> tuple[float, int, str]:
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            hull, _ = pcd.compute_convex_hull()
            volume_m3 = float(hull.get_volume())
            return volume_m3, len(points), "open3d"
        except Exception as exc:  # noqa: BLE001
            logger.warning("open3d volume computation failed (%s); using numpy fallback.", exc)
            return self._volume_numpy(points)

    def _volume_numpy(self, points: np.ndarray) -> tuple[float, int, str]:
        """
        Approximate cargo volume via bounding-box * fill-factor when
        scipy.spatial.ConvexHull is unavailable, otherwise a true convex hull.
        """
        try:
            from scipy.spatial import ConvexHull  # type: ignore

            hull = ConvexHull(points)
            return float(hull.volume), len(points), "numpy_hull"
        except Exception:  # noqa: BLE001 - scipy optional
            mins = points.min(axis=0)
            maxs = points.max(axis=0)
            bbox_volume = float(np.prod(maxs - mins))
            fill_factor = 0.55  # typical loose cane-stack packing density
            return bbox_volume * fill_factor, len(points), "numpy_bbox"

    # --------------------------------------------------------------- #
    # Synthetic point cloud generator
    # --------------------------------------------------------------- #
    def _synthetic_point_cloud(self, scan_id: str) -> tuple[np.ndarray, str]:
        """
        Generates a plausible truck-cargo-shaped point cloud: an elongated
        mound roughly matching a sugarcane truck trailer (approx 8m x 2.4m
        bed, 0.5-1.8m stack height), seeded deterministically by scan_id.
        """
        rng = random.Random(scan_id)
        np_rng = np.random.default_rng(abs(hash(scan_id)) % (2**32))

        length_m = rng.uniform(7.5, 9.0)
        width_m = rng.uniform(2.2, 2.6)
        max_height_m = rng.uniform(1.2, 2.0)

        n_points = 4000
        x = np_rng.uniform(0, length_m, n_points)
        y = np_rng.uniform(0, width_m, n_points)
        # Mound-shaped height profile via a smooth bump function.
        cx, cy = length_m / 2, width_m / 2
        norm_dist = np.sqrt(((x - cx) / (length_m / 2)) ** 2 + ((y - cy) / (width_m / 2)) ** 2)
        norm_dist = np.clip(norm_dist, 0, 1)
        z = max_height_m * (1 - norm_dist**2) * np_rng.uniform(0.85, 1.0, n_points)
        z = np.clip(z, 0.02, None)

        points = np.column_stack([x, y, z])
        return points, "synthetic"


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    logging.basicConfig(level=logging.INFO)
    engine = Lidar3DEngine()
    sample_ratios = {
        "clean_cane": 0.62,
        "dry_leaf": 0.15,
        "green_top": 0.10,
        "soil_dirt": 0.08,
        "rotten_stalk": 0.05,
    }
    result = engine.fuse(point_cloud=None, class_area_ratios=sample_ratios, scan_id="demo_scan_001")
    import json

    print(json.dumps(result.to_dict(), indent=2))
