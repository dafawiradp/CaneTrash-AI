"""
mock_gantry_cameras.py
========================
Synthetic image and LiDAR point-cloud generator standing in for the
overhead RGB gantry cameras and 3D LiDAR scanner.

This module intentionally does NOT require Pillow/OpenCV: it returns raw
placeholder JPEG-header-prefixed bytes for the RGB path (the segmentation
engine's synthetic mode only uses `frame_id` for its content anyway, so the
byte payload only needs to be a valid `bytes` object, not a decodable
image, to keep the phase-1 dependency footprint minimal). Swap in a real
frame grabber (e.g. GigE Vision SDK / OpenCV VideoCapture) behind the same
interface for production hardware.
"""

from __future__ import annotations

import logging
import random

import numpy as np

logger = logging.getLogger("canetrash.hardware_sim.mock_gantry_cameras")

# Minimal valid JPEG SOI/EOI markers so downstream code that sniffs the
# magic bytes (without fully decoding) doesn't choke on an empty payload.
_PLACEHOLDER_JPEG_HEADER = b"\xff\xd8\xff\xe0"
_PLACEHOLDER_JPEG_FOOTER = b"\xff\xd9"


class MockGantryCameras:
    """
    Usage:
        gantry = MockGantryCameras()
        frame_bytes = gantry.capture_rgb_frame("scan001_frame00")
        points = gantry.capture_point_cloud("scan001")
    """

    def __init__(self, frame_payload_size: int = 2048) -> None:
        self.frame_payload_size = frame_payload_size

    def capture_rgb_frame(self, frame_id: str) -> bytes:
        """
        Returns a deterministic pseudo-JPEG byte payload for the given
        frame_id. Real hardware integration should replace this with an
        actual frame-grab call returning encoded image bytes.
        """
        rng = random.Random(frame_id)
        body = bytes(rng.getrandbits(8) for _ in range(self.frame_payload_size))
        logger.debug("Captured synthetic RGB frame %s (%d bytes)", frame_id, len(body))
        return _PLACEHOLDER_JPEG_HEADER + body + _PLACEHOLDER_JPEG_FOOTER

    def capture_point_cloud(self, scan_id: str) -> "np.ndarray | None":
        """
        Returns None to signal the caller (Lidar3DEngine) to use its own
        internal synthetic cargo-mound generator, which produces a more
        physically plausible truck-bed point cloud than a naive random
        cloud would here. Real hardware integration should return an
        (N, 3) float array of XYZ points instead.
        """
        logger.debug("Point cloud capture for %s delegated to Lidar3DEngine synthetic mode.", scan_id)
        return None


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.DEBUG)
    gantry = MockGantryCameras()
    frame = gantry.capture_rgb_frame("demo_frame")
    print(f"Frame bytes: {len(frame)}, header={frame[:4]!r}")
    print("Point cloud:", gantry.capture_point_cloud("demo_scan"))
