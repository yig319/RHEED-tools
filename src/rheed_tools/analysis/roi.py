from __future__ import annotations

"""ROI utilities for offline diffraction-frame analysis."""

import numpy as np



def sanitize_roi(
    shape: tuple[int, int],
    roi: tuple[int, int, int, int] | None,
    fraction: float,
    corner: str = "center",
) -> tuple[int, int, int, int]:
    """Clip an ROI to image bounds or create a default ROI."""

    h, w = shape
    if roi is not None:
        y0, y1, x0, x1 = roi
        y0 = int(np.clip(y0, 0, h - 1))
        y1 = int(np.clip(y1, y0 + 1, h))
        x0 = int(np.clip(x0, 0, w - 1))
        x1 = int(np.clip(x1, x0 + 1, w))
        return y0, y1, x0, x1

    box_h = max(3, int(round(h * fraction)))
    box_w = max(3, int(round(w * fraction)))
    if corner == "top_left":
        return 0, min(box_h, h), 0, min(box_w, w)

    cy = h // 2
    cx = w // 2
    y0 = max(0, cy - box_h // 2)
    x0 = max(0, cx - box_w // 2)
    y1 = min(h, y0 + box_h)
    x1 = min(w, x0 + box_w)
    return y0, y1, x0, x1


def crop_frame(frame: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    """Return a cropped view of one frame."""

    arr = np.asarray(frame)
    if arr.ndim != 2:
        raise ValueError("frame must be a 2D array")
    y0, y1, x0, x1 = sanitize_roi(arr.shape, roi, fraction=0.2)
    return arr[y0:y1, x0:x1]


def crop_frames(frames: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a frame stack with one shared ROI."""

    arr = np.asarray(frames)
    if arr.ndim != 3:
        raise ValueError("frames must be a 3D array shaped (n_frames, height, width)")
    y0, y1, x0, x1 = sanitize_roi(arr.shape[1:], roi, fraction=0.2)
    return arr[:, y0:y1, x0:x1]


def recenter_roi(
    shape: tuple[int, int],
    center_y: float,
    center_x: float,
    box_height: int,
    box_width: int,
) -> tuple[int, int, int, int]:
    """Build a fixed-size ROI centered on a drifting feature."""

    h, w = shape
    box_h = int(np.clip(box_height, 1, h))
    box_w = int(np.clip(box_width, 1, w))
    cy = float(np.clip(center_y, 0.0, max(h - 1, 0)))
    cx = float(np.clip(center_x, 0.0, max(w - 1, 0)))
    y0 = int(round(cy - box_h / 2.0))
    x0 = int(round(cx - box_w / 2.0))
    y0 = int(np.clip(y0, 0, max(h - box_h, 0)))
    x0 = int(np.clip(x0, 0, max(w - box_w, 0)))
    return y0, y0 + box_h, x0, x0 + box_w

