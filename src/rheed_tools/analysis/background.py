from __future__ import annotations

"""Background-estimation helpers shared by offline RHEED analysis modules."""

import numpy as np


def estimate_rolling_background(
    values: np.ndarray,
    window_size: int = 21,
    percentile: float = 20.0,
) -> np.ndarray:
    """Estimate a slow baseline or lower envelope from a 1D trace."""

    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("values must be a 1D array")
    if arr.size == 0:
        return arr.copy()

    window = max(1, int(window_size))
    if window % 2 == 0:
        window += 1
    if window == 1:
        return np.full(arr.shape, np.percentile(arr, percentile), dtype=float)

    pad = window // 2
    padded = np.pad(arr, pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.percentile(windows, percentile, axis=1)


def subtract_rolling_background(
    values: np.ndarray,
    window_size: int = 21,
    percentile: float = 20.0,
    normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove slow baseline drift from a 1D intensity trace."""

    arr = np.asarray(values, dtype=float)
    background = estimate_rolling_background(arr, window_size=window_size, percentile=percentile)
    corrected = arr - background

    if not normalize or corrected.size == 0:
        return corrected, background

    scale = float(np.percentile(corrected, 95) - np.percentile(corrected, 5))
    if abs(scale) < 1e-12:
        return corrected, background
    return corrected / scale, background

