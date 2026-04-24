from __future__ import annotations

"""Diffuse-scattering and halo metrics for RHEED images."""

from dataclasses import dataclass

import numpy as np

from .diffraction_2d import analyze_rheed_frame
from .roi import sanitize_roi


@dataclass(slots=True)
class DiffuseFrameMetrics:
    """Diffuse-scattering summary for one frame."""

    signal_roi: tuple[int, int, int, int]
    diffuse_roi: tuple[int, int, int, int]
    diffuse_sum: float
    diffuse_mean: float
    diffuse_std: float
    diffuse_to_signal_ratio: float
    halo_sum: float
    halo_mean: float
    halo_ratio: float
    anisotropy: float


@dataclass(slots=True)
class DiffuseSeriesMetrics:
    """Diffuse-scattering trends across a movie."""

    ts: np.ndarray
    diffuse_sum: np.ndarray
    diffuse_mean: np.ndarray
    diffuse_std: np.ndarray
    diffuse_to_signal_ratio: np.ndarray
    halo_sum: np.ndarray
    halo_mean: np.ndarray
    halo_ratio: np.ndarray
    anisotropy: np.ndarray


def _annulus_mask(
    shape: tuple[int, int],
    *,
    center_x: float,
    center_y: float,
    inner_radius_px: float,
    outer_radius_px: float,
) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=float)
    rr = np.hypot(xx - center_x, yy - center_y)
    return (rr >= inner_radius_px) & (rr <= outer_radius_px)


def analyze_diffuse_scattering(
    frame: np.ndarray,
    *,
    signal_roi: tuple[int, int, int, int] | None = None,
    diffuse_roi: tuple[int, int, int, int] | None = None,
    background_roi: tuple[int, int, int, int] | None = None,
    halo_inner_scale: float = 1.5,
    halo_outer_scale: float = 3.0,
) -> DiffuseFrameMetrics:
    """Measure diffuse background and halo intensity around one signal ROI.

    Tuning guidance:
    - `signal_roi` should bound the main spot or streak family
    - `diffuse_roi` can be the full detector or a larger cropped region
    - `halo_inner_scale` and `halo_outer_scale` define the annulus around the
      spot where diffuse halo intensity is summarized
    """

    arr = np.asarray(frame, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("frame must be a non-empty 2D array")
    if halo_inner_scale <= 0 or halo_outer_scale <= halo_inner_scale:
        raise ValueError("halo scales must satisfy 0 < halo_inner_scale < halo_outer_scale")

    signal = analyze_rheed_frame(arr, roi=signal_roi, background_roi=background_roi)
    signal_y0, signal_y1, signal_x0, signal_x1 = signal.roi
    diff_y0, diff_y1, diff_x0, diff_x1 = sanitize_roi(arr.shape, diffuse_roi, fraction=0.75)

    corrected = np.clip(arr[diff_y0:diff_y1, diff_x0:diff_x1] - signal.background_mean, 0.0, None)
    local_center_x = signal.centroid_x - diff_x0
    local_center_y = signal.centroid_y - diff_y0
    signal_mask = np.zeros_like(corrected, dtype=bool)
    signal_mask[
        max(0, signal_y0 - diff_y0) : min(corrected.shape[0], signal_y1 - diff_y0),
        max(0, signal_x0 - diff_x0) : min(corrected.shape[1], signal_x1 - diff_x0),
    ] = True

    diffuse_pixels = corrected[~signal_mask]
    diffuse_sum = float(np.sum(diffuse_pixels))
    diffuse_mean = float(np.mean(diffuse_pixels)) if diffuse_pixels.size else 0.0
    diffuse_std = float(np.std(diffuse_pixels)) if diffuse_pixels.size else 0.0
    diffuse_to_signal_ratio = diffuse_sum / (abs(signal.corrected_sum) + 1e-9)

    halo_inner_radius = float(max(signal.fwhm_x, signal.fwhm_y) * halo_inner_scale / 2.0)
    halo_outer_radius = float(max(signal.fwhm_x, signal.fwhm_y) * halo_outer_scale / 2.0)
    halo_mask = _annulus_mask(
        corrected.shape,
        center_x=local_center_x,
        center_y=local_center_y,
        inner_radius_px=halo_inner_radius,
        outer_radius_px=halo_outer_radius,
    )
    halo_pixels = corrected[halo_mask]
    halo_sum = float(np.sum(halo_pixels)) if halo_pixels.size else 0.0
    halo_mean = float(np.mean(halo_pixels)) if halo_pixels.size else 0.0
    halo_ratio = halo_sum / (abs(signal.corrected_sum) + 1e-9)

    yy, xx = np.indices(corrected.shape, dtype=float)
    weights = corrected.copy()
    weights[signal_mask] = 0.0
    total = float(np.sum(weights))
    if total <= 1e-12:
        anisotropy = 0.0
    else:
        dx = xx - local_center_x
        dy = yy - local_center_y
        var_x = float(np.sum(weights * dx * dx) / total)
        var_y = float(np.sum(weights * dy * dy) / total)
        anisotropy = float(abs(var_x - var_y) / (var_x + var_y + 1e-9))

    return DiffuseFrameMetrics(
        signal_roi=signal.roi,
        diffuse_roi=(diff_y0, diff_y1, diff_x0, diff_x1),
        diffuse_sum=diffuse_sum,
        diffuse_mean=diffuse_mean,
        diffuse_std=diffuse_std,
        diffuse_to_signal_ratio=float(diffuse_to_signal_ratio),
        halo_sum=halo_sum,
        halo_mean=halo_mean,
        halo_ratio=float(halo_ratio),
        anisotropy=anisotropy,
    )


def analyze_diffuse_scattering_series(
    frames: np.ndarray,
    *,
    ts: np.ndarray | None = None,
    signal_roi: tuple[int, int, int, int] | None = None,
    diffuse_roi: tuple[int, int, int, int] | None = None,
    background_roi: tuple[int, int, int, int] | None = None,
    halo_inner_scale: float = 1.5,
    halo_outer_scale: float = 3.0,
) -> DiffuseSeriesMetrics:
    """Run diffuse-scattering analysis across a frame stack."""

    arr = np.asarray(frames, dtype=float)
    if arr.ndim != 3 or arr.shape[0] == 0:
        raise ValueError("frames must be a non-empty 3D array shaped (n_frames, height, width)")
    if ts is None:
        ts_arr = np.arange(arr.shape[0], dtype=float)
    else:
        ts_arr = np.asarray(ts, dtype=float)
        if ts_arr.shape != (arr.shape[0],):
            raise ValueError("ts must have one timestamp per frame")

    metrics = [
        analyze_diffuse_scattering(
            frame,
            signal_roi=signal_roi,
            diffuse_roi=diffuse_roi,
            background_roi=background_roi,
            halo_inner_scale=halo_inner_scale,
            halo_outer_scale=halo_outer_scale,
        )
        for frame in arr
    ]

    return DiffuseSeriesMetrics(
        ts=ts_arr,
        diffuse_sum=np.asarray([item.diffuse_sum for item in metrics], dtype=float),
        diffuse_mean=np.asarray([item.diffuse_mean for item in metrics], dtype=float),
        diffuse_std=np.asarray([item.diffuse_std for item in metrics], dtype=float),
        diffuse_to_signal_ratio=np.asarray([item.diffuse_to_signal_ratio for item in metrics], dtype=float),
        halo_sum=np.asarray([item.halo_sum for item in metrics], dtype=float),
        halo_mean=np.asarray([item.halo_mean for item in metrics], dtype=float),
        halo_ratio=np.asarray([item.halo_ratio for item in metrics], dtype=float),
        anisotropy=np.asarray([item.anisotropy for item in metrics], dtype=float),
    )


__all__ = [
    "DiffuseFrameMetrics",
    "DiffuseSeriesMetrics",
    "analyze_diffuse_scattering",
    "analyze_diffuse_scattering_series",
]

