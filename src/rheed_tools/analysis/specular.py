from __future__ import annotations

"""Specular-spot analysis for intensity, drift, width, and oscillation damping."""

from dataclasses import dataclass

import numpy as np

from .diffraction_2d import analyze_roi_frame
from .spot_fit import extract_spot_patch, fit_gaussian_2d, locate_and_fit_spot
from .trace_1d import OscillationSummary, summarize_oscillation_signal


@dataclass(slots=True)
class SpecularFrameMetrics:
    """Intensity and Gaussian-fit metrics for one specular spot ROI."""

    roi: tuple[int, int, int, int]
    background_roi: tuple[int, int, int, int]
    raw_sum: float
    corrected_sum: float
    raw_mean: float
    raw_min: float
    raw_max: float
    raw_std: float
    background_mean: float
    center_x: float
    center_y: float
    fwhm_x: float
    fwhm_y: float
    rotation_deg: float
    rmse: float


@dataclass(slots=True)
class SpecularSeriesMetrics:
    """Time-series specular metrics plus an oscillation summary."""

    roi: tuple[int, int, int, int]
    ts: np.ndarray
    raw_sum: np.ndarray
    corrected_sum: np.ndarray
    raw_mean: np.ndarray
    raw_min: np.ndarray
    raw_max: np.ndarray
    raw_std: np.ndarray
    background_mean: np.ndarray
    center_x: np.ndarray
    center_y: np.ndarray
    fwhm_x: np.ndarray
    fwhm_y: np.ndarray
    rotation_deg: np.ndarray
    rmse: np.ndarray
    oscillation_summary: OscillationSummary


def analyze_specular_frame(
    frame: np.ndarray,
    *,
    roi: tuple[int, int, int, int] | None = None,
    background_roi: tuple[int, int, int, int] | None = None,
    expected_center: tuple[float, float] | None = None,
    patch_size: tuple[int, int] = (31, 31),
    allow_rotation: bool = False,
) -> SpecularFrameMetrics:
    """Analyze one specular spot region.

    Recommended usage:
    - if you already know the spot ROI, pass `roi`
    - if you only know the approximate center, pass `expected_center`
    - keep `allow_rotation=False` for compact specular spots unless you expect
      a strongly elongated feature
    """

    arr = np.asarray(frame, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("frame must be a non-empty 2D array")

    if roi is None and expected_center is not None:
        _, roi = extract_spot_patch(
            arr,
            center_x=float(expected_center[0]),
            center_y=float(expected_center[1]),
            patch_size=patch_size,
        )
    elif roi is None:
        _, fit = locate_and_fit_spot(
            arr,
            search_roi=None,
            threshold_rel=0.35,
            min_distance_px=max(4, patch_size[0] // 5),
            patch_size=patch_size,
            allow_rotation=allow_rotation,
        )
        roi = fit.roi

    roi_metrics = analyze_roi_frame(arr, roi=roi, background_roi=background_roi)
    y0, y1, x0, x1 = roi_metrics.roi
    patch = arr[y0:y1, x0:x1]
    fit = fit_gaussian_2d(patch, allow_rotation=allow_rotation)

    return SpecularFrameMetrics(
        roi=roi_metrics.roi,
        background_roi=roi_metrics.background_roi,
        raw_sum=float(np.sum(patch)),
        corrected_sum=float(roi_metrics.corrected_sum),
        raw_mean=float(np.mean(patch)),
        raw_min=float(np.min(patch)),
        raw_max=float(np.max(patch)),
        raw_std=float(np.std(patch)),
        background_mean=float(roi_metrics.background_mean),
        center_x=float(x0 + fit.center_x),
        center_y=float(y0 + fit.center_y),
        fwhm_x=float(fit.fwhm_x),
        fwhm_y=float(fit.fwhm_y),
        rotation_deg=float(fit.rotation_deg),
        rmse=float(fit.rmse),
    )


def _detect_trace_peaks(trace: np.ndarray) -> np.ndarray:
    try:
        from scipy.signal import find_peaks
    except ImportError as exc:
        raise ImportError(
            "scipy is required for analyze_specular_series(). Install notebook/dev dependencies first."
        ) from exc

    y = np.asarray(trace, dtype=float)
    shifted = y - float(np.min(y))
    if shifted.size < 4 or float(np.max(shifted)) <= 0:
        return np.asarray([], dtype=int)
    prominence = max(float(np.std(shifted) * 0.35), float(np.max(shifted) * 0.08))
    peak_idx, _ = find_peaks(shifted, prominence=prominence)
    return np.asarray(peak_idx, dtype=int)


def analyze_specular_series(
    frames: np.ndarray,
    *,
    ts: np.ndarray | None = None,
    roi: tuple[int, int, int, int] | None = None,
    background_roi: tuple[int, int, int, int] | None = None,
    expected_center: tuple[float, float] | None = None,
    patch_size: tuple[int, int] = (31, 31),
    allow_rotation: bool = False,
    fit_every_n: int = 1,
) -> SpecularSeriesMetrics:
    """Track specular intensity and width over time.

    Tuning guidance:
    - `roi` should be the fixed analysis region chosen earlier in the workflow
    - `fit_every_n` is the main speed control for long videos
    - when only intensity trends matter, increase `fit_every_n`
    """

    arr = np.asarray(frames, dtype=float)
    if arr.ndim != 3 or arr.shape[0] == 0:
        raise ValueError("frames must be a non-empty 3D array shaped (n_frames, height, width)")
    if fit_every_n < 1:
        raise ValueError("fit_every_n must be >= 1")
    if ts is None:
        ts_arr = np.arange(arr.shape[0], dtype=float)
    else:
        ts_arr = np.asarray(ts, dtype=float)
        if ts_arr.shape != (arr.shape[0],):
            raise ValueError("ts must have one timestamp per frame")

    frame_metrics = [
        analyze_specular_frame(
            frame,
            roi=roi,
            background_roi=background_roi,
            expected_center=expected_center,
            patch_size=patch_size,
            allow_rotation=allow_rotation,
        )
        for frame in arr
    ]

    center_x = np.full(arr.shape[0], np.nan, dtype=float)
    center_y = np.full(arr.shape[0], np.nan, dtype=float)
    fwhm_x = np.full(arr.shape[0], np.nan, dtype=float)
    fwhm_y = np.full(arr.shape[0], np.nan, dtype=float)
    rotation_deg = np.full(arr.shape[0], np.nan, dtype=float)
    rmse = np.full(arr.shape[0], np.nan, dtype=float)
    for idx, metrics in enumerate(frame_metrics):
        if idx % fit_every_n == 0:
            center_x[idx] = metrics.center_x
            center_y[idx] = metrics.center_y
            fwhm_x[idx] = metrics.fwhm_x
            fwhm_y[idx] = metrics.fwhm_y
            rotation_deg[idx] = metrics.rotation_deg
            rmse[idx] = metrics.rmse

    corrected_sum = np.asarray([item.corrected_sum for item in frame_metrics], dtype=float)
    peaks = _detect_trace_peaks(corrected_sum)
    oscillation_summary = summarize_oscillation_signal(ts_arr, corrected_sum, peak_indices=peaks)

    return SpecularSeriesMetrics(
        roi=frame_metrics[0].roi,
        ts=ts_arr,
        raw_sum=np.asarray([item.raw_sum for item in frame_metrics], dtype=float),
        corrected_sum=corrected_sum,
        raw_mean=np.asarray([item.raw_mean for item in frame_metrics], dtype=float),
        raw_min=np.asarray([item.raw_min for item in frame_metrics], dtype=float),
        raw_max=np.asarray([item.raw_max for item in frame_metrics], dtype=float),
        raw_std=np.asarray([item.raw_std for item in frame_metrics], dtype=float),
        background_mean=np.asarray([item.background_mean for item in frame_metrics], dtype=float),
        center_x=center_x,
        center_y=center_y,
        fwhm_x=fwhm_x,
        fwhm_y=fwhm_y,
        rotation_deg=rotation_deg,
        rmse=rmse,
        oscillation_summary=oscillation_summary,
    )


__all__ = [
    "SpecularFrameMetrics",
    "SpecularSeriesMetrics",
    "analyze_specular_frame",
    "analyze_specular_series",
]

