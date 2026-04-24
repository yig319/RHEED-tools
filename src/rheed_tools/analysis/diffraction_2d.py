from __future__ import annotations

"""Offline diffraction-image analysis for spots, streaks, and drift."""

from dataclasses import dataclass

import numpy as np

from .morphology import describe_shape, weighted_moments
from .roi import recenter_roi, sanitize_roi


@dataclass(slots=True)
class RoiFrameMetrics:
    """ROI-only intensity metrics for one diffraction frame."""

    roi: tuple[int, int, int, int]
    background_roi: tuple[int, int, int, int]
    raw_sum: float
    raw_mean: float
    background_mean: float
    corrected_sum: float
    corrected_mean: float
    contrast: float


@dataclass(slots=True)
class FrameMetrics(RoiFrameMetrics):
    """Background-corrected spot/streak metrics for one diffraction frame."""

    centroid_x: float
    centroid_y: float
    sigma_x: float
    sigma_y: float
    fwhm_x: float
    fwhm_y: float
    aspect_ratio: float
    orientation_deg: float
    streakiness: float


@dataclass(slots=True)
class RoiSeriesMetrics:
    """ROI-only intensity traces across a time series of diffraction frames."""

    ts: np.ndarray
    raw_sum: np.ndarray
    raw_mean: np.ndarray
    corrected_sum: np.ndarray
    corrected_mean: np.ndarray
    background_mean: np.ndarray
    contrast: np.ndarray


@dataclass(slots=True)
class FrameSeriesMetrics(RoiSeriesMetrics):
    """Stacked image-derived metrics across a time series of frames."""

    centroid_x: np.ndarray
    centroid_y: np.ndarray
    fwhm_x: np.ndarray
    fwhm_y: np.ndarray
    aspect_ratio: np.ndarray
    orientation_deg: np.ndarray
    streakiness: np.ndarray


@dataclass(slots=True)
class SpotTrack:
    """Tracking result for one drifting diffraction feature."""

    ts: np.ndarray
    roi_history: list[tuple[int, int, int, int]]
    centroid_x: np.ndarray
    centroid_y: np.ndarray
    corrected_sum: np.ndarray
    aspect_ratio: np.ndarray
    streakiness: np.ndarray


def analyze_roi_frame(
    frame: np.ndarray,
    roi: tuple[int, int, int, int] | None = None,
    background_roi: tuple[int, int, int, int] | None = None,
) -> RoiFrameMetrics:
    """Extract ROI-only intensity metrics from one RHEED frame.

    Use this first when you are still deciding which diffraction feature is
    worth following. It keeps the analysis limited to the user-chosen box and
    does not assume centroid or shape tracking is meaningful yet.
    """

    img = np.asarray(frame, dtype=float)
    if img.ndim != 2:
        raise ValueError("frame must be a 2D array")
    if img.size == 0:
        raise ValueError("frame must not be empty")

    y0, y1, x0, x1 = sanitize_roi(img.shape, roi, fraction=0.18)
    bg_y0, bg_y1, bg_x0, bg_x1 = sanitize_roi(img.shape, background_roi, fraction=0.14, corner="top_left")

    roi_img = img[y0:y1, x0:x1]
    bg_img = img[bg_y0:bg_y1, bg_x0:bg_x1]
    background_mean = float(np.mean(bg_img))
    corrected_roi = roi_img - background_mean

    raw_sum = float(np.sum(roi_img))
    raw_mean = float(np.mean(roi_img))
    corrected_sum = float(np.sum(corrected_roi))
    corrected_mean = float(np.mean(corrected_roi))
    contrast = float((raw_mean - background_mean) / (abs(background_mean) + 1e-9))

    return RoiFrameMetrics(
        roi=(y0, y1, x0, x1),
        background_roi=(bg_y0, bg_y1, bg_x0, bg_x1),
        raw_sum=raw_sum,
        raw_mean=raw_mean,
        background_mean=background_mean,
        corrected_sum=corrected_sum,
        corrected_mean=corrected_mean,
        contrast=contrast,
    )


def analyze_rheed_frame(
    frame: np.ndarray,
    roi: tuple[int, int, int, int] | None = None,
    background_roi: tuple[int, int, int, int] | None = None,
) -> FrameMetrics:
    """Extract intensity, drift, and morphology metrics from one RHEED frame."""

    roi_metrics = analyze_roi_frame(frame, roi=roi, background_roi=background_roi)
    img = np.asarray(frame, dtype=float)
    y0, y1, x0, x1 = roi_metrics.roi
    roi_img = img[y0:y1, x0:x1]
    corrected_roi = roi_img - roi_metrics.background_mean
    corrected_positive = np.clip(corrected_roi, 0.0, None)

    centroid_x, centroid_y, _, _, _ = weighted_moments(corrected_positive)
    shape = describe_shape(corrected_positive)
    centroid_x += x0
    centroid_y += y0

    return FrameMetrics(
        roi=roi_metrics.roi,
        background_roi=roi_metrics.background_roi,
        raw_sum=roi_metrics.raw_sum,
        raw_mean=roi_metrics.raw_mean,
        background_mean=roi_metrics.background_mean,
        corrected_sum=roi_metrics.corrected_sum,
        corrected_mean=roi_metrics.corrected_mean,
        contrast=roi_metrics.contrast,
        centroid_x=float(centroid_x),
        centroid_y=float(centroid_y),
        sigma_x=float(shape.sigma_x),
        sigma_y=float(shape.sigma_y),
        fwhm_x=float(shape.fwhm_x),
        fwhm_y=float(shape.fwhm_y),
        aspect_ratio=float(shape.aspect_ratio),
        orientation_deg=float(shape.orientation_deg),
        streakiness=float(shape.streakiness),
    )


def analyze_roi_frames(
    frames: np.ndarray,
    ts: np.ndarray | None = None,
    roi: tuple[int, int, int, int] | None = None,
    background_roi: tuple[int, int, int, int] | None = None,
) -> RoiSeriesMetrics:
    """Convert a stack of RHEED frames into ROI-only intensity time traces."""

    arr = np.asarray(frames, dtype=float)
    if arr.ndim != 3:
        raise ValueError("frames must be a 3D array shaped (n_frames, height, width)")

    if ts is None:
        ts_arr = np.arange(arr.shape[0], dtype=float)
    else:
        ts_arr = np.asarray(ts, dtype=float)
        if ts_arr.shape != (arr.shape[0],):
            raise ValueError("ts must have one timestamp per frame")

    metrics = [analyze_roi_frame(frame, roi=roi, background_roi=background_roi) for frame in arr]
    return RoiSeriesMetrics(
        ts=ts_arr,
        raw_sum=np.asarray([item.raw_sum for item in metrics], dtype=float),
        raw_mean=np.asarray([item.raw_mean for item in metrics], dtype=float),
        corrected_sum=np.asarray([item.corrected_sum for item in metrics], dtype=float),
        corrected_mean=np.asarray([item.corrected_mean for item in metrics], dtype=float),
        background_mean=np.asarray([item.background_mean for item in metrics], dtype=float),
        contrast=np.asarray([item.contrast for item in metrics], dtype=float),
    )


def analyze_rheed_frames(
    frames: np.ndarray,
    ts: np.ndarray | None = None,
    roi: tuple[int, int, int, int] | None = None,
    background_roi: tuple[int, int, int, int] | None = None,
) -> FrameSeriesMetrics:
    """Convert a stack of RHEED frames into image-derived time traces."""

    roi_series = analyze_roi_frames(frames, ts=ts, roi=roi, background_roi=background_roi)
    arr = np.asarray(frames, dtype=float)
    metrics = [analyze_rheed_frame(frame, roi=roi, background_roi=background_roi) for frame in arr]
    return FrameSeriesMetrics(
        ts=roi_series.ts,
        raw_sum=roi_series.raw_sum,
        raw_mean=roi_series.raw_mean,
        corrected_sum=roi_series.corrected_sum,
        corrected_mean=roi_series.corrected_mean,
        background_mean=roi_series.background_mean,
        contrast=roi_series.contrast,
        centroid_x=np.asarray([item.centroid_x for item in metrics], dtype=float),
        centroid_y=np.asarray([item.centroid_y for item in metrics], dtype=float),
        fwhm_x=np.asarray([item.fwhm_x for item in metrics], dtype=float),
        fwhm_y=np.asarray([item.fwhm_y for item in metrics], dtype=float),
        aspect_ratio=np.asarray([item.aspect_ratio for item in metrics], dtype=float),
        orientation_deg=np.asarray([item.orientation_deg for item in metrics], dtype=float),
        streakiness=np.asarray([item.streakiness for item in metrics], dtype=float),
    )


def track_diffraction_spot(
    frames: np.ndarray,
    ts: np.ndarray | None = None,
    initial_roi: tuple[int, int, int, int] | None = None,
    search_margin_px: int = 8,
    background_roi: tuple[int, int, int, int] | None = None,
) -> SpotTrack:
    """Track a drifting diffraction feature by recentering the ROI each frame."""

    arr = np.asarray(frames, dtype=float)
    if arr.ndim != 3:
        raise ValueError("frames must be a 3D array shaped (n_frames, height, width)")
    if arr.shape[0] == 0:
        raise ValueError("frames must contain at least one frame")

    if ts is None:
        ts_arr = np.arange(arr.shape[0], dtype=float)
    else:
        ts_arr = np.asarray(ts, dtype=float)
        if ts_arr.shape != (arr.shape[0],):
            raise ValueError("ts must have one timestamp per frame")

    roi = sanitize_roi(arr.shape[1:], initial_roi, fraction=0.18)
    box_height = roi[1] - roi[0]
    box_width = roi[3] - roi[2]

    roi_history: list[tuple[int, int, int, int]] = []
    centroid_x: list[float] = []
    centroid_y: list[float] = []
    corrected_sum: list[float] = []
    aspect_ratio: list[float] = []
    streakiness: list[float] = []

    for frame in arr:
        metrics = analyze_rheed_frame(frame, roi=roi, background_roi=background_roi)
        roi_history.append(metrics.roi)
        centroid_x.append(metrics.centroid_x)
        centroid_y.append(metrics.centroid_y)
        corrected_sum.append(metrics.corrected_sum)
        aspect_ratio.append(metrics.aspect_ratio)
        streakiness.append(metrics.streakiness)

        roi = recenter_roi(
            arr.shape[1:],
            center_y=metrics.centroid_y,
            center_x=metrics.centroid_x,
            box_height=box_height + 2 * int(search_margin_px),
            box_width=box_width + 2 * int(search_margin_px),
        )
        box_height = metrics.roi[1] - metrics.roi[0]
        box_width = metrics.roi[3] - metrics.roi[2]

    return SpotTrack(
        ts=ts_arr,
        roi_history=roi_history,
        centroid_x=np.asarray(centroid_x, dtype=float),
        centroid_y=np.asarray(centroid_y, dtype=float),
        corrected_sum=np.asarray(corrected_sum, dtype=float),
        aspect_ratio=np.asarray(aspect_ratio, dtype=float),
        streakiness=np.asarray(streakiness, dtype=float),
    )

