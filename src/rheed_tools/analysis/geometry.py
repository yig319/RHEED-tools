from __future__ import annotations

"""Geometry-focused RHEED analysis for spots, streaks, tilt, and spacing."""

from dataclasses import dataclass

import numpy as np

from .diffraction_2d import analyze_rheed_frame
from .roi import sanitize_roi

_FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))


@dataclass(slots=True)
class PeakSpacingMetrics:
    """Peak positions and spacing inferred from a 1D line profile."""

    peak_positions_px: np.ndarray
    peak_heights: np.ndarray
    spacing_px: float | None
    spacing_std_px: float | None


@dataclass(slots=True)
class GeometryFrameMetrics:
    """Geometry summary for one diffraction ROI."""

    roi: tuple[int, int, int, int]
    center_x: float
    center_y: float
    horizontal_fwhm_px: float
    vertical_fwhm_px: float
    major_fwhm_px: float
    minor_fwhm_px: float
    aspect_ratio: float
    tilt_deg: float
    streak_length_px: float
    streak_width_px: float
    spacing_px: float | None
    spacing_std_px: float | None
    split_detected: bool
    split_separation_px: float | None
    split_balance: float | None
    peak_positions_px: np.ndarray


@dataclass(slots=True)
class GeometrySeriesMetrics:
    """Frame-by-frame geometry metrics across a diffraction movie."""

    ts: np.ndarray
    center_x: np.ndarray
    center_y: np.ndarray
    horizontal_fwhm_px: np.ndarray
    vertical_fwhm_px: np.ndarray
    major_fwhm_px: np.ndarray
    minor_fwhm_px: np.ndarray
    aspect_ratio: np.ndarray
    tilt_deg: np.ndarray
    streak_length_px: np.ndarray
    streak_width_px: np.ndarray
    spacing_px: np.ndarray
    spacing_std_px: np.ndarray
    split_detected: np.ndarray
    split_separation_px: np.ndarray
    split_balance: np.ndarray


def extract_axis_profile(
    frame: np.ndarray,
    *,
    roi: tuple[int, int, int, int] | None = None,
    background_roi: tuple[int, int, int, int] | None = None,
    axis: str = "x",
    reducer: str = "mean",
) -> np.ndarray:
    """Return a background-corrected 1D profile from a diffraction ROI.

    This is the base helper used by spacing and reconstruction analysis.

    Tuning guidance:
    - `axis="x"` averages rows and keeps horizontal peak spacing.
    - `axis="y"` averages columns and keeps vertical peak spacing.
    - `reducer="mean"` is smoother and better for weak streaks.
    - `reducer="sum"` emphasizes bright peaks when the ROI height or width is
      already well controlled.
    """

    metrics = analyze_rheed_frame(frame, roi=roi, background_roi=background_roi)
    arr = np.asarray(frame, dtype=float)
    y0, y1, x0, x1 = metrics.roi
    patch = np.clip(arr[y0:y1, x0:x1] - metrics.background_mean, 0.0, None)
    if axis not in {"x", "y"}:
        raise ValueError("axis must be 'x' or 'y'")
    if reducer not in {"mean", "sum", "max"}:
        raise ValueError("reducer must be 'mean', 'sum', or 'max'")
    axis_index = 0 if axis == "x" else 1
    if reducer == "mean":
        return np.mean(patch, axis=axis_index)
    if reducer == "sum":
        return np.sum(patch, axis=axis_index)
    return np.max(patch, axis=axis_index)


def measure_profile_spacing(
    profile: np.ndarray,
    *,
    min_rel_height: float = 0.2,
    min_distance_px: int = 6,
) -> PeakSpacingMetrics:
    """Detect bright peaks in a line profile and estimate their spacing.

    This is a lightweight first-pass spacing estimator for streak spacing,
    reconstruction spacing, or reciprocal-space trends.

    Tuning guidance:
    - raise `min_rel_height` when noise creates too many weak peaks
    - lower `min_rel_height` when real side peaks are being missed
    - increase `min_distance_px` if one broad streak produces duplicate peaks
    """

    arr = np.asarray(profile, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("profile must be a non-empty 1D array")
    if not (0.0 <= min_rel_height < 1.0):
        raise ValueError("min_rel_height must be between 0 and 1")
    if min_distance_px < 1:
        raise ValueError("min_distance_px must be >= 1")

    try:
        from scipy.signal import find_peaks
    except ImportError as exc:
        raise ImportError(
            "scipy is required for measure_profile_spacing(). Install notebook/dev dependencies first."
        ) from exc

    shifted = arr - float(np.min(arr))
    peak_height = float(np.max(shifted)) * float(min_rel_height)
    if peak_height <= 0.0:
        return PeakSpacingMetrics(
            peak_positions_px=np.asarray([], dtype=float),
            peak_heights=np.asarray([], dtype=float),
            spacing_px=None,
            spacing_std_px=None,
        )

    peak_idx, props = find_peaks(shifted, height=peak_height, distance=int(min_distance_px))
    if peak_idx.size == 0:
        return PeakSpacingMetrics(
            peak_positions_px=np.asarray([], dtype=float),
            peak_heights=np.asarray([], dtype=float),
            spacing_px=None,
            spacing_std_px=None,
        )

    positions = peak_idx.astype(float)
    heights = np.asarray(props["peak_heights"], dtype=float)
    if positions.size < 2:
        spacing = None
        spacing_std = None
    else:
        diffs = np.diff(np.sort(positions))
        spacing = float(np.median(diffs))
        spacing_std = float(np.std(diffs))

    return PeakSpacingMetrics(
        peak_positions_px=positions,
        peak_heights=heights,
        spacing_px=spacing,
        spacing_std_px=spacing_std,
    )


def measure_peak_spacing(
    frame: np.ndarray,
    *,
    roi: tuple[int, int, int, int] | None = None,
    background_roi: tuple[int, int, int, int] | None = None,
    axis: str = "x",
    reducer: str = "mean",
    min_rel_height: float = 0.2,
    min_distance_px: int = 6,
) -> PeakSpacingMetrics:
    """Measure peak spacing directly from a 2D diffraction image."""

    profile = extract_axis_profile(
        frame,
        roi=roi,
        background_roi=background_roi,
        axis=axis,
        reducer=reducer,
    )
    return measure_profile_spacing(
        profile,
        min_rel_height=min_rel_height,
        min_distance_px=min_distance_px,
    )


def _profile_width(profile: np.ndarray, threshold_fraction: float) -> float:
    arr = np.asarray(profile, dtype=float)
    if arr.size == 0:
        return 0.0
    arr = arr - float(np.min(arr))
    peak = float(np.max(arr))
    if peak <= 0.0:
        return 0.0
    mask = arr >= peak * float(threshold_fraction)
    if not np.any(mask):
        return 0.0
    idx = np.flatnonzero(mask)
    return float(idx[-1] - idx[0] + 1)


def _split_summary(
    profile: np.ndarray,
    *,
    min_rel_height: float,
    min_distance_px: int,
) -> tuple[bool, float | None, float | None]:
    spacing = measure_profile_spacing(
        profile,
        min_rel_height=min_rel_height,
        min_distance_px=min_distance_px,
    )
    if spacing.peak_positions_px.size < 2:
        return False, None, None

    order = np.argsort(spacing.peak_heights)[::-1]
    top_a = int(order[0])
    top_b = int(order[1])
    pos_a = float(spacing.peak_positions_px[top_a])
    pos_b = float(spacing.peak_positions_px[top_b])
    height_a = float(spacing.peak_heights[top_a])
    height_b = float(spacing.peak_heights[top_b])
    split_separation = abs(pos_a - pos_b)
    split_balance = min(height_a, height_b) / (max(height_a, height_b) + 1e-9)
    return split_separation > 0.0, float(split_separation), float(split_balance)


def measure_spot_streak_geometry(
    frame: np.ndarray,
    *,
    roi: tuple[int, int, int, int] | None = None,
    background_roi: tuple[int, int, int, int] | None = None,
    width_fraction: float = 0.2,
    spacing_axis: str = "x",
    spacing_reducer: str = "mean",
    spacing_min_rel_height: float = 0.2,
    spacing_min_distance_px: int = 6,
    split_prominence_rel: float = 0.25,
) -> GeometryFrameMetrics:
    """Summarize one diffraction feature as spot/streak geometry.

    This is the main entry point for:
    - streak length and width
    - elongation / aspect ratio
    - tilt angle
    - line-profile peak spacing
    - simple split-streak heuristics

    Tuning guidance:
    - `roi` is the most important parameter. Use it to isolate one spot family.
    - `width_fraction` controls how aggressively width is measured from the
      profile tails. Lower values keep more diffuse signal.
    - `spacing_axis` chooses whether spacing is extracted horizontally or
      vertically.
    - `split_prominence_rel` should be raised when noise causes false split
      detections.
    """

    if not (0.0 < width_fraction < 1.0):
        raise ValueError("width_fraction must be between 0 and 1")
    if not (0.0 < split_prominence_rel < 1.0):
        raise ValueError("split_prominence_rel must be between 0 and 1")

    base = analyze_rheed_frame(frame, roi=roi, background_roi=background_roi)
    arr = np.asarray(frame, dtype=float)
    y0, y1, x0, x1 = base.roi
    patch = np.clip(arr[y0:y1, x0:x1] - base.background_mean, 0.0, None)
    horizontal_profile = np.mean(patch, axis=0)
    vertical_profile = np.mean(patch, axis=1)

    horizontal_width = _profile_width(horizontal_profile, threshold_fraction=width_fraction)
    vertical_width = _profile_width(vertical_profile, threshold_fraction=width_fraction)

    spacing_metrics = measure_peak_spacing(
        frame,
        roi=base.roi,
        background_roi=background_roi,
        axis=spacing_axis,
        reducer=spacing_reducer,
        min_rel_height=spacing_min_rel_height,
        min_distance_px=spacing_min_distance_px,
    )
    split_detected, split_separation, split_balance = _split_summary(
        horizontal_profile if spacing_axis == "x" else vertical_profile,
        min_rel_height=split_prominence_rel,
        min_distance_px=spacing_min_distance_px,
    )

    return GeometryFrameMetrics(
        roi=base.roi,
        center_x=float(base.centroid_x),
        center_y=float(base.centroid_y),
        horizontal_fwhm_px=float(horizontal_width),
        vertical_fwhm_px=float(vertical_width),
        major_fwhm_px=float(max(base.fwhm_x, base.fwhm_y)),
        minor_fwhm_px=float(min(base.fwhm_x, base.fwhm_y)),
        aspect_ratio=float(base.aspect_ratio),
        tilt_deg=float(base.orientation_deg),
        streak_length_px=float(max(horizontal_width, vertical_width)),
        streak_width_px=float(min(horizontal_width, vertical_width)),
        spacing_px=spacing_metrics.spacing_px,
        spacing_std_px=spacing_metrics.spacing_std_px,
        split_detected=bool(split_detected),
        split_separation_px=split_separation,
        split_balance=split_balance,
        peak_positions_px=spacing_metrics.peak_positions_px,
    )


def measure_spot_streak_geometry_series(
    frames: np.ndarray,
    *,
    ts: np.ndarray | None = None,
    roi: tuple[int, int, int, int] | None = None,
    background_roi: tuple[int, int, int, int] | None = None,
    width_fraction: float = 0.2,
    spacing_axis: str = "x",
    spacing_reducer: str = "mean",
    spacing_min_rel_height: float = 0.2,
    spacing_min_distance_px: int = 6,
    split_prominence_rel: float = 0.25,
) -> GeometrySeriesMetrics:
    """Run geometry analysis across a frame stack."""

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
        measure_spot_streak_geometry(
            frame,
            roi=roi,
            background_roi=background_roi,
            width_fraction=width_fraction,
            spacing_axis=spacing_axis,
            spacing_reducer=spacing_reducer,
            spacing_min_rel_height=spacing_min_rel_height,
            spacing_min_distance_px=spacing_min_distance_px,
            split_prominence_rel=split_prominence_rel,
        )
        for frame in arr
    ]

    def optional_array(name: str) -> np.ndarray:
        values = [getattr(item, name) for item in metrics]
        return np.asarray([np.nan if value is None else float(value) for value in values], dtype=float)

    return GeometrySeriesMetrics(
        ts=ts_arr,
        center_x=np.asarray([item.center_x for item in metrics], dtype=float),
        center_y=np.asarray([item.center_y for item in metrics], dtype=float),
        horizontal_fwhm_px=np.asarray([item.horizontal_fwhm_px for item in metrics], dtype=float),
        vertical_fwhm_px=np.asarray([item.vertical_fwhm_px for item in metrics], dtype=float),
        major_fwhm_px=np.asarray([item.major_fwhm_px for item in metrics], dtype=float),
        minor_fwhm_px=np.asarray([item.minor_fwhm_px for item in metrics], dtype=float),
        aspect_ratio=np.asarray([item.aspect_ratio for item in metrics], dtype=float),
        tilt_deg=np.asarray([item.tilt_deg for item in metrics], dtype=float),
        streak_length_px=np.asarray([item.streak_length_px for item in metrics], dtype=float),
        streak_width_px=np.asarray([item.streak_width_px for item in metrics], dtype=float),
        spacing_px=optional_array("spacing_px"),
        spacing_std_px=optional_array("spacing_std_px"),
        split_detected=np.asarray([item.split_detected for item in metrics], dtype=bool),
        split_separation_px=optional_array("split_separation_px"),
        split_balance=optional_array("split_balance"),
    )


__all__ = [
    "GeometryFrameMetrics",
    "GeometrySeriesMetrics",
    "PeakSpacingMetrics",
    "extract_axis_profile",
    "measure_peak_spacing",
    "measure_profile_spacing",
    "measure_spot_streak_geometry",
    "measure_spot_streak_geometry_series",
]

