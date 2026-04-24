from __future__ import annotations

"""Surface-reconstruction helpers for fractional-order streak detection."""

from dataclasses import dataclass

import numpy as np

from .geometry import extract_axis_profile, measure_profile_spacing


@dataclass(slots=True)
class FractionalOrderPeak:
    """One detected fractional-order peak in a diffraction profile."""

    position_px: float
    height: float
    relative_order: float
    nearest_fractional_order: float
    distance_to_fractional_order: float


@dataclass(slots=True)
class ReconstructionFrameMetrics:
    """Surface reconstruction summary for one frame."""

    fundamental_spacing_px: float | None
    fractional_peaks: list[FractionalOrderPeak]
    fractional_peak_count: int
    strongest_fractional_ratio: float
    reconstruction_present: bool


@dataclass(slots=True)
class ReconstructionSeriesMetrics:
    """Frame-by-frame reconstruction metrics."""

    ts: np.ndarray
    fundamental_spacing_px: np.ndarray
    fractional_peak_count: np.ndarray
    strongest_fractional_ratio: np.ndarray
    reconstruction_present: np.ndarray


def detect_fractional_order_peaks(
    profile: np.ndarray,
    *,
    fundamental_spacing_px: float,
    center_index: float | None = None,
    fractional_orders: tuple[float, ...] = (0.5, 1.0 / 3.0, 2.0 / 3.0),
    tolerance_fraction: float = 0.12,
    min_rel_height: float = 0.12,
    min_distance_px: int = 4,
) -> list[FractionalOrderPeak]:
    """Detect half-order or fractional-order peaks in a line profile.

    Tuning guidance:
    - pass a known `fundamental_spacing_px` when you already know the main-order
      spacing from geometry or calibration
    - increase `tolerance_fraction` slightly when peaks drift during growth
    - raise `min_rel_height` when diffuse background creates false fractional
      peaks
    """

    if fundamental_spacing_px <= 0:
        raise ValueError("fundamental_spacing_px must be > 0")
    if center_index is None:
        center_index = float(np.argmax(profile))

    spacing = measure_profile_spacing(
        profile,
        min_rel_height=min_rel_height,
        min_distance_px=min_distance_px,
    )
    if spacing.peak_positions_px.size == 0:
        return []

    allowed = np.asarray(sorted(set(float(value) for value in fractional_orders)), dtype=float)
    peaks: list[FractionalOrderPeak] = []
    max_height = float(np.max(spacing.peak_heights)) if spacing.peak_heights.size else 1.0
    for position, height in zip(spacing.peak_positions_px, spacing.peak_heights):
        relative_order = abs(float(position) - float(center_index)) / float(fundamental_spacing_px)
        nearest_integer = np.round(relative_order)
        fractional_part = abs(relative_order - nearest_integer)
        nearest_fraction = float(allowed[np.argmin(np.abs(allowed - fractional_part))])
        distance = abs(fractional_part - nearest_fraction)
        if distance <= tolerance_fraction and height > max_height * min_rel_height:
            peaks.append(
                FractionalOrderPeak(
                    position_px=float(position),
                    height=float(height),
                    relative_order=float(relative_order),
                    nearest_fractional_order=float(nearest_fraction),
                    distance_to_fractional_order=float(distance),
                )
            )
    return peaks


def analyze_surface_reconstruction(
    frame: np.ndarray,
    *,
    roi: tuple[int, int, int, int] | None = None,
    background_roi: tuple[int, int, int, int] | None = None,
    axis: str = "x",
    reducer: str = "mean",
    expected_spacing_px: float | None = None,
    fractional_orders: tuple[float, ...] = (0.5, 1.0 / 3.0, 2.0 / 3.0),
    tolerance_fraction: float = 0.12,
    min_rel_height: float = 0.12,
    min_distance_px: int = 4,
) -> ReconstructionFrameMetrics:
    """Analyze one frame for fractional-order streaks or reconstruction peaks."""

    profile = extract_axis_profile(
        frame,
        roi=roi,
        background_roi=background_roi,
        axis=axis,
        reducer=reducer,
    )
    spacing = measure_profile_spacing(
        profile,
        min_rel_height=max(min_rel_height, 0.2),
        min_distance_px=max(min_distance_px, 4),
    )
    fundamental_spacing = expected_spacing_px if expected_spacing_px is not None else spacing.spacing_px
    if fundamental_spacing is None:
        return ReconstructionFrameMetrics(
            fundamental_spacing_px=None,
            fractional_peaks=[],
            fractional_peak_count=0,
            strongest_fractional_ratio=0.0,
            reconstruction_present=False,
        )

    center_index = float(np.argmax(profile))
    peaks = detect_fractional_order_peaks(
        profile,
        fundamental_spacing_px=float(fundamental_spacing),
        center_index=center_index,
        fractional_orders=fractional_orders,
        tolerance_fraction=tolerance_fraction,
        min_rel_height=min_rel_height,
        min_distance_px=min_distance_px,
    )
    max_profile = float(np.max(profile)) if profile.size else 1.0
    strongest_fractional_ratio = 0.0 if not peaks else float(max(item.height for item in peaks) / (max_profile + 1e-9))

    return ReconstructionFrameMetrics(
        fundamental_spacing_px=float(fundamental_spacing),
        fractional_peaks=peaks,
        fractional_peak_count=len(peaks),
        strongest_fractional_ratio=strongest_fractional_ratio,
        reconstruction_present=bool(peaks),
    )


def analyze_surface_reconstruction_series(
    frames: np.ndarray,
    *,
    ts: np.ndarray | None = None,
    roi: tuple[int, int, int, int] | None = None,
    background_roi: tuple[int, int, int, int] | None = None,
    axis: str = "x",
    reducer: str = "mean",
    expected_spacing_px: float | None = None,
    fractional_orders: tuple[float, ...] = (0.5, 1.0 / 3.0, 2.0 / 3.0),
    tolerance_fraction: float = 0.12,
    min_rel_height: float = 0.12,
    min_distance_px: int = 4,
) -> ReconstructionSeriesMetrics:
    """Run reconstruction analysis across a movie."""

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
        analyze_surface_reconstruction(
            frame,
            roi=roi,
            background_roi=background_roi,
            axis=axis,
            reducer=reducer,
            expected_spacing_px=expected_spacing_px,
            fractional_orders=fractional_orders,
            tolerance_fraction=tolerance_fraction,
            min_rel_height=min_rel_height,
            min_distance_px=min_distance_px,
        )
        for frame in arr
    ]

    return ReconstructionSeriesMetrics(
        ts=ts_arr,
        fundamental_spacing_px=np.asarray(
            [np.nan if item.fundamental_spacing_px is None else float(item.fundamental_spacing_px) for item in metrics],
            dtype=float,
        ),
        fractional_peak_count=np.asarray([item.fractional_peak_count for item in metrics], dtype=int),
        strongest_fractional_ratio=np.asarray([item.strongest_fractional_ratio for item in metrics], dtype=float),
        reconstruction_present=np.asarray([item.reconstruction_present for item in metrics], dtype=bool),
    )


__all__ = [
    "FractionalOrderPeak",
    "ReconstructionFrameMetrics",
    "ReconstructionSeriesMetrics",
    "analyze_surface_reconstruction",
    "analyze_surface_reconstruction_series",
    "detect_fractional_order_peaks",
]

