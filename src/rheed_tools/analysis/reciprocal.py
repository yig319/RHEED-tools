from __future__ import annotations

"""Reciprocal-space helpers built on top of diffraction spacing analysis."""

from dataclasses import dataclass

import numpy as np

from .geometry import measure_peak_spacing, measure_spot_streak_geometry


@dataclass(slots=True)
class ReciprocalCalibration:
    """Pixel-to-reciprocal-space calibration."""

    reciprocal_per_pixel: float
    reciprocal_unit: str = "arb"


@dataclass(slots=True)
class ReciprocalFrameMetrics:
    """Reciprocal-space summary for one diffraction ROI."""

    spacing_px: float | None
    spacing_std_px: float | None
    delta_k: float | None
    lattice_constant: float | None
    strain_percent: float | None
    tilt_deg: float


@dataclass(slots=True)
class ReciprocalSeriesMetrics:
    """Reciprocal-space trends across a frame stack."""

    ts: np.ndarray
    spacing_px: np.ndarray
    spacing_std_px: np.ndarray
    delta_k: np.ndarray
    lattice_constant: np.ndarray
    strain_percent: np.ndarray
    tilt_deg: np.ndarray


def calibrate_reciprocal_space(
    *,
    reciprocal_per_pixel: float,
    reciprocal_unit: str = "arb",
) -> ReciprocalCalibration:
    """Create a calibration object for converting pixel spacing to delta-k.

    Example:
    - if one pixel corresponds to `0.015 1/Angstrom`, pass
      `reciprocal_per_pixel=0.015`
    """

    if reciprocal_per_pixel <= 0:
        raise ValueError("reciprocal_per_pixel must be > 0")
    return ReciprocalCalibration(
        reciprocal_per_pixel=float(reciprocal_per_pixel),
        reciprocal_unit=str(reciprocal_unit),
    )


def pixel_spacing_to_delta_k(
    spacing_px: float | None,
    calibration: ReciprocalCalibration | None,
) -> float | None:
    """Convert a spacing in pixels to reciprocal-space spacing."""

    if spacing_px is None or calibration is None:
        return None
    return float(spacing_px) * calibration.reciprocal_per_pixel


def estimate_in_plane_lattice_constant(
    delta_k: float | None,
) -> float | None:
    """Estimate lattice constant from reciprocal spacing using 2*pi / delta_k.

    When the calibration is not absolute, this still provides a useful
    relative trend even if the resulting lattice constant is in arbitrary
    units.
    """

    if delta_k is None or delta_k <= 0:
        return None
    return float(2.0 * np.pi / delta_k)


def estimate_strain_percent(
    lattice_constant: float | None,
    *,
    reference_lattice_constant: float | None = None,
) -> float | None:
    """Estimate percent strain relative to a reference lattice constant."""

    if lattice_constant is None or reference_lattice_constant in (None, 0):
        return None
    return float((lattice_constant - reference_lattice_constant) / reference_lattice_constant * 100.0)


def analyze_reciprocal_frame(
    frame: np.ndarray,
    *,
    roi: tuple[int, int, int, int] | None = None,
    background_roi: tuple[int, int, int, int] | None = None,
    axis: str = "x",
    reducer: str = "mean",
    min_rel_height: float = 0.2,
    min_distance_px: int = 6,
    calibration: ReciprocalCalibration | None = None,
    reference_lattice_constant: float | None = None,
) -> ReciprocalFrameMetrics:
    """Measure streak spacing and convert it into lattice / strain metrics.

    Tuning guidance:
    - use the same `roi` as your geometry analysis so the same spot family is
      being measured
    - `axis` should follow the direction where periodic streak spacing appears
    - calibration can be omitted at first if you only need relative spacing
      trends during growth
    """

    spacing = measure_peak_spacing(
        frame,
        roi=roi,
        background_roi=background_roi,
        axis=axis,
        reducer=reducer,
        min_rel_height=min_rel_height,
        min_distance_px=min_distance_px,
    )
    geom = measure_spot_streak_geometry(
        frame,
        roi=roi,
        background_roi=background_roi,
        spacing_axis=axis,
        spacing_reducer=reducer,
        spacing_min_rel_height=min_rel_height,
        spacing_min_distance_px=min_distance_px,
    )
    delta_k = pixel_spacing_to_delta_k(spacing.spacing_px, calibration)
    lattice_constant = estimate_in_plane_lattice_constant(delta_k)
    strain_percent = estimate_strain_percent(
        lattice_constant,
        reference_lattice_constant=reference_lattice_constant,
    )
    return ReciprocalFrameMetrics(
        spacing_px=spacing.spacing_px,
        spacing_std_px=spacing.spacing_std_px,
        delta_k=delta_k,
        lattice_constant=lattice_constant,
        strain_percent=strain_percent,
        tilt_deg=float(geom.tilt_deg),
    )


def analyze_reciprocal_series(
    frames: np.ndarray,
    *,
    ts: np.ndarray | None = None,
    roi: tuple[int, int, int, int] | None = None,
    background_roi: tuple[int, int, int, int] | None = None,
    axis: str = "x",
    reducer: str = "mean",
    min_rel_height: float = 0.2,
    min_distance_px: int = 6,
    calibration: ReciprocalCalibration | None = None,
    reference_lattice_constant: float | None = None,
) -> ReciprocalSeriesMetrics:
    """Run reciprocal-space analysis across a frame stack."""

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
        analyze_reciprocal_frame(
            frame,
            roi=roi,
            background_roi=background_roi,
            axis=axis,
            reducer=reducer,
            min_rel_height=min_rel_height,
            min_distance_px=min_distance_px,
            calibration=calibration,
            reference_lattice_constant=reference_lattice_constant,
        )
        for frame in arr
    ]

    def optional_array(name: str) -> np.ndarray:
        values = [getattr(item, name) for item in metrics]
        return np.asarray([np.nan if value is None else float(value) for value in values], dtype=float)

    return ReciprocalSeriesMetrics(
        ts=ts_arr,
        spacing_px=optional_array("spacing_px"),
        spacing_std_px=optional_array("spacing_std_px"),
        delta_k=optional_array("delta_k"),
        lattice_constant=optional_array("lattice_constant"),
        strain_percent=optional_array("strain_percent"),
        tilt_deg=np.asarray([item.tilt_deg for item in metrics], dtype=float),
    )


__all__ = [
    "ReciprocalCalibration",
    "ReciprocalFrameMetrics",
    "ReciprocalSeriesMetrics",
    "analyze_reciprocal_frame",
    "analyze_reciprocal_series",
    "calibrate_reciprocal_space",
    "estimate_in_plane_lattice_constant",
    "estimate_strain_percent",
    "pixel_spacing_to_delta_k",
]

