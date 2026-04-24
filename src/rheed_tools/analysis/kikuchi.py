from __future__ import annotations

"""Approximate Kikuchi-line detection for notebook and real-time screening."""

from dataclasses import dataclass

import numpy as np

from .roi import sanitize_roi


@dataclass(slots=True)
class KikuchiLine:
    """One line-like Kikuchi feature detected from a gradient projection."""

    angle_deg: float
    offset_px: float
    strength: float
    sharpness: float


@dataclass(slots=True)
class KikuchiFrameMetrics:
    """Kikuchi-line summary for one frame."""

    roi: tuple[int, int, int, int]
    lines: list[KikuchiLine]
    dominant_angle_deg: float | None
    mean_line_sharpness: float | None
    zone_axis_x: float | None
    zone_axis_y: float | None


@dataclass(slots=True)
class KikuchiSeriesMetrics:
    """Kikuchi-line trends across a frame series."""

    ts: np.ndarray
    dominant_angle_deg: np.ndarray
    mean_line_sharpness: np.ndarray
    zone_axis_x: np.ndarray
    zone_axis_y: np.ndarray
    line_count: np.ndarray


def _resolve_zone_axis(
    lines: list[KikuchiLine],
    *,
    center_x: float,
    center_y: float,
) -> tuple[float | None, float | None]:
    if len(lines) < 2:
        return None, None
    best_pair = None
    best_delta = None
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            delta = abs(lines[i].angle_deg - lines[j].angle_deg)
            delta = min(delta, abs(delta - 180.0))
            if delta < 10.0:
                continue
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_pair = (lines[i], lines[j])
    if best_pair is None:
        return None, None

    line_a, line_b = best_pair
    theta_a = np.deg2rad(line_a.angle_deg)
    theta_b = np.deg2rad(line_b.angle_deg)
    normal_a = np.array([np.cos(theta_a), np.sin(theta_a)], dtype=float)
    normal_b = np.array([np.cos(theta_b), np.sin(theta_b)], dtype=float)
    lhs = np.stack([normal_a, normal_b], axis=0)
    rhs = np.array([line_a.offset_px, line_b.offset_px], dtype=float)
    if abs(np.linalg.det(lhs)) < 1e-9:
        return None, None
    dx, dy = np.linalg.solve(lhs, rhs)
    return float(center_x + dx), float(center_y + dy)


def detect_kikuchi_lines(
    frame: np.ndarray,
    *,
    roi: tuple[int, int, int, int] | None = None,
    angle_step_deg: float = 2.0,
    top_n: int = 4,
    min_angle_separation_deg: float = 12.0,
) -> KikuchiFrameMetrics:
    """Detect approximate Kikuchi-line orientations from gradient projections.

    This is a lightweight line detector meant for screening and tracking.
    It is not a full crystallographic solver, but it is usually enough to
    monitor line orientation drift and line sharpness.
    """

    try:
        from scipy import ndimage
    except ImportError as exc:
        raise ImportError(
            "scipy is required for detect_kikuchi_lines(). Install notebook/dev dependencies first."
        ) from exc

    arr = np.asarray(frame, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("frame must be a non-empty 2D array")
    if angle_step_deg <= 0:
        raise ValueError("angle_step_deg must be > 0")
    if top_n < 1:
        raise ValueError("top_n must be >= 1")

    y0, y1, x0, x1 = sanitize_roi(arr.shape, roi, fraction=0.8)
    patch = arr[y0:y1, x0:x1]
    grad_y = ndimage.sobel(patch, axis=0, mode="reflect")
    grad_x = ndimage.sobel(patch, axis=1, mode="reflect")
    edge = np.hypot(grad_x, grad_y)
    edge = edge - float(np.min(edge))

    candidates: list[KikuchiLine] = []
    angles = np.arange(-88.0, 90.0, float(angle_step_deg), dtype=float)
    for angle in angles:
        rotated = ndimage.rotate(edge, angle, reshape=False, order=1, mode="nearest")
        profile = np.mean(rotated, axis=0)
        profile = profile - float(np.median(profile))
        strength = float(np.max(profile))
        if strength <= 0:
            continue
        idx = int(np.argmax(profile))
        offset_px = float(idx - (profile.size - 1) / 2.0)
        sharpness = strength / (float(np.std(profile)) + 1e-9)
        candidates.append(
            KikuchiLine(
                angle_deg=float(angle),
                offset_px=offset_px,
                strength=strength,
                sharpness=float(sharpness),
            )
        )

    candidates.sort(key=lambda item: item.strength, reverse=True)
    lines: list[KikuchiLine] = []
    for candidate in candidates:
        if all(
            min(abs(candidate.angle_deg - existing.angle_deg), abs(abs(candidate.angle_deg - existing.angle_deg) - 180.0))
            >= min_angle_separation_deg
            for existing in lines
        ):
            lines.append(candidate)
        if len(lines) >= top_n:
            break

    center_x = x0 + (x1 - x0 - 1) / 2.0
    center_y = y0 + (y1 - y0 - 1) / 2.0
    zone_axis_x, zone_axis_y = _resolve_zone_axis(lines, center_x=center_x, center_y=center_y)
    dominant_angle = None if not lines else float(lines[0].angle_deg)
    mean_sharpness = None if not lines else float(np.mean([item.sharpness for item in lines]))

    return KikuchiFrameMetrics(
        roi=(y0, y1, x0, x1),
        lines=lines,
        dominant_angle_deg=dominant_angle,
        mean_line_sharpness=mean_sharpness,
        zone_axis_x=zone_axis_x,
        zone_axis_y=zone_axis_y,
    )


def analyze_kikuchi_series(
    frames: np.ndarray,
    *,
    ts: np.ndarray | None = None,
    roi: tuple[int, int, int, int] | None = None,
    angle_step_deg: float = 2.0,
    top_n: int = 4,
    min_angle_separation_deg: float = 12.0,
) -> KikuchiSeriesMetrics:
    """Run Kikuchi-line analysis across a movie."""

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
        detect_kikuchi_lines(
            frame,
            roi=roi,
            angle_step_deg=angle_step_deg,
            top_n=top_n,
            min_angle_separation_deg=min_angle_separation_deg,
        )
        for frame in arr
    ]

    def optional_array(name: str) -> np.ndarray:
        values = [getattr(item, name) for item in metrics]
        return np.asarray([np.nan if value is None else float(value) for value in values], dtype=float)

    return KikuchiSeriesMetrics(
        ts=ts_arr,
        dominant_angle_deg=optional_array("dominant_angle_deg"),
        mean_line_sharpness=optional_array("mean_line_sharpness"),
        zone_axis_x=optional_array("zone_axis_x"),
        zone_axis_y=optional_array("zone_axis_y"),
        line_count=np.asarray([len(item.lines) for item in metrics], dtype=int),
    )


__all__ = [
    "KikuchiFrameMetrics",
    "KikuchiLine",
    "KikuchiSeriesMetrics",
    "analyze_kikuchi_series",
    "detect_kikuchi_lines",
]

