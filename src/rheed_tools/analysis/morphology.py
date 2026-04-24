from __future__ import annotations

"""Shape descriptors for diffraction spots and streaks."""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ShapeMetrics:
    """Compact morphology summary for one background-corrected ROI."""

    sigma_x: float
    sigma_y: float
    fwhm_x: float
    fwhm_y: float
    aspect_ratio: float
    orientation_deg: float
    streakiness: float


_FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))


def weighted_moments(img: np.ndarray) -> tuple[float, float, float, float, float]:
    """Return weighted centroid, spread, and principal-axis angle."""

    arr = np.asarray(img, dtype=float)
    if arr.ndim != 2:
        raise ValueError("img must be a 2D array")

    h, w = arr.shape
    total = float(np.sum(arr))
    if total <= 1e-12:
        return w / 2.0, h / 2.0, 0.0, 0.0, 0.0

    yy, xx = np.indices(arr.shape)
    x_cm = float(np.sum(xx * arr) / total)
    y_cm = float(np.sum(yy * arr) / total)

    dx = xx - x_cm
    dy = yy - y_cm
    cov_xx = float(np.sum(dx * dx * arr) / total)
    cov_yy = float(np.sum(dy * dy * arr) / total)
    cov_xy = float(np.sum(dx * dy * arr) / total)
    cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=float)

    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    major = float(np.sqrt(eigvals[-1]))
    minor = float(np.sqrt(eigvals[0]))
    angle = float(np.degrees(np.arctan2(eigvecs[1, -1], eigvecs[0, -1])))
    return x_cm, y_cm, major, minor, angle


def describe_shape(img: np.ndarray) -> ShapeMetrics:
    """Summarize a diffraction feature as spot-like or streak-like."""

    _, _, sigma_major, sigma_minor, angle = weighted_moments(img)
    sigma_x = sigma_major
    sigma_y = sigma_minor
    aspect_ratio = float(sigma_major / sigma_minor) if sigma_minor > 1e-12 else float("inf")
    streakiness = 0.0 if not np.isfinite(aspect_ratio) else float((aspect_ratio - 1.0) / (aspect_ratio + 1.0))
    return ShapeMetrics(
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        fwhm_x=float(_FWHM_FACTOR * sigma_x),
        fwhm_y=float(_FWHM_FACTOR * sigma_y),
        aspect_ratio=aspect_ratio,
        orientation_deg=angle,
        streakiness=streakiness,
    )

