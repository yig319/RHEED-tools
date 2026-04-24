from __future__ import annotations

"""Bright-spot detection and local Gaussian fitting for diffraction analysis."""

from dataclasses import dataclass

import numpy as np

from .roi import recenter_roi, sanitize_roi


@dataclass(slots=True)
class SpotCandidate:
    """One bright candidate detected in a frame."""

    peak_x: float
    peak_y: float
    peak_value: float
    integrated_intensity: float
    roi: tuple[int, int, int, int]


@dataclass(slots=True)
class GaussianFitResult:
    """2D Gaussian fit result for one local diffraction patch."""

    amplitude: float
    background: float
    center_x: float
    center_y: float
    sigma_x: float
    sigma_y: float
    rotation_deg: float
    rmse: float
    roi: tuple[int, int, int, int]

    @property
    def fwhm_x(self) -> float:
        return float(2.0 * np.sqrt(2.0 * np.log(2.0)) * self.sigma_x)

    @property
    def fwhm_y(self) -> float:
        return float(2.0 * np.sqrt(2.0 * np.log(2.0)) * self.sigma_y)


@dataclass(slots=True)
class SpotCropResult:
    """Detected spot, Gaussian fit, and cropped patch driven by the fit."""

    candidate: SpotCandidate
    initial_fit: GaussianFitResult
    crop_patch: np.ndarray
    crop_roi: tuple[int, int, int, int]
    refined_fit: GaussianFitResult | None = None


@dataclass(slots=True)
class SpotRegionTrack:
    """One tracked spot across a frame series plus its union ROI."""

    track_id: int
    frame_indices: np.ndarray
    ts: np.ndarray
    center_x: np.ndarray
    center_y: np.ndarray
    fwhm_x: np.ndarray
    fwhm_y: np.ndarray
    rotation_deg: np.ndarray
    rmse: np.ndarray
    crop_rois: list[tuple[int, int, int, int]]
    union_roi: tuple[int, int, int, int]


@dataclass(slots=True)
class SpotRegionSeries:
    """Statistics and Gaussian metrics for one fixed spot ROI over time."""

    roi: tuple[int, int, int, int]
    ts: np.ndarray
    raw_sum: np.ndarray
    raw_mean: np.ndarray
    raw_min: np.ndarray
    raw_max: np.ndarray
    raw_std: np.ndarray
    center_x: np.ndarray
    center_y: np.ndarray
    fwhm_x: np.ndarray
    fwhm_y: np.ndarray
    rotation_deg: np.ndarray
    rmse: np.ndarray


@dataclass(slots=True)
class SpotCandidateMetrics:
    """Local morphology and fit-quality metrics for one detected candidate."""

    candidate: SpotCandidate
    fit: GaussianFitResult
    aspect_ratio: float
    elongation: float
    normalized_rmse: float
    compact_weight: float
    streak_weight: float


@dataclass(slots=True)
class CandidatePatternSummary:
    """Scene-level summary used to separate streak-like and discrete-spot frames."""

    candidate_count: int
    effective_compact_spot_count: float
    effective_streak_fragment_count: float
    mean_aspect_ratio: float | None
    median_aspect_ratio: float | None
    mean_normalized_rmse: float | None
    orientation_consistency: float | None
    center_line_rmse_px: float | None
    center_arc_rmse_px: float | None
    center_anisotropy: float | None
    neighbor_spacing_cv: float | None
    streak_like_score: float
    discrete_spot_score: float
    pattern_label: str


def gaussian_2d(
    x: np.ndarray,
    y: np.ndarray,
    amplitude: float,
    center_x: float,
    center_y: float,
    sigma_x: float,
    sigma_y: float,
    rotation_deg: float = 0.0,
    background: float = 0.0,
) -> np.ndarray:
    """Evaluate a rotated 2D Gaussian on a mesh grid.

    This is the low-level model used by the fitting helpers in this module.
    Most users will call `fit_gaussian_2d()` or `locate_fit_and_crop_spot()`
    instead of calling `gaussian_2d()` directly.

    Input tuning:
    - `amplitude`: peak height above the local background.
    - `center_x`, `center_y`: subpixel center of the diffraction spot.
    - `sigma_x`, `sigma_y`: Gaussian widths in pixels. Increase them when you
      expect a broader spot or a streak-like feature.
    - `rotation_deg`: useful when the feature is elongated and not aligned with
      the image axes.
    - `background`: local offset level under the spot.
    """

    sigma_x = max(float(sigma_x), 1e-9)
    sigma_y = max(float(sigma_y), 1e-9)
    theta = np.deg2rad(float(rotation_deg))
    x_shift = x - float(center_x)
    y_shift = y - float(center_y)
    xp = x_shift * np.cos(theta) + y_shift * np.sin(theta)
    yp = -x_shift * np.sin(theta) + y_shift * np.cos(theta)
    expo = -0.5 * ((xp / sigma_x) ** 2 + (yp / sigma_y) ** 2)
    return float(background) + float(amplitude) * np.exp(expo)


def gaussian_function(
    amplitude: float,
    center_x: float,
    center_y: float,
    sigma_x: float,
    sigma_y: float,
    rotation_deg: float = 0.0,
    background: float = 0.0,
):
    """Return a callable Gaussian model similar to the user's earlier helper.

    This wrapper is mainly useful when you want a function-style interface for
    plotting or recreating a fitted spot from stored Gaussian parameters.
    """

    def model(y: np.ndarray, x: np.ndarray) -> np.ndarray:
        return gaussian_2d(
            x,
            y,
            amplitude=amplitude,
            center_x=center_x,
            center_y=center_y,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            rotation_deg=rotation_deg,
            background=background,
        )

    return model


def estimate_gaussian_moments(data: np.ndarray) -> tuple[float, float, float, float, float, float, float]:
    """Estimate Gaussian parameters from image moments.

    This provides the initial guess for the nonlinear least-squares fit.
    If the fit is unstable, the problem is often not the optimizer itself but
    that the input patch is too large, too noisy, or contains multiple spots.

    What to adjust first:
    - tighten the patch around one isolated spot
    - improve background subtraction before fitting
    - reduce contamination from neighboring peaks or streaks
    """

    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("data must be a non-empty 2D array")

    background = float(np.percentile(arr, 10))
    signal = np.clip(arr - background, 0.0, None)
    total = float(signal.sum())
    yy, xx = np.indices(arr.shape, dtype=float)

    if total <= 1e-12:
        center_x = (arr.shape[1] - 1) / 2.0
        center_y = (arr.shape[0] - 1) / 2.0
        return 0.0, background, center_x, center_y, 1.0, 1.0, 0.0

    center_x = float((xx * signal).sum() / total)
    center_y = float((yy * signal).sum() / total)
    dx = xx - center_x
    dy = yy - center_y
    cov_xx = float((signal * dx * dx).sum() / total)
    cov_yy = float((signal * dy * dy).sum() / total)
    cov_xy = float((signal * dx * dy).sum() / total)
    cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=float)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1e-9, None)
    sigma_x = float(np.sqrt(eigvals[1]))
    sigma_y = float(np.sqrt(eigvals[0]))
    angle = float(np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1])))
    amplitude = float(np.max(signal))
    return amplitude, background, center_x, center_y, sigma_x, sigma_y, angle


def fit_gaussian_2d(
    data: np.ndarray,
    *,
    allow_rotation: bool = True,
) -> GaussianFitResult:
    """Fit a local diffraction patch with a rotated 2D Gaussian.

    Use this when you already have a local patch that mostly contains one
    diffraction spot. The fit returns center, width, rotation, background, and
    an `rmse` value to judge fit quality.

    What to adjust first when the fit looks wrong:
    - shrink the input patch so it contains one dominant spot
    - improve the upstream candidate detection or ROI selection
    - set `allow_rotation=False` if the spot is compact and rotation is causing
      unstable fits

    Interpretation:
    - lower `rmse` usually means the Gaussian describes the patch well
    - large `sigma_x` / `sigma_y` often means the feature is broad or streaking
    - very different `sigma_x` and `sigma_y` indicates elongation
    """

    try:
        from scipy import optimize
    except ImportError as exc:
        raise ImportError(
            "scipy is required for fit_gaussian_2d(). Install notebook/dev dependencies first."
        ) from exc

    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("data must be a non-empty 2D array")

    amp0, bg0, x0, y0, sx0, sy0, ang0 = estimate_gaussian_moments(arr)
    yy, xx = np.indices(arr.shape, dtype=float)

    if allow_rotation:
        p0 = np.array([amp0, bg0, x0, y0, sx0, sy0, ang0], dtype=float)

        def residuals(p: np.ndarray) -> np.ndarray:
            model = gaussian_2d(xx, yy, p[0], p[2], p[3], abs(p[4]), abs(p[5]), p[6], p[1])
            return np.ravel(model - arr)

        p_opt, _ = optimize.leastsq(residuals, p0)
        amplitude, background, center_x, center_y, sigma_x, sigma_y, angle = p_opt
    else:
        p0 = np.array([amp0, bg0, x0, y0, sx0, sy0], dtype=float)

        def residuals(p: np.ndarray) -> np.ndarray:
            model = gaussian_2d(xx, yy, p[0], p[2], p[3], abs(p[4]), abs(p[5]), 0.0, p[1])
            return np.ravel(model - arr)

        p_opt, _ = optimize.leastsq(residuals, p0)
        amplitude, background, center_x, center_y, sigma_x, sigma_y = p_opt
        angle = 0.0

    fit = gaussian_2d(xx, yy, amplitude, center_x, center_y, abs(sigma_x), abs(sigma_y), angle, background)
    rmse = float(np.sqrt(np.mean((fit - arr) ** 2)))
    return GaussianFitResult(
        amplitude=float(amplitude),
        background=float(background),
        center_x=float(center_x),
        center_y=float(center_y),
        sigma_x=float(abs(sigma_x)),
        sigma_y=float(abs(sigma_y)),
        rotation_deg=float(angle),
        rmse=rmse,
        roi=(0, arr.shape[0], 0, arr.shape[1]),
    )


def reconstruct_gaussian_patch(
    shape: tuple[int, int],
    fit: GaussianFitResult,
) -> np.ndarray:
    """Recreate a fitted Gaussian patch on a local ROI grid.

    This is mainly for inspection: compare the reconstructed Gaussian with the
    raw patch and inspect the residual to see whether the feature is reasonably
    Gaussian-like.
    """

    yy, xx = np.indices(shape, dtype=float)
    return gaussian_2d(
        xx,
        yy,
        amplitude=fit.amplitude,
        center_x=fit.center_x,
        center_y=fit.center_y,
        sigma_x=fit.sigma_x,
        sigma_y=fit.sigma_y,
        rotation_deg=fit.rotation_deg,
        background=fit.background,
    )


def detect_bright_spots(
    frame: np.ndarray,
    *,
    search_roi: tuple[int, int, int, int] | None = None,
    threshold_rel: float = 0.35,
    min_distance_px: int = 9,
    max_candidates: int = 10,
    patch_size: tuple[int, int] = (31, 31),
    expected_center: tuple[float, float] | list[tuple[float, float]] | None = None,
    max_offset_px: float | None = None,
    sort_by: str = "intensity",
) -> list[SpotCandidate]:
    """Detect bright local maxima in a full diffraction image or search ROI.

    This is the first stage in a multi-spot workflow: find candidate bright
    peaks, then choose which diffraction feature is physically meaningful.

    What to adjust first:
    - `search_roi`: the most important control. Use it to limit the search to
      the physically relevant part of the detector.
    - `threshold_rel`: raise it when too many weak/noisy peaks are detected;
      lower it when a real dim spot is being missed.
    - `min_distance_px`: increase it if multiple detections are landing on the
      same physical spot; decrease it if nearby spots are being merged.
    - `max_candidates`: raise it only if you genuinely want to inspect more
      possible spots.
    - `patch_size`: this is only the coarse extraction size used before
      Gaussian-based resizing. It does not need to be the final patch size.
    - `expected_center`: use this when you already know roughly where the
      desired spot should be. This may be one `(x, y)` pair or a list of such
      pairs. It does not force a detection there; it only helps rank or filter
      candidates around those locations.
    - `max_offset_px`: optional radius around `expected_center`. Candidates
      outside this radius are ignored. If no candidate falls inside the radius,
      the function returns an empty list rather than forcing a fit.
    - `sort_by`: `"intensity"` keeps the brightest-first behavior.
      `"distance_then_intensity"` is usually better once you know the spot's
      approximate location from an earlier frame.
    """

    img = np.asarray(frame, dtype=float)
    if img.ndim != 2 or img.size == 0:
        raise ValueError("frame must be a non-empty 2D array")
    if sort_by not in {"intensity", "distance_then_intensity"}:
        raise ValueError("sort_by must be 'intensity' or 'distance_then_intensity'")
    if max_offset_px is not None and max_offset_px < 0:
        raise ValueError("max_offset_px must be >= 0")

    expected_points: list[tuple[float, float]] | None = None
    if expected_center is not None:
        if (
            isinstance(expected_center, tuple)
            and len(expected_center) == 2
            and np.isscalar(expected_center[0])
            and np.isscalar(expected_center[1])
        ):
            expected_points = [(float(expected_center[0]), float(expected_center[1]))]
        else:
            expected_points = [(float(item[0]), float(item[1])) for item in expected_center]
        if len(expected_points) == 0:
            expected_points = None

    y0, y1, x0, x1 = sanitize_roi(img.shape, search_roi, fraction=0.8)
    view = img[y0:y1, x0:x1]
    baseline = float(np.percentile(view, 20))
    corrected = np.clip(view - baseline, 0.0, None)
    if corrected.size == 0 or float(np.max(corrected)) <= 0.0:
        return []

    peak_threshold = float(np.max(corrected) * threshold_rel)
    padded = np.pad(corrected, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    is_peak = corrected >= peak_threshold
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            neighbor = padded[1 + dy : 1 + dy + corrected.shape[0], 1 + dx : 1 + dx + corrected.shape[1]]
            is_peak &= center >= neighbor

    coords = np.argwhere(is_peak)
    if coords.size == 0:
        return []

    values = corrected[coords[:, 0], coords[:, 1]]
    order = np.argsort(values)[::-1]
    candidate_pool: list[SpotCandidate] = []
    half_h = patch_size[0] // 2
    half_w = patch_size[1] // 2

    for idx in order:
        cy_local, cx_local = coords[idx]
        peak_y = int(cy_local + y0)
        peak_x = int(cx_local + x0)
        too_close = False
        for item in candidate_pool:
            if np.hypot(item.peak_x - peak_x, item.peak_y - peak_y) < float(min_distance_px):
                too_close = True
                break
        if too_close:
            continue

        roi = recenter_roi(img.shape, center_y=peak_y, center_x=peak_x, box_height=2 * half_h + 1, box_width=2 * half_w + 1)
        py0, py1, px0, px1 = roi
        patch = img[py0:py1, px0:px1]
        integrated = float(np.sum(np.clip(patch - baseline, 0.0, None)))
        candidate_pool.append(
            SpotCandidate(
                peak_x=float(peak_x),
                peak_y=float(peak_y),
                peak_value=float(img[peak_y, peak_x]),
                integrated_intensity=integrated,
                roi=roi,
            )
        )

    def min_expected_distance(item: SpotCandidate) -> float:
        if expected_points is None:
            return 0.0
        return float(
            min(np.hypot(item.peak_x - expected_x, item.peak_y - expected_y) for expected_x, expected_y in expected_points)
        )

    if expected_points is not None:
        if max_offset_px is not None:
            candidate_pool = [
                item
                for item in candidate_pool
                if min_expected_distance(item) <= float(max_offset_px)
            ]
        if sort_by == "distance_then_intensity":
            candidate_pool.sort(
                key=lambda item: (
                    min_expected_distance(item),
                    -float(item.integrated_intensity),
                    -float(item.peak_value),
                )
            )
        else:
            candidate_pool.sort(
                key=lambda item: (
                    -float(item.integrated_intensity),
                    -float(item.peak_value),
                    min_expected_distance(item),
                )
            )
    else:
        candidate_pool.sort(key=lambda item: (-float(item.integrated_intensity), -float(item.peak_value)))

    return candidate_pool[:max_candidates]


def analyze_spot_candidates(
    frame: np.ndarray,
    candidates: list[SpotCandidate],
    *,
    patch_size: tuple[int, int] | None = None,
    allow_rotation: bool = True,
) -> list[SpotCandidateMetrics]:
    """Fit each candidate locally and derive compactness/streakiness metrics.

    Use this after `detect_bright_spots()` when plain candidate count is not
    informative enough. The resulting metrics help separate:
    - compact isolated spots that behave like true diffraction peaks
    - elongated or diffuse fragments that are more likely part of a streak

    What to adjust first:
    - `patch_size`: set this when you want a separate local fit window for
      morphology classification. A square patch often works better than
      reusing a wide detection ROI because it reduces bias toward elongated
      fits. If omitted, the candidate's own ROI is reused.
    - `allow_rotation`: keep this enabled for streak-like or tilted features.
      Disable it if compact spots are fit more stably without a free angle.
    """

    arr = np.asarray(frame, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("frame must be a non-empty 2D array")

    metrics: list[SpotCandidateMetrics] = []
    for candidate in candidates:
        if patch_size is None:
            y0, y1, x0, x1 = candidate.roi
            patch = arr[y0:y1, x0:x1]
            roi = candidate.roi
        else:
            patch, roi = extract_spot_patch(
                arr,
                candidate.peak_x,
                candidate.peak_y,
                patch_size=patch_size,
            )

        fit = fit_gaussian_2d(patch, allow_rotation=allow_rotation)
        fit.roi = roi
        fit.center_x = float(roi[2] + fit.center_x)
        fit.center_y = float(roi[0] + fit.center_y)

        major = max(float(fit.fwhm_x), float(fit.fwhm_y), 1e-9)
        minor = max(min(float(fit.fwhm_x), float(fit.fwhm_y)), 1e-9)
        aspect_ratio = float(major / minor)
        elongation = float((major - minor) / (major + minor))
        norm_denom = max(abs(float(fit.amplitude)), float(candidate.peak_value), 1e-9)
        normalized_rmse = float(fit.rmse / norm_denom)
        compact_weight = float(np.exp(-max(aspect_ratio - 1.0, 0.0)) * np.exp(-3.0 * normalized_rmse))
        streak_weight = float(max(aspect_ratio - 1.0, 0.0) * np.exp(-2.0 * normalized_rmse))
        metrics.append(
            SpotCandidateMetrics(
                candidate=candidate,
                fit=fit,
                aspect_ratio=aspect_ratio,
                elongation=elongation,
                normalized_rmse=normalized_rmse,
                compact_weight=compact_weight,
                streak_weight=streak_weight,
            )
        )

    return metrics


def summarize_candidate_geometry(
    candidate_metrics: list[SpotCandidateMetrics],
    *,
    streak_aspect_ratio_center: float = 1.6,
    discrete_aspect_ratio_ceiling: float = 1.45,
    label_margin: float = 0.22,
) -> CandidatePatternSummary:
    """Summarize a set of candidates with geometry-based growth heuristics.

    This is meant for frames where raw spot count is ambiguous. It combines:
    - local candidate shape from Gaussian fits
    - global arrangement of candidate centers

    Practical interpretation:
    - high `effective_compact_spot_count` favors discrete spot-like growth
    - high `effective_streak_fragment_count` plus good alignment favors a
      streak family broken into several detected maxima
    - `pattern_label` is a heuristic guide, not a hard physical truth

    Tuning guidance:
    - `streak_aspect_ratio_center`: raise this if compact spots are being
      called streaky too often. Lower it only if clearly elongated features
      are not receiving enough streak credit.
    - `discrete_aspect_ratio_ceiling`: raise this slightly if compact but
      imperfect spots are being penalized too hard.
    - `label_margin`: increase this to make the `mixed` class more common and
      reduce overconfident streak/spot labels.
    """

    if not candidate_metrics:
        return CandidatePatternSummary(
            candidate_count=0,
            effective_compact_spot_count=0.0,
            effective_streak_fragment_count=0.0,
            mean_aspect_ratio=None,
            median_aspect_ratio=None,
            mean_normalized_rmse=None,
            orientation_consistency=None,
            center_line_rmse_px=None,
            center_arc_rmse_px=None,
            center_anisotropy=None,
            neighbor_spacing_cv=None,
            streak_like_score=0.0,
            discrete_spot_score=0.0,
            pattern_label="no_candidates",
        )

    aspect = np.asarray([item.aspect_ratio for item in candidate_metrics], dtype=float)
    norm_rmse = np.asarray([item.normalized_rmse for item in candidate_metrics], dtype=float)
    compact = np.asarray([item.compact_weight for item in candidate_metrics], dtype=float)
    streak = np.asarray([item.streak_weight for item in candidate_metrics], dtype=float)
    centers = np.asarray([(item.fit.center_x, item.fit.center_y) for item in candidate_metrics], dtype=float)

    effective_compact_spot_count = float(np.sum(compact))
    effective_streak_fragment_count = float(np.sum(streak))
    mean_aspect_ratio = float(np.mean(aspect))
    median_aspect_ratio = float(np.median(aspect))
    mean_normalized_rmse = float(np.mean(norm_rmse))
    orientation_consistency = _orientation_consistency(candidate_metrics)
    center_line_rmse_px, center_anisotropy = _line_alignment_metrics(centers)
    center_arc_rmse_px = _circle_fit_rmse(centers)
    neighbor_spacing_cv = _nearest_neighbor_spacing_cv(centers)

    extent_x = float(np.max(centers[:, 0]) - np.min(centers[:, 0])) if centers.shape[0] >= 2 else 0.0
    extent_y = float(np.max(centers[:, 1]) - np.min(centers[:, 1])) if centers.shape[0] >= 2 else 0.0
    diag = float(np.hypot(extent_x, extent_y))
    line_alignment = _alignment_score(center_line_rmse_px, diag)
    arc_alignment = _alignment_score(center_arc_rmse_px, diag)
    alignment_score = max(line_alignment, arc_alignment)
    elongation_score = float(
        np.clip((mean_aspect_ratio - streak_aspect_ratio_center) / max(streak_aspect_ratio_center, 1e-9), 0.0, 1.0)
    )
    anisotropy_score = 0.0 if center_anisotropy is None else float(np.clip((center_anisotropy - 1.2) / 2.0, 0.0, 1.0))
    streak_density_score = float(
        np.clip(effective_streak_fragment_count / max(len(candidate_metrics), 1), 0.0, 1.0)
    )
    compact_score = float(np.clip(effective_compact_spot_count / max(len(candidate_metrics), 1), 0.0, 1.0))
    compact_shape_score = float(
        np.clip((discrete_aspect_ratio_ceiling - mean_aspect_ratio) / max(discrete_aspect_ratio_ceiling - 1.0, 1e-9), 0.0, 1.0)
    )
    sparse_alignment_score = 1.0 - alignment_score

    streak_like_score = float(
        np.mean([elongation_score, orientation_consistency, alignment_score, anisotropy_score, streak_density_score])
    )
    discrete_spot_score = float(
        np.mean([compact_score, compact_shape_score, sparse_alignment_score, 1.0 - 0.5 * orientation_consistency])
    )

    if streak_like_score >= discrete_spot_score + label_margin:
        pattern_label = "streak_family"
    elif discrete_spot_score >= streak_like_score + label_margin:
        pattern_label = "discrete_spots"
    else:
        pattern_label = "mixed"

    return CandidatePatternSummary(
        candidate_count=len(candidate_metrics),
        effective_compact_spot_count=effective_compact_spot_count,
        effective_streak_fragment_count=effective_streak_fragment_count,
        mean_aspect_ratio=mean_aspect_ratio,
        median_aspect_ratio=median_aspect_ratio,
        mean_normalized_rmse=mean_normalized_rmse,
        orientation_consistency=orientation_consistency,
        center_line_rmse_px=center_line_rmse_px,
        center_arc_rmse_px=center_arc_rmse_px,
        center_anisotropy=center_anisotropy,
        neighbor_spacing_cv=neighbor_spacing_cv,
        streak_like_score=streak_like_score,
        discrete_spot_score=discrete_spot_score,
        pattern_label=pattern_label,
    )


def classify_growth_pattern_from_candidates(
    frame: np.ndarray,
    candidates: list[SpotCandidate],
    *,
    patch_size: tuple[int, int] | None = None,
    allow_rotation: bool = True,
    streak_aspect_ratio_center: float = 1.6,
    discrete_aspect_ratio_ceiling: float = 1.45,
    label_margin: float = 0.22,
) -> CandidatePatternSummary:
    """Convenience wrapper: fit candidates locally and summarize the frame."""

    metrics = analyze_spot_candidates(
        frame,
        candidates,
        patch_size=patch_size,
        allow_rotation=allow_rotation,
    )
    return summarize_candidate_geometry(
        metrics,
        streak_aspect_ratio_center=streak_aspect_ratio_center,
        discrete_aspect_ratio_ceiling=discrete_aspect_ratio_ceiling,
        label_margin=label_margin,
    )


def extract_spot_patch(
    frame: np.ndarray,
    center_x: float,
    center_y: float,
    *,
    patch_size: tuple[int, int] = (31, 31),
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop an initial local patch around one spot center.

    This is the coarse patch used before Gaussian-driven resizing. For final
    spot extraction, prefer `crop_patch_from_gaussian_fit()` or
    `locate_fit_and_crop_spot()`.

    What to adjust first:
    - `patch_size`: make it just large enough to include the whole spot and a
      little background. If it is too large, neighboring features will bias the
      Gaussian fit. If it is too small, the Gaussian tails will be clipped.
    """

    arr = np.asarray(frame, dtype=float)
    if arr.ndim != 2:
        raise ValueError("frame must be a 2D array")
    roi = recenter_roi(arr.shape, center_y=center_y, center_x=center_x, box_height=patch_size[0], box_width=patch_size[1])
    y0, y1, x0, x1 = roi
    return arr[y0:y1, x0:x1], roi


def crop_patch_from_gaussian_fit(
    frame: np.ndarray,
    fit: GaussianFitResult,
    *,
    amplitude_fraction: float = 0.02,
    edge_padding_px: int = 3,
    keep_reference_aspect: bool = False,
    reference_patch_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop a patch based on where the fitted Gaussian still carries signal.

    The Gaussian support is truncated where the model falls below
    `amplitude_fraction` of its peak amplitude, then `edge_padding_px` pixels
    are added on all sides.

    What to adjust first:
    - `amplitude_fraction`: the main crop aggressiveness control.
      Smaller values keep more of the Gaussian tails.
      Larger values crop more tightly to the bright core.
      Typical values are around `0.01-0.10`.
    - `edge_padding_px`: use this when you want to keep a safety border after
      trimming. Typical values are `2-5` pixels.
    - `keep_reference_aspect`: keep the final crop aligned with the aspect
      ratio of the initial coarse patch instead of letting it drift toward a
      near-square Gaussian support.
    - `reference_patch_size`: the `(height, width)` used to define that aspect
      ratio, typically the same as the initial detection/fitting patch size.

    Practical rule of thumb:
    - use a smaller `amplitude_fraction` for weak, diffuse, or broad spots
    - use a larger `amplitude_fraction` for compact bright spots when you want
      a tighter patch for downstream fitting
    """

    arr = np.asarray(frame, dtype=float)
    if arr.ndim != 2:
        raise ValueError("frame must be a 2D array")
    if not (0.0 < amplitude_fraction < 1.0):
        raise ValueError("amplitude_fraction must be between 0 and 1")
    if edge_padding_px < 0:
        raise ValueError("edge_padding_px must be >= 0")
    if keep_reference_aspect:
        if reference_patch_size is None:
            raise ValueError("reference_patch_size is required when keep_reference_aspect=True")
        ref_h = int(reference_patch_size[0])
        ref_w = int(reference_patch_size[1])
        if ref_h <= 0 or ref_w <= 0:
            raise ValueError("reference_patch_size must contain positive integers")

    k = float(np.sqrt(-2.0 * np.log(amplitude_fraction)))
    half_h = int(np.ceil(k * fit.sigma_y)) + int(edge_padding_px)
    half_w = int(np.ceil(k * fit.sigma_x)) + int(edge_padding_px)
    box_height = 2 * half_h + 1
    box_width = 2 * half_w + 1

    if keep_reference_aspect:
        target_ratio = ref_w / ref_h
        current_ratio = box_width / box_height
        if current_ratio < target_ratio:
            box_width = int(np.ceil(box_height * target_ratio))
        else:
            box_height = int(np.ceil(box_width / target_ratio))

    roi = recenter_roi(
        arr.shape,
        center_y=fit.center_y,
        center_x=fit.center_x,
        box_height=box_height,
        box_width=box_width,
    )
    y0, y1, x0, x1 = roi
    return arr[y0:y1, x0:x1], roi


def locate_and_fit_spot(
    frame: np.ndarray,
    *,
    search_roi: tuple[int, int, int, int] | None = None,
    threshold_rel: float = 0.35,
    min_distance_px: int = 9,
    patch_size: tuple[int, int] = (31, 31),
    allow_rotation: bool = True,
) -> tuple[SpotCandidate, GaussianFitResult]:
    """Detect the brightest spot candidate and fit it with a local 2D Gaussian.

    This is the shortest path when you want one spot from one frame and do not
    need to inspect all candidates first.

    What to adjust first:
    - `search_roi` to restrict the search region
    - `threshold_rel` to control candidate sensitivity
    - `patch_size` to set the coarse fit window
    - `allow_rotation` depending on whether the feature is spot-like or
      elongated
    """

    candidates = detect_bright_spots(
        frame,
        search_roi=search_roi,
        threshold_rel=threshold_rel,
        min_distance_px=min_distance_px,
        max_candidates=1,
        patch_size=patch_size,
    )
    if not candidates:
        raise ValueError("no bright spot candidates found in the requested search region")

    candidate = candidates[0]
    patch, roi = extract_spot_patch(frame, candidate.peak_x, candidate.peak_y, patch_size=patch_size)
    fit = fit_gaussian_2d(patch, allow_rotation=allow_rotation)
    fit.roi = roi
    fit.center_x = float(roi[2] + fit.center_x)
    fit.center_y = float(roi[0] + fit.center_y)
    return candidate, fit


def locate_fit_and_crop_spot(
    frame: np.ndarray,
    *,
    search_roi: tuple[int, int, int, int] | None = None,
    threshold_rel: float = 0.35,
    min_distance_px: int = 9,
    patch_size: tuple[int, int] = (31, 31),
    allow_rotation: bool = True,
    amplitude_fraction: float = 0.02,
    edge_padding_px: int = 3,
    keep_reference_aspect: bool = False,
    refit_cropped: bool = True,
) -> SpotCropResult:
    """Detect, fit, and then crop a spot using the fitted Gaussian support.

    This is the main convenience workflow for one spot in one frame:
    detect -> fit -> trim from Gaussian support -> optional refit.

    What to adjust first:
    - `search_roi`: almost always the first thing to tune
    - `patch_size`: the coarse pre-fit window
    - `amplitude_fraction`: how aggressively the Gaussian-support crop trims
      low-signal edges
    - `edge_padding_px`: how much border to keep after trimming
    - `keep_reference_aspect`: preserve the coarse patch aspect ratio in the
      final Gaussian-driven crop
    - `refit_cropped=True`: keep this on when you want final metrics from the
      tighter cropped patch
    """

    candidate, fit = locate_and_fit_spot(
        frame,
        search_roi=search_roi,
        threshold_rel=threshold_rel,
        min_distance_px=min_distance_px,
        patch_size=patch_size,
        allow_rotation=allow_rotation,
    )
    crop_patch, crop_roi = crop_patch_from_gaussian_fit(
        frame,
        fit,
        amplitude_fraction=amplitude_fraction,
        edge_padding_px=edge_padding_px,
        keep_reference_aspect=keep_reference_aspect,
        reference_patch_size=patch_size,
    )

    refined_fit = None
    if refit_cropped:
        refined_fit = fit_gaussian_2d(crop_patch, allow_rotation=allow_rotation)
        refined_fit.roi = crop_roi
        refined_fit.center_x = float(crop_roi[2] + refined_fit.center_x)
        refined_fit.center_y = float(crop_roi[0] + refined_fit.center_y)

    return SpotCropResult(
        candidate=candidate,
        initial_fit=fit,
        crop_patch=crop_patch,
        crop_roi=crop_roi,
        refined_fit=refined_fit,
    )


def extract_multiple_gaussian_spot_patches(
    frame: np.ndarray,
    *,
    search_roi: tuple[int, int, int, int] | None = None,
    threshold_rel: float = 0.35,
    min_distance_px: int = 9,
    max_candidates: int = 10,
    initial_patch_size: tuple[int, int] = (31, 31),
    allow_rotation: bool = True,
    amplitude_fraction: float = 0.02,
    edge_padding_px: int = 3,
    keep_reference_aspect: bool = False,
    refit_cropped: bool = True,
) -> list[SpotCropResult]:
    """Detect several bright spots and extract Gaussian-sized patches for each.

    The workflow is:
    1. detect bright candidates in the main image or search ROI
    2. extract one coarse patch per candidate using `initial_patch_size`
    3. fit a 2D Gaussian in that coarse patch
    4. resize the patch from the fitted Gaussian support
    5. optionally refit inside the tighter crop

    This is useful when one main image contains multiple diffraction spots and
    you want each extracted patch size to be determined by its own fitted spot
    width instead of one global fixed patch size.

    What to adjust first:
    - `search_roi`: keep the multi-spot search limited to the relevant region
    - `threshold_rel`: controls how many peaks become candidates
    - `min_distance_px`: separates nearby spots
    - `initial_patch_size`: only for the first fit, not the final crop
    - `amplitude_fraction` and `edge_padding_px`: determine each final
      Gaussian-sized patch
    - `keep_reference_aspect`: preserve the initial patch aspect ratio in each
      final crop
    """

    candidates = detect_bright_spots(
        frame,
        search_roi=search_roi,
        threshold_rel=threshold_rel,
        min_distance_px=min_distance_px,
        max_candidates=max_candidates,
        patch_size=initial_patch_size,
    )
    results: list[SpotCropResult] = []
    for candidate in candidates:
        patch, roi = extract_spot_patch(
            frame,
            candidate.peak_x,
            candidate.peak_y,
            patch_size=initial_patch_size,
        )
        fit = fit_gaussian_2d(patch, allow_rotation=allow_rotation)
        fit.roi = roi
        fit.center_x = float(roi[2] + fit.center_x)
        fit.center_y = float(roi[0] + fit.center_y)

        crop_patch, crop_roi = crop_patch_from_gaussian_fit(
            frame,
            fit,
            amplitude_fraction=amplitude_fraction,
            edge_padding_px=edge_padding_px,
            keep_reference_aspect=keep_reference_aspect,
            reference_patch_size=initial_patch_size,
        )

        refined_fit = None
        if refit_cropped:
            refined_fit = fit_gaussian_2d(crop_patch, allow_rotation=allow_rotation)
            refined_fit.roi = crop_roi
            refined_fit.center_x = float(crop_roi[2] + refined_fit.center_x)
            refined_fit.center_y = float(crop_roi[0] + refined_fit.center_y)

        results.append(
            SpotCropResult(
                candidate=candidate,
                initial_fit=fit,
                crop_patch=crop_patch,
                crop_roi=crop_roi,
                refined_fit=refined_fit,
            )
        )
    return results


def track_spot_regions_in_video(
    frames: np.ndarray,
    *,
    ts: np.ndarray | None = None,
    search_roi: tuple[int, int, int, int] | None = None,
    threshold_rel: float = 0.35,
    min_distance_px: int = 9,
    max_candidates: int = 10,
    initial_patch_size: tuple[int, int] = (31, 31),
    allow_rotation: bool = True,
    amplitude_fraction: float = 0.02,
    edge_padding_px: int = 3,
    keep_reference_aspect: bool = False,
    refit_cropped: bool = True,
    max_match_distance_px: float = 18.0,
    union_padding_px: int = 2,
) -> list[SpotRegionTrack]:
    """Track multiple spots over a video and build one union ROI per spot.

    The goal is not a per-frame adaptive ROI for analysis, but one consolidated
    region per spot that is large enough to include motion, diffusion, or
    solidification over the whole video.
    """

    arr = np.asarray(frames, dtype=float)
    if arr.ndim != 3 or arr.shape[0] == 0:
        raise ValueError("frames must be a non-empty 3D array shaped (n_frames, height, width)")

    if ts is None:
        ts_arr = np.arange(arr.shape[0], dtype=float)
    else:
        ts_arr = np.asarray(ts, dtype=float)
        if ts_arr.shape != (arr.shape[0],):
            raise ValueError("ts must have one timestamp per frame")

    track_builders: list[dict[str, object]] = []
    next_track_id = 0

    for frame_idx, (frame, t_value) in enumerate(zip(arr, ts_arr)):
        detections = extract_multiple_gaussian_spot_patches(
            frame,
            search_roi=search_roi,
            threshold_rel=threshold_rel,
            min_distance_px=min_distance_px,
            max_candidates=max_candidates,
            initial_patch_size=initial_patch_size,
            allow_rotation=allow_rotation,
            amplitude_fraction=amplitude_fraction,
            edge_padding_px=edge_padding_px,
            keep_reference_aspect=keep_reference_aspect,
            refit_cropped=refit_cropped,
        )

        used_tracks: set[int] = set()
        for detection in detections:
            fit = detection.refined_fit if detection.refined_fit is not None else detection.initial_fit
            center_x = float(fit.center_x)
            center_y = float(fit.center_y)

            best_idx = None
            best_dist = None
            for idx, track in enumerate(track_builders):
                if idx in used_tracks:
                    continue
                last_x = float(track["center_x"][-1])
                last_y = float(track["center_y"][-1])
                dist = float(np.hypot(center_x - last_x, center_y - last_y))
                if dist <= max_match_distance_px and (best_dist is None or dist < best_dist):
                    best_dist = dist
                    best_idx = idx

            if best_idx is None:
                track_builders.append(
                    {
                        "track_id": next_track_id,
                        "frame_indices": [frame_idx],
                        "ts": [float(t_value)],
                        "center_x": [center_x],
                        "center_y": [center_y],
                        "fwhm_x": [float(fit.fwhm_x)],
                        "fwhm_y": [float(fit.fwhm_y)],
                        "rotation_deg": [float(fit.rotation_deg)],
                        "rmse": [float(fit.rmse)],
                        "crop_rois": [detection.crop_roi],
                    }
                )
                used_tracks.add(len(track_builders) - 1)
                next_track_id += 1
            else:
                track = track_builders[best_idx]
                track["frame_indices"].append(frame_idx)
                track["ts"].append(float(t_value))
                track["center_x"].append(center_x)
                track["center_y"].append(center_y)
                track["fwhm_x"].append(float(fit.fwhm_x))
                track["fwhm_y"].append(float(fit.fwhm_y))
                track["rotation_deg"].append(float(fit.rotation_deg))
                track["rmse"].append(float(fit.rmse))
                track["crop_rois"].append(detection.crop_roi)
                used_tracks.add(best_idx)

    tracks: list[SpotRegionTrack] = []
    h, w = arr.shape[1:]
    for track in track_builders:
        rois = track["crop_rois"]
        y0 = max(0, min(roi[0] for roi in rois) - union_padding_px)
        y1 = min(h, max(roi[1] for roi in rois) + union_padding_px)
        x0 = max(0, min(roi[2] for roi in rois) - union_padding_px)
        x1 = min(w, max(roi[3] for roi in rois) + union_padding_px)
        tracks.append(
            SpotRegionTrack(
                track_id=int(track["track_id"]),
                frame_indices=np.asarray(track["frame_indices"], dtype=int),
                ts=np.asarray(track["ts"], dtype=float),
                center_x=np.asarray(track["center_x"], dtype=float),
                center_y=np.asarray(track["center_y"], dtype=float),
                fwhm_x=np.asarray(track["fwhm_x"], dtype=float),
                fwhm_y=np.asarray(track["fwhm_y"], dtype=float),
                rotation_deg=np.asarray(track["rotation_deg"], dtype=float),
                rmse=np.asarray(track["rmse"], dtype=float),
                crop_rois=list(rois),
                union_roi=(y0, y1, x0, x1),
            )
        )
    return tracks


def analyze_spot_region_series(
    frames: np.ndarray,
    roi: tuple[int, int, int, int],
    *,
    ts: np.ndarray | None = None,
    allow_rotation: bool = True,
) -> SpotRegionSeries:
    """Analyze one fixed spot region over time after the ROI has been chosen."""

    arr = np.asarray(frames, dtype=float)
    if arr.ndim != 3 or arr.shape[0] == 0:
        raise ValueError("frames must be a non-empty 3D array shaped (n_frames, height, width)")

    if ts is None:
        ts_arr = np.arange(arr.shape[0], dtype=float)
    else:
        ts_arr = np.asarray(ts, dtype=float)
        if ts_arr.shape != (arr.shape[0],):
            raise ValueError("ts must have one timestamp per frame")

    y0, y1, x0, x1 = sanitize_roi(arr.shape[1:], roi, fraction=0.5)
    raw_sum = []
    raw_mean = []
    raw_min = []
    raw_max = []
    raw_std = []
    center_x = []
    center_y = []
    fwhm_x = []
    fwhm_y = []
    rotation_deg = []
    rmse = []

    for frame in arr:
        patch = frame[y0:y1, x0:x1]
        fit = fit_gaussian_2d(patch, allow_rotation=allow_rotation)
        raw_sum.append(float(np.sum(patch)))
        raw_mean.append(float(np.mean(patch)))
        raw_min.append(float(np.min(patch)))
        raw_max.append(float(np.max(patch)))
        raw_std.append(float(np.std(patch)))
        center_x.append(float(x0 + fit.center_x))
        center_y.append(float(y0 + fit.center_y))
        fwhm_x.append(float(fit.fwhm_x))
        fwhm_y.append(float(fit.fwhm_y))
        rotation_deg.append(float(fit.rotation_deg))
        rmse.append(float(fit.rmse))

    return SpotRegionSeries(
        roi=(y0, y1, x0, x1),
        ts=ts_arr,
        raw_sum=np.asarray(raw_sum, dtype=float),
        raw_mean=np.asarray(raw_mean, dtype=float),
        raw_min=np.asarray(raw_min, dtype=float),
        raw_max=np.asarray(raw_max, dtype=float),
        raw_std=np.asarray(raw_std, dtype=float),
        center_x=np.asarray(center_x, dtype=float),
        center_y=np.asarray(center_y, dtype=float),
        fwhm_x=np.asarray(fwhm_x, dtype=float),
        fwhm_y=np.asarray(fwhm_y, dtype=float),
        rotation_deg=np.asarray(rotation_deg, dtype=float),
        rmse=np.asarray(rmse, dtype=float),
    )


def _orientation_consistency(candidate_metrics: list[SpotCandidateMetrics]) -> float:
    weights = np.asarray([max(item.elongation, 0.0) for item in candidate_metrics], dtype=float)
    if np.all(weights <= 1e-9):
        return 0.0
    theta = np.deg2rad(np.asarray([item.fit.rotation_deg for item in candidate_metrics], dtype=float))
    vector = np.sum(weights * np.exp(2j * theta))
    return float(np.clip(np.abs(vector) / np.sum(weights), 0.0, 1.0))


def _line_alignment_metrics(centers: np.ndarray) -> tuple[float | None, float | None]:
    if centers.shape[0] < 2:
        return None, None
    centered = centers - np.mean(centers, axis=0, keepdims=True)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    major = float(max(singular[0], 1e-9))
    minor = float(singular[1]) if singular.size > 1 else 0.0
    line_rmse = float(minor / np.sqrt(max(centers.shape[0], 1)))
    anisotropy = float(major / max(minor, 1e-9))
    return line_rmse, anisotropy


def _circle_fit_rmse(centers: np.ndarray) -> float | None:
    if centers.shape[0] < 3:
        return None
    x = centers[:, 0]
    y = centers[:, 1]
    design = np.column_stack([x, y, np.ones_like(x)])
    rhs = -(x**2 + y**2)
    coef, *_ = np.linalg.lstsq(design, rhs, rcond=None)
    a, b, c = coef
    center_x = -0.5 * float(a)
    center_y = -0.5 * float(b)
    radius_sq = center_x**2 + center_y**2 - float(c)
    if radius_sq <= 0:
        return None
    radius = float(np.sqrt(radius_sq))
    radial = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    return float(np.sqrt(np.mean((radial - radius) ** 2)))


def _nearest_neighbor_spacing_cv(centers: np.ndarray) -> float | None:
    if centers.shape[0] < 3:
        return None
    diff = centers[:, None, :] - centers[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    nn = np.min(dist, axis=1)
    mean_nn = float(np.mean(nn))
    if mean_nn <= 1e-9:
        return None
    return float(np.std(nn) / mean_nn)


def _alignment_score(rmse_px: float | None, scene_diag_px: float) -> float:
    if rmse_px is None:
        return 0.0
    scale = max(0.12 * max(scene_diag_px, 1.0), 2.0)
    return float(np.clip(1.0 - rmse_px / scale, 0.0, 1.0))


__all__ = [
    "CandidatePatternSummary",
    "GaussianFitResult",
    "SpotCandidateMetrics",
    "SpotCropResult",
    "SpotRegionSeries",
    "SpotRegionTrack",
    "analyze_spot_candidates",
    "analyze_spot_region_series",
    "classify_growth_pattern_from_candidates",
    "crop_patch_from_gaussian_fit",
    "SpotCandidate",
    "detect_bright_spots",
    "estimate_gaussian_moments",
    "extract_spot_patch",
    "extract_multiple_gaussian_spot_patches",
    "fit_gaussian_2d",
    "gaussian_2d",
    "gaussian_function",
    "locate_fit_and_crop_spot",
    "locate_and_fit_spot",
    "reconstruct_gaussian_patch",
    "summarize_candidate_geometry",
    "track_spot_regions_in_video",
]

