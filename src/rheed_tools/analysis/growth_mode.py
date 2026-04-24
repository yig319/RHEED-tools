from __future__ import annotations

"""Heuristic growth-mode classification from combined RHEED features."""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class GrowthFeatureVector:
    """Compact combined feature set for one frame or one summary interval."""

    specular_intensity: float | None = None
    oscillation_amplitude: float | None = None
    damping_tau_s: float | None = None
    aspect_ratio: float | None = None
    streakiness: float | None = None
    tilt_deg: float | None = None
    diffuse_ratio: float | None = None
    lattice_spacing_px: float | None = None
    background_level: float | None = None


@dataclass(slots=True)
class GrowthModeDecision:
    """Heuristic growth-mode label and explanation."""

    label: str
    confidence: float
    scores: dict[str, float]
    reasons: list[str]


@dataclass(slots=True)
class GrowthModeSeries:
    """Per-frame growth-mode decisions."""

    ts: np.ndarray
    labels: list[str]
    confidence: np.ndarray
    layer_by_layer_score: np.ndarray
    step_flow_score: np.ndarray
    island_3d_score: np.ndarray
    mixed_score: np.ndarray


def build_growth_feature_vector(
    *,
    specular_metrics=None,
    geometry_metrics=None,
    diffuse_metrics=None,
    reciprocal_metrics=None,
    oscillation_amplitude: float | None = None,
    damping_tau_s: float | None = None,
) -> GrowthFeatureVector:
    """Build one combined feature vector from module outputs.

    This helper accepts metrics objects from the new analysis modules so the
    control layer can work with one compact feature record.
    """

    feature = GrowthFeatureVector(
        oscillation_amplitude=oscillation_amplitude,
        damping_tau_s=damping_tau_s,
    )
    if specular_metrics is not None:
        feature.specular_intensity = float(getattr(specular_metrics, "corrected_sum", np.nan))
        feature.background_level = float(getattr(specular_metrics, "background_mean", np.nan))
    if geometry_metrics is not None:
        feature.aspect_ratio = float(getattr(geometry_metrics, "aspect_ratio", np.nan))
        if hasattr(geometry_metrics, "streak_length_px") and hasattr(geometry_metrics, "streak_width_px"):
            length = float(getattr(geometry_metrics, "streak_length_px"))
            width = float(getattr(geometry_metrics, "streak_width_px"))
            feature.streakiness = float((length - width) / (length + width + 1e-9))
        feature.tilt_deg = float(getattr(geometry_metrics, "tilt_deg", np.nan))
        feature.lattice_spacing_px = float(getattr(geometry_metrics, "spacing_px", np.nan))
    if diffuse_metrics is not None:
        feature.diffuse_ratio = float(getattr(diffuse_metrics, "diffuse_to_signal_ratio", np.nan))
    if reciprocal_metrics is not None and feature.lattice_spacing_px is None:
        feature.lattice_spacing_px = float(getattr(reciprocal_metrics, "spacing_px", np.nan))

    def normalize_optional(value: float | None) -> float | None:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return None
        return float(value)

    return GrowthFeatureVector(
        specular_intensity=normalize_optional(feature.specular_intensity),
        oscillation_amplitude=normalize_optional(feature.oscillation_amplitude),
        damping_tau_s=normalize_optional(feature.damping_tau_s),
        aspect_ratio=normalize_optional(feature.aspect_ratio),
        streakiness=normalize_optional(feature.streakiness),
        tilt_deg=normalize_optional(feature.tilt_deg),
        diffuse_ratio=normalize_optional(feature.diffuse_ratio),
        lattice_spacing_px=normalize_optional(feature.lattice_spacing_px),
        background_level=normalize_optional(feature.background_level),
    )


def classify_growth_mode(
    features: GrowthFeatureVector,
    *,
    oscillation_threshold: float = 0.12,
    diffuse_threshold: float = 0.45,
    streakiness_threshold: float = 0.22,
    tilt_threshold_deg: float = 12.0,
) -> GrowthModeDecision:
    """Classify growth mode from combined RHEED features.

    Heuristic intent:
    - strong oscillation + streaky + low diffuse -> layer-by-layer
    - low oscillation + tilted streaks + low diffuse -> step-flow
    - high diffuse + spotty / weak streakiness -> 3D island
    - everything in between -> mixed
    """

    oscillation = 0.0 if features.oscillation_amplitude is None else max(0.0, float(features.oscillation_amplitude))
    diffuse = 0.0 if features.diffuse_ratio is None else max(0.0, float(features.diffuse_ratio))
    streakiness = 0.0 if features.streakiness is None else max(0.0, float(features.streakiness))
    tilt = 0.0 if features.tilt_deg is None else abs(float(features.tilt_deg))
    damping_bonus = 0.0
    if features.damping_tau_s is not None and np.isfinite(features.damping_tau_s):
        damping_bonus = min(float(features.damping_tau_s) / 50.0, 1.0)

    layer_by_layer = 0.45 * min(oscillation / max(oscillation_threshold, 1e-9), 1.5)
    layer_by_layer += 0.30 * min(streakiness / max(streakiness_threshold, 1e-9), 1.5)
    layer_by_layer += 0.15 * (1.0 - min(diffuse / max(diffuse_threshold, 1e-9), 1.5))
    layer_by_layer += 0.10 * damping_bonus

    step_flow = 0.45 * min(tilt / max(tilt_threshold_deg, 1e-9), 1.5)
    step_flow += 0.30 * min(streakiness / max(streakiness_threshold, 1e-9), 1.5)
    step_flow += 0.25 * (1.0 - min(oscillation / max(oscillation_threshold, 1e-9), 1.5))

    island_3d = 0.50 * min(diffuse / max(diffuse_threshold, 1e-9), 1.5)
    island_3d += 0.30 * (1.0 - min(streakiness / max(streakiness_threshold, 1e-9), 1.5))
    island_3d += 0.20 * (1.0 - min(oscillation / max(oscillation_threshold, 1e-9), 1.5))

    mixed = 1.0 - float(np.std([layer_by_layer, step_flow, island_3d]))
    scores = {
        "layer_by_layer": float(layer_by_layer),
        "step_flow": float(step_flow),
        "island_3d": float(island_3d),
        "mixed": float(mixed),
    }
    label = max(scores, key=scores.get)
    sorted_scores = sorted(scores.values(), reverse=True)
    confidence = 0.0 if len(sorted_scores) < 2 else max(0.0, min(1.0, sorted_scores[0] - sorted_scores[1] + 0.5))

    reasons: list[str] = []
    if label == "layer_by_layer":
        reasons.append("oscillation and streakiness are both relatively strong")
        if features.diffuse_ratio is not None:
            reasons.append("diffuse background remains comparatively low")
    elif label == "step_flow":
        reasons.append("tilted streak geometry is stronger than oscillatory behavior")
        reasons.append("the feature remains elongated rather than spotty")
    elif label == "island_3d":
        reasons.append("diffuse scattering is elevated relative to the main signal")
        reasons.append("the feature is comparatively spot-like or weakly streaky")
    else:
        reasons.append("multiple signatures are present without one dominant growth fingerprint")

    return GrowthModeDecision(
        label=label,
        confidence=float(confidence),
        scores=scores,
        reasons=reasons,
    )


def classify_growth_mode_series(
    feature_vectors: list[GrowthFeatureVector],
    *,
    ts: np.ndarray | None = None,
    oscillation_threshold: float = 0.12,
    diffuse_threshold: float = 0.45,
    streakiness_threshold: float = 0.22,
    tilt_threshold_deg: float = 12.0,
) -> GrowthModeSeries:
    """Classify growth mode for a list of feature vectors."""

    if len(feature_vectors) == 0:
        raise ValueError("feature_vectors must not be empty")
    if ts is None:
        ts_arr = np.arange(len(feature_vectors), dtype=float)
    else:
        ts_arr = np.asarray(ts, dtype=float)
        if ts_arr.shape != (len(feature_vectors),):
            raise ValueError("ts must have one timestamp per feature vector")

    decisions = [
        classify_growth_mode(
            features,
            oscillation_threshold=oscillation_threshold,
            diffuse_threshold=diffuse_threshold,
            streakiness_threshold=streakiness_threshold,
            tilt_threshold_deg=tilt_threshold_deg,
        )
        for features in feature_vectors
    ]

    return GrowthModeSeries(
        ts=ts_arr,
        labels=[item.label for item in decisions],
        confidence=np.asarray([item.confidence for item in decisions], dtype=float),
        layer_by_layer_score=np.asarray([item.scores["layer_by_layer"] for item in decisions], dtype=float),
        step_flow_score=np.asarray([item.scores["step_flow"] for item in decisions], dtype=float),
        island_3d_score=np.asarray([item.scores["island_3d"] for item in decisions], dtype=float),
        mixed_score=np.asarray([item.scores["mixed"] for item in decisions], dtype=float),
    )


def detect_growth_transitions(labels: list[str]) -> list[tuple[int, str, str]]:
    """Return index and label pairs whenever the classified growth mode changes."""

    transitions: list[tuple[int, str, str]] = []
    for idx in range(1, len(labels)):
        if labels[idx] != labels[idx - 1]:
            transitions.append((idx, labels[idx - 1], labels[idx]))
    return transitions


__all__ = [
    "GrowthFeatureVector",
    "GrowthModeDecision",
    "GrowthModeSeries",
    "build_growth_feature_vector",
    "classify_growth_mode",
    "classify_growth_mode_series",
    "detect_growth_transitions",
]

