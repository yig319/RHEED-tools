from __future__ import annotations

"""Normalized exponential curve helpers migrated from RHEED-Learn."""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ExponentialFit:
    """Result from fitting one normalized exponential curve."""

    amplitude: float
    offset: float
    tau_s: float
    rmse: float
    mode: str
    y_fit: np.ndarray


def normalize_0_1(
    values: np.ndarray,
    start: float | None = None,
    end: float | None = None,
    amplitude: float | None = None,
    *,
    unify: bool = True,
) -> np.ndarray:
    """Normalize a curve with the convention used by archived RHEED fits."""

    y = np.asarray(values, dtype=float)
    if y.size == 0:
        return y.copy()
    y_start = float(y[0] if start is None else start)
    y_end = float(y[-1] if end is None else end)
    y_amp = float(abs(y_end - y_start) if amplitude is None else amplitude)
    if y_amp <= 0:
        return np.zeros_like(y)
    out = (y - y_start) / y_amp
    if unify and y_end < y_start:
        out = -out
    return out


def denormalize_0_1(
    normalized_values: np.ndarray,
    start: float,
    end: float,
    amplitude: float | None = None,
    *,
    unify: bool = True,
) -> np.ndarray:
    """Invert `normalize_0_1`."""

    y_norm = np.asarray(normalized_values, dtype=float)
    y_amp = float(abs(end - start) if amplitude is None else amplitude)
    signed = -y_norm if unify and end < start else y_norm
    return signed * y_amp + float(start)


def exponential_growth(x: np.ndarray, amplitude: float, offset: float, tau_s: float) -> np.ndarray:
    """Increasing exponential saturation model."""

    tau = max(float(tau_s), 1e-12)
    return float(amplitude) * (1.0 - np.exp(-np.asarray(x, dtype=float) / tau)) + float(offset)


def exponential_decay(x: np.ndarray, amplitude: float, offset: float, tau_s: float) -> np.ndarray:
    """Decreasing exponential decay model."""

    tau = max(float(tau_s), 1e-12)
    return float(amplitude) * np.exp(-np.asarray(x, dtype=float) / tau) + float(offset)


def fit_exponential_curve(
    x: np.ndarray,
    y: np.ndarray,
    *,
    mode: str = "auto",
    initial: tuple[float, float, float] = (1.0, 0.0, 0.4),
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]] = ((-np.inf, -np.inf, 1e-6), (np.inf, np.inf, np.inf)),
) -> ExponentialFit:
    """Fit a growth or decay exponential model to one curve.

    Uses SciPy when available and falls back to a log-linear tau estimate for
    simple monotonic curves.
    """

    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    if xs.size != ys.size:
        raise ValueError("x and y must have the same length")
    if xs.size < 3:
        raise ValueError("at least three points are required")
    xs_rel = xs - xs[0]

    fit_mode = _choose_mode(ys) if mode == "auto" else mode
    if fit_mode not in {"growth", "decay"}:
        raise ValueError("mode must be 'auto', 'growth', or 'decay'")
    model = exponential_growth if fit_mode == "growth" else exponential_decay

    try:
        from scipy.optimize import curve_fit

        params, _ = curve_fit(model, xs_rel, ys, p0=initial, bounds=bounds, maxfev=20_000)
        y_fit = model(xs_rel, *params)
        rmse = float(np.sqrt(np.mean((ys - y_fit) ** 2)))
        return ExponentialFit(
            amplitude=float(params[0]),
            offset=float(params[1]),
            tau_s=float(params[2]),
            rmse=rmse,
            mode=fit_mode,
            y_fit=y_fit,
        )
    except Exception:
        tau = _log_linear_tau(xs_rel, ys, fit_mode)
        offset = float(np.min(ys) if fit_mode == "growth" else ys[-1])
        amplitude = float((np.max(ys) - offset) if fit_mode == "growth" else (ys[0] - offset))
        y_fit = model(xs_rel, amplitude, offset, tau)
        rmse = float(np.sqrt(np.mean((ys - y_fit) ** 2)))
        return ExponentialFit(amplitude=amplitude, offset=offset, tau_s=tau, rmse=rmse, mode=fit_mode, y_fit=y_fit)


def _choose_mode(y: np.ndarray) -> str:
    return "growth" if float(y[-1]) >= float(y[0]) else "decay"


def _log_linear_tau(x: np.ndarray, y: np.ndarray, mode: str) -> float:
    if mode == "growth":
        target = float(np.max(y))
        signal = target - y
    else:
        target = float(np.min(y))
        signal = y - target
    keep = signal > max(1e-12, 0.05 * float(np.max(signal)))
    if np.count_nonzero(keep) < 2:
        return float(max(x[-1] - x[0], 1e-6))
    slope, _ = np.polyfit(x[keep], np.log(signal[keep]), 1)
    if slope >= 0:
        return float(max(x[-1] - x[0], 1e-6))
    return float(-1.0 / slope)
