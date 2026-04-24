from __future__ import annotations

"""Low-level signal-processing primitives adapted from the legacy draft code.

These functions are intentionally small and dependency-light. Use them when you
want one specific operation without invoking the higher-level offline pipeline.
`offline_analysis.py` is the recommended entry point for most users.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class TauEstimate:
    """Result of fitting a single growth/decay relaxation segment."""

    tau_s: float | None
    mode: str
    loss: float | None
    n_points: int
    start_ts: float | None = None
    end_ts: float | None = None


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Smooth a 1D signal with a centered moving average.

    Use this only for light smoothing. If cycle edges matter, keep `window`
    small because large values smear the shape of each oscillation.
    """

    arr = np.asarray(values, dtype=float)
    if window <= 1 or arr.size == 0:
        return arr.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(arr, kernel, mode="same")


def median_filter_1d(values: np.ndarray, kernel_size: int) -> np.ndarray:
    """Median filter with edge-padding to keep original length.

    Best for isolated spikes or impulsive camera/software noise.
    `kernel_size=3-7` is a practical starting range.
    """

    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or kernel_size <= 1:
        return arr.copy()

    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    padded = np.pad(arr, pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel_size)
    return np.median(windows, axis=1)


def bandpass_filter_fft(
    values: np.ndarray,
    low_cutoff: float,
    high_cutoff: float,
    sample_frequency: float,
) -> np.ndarray:
    """Apply a simple FFT band-pass mask and inverse-transform.

    Set `low_cutoff` above the slow baseline drift and `high_cutoff` above the
    oscillation frequency but below obvious high-frequency noise.
    """

    arr = np.asarray(values, dtype=float)
    if arr.size < 8 or sample_frequency <= 0 or high_cutoff <= 0:
        return arr.copy()

    low = max(0.0, float(low_cutoff))
    high = min(float(high_cutoff), 0.5 * sample_frequency)
    if high <= low:
        return arr.copy()

    fft = np.fft.rfft(arr)
    freq = np.fft.rfftfreq(arr.size, d=1.0 / sample_frequency)
    mask = (freq >= low) & (freq <= high)
    return np.fft.irfft(fft * mask, n=arr.size)


def detect_peaks_1d(values: np.ndarray, min_distance: int, prominence: float = 0.0) -> np.ndarray:
    """Detect local maxima with spacing and prominence constraints.

    Use this when the trace already has obvious local maxima. Increase
    `min_distance` to avoid multiple detections in one cycle, and increase
    `prominence` to reject noise peaks.
    """

    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n < 3:
        return np.array([], dtype=int)

    min_distance = max(1, int(min_distance))
    candidates: list[int] = []
    for i in range(1, n - 1):
        if arr[i - 1] < arr[i] >= arr[i + 1]:
            left = arr[max(0, i - min_distance) : i + 1]
            right = arr[i : min(n, i + min_distance + 1)]
            local_prom = arr[i] - max(float(np.min(left)), float(np.min(right)))
            if local_prom >= prominence:
                candidates.append(i)

    if not candidates:
        return np.array([], dtype=int)

    selected: list[int] = []
    for idx in sorted(candidates, key=lambda j: arr[j], reverse=True):
        if all(abs(idx - kept) >= min_distance for kept in selected):
            selected.append(idx)
    selected.sort()
    return np.asarray(selected, dtype=int)


def detect_peaks_step_1d(
    values: np.ndarray,
    min_distance: int,
    convolve_step: int = 5,
    prominence: float = 0.0,
    mode: str = "same",
) -> np.ndarray:
    """Detect cycle boundaries after a step-convolution transform.

    This is often better than direct peak finding for growth-like signals with
    sharp resets and gradual recoveries. `convolve_step=3-7` is a common range.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size < 3:
        return np.array([], dtype=int)

    if convolve_step <= 1:
        return detect_peaks_1d(arr, min_distance=min_distance, prominence=prominence)

    step = np.hstack((np.ones(convolve_step), -1.0 * np.ones(convolve_step)))
    conv = np.convolve(arr, step, mode=mode)
    conv_abs = np.abs(conv) / float(convolve_step)
    return detect_peaks_1d(conv_abs, min_distance=min_distance, prominence=prominence)


def segment_cycles(ts: np.ndarray, values: np.ndarray, peak_indices: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split a full signal into peak-to-peak cycle segments."""

    t = np.asarray(ts, dtype=float)
    y = np.asarray(values, dtype=float)
    peaks = np.asarray(peak_indices, dtype=int)
    if t.size != y.size or t.size < 3 or peaks.size < 2:
        return []

    cycles: list[tuple[np.ndarray, np.ndarray]] = []
    for left, right in zip(peaks[:-1], peaks[1:]):
        if right - left < 3:
            continue
        cycles.append((t[left : right + 1], y[left : right + 1]))
    return cycles


def normalize_0_1(
    y: np.ndarray,
    i_start: float | None = None,
    i_end: float | None = None,
    i_diff: float | None = None,
    unify: bool = True,
) -> tuple[np.ndarray, float, float, float]:
    """Normalize one cycle to an approximate 0-to-1 scale.

    Use shared `i_start`, `i_end`, or `i_diff` when multiple cycles should be
    compared on one consistent scale instead of each being normalized
    independently.
    """

    arr = np.asarray(y, dtype=float)
    if arr.size == 0:
        return arr.copy(), 0.0, 0.0, 1.0

    n_avg = max(3, arr.size // 100 + 3)
    i_start = float(np.mean(arr[:n_avg])) if i_start is None else float(i_start)
    i_end = float(np.mean(arr[-n_avg:])) if i_end is None else float(i_end)
    i_diff = float(i_end - i_start) if i_diff is None else float(i_diff)

    if abs(i_diff) < 1e-12:
        return arr - i_start, i_start, i_end, 1.0

    if unify:
        y_nor = (arr - i_start) / i_diff
    else:
        if i_end < i_start:
            y_nor = (arr - i_end) / abs(i_diff)
        else:
            y_nor = (arr - i_start) / abs(i_diff)
    return y_nor, i_start, i_end, i_diff


def fit_relaxation_tau(
    x: np.ndarray,
    y: np.ndarray,
    mode: str = "auto",
    min_points: int = 8,
) -> TauEstimate:
    """Estimate a growth/decay relaxation time from one cycle.

    Use `mode="growth"` for monotonic recovery toward saturation, `mode="decay"`
    for decays, and `mode="auto"` if the direction is not known in advance.
    `min_points` should be large enough that one cycle contains a visible shape,
    not just a few samples.
    """

    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)

    if xs.size != ys.size or xs.size < min_points:
        return TauEstimate(tau_s=None, mode="insufficient_data", loss=None, n_points=xs.size)

    xs = xs - xs[0]
    dt = float(np.median(np.diff(xs))) if xs.size > 1 else 0.0
    x_span = float(xs[-1] - xs[0]) if xs.size > 1 else 0.0
    if dt <= 0 or x_span <= 0:
        return TauEstimate(tau_s=None, mode="insufficient_data", loss=None, n_points=xs.size)

    tau_min = max(2.0 * dt, x_span / 150.0)
    tau_max = max(5.0 * dt, 3.0 * x_span)
    tau_grid = np.geomspace(tau_min, tau_max, 120)

    def best_fit(candidate_mode: str) -> tuple[float, float] | None:
        best_tau: float | None = None
        best_loss = np.inf
        for tau in tau_grid:
            if candidate_mode == "growth":
                phi = 1.0 - np.exp(-xs / tau)
            else:
                phi = np.exp(-xs / tau)
            design = np.column_stack([np.ones_like(phi), phi])
            coef, *_ = np.linalg.lstsq(design, ys, rcond=None)
            y_hat = design @ coef
            loss = float(np.mean((ys - y_hat) ** 2))
            if loss < best_loss:
                best_loss = loss
                best_tau = float(tau)
        if best_tau is None:
            return None
        return best_tau, best_loss

    results: list[tuple[str, float, float]] = []
    if mode in ("auto", "growth"):
        fit = best_fit("growth")
        if fit is not None:
            results.append(("growth", fit[0], fit[1]))

    if mode in ("auto", "decay"):
        fit = best_fit("decay")
        if fit is not None:
            results.append(("decay", fit[0], fit[1]))

    if not results:
        return TauEstimate(tau_s=None, mode="fit_failed", loss=None, n_points=xs.size)

    best_mode, best_tau, best_loss = min(results, key=lambda item: item[2])
    return TauEstimate(
        tau_s=best_tau,
        mode=best_mode,
        loss=best_loss,
        n_points=xs.size,
        start_ts=float(x[0]),
        end_ts=float(x[-1]),
    )


def estimate_latest_cycle_tau(
    ts: np.ndarray,
    values: np.ndarray,
    min_distance: int,
    prominence: float,
    mode: str = "auto",
    min_points: int = 8,
) -> TauEstimate | None:
    """Fit only the most recent detected cycle in a rolling trace.

    This is the bridge used by the streaming tracker to surface a latest-cycle
    `tau` estimate without reprocessing the full history.
    """

    t = np.asarray(ts, dtype=float)
    y = np.asarray(values, dtype=float)
    if t.size != y.size or t.size < min_points:
        return None

    peaks = detect_peaks_1d(y, min_distance=min_distance, prominence=prominence)
    if peaks.size < 2:
        return None

    start, end = int(peaks[-2]), int(peaks[-1])
    if end - start + 1 < min_points:
        return None

    est = fit_relaxation_tau(t[start : end + 1], y[start : end + 1], mode=mode, min_points=min_points)
    if est.tau_s is None:
        return None
    return est


def trim_cycle_tail(y: np.ndarray, ratio: float = 0.1) -> np.ndarray:
    """Replace the trailing ratio of one cycle with earlier points.

    Use this only when the cycle tail is known to be corrupted by repeated
    readout artifacts. Keep `ratio` small; `0.05-0.15` is usually enough.
    """
    arr = np.asarray(y, dtype=float).copy()
    if arr.size < 4 or ratio <= 0.0:
        return arr

    n = max(1, int(arr.size * ratio))
    if 2 * n >= arr.size:
        return arr
    arr[-n:] = arr[-2 * n : -n]
    return arr


def remove_linear_background(x: np.ndarray, y: np.ndarray, linear_ratio: float = 0.8) -> np.ndarray:
    """Subtract a fitted linear background from the trailing part of a cycle.

    `linear_ratio` sets how much of the trailing cycle is used to estimate the
    line. Increase it if the end of the cycle is mostly baseline; decrease it if
    the end still contains strong nonlinear signal.
    """
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    if xs.size != ys.size or xs.size < 4:
        return ys.copy()
    if linear_ratio <= 0:
        return ys.copy()

    length = max(3, int(xs.size * linear_ratio))
    x_tail = xs[-length:]
    y_tail = ys[-length:]

    slope, intercept = np.polyfit(x_tail, y_tail, 1)
    bg = slope * xs + intercept
    return ys - bg

