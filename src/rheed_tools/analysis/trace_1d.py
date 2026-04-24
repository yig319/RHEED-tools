from __future__ import annotations

"""Offline post-processing helpers for 1D RHEED traces."""

from dataclasses import dataclass

import numpy as np

from rheed_tools.io.trace_io import load_trace_file

from .background import estimate_rolling_background, subtract_rolling_background
from ..signals import (
    TauEstimate,
    bandpass_filter_fft,
    detect_peaks_step_1d,
    fit_relaxation_tau,
    median_filter_1d,
    remove_linear_background,
    trim_cycle_tail,
)


@dataclass(slots=True)
class CycleFit:
    """Container for one peak-to-peak cycle and its fit output."""

    cycle_index: int
    start_ts: float
    end_ts: float
    tau: TauEstimate
    x: np.ndarray
    y: np.ndarray
    y_processed: np.ndarray


@dataclass(slots=True)
class CycleMetrics:
    """Common per-cycle intensity summary metrics."""

    cycle_index: int
    start_ts: float
    end_ts: float
    duration_s: float
    trough_intensity: float
    peak_intensity: float
    amplitude: float
    mean_intensity: float
    integrated_intensity: float
    end_minus_start: float


@dataclass(slots=True)
class OscillationSummary:
    """Common trace-level RHEED oscillation metrics."""

    dominant_frequency_hz: float | None
    dominant_period_s: float | None
    fft_power: float | None
    cycle_count: int
    median_cycle_period_s: float | None
    median_cycle_amplitude: float | None
    rms_amplitude: float | None
    damping_tau_s: float | None


@dataclass(slots=True)
class PulseTrace:
    """One laser-pulse-resolved trace segment cut from a continuous signal."""

    pulse_index: int
    start_ts: float
    end_ts: float
    x: np.ndarray
    y: np.ndarray


@dataclass(slots=True)
class PulseRelaxationFit:
    """One pulse-resolved relaxation fit and its cleaned trace."""

    pulse_index: int
    start_ts: float
    end_ts: float
    tau: TauEstimate
    x: np.ndarray
    y: np.ndarray
    y_processed: np.ndarray


def select_range(data: np.ndarray, start: float, end: float, y_col: int = 1) -> np.ndarray:
    """Select one time window from exported data."""

    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2 or arr.shape[1] <= y_col:
        raise ValueError("data must be a 2D array with enough columns")
    mask = (arr[:, 0] > start) & (arr[:, 0] < end)
    return np.stack([arr[mask, 0], arr[mask, y_col]], axis=1)


def preprocess_signal(
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    sample_rate_hz: float,
    median_kernel_size: int | None = 5,
    fft_band: tuple[float, float] | None = (0.05, 5.0),
    smooth_window: int | None = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply denoising passes to a recorded 1D RHEED trace."""

    x = np.asarray(sample_x, dtype=float)
    y = np.asarray(sample_y, dtype=float)
    if x.size != y.size:
        raise ValueError("sample_x and sample_y must have equal length")

    y_proc = y.copy()
    if median_kernel_size is not None:
        y_proc = median_filter_1d(y_proc, median_kernel_size)
    if fft_band is not None:
        y_proc = bandpass_filter_fft(y_proc, fft_band[0], fft_band[1], sample_rate_hz)
    if smooth_window is not None and smooth_window > 1:
        kernel = np.ones(smooth_window, dtype=float) / float(smooth_window)
        y_proc = np.convolve(y_proc, kernel, mode="same")
    return x, y_proc


def detect_cycle_boundaries(
    sample_y: np.ndarray,
    camera_freq: float,
    laser_freq: float,
    convolve_step: int = 5,
    prominence: float = 0.1,
) -> np.ndarray:
    """Detect cycle boundaries using a frequency prior and local prominence."""

    if camera_freq <= 0 or laser_freq <= 0:
        raise ValueError("camera_freq and laser_freq must be > 0")
    min_distance = max(1, int(camera_freq / laser_freq * 0.6))
    return detect_peaks_step_1d(
        np.asarray(sample_y, dtype=float),
        min_distance=min_distance,
        convolve_step=convolve_step,
        prominence=prominence,
        mode="same",
    )


def split_cycles(
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    peak_indices: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split one trace into peak-to-peak cycle arrays."""

    x = np.asarray(sample_x, dtype=float)
    y = np.asarray(sample_y, dtype=float)
    peaks = np.asarray(peak_indices, dtype=int)
    if x.size != y.size:
        raise ValueError("sample_x and sample_y must have equal length")
    if peaks.size < 2:
        return []

    cycles: list[tuple[np.ndarray, np.ndarray]] = []
    for left, right in zip(peaks[:-1], peaks[1:]):
        if right - left < 3:
            continue
        cycles.append((x[left:right], y[left:right]))
    return cycles


def split_pulse_traces(
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    *,
    laser_rate_hz: float,
    phase_offset_s: float = 0.0,
    min_points: int = 8,
) -> list[PulseTrace]:
    """Split a continuous trace into one segment per laser pulse interval.

    This is useful when each pulse generates one recovery-like intensity trace
    that should be fitted independently for a per-pulse relaxation `tau`.

    Tuning guidance:
    - `laser_rate_hz` is the pulse repetition rate in pulses per second.
    - `phase_offset_s` shifts the pulse boundaries relative to the first sample.
      Adjust it when the pulse onset does not line up with the first timestamp.
    - `min_points` should stay large enough that the pulse fit still sees a
      meaningful curve shape.
    """

    x = np.asarray(sample_x, dtype=float)
    y = np.asarray(sample_y, dtype=float)
    if x.size != y.size:
        raise ValueError("sample_x and sample_y must have equal length")
    if x.size == 0:
        return []
    if laser_rate_hz <= 0:
        raise ValueError("laser_rate_hz must be > 0")
    if min_points < 2:
        raise ValueError("min_points must be >= 2")

    period_s = 1.0 / float(laser_rate_hz)
    rel_t = x - float(x[0]) - float(phase_offset_s)
    pulse_ids = np.floor(rel_t / period_s).astype(int)
    keep = pulse_ids >= 0
    if not np.any(keep):
        return []

    x_keep = x[keep]
    y_keep = y[keep]
    pulse_ids_keep = pulse_ids[keep]

    traces: list[PulseTrace] = []
    for pulse_id in np.unique(pulse_ids_keep):
        mask = pulse_ids_keep == pulse_id
        if np.count_nonzero(mask) < min_points:
            continue
        x_seg = x_keep[mask]
        y_seg = y_keep[mask]
        traces.append(
            PulseTrace(
                pulse_index=int(pulse_id),
                start_ts=float(x_seg[0]),
                end_ts=float(x_seg[-1]),
                x=x_seg,
                y=y_seg,
            )
        )
    return traces


def compute_cycle_metrics(
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    peak_indices: np.ndarray,
) -> list[CycleMetrics]:
    """Summarize each cycle with basic intensity statistics."""

    cycles = split_cycles(sample_x, sample_y, peak_indices)
    metrics: list[CycleMetrics] = []
    for idx, (x_cycle, y_cycle) in enumerate(cycles):
        metrics.append(
            CycleMetrics(
                cycle_index=idx,
                start_ts=float(x_cycle[0]),
                end_ts=float(x_cycle[-1]),
                duration_s=float(x_cycle[-1] - x_cycle[0]),
                trough_intensity=float(np.min(y_cycle)),
                peak_intensity=float(np.max(y_cycle)),
                amplitude=float(np.max(y_cycle) - np.min(y_cycle)),
                mean_intensity=float(np.mean(y_cycle)),
                integrated_intensity=float(np.trapezoid(y_cycle, x_cycle)),
                end_minus_start=float(y_cycle[-1] - y_cycle[0]),
            )
        )
    return metrics


def summarize_oscillation_signal(
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    peak_indices: np.ndarray | None = None,
) -> OscillationSummary:
    """Estimate whole-trace oscillation metrics from a 1D signal."""

    x = np.asarray(sample_x, dtype=float)
    y = np.asarray(sample_y, dtype=float)
    if x.size != y.size:
        raise ValueError("sample_x and sample_y must have equal length")
    if x.size < 4:
        return OscillationSummary(None, None, None, 0, None, None, None, None)

    sample_rate_hz = _infer_sample_rate_hz(x)
    y_centered = y - np.mean(y)

    dominant_frequency_hz: float | None = None
    dominant_period_s: float | None = None
    fft_power: float | None = None
    if sample_rate_hz is not None:
        freq = np.fft.rfftfreq(x.size, d=1.0 / sample_rate_hz)
        spectrum = np.abs(np.fft.rfft(y_centered)) ** 2
        if spectrum.size > 1:
            idx = int(np.argmax(spectrum[1:]) + 1)
            dominant_frequency_hz = float(freq[idx])
            dominant_period_s = None if dominant_frequency_hz <= 0 else float(1.0 / dominant_frequency_hz)
            fft_power = float(spectrum[idx])

    peaks = np.asarray([], dtype=int) if peak_indices is None else np.asarray(peak_indices, dtype=int)
    cycle_metrics = compute_cycle_metrics(x, y, peaks) if peaks.size >= 2 else []

    median_cycle_period_s = None
    median_cycle_amplitude = None
    damping_tau_s = None
    if cycle_metrics:
        periods = np.asarray([item.duration_s for item in cycle_metrics], dtype=float)
        amplitudes = np.asarray([item.amplitude for item in cycle_metrics], dtype=float)
        times = np.asarray([item.start_ts for item in cycle_metrics], dtype=float)
        median_cycle_period_s = float(np.median(periods))
        median_cycle_amplitude = float(np.median(amplitudes))
        damping_tau_s = _fit_decay_tau(times, amplitudes)

    return OscillationSummary(
        dominant_frequency_hz=dominant_frequency_hz,
        dominant_period_s=dominant_period_s,
        fft_power=fft_power,
        cycle_count=len(cycle_metrics),
        median_cycle_period_s=median_cycle_period_s,
        median_cycle_amplitude=median_cycle_amplitude,
        rms_amplitude=float(np.sqrt(np.mean(y_centered**2))),
        damping_tau_s=damping_tau_s,
    )


def process_cycle_curve(
    x: np.ndarray,
    y: np.ndarray,
    tune_tail: bool = True,
    trim_first: int = 0,
    linear_ratio: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply per-cycle cleanup before exponential fitting."""

    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    y_proc = ys.copy()

    if tune_tail:
        y_proc = trim_cycle_tail(y_proc, ratio=0.1)

    if trim_first > 0 and trim_first < y_proc.size:
        y_proc = y_proc[trim_first:]
        xs = np.linspace(xs[0], xs[-1], y_proc.size)

    if linear_ratio is not None and linear_ratio > 0:
        y_proc = remove_linear_background(xs, y_proc, linear_ratio=linear_ratio)
    return xs, y_proc


def analyze_rheed_signal(
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    camera_freq: float,
    laser_freq: float,
    convolve_step: int = 5,
    prominence: float = 0.1,
    tune_tail: bool = True,
    trim_first: int = 0,
    linear_ratio: float = 0.8,
    fit_mode: str = "auto",
) -> list[CycleFit]:
    """Run the full legacy-style offline cycle analysis pipeline."""

    peaks = detect_cycle_boundaries(
        sample_y=sample_y,
        camera_freq=camera_freq,
        laser_freq=laser_freq,
        convolve_step=convolve_step,
        prominence=prominence,
    )
    cycles = split_cycles(sample_x, sample_y, peaks)

    results: list[CycleFit] = []
    for i, (x_cycle, y_cycle) in enumerate(cycles):
        x_proc, y_proc = process_cycle_curve(
            x_cycle,
            y_cycle,
            tune_tail=tune_tail,
            trim_first=trim_first,
            linear_ratio=linear_ratio,
        )
        tau = fit_relaxation_tau(x_proc, y_proc, mode=fit_mode, min_points=8)
        results.append(
            CycleFit(
                cycle_index=i,
                start_ts=float(x_cycle[0]),
                end_ts=float(x_cycle[-1]),
                tau=tau,
                x=x_cycle,
                y=y_cycle,
                y_processed=y_proc,
            )
        )
    return results


def analyze_pulse_relaxation(
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    *,
    laser_rate_hz: float,
    phase_offset_s: float = 0.0,
    min_points: int = 8,
    tune_tail: bool = True,
    trim_first: int = 0,
    linear_ratio: float = 0.8,
    fit_mode: str = "growth",
) -> list[PulseRelaxationFit]:
    """Fit one relaxation `tau` per laser pulse interval.

    This is the pulse-resolved version of the cycle-fitting workflow and is
    intended for traces like spot `raw_sum` or `raw_mean` where each pulse
    produces one exponential-like recovery curve.
    """

    pulse_traces = split_pulse_traces(
        sample_x,
        sample_y,
        laser_rate_hz=laser_rate_hz,
        phase_offset_s=phase_offset_s,
        min_points=min_points,
    )

    fits: list[PulseRelaxationFit] = []
    for pulse in pulse_traces:
        x_proc, y_proc = process_cycle_curve(
            pulse.x,
            pulse.y,
            tune_tail=tune_tail,
            trim_first=trim_first,
            linear_ratio=linear_ratio,
        )
        tau = fit_relaxation_tau(x_proc, y_proc, mode=fit_mode, min_points=min_points)
        fits.append(
            PulseRelaxationFit(
                pulse_index=pulse.pulse_index,
                start_ts=pulse.start_ts,
                end_ts=pulse.end_ts,
                tau=tau,
                x=pulse.x,
                y=pulse.y,
                y_processed=y_proc,
            )
        )
    return fits


def _infer_sample_rate_hz(x: np.ndarray) -> float | None:
    if x.size < 2:
        return None
    dt = float(np.median(np.diff(x)))
    if dt <= 0:
        return None
    return float(1.0 / dt)


def _fit_decay_tau(times: np.ndarray, amplitudes: np.ndarray) -> float | None:
    if times.size < 3 or amplitudes.size < 3 or times.size != amplitudes.size:
        return None

    baseline = float(np.min(amplitudes))
    signal = amplitudes - baseline
    keep = signal > max(1e-9, 0.05 * float(np.max(signal)))
    if np.count_nonzero(keep) < 3:
        return None

    xs = times[keep] - times[keep][0]
    ys = signal[keep]
    slope, _ = np.polyfit(xs, np.log(ys), 1)
    if slope >= 0:
        return None
    return float(-1.0 / slope)


__all__ = [
    "CycleFit",
    "CycleMetrics",
    "OscillationSummary",
    "PulseRelaxationFit",
    "PulseTrace",
    "analyze_pulse_relaxation",
    "analyze_rheed_signal",
    "compute_cycle_metrics",
    "detect_cycle_boundaries",
    "estimate_rolling_background",
    "load_trace_file",
    "preprocess_signal",
    "process_cycle_curve",
    "select_range",
    "split_pulse_traces",
    "split_cycles",
    "subtract_rolling_background",
    "summarize_oscillation_signal",
]

