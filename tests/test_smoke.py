import numpy as np

from rheed_tools.analysis.diffraction_2d import analyze_rheed_frame
from rheed_tools.analysis.curve_fitting import fit_exponential_curve, normalize_0_1
from rheed_tools.analysis.roi import crop_frame, sanitize_roi
from rheed_tools.analysis.spot_fit import fit_gaussian_2d, gaussian_2d
from rheed_tools.analysis.trace_1d import detect_cycle_boundaries, preprocess_signal
from rheed_tools.datasets import RheedParameterDataset, normalize_range
from rheed_tools.signals import fit_relaxation_tau
from rheed_tools.visualization import number_to_letters


def test_roi_and_frame_metrics_smoke() -> None:
    yy, xx = np.indices((48, 64), dtype=float)
    frame = 5.0 + gaussian_2d(
        xx,
        yy,
        amplitude=80.0,
        center_x=32.0,
        center_y=24.0,
        sigma_x=5.0,
        sigma_y=2.0,
        rotation_deg=10.0,
        background=0.0,
    )

    roi = sanitize_roi(frame.shape, (12, 36, 20, 44), fraction=0.5)
    cropped = crop_frame(frame, roi)
    metrics = analyze_rheed_frame(frame)
    fit = fit_gaussian_2d(cropped)

    assert cropped.shape == (24, 24)
    assert metrics.corrected_sum > 0
    assert fit.amplitude > 0


def test_trace_processing_smoke() -> None:
    sample_rate_hz = 50.0
    t = np.arange(0.0, 8.0, 1.0 / sample_rate_hz)
    y = 1.0 - np.exp(-np.mod(t, 1.0) / 0.2)

    _, y_proc = preprocess_signal(t, y, sample_rate_hz=sample_rate_hz, fft_band=(0.2, 8.0))
    peaks = detect_cycle_boundaries(y_proc, camera_freq=sample_rate_hz, laser_freq=1.0, prominence=0.02)
    tau = fit_relaxation_tau(t[:50], y[:50], mode="growth")

    assert peaks.size >= 5
    assert tau.tau_s is not None


def test_archived_curve_and_dataset_helpers_smoke(tmp_path) -> None:
    import h5py

    x = np.linspace(0.0, 2.0, 120)
    y = 0.2 + 1.4 * (1.0 - np.exp(-x / 0.35))
    fit = fit_exponential_curve(x, y, mode="growth")

    assert abs(fit.tau_s - 0.35) < 0.1
    assert np.allclose(normalize_range(np.array([2.0, 4.0])), np.array([0.0, 1.0]))
    assert normalize_0_1(y).shape == y.shape
    assert number_to_letters(27) == "ab"

    path = tmp_path / "params.h5"
    with h5py.File(path, "w") as h5:
        metric = h5.create_group("growth_a").create_group("spot_1").create_dataset("img_sum", data=y)

    dataset = RheedParameterDataset(path, camera_frequency_hz=10.0)
    t_loaded, y_loaded = dataset.load_curve("growth_a", "spot_1", "img_sum")

    assert np.allclose(y_loaded, y)
    assert np.allclose(t_loaded[:3], np.array([0.0, 0.1, 0.2]))
