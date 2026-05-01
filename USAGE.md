# RHEED-tools Usage Guide

`RHEED-tools` is responsible for reusable RHEED IMM/video I/O, ROI cropping,
spot detection/fitting/tracking, 2D diffraction metrics, 1D trace analysis,
dataset helpers, and RHEED-specific visualization.

## Install For Development

```bash
cd RHEED-tools
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

## Workflow: Inspect And Sample IMM Or Video Movies

Main entry points: `ImmMovie`, `inspect_movie_file`, `load_movie_frames`.
Helpers: `inspect_imm_file`, `load_imm_frames`, `load_imm_frame_raw`,
`load_imm_frame_headers`, `sample_frame_indices`, `frames_to_timestamps`,
`iter_movie_frames`, `load_video_frames`.

Use `ImmMovie` for k-Space IMM files and `load_movie_frames` for a uniform AVI/MP4/IMM
loader. Large files should be sampled with `every_n`, `start`, and `stop`.

```python
from rheed_tools.io import ImmMovie, load_movie_frames

movie = ImmMovie("YG070 RHEED.imm", fps=50.0)
print(movie.inspect())
frames, frame_indices = movie.load_frames(every_n=1000)
loaded = load_movie_frames("preview.avi", every_n=10, fps=50.0)
```

## Workflow: Crop ROI Data To HDF5 Or MP4

Main entry points: `crop_movie_to_h5`, `crop_movie_to_video`.
Helpers: `sanitize_roi`, `crop_frame`, `crop_frames`, `recenter_roi`,
`save_frames_h5`, `crop_and_save_h5`, `save_frames_video`, `crop_and_save_video`,
`save_image_stack`, `load_image_stack`, `save_image_sequence`, `export_video_frames`.

Use direct movie cropping for large raw files so the full movie is never held in
memory. ROI order is `(y0, y1, x0, x1)`.

```python
from rheed_tools.io import crop_movie_to_h5
from rheed_tools.analysis.roi import sanitize_roi

roi = sanitize_roi((492, 656), roi=(120, 260, 180, 340))
crop_movie_to_h5("YG070 RHEED.imm", "YG070 RHEED - ROI.h5", roi=roi, every_n=1000, fps=50.0)
```

## Workflow: Detect, Fit, Crop, And Track Spots

Main entry points: `detect_bright_spots`, `fit_gaussian_2d`,
`locate_fit_and_crop_spot`, `track_spot_regions_in_video`,
`analyze_spot_region_series`.
Helpers: `gaussian_2d`, `gaussian_function`, `estimate_gaussian_moments`,
`reconstruct_gaussian_patch`, `analyze_spot_candidates`,
`summarize_candidate_geometry`, `classify_growth_pattern_from_candidates`,
`extract_spot_patch`, `crop_patch_from_gaussian_fit`,
`extract_multiple_gaussian_spot_patches`, `locate_and_fit_spot`.

Use candidate detection on full frames, Gaussian fitting on local patches, and
fixed-region series analysis after choosing a stable ROI.

```python
from rheed_tools.analysis.spot_fit import detect_bright_spots, fit_gaussian_2d, analyze_spot_region_series
from rheed_tools.analysis.roi import crop_frame

candidates = detect_bright_spots(frames[0], min_distance=12)
patch = crop_frame(frames[0], roi=(100, 180, 200, 280))
fit = fit_gaussian_2d(patch)
series = analyze_spot_region_series(frames, roi=(100, 180, 200, 280))
```

## Workflow: Measure 2D Diffraction Features

Main entry points: `analyze_roi_frame`, `analyze_rheed_frame`,
`analyze_roi_frames`, `analyze_rheed_frames`, `track_diffraction_spot`,
`measure_spot_streak_geometry_series`, `analyze_specular_series`,
`analyze_diffuse_scattering_series`.
Helpers: morphology, geometry, reciprocal-space, reconstruction, Kikuchi, and
growth-mode functions.

Use ROI/frame analysis for intensity and drift traces. Use geometry/specular/
diffuse/reconstruction/Kikuchi modules for richer feature vectors. Use
`build_growth_feature_vector` and `classify_growth_mode_series` to combine module
outputs into heuristic growth labels.

```python
from rheed_tools.analysis import analyze_roi_frames, measure_spot_streak_geometry_series, analyze_specular_series

roi_metrics = analyze_roi_frames(frames, ts=timestamps, roi=(100, 180, 200, 280))
geometry = measure_spot_streak_geometry_series(frames, roi=(100, 180, 200, 280))
specular = analyze_specular_series(frames, roi=(100, 180, 200, 280), fit_every_n=10)
```

## Workflow: Analyze 1D Intensity Traces

Main entry points: `preprocess_signal`, `detect_cycle_boundaries`,
`analyze_rheed_signal`, `analyze_pulse_relaxation`.
Helpers: `select_range`, `split_cycles`, `split_pulse_traces`,
`compute_cycle_metrics`, `summarize_oscillation_signal`, `process_cycle_curve`,
background and signal helpers.

Use this path for exported intensity traces or ROI-derived `raw_sum`/`raw_mean`
curves. `preprocess_signal` returns `(x, y_processed)`, and cycle detection takes
the processed y-values.

```python
from rheed_tools.analysis.trace_1d import preprocess_signal, detect_cycle_boundaries, analyze_rheed_signal

x_clean, y_clean = preprocess_signal(ts, intensity, sample_rate_hz=50.0)
peaks = detect_cycle_boundaries(y_clean, camera_freq=50.0, laser_freq=1.0)
cycle_fits = analyze_rheed_signal(x_clean, y_clean, camera_freq=50.0, laser_freq=1.0)
```

## Workflow: Package RHEED Datasets

Main entry points: `RheedSpotDataset`, `RheedParameterDataset`,
`pack_image_sequence_to_h5`, `compress_h5_datasets`, `normalize_range`.
DataFed helpers: `create_collection`, `list_collection_items`, `upload_file`,
`download_file`, `update_record_metadata`.

Use dataset readers for archived HDF5 image/parameter files. Use DataFed helpers
only in environments with a configured DataFed client.

## Function Map

This compact map is for lookup after you know the workflow you need.

### `rheed_tools.analysis.background`
Functions: `estimate_rolling_background(values, window_size=21, percentile=20.0)`, `subtract_rolling_background(values, window_size=21, percentile=20.0, normalize=False)`

### `rheed_tools.analysis.curve_fitting`
Functions: `normalize_0_1(values, start=None, end=None, amplitude=None, *, unify=True)`, `denormalize_0_1(normalized_values, start, end, amplitude=None, *, unify=True)`, `exponential_growth(x, amplitude, offset, tau_s)`, `exponential_decay(x, amplitude, offset, tau_s)`, `fit_exponential_curve(x, y, *, mode='auto', initial=(1.0, 0.0, 0.4), bounds=((-np.inf, -np.inf, 1e-06), (np.inf, np.inf, np.inf)))`
Classes: `ExponentialFit`

### `rheed_tools.analysis.diffraction_2d`
Functions: `analyze_roi_frame(frame, roi=None, background_roi=None)`, `analyze_rheed_frame(frame, roi=None, background_roi=None)`, `analyze_roi_frames(frames, ts=None, roi=None, background_roi=None)`, `analyze_rheed_frames(frames, ts=None, roi=None, background_roi=None)`, `track_diffraction_spot(frames, ts=None, initial_roi=None, search_margin_px=8, background_roi=None)`
Classes: `RoiFrameMetrics`, `FrameMetrics`, `RoiSeriesMetrics`, `FrameSeriesMetrics`, `SpotTrack`

### `rheed_tools.analysis.diffuse`
Functions: `analyze_diffuse_scattering(frame, *, signal_roi=None, diffuse_roi=None, background_roi=None, halo_inner_scale=1.5, halo_outer_scale=3.0)`, `analyze_diffuse_scattering_series(frames, *, ts=None, signal_roi=None, diffuse_roi=None, background_roi=None, halo_inner_scale=1.5, halo_outer_scale=3.0)`
Classes: `DiffuseFrameMetrics`, `DiffuseSeriesMetrics`

### `rheed_tools.analysis.geometry`
Functions: `extract_axis_profile(frame, *, roi=None, background_roi=None, axis='x', reducer='mean')`, `measure_profile_spacing(profile, *, min_rel_height=0.2, min_distance_px=6)`, `measure_peak_spacing(frame, *, roi=None, background_roi=None, axis='x', reducer='mean', min_rel_height=0.2, min_distance_px=6)`, `measure_spot_streak_geometry(frame, *, roi=None, background_roi=None, width_fraction=0.2, spacing_axis='x', spacing_reducer='mean', spacing_min_rel_height=0.2, spacing_min_distance_px=6, split_prominence_rel=0.25)`, `measure_spot_streak_geometry_series(frames, *, ts=None, roi=None, background_roi=None, width_fraction=0.2, spacing_axis='x', spacing_reducer='mean', spacing_min_rel_height=0.2, spacing_min_distance_px=6, split_prominence_rel=0.25)`
Classes: `PeakSpacingMetrics`, `GeometryFrameMetrics`, `GeometrySeriesMetrics`

### `rheed_tools.analysis.growth_mode`
Functions: `build_growth_feature_vector(*, specular_metrics=None, geometry_metrics=None, diffuse_metrics=None, reciprocal_metrics=None, oscillation_amplitude=None, damping_tau_s=None)`, `classify_growth_mode(features, *, oscillation_threshold=0.12, diffuse_threshold=0.45, streakiness_threshold=0.22, tilt_threshold_deg=12.0)`, `classify_growth_mode_series(feature_vectors, *, ts=None, oscillation_threshold=0.12, diffuse_threshold=0.45, streakiness_threshold=0.22, tilt_threshold_deg=12.0)`, `detect_growth_transitions(labels)`
Classes: `GrowthFeatureVector`, `GrowthModeDecision`, `GrowthModeSeries`

### `rheed_tools.analysis.kikuchi`
Functions: `detect_kikuchi_lines(frame, *, roi=None, angle_step_deg=2.0, top_n=4, min_angle_separation_deg=12.0)`, `analyze_kikuchi_series(frames, *, ts=None, roi=None, angle_step_deg=2.0, top_n=4, min_angle_separation_deg=12.0)`
Classes: `KikuchiLine`, `KikuchiFrameMetrics`, `KikuchiSeriesMetrics`

### `rheed_tools.analysis.morphology`
Functions: `weighted_moments(img)`, `describe_shape(img)`
Classes: `ShapeMetrics`

### `rheed_tools.analysis.reciprocal`
Functions: `calibrate_reciprocal_space(*, reciprocal_per_pixel, reciprocal_unit='arb')`, `pixel_spacing_to_delta_k(spacing_px, calibration)`, `estimate_in_plane_lattice_constant(delta_k)`, `estimate_strain_percent(lattice_constant, *, reference_lattice_constant=None)`, `analyze_reciprocal_frame(frame, *, roi=None, background_roi=None, axis='x', reducer='mean', min_rel_height=0.2, min_distance_px=6, calibration=None, reference_lattice_constant=None)`, `analyze_reciprocal_series(frames, *, ts=None, roi=None, background_roi=None, axis='x', reducer='mean', min_rel_height=0.2, min_distance_px=6, calibration=None, reference_lattice_constant=None)`
Classes: `ReciprocalCalibration`, `ReciprocalFrameMetrics`, `ReciprocalSeriesMetrics`

### `rheed_tools.analysis.reconstruction`
Functions: `detect_fractional_order_peaks(profile, *, fundamental_spacing_px, center_index=None, fractional_orders=(0.5, 1.0 / 3.0, 2.0 / 3.0), tolerance_fraction=0.12, min_rel_height=0.12, min_distance_px=4)`, `analyze_surface_reconstruction(frame, *, roi=None, background_roi=None, axis='x', reducer='mean', expected_spacing_px=None, fractional_orders=(0.5, 1.0 / 3.0, 2.0 / 3.0), tolerance_fraction=0.12, min_rel_height=0.12, min_distance_px=4)`, `analyze_surface_reconstruction_series(frames, *, ts=None, roi=None, background_roi=None, axis='x', reducer='mean', expected_spacing_px=None, fractional_orders=(0.5, 1.0 / 3.0, 2.0 / 3.0), tolerance_fraction=0.12, min_rel_height=0.12, min_distance_px=4)`
Classes: `FractionalOrderPeak`, `ReconstructionFrameMetrics`, `ReconstructionSeriesMetrics`

### `rheed_tools.analysis.roi`
Functions: `sanitize_roi(shape, roi, fraction, corner='center')`, `crop_frame(frame, roi)`, `crop_frames(frames, roi)`, `recenter_roi(shape, center_y, center_x, box_height, box_width)`

### `rheed_tools.analysis.specular`
Functions: `analyze_specular_frame(frame, *, roi=None, background_roi=None, expected_center=None, patch_size=(31, 31), allow_rotation=False)`, `analyze_specular_series(frames, *, ts=None, roi=None, background_roi=None, expected_center=None, patch_size=(31, 31), allow_rotation=False, fit_every_n=1)`
Classes: `SpecularFrameMetrics`, `SpecularSeriesMetrics`

### `rheed_tools.analysis.spot_fit`
Functions: `gaussian_2d(x, y, amplitude, center_x, center_y, sigma_x, sigma_y, rotation_deg=0.0, background=0.0)`, `gaussian_function(amplitude, center_x, center_y, sigma_x, sigma_y, rotation_deg=0.0, background=0.0)`, `estimate_gaussian_moments(data)`, `fit_gaussian_2d(data, *, allow_rotation=True)`, `reconstruct_gaussian_patch(shape, fit)`, `detect_bright_spots(frame, *, search_roi=None, threshold_rel=0.35, min_distance_px=9, max_candidates=10, patch_size=(31, 31), expected_center=None, max_offset_px=None, sort_by='intensity')`, `analyze_spot_candidates(frame, candidates, *, patch_size=None, allow_rotation=True)`, `summarize_candidate_geometry(candidate_metrics, *, streak_aspect_ratio_center=1.6, discrete_aspect_ratio_ceiling=1.45, label_margin=0.22)`, `classify_growth_pattern_from_candidates(frame, candidates, *, patch_size=None, allow_rotation=True, streak_aspect_ratio_center=1.6, discrete_aspect_ratio_ceiling=1.45, label_margin=0.22)`, `extract_spot_patch(frame, center_x, center_y, *, patch_size=(31, 31))`, `crop_patch_from_gaussian_fit(frame, fit, *, amplitude_fraction=0.02, edge_padding_px=3, keep_reference_aspect=False, reference_patch_size=None)`, `locate_and_fit_spot(frame, *, search_roi=None, threshold_rel=0.35, min_distance_px=9, patch_size=(31, 31), allow_rotation=True)`, `locate_fit_and_crop_spot(frame, *, search_roi=None, threshold_rel=0.35, min_distance_px=9, patch_size=(31, 31), allow_rotation=True, amplitude_fraction=0.02, edge_padding_px=3, keep_reference_aspect=False, refit_cropped=True)`, `extract_multiple_gaussian_spot_patches(frame, *, search_roi=None, threshold_rel=0.35, min_distance_px=9, max_candidates=10, initial_patch_size=(31, 31), allow_rotation=True, amplitude_fraction=0.02, edge_padding_px=3, keep_reference_aspect=False, refit_cropped=True)`, `track_spot_regions_in_video(frames, *, ts=None, search_roi=None, threshold_rel=0.35, min_distance_px=9, max_candidates=10, initial_patch_size=(31, 31), allow_rotation=True, amplitude_fraction=0.02, edge_padding_px=3, keep_reference_aspect=False, refit_cropped=True, max_match_distance_px=18.0, union_padding_px=2)`, `analyze_spot_region_series(frames, roi, *, ts=None, allow_rotation=True)`
Classes: `SpotCandidate`, `GaussianFitResult` (fwhm_x, fwhm_y), `SpotCropResult`, `SpotRegionTrack`, `SpotRegionSeries`, `SpotCandidateMetrics`, `CandidatePatternSummary`

### `rheed_tools.analysis.trace_1d`
Functions: `select_range(data, start, end, y_col=1)`, `preprocess_signal(sample_x, sample_y, sample_rate_hz, median_kernel_size=5, fft_band=(0.05, 5.0), smooth_window=5)`, `detect_cycle_boundaries(sample_y, camera_freq, laser_freq, convolve_step=5, prominence=0.1)`, `split_cycles(sample_x, sample_y, peak_indices)`, `split_pulse_traces(sample_x, sample_y, *, laser_rate_hz, phase_offset_s=0.0, min_points=8)`, `compute_cycle_metrics(sample_x, sample_y, peak_indices)`, `summarize_oscillation_signal(sample_x, sample_y, peak_indices=None)`, `process_cycle_curve(x, y, tune_tail=True, trim_first=0, linear_ratio=0.8)`, `analyze_rheed_signal(sample_x, sample_y, camera_freq, laser_freq, convolve_step=5, prominence=0.1, tune_tail=True, trim_first=0, linear_ratio=0.8, fit_mode='auto')`, `analyze_pulse_relaxation(sample_x, sample_y, *, laser_rate_hz, phase_offset_s=0.0, min_points=8, tune_tail=True, trim_first=0, linear_ratio=0.8, fit_mode='growth')`
Classes: `CycleFit`, `CycleMetrics`, `OscillationSummary`, `PulseTrace`, `PulseRelaxationFit`

### `rheed_tools.analysis.visualization`
Functions: `plot_frame_with_crop(frame, roi, *, centroid_x=None, centroid_y=None, cmap='gray', figsize=(8.0, 3))`

### `rheed_tools.datasets.datafed`
Functions: `create_collection(collection_name, parent_id=None)`, `list_collection_items(collection_id, *, max_count=100)`, `upload_file(file_path, parent_id, metadata=None, *, wait=True)`, `download_file(file_id, output_dir, *, wait=True)`, `update_record_metadata(record_id, metadata)`

### `rheed_tools.datasets.hdf5`
Functions: `normalize_range(data, value_range=(0.0, 1.0))`, `pack_image_sequence_to_h5(h5_path, source_dir, dataset_names, *, output_names=None, image_suffixes=('.png', '.tif', '.tiff', '.jpg', '.jpeg'), compression='gzip', compression_opts=4)`, `compress_h5_datasets(input_path, output_path=None, *, compression='gzip', compression_opts=9)`
Classes: `RheedSpotDataset` (growth_names, growth_length, load_growth), `RheedParameterDataset` (growth_names, spot_names, metric_names, load_metric, load_curve, load_concatenated_curves)

### `rheed_tools.io.image_io`
Functions: `load_image_stack(path, array_name=None)`, `save_image_stack(path, frames)`, `save_image_sequence(output_dir, frames, *, prefix='frame', ext='.png')`, `save_frames_h5(path, frames, *, dataset_name='frames', timestamps=None, frame_indices=None, roi=None, metadata=None)`, `crop_and_save_h5(path, frames, roi, *, dataset_name='frames', timestamps=None, frame_indices=None, metadata=None)`, `crop_movie_to_h5(movie_path, output_path, roi, *, dataset_name='frames', every_n=1, start=0, stop=None, fps=None, as_gray=True, imm_duration_s=None, imm_frame_stride_bytes=646144, imm_header_bytes=640, imm_width=656, imm_height=492, imm_dtype='<u2', metadata=None)`

### `rheed_tools.io.imm_io`
Functions: `inspect_imm_file(path, *, frame_stride_bytes=646144, header_bytes=640, width=656, height=492, dtype='<u2', signature=b'KSA00F', duration_s=None)`, `load_imm_frames(path, *, every_n=1, start=0, stop=None, frame_stride_bytes=646144, header_bytes=640, width=656, height=492, dtype='<u2')`, `load_imm_frame_raw(path, frame_index=0, *, frame_stride_bytes=646144, header_bytes=640, width=656, height=492, dtype='<u2')`, `load_imm_frame_headers(path, frame_indices=None, *, frame_stride_bytes=646144, header_bytes=640)`
Classes: `ImmInfo`, `ImmMovie` (frame_count, shape, trailing_bytes, inspect, sample_frame_indices, timestamps, frame_index_from_time, frame_index_from_pulse_count, time_from_pulse_count, pulse_count_from_frame_index, time_from_frame_index, load_frame_raw, load_frame, load_frame_by_time, load_frame_by_pulse_count, load_frames, crop_to_h5, crop_to_video)

### `rheed_tools.io.trace_io`
Functions: `load_trace_file(path, time_col=0, intensity_col=1, delimiter=None, skiprows=0, comments='#', array_name=None, sample_rate_hz=None, dt=None)`

### `rheed_tools.io.video_io`
Functions: `sample_frame_indices(total_frames, every_n=1, start=0, stop=None)`, `frames_to_timestamps(frame_indices, fps)`, `inspect_movie_file(path, *, fps=None, imm_duration_s=None, imm_frame_stride_bytes=646144, imm_header_bytes=640, imm_width=656, imm_height=492, imm_dtype='<u2')`, `load_video_frames(path, every_n=1, start=0, stop=None, as_gray=True)`, `iter_movie_frames(path, *, every_n=1, start=0, stop=None, as_gray=True, imm_frame_stride_bytes=646144, imm_header_bytes=640, imm_width=656, imm_height=492, imm_dtype='<u2')`, `export_video_frames(video_path, output_dir, every_n=1, start=0, stop=None, prefix='frame')`, `save_frames_video(path, frames, *, fps=30.0, codec=None)`, `crop_and_save_video(path, frames, roi, *, fps=30.0, codec=None)`, `crop_movie_to_video(movie_path, output_path, roi, *, every_n=1, start=0, stop=None, fps=None, codec=None, as_gray=True, imm_duration_s=None, imm_frame_stride_bytes=646144, imm_header_bytes=640, imm_width=656, imm_height=492, imm_dtype='<u2')`, `load_movie_frames(path, *, every_n=1, start=0, stop=None, fps=None, as_gray=True, imm_duration_s=None, imm_frame_stride_bytes=646144, imm_header_bytes=640, imm_width=656, imm_height=492, imm_dtype='<u2')`
Classes: `MovieLoadResult`, `MovieInspection`

### `rheed_tools.notebook_utils`
Functions: `repo_data_path(*parts, start=None)`

### `rheed_tools.signals`
Functions: `moving_average(values, window)`, `median_filter_1d(values, kernel_size)`, `bandpass_filter_fft(values, low_cutoff, high_cutoff, sample_frequency)`, `detect_peaks_1d(values, min_distance, prominence=0.0)`, `detect_peaks_step_1d(values, min_distance, convolve_step=5, prominence=0.0, mode='same')`, `segment_cycles(ts, values, peak_indices)`, `normalize_0_1(y, i_start=None, i_end=None, i_diff=None, unify=True)`, `fit_relaxation_tau(x, y, mode='auto', min_points=8)`, `estimate_latest_cycle_tau(ts, values, min_distance, prominence, mode='auto', min_points=8)`, `trim_cycle_tail(y, ratio=0.1)`, `remove_linear_background(x, y, linear_ratio=0.8)`
Classes: `TauEstimate`
