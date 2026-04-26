# RHEED-tools Usage Guide

`RHEED-tools` owns reusable RHEED file I/O, trace analysis, ROI metrics,
spot fitting, diffraction analysis, and RHEED-specific visualization. Generic
notebook layout helpers come from `sci-viz-utils`.

## Install For Development

```bash
git clone https://github.com/yig319/RHEED-tools.git
cd RHEED-tools
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

## Load Frames

```python
from rheed_tools.io import load_movie_frames

frames = load_movie_frames("growth.imm", every_n=120, fps=30.1)
```

## Analyze A Trace

```python
from rheed_tools.analysis.trace_1d import analyze_rheed_signal

result = analyze_rheed_signal("intensity_trace.txt")
print(result)
```

## Visualize A Region Of Interest

```python
from rheed_tools.analysis.visualization import plot_frame_with_crop

fig, axes = plot_frame_with_crop(
    frames[0],
    roi=(100, 160, 220, 300),
    centroid_x=260,
    centroid_y=130,
)
```

## What Belongs Here

Keep RHEED-specific file readers, diffraction analysis, trace/oscillation
logic, ROI overlays, and spot-fitting workflows in `RHEED-tools`. Use
`sci-viz-utils` only for generic grid, label, scale-bar, and image-display
foundations.
