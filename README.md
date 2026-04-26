# RHEED-tools

Reusable Python utilities for RHEED data analysis, visualization, and file I/O.

This package is the base utility layer separated from the real-time analyzer.
It is meant to stay independent of instrument control, TSST adapters, feedback
policies, and live-loop orchestration.

## Package Layout

```text
src/rheed_tools/
  analysis/        # ROI, spot fitting, diffraction metrics, trace/curve analysis
  datasets/        # HDF5 packing/readers and optional DataFed wrappers
  io/              # IMM/video/image/trace loading and export helpers
  visualization/   # notebook/report plotting helpers from archived packages
  signals.py       # reusable 1D filtering, peak, cycle, and tau-fit utilities
  notebook_utils.py
tests/
docs/
```

## Install For Development

```bash
pip install -e ".[dev]"
```

## Quick Examples

```python
from rheed_tools.analysis.trace_1d import analyze_rheed_signal
from rheed_tools.datasets import RheedParameterDataset
from rheed_tools.io import load_movie_frames

movie = load_movie_frames("example.imm", every_n=120, fps=30.1)
```

See [`USAGE.md`](USAGE.md) for a practical guide to loading frames, analyzing
traces, visualizing ROIs, and the boundary between RHEED-tools and
`sci-viz-utils`.

## Migration Sources

This package consolidates reusable analysis and notebook utilities from:

- `RHEED_RealTimeAnalyzer` analysis/IO modules
- archived `RHEED-Learn` curve fitting, signal processing, and visualization helpers
- archived `RHEED_data_collect` HDF5 packing, dataset access, DataFed wrappers, and figure layout helpers

Runtime feedback/control, TSST adapters, and policy logic stay in the
real-time analyzer package.

## Publishing

The GitHub Actions workflow in `.github/workflows/main.yml` mirrors the release
flow used by the real-time analyzer:

- `#major` bumps `X.0.0`
- `#minor` bumps `x.Y.0`
- `#patch` bumps `x.y.Z`

Configure PyPI Trusted Publishing for:

- Repository: `yig319/RHEED-tools`
- Workflow: `main.yml`
- Environment name: any / unset
