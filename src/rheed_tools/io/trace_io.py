from __future__ import annotations

"""File-loading utilities for offline 1D trace analysis."""

from pathlib import Path

import numpy as np


def load_trace_file(
    path: str | Path,
    time_col: int = 0,
    intensity_col: int = 1,
    delimiter: str | None = None,
    skiprows: int = 0,
    comments: str | None = "#",
    array_name: str | None = None,
    sample_rate_hz: float | None = None,
    dt: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a RHEED trace from a text or NumPy file."""

    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(file_path)
    elif suffix == ".npz":
        archive = np.load(file_path)
        key = array_name or next(iter(archive.files), None)
        if key is None:
            raise ValueError("npz file does not contain any arrays")
        arr = archive[key]
    else:
        inferred_delimiter = "," if delimiter is None and suffix == ".csv" else delimiter
        arr = np.loadtxt(
            file_path,
            delimiter=inferred_delimiter,
            skiprows=skiprows,
            comments=comments,
            dtype=float,
        )

    data = np.asarray(arr, dtype=float)
    if data.ndim == 2 and 1 in data.shape:
        data = data.reshape(-1)

    if data.ndim == 1:
        y = data.astype(float)
        if dt is not None and dt <= 0:
            raise ValueError("dt must be > 0 when provided")
        if sample_rate_hz is not None and sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be > 0 when provided")
        sample_dt = dt if dt is not None else (1.0 / sample_rate_hz if sample_rate_hz is not None else 1.0)
        t = np.arange(y.size, dtype=float) * sample_dt
        return t, y

    if data.ndim != 2:
        raise ValueError("loaded trace must be 1D or 2D")

    max_col = max(time_col, intensity_col)
    if data.shape[0] <= max_col + 2 and data.shape[0] < data.shape[1]:
        data = data.T
    if data.shape[1] <= max_col and data.shape[0] > max_col:
        data = data.T
    if data.shape[1] <= max_col:
        raise ValueError("loaded trace does not contain the requested time/intensity columns")

    return data[:, time_col], data[:, intensity_col]

