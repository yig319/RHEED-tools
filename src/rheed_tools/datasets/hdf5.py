from __future__ import annotations

"""HDF5 dataset helpers migrated from RHEED-Learn/RHEED_data_collect.

The archived packages mixed dataset access, visualization, and DataFed upload
side effects in one module. This file keeps the base HDF5 operations small and
side-effect-light so they are safe to import from notebooks and scripts.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


def normalize_range(data: np.ndarray, value_range: tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """Scale numeric data into a target range.

    Constant arrays are mapped to the lower bound to avoid divide-by-zero NaNs.
    """

    arr = np.asarray(data, dtype=float)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    out_lo, out_hi = map(float, value_range)
    if hi <= lo:
        return np.full(arr.shape, out_lo, dtype=float)
    return ((arr - lo) * (out_hi - out_lo) / (hi - lo)) + out_lo


def pack_image_sequence_to_h5(
    h5_path: str | Path,
    source_dir: str | Path,
    dataset_names: Sequence[str],
    *,
    output_names: Sequence[str] | None = None,
    image_suffixes: Sequence[str] = (".png", ".tif", ".tiff", ".jpg", ".jpeg"),
    compression: str | None = "gzip",
    compression_opts: int | None = 4,
) -> Path:
    """Pack image sequences named by prefix into an HDF5 file.

    For each name in `dataset_names`, files matching `<name>*<suffix>` are read
    in sorted order and written as one HDF5 dataset. This generalizes the
    archived `pack_rheed_data` helper without assuming one exact file extension.
    """

    import imageio.v3 as iio
    import h5py

    source_path = Path(source_dir)
    output_path = Path(h5_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_names = dataset_names if output_names is None else output_names
    if len(output_names) != len(dataset_names):
        raise ValueError("output_names must match dataset_names length")

    with h5py.File(output_path, mode="a") as h5:
        for input_name, output_name in zip(dataset_names, output_names):
            files = _matching_image_files(source_path, input_name, image_suffixes)
            if not files:
                raise FileNotFoundError(f"No image files found for prefix {input_name!r}")

            first = np.asarray(iio.imread(files[0]))
            dataset = h5.create_dataset(
                str(output_name),
                shape=(len(files), *first.shape),
                dtype=first.dtype,
                compression=compression,
                compression_opts=compression_opts,
            )
            dataset[0] = first
            for idx, file_path in enumerate(files[1:], start=1):
                dataset[idx] = np.asarray(iio.imread(file_path))
    return output_path


def compress_h5_datasets(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    compression: str = "gzip",
    compression_opts: int = 9,
) -> Path:
    """Copy all HDF5 datasets/groups into a compressed output file."""

    import h5py

    src_path = Path(input_path)
    dst_path = Path(output_path) if output_path is not None else src_path.with_name(f"{src_path.stem}_compressed.h5")
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        _copy_h5_node(src, dst, compression=compression, compression_opts=compression_opts)
    return dst_path


class RheedSpotDataset:
    """Reader for HDF5 files that store one image stack per growth."""

    def __init__(self, path: str | Path, sample_name: str | None = None):
        self.path = Path(path)
        self.sample_name = sample_name

    def growth_names(self) -> list[str]:
        """Return top-level growth names in the HDF5 file."""

        import h5py

        with h5py.File(self.path, "r") as h5:
            return list(h5.keys())

    def growth_length(self, growth: str) -> int:
        """Return number of frames/items in one growth dataset."""

        import h5py

        with h5py.File(self.path, "r") as h5:
            return int(h5[growth].shape[0])

    def load_growth(self, growth: str, index: int | Sequence[int] | slice | None = None) -> np.ndarray:
        """Load all or part of one growth image stack."""

        import h5py

        with h5py.File(self.path, "r") as h5:
            dataset = h5[growth]
            if index is None:
                return np.asarray(dataset)
            return np.asarray(dataset[index])


class RheedParameterDataset:
    """Reader for growth/spot/metric HDF5 parameter files."""

    def __init__(self, path: str | Path, camera_frequency_hz: float, sample_name: str | None = None):
        if camera_frequency_hz <= 0:
            raise ValueError("camera_frequency_hz must be > 0")
        self.path = Path(path)
        self.camera_frequency_hz = float(camera_frequency_hz)
        self.sample_name = sample_name

    def growth_names(self) -> list[str]:
        """Return available growth names."""

        import h5py

        with h5py.File(self.path, "r") as h5:
            return list(h5.keys())

    def spot_names(self, growth: str) -> list[str]:
        """Return spot names inside one growth."""

        import h5py

        with h5py.File(self.path, "r") as h5:
            return list(h5[growth].keys())

    def metric_names(self, growth: str, spot: str) -> list[str]:
        """Return metric names for one growth/spot pair."""

        import h5py

        with h5py.File(self.path, "r") as h5:
            return list(h5[growth][spot].keys())

    def load_metric(
        self,
        growth: str,
        spot: str,
        metric: str,
        index: int | Sequence[int] | slice | None = None,
    ) -> np.ndarray:
        """Load one metric array from a growth/spot group."""

        import h5py

        with h5py.File(self.path, "r") as h5:
            dataset = h5[growth][spot][metric]
            if index is None:
                return np.asarray(dataset)
            return np.asarray(dataset[index])

    def load_curve(self, growth: str, spot: str, metric: str, *, start_frame: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Load one metric and return `(time_s, values)`."""

        values = np.asarray(self.load_metric(growth, spot, metric), dtype=float)
        frames = np.arange(start_frame, start_frame + values.shape[0], dtype=float)
        return frames / self.camera_frequency_hz, values

    def load_concatenated_curves(
        self,
        growths: Sequence[str],
        spot: str,
        metric: str,
        *,
        start_frame: int = 0,
        trim: tuple[int, int] = (100, 100),
        frame_gap: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load multiple growth curves and concatenate them with optional gaps."""

        x_all: list[np.ndarray] = []
        y_all: list[np.ndarray] = []
        cursor = int(start_frame)
        head, tail = trim
        for growth in growths:
            x, y = self.load_curve(growth, spot, metric, start_frame=cursor)
            end = None if tail <= 0 else -tail
            x = x[head:end]
            y = y[head:end]
            cursor += int(y.size + frame_gap)
            x_all.append(x)
            y_all.append(y)
        if not x_all:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        return np.concatenate(x_all), np.concatenate(y_all)


def _matching_image_files(source_dir: Path, prefix: str, suffixes: Sequence[str]) -> list[Path]:
    files: list[Path] = []
    for suffix in suffixes:
        files.extend(source_dir.glob(f"{prefix}*{suffix}"))
    return sorted(set(files))


def _copy_h5_node(src_node, dst_node, *, compression: str, compression_opts: int) -> None:
    import h5py

    for key, value in src_node.items():
        if isinstance(value, h5py.Dataset):
            dst_node.create_dataset(key, data=value[...], compression=compression, compression_opts=compression_opts)
            for attr_key, attr_value in value.attrs.items():
                dst_node[key].attrs[attr_key] = attr_value
        else:
            group = dst_node.create_group(key)
            for attr_key, attr_value in value.attrs.items():
                group.attrs[attr_key] = attr_value
            _copy_h5_node(value, group, compression=compression, compression_opts=compression_opts)
