from __future__ import annotations

"""Image-stack utilities for offline RHEED analysis."""

from pathlib import Path

import numpy as np

from rheed_tools.analysis.roi import crop_frame, crop_frames, sanitize_roi


def _prepare_frame_for_image_write(frame: np.ndarray, ext: str) -> np.ndarray:
    """Convert one frame into a dtype suitable for common image formats."""

    arr = np.asarray(frame)
    if np.issubdtype(arr.dtype, np.integer):
        return arr

    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo:
        if ext.lower() in {".tif", ".tiff", ".png"}:
            return np.zeros(arr.shape, dtype=np.uint16)
        return np.zeros(arr.shape, dtype=np.uint8)

    scaled = (arr - lo) / (hi - lo)
    if ext.lower() in {".tif", ".tiff", ".png"}:
        return np.clip(np.round(scaled * 65535.0), 0, 65535).astype(np.uint16)
    return np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)


def load_image_stack(path: str | Path, array_name: str | None = None) -> np.ndarray:
    """Load a saved image stack from `.npy` or `.npz`."""

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
        raise ValueError("load_image_stack currently supports only .npy and .npz")

    stack = np.asarray(arr, dtype=float)
    if stack.ndim == 2:
        return stack[np.newaxis, :, :]
    if stack.ndim != 3:
        raise ValueError("loaded image stack must be 2D or 3D")
    return stack


def save_image_stack(path: str | Path, frames: np.ndarray) -> None:
    """Save a frame stack to a NumPy file."""

    arr = np.asarray(frames)
    if arr.ndim != 3:
        raise ValueError("frames must be a 3D array shaped (n_frames, height, width)")
    np.save(Path(path), arr)


def save_image_sequence(
    output_dir: str | Path,
    frames: np.ndarray,
    *,
    prefix: str = "frame",
    ext: str = ".png",
) -> list[Path]:
    """Save a frame stack as sequential image files."""

    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise ImportError(
            "imageio is required for save_image_sequence(). Install notebook/dev dependencies first."
        ) from exc

    arr = np.asarray(frames)
    if arr.ndim != 3:
        raise ValueError("frames must be a 3D array shaped (n_frames, height, width)")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    suffix = ext if ext.startswith(".") else f".{ext}"

    written: list[Path] = []
    for idx, frame in enumerate(arr):
        path = output_path / f"{prefix}_{idx:05d}{suffix}"
        iio.imwrite(path, _prepare_frame_for_image_write(frame, suffix))
        written.append(path)
    return written


def save_frames_h5(
    path: str | Path,
    frames: np.ndarray,
    *,
    dataset_name: str = "frames",
    timestamps: np.ndarray | None = None,
    frame_indices: np.ndarray | None = None,
    roi: tuple[int, int, int, int] | None = None,
    metadata: dict[str, object] | None = None,
) -> Path:
    """Save a frame stack and optional metadata to an HDF5 file."""

    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required for save_frames_h5(). Install notebook/dev dependencies first."
        ) from exc

    arr = np.asarray(frames)
    if arr.ndim != 3:
        raise ValueError("frames must be a 3D array shaped (n_frames, height, width)")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5:
        h5.create_dataset(dataset_name, data=arr, compression="gzip")
        if timestamps is not None:
            ts = np.asarray(timestamps, dtype=float)
            if ts.shape != (arr.shape[0],):
                raise ValueError("timestamps must have one value per frame")
            h5.create_dataset("timestamps", data=ts)
        if frame_indices is not None:
            indices = np.asarray(frame_indices, dtype=int)
            if indices.shape != (arr.shape[0],):
                raise ValueError("frame_indices must have one value per frame")
            h5.create_dataset("frame_indices", data=indices)
        if roi is not None:
            h5.create_dataset("roi", data=np.asarray(roi, dtype=int))
        if metadata is not None:
            meta_group = h5.create_group("metadata")
            for key, value in metadata.items():
                if value is None:
                    continue
                meta_group.attrs[str(key)] = value
    return output_path


def crop_and_save_h5(
    path: str | Path,
    frames: np.ndarray,
    roi: tuple[int, int, int, int],
    *,
    dataset_name: str = "frames",
    timestamps: np.ndarray | None = None,
    frame_indices: np.ndarray | None = None,
    metadata: dict[str, object] | None = None,
) -> Path:
    """Crop a frame stack with one ROI and save it to HDF5."""

    cropped = crop_frames(frames, roi)
    return save_frames_h5(
        path,
        cropped,
        dataset_name=dataset_name,
        timestamps=timestamps,
        frame_indices=frame_indices,
        roi=roi,
        metadata=metadata,
    )


def crop_movie_to_h5(
    movie_path: str | Path,
    output_path: str | Path,
    roi: tuple[int, int, int, int],
    *,
    dataset_name: str = "frames",
    every_n: int = 1,
    start: int = 0,
    stop: int | None = None,
    fps: float | None = None,
    as_gray: bool = True,
    imm_duration_s: float | None = None,
    imm_frame_stride_bytes: int = 646_144,
    imm_header_bytes: int = 640,
    imm_width: int = 656,
    imm_height: int = 492,
    imm_dtype: str = "<u2",
    metadata: dict[str, object] | None = None,
) -> Path:
    """Crop a movie directly from the raw source file and save it to HDF5.

    Unlike `crop_and_save_h5()`, this function does not require an in-memory
    frame stack. It decodes or reads one sampled frame at a time.
    """

    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required for crop_movie_to_h5(). Install notebook/dev dependencies first."
        ) from exc

    from .video_io import frames_to_timestamps, inspect_movie_file, iter_movie_frames

    info = inspect_movie_file(
        movie_path,
        fps=fps,
        imm_duration_s=imm_duration_s,
        imm_frame_stride_bytes=imm_frame_stride_bytes,
        imm_header_bytes=imm_header_bytes,
        imm_width=imm_width,
        imm_height=imm_height,
        imm_dtype=imm_dtype,
    )
    if info.height is None or info.width is None:
        raise ValueError("movie frame size could not be determined for crop_movie_to_h5()")

    y0, y1, x0, x1 = sanitize_roi((info.height, info.width), roi, fraction=0.5)
    cropped_shape = (y1 - y0, x1 - x0)
    if cropped_shape[0] <= 0 or cropped_shape[1] <= 0:
        raise ValueError("roi produced an empty crop")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frame_indices_written: list[int] = []
    with h5py.File(out_path, "w") as h5:
        ds = None
        for out_idx, (frame_idx, frame) in enumerate(
            iter_movie_frames(
                movie_path,
                every_n=every_n,
                start=start,
                stop=stop,
                as_gray=as_gray,
                imm_frame_stride_bytes=imm_frame_stride_bytes,
                imm_header_bytes=imm_header_bytes,
                imm_width=imm_width,
                imm_height=imm_height,
                imm_dtype=imm_dtype,
            )
        ):
            cropped = np.asarray(frame[y0:y1, x0:x1])
            if ds is None:
                ds = h5.create_dataset(
                    dataset_name,
                    shape=(0, *cropped.shape),
                    maxshape=(None, *cropped.shape),
                    chunks=(1, *cropped.shape),
                    dtype=cropped.dtype,
                    compression="gzip",
                )
            ds.resize(out_idx + 1, axis=0)
            ds[out_idx] = cropped
            frame_indices_written.append(int(frame_idx))

        if ds is None:
            h5.create_dataset(
                dataset_name,
                shape=(0, *cropped_shape),
                maxshape=(None, *cropped_shape),
                chunks=(1, *cropped_shape),
                dtype=float,
                compression="gzip",
            )

        h5.create_dataset("roi", data=np.asarray((y0, y1, x0, x1), dtype=int))
        if frame_indices_written:
            index_arr = np.asarray(frame_indices_written, dtype=int)
            h5.create_dataset("frame_indices", data=index_arr)
            effective_fps = fps if fps is not None else info.fps
            if effective_fps is not None and effective_fps > 0:
                h5.create_dataset("timestamps", data=frames_to_timestamps(index_arr, fps=effective_fps))

        meta_group = h5.create_group("metadata")
        meta_group.attrs["source_path"] = str(Path(movie_path))
        meta_group.attrs["source_format"] = str(info.format)
        meta_group.attrs["sample_every_n"] = int(every_n)
        meta_group.attrs["start_frame"] = int(start)
        meta_group.attrs["stop_frame"] = -1 if stop is None else int(stop)
        if info.fps is not None:
            meta_group.attrs["source_fps"] = float(info.fps)
        if info.frame_count is not None:
            meta_group.attrs["source_frame_count"] = int(info.frame_count)
        if info.trailing_bytes is not None:
            meta_group.attrs["trailing_bytes"] = int(info.trailing_bytes)
        if metadata is not None:
            for key, value in metadata.items():
                if value is None:
                    continue
                meta_group.attrs[str(key)] = value

    return out_path


__all__ = [
    "crop_and_save_h5",
    "crop_movie_to_h5",
    "crop_frame",
    "crop_frames",
    "load_image_stack",
    "save_frames_h5",
    "save_image_sequence",
    "save_image_stack",
]

