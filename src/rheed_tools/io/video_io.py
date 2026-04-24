from __future__ import annotations

"""Video-to-frame helpers for offline RHEED workflows."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .imm_io import ImmInfo, ImmMovie, inspect_imm_file, load_imm_frame_raw, load_imm_frames
from rheed_tools.analysis.roi import crop_frames, sanitize_roi


@dataclass(slots=True)
class MovieLoadResult:
    """Unified result for movie loading across AVI and IMM sources."""

    frames: np.ndarray
    frame_indices: np.ndarray
    format: str
    fps: float | None
    metadata: ImmInfo | dict[str, object]


@dataclass(slots=True)
class MovieInspection:
    """Lightweight movie metadata preview without loading frame arrays."""

    path: Path
    format: str
    frame_count: int | None
    height: int | None
    width: int | None
    fps: float | None
    duration_s: float | None
    trailing_bytes: int | None
    metadata: ImmInfo | dict[str, object]


def _to_uint8_frames(frames: np.ndarray) -> np.ndarray:
    """Convert grayscale frames to uint8 for common video codecs."""

    arr = np.asarray(frames, dtype=float)
    if arr.ndim != 3:
        raise ValueError("frames must be a 3D array shaped (n_frames, height, width)")
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr - lo) / (hi - lo)
    return np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)


def _to_uint8_frame(frame: np.ndarray) -> np.ndarray:
    """Convert one grayscale frame to uint8 for streaming video output."""

    arr = np.asarray(frame)
    if arr.ndim != 2:
        raise ValueError("frame must be a 2D array")
    return _to_uint8_frames(arr[np.newaxis, :, :])[0]


def sample_frame_indices(
    total_frames: int,
    every_n: int = 1,
    start: int = 0,
    stop: int | None = None,
) -> np.ndarray:
    """Build frame indices for a sampled video export."""

    if total_frames < 0:
        raise ValueError("total_frames must be >= 0")
    if every_n <= 0:
        raise ValueError("every_n must be > 0")

    upper = total_frames if stop is None else min(stop, total_frames)
    return np.arange(start, upper, every_n, dtype=int)


def frames_to_timestamps(frame_indices: np.ndarray, fps: float) -> np.ndarray:
    """Convert frame indices into timestamps using a known frame rate."""

    if fps <= 0:
        raise ValueError("fps must be > 0")
    indices = np.asarray(frame_indices, dtype=float)
    return indices / float(fps)


def inspect_movie_file(
    path: str | Path,
    *,
    fps: float | None = None,
    imm_duration_s: float | None = None,
    imm_frame_stride_bytes: int = 646_144,
    imm_header_bytes: int = 640,
    imm_width: int = 656,
    imm_height: int = 492,
    imm_dtype: str = "<u2",
) -> MovieInspection:
    """Preview movie metadata without loading frames into memory.

    For `.imm` files this uses the known block layout and returns the full frame
    count immediately. For standard video files it performs a best-effort
    metadata read through `imageio` without decoding the whole movie.
    """

    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".imm":
        movie = ImmMovie(
            file_path,
            fps=fps,
            duration_s=imm_duration_s,
            frame_stride_bytes=imm_frame_stride_bytes,
            header_bytes=imm_header_bytes,
            width=imm_width,
            height=imm_height,
            dtype=imm_dtype,
        )
        info = movie.inspect()
        return MovieInspection(
            path=file_path,
            format="imm",
            frame_count=info.frame_count,
            height=info.height,
            width=info.width,
            fps=movie.fps,
            duration_s=movie.duration_s,
            trailing_bytes=info.trailing_bytes,
            metadata=info,
        )

    try:
        import imageio.v2 as iio
    except ImportError as exc:
        raise ImportError(
            "imageio is required for inspect_movie_file() on standard video files."
        ) from exc

    meta = dict(iio.immeta(file_path))
    size = meta.get("size")
    width = None
    height = None
    if isinstance(size, (tuple, list)) and len(size) >= 2:
        width = int(size[0])
        height = int(size[1])

    inferred_fps = fps
    if inferred_fps is None:
        meta_fps = meta.get("fps")
        if meta_fps is not None:
            inferred_fps = float(meta_fps)

    frame_count = None
    for key in ("nframes", "frame_count"):
        value = meta.get(key)
        if value not in (None, float("inf")):
            try:
                frame_count = int(value)
                break
            except (TypeError, ValueError):
                pass

    duration_s = None
    meta_duration = meta.get("duration")
    if meta_duration not in (None, float("inf")):
        try:
            duration_s = float(meta_duration)
        except (TypeError, ValueError):
            duration_s = None
    if duration_s is None and frame_count is not None and inferred_fps is not None and inferred_fps > 0:
        duration_s = float(frame_count / inferred_fps)

    return MovieInspection(
        path=file_path,
        format=suffix.lstrip(".") or "video",
        frame_count=frame_count,
        height=height,
        width=width,
        fps=inferred_fps,
        duration_s=duration_s,
        trailing_bytes=None,
        metadata=meta,
    )


def load_video_frames(
    path: str | Path,
    every_n: int = 1,
    start: int = 0,
    stop: int | None = None,
    as_gray: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load sampled frames from a video file with an optional imageio backend."""

    try:
        import imageio.v2 as iio
    except ImportError as exc:
        raise ImportError(
            "imageio is required for load_video_frames(). Install imageio to decode video files."
        ) from exc

    file_path = Path(path)
    frames = []
    indices: list[int] = []
    last_shape: tuple[int, int] | None = None

    iterator = iio.imiter(file_path)
    for idx, frame in enumerate(iterator):
        if idx < start:
            continue
        if stop is not None and idx >= stop:
            break
        if (idx - start) % every_n != 0:
            continue

        arr = np.asarray(frame, dtype=float)
        if arr.ndim == 3 and as_gray:
            arr = np.mean(arr, axis=2)
        elif arr.ndim != 2:
            raise ValueError("decoded frame must be 2D grayscale or 3D color")
        last_shape = arr.shape
        frames.append(arr)
        indices.append(idx)

    if not frames:
        if last_shape is None:
            try:
                first = np.asarray(iio.imread(file_path, index=0), dtype=float)
            except Exception:
                return np.empty((0, 0, 0), dtype=float), np.asarray(indices, dtype=int)
            if first.ndim == 3 and as_gray:
                first = np.mean(first, axis=2)
            last_shape = tuple(first.shape)
        return np.empty((0, *last_shape), dtype=float), np.asarray(indices, dtype=int)

    return np.asarray(frames, dtype=float), np.asarray(indices, dtype=int)


def iter_movie_frames(
    path: str | Path,
    *,
    every_n: int = 1,
    start: int = 0,
    stop: int | None = None,
    as_gray: bool = True,
    imm_frame_stride_bytes: int = 646_144,
    imm_header_bytes: int = 640,
    imm_width: int = 656,
    imm_height: int = 492,
    imm_dtype: str = "<u2",
):
    """Yield `(frame_index, frame)` pairs directly from a movie source.

    This is the low-memory path for large movies. It samples and decodes one
    frame at a time, which is useful for direct export to HDF5 or MP4 without
    materializing the full stack first.
    """

    if every_n <= 0:
        raise ValueError("every_n must be > 0")
    if start < 0:
        raise ValueError("start must be >= 0")

    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".imm":
        info = inspect_imm_file(
            file_path,
            frame_stride_bytes=imm_frame_stride_bytes,
            header_bytes=imm_header_bytes,
            width=imm_width,
            height=imm_height,
            dtype=imm_dtype,
        )
        stop_idx = info.frame_count if stop is None else min(int(stop), info.frame_count)
        if stop_idx < start:
            return
        for frame_idx in range(start, stop_idx, every_n):
            frame = load_imm_frame_raw(
                file_path,
                frame_index=frame_idx,
                frame_stride_bytes=imm_frame_stride_bytes,
                header_bytes=imm_header_bytes,
                width=imm_width,
                height=imm_height,
                dtype=imm_dtype,
            )
            yield int(frame_idx), np.asarray(frame, dtype=float)
        return

    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise ImportError(
            "imageio is required for iter_movie_frames() on standard video files."
        ) from exc

    iterator = iio.imiter(file_path)
    for idx, frame in enumerate(iterator):
        if idx < start:
            continue
        if stop is not None and idx >= stop:
            break
        if (idx - start) % every_n != 0:
            continue

        arr = np.asarray(frame, dtype=float)
        if arr.ndim == 3 and as_gray:
            arr = np.mean(arr, axis=2)
        elif arr.ndim != 2:
            raise ValueError("decoded frame must be 2D grayscale or 3D color")
        yield int(idx), arr


def export_video_frames(
    video_path: str | Path,
    output_dir: str | Path,
    every_n: int = 1,
    start: int = 0,
    stop: int | None = None,
    prefix: str = "frame",
) -> list[Path]:
    """Decode sampled video frames and save them as `.npy` arrays."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    frames, indices = load_video_frames(video_path, every_n=every_n, start=start, stop=stop)

    written: list[Path] = []
    for frame, idx in zip(frames, indices):
        path = output_path / f"{prefix}_{int(idx):05d}.npy"
        np.save(path, frame)
        written.append(path)
    return written


def save_frames_video(
    path: str | Path,
    frames: np.ndarray,
    *,
    fps: float = 30.0,
    codec: str | None = None,
) -> Path:
    """Save a grayscale frame stack as a video file.

    For odd-sized crops, the common MP4 default ``yuv420p`` path can fail because
    H.264 typically expects even width and height. In that case we switch to a
    more permissive pixel format so ROI exports do not break on tight crops.
    """

    try:
        import imageio.v2 as iio
    except ImportError as exc:
        raise ImportError(
            "imageio is required for save_frames_video(). Install notebook/dev dependencies first."
        ) from exc

    if fps <= 0:
        raise ValueError("fps must be > 0")
    video_frames = _to_uint8_frames(frames)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kwargs: dict[str, object] = {"fps": fps}
    kwargs["macro_block_size"] = 1
    height, width = video_frames.shape[1:]
    inferred_codec = codec
    if inferred_codec is None and output_path.suffix.lower() == ".mp4":
        inferred_codec = "libx264"
    if inferred_codec is not None:
        kwargs["codec"] = inferred_codec

    # Tight ROI crops often have odd dimensions; prefer yuv444p for MP4/H.264 in
    # that case so ffmpeg does not reject the export.
    if output_path.suffix.lower() == ".mp4" and (height % 2 != 0 or width % 2 != 0):
        kwargs["pixelformat"] = "yuv444p"

    with iio.get_writer(output_path, format="FFMPEG", **kwargs) as writer:
        for frame in video_frames:
            writer.append_data(frame)
    return output_path


def crop_and_save_video(
    path: str | Path,
    frames: np.ndarray,
    roi: tuple[int, int, int, int],
    *,
    fps: float = 30.0,
    codec: str | None = None,
) -> Path:
    """Crop a frame stack with one ROI and save the crop as a video."""

    cropped = crop_frames(frames, roi)
    return save_frames_video(path, cropped, fps=fps, codec=codec)


def crop_movie_to_video(
    movie_path: str | Path,
    output_path: str | Path,
    roi: tuple[int, int, int, int],
    *,
    every_n: int = 1,
    start: int = 0,
    stop: int | None = None,
    fps: float | None = None,
    codec: str | None = None,
    as_gray: bool = True,
    imm_duration_s: float | None = None,
    imm_frame_stride_bytes: int = 646_144,
    imm_header_bytes: int = 640,
    imm_width: int = 656,
    imm_height: int = 492,
    imm_dtype: str = "<u2",
) -> Path:
    """Crop a movie directly from the raw source file and save it as video.

    Unlike `crop_and_save_video()`, this function does not require an in-memory
    frame stack. It reads one sampled frame at a time from the source file.
    """

    try:
        import imageio.v2 as iio
    except ImportError as exc:
        raise ImportError(
            "imageio is required for crop_movie_to_video(). Install notebook/dev dependencies first."
        ) from exc

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
        raise ValueError("movie frame size could not be determined for crop_movie_to_video()")

    y0, y1, x0, x1 = sanitize_roi((info.height, info.width), roi, fraction=0.5)
    cropped_height = y1 - y0
    cropped_width = x1 - x0
    if cropped_height <= 0 or cropped_width <= 0:
        raise ValueError("roi produced an empty crop")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    writer_fps = fps if fps is not None else info.fps
    if writer_fps is None or writer_fps <= 0:
        writer_fps = 30.0

    kwargs: dict[str, object] = {"fps": writer_fps, "macro_block_size": 1}
    inferred_codec = codec
    if inferred_codec is None and output_file.suffix.lower() == ".mp4":
        inferred_codec = "libx264"
    if inferred_codec is not None:
        kwargs["codec"] = inferred_codec
    if output_file.suffix.lower() == ".mp4" and (cropped_height % 2 != 0 or cropped_width % 2 != 0):
        kwargs["pixelformat"] = "yuv444p"

    with iio.get_writer(output_file, format="FFMPEG", **kwargs) as writer:
        for _, frame in iter_movie_frames(
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
        ):
            writer.append_data(_to_uint8_frame(frame[y0:y1, x0:x1]))
    return output_file


def load_movie_frames(
    path: str | Path,
    *,
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
) -> MovieLoadResult:
    """Load sampled frames from a supported movie file and infer its format."""

    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".imm":
        movie = ImmMovie(
            file_path,
            fps=fps,
            duration_s=imm_duration_s,
            frame_stride_bytes=imm_frame_stride_bytes,
            header_bytes=imm_header_bytes,
            width=imm_width,
            height=imm_height,
            dtype=imm_dtype,
        )
        frames, frame_indices = movie.load_frames(every_n=every_n, start=start, stop=stop, as_float=True)
        info = movie.inspect()
        return MovieLoadResult(
            frames=frames,
            frame_indices=frame_indices,
            format="imm",
            fps=movie.fps,
            metadata=info,
        )

    frames, frame_indices = load_video_frames(
        file_path,
        every_n=every_n,
        start=start,
        stop=stop,
        as_gray=as_gray,
    )
    metadata: dict[str, object] = {
        "path": file_path,
        "as_gray": as_gray,
    }
    return MovieLoadResult(
        frames=frames,
        frame_indices=frame_indices,
        format=suffix.lstrip(".") or "video",
        fps=fps,
        metadata=metadata,
    )


__all__ = [
    "MovieInspection",
    "MovieLoadResult",
    "crop_movie_to_video",
    "export_video_frames",
    "frames_to_timestamps",
    "inspect_movie_file",
    "iter_movie_frames",
    "load_movie_frames",
    "load_video_frames",
    "sample_frame_indices",
]

