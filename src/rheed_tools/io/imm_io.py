from __future__ import annotations

"""Helpers for reading k-Space `.imm` movie files."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class ImmInfo:
    """Basic metadata inferred from a k-Space IMM movie."""

    path: Path
    frame_count: int
    height: int
    width: int
    dtype: str
    frame_stride_bytes: int
    header_bytes: int
    signature: str
    fps_estimate: float | None = None
    trailing_bytes: int = 0


class ImmMovie:
    """Object-oriented interface for working with one k-Space IMM movie.

    This is the notebook-friendly entry point when you want to:
    - inspect the file without loading the full movie
    - convert time or pulse count into an exact frame index
    - read one raw frame directly from disk
    - load a sampled stack for lightweight preview
    - export a cropped ROI to HDF5 or MP4 without holding the whole movie in memory
    """

    def __init__(
        self,
        path: str | Path,
        *,
        fps: float | None = None,
        duration_s: float | None = None,
        frame_stride_bytes: int = 646_144,
        header_bytes: int = 640,
        width: int = 656,
        height: int = 492,
        dtype: str = "<u2",
        signature: bytes = b"KSA00F",
    ) -> None:
        self.path = Path(path)
        self.frame_stride_bytes = int(frame_stride_bytes)
        self.header_bytes = int(header_bytes)
        self.width = int(width)
        self.height = int(height)
        self.dtype = np.dtype(dtype).str
        self.signature = bytes(signature)
        self.info = inspect_imm_file(
            self.path,
            frame_stride_bytes=self.frame_stride_bytes,
            header_bytes=self.header_bytes,
            width=self.width,
            height=self.height,
            dtype=self.dtype,
            signature=self.signature,
            duration_s=duration_s,
        )
        self.fps = None if fps is None else float(fps)
        if self.fps is None:
            self.fps = self.info.fps_estimate
        if duration_s is not None and duration_s > 0:
            self.duration_s = float(duration_s)
        elif self.fps is not None and self.fps > 0:
            self.duration_s = float(self.info.frame_count / self.fps)
        else:
            self.duration_s = None

    @property
    def frame_count(self) -> int:
        return int(self.info.frame_count)

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.info.height), int(self.info.width)

    @property
    def trailing_bytes(self) -> int:
        return int(self.info.trailing_bytes)

    def inspect(self) -> ImmInfo:
        """Return cached IMM metadata for this movie."""

        return self.info

    def sample_frame_indices(
        self,
        *,
        every_n: int = 1,
        start: int = 0,
        stop: int | None = None,
    ) -> np.ndarray:
        """Return sampled frame indices using frame-index units."""

        if every_n <= 0:
            raise ValueError("every_n must be > 0")
        if start < 0:
            raise ValueError("start must be >= 0")
        stop_idx = self.frame_count if stop is None else min(int(stop), self.frame_count)
        if stop_idx < start:
            stop_idx = start
        return np.arange(start, stop_idx, every_n, dtype=int)

    def timestamps(self, frame_indices: np.ndarray | list[int]) -> np.ndarray:
        """Convert frame indices into timestamps using the movie fps."""

        if self.fps is None or self.fps <= 0:
            raise ValueError("fps must be known to compute timestamps")
        indices = np.asarray(frame_indices, dtype=float)
        return indices / float(self.fps)

    def frame_index_from_time(self, time_s: float) -> int:
        """Convert a physical time in seconds into the nearest frame index."""

        if self.fps is None or self.fps <= 0:
            raise ValueError("fps must be known to map time to frame index")
        index = int(round(float(time_s) * float(self.fps)))
        return int(np.clip(index, 0, self.frame_count - 1))

    def frame_index_from_pulse_count(self, pulse_count: float, laser_rate_hz: float) -> int:
        """Convert a laser pulse count into the nearest frame index."""

        if laser_rate_hz <= 0:
            raise ValueError("laser_rate_hz must be > 0")
        time_s = float(pulse_count) / float(laser_rate_hz)
        return self.frame_index_from_time(time_s)

    def time_from_pulse_count(self, pulse_count: float, laser_rate_hz: float) -> float:
        """Convert pulse count into elapsed time in seconds."""

        if laser_rate_hz <= 0:
            raise ValueError("laser_rate_hz must be > 0")
        return float(pulse_count) / float(laser_rate_hz)

    def pulse_count_from_frame_index(self, frame_index: int, laser_rate_hz: float) -> float:
        """Estimate pulse count corresponding to one frame index."""

        if laser_rate_hz <= 0:
            raise ValueError("laser_rate_hz must be > 0")
        return float(self.time_from_frame_index(frame_index) * float(laser_rate_hz))

    def time_from_frame_index(self, frame_index: int) -> float:
        """Convert one frame index into elapsed time in seconds."""

        if self.fps is None or self.fps <= 0:
            raise ValueError("fps must be known to map frame index to time")
        return float(frame_index) / float(self.fps)

    def _read_payload(self, frame_index: int) -> np.ndarray:
        if frame_index < 0 or frame_index >= self.frame_count:
            raise ValueError(f"frame_index must be between 0 and {self.frame_count - 1}")
        pixel_dtype = np.dtype(self.dtype)
        payload_bytes = pixel_dtype.itemsize * self.width * self.height
        with self.path.open("rb") as fh:
            offset = frame_index * self.frame_stride_bytes + self.header_bytes
            fh.seek(offset)
            payload = fh.read(payload_bytes)
        if len(payload) != payload_bytes:
            raise ValueError(f"incomplete IMM frame payload at index {frame_index}")
        return np.frombuffer(payload, dtype=pixel_dtype).reshape(self.height, self.width)

    def load_frame_raw(self, frame_index: int = 0) -> np.ndarray:
        """Read one frame directly from disk without orientation changes."""

        return self._read_payload(int(frame_index))

    def load_frame(self, frame_index: int = 0, *, as_float: bool = True) -> np.ndarray:
        """Read one frame, optionally converted to float for analysis."""

        frame = self.load_frame_raw(frame_index)
        return np.asarray(frame, dtype=float) if as_float else frame

    def load_frame_by_time(self, time_s: float, *, as_float: bool = True) -> tuple[int, np.ndarray]:
        """Load the frame nearest to the requested time."""

        frame_index = self.frame_index_from_time(time_s)
        return frame_index, self.load_frame(frame_index, as_float=as_float)

    def load_frame_by_pulse_count(
        self,
        pulse_count: float,
        *,
        laser_rate_hz: float,
        as_float: bool = True,
    ) -> tuple[int, np.ndarray]:
        """Load the frame nearest to the requested pulse count."""

        frame_index = self.frame_index_from_pulse_count(pulse_count, laser_rate_hz)
        return frame_index, self.load_frame(frame_index, as_float=as_float)

    def load_frames(
        self,
        *,
        every_n: int = 1,
        start: int = 0,
        stop: int | None = None,
        as_float: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load a sampled frame stack into memory for lightweight preview."""

        frame_indices = self.sample_frame_indices(every_n=every_n, start=start, stop=stop)
        if frame_indices.size == 0:
            dtype = float if as_float else np.dtype(self.dtype)
            return np.empty((0, self.height, self.width), dtype=dtype), frame_indices
        frames = [self.load_frame(int(frame_idx), as_float=as_float) for frame_idx in frame_indices]
        return np.asarray(frames), frame_indices

    def crop_to_h5(
        self,
        output_path: str | Path,
        *,
        roi: tuple[int, int, int, int],
        every_n: int = 1,
        start: int = 0,
        stop: int | None = None,
        metadata: dict[str, object] | None = None,
    ) -> Path:
        """Crop the movie directly from disk and save the ROI stack to HDF5."""

        from .image_io import crop_movie_to_h5

        return crop_movie_to_h5(
            self.path,
            output_path,
            roi=roi,
            every_n=every_n,
            start=start,
            stop=stop,
            fps=self.fps,
            imm_frame_stride_bytes=self.frame_stride_bytes,
            imm_header_bytes=self.header_bytes,
            imm_width=self.width,
            imm_height=self.height,
            imm_dtype=self.dtype,
            metadata=metadata,
        )

    def crop_to_video(
        self,
        output_path: str | Path,
        *,
        roi: tuple[int, int, int, int],
        every_n: int = 1,
        start: int = 0,
        stop: int | None = None,
        codec: str | None = None,
    ) -> Path:
        """Crop the movie directly from disk and save the ROI stack to MP4/video."""

        from .video_io import crop_movie_to_video

        return crop_movie_to_video(
            self.path,
            output_path,
            roi=roi,
            every_n=every_n,
            start=start,
            stop=stop,
            fps=self.fps,
            codec=codec,
            imm_frame_stride_bytes=self.frame_stride_bytes,
            imm_header_bytes=self.header_bytes,
            imm_width=self.width,
            imm_height=self.height,
            imm_dtype=self.dtype,
        )


def inspect_imm_file(
    path: str | Path,
    *,
    frame_stride_bytes: int = 646_144,
    header_bytes: int = 640,
    width: int = 656,
    height: int = 492,
    dtype: str = "<u2",
    signature: bytes = b"KSA00F",
    duration_s: float | None = None,
) -> ImmInfo:
    """Inspect a k-Space IMM file using the known block layout."""

    file_path = Path(path)
    size_bytes = file_path.stat().st_size
    if size_bytes < header_bytes:
        raise ValueError("IMM file is smaller than one frame header")
    if frame_stride_bytes <= header_bytes:
        raise ValueError("frame_stride_bytes must exceed header_bytes")

    payload_bytes = frame_stride_bytes - header_bytes
    expected_payload = int(np.dtype(dtype).itemsize) * width * height
    if payload_bytes != expected_payload:
        raise ValueError(
            "IMM payload size does not match dtype * width * height: "
            f"{payload_bytes} != {expected_payload}"
        )
    trailing_bytes = int(size_bytes % frame_stride_bytes)
    frame_count = size_bytes // frame_stride_bytes
    if frame_count == 0:
        raise ValueError("IMM file does not contain one complete frame with the inferred frame stride")

    with file_path.open("rb") as fh:
        first_header = fh.read(header_bytes)
    if signature not in first_header:
        raise ValueError(f"IMM signature {signature!r} not found in the first frame header")

    fps_estimate = None if duration_s is None or duration_s <= 0 else float(frame_count / duration_s)
    return ImmInfo(
        path=file_path,
        frame_count=int(frame_count),
        height=int(height),
        width=int(width),
        dtype=np.dtype(dtype).str,
        frame_stride_bytes=int(frame_stride_bytes),
        header_bytes=int(header_bytes),
        signature=signature.decode("ascii", errors="replace"),
        fps_estimate=fps_estimate,
        trailing_bytes=trailing_bytes,
    )


def load_imm_frames(
    path: str | Path,
    *,
    every_n: int = 1,
    start: int = 0,
    stop: int | None = None,
    frame_stride_bytes: int = 646_144,
    header_bytes: int = 640,
    width: int = 656,
    height: int = 492,
    dtype: str = "<u2",
) -> tuple[np.ndarray, np.ndarray, ImmInfo]:
    """Load sampled image frames from a k-Space IMM movie."""

    movie = ImmMovie(
        path,
        frame_stride_bytes=frame_stride_bytes,
        header_bytes=header_bytes,
        width=width,
        height=height,
        dtype=dtype,
    )
    frames, frame_indices = movie.load_frames(every_n=every_n, start=start, stop=stop, as_float=True)
    return frames, frame_indices, movie.inspect()


def load_imm_frame_raw(
    path: str | Path,
    frame_index: int = 0,
    *,
    frame_stride_bytes: int = 646_144,
    header_bytes: int = 640,
    width: int = 656,
    height: int = 492,
    dtype: str = "<u2",
) -> np.ndarray:
    """Read one IMM frame with only header skip + reshape applied.

    This helper intentionally avoids float conversion, normalization, transpose,
    flips, or any orientation changes so the caller can inspect the raw camera
    layout directly.
    """

    movie = ImmMovie(
        path,
        frame_stride_bytes=frame_stride_bytes,
        header_bytes=header_bytes,
        width=width,
        height=height,
        dtype=dtype,
    )
    return movie.load_frame_raw(frame_index)


def load_imm_frame_headers(
    path: str | Path,
    frame_indices: np.ndarray | list[int] | None = None,
    *,
    frame_stride_bytes: int = 646_144,
    header_bytes: int = 640,
) -> dict[int, bytes]:
    """Read raw per-frame header blocks from a k-Space IMM movie."""

    file_path = Path(path)
    size_bytes = file_path.stat().st_size
    frame_count = size_bytes // frame_stride_bytes
    if frame_count == 0:
        raise ValueError("IMM file does not contain one complete frame with the inferred frame stride")

    if frame_indices is None:
        indices = np.arange(frame_count, dtype=int)
    else:
        indices = np.asarray(frame_indices, dtype=int)
    headers: dict[int, bytes] = {}
    with file_path.open("rb") as fh:
        for idx in indices:
            if idx < 0 or idx >= frame_count:
                raise ValueError(f"frame index out of range: {idx}")
            fh.seek(idx * frame_stride_bytes)
            headers[int(idx)] = fh.read(header_bytes)
    return headers

__all__ = ["ImmInfo", "ImmMovie", "inspect_imm_file", "load_imm_frame_headers", "load_imm_frame_raw", "load_imm_frames"]

