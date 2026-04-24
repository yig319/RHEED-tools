from __future__ import annotations

"""Notebook-friendly plotting helpers for diffraction-frame inspection."""

import numpy as np

from .roi import crop_frame


def plot_frame_with_crop(
    frame: np.ndarray,
    roi: tuple[int, int, int, int],
    *,
    centroid_x: float | None = None,
    centroid_y: float | None = None,
    cmap: str = "gray",
    figsize: tuple[float, float] = (8.0, 3),
):
    """Plot a full frame with ROI overlay next to the cropped ROI."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plot_frame_with_crop(). Install notebook/dev dependencies first."
        ) from exc

    arr = np.asarray(frame, dtype=float)
    crop = crop_frame(arr, roi)
    y0, y1, x0, x1 = roi

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].imshow(arr, cmap=cmap)
    axes[0].set_title("Frame with ROI")
    axes[0].add_patch(
        plt.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor="cyan",
            linewidth=1.5,
        )
    )
    if centroid_x is not None and centroid_y is not None:
        axes[0].plot(centroid_x, centroid_y, "ro", ms=5)
    axes[0].set_axis_off()

    axes[1].imshow(crop, cmap=cmap)
    axes[1].set_title("ROI crop")
    axes[1].set_axis_off()

    fig.tight_layout()
    return fig, axes


__all__ = ["plot_frame_with_crop"]

