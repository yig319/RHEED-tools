from __future__ import annotations

"""Notebook/report plotting helpers migrated from archived RHEED packages."""

from collections.abc import Sequence
from math import ceil

import numpy as np


def set_axis_labels(
    ax,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    yaxis_style: str | None = "sci",
    logscale: bool = False,
    legend: Sequence[str] | None = None,
    ticks_both_sides: bool = True,
) -> None:
    """Apply common axis labels, limits, and tick styling.

    This is a cleaned version of the old `set_labels` helpers. It intentionally
    accepts one Matplotlib axis and mutates it in place.
    """

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if yaxis_style == "sci":
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useLocale=False)
    if logscale:
        ax.set_yscale("log")
    if legend:
        ax.legend(legend)
    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    if ticks_both_sides:
        ax.yaxis.set_ticks_position("both")
        ax.xaxis.set_ticks_position("both")


def trim_axes(axs, n_axes: int):
    """Remove extra axes from a Matplotlib axes array and return the kept axes."""

    flat_axes = np.asarray(axs, dtype=object).reshape(-1)
    for ax in flat_axes[n_axes:]:
        ax.remove()
    return flat_axes[:n_axes]


def number_to_letters(number: int) -> str:
    """Convert zero-based panel number to letters: 0 -> a, 25 -> z, 26 -> aa."""

    if number < 0:
        raise ValueError("number must be non-negative")
    letters = ""
    value = int(number)
    while value >= 0:
        value, remainder = divmod(value, 26)
        letters = chr(97 + remainder) + letters
        value -= 1
    return letters


def make_figure_grid(
    n_plots: int,
    *,
    columns: int | None = None,
    figsize: tuple[float, float] | None = None,
    layout: str = "compressed",
):
    """Create a compact Matplotlib figure grid for a known number of panels."""

    if n_plots <= 0:
        raise ValueError("n_plots must be > 0")

    import matplotlib.pyplot as plt

    if columns is None:
        if n_plots < 3:
            columns = 2
        elif n_plots < 5:
            columns = 3
        elif n_plots < 10:
            columns = 4
        elif n_plots < 17:
            columns = 5
        elif n_plots < 26:
            columns = 6
        else:
            columns = 7

    rows = ceil(n_plots / columns)
    if figsize is None:
        figsize = (3.0 * columns, 3.0 * rows)

    fig, axs = plt.subplots(rows, columns, figsize=figsize, layout=layout)
    axes = trim_axes(axs, n_plots)
    return fig, axes


def show_image_grid(
    images: Sequence[np.ndarray],
    *,
    labels: Sequence[str] | None = None,
    images_per_row: int = 8,
    image_height: float = 1.0,
    show_colorbar: bool = False,
    clim_sigma: float | Sequence[float] = 3.0,
    scale_0_1: bool = False,
    hist_bins: int | None = None,
    show_axis: bool = False,
):
    """Display a list of images in a compact notebook grid."""

    if not images:
        raise ValueError("images must contain at least one image")

    import matplotlib.pyplot as plt

    arrays = [np.asarray(image) for image in images]
    labels = list(range(len(arrays))) if labels is None else labels
    panel_rows = 2 if hist_bins else 1
    rows = ceil(len(arrays) / images_per_row) * panel_rows
    height = max(1.0, arrays[0].shape[0] / max(arrays[0].shape[1], 1) * image_height + 1.0)
    fig, axs = plt.subplots(rows, images_per_row, figsize=(16, height * rows))
    axes = trim_axes(axs, len(arrays) * panel_rows)

    for idx, image in enumerate(arrays):
        image_to_show = _scale_image_0_1(image) if scale_0_1 else image
        image_ax = axes[idx * panel_rows]
        image_ax.set_title(str(labels[idx]))
        im = image_ax.imshow(image_to_show)
        if show_colorbar:
            sigma = clim_sigma[idx] if isinstance(clim_sigma, Sequence) else clim_sigma
            mean = float(np.mean(image_to_show))
            std = float(np.std(image_to_show))
            im.set_clim(mean - sigma * std, mean + sigma * std)
            fig.colorbar(im, ax=image_ax)
        if show_axis:
            image_ax.tick_params(axis="x", direction="in", top=True)
            image_ax.tick_params(axis="y", direction="in", right=True)
        else:
            image_ax.axis("off")
        if hist_bins:
            axes[idx * panel_rows + 1].hist(image_to_show.reshape(-1), bins=hist_bins)

    return fig, axes


def plot_image_map(
    ax,
    data: np.ndarray,
    *,
    colorbar: bool = True,
    clim: tuple[float, float] | None = None,
    cbar_number_format: str = "%.1e",
    cmap: str = "viridis",
):
    """Plot one 2D array with report-style axis cleanup and optional colorbar."""

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    arr = np.asarray(data)
    if arr.ndim == 1:
        side = int(np.sqrt(arr.size))
        if side * side != arr.size:
            raise ValueError("1D data can only be reshaped when its length is a square")
        arr = arr.reshape(side, side)
    if arr.ndim != 2:
        raise ValueError("data must be a 2D array or square-length 1D array")

    im = ax.imshow(arr, cmap=plt.get_cmap(cmap), clim=clim)
    ax.set_yticks([])
    ax.set_xticks([])
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="10%", pad=0.05)
        plt.colorbar(im, cax=cax, format=cbar_number_format)
    return im


def label_panel(
    ax,
    number: int | None = None,
    *,
    style: str = "wb",
    loc: str = "tl",
    prefix: str = "",
    size: float = 8,
    inset_fraction: tuple[float, float] = (0.15, 0.15),
    **kwargs,
):
    """Add a small panel label to a Matplotlib axis."""

    from matplotlib import patheffects

    formatting_key = {
        "wb": dict(color="w", linewidth=0.75),
        "b": dict(color="k", linewidth=0.0),
        "w": dict(color="w", linewidth=0.0),
    }
    if style not in formatting_key:
        raise ValueError("style must be one of 'wb', 'b', or 'w'")

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_inset = (xlim[1] - xlim[0]) * inset_fraction[1]
    y_inset = (ylim[1] - ylim[0]) * inset_fraction[0]
    locs = {
        "tl": (xlim[0] + x_inset, ylim[1] - y_inset),
        "tr": (xlim[1] - x_inset, ylim[1] - y_inset),
        "bl": (xlim[0] + x_inset, ylim[0] + y_inset),
        "br": (xlim[1] - x_inset, ylim[0] + y_inset),
        "ct": ((xlim[0] + xlim[1]) / 2, ylim[1] - y_inset),
        "cb": ((xlim[0] + xlim[1]) / 2, ylim[0] + y_inset),
    }
    if loc not in locs:
        raise ValueError("loc must be one of 'tl', 'tr', 'bl', 'br', 'ct', or 'cb'")

    text = prefix
    if number is not None:
        text += number_to_letters(number)
    formatting = formatting_key[style]
    artist = ax.text(
        *locs[loc],
        text,
        va="center",
        ha="center",
        path_effects=[patheffects.withStroke(linewidth=formatting["linewidth"], foreground="k")],
        color=formatting["color"],
        size=size,
        **kwargs,
    )
    artist.set_zorder(np.inf)
    return artist


def add_scalebar(
    ax,
    image_size: float,
    scale_size: float,
    *,
    units: str = "nm",
    loc: str = "br",
    color: str = "white",
    linewidth: float = 2.0,
):
    """Draw a simple scalebar on an image axis.

    `image_size` and `scale_size` must use the same physical units. The bar
    length is converted to axis pixels using the current x-axis span.
    """

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    width_px = abs(xlim[1] - xlim[0])
    bar_px = width_px * float(scale_size) / float(image_size)
    x_pad = 0.08 * width_px
    y_pad = 0.08 * abs(ylim[1] - ylim[0])

    if loc.endswith("r"):
        x0 = max(xlim) - x_pad - bar_px
        x1 = max(xlim) - x_pad
    else:
        x0 = min(xlim) + x_pad
        x1 = x0 + bar_px
    y = min(ylim) + y_pad if loc.startswith("b") else max(ylim) - y_pad

    ax.plot([x0, x1], [y, y], color=color, linewidth=linewidth)
    ax.text((x0 + x1) / 2, y, f"{scale_size:g} {units}", color=color, ha="center", va="bottom")


def _scale_image_0_1(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=float)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)
