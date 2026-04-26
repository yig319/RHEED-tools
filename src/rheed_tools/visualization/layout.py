"""Compatibility aliases for generic notebook/report plotting helpers.

RHEED-specific frame/ROI plotting remains in :mod:`rheed_tools.analysis`.
Shared layout, image-grid, labels, and scale-bar primitives are owned by
:mod:`sci_viz_utils.figures`.
"""

from sci_viz_utils.figures import (
    add_scalebar,
    label_panel,
    make_figure_grid,
    number_to_letters,
    plot_image_map,
    set_axis_labels,
    show_image_grid,
    trim_axes,
)

__all__ = [
    "add_scalebar",
    "label_panel",
    "make_figure_grid",
    "number_to_letters",
    "plot_image_map",
    "set_axis_labels",
    "show_image_grid",
    "trim_axes",
]
