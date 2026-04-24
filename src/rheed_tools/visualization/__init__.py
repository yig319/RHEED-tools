"""Plotting helpers for RHEED notebooks and reports.

The functions in this subpackage are optional convenience utilities. They
import Matplotlib only inside plotting calls so the core package can still be
used in headless analysis environments.
"""

from .layout import (
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
