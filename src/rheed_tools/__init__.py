"""Reusable RHEED analysis, visualization, and data I/O utilities.

`rheed_tools` is intentionally runtime-agnostic: it should not know about TSST,
live feedback policies, or instrument command submission. Keep those pieces in
the real-time analyzer package and use this package for base-level processing
that is useful in notebooks, offline review, model-building, and replay.
"""

__all__ = ["__version__"]
__version__ = "0.2.2"
