"""Small helpers for keeping demo notebooks concise and consistent."""

from __future__ import annotations

from pathlib import Path

from sci_viz_utils.paths import find_repo_root


def repo_data_path(*parts: str | Path, start: Path | None = None) -> Path:
    """Build a path inside the repository ``data/`` directory."""
    return find_repo_root(start=start) / "data" / Path(*parts)


__all__ = ["find_repo_root", "repo_data_path"]

