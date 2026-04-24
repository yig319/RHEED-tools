"""Small helpers for keeping demo notebooks concise and consistent."""

from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Return the repository root by walking upward until ``pyproject.toml`` is found."""
    path = (start or Path.cwd()).resolve()
    for candidate in [path, *path.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return path


def repo_data_path(*parts: str | Path, start: Path | None = None) -> Path:
    """Build a path inside the repository ``data/`` directory."""
    return find_repo_root(start=start) / "data" / Path(*parts)


__all__ = ["find_repo_root", "repo_data_path"]

