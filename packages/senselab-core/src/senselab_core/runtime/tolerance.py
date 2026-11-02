"""Where the tolerance profile of one backend of one task lives."""

from importlib.resources.abc import Traversable
from pathlib import Path
from typing import TypeVar

Folder = TypeVar("Folder", Path, Traversable)


def tolerance_profile_path(folder: Folder, task: str, backend: str) -> Folder:
    """Return `<folder>/<task>/<backend>.yaml`: a backend name is only unique within its task."""
    return folder.joinpath(task, f"{backend}.yaml")
