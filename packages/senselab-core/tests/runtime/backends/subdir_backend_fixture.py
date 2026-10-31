"""A backend that makes a directory in the array directory on every request.

The directory is named `init["subdir"]` (default: an array file name) and holds a file and a symlink
to `init["victim"]` when one is given.
"""

import os
from pathlib import Path
from typing import Any

from backends import modes_fixture as _modes

ARRAY_LIKE = "f" * 64 + ".npy"


def load(init: dict[str, Any]) -> object:
    """Return `init` as the state."""
    return init


def handle(state: Any, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:  # noqa: ANN401
    """Make the directory, then act out the mode."""
    sub = Path(state["array_dir"]) / state.get("subdir", ARRAY_LIKE)
    sub.mkdir(exist_ok=True)
    (sub / "scratch.bin").write_bytes(b"x")
    if "victim" in state and not (sub / "victim").is_symlink():
        os.symlink(state["victim"], sub / "victim")
    _modes.apply(state, params)
    return [{"outputs": {"made": sub.name}} for _ in items]
