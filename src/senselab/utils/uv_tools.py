"""Shared uv and lock primitives: finding ``uv``, PEP 503 names, the requirements digest, the torch family.

Every hashed-lock format in senselab builds on these. The header formats themselves stay with their
owners (:mod:`senselab.utils.venv_lock` for the backend venv locks); only the pieces they have in
common live here. Standard library only, so the module imports in a lean environment.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Collection, Iterable, Mapping

logger = logging.getLogger("senselab")

UV_ENV = "SENSELAB_UV"
"""Names an absolute path to the ``uv`` executable to use instead of searching for one."""

TORCH_NAMES = frozenset({"torch", "torchvision", "torchaudio"})
"""The PyTorch distributions served from a PyTorch index."""

TORCH_VARIANT_PREFIXES = ("nvidia-", "triton", "pytorch-triton", "cuda-")
"""Prefixes of the CUDA-variant distributions that PyTorch wheels pull in."""


def _home_locations() -> list[Path]:
    return [Path.home() / ".local" / "bin" / "uv", Path.home() / ".cargo" / "bin" / "uv"]


def find_uv(*, install: bool = False) -> str:
    """Return the path of the ``uv`` executable.

    The search order is the ``SENSELAB_UV`` override, then ``PATH``, then the standard install
    locations under the home directory.

    Args:
        install: When nothing is found, ``pip install uv`` and search ``PATH`` again (for hosts
            such as notebooks where uv is not pre-installed).

    Returns:
        The path of ``uv``.

    Raises:
        FileNotFoundError: If ``SENSELAB_UV`` is set but is not an absolute path to an executable
            file, or no ``uv`` is found (after the install, when requested).
    """
    pinned = os.environ.get(UV_ENV)
    if pinned:
        if not os.path.isabs(pinned) or not os.path.isfile(pinned) or not os.access(pinned, os.X_OK):
            raise FileNotFoundError(f"{UV_ENV}={pinned!r} is not an absolute path to an executable")
        return pinned
    found = shutil.which("uv")
    if found:
        return found
    for candidate in _home_locations():
        if candidate.is_file():
            return str(candidate)
    if install:
        logger.info("uv not found — installing automatically...")
        result = subprocess.run(["pip", "install", "uv"], capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            found = shutil.which("uv")
            if found:
                return found
        raise FileNotFoundError("uv not found and auto-install failed. Install with: pip install uv")
    raise FileNotFoundError("uv not found on PATH or in the standard install locations")


def normalize_name(name: str) -> str:
    """Return a distribution name normalized per PEP 503."""
    return re.sub(r"[-_.]+", "-", name).lower()


def requirements_sha256(lines: Iterable[str], *, header_fields: Mapping[str, object]) -> str:
    """Digest of everything a lock is compiled from.

    Args:
        lines: The requirement lines; their order does not matter.
        header_fields: The other compile inputs a lock format records (JSON-serializable values),
            e.g. the Python version and the CUDA ceiling.

    Returns:
        A hex SHA-256 of ``{"requirements": sorted(lines), **header_fields}`` as sorted-key JSON.

    Raises:
        ValueError: If ``header_fields`` has a ``requirements`` key.
    """
    if "requirements" in header_fields:
        raise ValueError("header_fields must not carry a 'requirements' key; pass the lines instead")
    payload = {"requirements": sorted(lines), **header_fields}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def is_torch_family(name: str, *, roots: Collection[str] = TORCH_NAMES) -> bool:
    """Whether a distribution is a PyTorch root or one of the CUDA-variant packages PyTorch pulls in.

    Args:
        name: A distribution name, in any spelling.
        roots: The PyTorch distributions the caller serves from a PyTorch index.

    Returns:
        Whether the normalized name is in ``roots`` or starts with a CUDA-variant prefix.
    """
    name = normalize_name(name)
    return name in roots or name.startswith(TORCH_VARIANT_PREFIXES)
