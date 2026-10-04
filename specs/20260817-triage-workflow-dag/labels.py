"""The verified labels for this spec's reference recording, read from the spec's own artefact.

Measurement scripts in this directory import ``LABELS`` from here so that ground truth has one owner
inside senselab. Nothing here reaches outside this repository.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_DOC = json.loads((Path(__file__).parent / "ground-truth-2026-08-18.json").read_text())

RECORDING: str = _DOC["recording"]
SHA256: str = _DOC["sha256"]
LABELS: list[dict[str, Any]] = _DOC["labels"]
SCORED: list[dict[str, Any]] = [lab for lab in LABELS if lab["status"] == "scored"]


def label(name: str) -> dict[str, Any]:
    """Return the one label with this name.

    Args:
        name: The label's ``name`` field.

    Returns:
        The label.

    Raises:
        KeyError: If no label carries that name.
    """
    for lab in LABELS:
        if lab["name"] == name:
            return lab
    raise KeyError(f"no label named {name!r}; have {[lab['name'] for lab in LABELS]}")
