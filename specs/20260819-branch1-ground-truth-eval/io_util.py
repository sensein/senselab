"""Gzipped-JSON helpers, so the stored posterior matrices are small enough to keep in the repo."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any


def dump(obj: Any, path: Path) -> None:  # noqa: ANN401
    """Write ``obj`` as gzipped JSON."""
    with gzip.open(path, "wt") as handle:
        json.dump(obj, handle)


def load(path: Path) -> Any:  # noqa: ANN401
    """Read gzipped JSON, falling back to plain JSON at the same stem."""
    if path.exists():
        with gzip.open(path, "rt") as handle:
            return json.load(handle)
    plain = path.with_suffix("")
    with plain.open() as handle:
        return json.load(handle)
