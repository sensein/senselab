"""Fail when a file tracked under ``packages/`` breaks the fixture budget.

The budget is read from the git index: no tracked file under ``packages/`` exceeds 256 KB, the data files (every
file that is not ``.py``, ``.pyi`` or ``.ipynb``) total at most 2 MB, and no tracked entry is a symlink or submodule.

Usage::

    uv run python scripts/check_fixture_budget.py
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[1]
BUDGETED = "packages/"
FILE_MAX_BYTES = 256 * 1000
DATA_TOTAL_MAX_BYTES = 2 * 1000 * 1000
CODE_SUFFIXES = (".py", ".pyi", ".ipynb")
REGULAR_MODES = frozenset({"100644", "100755"})
SYMLINK_MODE = "120000"
GITLINK_MODE = "160000"
_FIX = "generate it with senselab_core.testing.synth, or pull it through a data manifest"


class TrackedObject(NamedTuple):
    """One entry of the git index: its path, mode and size in bytes (0 for a submodule)."""

    path: str
    mode: str
    size: int


def indexed_objects(root: Path) -> list[TrackedObject]:
    """Every entry of the git index of ``root``, with the size of its blob."""
    listing = subprocess.run(["git", "ls-files", "-s", "-z"], cwd=root, capture_output=True, check=True)
    rows = []
    for line in listing.stdout.decode().split("\0"):
        if line:
            meta, _, path = line.partition("\t")
            mode, oid, _stage = meta.split()
            rows.append((path, mode, oid))
    blobs = [oid for _, mode, oid in rows if mode != GITLINK_MODE]
    stdin = "".join(f"{oid}\n" for oid in blobs).encode()
    sizes = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objectsize)"], cwd=root, capture_output=True, check=True, input=stdin
    ).stdout.split()
    by_oid = dict(zip(blobs, (int(s) for s in sizes), strict=True))
    return [TrackedObject(path, mode, by_oid.get(oid, 0)) for path, mode, oid in rows]


def check(objects: Sequence[TrackedObject]) -> list[str]:
    """Return one message per violation among ``objects``."""
    out: list[str] = []
    total = 0
    for o in objects:
        if o.mode == SYMLINK_MODE:
            out.append(f"{o.path}: tracked symlink; commit the file itself")
        elif o.mode == GITLINK_MODE:
            out.append(f"{o.path}: tracked submodule; pull its data through a manifest")
        elif o.mode not in REGULAR_MODES:
            out.append(f"{o.path}: tracked object of mode {o.mode}; only regular files may be committed")
        elif o.path.startswith(BUDGETED):
            if o.size > FILE_MAX_BYTES:
                out.append(f"{o.path}: {o.size} bytes exceeds the {FILE_MAX_BYTES} byte per-file limit; {_FIX}")
            if not o.path.endswith(CODE_SUFFIXES):
                total += o.size
    if total > DATA_TOTAL_MAX_BYTES:
        out.append(f"total of data files under {BUDGETED}: {total} bytes exceeds the {DATA_TOTAL_MAX_BYTES} byte limit; {_FIX}")
    return out


def main(root: Path = ROOT) -> int:
    """Run the check over the index of ``root``; return the exit status."""
    violations = check(indexed_objects(root))
    for v in violations:
        print(v)
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
