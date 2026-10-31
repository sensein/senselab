"""Subprocess worker runtime: stdlib and numpy only."""

import functools
from pathlib import Path

from senselab_worker.digest import file_digest, identity_digest

WIRE_VERSION = 2


@functools.lru_cache(maxsize=1)
def runtime_identity() -> str:
    """Return `identity_digest("worker-runtime@1", files)` over this worker runtime's source.

    `files` is `[[relpath, file_digest("file@1", path)], ...]` over the package's `*.py` files, each named by its
    POSIX path relative to the package root and sorted by it.
    """
    root = Path(__file__).parent
    files = sorted((p.relative_to(root).as_posix(), p) for p in root.rglob("*.py"))
    return identity_digest("worker-runtime@1", [[rel, file_digest("file@1", path)] for rel, path in files])
