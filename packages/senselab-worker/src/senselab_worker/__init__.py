"""Subprocess worker runtime: stdlib and numpy only."""

import functools
from pathlib import Path

from senselab_worker.hashing import sha256_bytes

WIRE_VERSION = 2


@functools.lru_cache(maxsize=1)
def runtime_identity() -> str:
    """Return the hex sha256 identifying this worker runtime's source.

    Over the package's `*.py` files sorted by POSIX path relative to the package root, the digest
    covers, per file, the UTF-8 path, a NUL byte, the file bytes and a NUL byte.
    """
    root = Path(__file__).parent
    files = sorted((p.relative_to(root).as_posix(), p) for p in root.rglob("*.py"))
    return sha256_bytes(b"".join(rel.encode("utf-8") + b"\0" + path.read_bytes() + b"\0" for rel, path in files))
