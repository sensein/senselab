"""Content identities of bytes, files and weight sets (ADR-002, P-4)."""

import os
from pathlib import PurePath
from typing import Iterator, Sequence, TypedDict

from senselab_worker import _raw_digest
from senselab_worker.canonical import canonical_text, utf16_sort_key
from senselab_worker.digest import file_digest, identity_digest

__all__ = ["DEFAULT_CHUNK_SIZE", "ContentId", "content_identity", "weights_identity"]

DEFAULT_CHUNK_SIZE = 64 * 2**20


class ContentId(TypedDict):
    """Chunked identity of a file: per-chunk sha256 hex digests and the root over them."""

    algo: str
    chunk_size: int
    chunks: list[str]
    root: str


def _read_chunks(path: str | os.PathLike[str], chunk_size: int) -> Iterator[bytes]:
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be at least 1 byte, got {chunk_size}")
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            yield chunk


def content_identity(path: str | os.PathLike[str], *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> ContentId:
    """Return the chunked identity of a file, read one chunk at a time.

    `chunks` holds the sha256 hex of each consecutive `chunk_size`-byte chunk (the last may be shorter;
    an empty file has none). `root` is `identity_digest("content-root@2", {"chunk_size": chunk_size,
    "chunks": chunks})`, so the same bytes read in two chunk sizes have two roots.

    Raises:
        ValueError: if `chunk_size` is below 1.
    """
    chunks = [_raw_digest.bytes_digest("file@1", chunk) for chunk in _read_chunks(path, chunk_size)]
    return {
        "algo": "sha256",
        "chunk_size": chunk_size,
        "chunks": chunks,
        "root": identity_digest("content-root@2", {"chunk_size": chunk_size, "chunks": chunks}),
    }


def weights_identity(paths: Sequence[str | os.PathLike[str]], *, root: str | os.PathLike[str]) -> str:
    """Return the identity of a set of weight files under `root`, independent of listing order.

    Each file contributes `[relpath, sha256]`: its path relative to `root` with POSIX separators,
    in canonical form (`canonical_text`; symlinks are not resolved, so a link is named by where it sits and hashed by
    what it points to), and `file_digest("file@1", path)`. The identity is `identity_digest("weights@3", pairs)`
    over the pairs sorted by relpath in UTF-16 code-unit order, the order JCS uses for object keys.

    Raises:
        ValueError: if a path lies outside `root`, the same path is listed twice after
            normalisation, or a path holds a lone surrogate (`InvalidText`).
    """
    root_abs = os.path.abspath(root)
    pairs: dict[str, str] = {}
    for path in paths:
        rel = os.path.relpath(os.path.abspath(path), root_abs)
        if rel == os.curdir or rel == os.pardir or rel.startswith(os.pardir + os.sep):
            raise ValueError(f"weight file is not under {root_abs}: {path}")
        posix = canonical_text(PurePath(rel).as_posix())
        if posix in pairs:
            raise ValueError(f"weight file listed twice: {posix}")
        pairs[posix] = file_digest("file@1", path)
    ordered = sorted(pairs, key=utf16_sort_key)
    return identity_digest("weights@3", [[rel, pairs[rel]] for rel in ordered])
