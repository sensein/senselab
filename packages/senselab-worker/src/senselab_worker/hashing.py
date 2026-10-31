"""Content identities of bytes, files and weight sets (ADR-002, P-4)."""

import hashlib
import os
import unicodedata
from pathlib import PurePath
from typing import Iterator, Sequence, TypedDict

from senselab_worker.canonical import jcs_dumps, utf16_sort_key

DEFAULT_CHUNK_SIZE = 64 * 2**20


class ContentId(TypedDict):
    """Chunked identity of a file: per-chunk sha256 hex digests and the root over them."""

    algo: str
    chunk_size: int
    chunks: list[str]
    root: str


def sha256_bytes(b: bytes) -> str:
    """Return the lowercase hex sha256 of `b`."""
    return hashlib.sha256(b).hexdigest()


def _read_chunks(path: str | os.PathLike[str], chunk_size: int) -> Iterator[bytes]:
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be at least 1 byte, got {chunk_size}")
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            yield chunk


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return the lowercase hex sha256 of the file at `path`, read one chunk at a time."""
    digest = hashlib.sha256()
    for chunk in _read_chunks(path, DEFAULT_CHUNK_SIZE):
        digest.update(chunk)
    return digest.hexdigest()


def content_identity(path: str | os.PathLike[str], *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> ContentId:
    """Return the chunked identity of a file, read one chunk at a time.

    `chunks` holds the sha256 of each consecutive `chunk_size`-byte chunk (the last may be shorter;
    an empty file has none). `root` is the sha256 of the concatenated chunk hex digests.

    Raises:
        ValueError: if `chunk_size` is below 1.
    """
    chunks = [sha256_bytes(chunk) for chunk in _read_chunks(path, chunk_size)]
    return {
        "algo": "sha256",
        "chunk_size": chunk_size,
        "chunks": chunks,
        "root": sha256_bytes("".join(chunks).encode()),
    }


def weights_identity(paths: Sequence[str | os.PathLike[str]], *, root: str | os.PathLike[str]) -> str:
    """Return the identity of a set of weight files under `root`, independent of listing order.

    Each file contributes `[relpath, sha256]`: its path relative to `root` with POSIX separators,
    NFC-normalised (symlinks are not resolved, so a link is named by where it sits and hashed by
    what it points to), and the sha256 of its bytes. The identity is the sha256 of the JCS of the
    pairs sorted by relpath in UTF-16 code-unit order, the order JCS uses for object keys.

    Raises:
        ValueError: if a path lies outside `root` or the same path is listed twice after
            normalisation.
    """
    root_abs = os.path.abspath(root)
    pairs: dict[str, str] = {}
    for path in paths:
        rel = os.path.relpath(os.path.abspath(path), root_abs)
        if rel == os.curdir or rel == os.pardir or rel.startswith(os.pardir + os.sep):
            raise ValueError(f"weight file is not under {root_abs}: {path}")
        posix = unicodedata.normalize("NFC", PurePath(rel).as_posix())
        if posix in pairs:
            raise ValueError(f"weight file listed twice: {posix}")
        pairs[posix] = sha256_file(path)
    ordered = sorted(pairs, key=utf16_sort_key)
    return sha256_bytes(jcs_dumps([[rel, pairs[rel]] for rel in ordered]))
