"""The digest API: every digest names a registered tag and states what it hashes (ADR-006).

`identity_digest(tag, value)` digests a canonical JSON value under an identity tag; `file_digest(tag, file)` digests
a file on disk and `npy_digest(tag, array)` an array's canonical `.npy`. No public function digests caller bytes.
"""

import io
import os
from typing import BinaryIO

import numpy

__all__ = ["BLOB_TAGS", "FILE_TAGS", "IDENTITY_TAGS", "NPY_TAGS", "file_digest", "identity_digest", "npy_digest"]

IDENTITY_TAGS: frozenset[str] = frozenset(
    {
        "logical-key@3",
        "output@3",
        "context@3",
        "content-root@2",
        "audio@3",
        "weights@3",
        "value@2",
        "provenance-id@3",
        "data-manifest@3",
        "worker-runtime@1",
        "fake-output@1",
        "contract-probe@1",
    }
)
"""The registered tags of `identity_digest`; no tag is a prefix of another."""

FILE_TAGS: frozenset[str] = frozenset({"file@1"})
"""The registered tags of `file_digest`: the bytes of a file, as its published sha256 records them."""
NPY_TAGS: frozenset[str] = frozenset({"npy@1"})
"""The registered tags of `npy_digest`: an array's canonical `.npy` bytes."""
BLOB_TAGS: frozenset[str] = FILE_TAGS | NPY_TAGS
"""Every tag whose digest is the plain sha256 of content."""

_CHUNK_SIZE = 2**20
_FILE_STREAMS = (io.BufferedReader, io.BufferedRandom, io.FileIO)


def identity_digest(tag: str, value: object) -> str:
    """Return the sha256 hex of `tag` in ASCII, a zero byte and the JCS of `value`.

    Raises:
        ValueError: if `tag` is not in `IDENTITY_TAGS`, or `jcs_dumps` refuses `value`.
        TypeError: if `jcs_dumps` refuses `value`.
    """
    import hashlib

    from senselab_worker.canonical import jcs_dumps

    if tag not in IDENTITY_TAGS:
        raise ValueError(f"unregistered identity tag {tag!r}; registered: {sorted(IDENTITY_TAGS)}")
    return hashlib.sha256(tag.encode("ascii") + b"\x00" + jcs_dumps(value)).hexdigest()


def file_digest(tag: str, source: str | os.PathLike[str] | BinaryIO) -> str:
    """Return the lowercase sha256 hex of a file: the file at a path, or a binary file object opened on disk.

    The file is read in chunks from its current position to its end. The tag is checked and not hashed.

    Raises:
        ValueError: if `tag` is not in `FILE_TAGS`.
        TypeError: if `source` is neither a path nor a binary file opened on disk (in-memory streams are refused).
        OSError: if the file cannot be read.
    """
    import hashlib

    if tag not in FILE_TAGS:
        raise ValueError(f"unregistered file tag {tag!r}; registered: {sorted(FILE_TAGS)}")
    if isinstance(source, (str, os.PathLike)):
        with open(source, "rb") as f:
            return file_digest(tag, f)
    if not isinstance(source, _FILE_STREAMS):
        raise TypeError(f"file_digest takes a path or a binary file opened on disk, not {type(source).__name__}")
    digest = hashlib.sha256()
    while chunk := source.read(_CHUNK_SIZE):
        digest.update(chunk)
    return digest.hexdigest()


def npy_digest(tag: str, array: numpy.ndarray) -> str:
    """Return the lowercase sha256 hex of `canonical_npy_bytes(array)`; the tag is checked and not hashed.

    Raises:
        ValueError: if `tag` is not in `NPY_TAGS`, or the array has no canonical form.
        TypeError: if `array` is not a numpy array.
    """
    import hashlib

    from senselab_worker.canonical import canonical_npy_bytes

    if tag not in NPY_TAGS:
        raise ValueError(f"unregistered npy tag {tag!r}; registered: {sorted(NPY_TAGS)}")
    if not isinstance(array, numpy.ndarray):
        raise TypeError(f"npy_digest takes a numpy array, not {type(array).__name__}")
    return hashlib.sha256(canonical_npy_bytes(array)).hexdigest()
