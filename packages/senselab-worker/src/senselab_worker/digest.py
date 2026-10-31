"""The digest API: every digest names a registered tag (ADR-006).

`identity_digest(tag, value)` digests a canonical JSON value under an identity tag; `blob_digest(tag, data)`
digests raw content (bytes, a file or a binary stream) under a blob tag. No other digest function exists.
"""

import os
from typing import BinaryIO

__all__ = ["BLOB_TAGS", "IDENTITY_TAGS", "blob_digest", "identity_digest"]

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
    }
)
"""The registered tags of `identity_digest`; no tag is a prefix of another."""

BLOB_TAGS: frozenset[str] = frozenset({"file@1", "npy@1"})
"""The registered tags of `blob_digest`: `file@1` for a file's or a source's bytes, `npy@1` for an array's `.npy`."""

_CHUNK_SIZE = 2**20


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


def blob_digest(tag: str, data: bytes | bytearray | memoryview | str | os.PathLike[str] | BinaryIO) -> str:
    """Return the lowercase sha256 hex of raw content: bytes, the file at a path, or a binary stream read to its end.

    The tag names what the bytes are and is not hashed, so the digest is the content's plain sha256.

    Raises:
        ValueError: if `tag` is not in `BLOB_TAGS`.
        TypeError: if `data` is none of bytes, a path or a binary stream.
        OSError: if the file cannot be read.
    """
    import hashlib

    if tag not in BLOB_TAGS:
        raise ValueError(f"unregistered blob tag {tag!r}; registered: {sorted(BLOB_TAGS)}")
    if isinstance(data, (bytes, bytearray, memoryview)):
        return hashlib.sha256(data).hexdigest()
    digest = hashlib.sha256()
    if isinstance(data, (str, os.PathLike)):
        with open(data, "rb") as f:
            while chunk := f.read(_CHUNK_SIZE):
                digest.update(chunk)
        return digest.hexdigest()
    read = getattr(data, "read", None)
    if read is None:
        raise TypeError(f"blob_digest takes bytes, a path or a binary stream, not {type(data).__name__}")
    while chunk := read(_CHUNK_SIZE):
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise TypeError(f"blob_digest takes a binary stream; read() returned {type(chunk).__name__}")
        digest.update(chunk)
    return digest.hexdigest()
