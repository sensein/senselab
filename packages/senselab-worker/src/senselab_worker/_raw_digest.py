"""The sha256 of bytes already in memory, for the owners `.importlinter` names (ADR-006).

Callers import the module and call `_raw_digest.bytes_digest`, so no other namespace binds the function.
"""

from senselab_worker.digest import BLOB_TAGS


def bytes_digest(tag: str, data: bytes | bytearray | memoryview) -> str:
    """Return the lowercase sha256 hex of `data`; the tag is checked against `BLOB_TAGS` and not hashed.

    Raises:
        ValueError: if `tag` is not in `BLOB_TAGS`.
        TypeError: if `data` is not bytes-like.
    """
    import hashlib

    if tag not in BLOB_TAGS:
        raise ValueError(f"unregistered blob tag {tag!r}; registered: {sorted(BLOB_TAGS)}")
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError(f"bytes_digest takes bytes, not {type(data).__name__}")
    return hashlib.sha256(data).hexdigest()
