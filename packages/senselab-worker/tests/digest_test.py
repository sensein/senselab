"""The two tagged digests and the values they keep."""

import hashlib
import io
from pathlib import Path

import pytest
from senselab_worker.canonical import jcs_dumps
from senselab_worker.digest import BLOB_TAGS, IDENTITY_TAGS, blob_digest, identity_digest


@pytest.mark.logic
@pytest.mark.parametrize("tag", sorted(BLOB_TAGS))
def test_a_blob_digest_is_the_plain_sha256_of_the_content(tag: str, tmp_path: Path) -> None:
    """Bytes, a file at a path (str or Path) and a binary stream give the content's sha256, whatever the tag."""
    data = bytes(range(256)) * 9000
    expected = hashlib.sha256(data).hexdigest()
    path = tmp_path / "blob.bin"
    path.write_bytes(data)
    assert blob_digest(tag, data) == expected
    assert blob_digest(tag, bytearray(data)) == expected
    assert blob_digest(tag, memoryview(data)) == expected
    assert blob_digest(tag, path) == expected
    assert blob_digest(tag, str(path)) == expected
    assert blob_digest(tag, io.BytesIO(data)) == expected


@pytest.mark.logic
@pytest.mark.parametrize("tag", sorted(IDENTITY_TAGS))
def test_an_identity_digest_hashes_the_tag_a_zero_byte_and_the_jcs(tag: str) -> None:
    """`identity_digest` is `sha256(tag || 0x00 || JCS(value))`."""
    value = {"b": [1, 2.5, "x"], "a": None}
    assert identity_digest(tag, value) == hashlib.sha256(tag.encode() + b"\x00" + jcs_dumps(value)).hexdigest()


@pytest.mark.logic
@pytest.mark.parametrize("tag", ["", "file", "file@2", "output@3 ", "npy@1\x00"])
def test_a_blob_digest_refuses_an_unregistered_tag(tag: str) -> None:
    """A tag outside `BLOB_TAGS`, identity tags included, is refused."""
    with pytest.raises(ValueError, match="unregistered blob tag"):
        blob_digest(tag, b"x")


@pytest.mark.logic
@pytest.mark.parametrize("tag", ["file@1", "npy@1", "value@1", "output@2"])
def test_an_identity_digest_refuses_a_blob_or_retired_tag(tag: str) -> None:
    """Blob tags and retired majors are not identity tags."""
    with pytest.raises(ValueError, match="unregistered identity tag"):
        identity_digest(tag, 1)


@pytest.mark.logic
def test_a_blob_digest_refuses_text_and_text_streams(tmp_path: Path) -> None:
    """Text is read as a path, never hashed as content, and a text stream is refused."""
    with pytest.raises(FileNotFoundError):
        blob_digest("file@1", str(tmp_path / '{"a": 1}'))
    with pytest.raises(TypeError, match="binary stream"):
        blob_digest("file@1", io.StringIO("text"))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bytes, a path or a binary stream"):
        blob_digest("file@1", 3)  # type: ignore[arg-type]
