"""The tagged digests, the values they keep, and the inputs they refuse."""

import hashlib
import io
import json
from pathlib import Path

import numpy
import pytest
from senselab_worker import _raw_digest
from senselab_worker.canonical import canonical_npy_bytes, jcs_dumps
from senselab_worker.digest import FILE_TAGS, IDENTITY_TAGS, NPY_TAGS, file_digest, identity_digest, npy_digest


@pytest.mark.logic
@pytest.mark.parametrize("tag", sorted(FILE_TAGS))
def test_a_file_digest_is_the_plain_sha256_of_the_file(tag: str, tmp_path: Path) -> None:
    """A path (str or Path) and a binary file opened on disk give the file's sha256."""
    data = bytes(range(256)) * 9000
    path = tmp_path / "blob.bin"
    path.write_bytes(data)
    expected = hashlib.sha256(data).hexdigest()
    assert file_digest(tag, path) == expected
    assert file_digest(tag, str(path)) == expected
    with path.open("rb") as f:
        assert file_digest(tag, f) == expected
    with open(path, "rb", buffering=0) as raw:
        assert file_digest(tag, raw) == expected


@pytest.mark.logic
@pytest.mark.parametrize("tag", sorted(NPY_TAGS))
def test_an_npy_digest_is_the_sha256_of_the_canonical_npy(tag: str) -> None:
    """`npy_digest` hashes `canonical_npy_bytes`, so two layouts of one array have one digest."""
    a = numpy.arange(6, dtype="<f4").reshape(2, 3)
    expected = hashlib.sha256(canonical_npy_bytes(a)).hexdigest()
    assert npy_digest(tag, a) == expected
    assert npy_digest(tag, numpy.asfortranarray(a.astype(">f4"))) == expected


@pytest.mark.logic
@pytest.mark.parametrize("tag", sorted(IDENTITY_TAGS))
def test_an_identity_digest_hashes_the_tag_a_zero_byte_and_the_jcs(tag: str) -> None:
    """`identity_digest` is `sha256(tag || 0x00 || JCS(value))`."""
    value = {"b": [1, 2.5, "x"], "a": None}
    assert identity_digest(tag, value) == hashlib.sha256(tag.encode() + b"\x00" + jcs_dumps(value)).hexdigest()


@pytest.mark.logic
@pytest.mark.parametrize("tag", ["", "file", "file@2", "npy@1", "logical-key@3", "output@3", "file@1\x00"])
def test_a_file_digest_refuses_any_other_tag(tag: str, tmp_path: Path) -> None:
    """A tag outside `FILE_TAGS`, npy and identity tags included, is refused."""
    path = tmp_path / "x"
    path.write_bytes(b"x")
    with pytest.raises(ValueError, match="unregistered file tag"):
        file_digest(tag, path)


@pytest.mark.logic
@pytest.mark.parametrize("tag", ["", "npy@2", "file@1", "logical-key@3", "output@3"])
def test_an_npy_digest_refuses_any_other_tag(tag: str) -> None:
    """A tag outside `NPY_TAGS`, file and identity tags included, is refused."""
    with pytest.raises(ValueError, match="unregistered npy tag"):
        npy_digest(tag, numpy.zeros(1))


@pytest.mark.logic
@pytest.mark.parametrize("tag", ["file@1", "npy@1", "value@1", "output@2"])
def test_an_identity_digest_refuses_a_blob_or_retired_tag(tag: str) -> None:
    """Blob tags and retired majors are not identity tags."""
    with pytest.raises(ValueError, match="unregistered identity tag"):
        identity_digest(tag, 1)


SERIALISED = json.dumps({"a": [1, 2]}, sort_keys=True)


@pytest.mark.logic
@pytest.mark.parametrize(
    "source",
    [
        SERIALISED.encode(),
        bytearray(SERIALISED.encode()),
        memoryview(SERIALISED.encode()),
        io.BytesIO(SERIALISED.encode()),
    ],
    ids=["bytes", "bytearray", "memoryview", "BytesIO"],
)
def test_i1_serialised_bytes_cannot_reach_a_file_digest(source: object) -> None:
    """Bytes and in-memory streams of `json.dumps(...).encode()` are refused: only a file on disk is hashed."""
    with pytest.raises(TypeError, match="path or a binary file opened on disk"):
        file_digest("file@1", source)  # type: ignore[arg-type]


@pytest.mark.logic
def test_i1_serialised_text_is_read_as_a_path_never_hashed(tmp_path: Path) -> None:
    """A `str` is a path: serialised text names no file and raises, and a text stream is refused."""
    with pytest.raises(FileNotFoundError):
        file_digest("file@1", str(tmp_path / SERIALISED))
    with pytest.raises(TypeError):
        file_digest("file@1", io.StringIO(SERIALISED))  # type: ignore[arg-type]


@pytest.mark.logic
@pytest.mark.parametrize("value", [SERIALISED, SERIALISED.encode(), [1, 2], 3.0], ids=["str", "bytes", "list", "float"])
def test_i1_serialised_data_cannot_reach_an_npy_digest(value: object) -> None:
    """`npy_digest` takes an array only; text, bytes and other values are refused by type."""
    with pytest.raises(TypeError, match="takes a numpy array"):
        npy_digest("npy@1", value)  # type: ignore[arg-type]


@pytest.mark.logic
def test_the_raw_form_checks_its_tag_and_type() -> None:
    """The owners' raw form is the plain sha256 under a blob tag, and refuses other tags and non-bytes."""
    assert _raw_digest.bytes_digest("npy@1", b"abc") == hashlib.sha256(b"abc").hexdigest()
    with pytest.raises(ValueError, match="unregistered blob tag"):
        _raw_digest.bytes_digest("output@3", b"abc")
    with pytest.raises(TypeError):
        _raw_digest.bytes_digest("file@1", "abc")  # type: ignore[arg-type]
