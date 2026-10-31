"""Wire codec: framing, array references and their integrity checks."""

import json
import os
import time
from pathlib import Path

import numpy
import pytest
from senselab_worker import runtime_identity
from senselab_worker.canonical import canonical_npy_bytes
from senselab_worker.codec import (
    MAX_LINE_BYTES,
    Deadline,
    array_paths,
    decode,
    encode,
    from_wire,
    parse_line,
    to_wire,
)
from senselab_worker.faults import ArrayIntegrityError, ProtocolError
from senselab_worker.hashing import sha256_bytes


@pytest.mark.logic
def test_runtime_identity_is_stable_and_hex() -> None:
    """The identity is a 64-character hex digest, the same on every call."""
    assert runtime_identity() == runtime_identity() and len(runtime_identity()) == 64
    int(runtime_identity(), 16)


@pytest.mark.logic
def test_array_roundtrip_across_numpy_majors(tmp_path: Path) -> None:
    """An array survives encode and decode bit for bit."""
    a = numpy.linspace(-1, 1, 16000, dtype="<f4").reshape(1, -1)
    msg = decode(encode({"x": a}, array_dir=tmp_path), array_dir=tmp_path)
    assert msg["x"].tobytes() == a.tobytes()


@pytest.mark.logic
def test_array_file_is_canonical_npy_named_by_its_sha256(tmp_path: Path) -> None:
    """The file holds `canonical_npy_bytes` and its name and reference carry their sha256."""
    a = numpy.arange(6, dtype=">i4").reshape(2, 3)[:, ::2]
    ref = to_wire(a, array_dir=tmp_path)["__array__"]
    data = canonical_npy_bytes(a)
    assert Path(ref["path"]).read_bytes() == data
    assert ref["sha256"] == sha256_bytes(data) and Path(ref["path"]).name == f"{ref['sha256']}.npy"
    assert ref["dtype"] == "<i4" and ref["shape"] == [2, 2]
    assert numpy.array_equal(from_wire({"__array__": ref}, array_dir=tmp_path), a)


@pytest.mark.logic
def test_equal_arrays_share_one_file_and_no_temp_file_remains(tmp_path: Path) -> None:
    """Writing equal arrays twice leaves exactly one file."""
    a = numpy.ones(4, dtype="<f8")
    to_wire([a, a.copy()], array_dir=tmp_path)
    assert [p.suffix for p in tmp_path.iterdir()] == [".npy"]


@pytest.mark.logic
@pytest.mark.parametrize(
    "value",
    [
        numpy.array([object()]),
        numpy.zeros(2, dtype=[("a", "<f4")]),
        float("nan"),
        float("-inf"),
        numpy.float32("nan"),
        {"__array__": {}},
        {1: "x"},
        b"bytes",
        "\ud800",
        1 + 2j,
    ],
)
def test_unencodable_values_are_content_errors(tmp_path: Path, value: object) -> None:
    """Values with no wire form raise `ValueError` or `TypeError`."""
    with pytest.raises((ValueError, TypeError)):
        encode({"x": value}, array_dir=tmp_path)


@pytest.mark.logic
def test_numpy_scalars_travel_as_items(tmp_path: Path) -> None:
    """Numpy scalars are written as their `.item()`."""
    line = encode({"a": numpy.float32(0.5), "b": numpy.int64(3), "c": numpy.bool_(True)}, array_dir=tmp_path)
    assert json.loads(line) == {"a": 0.5, "b": 3, "c": True}
    assert line.endswith(b"\n") and line.count(b"\n") == 1


def _ref(tmp_path: Path, a: numpy.ndarray) -> dict:
    return to_wire(a, array_dir=tmp_path)


@pytest.mark.logic
@pytest.mark.parametrize("field", ["sha256", "dtype", "shape", "path", "missing", "bytes", "malformed"])
def test_bad_array_reference_is_integrity_error(tmp_path: Path, field: str) -> None:
    """Each kind of damaged reference raises `ArrayIntegrityError`."""
    ref = _ref(tmp_path, numpy.arange(3, dtype="<f4"))
    inner = ref["__array__"]
    if field == "sha256":
        inner["sha256"] = "0" * 64
    elif field == "dtype":
        inner["dtype"] = "<f8"
    elif field == "shape":
        inner["shape"] = [1, 3]
    elif field == "path":
        outside = tmp_path.parent / Path(inner["path"]).name
        outside.write_bytes(Path(inner["path"]).read_bytes())
        inner["path"] = str(outside)
    elif field == "missing":
        Path(inner["path"]).unlink()
    elif field == "bytes":
        Path(inner["path"]).write_bytes(Path(inner["path"]).read_bytes()[:-1])
    else:
        ref = {"__array__": "nope"}
    with pytest.raises(ArrayIntegrityError):
        from_wire({"x": [ref]}, array_dir=tmp_path)


@pytest.mark.logic
@pytest.mark.parametrize(
    "line",
    [b"[1]\n", b"not json\n", b'{"x": NaN}\n', b'{"x": Infinity}\n', b"\xff\n", b" " * (MAX_LINE_BYTES + 1)],
)
def test_bad_lines_are_protocol_errors(line: bytes) -> None:
    """A line that is not a JSON object within the limit raises `ProtocolError`."""
    with pytest.raises(ProtocolError):
        parse_line(line)


@pytest.mark.logic
def test_oversized_line_is_refused_on_write(tmp_path: Path) -> None:
    """The writer refuses a line over the limit."""
    with pytest.raises(ProtocolError):
        encode({"x": "a" * MAX_LINE_BYTES}, array_dir=tmp_path)


@pytest.mark.logic
def test_array_paths_lists_every_reference(tmp_path: Path) -> None:
    """`array_paths` finds references at any depth."""
    a, b = numpy.zeros(2), numpy.ones(2)
    wire = to_wire({"p": [a, {"q": b}], "r": 1}, array_dir=tmp_path)
    assert array_paths(wire) == {Path(p) for p in tmp_path.iterdir()}


@pytest.mark.logic
@pytest.mark.parametrize(
    "data",
    [b"", b"\x93NUMPY\x01\x00", b"\x93NUMPY\x01\x00\x10\x00{'descr': '<f4'", b"not an npy file", b"\x93NUMPY\x09\x00"],
)
def test_malformed_file_with_matching_sha256_is_integrity_error(tmp_path: Path, data: bytes) -> None:
    """A file whose sha256 matches its reference but which numpy cannot load raises `ArrayIntegrityError`."""
    sha = sha256_bytes(data)
    (tmp_path / f"{sha}.npy").write_bytes(data)
    ref = {"__array__": {"path": str(tmp_path / f"{sha}.npy"), "sha256": sha, "dtype": "<f4", "shape": [1]}}
    with pytest.raises(ArrayIntegrityError):
        from_wire(ref, array_dir=tmp_path)


@pytest.mark.logic
def test_deadline_counts_down_and_names_what_expired() -> None:
    """`None` never expires; a spent deadline raises `TimeoutError` naming the step."""
    assert Deadline(None).remaining() is None and not Deadline(None).expired()
    Deadline(None).check("anything")
    live = Deadline(60)
    remaining = live.remaining()
    assert remaining is not None and 0 < remaining <= 60 and not live.expired()
    spent = Deadline(0)
    assert spent.expired() and spent.remaining() == 0
    with pytest.raises(TimeoutError, match="reading arrays"):
        spent.check("reading arrays")


@pytest.mark.logic
def test_fifo_array_ref_is_refused_without_blocking(tmp_path: Path) -> None:
    """A FIFO at `<sha>.npy` is not a regular file: refused at once, never opened for a blocking read."""
    sha = "b" * 64
    os.mkfifo(tmp_path / f"{sha}.npy")
    ref = {"__array__": {"path": str(tmp_path / f"{sha}.npy"), "sha256": sha, "dtype": "<f4", "shape": [1]}}
    t0 = time.monotonic()
    with pytest.raises(ArrayIntegrityError, match="regular file"):
        from_wire(ref, array_dir=tmp_path)
    assert time.monotonic() - t0 < 1


@pytest.mark.logic
@pytest.mark.parametrize("target", ["inside", "outside"])
def test_symlinked_array_ref_is_refused(tmp_path: Path, target: str) -> None:
    """A symlink at `<sha>.npy` is refused even when it points at the right bytes."""
    root = tmp_path / "arrays"
    root.mkdir()
    ref = to_wire(numpy.arange(3, dtype="<f4"), array_dir=root)
    real = Path(ref["__array__"]["path"])
    elsewhere = (root if target == "inside" else tmp_path) / "real.bin"
    real.replace(elsewhere)
    os.symlink(elsewhere, real)
    with pytest.raises(ArrayIntegrityError):
        from_wire(ref, array_dir=root)


@pytest.mark.logic
def test_array_read_stops_at_the_deadline(tmp_path: Path) -> None:
    """A spent deadline stops the read with `TimeoutError`, not an integrity error."""
    ref = to_wire(numpy.zeros(2**20, dtype="<f8"), array_dir=tmp_path)
    with pytest.raises(TimeoutError):
        from_wire(ref, array_dir=tmp_path, deadline=Deadline(0))


@pytest.mark.logic
def test_writer_leaves_no_temporary_after_a_failed_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A write that fails after the temporary file exists removes it."""
    monkeypatch.setattr(os, "replace", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        to_wire(numpy.ones(3), array_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []
