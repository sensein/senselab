"""Wire codec: framing, array references and their integrity checks."""

import hashlib
import json
import os
import time
from pathlib import Path

import numpy
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from senselab_worker import runtime_identity
from senselab_worker.canonical import npy_bytes
from senselab_worker.codec import (
    MAX_DEPTH,
    MAX_LINE_BYTES,
    Deadline,
    LineTooLarge,
    array_paths,
    decode,
    dump_line,
    encode,
    from_wire,
    parse_line,
    to_wire,
    validate_message,
)
from senselab_worker.faults import (
    ArrayIntegrityError,
    ContentFault,
    ProtocolError,
    ValueTooLarge,
    WireValueError,
    error_record,
)


@pytest.mark.logic
def test_runtime_identity_is_stable_and_hex() -> None:
    """The identity is a 64-character hex digest, the same on every call."""
    assert runtime_identity() == runtime_identity() and len(runtime_identity()) == 64
    int(runtime_identity(), 16)


@pytest.mark.logic
def test_array_roundtrip_across_numpy_majors(tmp_path: Path) -> None:
    """An array survives encode and decode bit for bit."""
    a = numpy.linspace(-1, 1, 16000, dtype="<f4").reshape(1, -1)
    request = {"id": "r", "items": [{"id": "i", "inputs": {"x": a}}], "params": {}}
    msg = decode(encode(request, "request", array_dir=tmp_path), "request", array_dir=tmp_path)
    assert msg["items"][0]["inputs"]["x"].tobytes() == a.tobytes()


@pytest.mark.logic
def test_array_file_is_npy_named_by_its_sha256(tmp_path: Path) -> None:
    """The file holds `npy_bytes` and its name and reference carry their sha256."""
    a = numpy.arange(6, dtype=">i4").reshape(2, 3)[:, ::2]
    ref = to_wire(a, array_dir=tmp_path, at="line")["__array__"]
    data = npy_bytes(a)
    assert Path(ref["path"]).read_bytes() == data
    assert ref["sha256"] == hashlib.sha256(data).hexdigest() and Path(ref["path"]).name == f"{ref['sha256']}.npy"
    assert ref["dtype"] == "<i4" and ref["shape"] == [2, 2]
    assert numpy.array_equal(from_wire({"__array__": ref}, array_dir=tmp_path), a)


@pytest.mark.logic
def test_the_wire_carries_every_value_bit(tmp_path: Path) -> None:
    """A NaN payload, a NaN sign, `-0.0` and a bool byte of 2 arrive as sent; only identity canonicalises them."""
    floats = numpy.array([0x7FF8000000000001, 0xFFF8000000000000, 0x8000000000000000], dtype="<u8").view("<f8")
    bools = numpy.frombuffer(bytes([0, 1, 2]), dtype=numpy.bool_)
    for a in (floats, bools):
        back = from_wire(to_wire(a, array_dir=tmp_path, at="line"), array_dir=tmp_path)
        assert back.tobytes() == a.tobytes()


@pytest.mark.logic
def test_equal_arrays_share_one_file_and_no_temp_file_remains(tmp_path: Path) -> None:
    """Writing equal arrays twice leaves exactly one file."""
    a = numpy.ones(4, dtype="<f8")
    to_wire([a, a.copy()], array_dir=tmp_path, at="line")
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
    """Values with no wire form raise `WireValueError`, a typed content fault and a `ValueError`."""
    with pytest.raises(WireValueError) as info:
        to_wire({"x": value}, array_dir=tmp_path, at="line")
    assert isinstance(info.value, (ContentFault, ValueError))
    assert (error_record(info.value)["fault"], error_record(info.value)["gap"]) == ("content", False)


@pytest.mark.logic
def test_numpy_scalars_travel_as_items(tmp_path: Path) -> None:
    """Numpy scalars are written as their `.item()`."""
    wire = to_wire(
        {"a": numpy.float32(0.5), "b": numpy.int64(3), "c": numpy.bool_(True)}, array_dir=tmp_path, at="line"
    )
    assert wire == {"a": 0.5, "b": 3, "c": True}
    assert [type(v) for v in wire.values()] == [float, int, bool]


def _ref(tmp_path: Path, a: numpy.ndarray) -> dict:
    return to_wire(a, array_dir=tmp_path, at="line")


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
    with pytest.raises(LineTooLarge):
        dump_line({"id": "r", "item": "i", "outputs": {"x": "a" * (MAX_LINE_BYTES - 30)}}, "result")
    with pytest.raises(ValueTooLarge):
        encode({"id": "r", "item": "i", "outputs": {"x": "a" * MAX_LINE_BYTES}}, "result", array_dir=tmp_path)


@pytest.mark.logic
def test_array_paths_lists_every_reference(tmp_path: Path) -> None:
    """`array_paths` finds references at any depth."""
    a, b = numpy.zeros(2), numpy.ones(2)
    wire = to_wire({"p": [a, {"q": b}], "r": 1}, array_dir=tmp_path, at="line")
    assert array_paths(wire) == {Path(p) for p in tmp_path.iterdir()}


@pytest.mark.logic
@pytest.mark.parametrize(
    "data",
    [b"", b"\x93NUMPY\x01\x00", b"\x93NUMPY\x01\x00\x10\x00{'descr': '<f4'", b"not an npy file", b"\x93NUMPY\x09\x00"],
)
def test_malformed_file_with_matching_sha256_is_integrity_error(tmp_path: Path, data: bytes) -> None:
    """A file whose sha256 matches its reference but which numpy cannot load raises `ArrayIntegrityError`."""
    sha = hashlib.sha256(data).hexdigest()
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
    ref = to_wire(numpy.arange(3, dtype="<f4"), array_dir=root, at="line")
    real = Path(ref["__array__"]["path"])
    elsewhere = (root if target == "inside" else tmp_path) / "real.bin"
    real.replace(elsewhere)
    os.symlink(elsewhere, real)
    with pytest.raises(ArrayIntegrityError):
        from_wire(ref, array_dir=root)


@pytest.mark.logic
def test_array_read_stops_at_the_deadline(tmp_path: Path) -> None:
    """A spent deadline stops the read with `TimeoutError`, not an integrity error."""
    ref = to_wire(numpy.zeros(2**20, dtype="<f8"), array_dir=tmp_path, at="line")
    with pytest.raises(TimeoutError):
        from_wire(ref, array_dir=tmp_path, deadline=Deadline(0))


@pytest.mark.logic
def test_writer_leaves_no_temporary_after_a_failed_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A write that fails after the temporary file exists removes it."""
    monkeypatch.setattr(os, "replace", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        to_wire(numpy.ones(3), array_dir=tmp_path, at="line")
    assert list(tmp_path.iterdir()) == []


_ERROR = {"type": "ValueError", "message": "m", "fault": "content", "traceback": "", "gap": True}
VALID: dict[str, dict] = {
    "hello": {"hello": {"wire": 2, "runtime": "a" * 64, "python": "3.12.0", "numpy": "2.0.0", "pid": 7}},
    "init": {"init": {"array_dir": "/tmp/x", "backend_key": 1}},
    "ready": {"ready": {"load_s": 0.5}},
    "request": {"id": "r", "items": [{"id": "a", "inputs": {}}, {"id": "b", "inputs": {"x": 1}}], "params": {}},
    "result": {"id": "r", "item": "a", "outputs": {"y": [1, 2]}},
    "done": {"id": "r", "done": 2},
    "fatal": {"fatal": True, "error": dict(_ERROR)},
    "stop": {"stop": True},
}
INT, NUMBER, STR, DICT, LIST, BOOL, TRUE = (
    {int},
    {int, float},
    {str},
    {dict},
    {list},
    {bool},
    {"true"},
)
FIELDS: dict[str, list[tuple[tuple[object, ...], set]]] = {
    "hello": [
        (("hello",), DICT),
        (("hello", "wire"), INT),
        (("hello", "runtime"), STR),
        (("hello", "python"), STR),
        (("hello", "numpy"), STR),
        (("hello", "pid"), INT),
    ],
    "init": [(("init",), DICT), (("init", "array_dir"), STR)],
    "ready": [(("ready",), DICT), (("ready", "load_s"), NUMBER)],
    "request": [
        (("id",), STR),
        (("items",), LIST),
        (("items", 0), DICT),
        (("items", 0, "id"), STR),
        (("items", 1, "inputs"), DICT),
        (("params",), DICT),
    ],
    "result": [(("id",), STR), (("item",), STR), (("outputs",), DICT)],
    "done": [(("id",), STR), (("done",), INT)],
    "fatal": [(("fatal",), TRUE)],
    "stop": [(("stop",), TRUE)],
}
WRONG: list[object] = [True, False, 1, 1.0, "1", "true", None, [], {}]


def _accepts(ok: set, value: object) -> bool:
    if ok == TRUE:
        return value is True
    return type(value) in ok


def _set(msg: dict, path: tuple[object, ...], value: object) -> dict:
    out = json.loads(json.dumps(msg))
    target = out
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]
    return out


def _delete(msg: dict, path: tuple[object, ...]) -> dict:
    out = json.loads(json.dumps(msg))
    target = out
    for key in path[:-1]:
        target = target[key]
    if isinstance(path[-1], int):
        target[path[-1]] = "gone"
    else:
        del target[path[-1]]
    return out


@pytest.mark.logic
@pytest.mark.parametrize("kind", sorted(VALID))
def test_validate_message_accepts_each_valid_kind_and_ignores_unknown_keys(kind: str) -> None:
    """Each kind's sample passes, with an unknown key added at the top and inside, and comes back unchanged."""
    from senselab_worker.codec import validate_message

    msg = VALID[kind]
    assert validate_message(msg, kind) is msg  # type: ignore[arg-type]
    extra = {**json.loads(json.dumps(msg)), "zzz_unknown": 1}
    validate_message(extra, kind)  # type: ignore[arg-type]


@pytest.mark.logic
@pytest.mark.parametrize(("kind", "path", "ok"), [(k, p, ok) for k, fields in FIELDS.items() for p, ok in fields])
def test_validate_message_refuses_every_wrong_type_and_missing_field(kind: str, path: tuple, ok: set) -> None:
    """Every required field, missing or of any other JSON type, is a `ProtocolError`; ints are exactly `int`."""
    from senselab_worker.codec import validate_message

    with pytest.raises(ProtocolError):
        validate_message(_delete(VALID[kind], path), kind)  # type: ignore[arg-type]
    for value in WRONG:
        if not _accepts(ok, value):
            with pytest.raises(ProtocolError):
                validate_message(_set(VALID[kind], path, value), kind)  # type: ignore[arg-type]


@pytest.mark.logic
@pytest.mark.parametrize(
    "msg",
    [
        {"id": "r", "item": "a", "outputs": {}, "error": _ERROR},
        {"id": "r", "item": "a"},
        {"id": "r", "item": "a", "error": {**_ERROR, "fault": "weird"}},
        {"id": "r", "item": "a", "error": {**_ERROR, "gap": 1}},
        {"id": "r", "item": "a", "error": {k: v for k, v in _ERROR.items() if k != "traceback"}},
        {"id": "r", "item": "a", "error": {**_ERROR, "message": 3}},
        {"id": "r", "item": "a\ud800", "outputs": {}},
        {"id": "r", "item": "x" * 1025, "outputs": {}},
    ],
)
def test_validate_message_refuses_malformed_results(msg: dict) -> None:
    """A result has exactly one of outputs and a complete error record, and ids that are short valid text."""
    from senselab_worker.codec import validate_message

    with pytest.raises(ProtocolError):
        validate_message(msg, "result")


@pytest.mark.logic
@pytest.mark.parametrize(
    "items",
    [
        [{"id": "a", "inputs": {}}, {"id": "a", "inputs": {}}],
        [{"id": "\udcff", "inputs": {}}],
        [{"id": "x" * 1025, "inputs": {}}],
    ],
)
def test_validate_message_refuses_malformed_requests(items: list) -> None:
    """Item ids are unique, valid Unicode and at most `MAX_ID_CHARS` long."""
    from senselab_worker.codec import validate_message

    with pytest.raises(ProtocolError):
        validate_message({"id": "r", "items": items, "params": {}}, "request")


@pytest.mark.logic
@pytest.mark.parametrize("done", [-1, True, 1.0])
def test_done_counts_items(done: object) -> None:
    """`done` is a non-negative `int`."""
    from senselab_worker.codec import validate_message

    with pytest.raises(ProtocolError):
        validate_message({"id": "r", "done": done}, "done")


@pytest.mark.logic
@pytest.mark.parametrize("wire", [True, 1.0, "1"])
def test_hello_wire_must_be_int(wire: object) -> None:
    """The hello's `wire` is an `int`: `True`, `1.0` and `"1"` are not wire versions."""
    from senselab_worker.codec import validate_message

    hello = {"hello": {**VALID["hello"]["hello"], "wire": wire}}
    with pytest.raises(ProtocolError):
        validate_message(hello, "hello")


@pytest.mark.logic
@pytest.mark.parametrize(
    "line",
    [
        b'{"x": 1e999}\n',
        b'{"x": -1e999}\n',
        b'{"x": [1, 2e400]}\n',
        b'{"a": 1, "a": 2}\n',
        b'{"a": {"b": 1, "b": 1}}\n',
        b'{"a": [{"k": 1, "k": 2}]}\n',
    ],
)
def test_parse_line_refuses_overflow_and_duplicate_keys(line: bytes) -> None:
    """A number that overflows to infinity and a key repeated in one object are protocol errors."""
    with pytest.raises(ProtocolError):
        parse_line(line)


@pytest.mark.logic
def test_a34_lax_wire_values_are_refused() -> None:
    """The probe: `"wire": true` and `1.0` passed the hello, `1e999` and a duplicate key passed `parse_line`."""
    from senselab_worker.codec import validate_message

    for wire in (True, 1.0):
        with pytest.raises(ProtocolError):
            validate_message({"hello": {**VALID["hello"]["hello"], "wire": wire}}, "hello")
    for line in (b'{"x": 1e999}\n', b'{"id": "a", "id": "b"}\n'):
        with pytest.raises(ProtocolError):
            parse_line(line)


def _nested(depth: int) -> object:
    value: object = {}
    for i in range(depth - 1):
        value = [value] if i % 2 else {"k": value}
    return value


@pytest.mark.logic
def test_nesting_bound_is_the_same_for_reader_writer_encoder_and_jcs(tmp_path: Path) -> None:
    """`d` nested containers pass `parse_line`, `dump_line`, `to_wire` and `jcs_dumps` exactly when d <= MAX_DEPTH."""
    from senselab_worker.canonical import jcs_dumps
    from senselab_worker.codec import MAX_DEPTH

    assert MAX_DEPTH == 64
    for depth in range(1, MAX_DEPTH + 8):
        accepted = depth <= MAX_DEPTH
        value = _nested(depth)
        line = json.dumps({"k": _nested(depth - 1)} if depth > 1 else {}).encode() + b"\n"
        result = {"id": "r", "item": "a", "outputs": {"k": _nested(depth - 2)}} if depth > 2 else None
        checks = [
            (lambda: parse_line(line), ProtocolError),
            (lambda: to_wire(value, array_dir=tmp_path, at="line"), WireValueError),
            (lambda: jcs_dumps(value), ValueError),
        ]
        if result is not None:
            checks.append((lambda: dump_line(result, "result"), ProtocolError))
        for call, error in checks:
            if accepted:
                call()
            else:
                with pytest.raises(error):
                    call()


@pytest.mark.logic
@pytest.mark.parametrize("depth", [10**3, 10**5])
def test_very_deep_lines_are_protocol_errors_not_recursion_errors(depth: int) -> None:
    """Nesting far past the bound is refused as a `ProtocolError` before any recursion limit."""
    for line in (b"[" * depth + b"]" * depth, b'{"a":' * depth + b"1" + b"}" * depth):
        with pytest.raises(ProtocolError):
            parse_line(line + b"\n")


@pytest.mark.logic
def test_to_wire_cycle_is_a_content_error(tmp_path: Path) -> None:
    """A container that contains itself is a typed content error, not a `RecursionError`; a shared child is fine."""
    cyclic_list: list = [1]
    cyclic_list.append(cyclic_list)
    cyclic_dict: dict = {}
    cyclic_dict["self"] = {"inner": cyclic_dict}
    for value in (cyclic_list, cyclic_dict, {"x": [cyclic_list]}):
        with pytest.raises(WireValueError, match="contains itself"):
            to_wire(value, array_dir=tmp_path, at="line")
    shared = [1, 2]
    assert to_wire({"a": shared, "b": [shared, shared]}, array_dir=tmp_path, at="line") == {
        "a": [1, 2],
        "b": [[1, 2], [1, 2]],
    }


def _numpy_scalars() -> list[object]:
    out = []
    for t in sorted(set(numpy.sctypeDict.values()), key=lambda t: t.__name__):
        try:
            out.append(t(0) if t is not numpy.datetime64 else t(0, "s"))
        except (TypeError, ValueError):
            continue
    return out


@pytest.mark.logic
def test_every_numpy_scalar_is_encoded_by_canonical_scalar_or_refused(tmp_path: Path) -> None:
    """`to_wire` encodes a numpy scalar exactly as `canonical_scalar` does, or refuses it as a content error."""
    from senselab_worker.canonical import canonical_scalar

    scalars = _numpy_scalars()
    assert len(scalars) > 15
    for v in scalars:
        try:
            expected = canonical_scalar(v)
        except TypeError:
            with pytest.raises(WireValueError):
                to_wire(v, array_dir=tmp_path, at="line")
            continue
        if isinstance(expected, float) and not numpy.isfinite(expected):
            continue
        got = to_wire(v, array_dir=tmp_path, at="line")
        assert got == expected and type(got) is type(expected), v


class _Str(str):
    pass


class _Dict(dict):
    pass


@pytest.mark.logic
@pytest.mark.parametrize("value", [_Str("x"), _Dict(a=1), {"k": _Str("x")}, {_Str("k"): 1}, numpy.ma.array([1.0])])
def test_to_wire_accepts_exact_types_only(tmp_path: Path, value: object) -> None:
    """Subclasses of the JSON types and of `ndarray` are refused, as by the canonical encoder."""
    with pytest.raises(WireValueError):
        to_wire(value, array_dir=tmp_path, at="line")


def _chain(height: int, leaf: object) -> object:
    value = leaf
    for i in range(height):
        value = [value] if i % 2 else {"k": value}
    return value


def _line_for(slot: str, wire: dict) -> tuple[dict, str]:
    if slot == "outputs":
        return {"id": "r", "item": "a", "outputs": wire}, "result"
    if slot == "inputs":
        return {"id": "r", "items": [{"id": "a", "inputs": wire}], "params": {}}, "request"
    if slot == "params":
        return {"id": "r", "items": [], "params": wire}, "request"
    return {"init": {**wire, "array_dir": "/tmp/x"}}, "init"


@pytest.mark.contract
@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    slot=st.sampled_from(["outputs", "inputs", "params", "init"]),
    offset=st.integers(-4, 1),
    leaf=st.sampled_from(["scalar", "empty", "array"]),
)
def test_depth_band_composes_to_wire_and_the_line(tmp_path: Path, slot: str, offset: int, leaf: str) -> None:
    """Across the depth band, a value passes `to_wire` and its whole line, or `to_wire` refuses it.

    The band runs below and past `MAX_DEPTH`. A refusal is a `WireValueError`, never a `ProtocolError`
    from `dump_line` or `parse_line`.
    """
    from senselab_worker.codec import ARRAY_REF_HEIGHT, ENVELOPE, MAX_DEPTH

    leaf_value, leaf_height = {"scalar": (0, 0), "empty": ({}, 1), "array": (numpy.zeros(2), ARRAY_REF_HEIGHT)}[leaf]
    height = MAX_DEPTH - ENVELOPE[slot] + offset
    chain_height = height - 1 - leaf_height
    value = {"k": _chain(chain_height, leaf_value)}
    fits = ENVELOPE[slot] + height <= MAX_DEPTH
    if not fits:
        with pytest.raises(WireValueError, match="deeper"):
            to_wire(value, array_dir=tmp_path, at=slot)  # type: ignore[arg-type]
        return
    wire = to_wire(value, array_dir=tmp_path, at=slot)  # type: ignore[arg-type]
    line, kind = _line_for(slot, wire)
    parsed = parse_line(dump_line(line, kind))  # type: ignore[arg-type]
    validate_message(parsed, kind)  # type: ignore[arg-type]


def _doubling(levels: int) -> object:
    value: object = 0
    for _ in range(levels):
        value = [value, value]
    return value


@pytest.mark.logic
def test_a10_exponentially_shared_value_is_refused_after_linear_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The probe: `l = [l, l]` sixty times never finished in `to_wire`; it now stops at the line budget.

    It is small in memory but 2**60 nodes as a tree. Each container is encoded once and charged per use.
    """
    import senselab_worker.codec as codec

    entries = 0
    real = codec.entered

    def counting(*args: object) -> object:
        nonlocal entries
        entries += 1
        return real(*args)  # type: ignore[arg-type]

    monkeypatch.setattr(codec, "entered", counting)
    with pytest.raises(ValueTooLarge):
        to_wire({"s": _doubling(60)}, array_dir=tmp_path, at="outputs")
    assert entries <= 61
    assert error_record(ValueTooLarge("x"))["fault"] == "resource"
    small = to_wire({"s": _doubling(4)}, array_dir=tmp_path, at="outputs")
    assert json.dumps(small) == json.dumps({"s": json.loads(json.dumps(_doubling(4)))})


@pytest.mark.logic
def test_shared_array_is_written_once(tmp_path: Path) -> None:
    """An array met many times is written once and referenced each time."""
    a = numpy.arange(3.0)
    wire = to_wire({"a": [a] * 1000}, array_dir=tmp_path, at="outputs")
    assert len({json.dumps(ref) for ref in wire["a"]}) == 1 and len(list(tmp_path.iterdir())) == 1


TREE_NODES = 5000
"""Bound on a generated message's size as a tree, so that its JSON text stays small."""


@st.composite
def _shared_dags(draw: st.DrawFn) -> tuple[dict[str, object], int, int]:
    """A result message whose containers reuse earlier ones, its height as built, and its distinct containers."""
    nodes: list[object] = [0]
    heights = [0]
    paths = [1]
    for _ in range(draw(st.integers(1, 80))):
        picks = draw(st.lists(st.integers(0, len(nodes) - 1), max_size=3)) + [len(nodes) - 1]
        if sum(paths[i] for i in picks) > TREE_NODES:
            picks = picks[-1:]
        children = [nodes[i] for i in picks]
        nodes.append(children if draw(st.booleans()) else {str(k): c for k, c in enumerate(children)})
        heights.append(1 + max(heights[i] for i in picks))
        paths.append(1 + sum(paths[i] for i in picks))
    return {"id": "r", "item": "i", "outputs": {"x": nodes[-1]}}, heights[-1] + 2, len(nodes) + 1


@pytest.mark.contract
@settings(max_examples=200, deadline=None)
@given(case=_shared_dags())
def test_depth_check_of_a_shared_dag_matches_the_tree_and_enters_each_container_once(
    case: tuple[dict[str, object], int, int],
) -> None:
    """`dump_line` refuses exactly the messages whose tree is deeper than `MAX_DEPTH`, entering each container once."""
    import senselab_worker.codec as codec

    msg, height, distinct = case
    entries = 0
    real = codec.entered

    def counting(*args: object) -> object:
        nonlocal entries
        entries += 1
        return real(*args)  # type: ignore[arg-type]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(codec, "entered", counting)
        if height > MAX_DEPTH:
            with pytest.raises(ProtocolError, match="deeper"):
                dump_line(msg, "result")
        else:
            dump_line(msg, "result")
    assert entries <= distinct
