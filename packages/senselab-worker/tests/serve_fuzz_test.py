"""`serve` is total: whatever `handle` returns or raises, every item gets one valid result line and serving goes on."""

import enum
import io
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, NoReturn

import numpy
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from senselab_worker.codec import ENVELOPE, MAX_DEPTH, MAX_LINE_BYTES, validate_message
from senselab_worker.faults import ContentFault, ResourceFault
from senselab_worker.serve import serve

REQ = "req-fuzz"
GOOD = "req-good"

Entry = Callable[[], object]


class _Escape(BaseException):
    """A `BaseException` that is neither `KeyboardInterrupt` nor `SystemExit`."""


class _Unprintable(Exception):
    def __str__(self) -> str:
        raise RuntimeError("no str")

    def __repr__(self) -> str:
        raise _Escape("no repr")


class _NotStr(ContentFault):
    def __str__(self) -> str:
        return 3  # type: ignore[return-value]


class _Str(str):
    pass


class _Colour(enum.IntEnum):
    RED = 1


class _HostileList(list):
    def __len__(self) -> int:
        raise _Escape("len")

    def __iter__(self) -> NoReturn:
        raise RuntimeError("iter")


def _nested(depth: int) -> object:
    value: object = 0
    for i in range(depth):
        value = [value] if i % 2 else {"k": value}
    return value


def _depth_expectation(outputs_height: int) -> tuple[str, str | None]:
    """An outputs value `outputs_height` containers tall fits its line iff the line's depth is within MAX_DEPTH."""
    fits = ENVELOPE["outputs"] + outputs_height <= MAX_DEPTH
    return ("outputs", None) if fits else ("error", "WireValueError")


def _shared(levels: int, as_dict: bool) -> object:
    """A value of `levels` containers, each holding its child twice: linear in memory, exponential as a tree."""
    value: object = 0
    for _ in range(levels):
        value = {"a": value, "b": value} if as_dict else [value, value]
    return {"s": value}


def _shared_expectation(levels: int) -> tuple[str, str | None]:
    if ENVELOPE["outputs"] + levels + 1 > MAX_DEPTH:
        return ("error", "WireValueError")
    return ("outputs", None) if levels <= 16 else ("error", "ResponseTooLarge")


def _cyclic_list() -> object:
    a: list = [1]
    a.append([a])
    return a


def _cyclic_dict() -> object:
    d: dict = {}
    d["self"] = d
    return d


_JSON = st.recursive(
    st.none()
    | st.booleans()
    | st.integers(-(2**63), 2**63)
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=20),
    lambda children: (
        st.lists(children, max_size=4)
        | st.dictionaries(
            st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=5), children, max_size=4
        )
    ),
    max_leaves=20,
)

_ARRAYS: list[Callable[[], object]] = [
    lambda: numpy.arange(6, dtype="<f4").reshape(2, 3),
    lambda: numpy.arange(4, dtype=">i8")[::2],
    lambda: numpy.array(["a", "é"]),
    lambda: numpy.array([object()]),
    lambda: numpy.zeros(2, dtype=[("a", "<f4")]),
    lambda: numpy.ma.array([1.0, 2.0]),
    lambda: numpy.zeros(3, dtype=numpy.longdouble),
]

_NON_JSON: list[Callable[[], object]] = [
    object,
    lambda: b"bytes",
    lambda: 1 + 2j,
    lambda: {1, 2},
    lambda: float("nan"),
    lambda: float("inf"),
    lambda: numpy.float32("nan"),
    lambda: {"__array__": 1},
    lambda: {1: "x"},
    lambda: _Str("x"),
    lambda: _Colour.RED,
    lambda: numpy.datetime64(0, "s"),
    lambda: "\ud800",
    lambda: {"\udcff": 1},
    lambda: [1, "a\udc80b"],
    _cyclic_list,
    _cyclic_dict,
]

_ERRORS: list[Callable[[], object]] = [
    lambda: ValueError("plain"),
    lambda: KeyError("k"),
    lambda: ContentFault("typed content"),
    lambda: ResourceFault("typed resource"),
    _Unprintable,
    _NotStr,
    lambda: OSError("cannot open /data/\udcff.wav"),
    lambda: RuntimeError("x" * (2 * MAX_LINE_BYTES)),
    lambda: _Escape("base"),
    lambda: GeneratorExit(),
]

_SIZES = [MAX_LINE_BYTES - 200, MAX_LINE_BYTES, MAX_LINE_BYTES + 1, 3 * MAX_LINE_BYTES]


@dataclass
class Item:
    """How `handle` answers one item, and what the answer must be when the batch is well formed."""

    make: Callable[[], dict]
    expect: tuple[str, str | None]


def _items() -> st.SearchStrategy[Item]:
    json_ok = _JSON.map(lambda v: Item(lambda: {"outputs": {"v": v}}, ("outputs", None)))
    array = st.sampled_from(_ARRAYS).map(lambda f: Item(lambda: {"outputs": {"a": f()}}, ("any", None)))
    non_json = st.sampled_from(_NON_JSON).map(
        lambda f: Item(lambda: {"outputs": {"x": f()}}, ("error", "WireValueError"))
    )
    deep = st.sampled_from([10, *range(MAX_DEPTH - 6, MAX_DEPTH + 2), 200, 5000]).map(
        lambda d: Item(lambda: {"outputs": {"x": _nested(d)}}, _depth_expectation(d + 1))
    )
    shared = st.tuples(st.sampled_from([3, 10, 16, 30, 60, 62, 63, 70]), st.booleans()).map(
        lambda kd: Item(lambda: {"outputs": _shared(*kd)}, _shared_expectation(kd[0]))
    )
    shared_arrays = st.integers(1, 2000).map(
        lambda n: Item(lambda: {"outputs": {"a": [numpy.zeros(3)] * n}}, ("outputs", None))
    )
    huge = st.sampled_from(_SIZES).map(
        lambda n: Item(
            lambda: {"outputs": {"s": "x" * n}},
            ("outputs", None) if n < MAX_LINE_BYTES - 100 else ("error", "ResponseTooLarge"),
        )
    )
    error = st.sampled_from(_ERRORS).map(lambda f: Item(lambda: {"error": f()}, ("error", None)))
    return json_ok | array | non_json | deep | shared | shared_arrays | huge | error


_MALFORMED_ENTRIES: list[Callable[[], object]] = [
    lambda: {"outputs": [1]},
    lambda: {"outputs": None},
    lambda: {"error": "not an exception"},
    lambda: {"outputs": {}, "error": ValueError()},
    lambda: {},
    lambda: None,
    lambda: ["outputs"],
]


@dataclass
class Behaviour:
    """A hostile `handle`: per-item answers, optionally broken as a whole batch."""

    items: list[Item]
    item_ids: list[str]
    batch: str
    malformed: Callable[[], object]

    @property
    def n(self) -> int:
        """The number of items in the request."""
        return len(self.item_ids)

    def handle(self, state: object, items: list[dict], params: dict) -> object:
        """Answer the hostile request as this behaviour says; answer any other well."""
        if params.get("good"):
            return [{"outputs": {"ok": True}} for _ in items]
        entries = [item.make() for item in self.items]
        if self.batch == "raise":
            raise RuntimeError("handle failed")
        if self.batch == "raise_base":
            raise _Escape("handle escaped")
        if self.batch == "none":
            return None
        if self.batch == "tuple":
            return tuple(entries)
        if self.batch == "short":
            return entries[:-1]
        if self.batch == "long":
            return [*entries, {"outputs": {}}]
        if self.batch == "hostile_list":
            return _HostileList(entries)
        if self.batch == "one_malformed" and entries:
            entries[self.malformed_at] = self.malformed()  # type: ignore[call-overload]
        return entries

    @property
    def malformed_at(self) -> int:
        """The index of the item whose entry `one_malformed` breaks."""
        return self.n // 2


@st.composite
def handle_behaviours(draw: st.DrawFn) -> Behaviour:
    """A `handle` whose outputs and errors are drawn from the hostile kinds above."""
    ids = draw(
        st.lists(
            st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=1, max_size=8),
            max_size=6,
            unique=True,
        )
    )
    items = [draw(_items()) for _ in ids]
    batch = draw(
        st.sampled_from(
            ["well", "well", "well", "raise", "raise_base", "none", "tuple", "short", "long", "hostile_list"]
            + ["one_malformed"]
        )
    )
    return Behaviour(items, ids, batch, draw(st.sampled_from(_MALFORMED_ENTRIES)))


def _frame(msg: dict) -> bytes:
    return json.dumps(msg).encode() + b"\n"


def _serve_in_process(behaviour: Behaviour, array_dir: Path) -> tuple[int, list[dict]]:
    """Run the real `serve` over in-memory streams: init, the hostile request, a good request, stop."""
    hostile = {"id": REQ, "items": [{"id": i, "inputs": {}} for i in behaviour.item_ids], "params": {}}
    good = {"id": GOOD, "items": [{"id": "g", "inputs": {}}], "params": {"good": True}}
    stream: list[dict] = [{"init": {"array_dir": str(array_dir)}}, hostile, good, {"stop": True}]
    rx = io.BytesIO(b"".join(map(_frame, stream)))
    tx = io.BytesIO()
    code = serve(lambda init: None, behaviour.handle, rx, tx)  # type: ignore[arg-type]
    lines = tx.getvalue().splitlines(keepends=True)
    assert all(len(line) <= MAX_LINE_BYTES and line.endswith(b"\n") for line in lines)
    return code, [json.loads(line) for line in lines]


@pytest.mark.contract
@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
@given(behaviour=handle_behaviours())
def test_serve_is_total_fuzz(behaviour: Behaviour) -> None:
    """Every item gets exactly one valid result line, in order, then `done`; the worker keeps serving."""
    with tempfile.TemporaryDirectory(prefix="serve-fuzz-") as scratch:
        code, out = _serve_in_process(behaviour, Path(scratch))
    assert code == 0
    assert [next(iter(m)) for m in out[:2]] == ["hello", "ready"]
    replies = out[2 : 2 + behaviour.n + 1]
    assert [r["item"] for r in replies[:-1]] == behaviour.item_ids
    assert replies[-1] == {"id": REQ, "done": behaviour.n}
    for r in replies[:-1]:
        assert ("outputs" in r) != ("error" in r)
        validate_message(r, "result")
        assert r["id"] == REQ
    if behaviour.batch in ("well", "one_malformed"):
        for i, (item, r) in enumerate(zip(behaviour.items, replies[:-1])):
            kind, type_name = item.expect
            if behaviour.batch == "one_malformed" and i == behaviour.malformed_at:
                kind, type_name = "error", "ProtocolError"
            if kind != "any":
                assert kind in r, (item.expect, r)
            if type_name is not None:
                assert r["error"]["type"] == type_name, r["error"]
    else:
        assert all("error" in r for r in replies[:-1])
    for r in replies[:-1]:
        assert "error" not in r or r["error"]["type"] != "ProtocolError" or behaviour.batch != "well"
    assert out[2 + behaviour.n + 1 :] == [{"id": GOOD, "item": "g", "outputs": {"ok": True}}, {"id": GOOD, "done": 1}]
