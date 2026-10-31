"""The worker serve loop, driven over in-memory streams."""

import io
import json
from pathlib import Path
from typing import Callable, NoReturn

import numpy
import pytest
from senselab_worker import WIRE_VERSION, runtime_identity
from senselab_worker.codec import ARRAY_REF_HEIGHT, ENVELOPE, MAX_DEPTH, MAX_LINE_BYTES, to_wire, validate_message
from senselab_worker.faults import ContentFault
from senselab_worker.serve import EXIT_FATAL, serve

Handle = Callable[[object, list[dict], dict], list[dict]]


def _echo(state: object, items: list[dict], params: dict) -> list[dict]:
    return [{"outputs": {"seen": sorted(item["inputs"]), "state": state}} for item in items]


def _load(init: dict) -> object:
    return init.get("tag", "loaded")


def _session(
    array_dir: Path,
    messages: list[dict | bytes],
    *,
    load: Callable[[dict], object] = _load,
    handle: Handle = _echo,
) -> tuple[int, list[dict]]:
    rx = io.BytesIO(b"".join(m if isinstance(m, bytes) else _line(m, array_dir) for m in messages))
    tx = io.BytesIO()
    code = serve(load, handle, rx, tx)
    return code, [json.loads(line) for line in tx.getvalue().splitlines()]


def _line(msg: dict, array_dir: Path) -> bytes:
    """Frame `msg` without validating it, so tests can send what a host never would."""
    return json.dumps(to_wire(msg, array_dir=array_dir, at="line")).encode() + b"\n"


def _replies(out: list[dict], request_id: str = "req-1") -> list[dict]:
    """The lines answering one request: one per item, then `done`, each valid for its kind."""
    start = next(i for i, m in enumerate(out) if m.get("id") == request_id)
    end = next(i for i in range(start, len(out)) if "done" in out[i])
    for m in out[start:end]:
        validate_message(m, "result")
    validate_message(out[end], "done")
    assert out[end] == {"id": request_id, "done": end - start}
    return out[start:end]


def _init(array_dir: Path, **extra: object) -> dict:
    return {"init": {"array_dir": str(array_dir), **extra}}


def _request(*item_ids: str, params: dict | None = None, inputs: dict | None = None) -> dict:
    items = [{"id": i, "inputs": dict(inputs or {"k": 1})} for i in item_ids]
    return {"id": "req-1", "items": items, "params": params or {}}


@pytest.mark.logic
def test_session_hello_ready_response_stop(tmp_path: Path) -> None:
    """A full session answers hello, ready and one response per request, then exits 0 on stop."""
    code, out = _session(tmp_path, [_init(tmp_path), _request("a", "b"), {"stop": True}, _request("c")])
    hello, ready, *response = out
    assert code == 0 and WIRE_VERSION == 2
    validate_message(hello, "hello")
    validate_message(ready, "ready")
    assert hello["hello"]["wire"] == WIRE_VERSION and hello["hello"]["runtime"] == runtime_identity()
    assert ready["ready"]["load_s"] >= 0
    assert response == [
        {"id": "req-1", "item": "a", "outputs": {"seen": ["k"], "state": "loaded"}},
        {"id": "req-1", "item": "b", "outputs": {"seen": ["k"], "state": "loaded"}},
        {"id": "req-1", "done": 2},
    ]


@pytest.mark.logic
@pytest.mark.parametrize("ending", [[], [{"stop": True}]])
def test_stop_or_eof_before_init_exits_zero_without_reply(tmp_path: Path, ending: list[dict]) -> None:
    """Stop or EOF before init exits 0 after the hello alone."""
    code, out = _session(tmp_path, list(ending))
    assert code == 0 and [next(iter(m)) for m in out] == ["hello"]


@pytest.mark.logic
def test_eof_after_init_exits_zero(tmp_path: Path) -> None:
    """EOF on stdin is a stop."""
    code, out = _session(tmp_path, [_init(tmp_path)])
    assert code == 0 and [next(iter(m)) for m in out] == ["hello", "ready"]


@pytest.mark.logic
def test_unknown_keys_in_known_messages_are_ignored(tmp_path: Path) -> None:
    """Extra keys inside init, a request or stop change nothing."""
    request = {**_request("a"), "extra": 1}
    code, out = _session(tmp_path, [_init(tmp_path, tag="t"), request, {"stop": True, "why": "done"}])
    assert code == 0 and _replies(out)[0]["outputs"]["state"] == "t"


def _fatal(out: list[dict]) -> dict:
    last = out[-1]
    validate_message(last, "fatal")
    return last["error"]


def _raise_load(init: dict) -> object:
    raise KeyError("model")


@pytest.mark.logic
@pytest.mark.parametrize(
    "messages",
    [
        [{"init": {}}],
        [{"init": {"array_dir": 1}}],
        [_request("a")],
        [b"[]\n"],
        ["INIT", b"{oops\n"],
        ["INIT", {"what": 1}],
        ["INIT", {"id": "r", "items": [{"id": "a", "inputs": {}}, {"id": "a", "inputs": {}}], "params": {}}],
        ["INIT", {"id": "r", "items": [{"id": "a"}], "params": {}}],
        ["INIT", {"id": "r", "items": [{"id": 1, "inputs": {}}], "params": {}}],
        ["INIT", {"id": 3, "items": [], "params": {}}],
        ["INIT", {"id": "r", "items": {}, "params": {}}],
        ["INIT", {"id": "r", "items": [], "params": []}],
        ["INIT", {"id": "r", "items": []}],
        [{"stop": False}],
        ["INIT", {"stop": False}],
        ["INIT", {"stop": 1}],
        ["INIT", {"stop": "true"}],
        ["INIT", {"stop": None, "id": "r", "items": [], "params": {}}],
        ["INIT", b'{"id": "r", "id": "s", "items": [], "params": {}}\n'],
        ["INIT", b'{"id": "r", "items": [], "params": {"w": 1e999}}\n'],
        ["INIT", b'{"id": "r", "items": [], "params": ' + b"[" * 100 + b"]" * 100 + b"}\n"],
        ["INIT", {"id": "r", "items": [{"id": "a" * 1025, "inputs": {}}], "params": {}}],
    ],
)
def test_unusable_lines_are_fatal_and_exit_nonzero(tmp_path: Path, messages: list[object]) -> None:
    """A line the worker cannot act on is answered with `fatal` and a non-zero exit."""
    lines = [_init(tmp_path) if m == "INIT" else m for m in messages]
    code, out = _session(tmp_path, lines)  # type: ignore[arg-type]
    error = _fatal(out)
    assert code == EXIT_FATAL and (error["type"], error["fault"], error["gap"]) == (
        "ProtocolError",
        "infrastructure",
        False,
    )


@pytest.mark.logic
@pytest.mark.parametrize("stop", [False, None, 0, 1, "true", [], {}])
def test_stop_not_exactly_true_is_a_fatal_protocol_error_and_exits_fatal(tmp_path: Path, stop: object) -> None:
    """A message whose `stop` is not exactly `true` is answered with a fatal `ProtocolError`, then `EXIT_FATAL`.

    The probe: `{"stop": false, ...}` on a request made the worker exit 0 without a reply.
    """
    request = {**_request("a"), "stop": stop}
    code, out = _session(tmp_path, [_init(tmp_path), json.dumps(request).encode() + b"\n"])
    error = _fatal(out)
    assert code == EXIT_FATAL and error["type"] == "ProtocolError" and "stop" in error["message"]


@pytest.mark.logic
def test_load_failure_is_fatal(tmp_path: Path) -> None:
    """An exception in `load` is reported as `fatal` with its classification."""
    code, out = _session(tmp_path, [_init(tmp_path), _request("a")], load=_raise_load)
    error = _fatal(out)
    assert code == EXIT_FATAL and (error["type"], error["fault"], error["gap"]) == ("KeyError", "content", True)
    assert len(out) == 2


@pytest.mark.logic
def test_handle_raising_fails_every_item(tmp_path: Path) -> None:
    """If `handle` raises, every item gets that error."""

    def handle(state: object, items: list[dict], params: dict) -> list[dict]:
        raise MemoryError("full")

    _, out = _session(tmp_path, [_init(tmp_path), _request("a", "b")], handle=handle)
    errors = [r["error"] for r in _replies(out)]
    assert [e["type"] for e in errors] == ["MemoryError"] * 2 and {e["fault"] for e in errors} == {"resource"}


class _Escape(BaseException):
    """A `BaseException` that is neither `KeyboardInterrupt` nor `SystemExit`."""


@pytest.mark.logic
@pytest.mark.parametrize("raised", [_Escape("out"), GeneratorExit(), StopIteration()])
def test_handle_raising_a_base_exception_fails_every_item_and_serving_continues(
    tmp_path: Path, raised: BaseException
) -> None:
    """A `BaseException` from `handle`, other than `KeyboardInterrupt` and `SystemExit`, is every item's error."""

    def handle(state: object, items: list[dict], params: dict) -> list[dict]:
        if params.get("raise"):
            raise raised
        return _echo(state, items, params)

    first = {**_request("a", "b"), "params": {"raise": True}}
    second = {**_request("c"), "id": "req-2"}
    code, out = _session(tmp_path, [_init(tmp_path), first, second, {"stop": True}], handle=handle)
    assert code == 0
    assert [r["error"]["type"] for r in _replies(out)] == [type(raised).__name__] * 2
    assert "outputs" in _replies(out, "req-2")[0]


@pytest.mark.logic
@pytest.mark.parametrize("raised", [KeyboardInterrupt(), SystemExit(3)])
def test_keyboard_interrupt_and_system_exit_from_handle_end_serving(tmp_path: Path, raised: BaseException) -> None:
    """`KeyboardInterrupt` and `SystemExit` are not item errors: they leave `serve`."""

    def handle(state: object, items: list[dict], params: dict) -> list[dict]:
        raise raised

    with pytest.raises(type(raised)):
        _session(tmp_path, [_init(tmp_path), _request("a")], handle=handle)


class _List(list):
    def __len__(self) -> int:
        raise RuntimeError("hostile len")


class _Dict(dict):
    def keys(self) -> NoReturn:  # type: ignore[override]
        raise RuntimeError("hostile keys")


@pytest.mark.logic
@pytest.mark.parametrize(
    "bad",
    [
        lambda items: [],
        lambda items: [{"outputs": {}}] * (len(items) + 1),
        lambda items: [{"outputs": {}, "error": ValueError()} for _ in items],
        lambda items: [{} for _ in items],
        lambda items: [{"error": "not an exception"} for _ in items],
        lambda items: [{"outputs": [1]} for _ in items],
        lambda items: None,
        lambda items: tuple({"outputs": {}} for _ in items),
        lambda items: _List({"outputs": {}} for _ in items),
        lambda items: [_Dict(outputs={}) for _ in items],
        lambda items: [{"outputs": _Dict()} for _ in items],
    ],
)
def test_malformed_handle_result_is_protocol_error_for_every_item(
    tmp_path: Path, bad: Callable[[list[dict]], object]
) -> None:
    """A result that is not a list of one entry per item, or whose every entry is malformed, fails every item.

    Each item gets an infrastructure `ProtocolError`.
    """
    _, out = _session(tmp_path, [_init(tmp_path), _request("a", "b")], handle=lambda s, items, p: bad(items))  # type: ignore[arg-type, return-value]
    errors = [r["error"] for r in _replies(out)]
    assert {(e["type"], e["fault"], e["gap"]) for e in errors} == {("ProtocolError", "infrastructure", False)}


@pytest.mark.logic
def test_per_item_errors_and_output_encoding_failures_stay_with_their_item(tmp_path: Path) -> None:
    """An item's own error, or outputs with no wire form, fail only that item."""

    def handle(state: object, items: list[dict], params: dict) -> list[dict]:
        by_id: dict[str, dict] = {
            "ok": {"outputs": {"x": numpy.arange(3, dtype="<i2")}},
            "err": {"error": TypeError("bad input")},
            "typed": {"error": ContentFault("bad input")},
            "obj": {"outputs": {"x": numpy.array([object()])}},
            "nan": {"outputs": {"x": float("nan")}},
            "odd": {"error": RuntimeError("odd")},
        }
        return [by_id[item["id"]] for item in items]

    ids = ["ok", "err", "typed", "obj", "nan", "odd"]
    _, out = _session(tmp_path, [_init(tmp_path), _request(*ids)], handle=handle)
    results = _replies(out)
    assert [r["item"] for r in results] == ids
    assert results[0]["outputs"]["x"]["__array__"]["dtype"] == "<i2"
    assert [(r["error"]["type"], r["error"]["fault"], r["error"]["gap"]) for r in results[1:]] == [
        ("TypeError", "content", True),
        ("ContentFault", "content", False),
        ("WireValueError", "content", False),
        ("WireValueError", "content", False),
        ("RuntimeError", "infrastructure", True),
    ]


def _damaged_ref(tmp_path: Path) -> dict:
    wire = to_wire(numpy.zeros(2), array_dir=tmp_path, at="line")
    wire["__array__"]["sha256"] = "f" * 64
    return wire


@pytest.mark.logic
def test_bad_array_in_an_item_fails_only_that_item_before_handle(tmp_path: Path) -> None:
    """An item whose array fails integrity checks is not passed to `handle`."""
    seen: list[str] = []

    def handle(state: object, items: list[dict], params: dict) -> list[dict]:
        seen.extend(item["id"] for item in items)
        return [{"outputs": {"sum": float(item["inputs"]["x"].sum())}} for item in items]

    good = to_wire(numpy.ones(2), array_dir=tmp_path, at="line")
    request = {
        "id": "r",
        "items": [{"id": "a", "inputs": {"x": good}}, {"id": "b", "inputs": {"x": _damaged_ref(tmp_path)}}],
        "params": {},
    }
    _, out = _session(tmp_path, [_init(tmp_path), json.dumps(request).encode() + b"\n"], handle=handle)
    a, b = _replies(out, "r")
    assert seen == ["a"] and a["outputs"] == {"sum": 2.0}
    assert (b["error"]["type"], b["error"]["fault"]) == ("ArrayIntegrityError", "infrastructure")


@pytest.mark.logic
def test_bad_array_in_params_fails_every_item(tmp_path: Path) -> None:
    """An integrity failure in `params` fails the whole request."""
    items = [{"id": "a", "inputs": {}}, {"id": "b", "inputs": {}}]
    request = {"id": "r", "items": items, "params": {"w": _damaged_ref(tmp_path)}}
    _, out = _session(tmp_path, [_init(tmp_path), json.dumps(request).encode() + b"\n"])
    assert {r["error"]["type"] for r in _replies(out, "r")} == {"ArrayIntegrityError"}


@pytest.mark.logic
def test_bad_array_in_init_is_fatal(tmp_path: Path) -> None:
    """An integrity failure in `init` is fatal."""
    init = {"init": {"array_dir": str(tmp_path), "w": _damaged_ref(tmp_path)}}
    code, out = _session(tmp_path, [json.dumps(init).encode() + b"\n"])
    assert code != 0 and _fatal(out)["type"] == "ArrayIntegrityError"


@pytest.mark.logic
def test_empty_request_gets_only_done(tmp_path: Path) -> None:
    """A request with no items is answered by `done` alone."""
    _, out = _session(tmp_path, [_init(tmp_path), {"id": "r", "items": [], "params": {}}])
    assert out[2:] == [{"id": "r", "done": 0}]


def _sized(n_bytes: int) -> dict:
    return {"outputs": {"blob": "x" * n_bytes}}


@pytest.mark.logic
def test_one_oversize_item_does_not_fail_batch_mates(tmp_path: Path) -> None:
    """The line budget is per item: an item over it is that item's resource error, and its batch-mates keep outputs."""
    sizes = {f"i{n}": 2**20 for n in range(20)} | {"big": MAX_LINE_BYTES, "edge": MAX_LINE_BYTES - 200}

    def handle(state: object, items: list[dict], params: dict) -> list[dict]:
        return [_sized(sizes[item["id"]]) for item in items]

    ids = list(sizes)
    code, out = _session(tmp_path, [_init(tmp_path), _request(*ids), {"stop": True}], handle=handle)
    results = _replies(out)
    assert code == 0 and [r["item"] for r in results] == ids
    for r in results:
        line = len(json.dumps(r, ensure_ascii=False, separators=(",", ":")).encode()) + 1
        assert line <= MAX_LINE_BYTES
        if r["item"] == "big":
            assert (r["error"]["type"], r["error"]["fault"], r["error"]["gap"]) == (
                "ResponseTooLarge",
                "resource",
                False,
            )
        else:
            assert len(r["outputs"]["blob"]) == sizes[r["item"]]


@pytest.mark.logic
def test_a10_aggregate_reply_over_the_budget_fails_no_item(tmp_path: Path) -> None:
    """The probe: 20 items of 1 MB each (20 MB in all) killed the worker; now every item gets its outputs."""

    def handle(state: object, items: list[dict], params: dict) -> list[dict]:
        return [_sized(10**6) for _ in items]

    code, out = _session(tmp_path, [_init(tmp_path), _request(*map(str, range(20))), {"stop": True}], handle=handle)
    assert code == 0 and all(len(r["outputs"]["blob"]) == 10**6 for r in _replies(out))


class _Unprintable(Exception):
    def __str__(self) -> str:
        raise RuntimeError("no str")

    def __repr__(self) -> str:
        raise RuntimeError("no repr")


@pytest.mark.logic
def test_a10_unprintable_or_surrogate_error_fails_only_its_item(tmp_path: Path) -> None:
    """The probe: an error whose `__str__` raises, or whose message holds a lone surrogate, killed the worker."""

    def handle(state: object, items: list[dict], params: dict) -> list[dict]:
        errors = {"str": _Unprintable(), "surrogate": OSError("cannot open /data/\udcff.wav")}
        return [{"error": errors[i["id"]]} if i["id"] in errors else {"outputs": {"ok": 1}} for i in items]

    second = {**_request("after"), "id": "req-2"}
    code, out = _session(tmp_path, [_init(tmp_path), _request("str", "ok", "surrogate"), second], handle=handle)
    first = _replies(out)
    assert code == 0 and first[1]["outputs"] == {"ok": 1}
    assert first[0]["error"]["message"] == "<unprintable _Unprintable>"
    assert first[2]["error"]["message"] == "cannot open /data/\\udcff.wav"
    assert _replies(out, "req-2")[0]["outputs"] == {"ok": 1}


@pytest.mark.logic
@pytest.mark.parametrize(
    "malformed", [{"outputs": [1]}, {"error": "text"}, {}, None, {"outputs": {}, "error": ValueError()}, ["outputs"]]
)
def test_one_malformed_entry_fails_only_its_item(tmp_path: Path, malformed: object) -> None:
    """In a list of the right length, a malformed entry is that item's `ProtocolError`; its batch-mates keep theirs."""

    def handle(state: object, items: list[dict], params: dict) -> list[object]:
        return [malformed if item["id"] == "b" else {"outputs": {"id": item["id"]}} for item in items]

    _, out = _session(tmp_path, [_init(tmp_path), _request("a", "b", "c")], handle=handle)  # type: ignore[arg-type]
    a, b, c = _replies(out)
    assert (a["outputs"], c["outputs"]) == ({"id": "a"}, {"id": "c"})
    assert (b["error"]["type"], b["error"]["fault"], b["error"]["gap"]) == ("ProtocolError", "infrastructure", False)


def _chain(height: int, leaf: object) -> object:
    value = leaf
    for i in range(height):
        value = [value] if i % 2 else {"k": value}
    return value


@pytest.mark.contract
@pytest.mark.parametrize("leaf", ["scalar", "empty", "array"])
def test_output_depth_band_is_outputs_or_a_content_fault(tmp_path: Path, leaf: str) -> None:
    """Outputs across the depth band are sent, or are that item's `WireValueError`, never a `ProtocolError`.

    The band runs from MAX_DEPTH - 4 to MAX_DEPTH + 1 on the result line; the batch-mate is unaffected.
    """
    leaf_value, leaf_height = {"scalar": (0, 0), "empty": ({}, 1), "array": (numpy.zeros(2), ARRAY_REF_HEIGHT)}[leaf]
    band = range(MAX_DEPTH - ENVELOPE["outputs"] - 4, MAX_DEPTH - ENVELOPE["outputs"] + 2)

    def handle(state: object, items: list[dict], params: dict) -> list[dict]:
        heights = {f"h{h}": h for h in band}
        return [
            {"outputs": {"k": _chain(heights[i["id"]] - 1 - leaf_height, leaf_value)}}
            if i["id"] in heights
            else {"outputs": {"ok": 1}}
            for i in items
        ]

    ids = [f"h{h}" for h in band] + ["mate"]
    code, out = _session(tmp_path, [_init(tmp_path), _request(*ids), {"stop": True}], handle=handle)
    replies = _replies(out)
    assert code == 0 and replies[-1]["outputs"] == {"ok": 1}
    for h, r in zip(band, replies):
        if ENVELOPE["outputs"] + h <= MAX_DEPTH:
            assert "outputs" in r, (h, r)
        else:
            assert (r["error"]["type"], r["error"]["fault"], r["error"]["gap"]) == ("WireValueError", "content", False)


@pytest.mark.logic
def test_a10_exponentially_shared_output_fails_only_its_item(tmp_path: Path) -> None:
    """The probe: an output `l = [l, l]` sixty times hung the worker. It is now that item's `ResponseTooLarge`."""

    def handle(state: object, items: list[dict], params: dict) -> list[dict]:
        value: object = 0
        for _ in range(60):
            value = [value, value]
        return [{"outputs": {"s": value}} if i["id"] == "dag" else {"outputs": {"ok": 1}} for i in items]

    code, out = _session(tmp_path, [_init(tmp_path), _request("dag", "mate"), {"stop": True}], handle=handle)
    dag, mate = _replies(out)
    assert code == 0 and mate["outputs"] == {"ok": 1}
    assert (dag["error"]["type"], dag["error"]["fault"], dag["error"]["gap"]) == ("ResponseTooLarge", "resource", False)
