"""The worker serve loop, driven over in-memory streams."""

import io
import json
from pathlib import Path
from typing import Callable

import numpy
import pytest
from senselab_worker import WIRE_VERSION, runtime_identity
from senselab_worker.codec import encode
from senselab_worker.serve import serve

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
    rx = io.BytesIO(b"".join(m if isinstance(m, bytes) else encode(m, array_dir=array_dir) for m in messages))
    tx = io.BytesIO()
    code = serve(load, handle, rx, tx)
    return code, [json.loads(line) for line in tx.getvalue().splitlines()]


def _init(array_dir: Path, **extra: object) -> dict:
    return {"init": {"array_dir": str(array_dir), **extra}}


def _request(*item_ids: str, params: dict | None = None, inputs: dict | None = None) -> dict:
    items = [{"id": i, "inputs": dict(inputs or {"k": 1})} for i in item_ids]
    return {"id": "req-1", "items": items, "params": params or {}}


@pytest.mark.logic
def test_session_hello_ready_response_stop(tmp_path: Path) -> None:
    """A full session answers hello, ready and one response per request, then exits 0 on stop."""
    code, out = _session(tmp_path, [_init(tmp_path), _request("a", "b"), {"stop": True}, _request("c")])
    hello, ready, response = out
    assert code == 0
    assert hello["hello"]["wire"] == WIRE_VERSION and hello["hello"]["runtime"] == runtime_identity()
    assert {"python", "numpy", "pid"} <= hello["hello"].keys()
    assert ready["ready"]["load_s"] >= 0
    assert response == {
        "id": "req-1",
        "results": [
            {"id": "a", "outputs": {"seen": ["k"], "state": "loaded"}},
            {"id": "b", "outputs": {"seen": ["k"], "state": "loaded"}},
        ],
    }


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
    assert code == 0 and out[2]["results"][0]["outputs"]["state"] == "t"


def _fatal(out: list[dict]) -> dict:
    last = out[-1]
    assert last["fatal"] is True
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
    ],
)
def test_unusable_lines_are_fatal_and_exit_nonzero(tmp_path: Path, messages: list[object]) -> None:
    """A line the worker cannot act on is answered with `fatal` and a non-zero exit."""
    lines = [_init(tmp_path) if m == "INIT" else m for m in messages]
    code, out = _session(tmp_path, lines)  # type: ignore[arg-type]
    error = _fatal(out)
    assert code != 0 and (error["type"], error["fault"]) == ("ProtocolError", "infrastructure")


@pytest.mark.logic
def test_load_failure_is_fatal(tmp_path: Path) -> None:
    """An exception in `load` is reported as `fatal` with its classification."""
    code, out = _session(tmp_path, [_init(tmp_path), _request("a")], load=_raise_load)
    error = _fatal(out)
    assert code != 0 and error["type"] == "KeyError" and error["fault"] == "content"
    assert len(out) == 2


@pytest.mark.logic
def test_handle_raising_fails_every_item(tmp_path: Path) -> None:
    """If `handle` raises, every item gets that error."""

    def handle(state: object, items: list[dict], params: dict) -> list[dict]:
        raise MemoryError("full")

    _, out = _session(tmp_path, [_init(tmp_path), _request("a", "b")], handle=handle)
    errors = [r["error"] for r in out[2]["results"]]
    assert [e["type"] for e in errors] == ["MemoryError"] * 2 and {e["fault"] for e in errors} == {"resource"}


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
    ],
)
def test_malformed_handle_result_is_protocol_error_for_every_item(
    tmp_path: Path, bad: Callable[[list[dict]], object]
) -> None:
    """A wrong count or a malformed entry gives every item an infrastructure `ProtocolError`."""
    _, out = _session(tmp_path, [_init(tmp_path), _request("a", "b")], handle=lambda s, items, p: bad(items))  # type: ignore[arg-type, return-value]
    errors = [r["error"] for r in out[2]["results"]]
    assert {(e["type"], e["fault"], e["gap"]) for e in errors} == {("ProtocolError", "infrastructure", False)}


@pytest.mark.logic
def test_per_item_errors_and_output_encoding_failures_stay_with_their_item(tmp_path: Path) -> None:
    """An item's own error, or outputs with no wire form, fail only that item."""

    def handle(state: object, items: list[dict], params: dict) -> list[dict]:
        by_id: dict[str, dict] = {
            "ok": {"outputs": {"x": numpy.arange(3, dtype="<i2")}},
            "err": {"error": TypeError("bad input")},
            "obj": {"outputs": {"x": numpy.array([object()])}},
            "nan": {"outputs": {"x": float("nan")}},
            "odd": {"error": RuntimeError("odd")},
        }
        return [by_id[item["id"]] for item in items]

    _, out = _session(tmp_path, [_init(tmp_path), _request("ok", "err", "obj", "nan", "odd")], handle=handle)
    results = out[2]["results"]
    assert [r["id"] for r in results] == ["ok", "err", "obj", "nan", "odd"]
    assert results[0]["outputs"]["x"]["__array__"]["dtype"] == "<i2"
    assert [(r["error"]["type"], r["error"]["fault"], r["error"]["gap"]) for r in results[1:]] == [
        ("TypeError", "content", False),
        ("ValueError", "content", False),
        ("ValueError", "content", False),
        ("RuntimeError", "infrastructure", True),
    ]


def _damaged_ref(tmp_path: Path) -> dict:
    wire = json.loads(encode({"x": numpy.zeros(2)}, array_dir=tmp_path))["x"]
    wire["__array__"]["sha256"] = "f" * 64
    return wire


@pytest.mark.logic
def test_bad_array_in_an_item_fails_only_that_item_before_handle(tmp_path: Path) -> None:
    """An item whose array fails integrity checks is not passed to `handle`."""
    seen: list[str] = []

    def handle(state: object, items: list[dict], params: dict) -> list[dict]:
        seen.extend(item["id"] for item in items)
        return [{"outputs": {"sum": float(item["inputs"]["x"].sum())}} for item in items]

    good = json.loads(encode({"x": numpy.ones(2)}, array_dir=tmp_path))["x"]
    request = {
        "id": "r",
        "items": [{"id": "a", "inputs": {"x": good}}, {"id": "b", "inputs": {"x": _damaged_ref(tmp_path)}}],
        "params": {},
    }
    _, out = _session(tmp_path, [_init(tmp_path), json.dumps(request).encode() + b"\n"], handle=handle)
    a, b = out[2]["results"]
    assert seen == ["a"] and a["outputs"] == {"sum": 2.0}
    assert (b["error"]["type"], b["error"]["fault"]) == ("ArrayIntegrityError", "infrastructure")


@pytest.mark.logic
def test_bad_array_in_params_fails_every_item(tmp_path: Path) -> None:
    """An integrity failure in `params` fails the whole request."""
    items = [{"id": "a", "inputs": {}}, {"id": "b", "inputs": {}}]
    request = {"id": "r", "items": items, "params": {"w": _damaged_ref(tmp_path)}}
    _, out = _session(tmp_path, [_init(tmp_path), json.dumps(request).encode() + b"\n"])
    assert {r["error"]["type"] for r in out[2]["results"]} == {"ArrayIntegrityError"}


@pytest.mark.logic
def test_bad_array_in_init_is_fatal(tmp_path: Path) -> None:
    """An integrity failure in `init` is fatal."""
    init = {"init": {"array_dir": str(tmp_path), "w": _damaged_ref(tmp_path)}}
    code, out = _session(tmp_path, [json.dumps(init).encode() + b"\n"])
    assert code != 0 and _fatal(out)["type"] == "ArrayIntegrityError"


@pytest.mark.logic
def test_empty_request_gets_empty_results(tmp_path: Path) -> None:
    """A request with no items is answered with no results."""
    _, out = _session(tmp_path, [_init(tmp_path), {"id": "r", "items": [], "params": {}}])
    assert out[2] == {"id": "r", "results": []}
