"""The host side of the wire protocol, against the fake worker and a scripted one."""

import errno
import gc
import hashlib
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy
import pytest
import senselab_worker
from backends.subdir_backend_fixture import ARRAY_LIKE
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from lifecycle_probes_fixture import alive, gate, group_alive, interpreter, listing, recorded_pids, state
from senselab_core.kernel.declare import Env
from senselab_core.runtime import worker_client
from senselab_core.runtime.fake_backend import FAKE as FAKE_SPEC
from senselab_core.runtime.host_worker import HostWorker
from senselab_core.runtime.worker_client import (
    DEFAULT_GRACES,
    ConcurrentUse,
    ProtocolMismatch,
    ShutdownGraces,
    WorkerClient,
    WorkerFatal,
    WorkerLost,
    WorkerTimeout,
)
from senselab_core.testing.interpreter import source_tree_python
from senselab_worker import codec, runtime_identity
from senselab_worker.canonical import MAX_DEPTH, canonical_npy_bytes
from senselab_worker.codec import ARRAY_REF_HEIGHT, ENVELOPE, MAX_LINE_BYTES, LineTooLarge, to_wire
from senselab_worker.digest import identity_digest
from senselab_worker.fake import EXIT_CRASH, FAULT_KEY
from senselab_worker.faults import FaultClass, ProtocolError, ValueTooLarge, classify, error_record
from senselab_worker.model_cache import ModelCache

FAKE = "senselab_worker.fake"
SCRIPTED = Path(__file__).parent / "scripted_worker_fixture.py"
posix_only = pytest.mark.skipif(os.name == "nt", reason="stands in for the interpreter with a shell script")

PYTHON = source_tree_python()


@pytest.fixture(autouse=True)
def no_leaked_processes(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[subprocess.Popen[bytes]]]:
    """Fail a test that leaves a worker process running or unreaped.

    Records every ``Popen`` the client creates and asserts each has been waited on
    (``returncode`` is set only by ``poll``/``wait``), so a zombie counts as a leak and
    children spawned by other tests, such as multiprocessing helpers, are ignored.
    """
    spawned: list[subprocess.Popen[bytes]] = []
    real = subprocess.Popen

    class _Recording(real):  # type: ignore[valid-type, misc]
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            spawned.append(self)

    monkeypatch.setattr(subprocess, "Popen", _Recording)
    yield spawned
    monkeypatch.undo()
    leaked = [p.pid for p in spawned if p.returncode is None]
    for p in spawned:
        if p.returncode is None:
            p.kill()
            p.wait()
    assert leaked == []


@pytest.fixture
def array_parent(tmp_path: Path) -> Iterator[Path]:
    """A parent for the client's array directory, which must hold nothing after the test."""
    path = tmp_path / "arrays"
    path.mkdir()
    yield path
    assert list(path.iterdir()) == []


@pytest.fixture
def worker(array_parent: Path) -> Iterator[WorkerClient]:
    """A client over the fake worker."""
    client = WorkerClient(PYTHON, FAKE, init={}, array_parent=array_parent)
    yield client
    client.close()
    assert not client.alive


SMALL_LINE_BYTES = 2048
"""The line limit of `small_line_worker`, below the worker's own."""


@pytest.fixture
def small_line_worker(array_parent: Path) -> Iterator[WorkerClient]:
    """A client over the fake worker whose lines are limited to `SMALL_LINE_BYTES`."""
    client = WorkerClient(PYTHON, FAKE, init={}, array_parent=array_parent, max_line_bytes=SMALL_LINE_BYTES)
    yield client
    client.close()
    assert not client.alive


def _lines(reply: dict[str, Any] | list[Any]) -> list[str]:
    return [
        part if isinstance(part, str) else json.dumps(part) for part in (reply if isinstance(reply, list) else [reply])
    ]


def _scripted(
    array_parent: Path, *replies: dict[str, Any] | list[Any], max_line_bytes: int = MAX_LINE_BYTES, **init: object
) -> WorkerClient:
    """A client over the scripted worker; each reply is one line (a dict) or a list of lines."""
    python = interpreter(array_parent.parent, f'exec "{sys.executable}" "{SCRIPTED}"')
    return WorkerClient(
        python,
        "unused",
        init={"replies": [_lines(r) for r in replies], **init},
        array_parent=array_parent,
        max_line_bytes=max_line_bytes,
    )


def _value(inputs: dict[str, Any], params: dict[str, Any]) -> str:
    return identity_digest("fake-output@1", {"inputs": inputs, "params": params})


@pytest.mark.logic
def test_one_item_failure_isolated(worker: WorkerClient) -> None:
    """One failing item leaves its batch-mates' outputs intact."""
    out = worker.run([{"id": "a", "inputs": {}}, {"id": "b", "inputs": {}}], {FAULT_KEY: {"fail_items": ["b"]}})
    assert out[0] == {"id": "a", "outputs": {"value": _value({}, {})}}
    assert out[1]["error"]["fault"] == "content" and out[1]["error"]["fault"] is FaultClass.CONTENT


@pytest.mark.logic
def test_oom_is_resource_fault(worker: WorkerClient) -> None:
    """An out-of-memory error in one item is a resource fault for that item."""
    out = worker.run([{"id": "a", "inputs": {}}, {"id": "b", "inputs": {}}], {FAULT_KEY: {"oom_items": ["a"]}})
    assert (out[0]["error"]["type"], out[0]["error"]["fault"]) == ("OutOfMemoryError", FaultClass.RESOURCE)
    assert "outputs" in out[1]


@pytest.mark.logic
def test_arrays_travel_and_request_files_are_removed(worker: WorkerClient) -> None:
    """An input array reaches the worker, and its file is deleted once the response is read."""
    a = numpy.linspace(0, 1, 32, dtype="<f4")
    [out] = worker.run([{"id": "a", "inputs": {"x": a}}], {"k": 1})
    assert out["outputs"]["value"] == _value({"x": hashlib.sha256(canonical_npy_bytes(a)).hexdigest()}, {"k": 1})
    assert list(worker.array_dir.iterdir()) == []


@pytest.mark.logic
def test_unencodable_item_fails_alone_on_the_host(worker: WorkerClient) -> None:
    """An item with an object array fails as content before sending; its batch-mates run."""
    items = [{"id": "a", "inputs": {"x": numpy.array([object()])}}, {"id": "b", "inputs": {}}]
    out = worker.run(items, {})
    assert (out[0]["error"]["type"], out[0]["error"]["fault"]) == ("WireValueError", FaultClass.CONTENT)
    assert out[0]["error"]["gap"] is False and out[1]["outputs"]["value"] == _value({}, {})


def _oversized_ints() -> list[object]:
    big = 10 ** sys.get_int_max_str_digits()
    return [big, -big, [1, big], {"k": {"j": big}}, (big,)]


@pytest.mark.logic
@pytest.mark.parametrize("value", _oversized_ints(), ids=["pos", "neg", "list", "nested", "tuple"])
def test_an_oversized_int_in_one_items_inputs_fails_only_that_item(worker: WorkerClient, value: object) -> None:
    """An int past the reader's digit limit is that item's content `WireValueError`; its batch-mates run."""
    out = worker.run([{"id": "a", "inputs": {"n": value}}, {"id": "b", "inputs": {"n": 1}}], {})
    assert (out[0]["error"]["type"], out[0]["error"]["fault"], out[0]["error"]["gap"]) == (
        "WireValueError",
        FaultClass.CONTENT,
        False,
    )
    assert out[1]["outputs"]["value"] == _value({"n": 1}, {})
    with pytest.raises(ValueError):
        worker.run([{"id": "a", "inputs": {}}], {"n": value})
    assert worker.run([{"id": "a", "inputs": {}}], {})[0]["outputs"]


_GRACE_FIELDS = ("stop", "term", "kill", "join", "exit_code_wait")
_BAD_GRACES = [float("nan"), float("inf"), float("-inf"), -1.0, -1e-9, 1e20, threading.TIMEOUT_MAX * 2, True, "1", None]


@pytest.mark.logic
@pytest.mark.parametrize("field", _GRACE_FIELDS)
@pytest.mark.parametrize("bad", _BAD_GRACES, ids=repr)
def test_every_grace_outside_its_range_is_refused_at_construction(field: str, bad: object) -> None:
    """A non-finite, negative, oversize or non-numeric grace raises `ValueError` before any client uses it."""
    with pytest.raises(ValueError, match=f"grace {field}"):
        ShutdownGraces(**{field: bad})  # type: ignore[arg-type]


@pytest.mark.logic
def test_grace_ranges_at_their_edges() -> None:
    """`term` may be 0 or all of `stop`; a zero grace elsewhere, `term` past `stop` or an oversize bound is refused."""
    assert ShutdownGraces(stop=1.0, term=0.0).term == 0.0
    assert ShutdownGraces(stop=1.0, term=1.0).term == 1.0
    for field in ("stop", "kill", "join", "exit_code_wait"):
        with pytest.raises(ValueError, match=field):
            ShutdownGraces(**{field: 0.0, **({"term": 0.0} if field == "stop" else {})})
    with pytest.raises(ValueError, match="term"):
        ShutdownGraces(stop=1.0, term=1.5)
    third = threading.TIMEOUT_MAX / 2
    with pytest.raises(ValueError, match="bound"):
        ShutdownGraces(stop=third, term=0.0, kill=third, join=third)


@pytest.mark.logic
@pytest.mark.parametrize("field", _GRACE_FIELDS)
def test_graces_overflow_is_value_error(field: str) -> None:
    """An int too large for a float is refused as `ValueError`, as every other grace out of range is."""
    with pytest.raises(ValueError, match=f"grace {field}"):
        ShutdownGraces(**{field: 10**400})


@pytest.mark.logic
def test_exit_observed_by_one_wait() -> None:
    """One `_wait_exit` blocks for an exit or, unblocked, asks; neither reaps, and no second probe exists."""
    proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        assert worker_client._wait_exit(proc.pid, block=False) is False
        proc.stdin.close()
        assert worker_client._wait_exit(proc.pid) is True
        assert worker_client._wait_exit(proc.pid, block=False) is True
        assert proc.returncode is None
    finally:
        proc.stdin.close()
        proc.wait(timeout=30)
    assert not hasattr(worker_client, "_has_exited")


@pytest.mark.logic
def test_max_line_bytes_out_of_range_is_refused(array_parent: Path) -> None:
    """A line limit must be a positive int no larger than the worker's own."""
    for bad in (0, MAX_LINE_BYTES + 1, 2048.0, True):
        with pytest.raises(ValueError, match="max_line_bytes"):
            WorkerClient(PYTHON, FAKE, init={}, array_parent=array_parent, max_line_bytes=bad)  # type: ignore[arg-type]


@pytest.mark.logic
def test_response_line_bounded(array_parent: Path) -> None:
    """A response line longer than the client's `max_line_bytes` is `LineTooLarge`, and the worker is discarded."""
    long = {"id": "$ID", "item": "a", "outputs": {"x": "$PAD"}}
    client = _scripted(array_parent, [long, _done(1)], max_line_bytes=SMALL_LINE_BYTES, pad=2 * SMALL_LINE_BYTES)
    with pytest.raises(LineTooLarge, match=str(SMALL_LINE_BYTES)):
        client.run([{"id": "a", "inputs": {}}], {})
    assert not client.alive
    client.close()


@pytest.mark.logic
def test_scripted_fixture_exhausted_raises_typed(array_parent: Path) -> None:
    """A request past the scripted worker's last reply ends it with `RepliesExhausted`, never `StopIteration`."""
    client = _scripted(array_parent)
    with pytest.raises(WorkerLost) as lost:
        client.run([{"id": "a", "inputs": {}}], {})
    assert "RepliesExhausted" in "\n".join(lost.value.__notes__)
    client.close()


@pytest.mark.logic
def test_a_clients_graces_cannot_be_replaced(worker: WorkerClient) -> None:
    """`graces` is read-only on a live client, and the graces themselves are frozen."""
    with pytest.raises(AttributeError):
        worker.graces = ShutdownGraces(stop=1.0, term=0.5)  # type: ignore[misc]
    with pytest.raises(AttributeError):
        worker.graces.stop = 1e20  # type: ignore[misc]
    assert worker.graces == DEFAULT_GRACES


@pytest.mark.logic
def test_unencodable_params_refuse_the_request(worker: WorkerClient) -> None:
    """Params with no wire form raise before sending, and the worker stays usable."""
    with pytest.raises(ValueError):
        worker.run([{"id": "a", "inputs": {}}], {"w": float("nan")})
    assert worker.run([{"id": "a", "inputs": {}}], {})[0]["outputs"]


@pytest.mark.logic
@pytest.mark.parametrize(
    ("items", "params"),
    [
        ([{"id": "a", "inputs": {}}, {"id": "a", "inputs": {}}], {}),
        ([{"id": "a"}], {}),
        ([{"id": 1, "inputs": {}}], {}),
        ([{"id": "a", "inputs": []}], {}),
        ([{"id": "a", "inputs": {}}], []),
    ],
)
def test_malformed_requests_are_refused_on_the_host(worker: WorkerClient, items: list[Any], params: object) -> None:
    """Item ids must be unique strings, inputs and params dicts."""
    with pytest.raises((ValueError, TypeError)):
        worker.run(items, params)  # type: ignore[arg-type]
    assert worker.alive


@pytest.mark.logic
def test_handshake_mismatch_refuses(array_parent: Path) -> None:
    """A worker whose runtime differs from the expected one is refused, naming both."""
    with pytest.raises(ProtocolMismatch) as info:
        WorkerClient(PYTHON, FAKE, init={}, array_parent=array_parent, expect_runtime="0" * 64)
    assert "0" * 64 in str(info.value) and runtime_identity() in str(info.value)
    assert classify(info.value) is FaultClass.INFRASTRUCTURE and isinstance(info.value, RuntimeError)


@pytest.mark.logic
@posix_only
def test_non_worker_answer_is_a_mismatch(array_parent: Path) -> None:
    """A first JSON message that is not a hello is a mismatch."""
    with pytest.raises(ProtocolMismatch):
        WorkerClient(interpreter(array_parent.parent, "echo '{\"x\": 1}'"), FAKE, init={}, array_parent=array_parent)


@pytest.mark.logic
@posix_only
@pytest.mark.parametrize(("body", "code", "stderr"), [("echo junk", 0, None), ("echo oops >&2; exit 3", 3, "oops")])
def test_exit_before_hello_is_worker_lost(array_parent: Path, body: str, code: int, stderr: str | None) -> None:
    """A worker that exits before its hello is lost, with its exit code and the stderr tail."""
    with pytest.raises(WorkerLost) as info:
        WorkerClient(interpreter(array_parent.parent, body), FAKE, init={}, array_parent=array_parent)
    assert info.value.returncode == code
    if stderr:
        assert any(stderr in note for note in info.value.__notes__)


@pytest.mark.logic
@pytest.mark.parametrize(
    ("module", "error_type"), [("no_such_backend", "ModuleNotFoundError"), ("json", "AttributeError")]
)
def test_unusable_backend_is_worker_fatal(array_parent: Path, module: str, error_type: str) -> None:
    """A backend module that cannot be imported, or lacks `load`, fails the load."""
    with pytest.raises(WorkerFatal) as info:
        WorkerClient(PYTHON, module, init={}, array_parent=array_parent)
    assert info.value.error["type"] == error_type


@pytest.mark.logic
def test_no_array_file_remains_when_every_item_fails_to_encode(worker: WorkerClient) -> None:
    """An array written before its item failed to encode is removed though nothing was sent."""
    out = worker.run([{"id": "a", "inputs": {"x": numpy.ones(3), "y": numpy.array([object()])}}], {})
    assert out[0]["error"]["fault"] is FaultClass.CONTENT
    assert list(worker.array_dir.iterdir()) == []


@pytest.mark.logic
def test_no_array_file_remains_when_params_fail_to_encode(worker: WorkerClient) -> None:
    """An array written before the params failed to encode is removed."""
    with pytest.raises(ValueError):
        worker.run([{"id": "a", "inputs": {}}], {"w": numpy.ones(3), "z": float("nan")})
    assert list(worker.array_dir.iterdir()) == []


@pytest.mark.logic
@pytest.mark.parametrize("init", [{"weights": "x"}, {"faults": {}}])
def test_load_failure_is_worker_fatal(array_parent: Path, init: dict[str, Any]) -> None:
    """A `fatal` during load raises `WorkerFatal`, an infrastructure fault carrying the error."""
    with pytest.raises(WorkerFatal) as info:
        WorkerClient(PYTHON, FAKE, init=init, array_parent=array_parent)
    assert info.value.error["fault"] == "content"
    assert classify(info.value) is FaultClass.INFRASTRUCTURE


@pytest.mark.logic
def test_reserved_init_key_is_refused(array_parent: Path) -> None:
    """`array_dir` in `init` is reserved."""
    with pytest.raises(ValueError):
        WorkerClient(PYTHON, FAKE, init={"array_dir": "x"}, array_parent=array_parent)


@pytest.mark.logic
def test_crash_mid_batch_is_worker_lost(array_parent: Path) -> None:
    """A worker that dies mid-request raises `WorkerLost` and is discarded."""
    client = WorkerClient(PYTHON, FAKE, init={}, array_parent=array_parent)
    with pytest.raises(WorkerLost) as info:
        client.run([{"id": "a", "inputs": {}}, {"id": "b", "inputs": {}}], {FAULT_KEY: {"crash_after": 1}})
    assert classify(info.value) is FaultClass.INFRASTRUCTURE and info.value.__notes__
    assert info.value.returncode == EXIT_CRASH and str(EXIT_CRASH) in str(info.value)
    assert not client.alive and list(array_parent.iterdir()) == []
    with pytest.raises(WorkerLost):
        client.run([{"id": "a", "inputs": {}}], {})
    client.close()


@pytest.mark.logic
def test_timeout_is_resource_fault(array_parent: Path) -> None:
    """A load or request past its timeout is `WorkerTimeout`, a typed resource fault, and the worker is killed."""
    client = WorkerClient(PYTHON, FAKE, init={}, array_parent=array_parent, request_timeout=0.5)
    with pytest.raises(WorkerTimeout) as info:
        client.run([{"id": "a", "inputs": {}}], {FAULT_KEY: {"latency_s": 60}})
    record = error_record(info.value)
    assert (record["fault"], record["gap"]) == ("resource", False) and isinstance(info.value, TimeoutError)
    assert not client.alive and info.value.__notes__
    client.close()
    with pytest.raises(WorkerTimeout):
        WorkerClient(PYTHON, FAKE, init={}, array_parent=array_parent, load_timeout=1e-9)


@pytest.mark.logic
def test_close_is_idempotent_and_context_managed(array_parent: Path) -> None:
    """`close` may be called twice; the client is a context manager."""
    with WorkerClient(PYTHON, FAKE, init={}, array_parent=array_parent) as client:
        assert client.alive
    client.close()
    assert not client.alive


def _ok(item: str, request: str = "$ID") -> dict[str, Any]:
    return {"id": request, "item": item, "outputs": {}}


def _done(n: int, request: str = "$ID") -> dict[str, Any]:
    return {"id": request, "done": n}


_ERROR = {"type": "ValueError", "message": "m", "fault": "content", "traceback": "", "gap": False}


@pytest.mark.logic
@pytest.mark.parametrize(
    "reply",
    [
        [_ok("a", "other"), _ok("b"), _done(2)],
        [_done(0)],
        [_ok("b"), _ok("a"), _done(2)],
        [_ok("a"), _done(1)],
        [_ok("a"), _ok("b"), _ok("b"), _done(2)],
        [_ok("a"), _ok("b"), _done(3)],
        [_ok("a"), _ok("b"), _done(2, "other")],
        [_ok("a"), _ok("b"), {"id": "$ID", "done": True}],
        [{"id": "$ID", "item": "a"}, _ok("b"), _done(2)],
        [{**_ok("a"), "error": _ERROR}, _ok("b"), _done(2)],
        [{"id": "$ID", "item": "a", "error": {**_ERROR, "fault": "weird"}}, _ok("b"), _done(2)],
        [{"id": "$ID", "item": "a", "error": {"fault": "content", "gap": False}}, _ok("b"), _done(2)],
        [{"id": "$ID", "item": "a", "outputs": [1]}, _ok("b"), _done(2)],
        ['{"id": "$ID", "item": "a", "item": "b", "outputs": {}}', _ok("b"), _done(2)],
        ['{"id": "$ID", "item": "a", "outputs": {"x": 1e999}}', _ok("b"), _done(2)],
        ["[1]"],
        [{"ready": {}}],
        [{"id": "$ID", "results": [_ok("a"), _ok("b")]}],
    ],
)
def test_malformed_responses_are_protocol_errors(array_parent: Path, reply: list[Any]) -> None:
    """A reply line with the wrong id, item, order, count or shape raises `ProtocolError`."""
    client = _scripted(array_parent, reply)
    with pytest.raises(ProtocolError):
        client.run([{"id": "a", "inputs": {}}, {"id": "b", "inputs": {}}], {})
    assert not client.alive
    client.close()


@pytest.mark.logic
def test_fatal_during_a_request_is_worker_fatal(array_parent: Path) -> None:
    """A `fatal` line in place of a response raises `WorkerFatal`, whatever fault it names."""
    error = {"type": "MemoryError", "message": "m", "fault": "resource", "traceback": "", "gap": False}
    client = _scripted(array_parent, {"error": error, "fatal": True})
    with pytest.raises(WorkerFatal) as info:
        client.run([{"id": "a", "inputs": {}}], {})
    assert info.value.error == error and classify(info.value) is FaultClass.INFRASTRUCTURE
    client.close()


@pytest.mark.logic
@pytest.mark.parametrize(
    ("error", "kept"),
    [
        ({**_ERROR, "fault": "weird"}, ("ValueError", "m")),
        ({k: v for k, v in _ERROR.items() if k != "gap"}, ("ValueError", "m")),
        ({**_ERROR, "gap": "no"}, ("ValueError", "m")),
        ({"message": "only a message"}, (None, "only a message")),
        ({"type": 3, "message": ["not", "text"]}, (None, None)),
        ({**_ERROR, "fault": ["content"]}, ("ValueError", "m")),
        ("a bare string", (None, None)),
        (None, (None, None)),
    ],
)
def test_unknown_fatal_keeps_message(array_parent: Path, error: object, kept: tuple[Any, ...]) -> None:
    """A `fatal` whose fault is unknown or whose `gap` is missing is `infrastructure` with a gap, and keeps its text."""
    client = _scripted(array_parent, {"error": error, "fatal": True})
    with pytest.raises(WorkerFatal) as info:
        client.run([{"id": "a", "inputs": {}}], {})
    got = info.value.error
    assert (got["fault"], got["gap"]) == (FaultClass.INFRASTRUCTURE, True)
    assert isinstance(got["type"], str) and isinstance(got["message"], str)
    type_name, message = kept
    if type_name is not None:
        assert got["type"] == type_name
    if message is not None:
        assert got["message"] == message and message in str(info.value)
    client.close()


_JSON = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False) | st.text(),
    lambda children: st.lists(children, max_size=3) | st.dictionaries(st.text(max_size=8), children, max_size=3),
    max_leaves=10,
)
_ERROR_FIELDS = st.fixed_dictionaries(
    {},
    optional={key: _JSON | st.sampled_from(["content", "resource", "weird", True, False]) for key in _ERROR},
)


@pytest.mark.logic
@settings(max_examples=300, deadline=None)
@given(error=_JSON | _ERROR_FIELDS)
def test_fatal_decode_is_total(error: object) -> None:
    """Any JSON value as a fatal's error decodes to a record: text fields are strings, an unknown fault is a gap."""
    record = worker_client._fatal_record(error)
    assert all(type(record[k]) is str for k in ("type", "message", "traceback"))
    assert isinstance(record["fault"], FaultClass) and type(record["gap"]) is bool
    sent = error if type(error) is dict else {}
    if sent.get("fault") not in ("content", "resource", "infrastructure") or type(sent.get("gap")) is not bool:
        assert (record["fault"], record["gap"]) == (FaultClass.INFRASTRUCTURE, True)
    else:
        assert (record["fault"].value, record["gap"]) == (sent["fault"], sent["gap"])
    for key in ("type", "message"):
        if type(sent.get(key)) is str:
            assert record[key] == sent[key]
    str(worker_client.WorkerFatal(record))


@pytest.mark.logic
@pytest.mark.parametrize(
    ("tail", "code"),
    [
        (["$EXIT:3"], 3),
        (['$RAW:{"id": "$ID", "item": "b", "outp', "$EXIT:4"], 4),
        ([_ok("b"), "$EXIT:0"], 0),
    ],
)
def test_crash_mid_reply_is_worker_lost_with_returncode(array_parent: Path, tail: list[Any], code: int) -> None:
    """A worker that exits after some of a reply's lines raises `WorkerLost` carrying its exit code."""
    client = _scripted(array_parent, [_ok("a"), *tail])
    with pytest.raises(WorkerLost) as info:
        client.run([{"id": "a", "inputs": {}}, {"id": "b", "inputs": {}}], {})
    assert info.value.returncode == code and f"exit code {code}" in str(info.value)
    assert not client.alive
    client.close()


@pytest.mark.logic
@posix_only
@pytest.mark.parametrize("wire", [True, 1.0, "2", 1, None])
def test_hello_wire_must_be_an_int_on_the_host(array_parent: Path, wire: object) -> None:
    """A hello whose `wire` is not the int 2, `true` and `1.0` included, is refused."""
    hello = {"wire": wire, "runtime": runtime_identity(), "python": "", "numpy": "", "pid": 0}
    line = json.dumps({"hello": hello})
    with pytest.raises(ProtocolMismatch):
        WorkerClient(interpreter(array_parent.parent, f"echo '{line}'"), FAKE, init={}, array_parent=array_parent)


@pytest.mark.contract
def test_serve_is_total_through_a_real_worker(worker: WorkerClient) -> None:
    """One case of the serve fuzz property over a real worker: typed, oversize and unprintable faults stay per item."""
    ids = ["fail", "oom", "big", "mute", "ok"]
    faults = {"fail_items": ["fail"], "oom_items": ["oom"], "oversize_items": ["big"], "unprintable_items": ["mute"]}
    out = worker.run([{"id": i, "inputs": {}} for i in ids], {FAULT_KEY: faults})
    assert [r["id"] for r in out] == ids
    summary = [(r["error"]["type"], r["error"]["fault"], r["error"]["gap"]) for r in out[:4]]
    assert summary == [
        ("ContentFault", FaultClass.CONTENT, False),
        ("OutOfMemoryError", FaultClass.RESOURCE, False),
        ("ResponseTooLarge", FaultClass.RESOURCE, False),
        ("UnprintableFault", FaultClass.CONTENT, False),
    ]
    assert out[3]["error"]["message"] == "<unprintable UnprintableFault>"
    assert out[4]["outputs"]["value"] == _value({}, {})
    assert worker.alive and worker.run([{"id": "x", "inputs": {}}], {})[0]["outputs"]


@pytest.mark.logic
def test_output_arrays_decode_per_item_and_init_files_survive(array_parent: Path, tmp_path: Path) -> None:
    """A damaged output array fails only its item; request and response files go, init files stay."""
    staging = tmp_path / "staging"
    staging.mkdir()
    good, bad = numpy.arange(4, dtype="<i8"), numpy.ones(3)
    good_ref, bad_ref = to_wire(good, array_dir=staging, at="line"), to_wire(bad, array_dir=staging, at="line")
    bad_ref["__array__"]["sha256"] = "e" * 64

    def placed(ref: dict[str, Any]) -> dict[str, Any]:
        moved = dict(ref["__array__"], path="$DIR/" + Path(ref["__array__"]["path"]).name)
        return {"__array__": moved}

    reply = [
        {"id": "$ID", "item": "a", "outputs": {"y": placed(good_ref)}},
        {"id": "$ID", "item": "b", "outputs": {"y": placed(bad_ref)}},
        _done(2),
    ]
    weights = numpy.full(5, 7.0)
    client = _scripted(array_parent, reply, w=weights)
    init_files = set(client.array_dir.iterdir())
    for path in staging.iterdir():
        path.replace(client.array_dir / path.name)
    out = client.run([{"id": "a", "inputs": {"x": numpy.zeros(2)}}, {"id": "b", "inputs": {}}], {})
    assert numpy.array_equal(out[0]["outputs"]["y"], good)
    assert (out[1]["error"]["type"], out[1]["error"]["fault"]) == ("ArrayIntegrityError", FaultClass.INFRASTRUCTURE)
    assert set(client.array_dir.iterdir()) == init_files
    client.close()


@pytest.mark.logic
def test_classification_gap_is_reported(array_parent: Path, caplog: pytest.LogCaptureFixture) -> None:
    """An error marked as a classification gap is logged by the host."""
    error = {"type": "OSError", "message": "m", "fault": "infrastructure", "traceback": "", "gap": True}
    client = _scripted(array_parent, [{"id": "$ID", "item": "a", "error": error}, _done(1)])
    with caplog.at_level(logging.WARNING):
        out = client.run([{"id": "a", "inputs": {}}], {})
    assert out[0]["error"]["gap"] is True and "OSError" in caplog.text
    client.close()


@pytest.mark.logic
def test_leak_fixture_records_a_child_until_it_is_waited_on(no_leaked_processes: list[subprocess.Popen[bytes]]) -> None:
    """An exited child nobody waited on still has no returncode, which is what the fixture rejects."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    assert proc in no_leaked_processes
    assert proc.returncode is None
    proc.wait()
    assert proc.returncode == 0


ITEMS = [{"id": "a", "inputs": {"x": numpy.arange(8, dtype="<i4")}}, {"id": "b", "inputs": {}}]
STOPPED_WAIT_S = 30.0


@pytest.mark.limits
def test_forking_backend_leaves_no_survivor(tmp_path: Path, backends_importable: None) -> None:
    """A timed-out request and `close()` end within their bound and leave no grandchild running."""
    pid_file = tmp_path / "pids"
    client = WorkerClient(
        PYTHON, "backends.forking_backend_fixture", init={"pid_file": str(pid_file)}, request_timeout=2
    )
    pgid = client.hello["pid"]
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        client.run([{"id": "a", "inputs": {}}], {"mode": "hang"})
    client.close()
    assert time.monotonic() - t0 < 2 + DEFAULT_GRACES.bound
    pids = recorded_pids(pid_file)
    assert len(pids) == 2 and not any(map(alive, pids))
    assert not group_alive(pgid)


@pytest.mark.limits
def test_timeout_returns_though_a_grandchild_holds_stderr(tmp_path: Path, backends_importable: None) -> None:
    """A backend that leaves a grandchild on stderr and hangs: `run` alone returns and kills it.

    The grandchildren start in `load`, so they exist before the request's timeout starts.
    """
    pid_file = tmp_path / "pids"
    init = {"pid_file": str(pid_file), "spawn_at_load": True}
    client = WorkerClient(PYTHON, "backends.forking_backend_fixture", init=init, request_timeout=2)
    assert len(recorded_pids(pid_file)) == 2
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        client.run([{"id": "a", "inputs": {}}], {"mode": "hang"})
    assert time.monotonic() - t0 < 2 + DEFAULT_GRACES.bound
    assert not client.alive and not any(map(alive, recorded_pids(pid_file)))
    client.close()


@pytest.mark.logic
def test_backend_child_cannot_read_requests(backends_importable: None) -> None:
    """Children started with the default stdin read nothing, and every request reaches the worker."""
    with WorkerClient(PYTHON, "backends.stdin_reading_backend_fixture", init={}, request_timeout=10) as c:
        for i in range(3):
            assert c.run([{"id": str(i), "inputs": {}}], {})[0]["outputs"] == {"stdin_bytes": 0}


@pytest.mark.logic
def test_client_deletes_only_what_it_created(tmp_path: Path) -> None:
    """Files the client did not write survive, in its parent and in its own directory."""
    parent = tmp_path / "p"
    parent.mkdir()
    (parent / "thesis.txt").write_text("x")
    with WorkerClient(PYTHON, FAKE, init={}, array_parent=parent) as c:
        (c.array_dir / "notes.md").write_text("y")
        c.run(ITEMS, {})
    assert (parent / "thesis.txt").exists() and (c.array_dir / "notes.md").exists()
    assert [p.name for p in c.array_dir.iterdir()] == ["notes.md"]


@pytest.mark.logic
def test_files_beside_the_array_dir_survive_runs_and_close(tmp_path: Path) -> None:
    """A directory holding `thesis.txt` and `notes.md` is left exactly as it was."""
    parent = tmp_path / "work"
    parent.mkdir()
    (parent / "thesis.txt").write_text("t")
    (parent / "notes.md").write_text("n")
    before = listing(parent)
    with WorkerClient(PYTHON, FAKE, init={}, array_parent=parent) as c:
        assert c.array_dir.parent == parent.absolute() and c.array_dir.name.startswith("senselab-arrays-")
        c.run(ITEMS, {})
        c.run(ITEMS, {"w": numpy.ones(3)})
    assert listing(parent) == before


@pytest.mark.logic
def test_array_dir_is_read_only(worker: WorkerClient) -> None:
    """`array_dir` cannot be reassigned."""
    with pytest.raises(AttributeError):
        worker.array_dir = Path("/")  # type: ignore[misc]


@pytest.mark.logic
def test_concurrent_run_is_refused(array_parent: Path, backends_importable: None) -> None:
    """A second `run` while one is in flight raises `ConcurrentUse`; the first call's results are intact."""
    first: list[object] = []
    with (
        gate() as g,
        WorkerClient(
            PYTHON,
            "backends.slow_reader_backend_fixture",
            init={"gate_port": g.port},
            array_parent=array_parent,
        ) as c,
    ):
        a = numpy.arange(32, dtype="<f8")
        thread = threading.Thread(
            target=lambda: first.append(c.run([{"id": "a", "inputs": {"x": a}}], {"mode": "gate"}))
        )
        thread.start()
        g.entered()
        with pytest.raises(ConcurrentUse):
            c.run([{"id": "b", "inputs": {"x": numpy.zeros(4)}}], {})
        g.release()
        thread.join(30)
        assert first and first[0][0]["outputs"] == {"ok": True}  # type: ignore[index]
        assert c.run([{"id": "c", "inputs": {}}], {})[0]["outputs"]


@pytest.mark.logic
def test_backend_subdir_in_array_dir_does_not_fail_runs(tmp_path: Path, backends_importable: None) -> None:
    """A directory a backend makes under an array name is refused, never entered, and runs succeed."""
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.npy").write_bytes(b"k")
    parent = tmp_path / "parent"
    parent.mkdir()
    with WorkerClient(
        PYTHON, "backends.subdir_backend_fixture", init={"victim": str(victim)}, array_parent=parent
    ) as c:
        for _ in range(3):
            assert c.run(ITEMS, {})[0]["outputs"] == {"made": ARRAY_LIKE}
            assert [p.name for p in c.array_dir.iterdir()] == [ARRAY_LIKE]
    assert [p.name for p in c.array_dir.iterdir()] == [ARRAY_LIKE]
    assert sorted(p.name for p in (c.array_dir / ARRAY_LIKE).iterdir()) == ["scratch.bin", "victim"]
    assert (victim / "keep.npy").read_bytes() == b"k"


@pytest.mark.logic
def test_a_worker_is_started_isolated_from_the_callers_environment(
    tmp_path: Path, backends_importable: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker runs under `-I` in its interpreter's directory with the declared families only."""
    monkeypatch.setenv("PYTHONSTARTUP", str(tmp_path / "startup.py"))
    monkeypatch.setenv("SENSELAB_AMBIENT", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.chdir(tmp_path)
    with WorkerClient(PYTHON, "backends.environ_backend_fixture", init={}, array_parent=tmp_path) as c:
        [result] = c.run([{"id": "i", "inputs": {}}], {})
    started = result["outputs"]
    assert started["isolated"] == 1 and os.path.realpath(started["cwd"]) == os.path.realpath(Path(PYTHON).parent)
    assert started["environ"]["OMP_NUM_THREADS"] == "3"
    assert not {"PYTHONSTARTUP", "SENSELAB_AMBIENT", "PYTHONPATH"} & set(started["environ"])


@pytest.mark.logic
@pytest.mark.parametrize("copies", [2, 5])
def test_one_array_in_every_items_outputs_is_read_once(tmp_path: Path, backends_importable: None, copies: int) -> None:
    """Results naming one array file are one read-only array; an array in one result alone stays writable."""
    a = numpy.linspace(0, 1, 2048)
    items = [{"id": f"i{n}", "inputs": {"w": a, "own": numpy.full(4, n)}} for n in range(copies)]
    with WorkerClient(PYTHON, "backends.echo_inputs_backend_fixture", init={}, array_parent=tmp_path) as c:
        out = c.run(items, {})
    shared = [r["outputs"]["w"] for r in out]
    assert all(w is shared[0] for w in shared) and not shared[0].flags.writeable
    assert numpy.array_equal(shared[0], a)
    assert all(r["outputs"]["own"].flags.writeable for r in out)


@pytest.mark.logic
def test_backend_subdir_leaves_runs_and_the_worker_usable(
    tmp_path: Path, backends_importable: None, caplog: pytest.LogCaptureFixture
) -> None:
    """A backend directory not named like an array is left in place and logged; runs succeed."""
    with (
        caplog.at_level(logging.WARNING),
        WorkerClient(PYTHON, "backends.subdir_backend_fixture", init={"subdir": "scratch"}, array_parent=tmp_path) as c,
    ):
        for _ in range(3):
            assert c.run(ITEMS, {})[0]["outputs"] == {"made": "scratch"}
            assert c.alive
    assert [p.name for p in c.array_dir.iterdir()] == ["scratch"]
    assert str(c.array_dir) in caplog.text


def _wait_stopped(pid: int) -> None:
    end = time.monotonic() + STOPPED_WAIT_S
    while not state(pid).startswith("T"):
        assert time.monotonic() < end, f"worker {pid} did not stop"


@pytest.mark.limits
def test_send_to_wedged_worker_respects_deadline(array_parent: Path, backends_importable: None) -> None:
    """An 8 MB request to a worker that has stopped reading times out at `request_timeout`."""
    client = WorkerClient(
        PYTHON, "backends.slow_reader_backend_fixture", init={}, array_parent=array_parent, request_timeout=2
    )
    client.run([{"id": "a", "inputs": {}}], {"mode": "stop_reading"})
    _wait_stopped(client.hello["pid"])
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        client.run([{"id": "a", "inputs": {}}], {"blob": "x" * 8 * 2**20})
    assert time.monotonic() - t0 < 2 + DEFAULT_GRACES.bound
    assert not group_alive(client.hello["pid"])
    client.close()


def _run_over_a_fifo_named_like_an_array(array_parent: Path) -> tuple[dict[str, Any], float]:
    """Run a request whose response names a FIFO at `<sha>.npy`; return the item's record and the time it took."""
    sha = "a" * 64
    ref = {"__array__": {"path": f"$DIR/{sha}.npy", "sha256": sha, "dtype": "<f4", "shape": [1]}}
    client = _scripted(array_parent, [{"id": "$ID", "item": "a", "outputs": {"y": ref}}, _done(1)])
    os.mkfifo(client.array_dir / f"{sha}.npy")
    t0 = time.monotonic()
    [out] = client.run([{"id": "a", "inputs": {}}], {})
    elapsed = time.monotonic() - t0
    client.close()
    return out, elapsed


@pytest.mark.logic
def test_fifo_named_like_an_array_does_not_hang_the_host(array_parent: Path) -> None:
    """A response naming a FIFO at `<sha>.npy` fails its item instead of blocking."""
    out, _ = _run_over_a_fifo_named_like_an_array(array_parent)
    assert out["error"]["type"] == "ArrayIntegrityError"


@pytest.mark.limits
def test_fifo_named_like_an_array_fails_its_item_within_a_second(array_parent: Path) -> None:
    """A response naming a FIFO at `<sha>.npy` fails its item in under 1 s."""
    _, elapsed = _run_over_a_fifo_named_like_an_array(array_parent)
    assert elapsed < 1


_LEFT_AT_EXIT = """
import logging, sys
from pathlib import Path
from senselab_core.runtime.worker_client import WorkerClient
from senselab_core.testing.interpreter import source_tree_python as python
logging.basicConfig(level=logging.WARNING, format="LOG %(levelname)s %(message)s")
c = WorkerClient(python(), "senselab_worker.fake", init={{}}, array_parent=Path(sys.argv[1]))
{body}
print(c._proc.pid, flush=True)
"""


@pytest.mark.limits
@pytest.mark.parametrize("body", ["pass", "c.run([{'id': 'a', 'inputs': {}}], {})", "holder = [c]"])
def test_a_client_left_open_at_interpreter_exit_finalises_at_once_and_quietly(array_parent: Path, body: str) -> None:
    """A client still open at exit is finalised well inside one grace, logs nothing, and leaves no file or process."""
    script = array_parent.parent / "left_open.py"
    script.write_text(_LEFT_AT_EXIT.format(body=body))
    start = time.monotonic()
    proc = subprocess.run(
        [sys.executable, str(script), str(array_parent)], capture_output=True, text=True, timeout=60, check=False
    )
    elapsed = time.monotonic() - start
    assert proc.returncode == 0, proc.stderr
    assert elapsed < DEFAULT_GRACES.term, (elapsed, proc.stderr)
    assert [line for line in proc.stderr.splitlines() if line.startswith("LOG")] == []
    assert not alive(int(proc.stdout.split()[-1]))


@pytest.mark.logic
def test_leaked_client_is_closed_with_a_resource_warning(array_parent: Path) -> None:
    """A client dropped without `close` warns, stops its worker and removes its directory."""
    client = WorkerClient(PYTHON, FAKE, init={}, array_parent=array_parent)
    pgid = client.hello["pid"]
    with pytest.warns(ResourceWarning):
        del client
        gc.collect()
    assert not group_alive(pgid)


@pytest.mark.contract
def test_print_from_a_pth_before_the_channel_is_claimed_is_discarded(
    array_parent: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A `.pth` that writes to fd 1 at interpreter start precedes the hello; the host logs and skips it."""
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True, timeout=120)
    [site] = venv.glob("lib/python*/site-packages")
    roots = {Path(senselab_worker.__file__).parent.parent, Path(numpy.__file__).parent.parent}
    (site / "a_host_packages.pth").write_text("".join(f"{r}\n" for r in sorted(roots)))
    (site / "b_noise.pth").write_text("import os; os.write(1, b'noise from a pth\\n')\n")
    with (
        caplog.at_level(logging.WARNING),
        WorkerClient(venv / "bin" / "python", FAKE, init={}, array_parent=array_parent) as c,
    ):
        assert c.run([{"id": "a", "inputs": {}}], {})[0]["outputs"]
    assert "noise from a pth" in caplog.text


def _record_killpg(monkeypatch: pytest.MonkeyPatch) -> list[tuple[float, int, int]]:
    sent: list[tuple[float, int, int]] = []
    real = os.killpg

    def record(pgid: int, sig: int) -> None:
        sent.append((time.monotonic(), pgid, sig))
        real(pgid, sig)

    monkeypatch.setattr(os, "killpg", record)
    return sent


@pytest.mark.logic
def test_late_close_after_the_leader_was_reaped_elsewhere_sends_no_signal(
    array_parent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once the leader is reaped outside the client its pid may be reused, so `close()` signals nothing."""
    client = WorkerClient(PYTHON, FAKE, init={}, array_parent=array_parent)
    pid = client.hello["pid"]
    os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)
    sent = _record_killpg(monkeypatch)
    client.close()
    assert [s for s in sent if s[2] != 0] == []


@pytest.mark.logic
def test_idle_crash_keeps_the_leader_unreaped_until_close(array_parent: Path) -> None:
    """A leader that dies while idle stays a zombie, holding its pid and group id, until `close()` reaps it."""
    client = WorkerClient(PYTHON, FAKE, init={}, array_parent=array_parent)
    pid = client.hello["pid"]
    os.kill(pid, signal.SIGKILL)
    end = time.monotonic() + STOPPED_WAIT_S
    while client.alive:
        assert time.monotonic() < end
    assert client.returncode is None and state(pid).startswith("Z")
    client.close()
    assert client.returncode == -signal.SIGKILL


@pytest.mark.limits
def test_graceful_close_leaves_sigterm_its_own_grace(
    array_parent: Path, backends_importable: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker that ignores the stop line gets SIGTERM with `graces.term` of the stop window left."""
    client = WorkerClient(
        PYTHON,
        "backends.slow_reader_backend_fixture",
        init={},
        array_parent=array_parent,
        graces=ShutdownGraces(stop=2.0, term=1.0),
    )
    client.run([{"id": "a", "inputs": {}}], {"mode": "stop_reading"})
    _wait_stopped(client.hello["pid"])
    sent = _record_killpg(monkeypatch)
    t0 = time.monotonic()
    client.close()
    terms = [t for t, _, sig in sent if sig == signal.SIGTERM]
    assert terms and terms[0] - t0 < 2.0 - 1.0 + 0.5
    assert signal.SIGKILL not in [sig for _, _, sig in sent]


def _probe_fails(monkeypatch: pytest.MonkeyPatch, how: str) -> list[float]:
    """Route the leader probe through `ps`, as where `waitid` is missing; make every `ps` fail as `how`."""
    calls: list[float] = []
    real_run = subprocess.run

    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:  # noqa: ANN401
        if argv[0] != worker_client._PS:
            return real_run(argv, **kwargs)
        timeout = kwargs["timeout"]
        assert isinstance(timeout, float) and timeout <= worker_client.LEADER_PROBE_TIMEOUT_S
        calls.append(timeout)
        if how == "missing":
            raise FileNotFoundError(2, "No such file or directory", argv[0])
        time.sleep(timeout)
        raise subprocess.TimeoutExpired(argv, timeout)

    monkeypatch.delattr(os, "waitid", raising=False)
    monkeypatch.setattr(subprocess, "run", run)
    return calls


def _close_after_an_idle_crash_with_failing_ps(array_parent: Path, monkeypatch: pytest.MonkeyPatch, how: str) -> float:
    """Crash an idle worker, make `ps` fail as `how`, close; assert nothing was signalled; return `close()`'s time."""
    client = WorkerClient(PYTHON, FAKE, init={}, array_parent=array_parent)
    pid = client.hello["pid"]
    os.kill(pid, signal.SIGKILL)
    end = time.monotonic() + STOPPED_WAIT_S
    while client.alive:
        assert time.monotonic() < end
    calls = _probe_fails(monkeypatch, how)
    sent = _record_killpg(monkeypatch)
    t0 = time.monotonic()
    client.close()
    elapsed = time.monotonic() - t0
    assert calls and [s for s in sent if s[2] != 0] == []
    assert client.returncode == -signal.SIGKILL and not alive(pid)
    return elapsed


@pytest.mark.logic
def test_linux_without_waitid_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """On Linux without `os.waitid`, the exit probe answers 'running' and the leader probe asks `ps`."""
    monkeypatch.setattr(sys, "platform", "linux")
    calls = _probe_fails(monkeypatch, "missing")
    exited = threading.Event()
    assert worker_client._exit_seen(os.getpid(), exited) is False
    exited.set()
    assert worker_client._leader_held(os.getpid(), exited, worker_client.LEADER_PROBE_TIMEOUT_S) is None
    assert len(calls) == 1


@pytest.mark.logic
@pytest.mark.parametrize("how", ["missing", "timeout"])
def test_unknown_leader_state_is_never_read_as_reaped(
    array_parent: Path, monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    """With `ps` missing or timing out after an idle crash, `close()` returns and signals nothing."""
    _close_after_an_idle_crash_with_failing_ps(array_parent, monkeypatch, how)


@pytest.mark.limits
@pytest.mark.parametrize("how", ["missing", "timeout"])
def test_unknown_leader_state_is_bounded(array_parent: Path, monkeypatch: pytest.MonkeyPatch, how: str) -> None:
    """With `ps` missing or timing out after an idle crash, `close()` returns within `DEFAULT_GRACES.bound`."""
    assert _close_after_an_idle_crash_with_failing_ps(array_parent, monkeypatch, how) < DEFAULT_GRACES.bound


def _time_out_a_forking_worker_with_failing_ps(
    tmp_path: Path, array_parent: Path, monkeypatch: pytest.MonkeyPatch, how: str
) -> float:
    """Time out a hung forking worker with `ps` failing as `how`; assert `ps` was not asked; return `run`'s time."""
    pid_file = tmp_path / "pids"
    client = WorkerClient(
        PYTHON,
        "backends.forking_backend_fixture",
        init={"pid_file": str(pid_file)},
        array_parent=array_parent,
        request_timeout=1,
    )
    pgid = client.hello["pid"]
    calls = _probe_fails(monkeypatch, how)
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        client.run([{"id": "a", "inputs": {}}], {"mode": "hang"})
    elapsed = time.monotonic() - t0
    client.close()
    assert calls == [] and not group_alive(pgid) and not any(map(alive, recorded_pids(pid_file)))
    return elapsed


@pytest.mark.logic
@pytest.mark.parametrize("how", ["missing", "timeout"])
def test_running_leader_is_stopped_without_asking_ps(
    tmp_path: Path, array_parent: Path, backends_importable: None, monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    """A leader whose exit has not been observed is held: a timed-out worker's group is stopped without `ps`."""
    _time_out_a_forking_worker_with_failing_ps(tmp_path, array_parent, monkeypatch, how)


@pytest.mark.limits
@pytest.mark.parametrize("how", ["missing", "timeout"])
def test_a_timed_out_worker_is_stopped_within_the_bound_without_asking_ps(
    tmp_path: Path, array_parent: Path, backends_importable: None, monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    """A timed-out worker's group is stopped within `1 s + DEFAULT_GRACES.bound` though `ps` fails."""
    elapsed = _time_out_a_forking_worker_with_failing_ps(tmp_path, array_parent, monkeypatch, how)
    assert elapsed < 1 + DEFAULT_GRACES.bound


@pytest.mark.logic
def test_unexpected_signal_errors_are_logged(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """ESRCH is silent; EPERM is silent only for a zombie leader on macOS; any other errno is logged."""

    def fail(err: int) -> None:
        def killpg(pgid: int, sig: int) -> None:
            raise OSError(err, os.strerror(err))

        monkeypatch.setattr(os, "killpg", killpg)

    with caplog.at_level(logging.WARNING):
        fail(errno.ESRCH)
        worker_client._signal_group(4242, signal.SIGTERM, leader_exited=False)
        assert caplog.text == ""
        fail(errno.EINVAL)
        worker_client._signal_group(4242, signal.SIGTERM, leader_exited=True)
        assert "SIGTERM" in caplog.text and "4242" in caplog.text
        caplog.clear()
        fail(errno.EPERM)
        worker_client._signal_group(4242, signal.SIGKILL, leader_exited=False)
        assert "SIGKILL" in caplog.text
        caplog.clear()
        worker_client._signal_group(4242, signal.SIGKILL, leader_exited=True)
        assert (caplog.text == "") is (sys.platform == "darwin")


def _chain(height: int, leaf: object) -> object:
    value = leaf
    for i in range(height):
        value = [value] if i % 2 else {"k": value}
    return value


@pytest.mark.contract
@pytest.mark.parametrize("leaf", ["scalar", "empty", "array"])
def test_input_depth_band_is_sent_or_a_content_fault(worker: WorkerClient, leaf: str) -> None:
    """Inputs across the depth band are sent, or are that item's `WireValueError`; `run` never raises.

    The band runs from MAX_DEPTH - 4 to MAX_DEPTH + 1 on the request line; the worker stays alive.
    """
    leaf_value, leaf_height = {"scalar": (0, 0), "empty": ({}, 1), "array": (numpy.zeros(2), ARRAY_REF_HEIGHT)}[leaf]
    band = range(MAX_DEPTH - ENVELOPE["inputs"] - 4, MAX_DEPTH - ENVELOPE["inputs"] + 2)
    items = [{"id": f"h{h}", "inputs": {"k": _chain(h - 1 - leaf_height, leaf_value)}} for h in band]
    out = worker.run([*items, {"id": "mate", "inputs": {}}], {})
    assert out[-1]["outputs"]["value"] == _value({}, {}) and worker.alive
    for h, r in zip(band, out):
        if ENVELOPE["inputs"] + h <= MAX_DEPTH:
            assert "outputs" in r, (h, r)
        else:
            assert (r["error"]["type"], r["error"]["fault"], r["error"]["gap"]) == (
                "WireValueError",
                FaultClass.CONTENT,
                False,
            )


@pytest.mark.logic
def test_exponentially_shared_input_fails_only_its_item(worker: WorkerClient) -> None:
    """The host's encoder is bounded too: an input `l = [l, l]` sixty times is that item's `ValueTooLarge`."""
    value: object = 0
    for _ in range(60):
        value = [value, value]
    out = worker.run([{"id": "dag", "inputs": {"s": value}}, {"id": "mate", "inputs": {}}], {})
    assert (out[0]["error"]["type"], out[0]["error"]["fault"], out[0]["error"]["gap"]) == (
        "ValueTooLarge",
        FaultClass.RESOURCE,
        False,
    )
    assert out[1]["outputs"]["value"] == _value({}, {}) and worker.alive


@pytest.mark.logic
def test_items_that_fit_alone_but_not_together_raise_value_too_large(worker: WorkerClient) -> None:
    """Items that each fit but together exceed the line budget refuse the request; the worker stays usable."""
    items = [{"id": str(i), "inputs": {"s": "x" * 10**6}} for i in range(20)]
    with pytest.raises(ValueTooLarge, match="does not fit one line"):
        worker.run(items, {})
    assert worker.alive and worker.run(items[:2], {})[0]["outputs"]


def _fits(items: list[dict[str, Any]], params: dict[str, Any], limit: int) -> bool:
    request = {"id": "0" * 36, "items": items, "params": params}
    try:
        codec.dump_line(request, "request", limit=limit)
    except codec.LineTooLarge:
        return False
    return True


@pytest.mark.contract
@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(sizes=st.lists(st.integers(0, 1500), min_size=1, max_size=6), params_size=st.integers(0, 1500))
def test_oversize_is_value_too_large_per_item_where_attributable(
    small_line_worker: WorkerClient, sizes: list[int], params_size: int
) -> None:
    """An oversized request never raises a plain `ValueError`: what one item causes is that item's error.

    Against the codec's own line check: an item whose request alone cannot fit fails alone as
    `ValueTooLarge` (resource); params that cannot fit, or items that fit only alone, raise `ValueTooLarge`.
    """
    worker, budget = small_line_worker, SMALL_LINE_BYTES
    params = {"p": "y" * params_size}
    items = [{"id": str(i), "inputs": {"s": "x" * n}} for i, n in enumerate(sizes)]
    alone = [_fits([item], params, budget) for item in items]
    kept = [item for item, ok in zip(items, alone) if ok]
    if not _fits([{"id": "", "inputs": {}}], params, budget) or (kept and not _fits(kept, params, budget)):
        with pytest.raises(ValueTooLarge):
            worker.run(items, params)
        return
    out = worker.run(items, params)
    for result, ok in zip(out, alone):
        if ok:
            assert "outputs" in result
        else:
            assert (result["error"]["type"], result["error"]["fault"]) == ("ValueTooLarge", FaultClass.RESOURCE)
    assert worker.alive


COUNTING = "backends.counting_backend_fixture"
FAKE_KEY = ("fake", "c" * 40, "cpu", "float32")


def _one_byte(_: object) -> int:
    return 1


@pytest.mark.contract
def test_host_and_isolated_share_handle(array_parent: Path) -> None:
    """One backend's `handle` gives the same results in process (`Env.host`) and in a worker (`Env.isolated`).

    The isolated env of the fake stack is the running interpreter (a declared test seam); no env is built.
    """
    isolated = FAKE_SPEC.model_copy(update={"env": Env.isolated("fake")})
    items = [{"id": i, "inputs": {"payload": i, "signal": numpy.arange(4.0)}} for i in "abc"]
    params = {"salt": "s", FAULT_KEY: {"fail_items": ["b"], "oom_items": ["c"]}}
    host = HostWorker(
        FAKE_SPEC.worker, init=dict(FAKE_SPEC.load_stub), key=FAKE_KEY, cache=ModelCache(1), size_of=_one_byte
    )
    with WorkerClient(PYTHON, isolated.worker, init=dict(isolated.load_stub), array_parent=array_parent) as w:
        remote = w.run(items, params)
    local = host.run(items, params)

    def comparable(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {**r, "error": {k: v for k, v in r["error"].items() if k != "traceback"}} if "error" in r else r
            for r in results
        ]

    assert comparable(local) == comparable(remote)
    assert [r.get("error", {}).get("fault") for r in local] == [None, FaultClass.CONTENT, FaultClass.RESOURCE]


@pytest.mark.logic
def test_host_worker_model_loaded_once_across_calls() -> None:
    """In process, calls and clients sharing a cache and key load the backend once."""
    from backends import counting_backend_fixture as counting  # noqa: PLC0415 - the module records loads

    counting.LOADS.clear()
    cache = ModelCache(1)
    for _ in range(2):
        worker = HostWorker(COUNTING, init={}, key=FAKE_KEY, cache=cache, size_of=_one_byte)
        worker.run([{"id": "a", "inputs": {}}], {})
        worker.run([{"id": "b", "inputs": {}}], {})
    assert counting.LOADS == [{}]


@pytest.mark.logic
def test_host_worker_reloads_an_evicted_model() -> None:
    """A run whose state was evicted by another model loads it again."""
    from backends import counting_backend_fixture as counting  # noqa: PLC0415 - the module records loads

    counting.LOADS.clear()
    cache = ModelCache(1)
    first = HostWorker(COUNTING, init={"device": "cpu"}, key=FAKE_KEY, cache=cache, size_of=_one_byte)
    HostWorker(COUNTING, init={}, key=("fake", "d" * 40, "cpu", "float32"), cache=cache, size_of=_one_byte)
    assert first.run([{"id": "a", "inputs": {}}], {})[0]["outputs"]["value"] == _value({}, {})
    assert counting.LOADS == [{"device": "cpu"}, {}, {"device": "cpu"}]


@pytest.mark.logic
@pytest.mark.parametrize(("module", "init"), [("no_such_backend_module", {}), (FAKE, {"unknown": 1})])
def test_host_worker_load_failure_is_worker_fatal(module: str, init: dict[str, Any]) -> None:
    """An import or load failure is `WorkerFatal`, as a worker reports it, and nothing is cached."""
    cache = ModelCache(1)
    with pytest.raises(WorkerFatal):
        HostWorker(module, init=init, key=FAKE_KEY, cache=cache, size_of=_one_byte)
    assert cache.keys() == []


@pytest.mark.logic
def test_host_worker_refuses_a_ref_and_reserved_init() -> None:
    """A ref as the model identity, or `array_dir` in `init`, is refused before anything loads."""
    cache = ModelCache(1)
    with pytest.raises(ValueError, match="never a ref"):
        HostWorker(FAKE, init={}, key=("fake", "main", "cpu", "float32"), cache=cache, size_of=_one_byte)
    with pytest.raises(ValueError, match="reserved"):
        HostWorker(FAKE, init={"array_dir": "/x"}, key=FAKE_KEY, cache=cache, size_of=_one_byte)
    assert cache.keys() == []


@pytest.mark.logic
def test_host_worker_malformed_request_is_value_error() -> None:
    """A malformed request is a `ValueError`, as from `WorkerClient.run`."""
    worker = HostWorker(FAKE, init={}, key=FAKE_KEY, cache=ModelCache(1), size_of=_one_byte)
    with pytest.raises(ValueError, match="malformed request"):
        worker.run([{"inputs": {}}], {})
