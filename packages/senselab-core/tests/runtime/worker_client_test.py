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

import numpy
import pytest
import senselab_worker
from backends.subdir_backend_fixture import ARRAY_LIKE
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from lifecycle_probes_fixture import alive, gate, group_alive, interpreter, listing, recorded_pids, state
from senselab_core.runtime import worker_client
from senselab_core.runtime.worker_client import (
    SHUTDOWN_BOUND_S,
    ConcurrentUse,
    ProtocolMismatch,
    WorkerClient,
    WorkerFatal,
    WorkerLost,
)
from senselab_worker import codec, runtime_identity
from senselab_worker.canonical import canonical_npy_bytes
from senselab_worker.codec import ARRAY_REF_HEIGHT, ENVELOPE, MAX_DEPTH, to_wire
from senselab_worker.digest import identity_digest
from senselab_worker.fake import EXIT_CRASH
from senselab_worker.faults import FaultClass, ProtocolError, ValueTooLarge, classify

FAKE = "senselab_worker.fake"
SCRIPTED = Path(__file__).parent / "scripted_worker_fixture.py"
posix_only = pytest.mark.skipif(os.name == "nt", reason="stands in for the interpreter with a shell script")


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
            super().__init__(*args, **kwargs)  # type: ignore[call-overload]
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
    client = WorkerClient(sys.executable, FAKE, init={}, array_parent=array_parent)
    yield client
    client.close()
    assert not client.alive


def _lines(reply: dict | list) -> list[str]:
    return [
        part if isinstance(part, str) else json.dumps(part) for part in (reply if isinstance(reply, list) else [reply])
    ]


def _scripted(array_parent: Path, *replies: dict | list, **init: object) -> WorkerClient:
    """A client over the scripted worker; each reply is one line (a dict) or a list of lines."""
    python = interpreter(array_parent.parent, f'exec "{sys.executable}" "{SCRIPTED}"')
    return WorkerClient(
        python, "unused", init={"replies": [_lines(r) for r in replies], **init}, array_parent=array_parent
    )


def _value(inputs: dict, params: dict) -> str:
    return identity_digest("fake-output@1", {"inputs": inputs, "params": params})


@pytest.mark.logic
def test_one_item_failure_isolated(worker: WorkerClient) -> None:
    """One failing item leaves its batch-mates' outputs intact."""
    out = worker.run([{"id": "a", "inputs": {}}, {"id": "b", "inputs": {}}], {"faults": {"fail_items": ["b"]}})
    assert out[0] == {"id": "a", "outputs": {"value": _value({}, {})}}
    assert out[1]["error"]["fault"] == "content" and out[1]["error"]["fault"] is FaultClass.CONTENT


@pytest.mark.logic
def test_oom_is_resource_fault(worker: WorkerClient) -> None:
    """An out-of-memory error in one item is a resource fault for that item."""
    out = worker.run([{"id": "a", "inputs": {}}, {"id": "b", "inputs": {}}], {"faults": {"oom_items": ["a"]}})
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
def test_malformed_requests_are_refused_on_the_host(worker: WorkerClient, items: list, params: object) -> None:
    """Item ids must be unique strings, inputs and params dicts."""
    with pytest.raises((ValueError, TypeError)):
        worker.run(items, params)  # type: ignore[arg-type]
    assert worker.alive


@pytest.mark.logic
def test_handshake_mismatch_refuses(array_parent: Path) -> None:
    """A worker whose runtime differs from the expected one is refused, naming both."""
    with pytest.raises(ProtocolMismatch) as info:
        WorkerClient(sys.executable, FAKE, init={}, array_parent=array_parent, expect_runtime="0" * 64)
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
        WorkerClient(sys.executable, module, init={}, array_parent=array_parent)
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
@pytest.mark.parametrize("init", [{"model": "x"}, {"faults": {"crash_after": "soon", "bogus": 1}}])
def test_load_failure_is_worker_fatal(array_parent: Path, init: dict) -> None:
    """A `fatal` during load raises `WorkerFatal`, an infrastructure fault carrying the error."""
    with pytest.raises(WorkerFatal) as info:
        WorkerClient(sys.executable, FAKE, init=init, array_parent=array_parent)
    assert info.value.error["fault"] == "content"
    assert classify(info.value) is FaultClass.INFRASTRUCTURE


@pytest.mark.logic
def test_reserved_init_key_is_refused(array_parent: Path) -> None:
    """`array_dir` in `init` is reserved."""
    with pytest.raises(ValueError):
        WorkerClient(sys.executable, FAKE, init={"array_dir": "x"}, array_parent=array_parent)


@pytest.mark.logic
def test_crash_mid_batch_is_worker_lost(array_parent: Path) -> None:
    """A worker that dies mid-request raises `WorkerLost` and is discarded."""
    client = WorkerClient(sys.executable, FAKE, init={"faults": {"crash_after": 1}}, array_parent=array_parent)
    with pytest.raises(WorkerLost) as info:
        client.run([{"id": "a", "inputs": {}}, {"id": "b", "inputs": {}}], {})
    assert classify(info.value) is FaultClass.INFRASTRUCTURE and info.value.__notes__
    assert info.value.returncode == EXIT_CRASH and str(EXIT_CRASH) in str(info.value)
    assert not client.alive and list(array_parent.iterdir()) == []
    with pytest.raises(WorkerLost):
        client.run([{"id": "a", "inputs": {}}], {})
    client.close()


@pytest.mark.logic
def test_request_timeout_is_resource_and_kills_the_worker(array_parent: Path) -> None:
    """A request past its timeout raises `TimeoutError` and the worker is killed."""
    client = WorkerClient(sys.executable, FAKE, init={}, array_parent=array_parent, request_timeout=0.5)
    with pytest.raises(TimeoutError) as info:
        client.run([{"id": "a", "inputs": {}}], {"faults": {"latency_s": 60}})
    assert classify(info.value) is FaultClass.RESOURCE and not client.alive
    client.close()


@pytest.mark.logic
def test_close_is_idempotent_and_context_managed(array_parent: Path) -> None:
    """`close` may be called twice; the client is a context manager."""
    with WorkerClient(sys.executable, FAKE, init={}, array_parent=array_parent) as client:
        assert client.alive
    client.close()
    assert not client.alive


def _ok(item: str, request: str = "$ID") -> dict:
    return {"id": request, "item": item, "outputs": {}}


def _done(n: int, request: str = "$ID") -> dict:
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
def test_malformed_responses_are_protocol_errors(array_parent: Path, reply: list) -> None:
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
def test_unknown_fatal_keeps_message(array_parent: Path, error: object, kept: tuple) -> None:
    """A `fatal` whose fault is unknown or whose `gap` is missing is `infrastructure` with a gap, and keeps its text.

    The probe: such a `fatal` became a `ProtocolError` and the worker's message was lost.
    """
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
def test_crash_mid_reply_is_worker_lost_with_returncode(array_parent: Path, tail: list, code: int) -> None:
    """A worker that exits after some of a reply's lines raises `WorkerLost` carrying its exit code.

    The probe: a crash mid-reply surfaced as `ProtocolError`, with no exit code.
    """
    client = _scripted(array_parent, [_ok("a"), *tail])
    with pytest.raises(WorkerLost) as info:
        client.run([{"id": "a", "inputs": {}}, {"id": "b", "inputs": {}}], {})
    assert info.value.returncode == code and f"exit code {code}" in str(info.value)
    assert not client.alive
    client.close()


@pytest.mark.logic
@posix_only
@pytest.mark.parametrize("wire", [True, 1.0, "2", 1, None])
def test_a34_hello_wire_must_be_an_int_on_the_host(array_parent: Path, wire: object) -> None:
    """The probe: a hello with `"wire": true` or `1.0` was accepted; a wire that is not the int 2 is refused."""
    hello = {"wire": wire, "runtime": runtime_identity(), "python": "", "numpy": "", "pid": 0}
    line = json.dumps({"hello": hello})
    with pytest.raises(ProtocolMismatch):
        WorkerClient(interpreter(array_parent.parent, f"echo '{line}'"), FAKE, init={}, array_parent=array_parent)


@pytest.mark.contract
def test_serve_is_total_through_a_real_worker(worker: WorkerClient) -> None:
    """One case of the serve fuzz property over a real worker: typed, oversize and unprintable faults stay per item."""
    ids = ["fail", "oom", "big", "mute", "ok"]
    faults = {"fail_items": ["fail"], "oom_items": ["oom"], "oversize_items": ["big"], "unprintable_items": ["mute"]}
    out = worker.run([{"id": i, "inputs": {}} for i in ids], {"faults": faults})
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

    def placed(ref: dict) -> dict:
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
        sys.executable, "backends.forking_backend_fixture", init={"pid_file": str(pid_file)}, request_timeout=2
    )
    pgid = client.hello["pid"]
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        client.run([{"id": "a", "inputs": {}}], {"mode": "hang"})
    client.close()
    assert time.monotonic() - t0 < 2 + SHUTDOWN_BOUND_S
    pids = recorded_pids(pid_file)
    assert len(pids) == 2 and not any(map(alive, pids))
    assert not group_alive(pgid)


@pytest.mark.limits
def test_a7_timeout_returns_though_a_grandchild_holds_stderr(tmp_path: Path, backends_importable: None) -> None:
    """The probe: a backend leaves a grandchild on stderr and hangs; `run` alone returns and kills it.

    The grandchildren start in `load`, so they exist before the request's timeout starts.
    """
    pid_file = tmp_path / "pids"
    init = {"pid_file": str(pid_file), "spawn_at_load": True}
    client = WorkerClient(sys.executable, "backends.forking_backend_fixture", init=init, request_timeout=2)
    assert len(recorded_pids(pid_file)) == 2
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        client.run([{"id": "a", "inputs": {}}], {"mode": "hang"})
    assert time.monotonic() - t0 < 2 + SHUTDOWN_BOUND_S
    assert not client.alive and not any(map(alive, recorded_pids(pid_file)))
    client.close()


@pytest.mark.logic
def test_backend_child_cannot_read_requests(backends_importable: None) -> None:
    """Children started with the default stdin read nothing, and every request reaches the worker."""
    with WorkerClient(sys.executable, "backends.stdin_reading_backend_fixture", init={}, request_timeout=10) as c:
        for i in range(3):
            assert c.run([{"id": str(i), "inputs": {}}], {})[0]["outputs"] == {"stdin_bytes": 0}


@pytest.mark.logic
def test_client_deletes_only_what_it_created(tmp_path: Path) -> None:
    """Files the client did not write survive, in its parent and in its own directory."""
    parent = tmp_path / "p"
    parent.mkdir()
    (parent / "thesis.txt").write_text("x")
    with WorkerClient(sys.executable, FAKE, init={}, array_parent=parent) as c:
        (c.array_dir / "notes.md").write_text("y")
        c.run(ITEMS, {})
    assert (parent / "thesis.txt").exists() and (c.array_dir / "notes.md").exists()
    assert [p.name for p in c.array_dir.iterdir()] == ["notes.md"]


@pytest.mark.logic
def test_a9_files_beside_the_array_dir_survive_runs_and_close(tmp_path: Path) -> None:
    """The probe: a directory holding `thesis.txt` and `notes.md` is left exactly as it was."""
    parent = tmp_path / "work"
    parent.mkdir()
    (parent / "thesis.txt").write_text("t")
    (parent / "notes.md").write_text("n")
    before = listing(parent)
    with WorkerClient(sys.executable, FAKE, init={"faults": {}}, array_parent=parent) as c:
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
            sys.executable,
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
        sys.executable, "backends.subdir_backend_fixture", init={"victim": str(victim)}, array_parent=parent
    ) as c:
        for _ in range(3):
            assert c.run(ITEMS, {})[0]["outputs"] == {"made": ARRAY_LIKE}
            assert [p.name for p in c.array_dir.iterdir()] == [ARRAY_LIKE]
    assert [p.name for p in c.array_dir.iterdir()] == [ARRAY_LIKE]
    assert sorted(p.name for p in (c.array_dir / ARRAY_LIKE).iterdir()) == ["scratch.bin", "victim"]
    assert (victim / "keep.npy").read_bytes() == b"k"


@pytest.mark.logic
def test_a22_backend_subdir_leaves_runs_and_the_worker_usable(
    tmp_path: Path, backends_importable: None, caplog: pytest.LogCaptureFixture
) -> None:
    """The probe: a backend directory not named like an array is left in place and logged; runs succeed."""
    with (
        caplog.at_level(logging.WARNING),
        WorkerClient(
            sys.executable, "backends.subdir_backend_fixture", init={"subdir": "scratch"}, array_parent=tmp_path
        ) as c,
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
        sys.executable, "backends.slow_reader_backend_fixture", init={}, array_parent=array_parent, request_timeout=2
    )
    client.run([{"id": "a", "inputs": {}}], {"mode": "stop_reading"})
    _wait_stopped(client.hello["pid"])
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        client.run([{"id": "a", "inputs": {}}], {"blob": "x" * 8 * 2**20})
    assert time.monotonic() - t0 < 2 + SHUTDOWN_BOUND_S
    assert not group_alive(client.hello["pid"])
    client.close()


@pytest.mark.logic
def test_a21_fifo_named_like_an_array_does_not_hang_the_host(array_parent: Path) -> None:
    """The probe: a response naming a FIFO at `<sha>.npy` fails its item at once instead of blocking."""
    sha = "a" * 64
    ref = {"__array__": {"path": f"$DIR/{sha}.npy", "sha256": sha, "dtype": "<f4", "shape": [1]}}
    client = _scripted(array_parent, [{"id": "$ID", "item": "a", "outputs": {"y": ref}}, _done(1)])
    os.mkfifo(client.array_dir / f"{sha}.npy")
    t0 = time.monotonic()
    [out] = client.run([{"id": "a", "inputs": {}}], {})
    assert time.monotonic() - t0 < 1
    assert out["error"]["type"] == "ArrayIntegrityError"
    client.close()


@pytest.mark.logic
def test_leaked_client_is_closed_with_a_resource_warning(array_parent: Path) -> None:
    """A client dropped without `close` warns, stops its worker and removes its directory."""
    client = WorkerClient(sys.executable, FAKE, init={}, array_parent=array_parent)
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
    client = WorkerClient(sys.executable, FAKE, init={}, array_parent=array_parent)
    pid = client.hello["pid"]
    os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)
    sent = _record_killpg(monkeypatch)
    client.close()
    assert [s for s in sent if s[2] != 0] == []


@pytest.mark.logic
def test_idle_crash_keeps_the_leader_unreaped_until_close(array_parent: Path) -> None:
    """A leader that dies while idle stays a zombie, holding its pid and group id, until `close()` reaps it."""
    client = WorkerClient(sys.executable, FAKE, init={}, array_parent=array_parent)
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
    """A worker that ignores the stop line gets SIGTERM with `TERM_GRACE_S` of the stop window left."""
    monkeypatch.setattr(worker_client, "STOP_GRACE_S", 2.0)
    monkeypatch.setattr(worker_client, "TERM_GRACE_S", 1.0)
    client = WorkerClient(sys.executable, "backends.slow_reader_backend_fixture", init={}, array_parent=array_parent)
    client.run([{"id": "a", "inputs": {}}], {"mode": "stop_reading"})
    _wait_stopped(client.hello["pid"])
    sent = _record_killpg(monkeypatch)
    t0 = time.monotonic()
    client.close()
    terms = [t for t, _, sig in sent if sig == signal.SIGTERM]
    assert terms and terms[0] - t0 < 2.0 - 1.0 + 0.5
    assert signal.SIGKILL not in [sig for _, _, sig in sent]


def _probe_fails(monkeypatch: pytest.MonkeyPatch, how: str) -> list[float]:
    """Route the leader probe through the `ps` path and make every run of it fail as `how`."""
    calls: list[float] = []

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        timeout = kwargs["timeout"]
        assert isinstance(timeout, float) and timeout <= worker_client.LEADER_PROBE_TIMEOUT_S
        calls.append(timeout)
        if how == "missing":
            raise FileNotFoundError(2, "No such file or directory", argv[0])
        time.sleep(timeout)
        raise subprocess.TimeoutExpired(argv, timeout)

    monkeypatch.setattr(worker_client, "_HAS_WAITID", False)
    monkeypatch.setattr(worker_client, "_PROBE_RUN", run)
    return calls


@pytest.mark.logic
@pytest.mark.parametrize("how", ["missing", "timeout"])
def test_unknown_leader_state_is_bounded_and_never_read_as_reaped(
    array_parent: Path, monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    """With `ps` missing or timing out after an idle crash, `close()` returns within the bound and signals nothing."""
    client = WorkerClient(sys.executable, FAKE, init={}, array_parent=array_parent)
    pid = client.hello["pid"]
    os.kill(pid, signal.SIGKILL)
    end = time.monotonic() + STOPPED_WAIT_S
    while client.alive:
        assert time.monotonic() < end
    calls = _probe_fails(monkeypatch, how)
    sent = _record_killpg(monkeypatch)
    t0 = time.monotonic()
    client.close()
    assert time.monotonic() - t0 < SHUTDOWN_BOUND_S
    assert calls and [s for s in sent if s[2] != 0] == []
    assert client.returncode == -signal.SIGKILL and not alive(pid)


@pytest.mark.logic
@pytest.mark.parametrize("how", ["missing", "timeout"])
def test_running_leader_is_stopped_without_asking_ps(
    tmp_path: Path, array_parent: Path, backends_importable: None, monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    """A leader whose exit has not been observed is held: a timed-out worker's group is stopped without `ps`."""
    pid_file = tmp_path / "pids"
    client = WorkerClient(
        sys.executable,
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
    assert time.monotonic() - t0 < 1 + SHUTDOWN_BOUND_S
    client.close()
    assert calls == [] and not group_alive(pgid) and not any(map(alive, recorded_pids(pid_file)))


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
def test_a10_exponentially_shared_input_fails_only_its_item(worker: WorkerClient) -> None:
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


def _fits(items: list[dict], params: dict) -> bool:
    request = {"id": "0" * 36, "items": items, "params": params}
    try:
        codec.dump_line(request, "request")
    except codec.LineTooLarge:
        return False
    return True


@pytest.mark.contract
@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(sizes=st.lists(st.integers(0, 1500), min_size=1, max_size=6), params_size=st.integers(0, 1500))
def test_oversize_is_value_too_large_per_item_where_attributable(
    worker: WorkerClient, monkeypatch: pytest.MonkeyPatch, sizes: list[int], params_size: int
) -> None:
    """An oversized request never raises a plain `ValueError`: what one item causes is that item's error.

    Against the codec's own line check: an item whose request alone cannot fit fails alone as
    `ValueTooLarge` (resource); params that cannot fit, or items that fit only alone, raise `ValueTooLarge`.
    """
    budget = 2048
    monkeypatch.setattr(codec, "MAX_LINE_BYTES", budget)
    monkeypatch.setattr(worker_client, "MAX_LINE_BYTES", budget)
    params = {"p": "y" * params_size}
    items = [{"id": str(i), "inputs": {"s": "x" * n}} for i, n in enumerate(sizes)]
    alone = [_fits([item], params) for item in items]
    kept = [item for item, ok in zip(items, alone) if ok]
    if not _fits([{"id": "", "inputs": {}}], params) or (kept and not _fits(kept, params)):
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
