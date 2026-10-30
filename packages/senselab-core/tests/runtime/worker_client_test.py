"""The host side of the wire protocol, against the fake worker and a scripted one."""

import json
import logging
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy
import pytest
from senselab_core.runtime.worker_client import ProtocolMismatch, WorkerClient, WorkerFatal, WorkerLost
from senselab_worker import runtime_identity
from senselab_worker.canonical import canonical_npy_bytes, jcs_dumps
from senselab_worker.codec import to_wire
from senselab_worker.faults import FaultClass, ProtocolError, classify
from senselab_worker.hashing import sha256_bytes

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
def array_dir(tmp_path: Path) -> Iterator[Path]:
    """A host array directory that must be gone after the test."""
    path = tmp_path / "arrays"
    yield path
    assert not path.exists()


@pytest.fixture
def worker(array_dir: Path) -> Iterator[WorkerClient]:
    """A client over the fake worker."""
    client = WorkerClient(sys.executable, FAKE, init={}, array_dir=array_dir)
    yield client
    client.close()
    assert not client.alive


def _interpreter(directory: Path, body: str) -> Path:
    """Write an executable that stands in for the worker's interpreter and ignores its arguments."""
    path = directory / "interpreter"
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)
    return path


def _scripted(array_dir: Path, *replies: dict, **init: object) -> WorkerClient:
    python = _interpreter(array_dir.parent, f'exec "{sys.executable}" "{SCRIPTED}"')
    return WorkerClient(
        python, "unused", init={"replies": [json.dumps(r) for r in replies], **init}, array_dir=array_dir
    )


def _value(inputs: dict, params: dict) -> str:
    return sha256_bytes(jcs_dumps({"inputs": inputs, "params": params}))


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
def test_arrays_travel_and_request_files_are_removed(worker: WorkerClient, array_dir: Path) -> None:
    """An input array reaches the worker, and its file is deleted once the response is read."""
    a = numpy.linspace(0, 1, 32, dtype="<f4")
    [out] = worker.run([{"id": "a", "inputs": {"x": a}}], {"k": 1})
    assert out["outputs"]["value"] == _value({"x": sha256_bytes(canonical_npy_bytes(a))}, {"k": 1})
    assert list(array_dir.iterdir()) == []


@pytest.mark.logic
def test_unencodable_item_fails_alone_on_the_host(worker: WorkerClient) -> None:
    """An item with an object array fails as content before sending; its batch-mates run."""
    items = [{"id": "a", "inputs": {"x": numpy.array([object()])}}, {"id": "b", "inputs": {}}]
    out = worker.run(items, {})
    assert (out[0]["error"]["type"], out[0]["error"]["fault"]) == ("ValueError", FaultClass.CONTENT)
    assert out[1]["outputs"]["value"] == _value({}, {})


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
def test_handshake_mismatch_refuses(array_dir: Path) -> None:
    """A worker whose runtime differs from the expected one is refused, naming both."""
    with pytest.raises(ProtocolMismatch) as info:
        WorkerClient(sys.executable, FAKE, init={}, array_dir=array_dir, expect_runtime="0" * 64)
    assert "0" * 64 in str(info.value) and runtime_identity() in str(info.value)
    assert classify(info.value) is FaultClass.INFRASTRUCTURE and isinstance(info.value, RuntimeError)


@pytest.mark.logic
@posix_only
@pytest.mark.parametrize(("body", "stderr"), [("echo junk", None), ("echo oops >&2; exit 3", "oops")])
def test_non_worker_is_a_mismatch(array_dir: Path, body: str, stderr: str | None) -> None:
    """A first line that is not a hello, or no line at all, is a mismatch carrying the stderr tail."""
    with pytest.raises(ProtocolMismatch) as info:
        WorkerClient(_interpreter(array_dir.parent, body), FAKE, init={}, array_dir=array_dir)
    if stderr:
        assert any(stderr in note for note in info.value.__notes__)


@pytest.mark.logic
@pytest.mark.parametrize(
    ("module", "error_type"), [("no_such_backend", "ModuleNotFoundError"), ("json", "AttributeError")]
)
def test_unusable_backend_is_worker_fatal(array_dir: Path, module: str, error_type: str) -> None:
    """A backend module that cannot be imported, or lacks `load`, fails the load."""
    with pytest.raises(WorkerFatal) as info:
        WorkerClient(sys.executable, module, init={}, array_dir=array_dir)
    assert info.value.error["type"] == error_type


@pytest.mark.logic
def test_no_array_file_remains_when_every_item_fails_to_encode(worker: WorkerClient, array_dir: Path) -> None:
    """An array written before its item failed to encode is removed though nothing was sent."""
    out = worker.run([{"id": "a", "inputs": {"x": numpy.ones(3), "y": numpy.array([object()])}}], {})
    assert out[0]["error"]["fault"] is FaultClass.CONTENT
    assert list(array_dir.iterdir()) == []


@pytest.mark.logic
def test_no_array_file_remains_when_params_fail_to_encode(worker: WorkerClient, array_dir: Path) -> None:
    """An array written before the params failed to encode is removed."""
    with pytest.raises(ValueError):
        worker.run([{"id": "a", "inputs": {}}], {"w": numpy.ones(3), "z": float("nan")})
    assert list(array_dir.iterdir()) == []


@pytest.mark.logic
@pytest.mark.parametrize("init", [{"model": "x"}, {"faults": {"crash_after": "soon", "bogus": 1}}])
def test_load_failure_is_worker_fatal(array_dir: Path, init: dict) -> None:
    """A `fatal` during load raises `WorkerFatal`, an infrastructure fault carrying the error."""
    with pytest.raises(WorkerFatal) as info:
        WorkerClient(sys.executable, FAKE, init=init, array_dir=array_dir)
    assert info.value.error["fault"] == "content"
    assert classify(info.value) is FaultClass.INFRASTRUCTURE


@pytest.mark.logic
def test_reserved_init_key_is_refused(array_dir: Path) -> None:
    """`array_dir` in `init` is reserved."""
    with pytest.raises(ValueError):
        WorkerClient(sys.executable, FAKE, init={"array_dir": "x"}, array_dir=array_dir)


@pytest.mark.logic
def test_crash_mid_batch_is_worker_lost(array_dir: Path) -> None:
    """A worker that dies mid-request raises `WorkerLost` and is discarded."""
    client = WorkerClient(sys.executable, FAKE, init={"faults": {"crash_after": 1}}, array_dir=array_dir)
    with pytest.raises(WorkerLost) as info:
        client.run([{"id": "a", "inputs": {}}, {"id": "b", "inputs": {}}], {})
    assert classify(info.value) is FaultClass.INFRASTRUCTURE and info.value.__notes__
    assert not client.alive and not array_dir.exists()
    with pytest.raises(WorkerLost):
        client.run([{"id": "a", "inputs": {}}], {})
    client.close()


@pytest.mark.logic
def test_request_timeout_is_resource_and_kills_the_worker(array_dir: Path) -> None:
    """A request past its timeout raises `TimeoutError` and the worker is killed."""
    client = WorkerClient(sys.executable, FAKE, init={}, array_dir=array_dir, request_timeout=0.5)
    with pytest.raises(TimeoutError) as info:
        client.run([{"id": "a", "inputs": {}}], {"faults": {"latency_s": 60}})
    assert classify(info.value) is FaultClass.RESOURCE and not client.alive
    client.close()


@pytest.mark.logic
def test_close_is_idempotent_and_context_managed(array_dir: Path) -> None:
    """`close` may be called twice; the client is a context manager."""
    with WorkerClient(sys.executable, FAKE, init={}, array_dir=array_dir) as client:
        assert client.alive
    client.close()
    assert not client.alive


def _ok(item: str) -> dict:
    return {"id": item, "outputs": {}}


@pytest.mark.logic
@pytest.mark.parametrize(
    "reply",
    [
        {"id": "other", "results": [_ok("a")]},
        {"id": "$ID", "results": []},
        {"id": "$ID", "results": [_ok("b"), _ok("a")]},
        {"id": "$ID", "results": [_ok("a")]},
        {"id": "$ID", "results": [{"id": "a"}, _ok("b")]},
        {"id": "$ID", "results": [{**_ok("a"), "error": {}}, _ok("b")]},
        {"id": "$ID", "results": [{"id": "a", "error": {"fault": "weird", "gap": False}}, _ok("b")]},
        {"ready": {}},
    ],
)
def test_malformed_responses_are_protocol_errors(array_dir: Path, reply: dict) -> None:
    """A response with the wrong id, count, order or entry shape raises `ProtocolError`."""
    client = _scripted(array_dir, reply)
    with pytest.raises(ProtocolError):
        client.run([{"id": "a", "inputs": {}}, {"id": "b", "inputs": {}}], {})
    assert not client.alive
    client.close()


@pytest.mark.logic
def test_fatal_during_a_request_is_worker_fatal(array_dir: Path) -> None:
    """A `fatal` line in place of a response raises `WorkerFatal`, whatever fault it names."""
    error = {"type": "MemoryError", "message": "m", "fault": "resource", "traceback": "", "gap": False}
    client = _scripted(array_dir, {"error": error, "fatal": True})
    with pytest.raises(WorkerFatal) as info:
        client.run([{"id": "a", "inputs": {}}], {})
    assert info.value.error == error and classify(info.value) is FaultClass.INFRASTRUCTURE
    client.close()


@pytest.mark.logic
def test_output_arrays_decode_per_item_and_init_files_survive(array_dir: Path, tmp_path: Path) -> None:
    """A damaged output array fails only its item; request and response files go, init files stay."""
    staging = tmp_path / "staging"
    staging.mkdir()
    good, bad = numpy.arange(4, dtype="<i8"), numpy.ones(3)
    good_ref, bad_ref = to_wire(good, array_dir=staging), to_wire(bad, array_dir=staging)
    bad_ref["__array__"]["sha256"] = "e" * 64

    def placed(ref: dict) -> dict:
        moved = dict(ref["__array__"], path=str(array_dir / Path(ref["__array__"]["path"]).name))
        return {"__array__": moved}

    reply = {
        "id": "$ID",
        "results": [{"id": "a", "outputs": {"y": placed(good_ref)}}, {"id": "b", "outputs": {"y": placed(bad_ref)}}],
    }
    weights = numpy.full(5, 7.0)
    client = _scripted(array_dir, reply, w=weights)
    init_files = set(array_dir.iterdir())
    for path in staging.iterdir():
        path.replace(array_dir / path.name)
    out = client.run([{"id": "a", "inputs": {"x": numpy.zeros(2)}}, {"id": "b", "inputs": {}}], {})
    assert numpy.array_equal(out[0]["outputs"]["y"], good)
    assert (out[1]["error"]["type"], out[1]["error"]["fault"]) == ("ArrayIntegrityError", FaultClass.INFRASTRUCTURE)
    assert set(array_dir.iterdir()) == init_files
    client.close()


@pytest.mark.logic
def test_classification_gap_is_reported(array_dir: Path, caplog: pytest.LogCaptureFixture) -> None:
    """An error marked as a classification gap is logged by the host."""
    error = {"type": "OSError", "message": "m", "fault": "infrastructure", "traceback": "", "gap": True}
    client = _scripted(array_dir, {"id": "$ID", "results": [{"id": "a", "error": error}]})
    with caplog.at_level(logging.WARNING):
        out = client.run([{"id": "a", "inputs": {}}], {})
    assert out[0]["error"]["gap"] is True and "OSError" in caplog.text
    client.close()


def test_leak_fixture_records_a_child_until_it_is_waited_on(no_leaked_processes: list[subprocess.Popen[bytes]]) -> None:
    """An exited child nobody waited on still has no returncode, which is what the fixture rejects."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    assert proc in no_leaked_processes
    assert proc.returncode is None
    proc.wait()
    assert proc.returncode == 0
