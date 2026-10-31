"""The host side of the wire protocol, against the fake worker and a scripted one."""

import gc
import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import numpy
import pytest
import senselab_worker
from lifecycle_probes_fixture import alive, gate, group_alive, listing, recorded_pids, state
from senselab_core.runtime.worker_client import (
    SHUTDOWN_BOUND_S,
    ConcurrentUse,
    ProtocolMismatch,
    WorkerClient,
    WorkerFatal,
    WorkerLost,
)
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


def _interpreter(directory: Path, body: str) -> Path:
    """Write an executable that stands in for the worker's interpreter and ignores its arguments."""
    path = directory / "interpreter"
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)
    return path


def _scripted(array_parent: Path, *replies: dict, **init: object) -> WorkerClient:
    python = _interpreter(array_parent.parent, f'exec "{sys.executable}" "{SCRIPTED}"')
    return WorkerClient(
        python, "unused", init={"replies": [json.dumps(r) for r in replies], **init}, array_parent=array_parent
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
def test_arrays_travel_and_request_files_are_removed(worker: WorkerClient) -> None:
    """An input array reaches the worker, and its file is deleted once the response is read."""
    a = numpy.linspace(0, 1, 32, dtype="<f4")
    [out] = worker.run([{"id": "a", "inputs": {"x": a}}], {"k": 1})
    assert out["outputs"]["value"] == _value({"x": sha256_bytes(canonical_npy_bytes(a))}, {"k": 1})
    assert list(worker.array_dir.iterdir()) == []


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
def test_handshake_mismatch_refuses(array_parent: Path) -> None:
    """A worker whose runtime differs from the expected one is refused, naming both."""
    with pytest.raises(ProtocolMismatch) as info:
        WorkerClient(sys.executable, FAKE, init={}, array_parent=array_parent, expect_runtime="0" * 64)
    assert "0" * 64 in str(info.value) and runtime_identity() in str(info.value)
    assert classify(info.value) is FaultClass.INFRASTRUCTURE and isinstance(info.value, RuntimeError)


@pytest.mark.logic
@posix_only
@pytest.mark.parametrize(("body", "stderr"), [("echo junk", None), ("echo oops >&2; exit 3", "oops")])
def test_non_worker_is_a_mismatch(array_parent: Path, body: str, stderr: str | None) -> None:
    """A first line that is not a hello, or no line at all, is a mismatch carrying the stderr tail."""
    with pytest.raises(ProtocolMismatch) as info:
        WorkerClient(_interpreter(array_parent.parent, body), FAKE, init={}, array_parent=array_parent)
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
def test_malformed_responses_are_protocol_errors(array_parent: Path, reply: dict) -> None:
    """A response with the wrong id, count, order or entry shape raises `ProtocolError`."""
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
def test_output_arrays_decode_per_item_and_init_files_survive(array_parent: Path, tmp_path: Path) -> None:
    """A damaged output array fails only its item; request and response files go, init files stay."""
    staging = tmp_path / "staging"
    staging.mkdir()
    good, bad = numpy.arange(4, dtype="<i8"), numpy.ones(3)
    good_ref, bad_ref = to_wire(good, array_dir=staging), to_wire(bad, array_dir=staging)
    bad_ref["__array__"]["sha256"] = "e" * 64

    def placed(ref: dict) -> dict:
        moved = dict(ref["__array__"], path="$DIR/" + Path(ref["__array__"]["path"]).name)
        return {"__array__": moved}

    reply = {
        "id": "$ID",
        "results": [{"id": "a", "outputs": {"y": placed(good_ref)}}, {"id": "b", "outputs": {"y": placed(bad_ref)}}],
    }
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
    client = _scripted(array_parent, {"id": "$ID", "results": [{"id": "a", "error": error}]})
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
    """The probe: a backend leaves a grandchild on stderr and hangs; `run` alone returns and kills it."""
    pid_file = tmp_path / "pids"
    client = WorkerClient(
        sys.executable, "backends.forking_backend_fixture", init={"pid_file": str(pid_file)}, request_timeout=2
    )
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
            sys.executable, "backends.subdir_backend_fixture", init={"gate_port": g.port}, array_parent=array_parent
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
        assert first and first[0][0]["outputs"] == {"made": "f" * 64 + ".npy"}  # type: ignore[index]
        assert c.run([{"id": "c", "inputs": {}}], {})[0]["outputs"]


@pytest.mark.logic
def test_backend_subdir_in_array_dir_does_not_fail_runs(
    array_parent: Path, tmp_path: Path, backends_importable: None
) -> None:
    """A directory a backend makes under an array name is removed as a tree, never following its symlinks."""
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.npy").write_bytes(b"k")
    with WorkerClient(
        sys.executable, "backends.subdir_backend_fixture", init={"victim": str(victim)}, array_parent=array_parent
    ) as c:
        for _ in range(3):
            assert c.run(ITEMS, {})[0]["outputs"] == {"made": "f" * 64 + ".npy"}
            assert list(c.array_dir.iterdir()) == []
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
    client = _scripted(array_parent, {"id": "$ID", "results": [{"id": "a", "outputs": {"y": ref}}]})
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
