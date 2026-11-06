"""Pool workers: quanta across classes, bisection of whole-request failures, renewal, SIGTERM and OOM kills."""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from pool_stack_fixture import packaged, planned, registry, scoring
from senselab_core.runtime.call import RunConfig
from senselab_core.runtime.execution.executor import Admission, PlanFragment, ResultRef
from senselab_core.runtime.execution.in_process import job_key, node_fragment
from senselab_core.runtime.execution.job import Job, JobOutcome, execute
from senselab_core.runtime.execution.pool_worker import PoolWorker, Send, serve_pool
from senselab_core.runtime.execution.queue import RUN_QUEUES, QueueSet
from senselab_core.runtime.execution.run import prepare
from senselab_core.runtime.execution.workers import WorkerSet
from senselab_core.runtime.keys import Placement
from senselab_core.runtime.planner import Plan
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.store import ResultStore
from senselab_core.runtime.worker_client import WorkerClient, WorkerLost, WorkerOOMKilled
from senselab_core.testing.interpreter import source_tree_python
from senselab_worker.faults import FaultClass, classify, error_record

TASK = scoring("pool_worker_test.score")
REGISTRY = registry(TASK)
WORKERS = WorkerSet(2)
OVERRIDE: dict[str, Any] = {
    "devices": ["cpu"],
    "resource_classes": {
        "cpu": {"device": "cpu", "mem_bytes": 2**30, "max_workers": 1},
        "cpu:b": {"device": "cpu", "mem_bytes": 2**31, "max_workers": 1},
    },
    "pool": {"poll_s": 0.05, "first_poll_s": 0.01, "quantum_items": 2, "quantum_s": 60},
    "lease": {"stale_after_s": 0.4, "renew_fraction": 0.2, "attribute_delay_s": 0},
}


@pytest.fixture(autouse=True)
def stack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """This module's registry, every env this interpreter under an MIT lock; the cache under `tmp_path`."""
    with packaged(REGISTRY, tmp_path / "cache", monkeypatch):
        yield
    WORKERS.close()


def _profile(**pool: object) -> ExecutionProfile:
    return ExecutionProfile.load("laptop", {**OVERRIDE, "pool": {**OVERRIDE["pool"], **pool}})


def _run(tmp_path: Path, profile: ExecutionProfile, texts: Sequence[str]) -> tuple[Plan, ResultStore, Path]:
    """Prepare a run of one job per text in `tmp_path/run`; return its plan, its store and its queues directory."""
    placed, _ = planned(TASK, profile, texts)
    store = ResultStore(tmp_path / "store")
    prepare(placed, tmp_path / "run", store)
    return placed, store, tmp_path / "run" / RUN_QUEUES


def _put(placed: Plan, pool_dir: Path, element: int, queue: str = "cpu") -> str:
    """Put the one-job fragment of `element` on `queue`; return the text it scores."""
    admission = Admission(id=f"run-{placed.hash}", plan_hash=placed.hash)
    fragment = node_fragment(placed, admission, elements=[(element,)])
    fragment = fragment.model_copy(
        update={"placement": fragment.placement.model_copy(update={"resource_class": queue})}
    )
    QueueSet(pool_dir, placed.execution).queue(queue).put(fragment)
    time.sleep(0.01)
    return str(placed.manifest.rows[element].id)


def _sender(store: ResultStore, profile: ExecutionProfile) -> Send:
    def send(jobs: Sequence[Job], placement: Placement) -> list[JobOutcome]:
        return execute(jobs, placement, store=store, run=RunConfig(), profile=profile, workers=WORKERS)

    return send


def _outcomes(pool_dir: Path, profile: ExecutionProfile, queue: str = "cpu") -> list[dict[str, Any]]:
    directory = QueueSet(pool_dir, profile).queue(queue).path / "outcome"
    return [json.loads(p.read_bytes()) for p in sorted(directory.iterdir()) if not p.name.startswith(".")]


def _text(job: Job) -> object:
    return next(iter(job.values.values()))


@pytest.mark.logic
def test_worker_yields_after_quantum_to_older_queue(tmp_path: Path) -> None:
    """After a quantum of 2 items on `cpu`, the worker claims the older item waiting on `cpu:b`, then goes back."""
    profile = _profile()
    placed, store, pool_dir = _run(tmp_path, profile, ["b", "a1", "a2", "a3", "a4"])
    _put(placed, pool_dir, 0, "cpu:b")
    for element in range(1, 5):
        _put(placed, pool_dir, element)
    other = QueueSet(pool_dir, profile).queue("cpu:b").claim("other")
    assert other is not None
    send = _sender(store, profile)

    def released_once(jobs: Sequence[Job], placement: Placement) -> list[JobOutcome]:
        other.release()
        return send(jobs, placement)

    worker = PoolWorker(pool_dir, ["cpu", "cpu:b"], profile=profile, store=store, send=released_once)
    assert worker.serve() == 0
    assert [name for name, _ in worker.served] == ["cpu", "cpu", "cpu:b", "cpu", "cpu"]
    assert len(_outcomes(pool_dir, profile)) == 4 and len(_outcomes(pool_dir, profile, "cpu:b")) == 1


@pytest.mark.logic
def test_failing_batch_bisected(tmp_path: Path) -> None:
    """One poisoned item among 8 claimed together fails alone with `WorkerLost`; the other 7 commit their results."""
    profile = _profile(quantum_items=8)
    texts = [f"t{i}" for i in range(8)]
    texts[5] = "poison"
    placed, store, pool_dir = _run(tmp_path, profile, texts)
    for element in range(8):
        _put(placed, pool_dir, element)
    send, sizes = _sender(store, profile), []

    def poisoned(jobs: Sequence[Job], placement: Placement) -> list[JobOutcome]:
        sizes.append(len(jobs))
        if any(_text(j) == "poison" for j in jobs):
            raise WorkerLost("the worker died on its request", returncode=-11)
        return send(jobs, placement)

    worker = PoolWorker(pool_dir, ["cpu"], profile=profile, store=store, send=poisoned)
    assert worker.serve() == 0
    refs = [ResultRef.model_validate(j) for o in _outcomes(pool_dir, profile) for j in o["jobs"]]
    failed = [r for r in refs if r.failure is not None]
    assert len(refs) == 8 and len(failed) == 1 and sizes == [8, 4, 4, 2, 1, 1, 2]
    assert failed[0].failure is not None and failed[0].failure.type == "WorkerLost"
    poison = node_fragment(placed, Admission(id="x", plan_hash=placed.hash), elements=[(5,)])
    assert failed[0].key_digest == job_key(placed, "a", poison.inputs[0], _placement(poison)).digest()
    assert all(store.outputs_of(r.key_digest, r.output_identity) is not None for r in refs if r.output_identity)


def _placement(fragment: PlanFragment) -> Any:  # noqa: ANN401 - the fragment's placement on queue `cpu`
    return fragment.placement.model_copy(update={"resource_class": "cpu"})


@pytest.mark.logic
def test_claim_renewed_during_long_item(tmp_path: Path) -> None:
    """An item running past twice the lease's stale bound stays claimed by its worker, which commits it."""
    profile = _profile()
    placed, store, pool_dir = _run(tmp_path, profile, ["slow"])
    _put(placed, pool_dir, 0)
    send, stolen = _sender(store, profile), []

    def slow(jobs: Sequence[Job], placement: Placement) -> list[JobOutcome]:
        until = time.monotonic() + 2.5 * profile.lease.stale_after_s
        while time.monotonic() < until:
            stolen.append(QueueSet(pool_dir, profile).queue("cpu").claim("thief"))
            time.sleep(0.05)
        return send(jobs, placement)

    worker = PoolWorker(pool_dir, ["cpu"], profile=profile, store=store, send=slow, name="owner")
    assert worker.serve() == 0
    [outcome] = _outcomes(pool_dir, profile)
    assert not any(stolen) and len(stolen) > 5 and (outcome["outcome"], outcome["worker"]) == ("done", "owner")


CHILD = """
import json, sys, time
from pathlib import Path
import pytest
from pool_stack_fixture import packaged, registry, scoring
from senselab_core.runtime.call import RunConfig
from senselab_core.runtime.execution.job import execute
from senselab_core.runtime.execution.pool_worker import serve_pool
from senselab_core.runtime.execution.workers import WorkerSet
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.store import ResultStore

pool_dir, cache, override = Path(sys.argv[1]), Path(sys.argv[2]), json.loads(sys.argv[3])
profile, store, workers = ExecutionProfile.load("laptop", override), ResultStore(Path(sys.argv[4])), WorkerSet(1)

def send(jobs, placement):
    if any(next(iter(j.values.values())) == "slow" for j in jobs):
        (pool_dir.parent / "slow-started").touch()
        time.sleep(600)
    return execute(jobs, placement, store=store, run=RunConfig(), profile=profile, workers=workers)

with packaged(registry(scoring("pool_worker_test.score")), cache, pytest.MonkeyPatch()):
    sys.exit(serve_pool(pool_dir, ["cpu"], profile=profile, store=store, send=send))
"""


@pytest.mark.logic
def test_sigterm_releases_claims(tmp_path: Path) -> None:
    """On SIGTERM a worker process commits the item that finished, releases the one running and exits 0, at once."""
    override = {
        **OVERRIDE,
        "pool": {**OVERRIDE["pool"], "quantum_items": 1, "poll_s": 5},
        "lease": {"stale_after_s": 300},
    }
    profile = ExecutionProfile.load("laptop", override)
    placed, _, pool_dir = _run(tmp_path, profile, ["fast", "slow"])
    _put(placed, pool_dir, 0)
    _put(placed, pool_dir, 1)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    args = [str(pool_dir), str(tmp_path / "cache"), json.dumps(override), str(tmp_path / "store")]
    child = subprocess.Popen([sys.executable, "-c", CHILD, *args], env=env)
    try:
        deadline = time.monotonic() + 60
        while not (tmp_path / "run" / "slow-started").exists() and time.monotonic() < deadline:
            assert child.poll() is None
            time.sleep(0.05)
        child.send_signal(signal.SIGTERM)
        assert child.wait(timeout=30) == 0
    finally:
        child.kill()
        child.wait()
    [outcome] = _outcomes(pool_dir, profile)
    assert outcome["outcome"] == "done"
    queue = QueueSet(pool_dir, profile).queue("cpu")
    assert not queue.claimed_any()
    again = queue.claim("next", max_items=2)
    assert again is not None and len(again.items) == 1
    again.release()


@pytest.mark.logic
@pytest.mark.parametrize(("planted", "raised"), [(False, WorkerLost), (True, WorkerOOMKilled)])
def test_oom_kill_is_resource_fault(tmp_path: Path, planted: bool, raised: type) -> None:
    """A worker killed by SIGKILL is `WorkerOOMKilled`, a resource fault, once its cgroup records an OOM kill."""
    events = tmp_path / "memory.events"
    events.write_text("oom 0\noom_kill 0\n")
    module, python = "backends.sigkill_backend_fixture", source_tree_python()
    client = WorkerClient(python, module, init={}, array_parent=tmp_path, memory_events=events)
    events.write_text(f"oom {int(planted)}\noom_kill {int(planted)}\n")
    with pytest.raises(raised) as info:
        client.run([{"id": "a", "inputs": {}}], {})
    assert getattr(info.value, "returncode", None) == -signal.SIGKILL and error_record(info.value)["gap"] is False
    assert classify(info.value) is (FaultClass.RESOURCE if planted else FaultClass.INFRASTRUCTURE)
    client.close()


@pytest.mark.logic
def test_serve_pool_restores_the_sigterm_handler(tmp_path: Path) -> None:
    """`serve_pool` on the main thread handles SIGTERM for the call only, and an idle pool returns 0."""
    profile = _profile()
    _, store, pool_dir = _run(tmp_path, profile, ["x"])
    before = signal.getsignal(signal.SIGTERM)
    assert serve_pool(pool_dir, ["cpu"], profile=profile, store=store) == 0
    assert signal.getsignal(signal.SIGTERM) is before
    assert threading.current_thread() is threading.main_thread()
