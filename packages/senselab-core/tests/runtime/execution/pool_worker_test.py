"""Pool workers: quanta across classes, bisection of whole-request failures, renewal, SIGTERM and OOM kills."""

import json
import os
import shutil
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
from senselab_core.kernel.declare import TaskSpec
from senselab_core.runtime.call import RunConfig
from senselab_core.runtime.execution.executor import Admission, PlanFragment, ResultRef
from senselab_core.runtime.execution.in_process import job_key, node_fragment
from senselab_core.runtime.execution.job import Job, JobOutcome, execute
from senselab_core.runtime.execution.pool import split
from senselab_core.runtime.execution.pool_worker import PoolWorker, Send, serve_pool
from senselab_core.runtime.execution.queue import RUN_QUEUES, Claim, QueueSet, item_digest
from senselab_core.runtime.execution.run import prepare
from senselab_core.runtime.execution.workers import WorkerLoadFailed, WorkerSet
from senselab_core.runtime.keys import Placement
from senselab_core.runtime.locks import LeaseTimeout
from senselab_core.runtime.planner import Plan
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.sizing import Sample
from senselab_core.runtime.store import ResultStore
from senselab_core.runtime.worker_client import WorkerClient, WorkerLost, WorkerOOMKilled
from senselab_core.testing.interpreter import source_tree_python
from senselab_worker.faults import FaultClass, classify, error_record

TASK = scoring("pool_worker_test.score")
BROKEN = scoring("pool_worker_test.broken")
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
def stack(tmp_path: Path) -> Iterator[None]:
    """This module's registry, every env this interpreter under an MIT lock; the cache under `tmp_path`."""
    with packaged(REGISTRY, tmp_path / "cache"):
        yield
    WORKERS.close()


def _profile(**pool: object) -> ExecutionProfile:
    return ExecutionProfile.load("laptop", {**OVERRIDE, "pool": {**OVERRIDE["pool"], **pool}})


def _run(
    tmp_path: Path, profile: ExecutionProfile, texts: Sequence[str], task: TaskSpec = TASK
) -> tuple[Plan, ResultStore, Path]:
    """Prepare a run of `task`, one job per text, in `tmp_path/run`; return its plan, its store and its queues."""
    placed, _ = planned(task, profile, texts)
    store = ResultStore(tmp_path / "store")
    prepare(placed, tmp_path / "run", store)
    return placed, store, tmp_path / "run" / RUN_QUEUES


def _put(placed: Plan, pool_dir: Path, element: int | range, queue: str = "cpu", why: str = "") -> str:
    """Put the item of `element` (or of each in a range) on `queue`, its `why` set when given; return its digest."""
    admission = Admission(id=f"run-{placed.hash}", plan_hash=placed.hash)
    elements = [(e,) for e in element] if isinstance(element, range) else [(element,)]
    fragment = node_fragment(placed, admission, elements=elements)
    update: dict[str, object] = {"resource_class": queue, **({"why": why} if why else {})}
    fragment = fragment.model_copy(update={"placement": fragment.placement.model_copy(update=update)})
    QueueSet(pool_dir, placed.execution).queue(queue).put(fragment)
    time.sleep(0.01)
    return item_digest(fragment)


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
def test_a_worker_claims_one_item_at_a_time(tmp_path: Path) -> None:
    """A 1-job item and two 4-job items are claimed as 1, 4 and 4 jobs, never 1 + 4; each outcome holds its sample."""
    profile = _profile(quantum_items=4)
    placed, store, pool_dir = _run(tmp_path, profile, [f"t{i}" for i in range(9)])
    admission = Admission(id=f"run-{placed.hash}", plan_hash=placed.hash)
    queue = QueueSet(pool_dir, profile).queue("cpu")
    queue.put(node_fragment(placed, admission, elements=[(0,)]))
    queue.put_all(split(node_fragment(placed, admission, elements=[(i,) for i in range(1, 9)]), 4))
    send, sizes = _sender(store, profile), []

    def recorded(jobs: Sequence[Job], placement: Placement) -> list[JobOutcome]:
        sizes.append(len(jobs))
        return send(jobs, placement)

    worker = PoolWorker(pool_dir, ["cpu"], profile=profile, store=store, send=recorded)
    assert worker.serve() == 0
    assert sizes == [1, 4, 4] and len(worker.served) == 3 and queue.work() == 0
    samples = [Sample.model_validate(o["measure"]) for o in _outcomes(pool_dir, profile)]
    assert sorted((s.jobs, s.cold) for s in samples) == [(1, True), (4, False), (4, False)]
    assert all(s.input_s is None and s.peak_memory_bytes >= s.baseline_bytes and s.wall_s > 0 for s in samples)


@pytest.mark.logic
def test_failing_batch_bisected(tmp_path: Path) -> None:
    """One poisoned job of an 8-job item fails alone with `WorkerLost`; the other 7 commit their results."""
    profile = _profile(quantum_items=8)
    texts = [f"t{i}" for i in range(8)]
    texts[5] = "poison"
    placed, store, pool_dir = _run(tmp_path, profile, texts)
    _put(placed, pool_dir, range(8))
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

with packaged(registry(scoring("pool_worker_test.score")), cache):
    sys.exit(serve_pool(pool_dir, ["cpu"], profile=profile, store=store, send=send, join_s=0.2))
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


def _refs(pool_dir: Path, profile: ExecutionProfile) -> list[ResultRef]:
    return [ResultRef.model_validate(j) for o in _outcomes(pool_dir, profile) for j in o["jobs"]]


@pytest.mark.logic
def test_a_broken_env_costs_one_load_per_claim(tmp_path: Path) -> None:
    """A backend that cannot load fails a claim of 8 with one request and one load, each job `WorkerLoadFailed`."""
    from backends import failing_load_backend_fixture as failing  # noqa: PLC0415 - the module records loads

    failing.LOADS.clear()
    profile = _profile(quantum_items=8)
    with packaged(registry(BROKEN, worker="backends.failing_load_backend_fixture", name="broken"), tmp_path / "cache"):
        placed, store, pool_dir = _run(tmp_path, profile, [f"t{i}" for i in range(8)], BROKEN)
        _put(placed, pool_dir, range(8))
        send, sizes = _sender(store, profile), []

        def counted(jobs: Sequence[Job], placement: Placement) -> list[JobOutcome]:
            sizes.append(len(jobs))
            return send(jobs, placement)

        assert PoolWorker(pool_dir, ["cpu"], profile=profile, store=store, send=counted).serve() == 0
    refs = _refs(pool_dir, profile)
    assert sizes == [8] and len(failing.LOADS) == 1 and len(refs) == 8
    assert {(r.failure.type, r.failure.fault) for r in refs if r.failure} == {("WorkerLoadFailed", "infrastructure")}


@pytest.mark.logic
def test_a_load_failure_during_bisection_keeps_the_first_halfs_commits(tmp_path: Path) -> None:
    """A request lost as a whole is halved; the first half commits, and the second half, whose load fails, fails."""
    profile = _profile(quantum_items=8)
    placed, store, pool_dir = _run(tmp_path, profile, [f"t{i}" for i in range(8)])
    _put(placed, pool_dir, range(8))
    send, sizes = _sender(store, profile), []

    def flaky(jobs: Sequence[Job], placement: Placement) -> list[JobOutcome]:
        sizes.append(len(jobs))
        if len(sizes) == 1:
            raise WorkerLost("the worker died on its request", returncode=-11)
        if len(sizes) == 3:
            raise WorkerLoadFailed("WorkerFatal: the replacement worker did not load")
        return send(jobs, placement)

    assert PoolWorker(pool_dir, ["cpu"], profile=profile, store=store, send=flaky).serve() == 0
    refs = _refs(pool_dir, profile)
    done = [r for r in refs if r.failure is None]
    assert sizes == [8, 4, 4] and len(done) == 4 and len(refs) == 8
    assert all(store.outputs_of(r.key_digest, r.output_identity) is not None for r in done if r.output_identity)
    assert {r.failure.type for r in refs if r.failure} == {"WorkerLoadFailed"}


class _Abandoning(WorkerSet):
    """A worker set that records `abandon` instead of closing worker processes."""

    def __init__(self) -> None:
        super().__init__(1)
        self.abandoned = threading.Event()

    def abandon(self, thread: int) -> None:
        self.abandoned.set()


@pytest.mark.logic
def test_a_lost_claim_abandons_its_request(tmp_path: Path) -> None:
    """A claim broken mid-request: its worker set abandons the request, nothing is committed, nothing more is sent."""
    profile = _profile()
    placed, store, pool_dir = _run(tmp_path, profile, ["one", "two"])
    _put(placed, pool_dir, 0)
    _put(placed, pool_dir, 1, why="second")
    queue = QueueSet(pool_dir, profile).queue("cpu")
    workers, sends, thief, seen = _Abandoning(), [], [], []

    def stolen(jobs: Sequence[Job], placement: Placement) -> list[JobOutcome]:
        sends.append(len(jobs))
        for lease in (queue.path / "claims").iterdir():
            shutil.rmtree(lease)
        thief.append(queue.claim("thief", max_items=2))
        seen.append(workers.abandoned.wait(10))
        raise WorkerLost("closed by abandon")

    worker = PoolWorker(pool_dir, ["cpu"], profile=profile, store=store, workers=workers, send=stolen)
    assert worker.serve() == 0
    assert seen == [True] and sends == [1] and _outcomes(pool_dir, profile) == [] and thief[0] is not None
    assert len(thief[0].items) == 2
    thief[0].release()


@pytest.mark.logic
def test_halt_commits_finished_groups_and_releases_the_rest(tmp_path: Path) -> None:
    """Halted while a claim's second group runs, the worker commits the first group's item and releases the second."""
    profile = _profile()
    placed, store, pool_dir = _run(tmp_path, profile, ["fast", "slow"])
    _put(placed, pool_dir, 0)
    _put(placed, pool_dir, 1, why="second")
    send, release = _sender(store, profile), threading.Event()
    worker: PoolWorker

    def halting(jobs: Sequence[Job], placement: Placement) -> list[JobOutcome]:
        if _text(jobs[0]) == "slow":
            release.wait(10)
            raise WorkerLost("closed by abandon")
        outcomes = send(jobs, placement)
        worker.halt()
        return outcomes

    worker = PoolWorker(pool_dir, ["cpu"], profile=profile, store=store, send=halting, join_s=0.2)
    try:
        assert worker.serve() == 0
    finally:
        release.set()
    [outcome] = _outcomes(pool_dir, profile)
    queue = QueueSet(pool_dir, profile).queue("cpu")
    again = queue.claim("next", max_items=2)
    assert outcome["outcome"] == "done" and again is not None and len(again.items) == 1
    again.release()


@pytest.mark.logic
def test_a_lost_claim_releases_its_other_items_before_its_request_ends(tmp_path: Path) -> None:
    """A request that cannot be abandoned outlives the lease it lost; the claim's other item is claimable at once."""
    profile = _profile()
    placed, store, pool_dir = _run(tmp_path, profile, ["one", "two"])
    lost = _put(placed, pool_dir, 0)
    kept = _put(placed, pool_dir, 1, why="second")
    queue = QueueSet(pool_dir, profile).queue("cpu")
    taken: list[Claim] = []
    seen: list[str] = []

    def uninterruptible(jobs: Sequence[Job], placement: Placement) -> list[JobOutcome]:
        shutil.rmtree(queue.path / "claims" / f"{lost}.lease")
        until = time.monotonic() + 2.5 * profile.lease.stale_after_s
        while time.monotonic() < until and kept not in seen:
            claim = queue.claim("other", max_items=2)
            taken.extend([claim] if claim is not None else [])
            seen.extend(claim.items if claim is not None else ())
            time.sleep(0.02)
        time.sleep(max(0.0, until - time.monotonic()))
        raise WorkerLost("the request ended after its claim was lost")

    worker = PoolWorker(pool_dir, ["cpu"], profile=profile, store=store, workers=WORKERS, send=uninterruptible)
    try:
        assert worker.serve() == 0
    finally:
        for claim in taken:
            claim.release()
    assert kept in seen and _outcomes(pool_dir, profile) == []


@pytest.mark.logic
def test_a_busy_start_lease_gives_the_item_back_and_charges_nothing(tmp_path: Path) -> None:
    """A request whose start lease is busy gives its item back unreported, so no failure reaches the env's budget."""
    profile = _profile()
    placed, store, pool_dir = _run(tmp_path, profile, ["one"])
    _put(placed, pool_dir, 0)
    send, sends = _sender(store, profile), []

    def busy_once(jobs: Sequence[Job], placement: Placement) -> list[JobOutcome]:
        sends.append(len(jobs))
        if len(sends) == 1:
            raise LeaseTimeout("another holder is starting this server")
        return send(jobs, placement)

    worker = PoolWorker(pool_dir, ["cpu"], profile=profile, store=store, send=busy_once)
    assert worker.serve() == 0
    refs = _refs(pool_dir, profile)
    assert sends == [1, 1] and len(worker.served) == 2 and [r.failure for r in refs] == [None]
