"""`Pool`: fragments put on a run's queues, served by a fake substrate's worker thread through `execute`."""

import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy
import pytest
import soundfile
from lease_clock_fixture import ManualClock, newest
from pool_stack_fixture import packaged, planned, registry, scoring, unresolving
from senselab_core.cli import main as cli_main
from senselab_core.kernel import prov as kprov
from senselab_core.kernel.dataset import Record, ValueInput
from senselab_core.kernel.graph import Edge, Graph, Node
from senselab_core.kernel.identity import InputIdentity
from senselab_core.runtime.audio_io import open_audio
from senselab_core.runtime.call import RunConfig, TaskCall, bind_input
from senselab_core.runtime.clock import Clock, SystemClock
from senselab_core.runtime.datasets import RecordSet
from senselab_core.runtime.execution.epoch import Epoch, StaleCoordinator
from senselab_core.runtime.execution.events import EventLog, read_events, run_dir_of, status_from
from senselab_core.runtime.execution.executor import Admission, FaultRecord, PlanFragment, QueueHeld, ResultRef, follow
from senselab_core.runtime.execution.in_process import InProcess, job_key, node_fragment
from senselab_core.runtime.execution.job import Failed, JobOutcome
from senselab_core.runtime.execution.pool import Pool, larger_class, report, split
from senselab_core.runtime.execution.queue import RUN_QUEUES, Claim, QueueSet
from senselab_core.runtime.execution.records import add_plan_entity
from senselab_core.runtime.execution.run import INPUTS_FILE, RECORD_ATTEMPTS, Run, prepare, record_sources
from senselab_core.runtime.execution.substrates import QueueReady, WorkerHandle, mark_ready
from senselab_core.runtime.execution.workers import WorkerSet
from senselab_core.runtime.execution_settings import JobLease
from senselab_core.runtime.locks import LeaseUnavailable
from senselab_core.runtime.planner import Plan, plan, resume
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.provenance import ProvStore
from senselab_core.runtime.store import ResultStore
from senselab_worker.canonical import jcs_dumps
from syscall_faults_fixture import await_true, displace_before_proof, pause_replaces

CLUSTER = (
    next(p for p in Path(__file__).resolve().parents if p.name == "packages")
    / "senselab-core/tests/data/profiles/cluster.yaml"
)

SCORE = scoring("pool_test.score")
HEAR = SCORE.model_copy(update={"name": "pool_test.hear", "inputs": {"audio": "Audio"}})
REGISTRY = registry(SCORE, HEAR)
REGISTRY.add_backend(REGISTRY.backends(SCORE.name)[0].model_copy(update={"name": "alt"}))
CLASSES = {
    "cpu": {"device": "cpu", "mem_bytes": 2**30, "max_workers": 1},
    "cpu:large": {"device": "cpu", "mem_bytes": 2**33, "max_workers": 1},
}
WORKERS = WorkerSet(4)
CONTENT = {"content_attempts": 2, "infrastructure_replacements": 6, "resource_replacements": 1}
GAP = FaultRecord(fault="infrastructure", type="RuntimeError", message="unclassified", gap=True, charged=False)
CHARGED = FaultRecord(fault="content", type="ValueError", message="bad input", gap=False, charged=True)
LOST = FaultRecord(fault="infrastructure", type="WorkerLost", message="gone", gap=False, charged=False)
OOM = FaultRecord(fault="resource", type="ResourceFault", message="out of memory", gap=False, charged=False)


def _profile(**faults: int) -> ExecutionProfile:
    override = {
        "devices": ["cpu"],
        "resource_classes": CLASSES,
        "pool": {"poll_s": 0.2, "first_poll_s": 0.01},
        "faults": faults,
    }
    return ExecutionProfile.load("laptop", override)


@pytest.fixture(autouse=True)
def stack(tmp_path: Path) -> Iterator[None]:
    """This module's registry, every env this interpreter under an MIT lock; the cache under `tmp_path`."""
    with packaged(REGISTRY, tmp_path / "cache"):
        yield
    WORKERS.close()


def _planned(profile: ExecutionProfile, texts: Sequence[str]) -> tuple[Plan, dict[InputIdentity, object]]:
    return planned(SCORE, profile, texts)


Act = Callable[[PlanFragment], "FaultRecord | ResultStore | None"]


class FakeWorkers:
    """A `Substrate` whose thread claims each queue it serves, logs each job `claimed` and runs it through `execute`.

    `act` decides per fragment: a `FaultRecord` fails every job with it, a `ResultStore` executes into that store,
    None executes into `store`, whose provenance it writes to the run after each claim, as `PoolWorker` does.
    `served` holds each fragment claimed, in order. The thread claims only while `ready()` is true, which a test
    uses to hold it back until the coordinator has acted on what it served.
    """

    def __init__(self, run_dir: Path, store: ResultStore, placed: Plan, values: Mapping[InputIdentity, object]) -> None:
        """Serve the queues of `run_dir` for `placed`, its graph inputs `values`."""
        self.epochs = Epoch(run_dir)
        self.queues = QueueSet(run_dir / RUN_QUEUES, placed.execution, epoch=self.epochs.current)
        self.run_dir = run_dir
        self.store, self.placed, self.values = store, placed, values
        self.act: Act = lambda fragment: None
        self.served: list[PlanFragment] = []
        self.ready: Callable[[], bool] = lambda: True
        self.names: set[str] = set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, resource_class: str, count: int) -> list[WorkerHandle]:  # noqa: ARG002
        """Serve `resource_class` from the worker thread, started on the first call."""
        self.names.add(resource_class)
        mark_ready(self.run_dir, {resource_class: QueueReady(None, {})})
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return [WorkerHandle(resource_class, f"thread-{resource_class}")]

    def live(self) -> list[WorkerHandle]:
        """Return a handle for each queue the thread serves."""
        alive = self._thread is not None and self._thread.is_alive()
        return [WorkerHandle(name, f"thread-{name}") for name in sorted(self.names)] if alive else []

    def running(self, resource_class: str) -> int:
        """Return 1 for a queue the thread serves."""
        return int(resource_class in self.names and self._thread is not None and self._thread.is_alive())

    def stop(self, handles: Sequence[WorkerHandle] = ()) -> None:  # noqa: ARG002
        """Stop the worker thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(30)

    def _loop(self) -> None:
        while not self._stop.is_set():
            claims = []
            if self.ready():
                claims = [c for name in sorted(self.names) if (c := self.queues.queue(name).claim("fake")) is not None]
            for claim in claims:
                self._serve(claim)
            if not claims:
                time.sleep(0.01)

    def _log(self) -> EventLog:
        """Return a log of the current epoch's attempt (`Epoch.attempt`), `a<epoch>` when it recorded none."""
        epoch = self.epochs.current()
        attempt, ordinal = self.epochs.attempt(epoch) or (f"a{epoch}", max(epoch, 1))
        return EventLog(self.run_dir, attempt, ordinal, epoch=epoch)

    def _serve(self, claim: Claim) -> None:
        log = self._log()
        for digest, fragment in zip(claim.items, claim.fragments, strict=True):
            for job in fragment.jobs:
                log.record("claimed", job.node, job=job)
            self.served.append(fragment)
            report(claim, digest, self._outcomes(fragment))
        add_plan_entity(self.store.provenance, self.placed.hash)
        self.store.provenance.write(self.run_dir)

    def _outcomes(self, fragment: PlanFragment) -> list[JobOutcome]:
        act = self.act(fragment)
        if isinstance(act, FaultRecord):
            keys = [job_key(self.placed, fragment.node, inputs).digest() for inputs in fragment.inputs]
            return [Failed(job, key, act) for job, key in zip(fragment.jobs, keys, strict=True)]
        executor = InProcess(store=act or self.store, profile=self.placed.execution, run=RunConfig(), workers=WORKERS)
        admission = executor.admit(self.placed, self.values)
        handle = executor.submit(fragment.model_copy(update={"admission": admission.id}))
        return executor.outcomes(handle)


def _stack(
    tmp_path: Path, profile: ExecutionProfile, texts: Sequence[str]
) -> tuple[Plan, dict[InputIdentity, object], ResultStore, FakeWorkers]:
    placed, values = _planned(profile, texts)
    store = ResultStore(tmp_path / "store")
    return placed, values, store, FakeWorkers(tmp_path / "run", store, placed, values)


def _pool(  # noqa: PLR0913 - the run, its parts, and the epoch and clock a restart or a frozen clock sets
    tmp_path: Path,
    store: ResultStore,
    workers: FakeWorkers | Any,  # noqa: ANN401
    profile: ExecutionProfile,
    epoch: int | None = None,
    clock: Clock = SystemClock(),  # noqa: B008 - stateless
) -> Pool:
    """Return a pool of the next epoch over `tmp_path/run` (of `epoch`, as a restart within it, when given)."""
    run_dir = tmp_path / "run"
    epochs = Epoch(run_dir)
    epoch = epochs.take() if epoch is None else epoch
    return Pool(
        run_dir,
        queues=QueueSet(run_dir / RUN_QUEUES, profile, epoch=epochs.current),
        store=store,
        events=EventLog(run_dir, f"a{epoch}", epoch, epoch=epoch),
        profile=profile,
        substrate=workers,
        epoch=epoch,
        clock=clock,
        resolver=unresolving(profile),
    )


def _all(pool: Pool, placed: Plan, values: Mapping[InputIdentity, object]) -> list[ResultRef]:
    fragment = node_fragment(placed, pool.admit(placed, values))
    return list(follow(pool, pool.submit(fragment)))


def _kinds(run_dir: Path) -> list[str]:
    return [e.kind for e in read_events(run_dir) if e.kind not in ("claimed", "placement")]


@pytest.mark.contract
def test_two_coordinators_one_result_per_key(tmp_path: Path) -> None:
    """Two coordinators put one plan: one item, executed once, one result per key; the older one stops waiting."""
    profile = _profile(**CONTENT)
    placed, values, store, workers = _stack(tmp_path, profile, ["one", "two"])
    first = _pool(tmp_path, store, workers, profile)
    early = first.submit(node_fragment(placed, first.admit(placed, values)))
    second = _pool(tmp_path, store, workers, profile)
    late = second.submit(node_fragment(placed, second.admit(placed, values)))
    try:
        refs = list(follow(second, late))
        with pytest.raises(StaleCoordinator):
            list(follow(first, early))
    finally:
        workers.stop()
    assert all(r.failure is None for r in refs) and len(refs) == 2 and len(workers.served) == 1
    for ref in refs:
        entries = list((store.root).rglob(f"{ref.key_digest}/*"))
        assert len([e for e in entries if e.parent.name == ref.key_digest]) == 1


@pytest.mark.logic
def test_stale_coordinator_stops_after_new_epoch(tmp_path: Path) -> None:
    """Once a newer epoch is taken, the older coordinator's submit raises `StaleCoordinator` and puts nothing."""
    profile = _profile(**CONTENT)
    placed, values, store, workers = _stack(tmp_path, profile, ["one"])
    stale = _pool(tmp_path, store, workers, profile)
    assert Epoch(tmp_path / "run").take() == stale.epoch + 1
    with pytest.raises(StaleCoordinator) as raised:
        stale.submit(node_fragment(placed, stale.admit(placed, values)))
    assert (raised.value.mine, raised.value.current) == (1, 2)
    assert stale.queues.queue("cpu").depth() == 0 and not workers.names


@pytest.mark.contract
def test_content_fault_dead_after_budget(tmp_path: Path) -> None:
    """A content fault is charged: retried once, then dead at `content_attempts` 2, its failure read from `dead/`."""
    profile = _profile(**CONTENT)
    placed, values, store, workers = _stack(tmp_path, profile, ["fail"])
    pool = _pool(tmp_path, store, workers, profile)
    try:
        [ref] = _all(pool, placed, values)
    finally:
        workers.stop()
    assert ref.failure is not None and ref.failure.fault == "content" and ref.failure.charged
    assert [f.attempt for f in workers.served] == [1, 2]
    dead = json.loads((tmp_path / "run" / "dead" / f"{ref.key_digest}.json").read_text())
    assert FaultRecord.model_validate(dead["failure"]) == ref.failure
    assert _kinds(tmp_path / "run") == ["failed", "queued"]


@pytest.mark.contract
def test_gap_fault_never_dead(tmp_path: Path) -> None:
    """A gap fault is never charged: with one content attempt, two gaps are retried and the third attempt completes."""
    profile = _profile(content_attempts=1, infrastructure_replacements=6, resource_replacements=0)
    placed, values, store, workers = _stack(tmp_path, profile, ["one"])
    workers.act = lambda fragment: GAP if fragment.attempt < 3 else None
    pool = _pool(tmp_path, store, workers, profile)
    try:
        [ref] = _all(pool, placed, values)
    finally:
        workers.stop()
    assert ref.failure is None and store.outputs_of(ref.key_digest, str(ref.output_identity)) is not None
    assert [g.type for _, g in pool.gaps] == ["RuntimeError", "RuntimeError"] and not pool.broken
    assert not any((tmp_path / "run").glob("dead/*"))
    assert _kinds(tmp_path / "run") == ["failed", "queued", "failed", "queued"]


@pytest.mark.contract
def test_resource_fault_moves_to_larger_class_uncharged(tmp_path: Path) -> None:
    """A resource fault moves the job to `cpu:large` uncharged; with no move left it is a resource failure."""
    profile = _profile(content_attempts=1, infrastructure_replacements=6, resource_replacements=1)
    placed, values, store, workers = _stack(tmp_path, profile, ["one"])
    workers.act = lambda fragment: OOM if fragment.placement.resource_class == "cpu" else None
    try:
        [ref] = _all(_pool(tmp_path, store, workers, profile), placed, values)
    finally:
        workers.stop()
    assert ref.failure is None
    assert [(f.placement.resource_class, f.attempt) for f in workers.served] == [("cpu", 1), ("cpu:large", 2)]

    stuck = _profile(content_attempts=1, infrastructure_replacements=6, resource_replacements=0)
    placed, values, store, workers = _stack(tmp_path / "again", stuck, ["one"])
    workers.act = lambda fragment: OOM
    try:
        [failed] = _all(_pool(tmp_path / "again", store, workers, stuck), placed, values)
    finally:
        workers.stop()
    assert failed.failure == OOM and len(workers.served) == 1
    assert (tmp_path / "again" / "run" / "dead" / f"{failed.key_digest}.json").is_file()


@pytest.mark.contract
def test_a_load_oom_moves_to_the_larger_class_with_one_worker_start_per_claim(tmp_path: Path) -> None:
    """A backend that runs out of memory at load is re-placed on `cpu:large`, uncharged, one claim per class."""
    from fault_class_test import host_load_failure  # noqa: PLC0415 - a real checkout's WorkerLoadFailed

    load = FaultRecord.from_exception(host_load_failure(MemoryError("CUDA out of memory")))
    assert (load.fault, load.charged) == ("resource", False)
    profile = _profile(content_attempts=1, infrastructure_replacements=6, resource_replacements=1)
    placed, values, store, workers = _stack(tmp_path, profile, ["one"])
    workers.act = lambda fragment: load if fragment.placement.resource_class == "cpu" else None
    try:
        [ref] = _all(_pool(tmp_path, store, workers, profile), placed, values)
    finally:
        workers.stop()
    assert ref.failure is None
    assert [(f.placement.resource_class, f.attempt) for f in workers.served] == [("cpu", 1), ("cpu:large", 2)]


@pytest.mark.contract
def test_broken_env_reported_once(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Two infrastructure faults of one env break it: reported once with its cause, held, no job failed or dead."""
    profile = _profile(content_attempts=2, infrastructure_replacements=2, resource_replacements=0)
    placed, values, store, workers = _stack(tmp_path, profile, ["one", "two", "three"])
    workers.act = lambda fragment: LOST
    pool = _pool(tmp_path, store, workers, profile)
    admission = pool.admit(placed, values)
    handles = [pool.submit(node_fragment(placed, admission, elements=[(i,)])) for i in range(3)]
    try:
        for handle in handles:
            with pytest.raises(QueueHeld) as raised:
                list(follow(pool, handle))
            assert (raised.value.queue, raised.value.failures, raised.value.reason) == ("cpu", 2, LOST.message)
            assert LOST.message in str(raised.value)
    finally:
        workers.stop()
    [broken] = pool.broken.values()
    assert broken.env == "host" and broken.failures == 2 and broken.fault == LOST
    [logged] = [r.getMessage() for r in caplog.records if "is broken" in r.getMessage()]
    assert logged.endswith(f": {LOST.message}")
    events = read_events(tmp_path / "run")
    assert [e.kind for e in events if e.kind == "failed"] == ["failed", "failed"]
    assert status_from(events).counts["failed"] == 0
    assert [b.env for b in status_from(events, broken_after=2).broken] == ["host"]
    queue = pool.queues.queue("cpu")
    assert queue.held("host") == pool.epoch and queue.claim("w") is None
    assert not any((tmp_path / "run").glob("dead/*"))
    with pytest.raises(QueueHeld, match=LOST.message):
        pool.submit(node_fragment(placed, admission, elements=[(0,)]))


class _Idle:
    """A `Substrate` that starts no worker."""

    def start(self, resource_class: str, count: int) -> list[WorkerHandle]:  # noqa: ARG002
        """Start nothing."""
        return []

    def live(self) -> list[WorkerHandle]:
        """List nothing."""
        return []

    def running(self, resource_class: str) -> int:  # noqa: ARG002
        """Count nothing."""
        return 0

    def stop(self, handles: Sequence[WorkerHandle]) -> None:
        """Stop nothing."""


@pytest.mark.contract
def test_results_read_from_store_only(tmp_path: Path) -> None:
    """A result whose outputs the store lacks is a gap and retried; a restarted coordinator reads it from disk alone.

    The first attempt's worker stores into another store; once the retry stored the outputs, a restarted coordinator
    with no worker reads the first attempt's outcome and finds them in the store.
    """
    profile = _profile(**CONTENT)
    placed, values, store, workers = _stack(tmp_path, profile, ["one"])
    elsewhere = ResultStore(tmp_path / "elsewhere")
    workers.act = lambda fragment: elsewhere if fragment.attempt == 1 else None
    pool = _pool(tmp_path, store, workers, profile)
    try:
        [ref] = _all(pool, placed, values)
    finally:
        workers.stop()
    assert ref.output_identity is not None and store.outputs_of(ref.key_digest, ref.output_identity) is not None
    assert [f.attempt for f in workers.served] == [1, 2] and [g.type for _, g in pool.gaps] == ["MissingOutput"]
    restarted = _pool(tmp_path, store, _Idle(), profile)
    [again] = _all(restarted, placed, values)
    assert restarted.epoch == 2 and again == ref.model_copy(update={"served": "exact"})
    assert not restarted.gaps and [f.attempt for f in workers.served] == [1, 2]


def _wav(path: Path) -> Path:
    soundfile.write(path, numpy.zeros(1600, dtype="float32"), 16000)
    return path


def _heard(source: object, profile: ExecutionProfile) -> Plan:
    call = TaskCall.model_validate({"task": HEAR.name, "inputs": {"audio": source}}, context={"registry": REGISTRY})
    return plan(call, config=RunConfig(), execution=profile)


@pytest.mark.contract
def test_two_starters_keep_every_durable_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A starter holding audio in memory, its write paused, and one giving the path merge one after the other.

    The memory starter takes the lease first and is held at its rename; the path starter waits for the lease, reads
    the memory marker and keeps its path, so the durable source survives whichever finishes last.
    """
    profile = _profile(**CONTENT)
    path = _wav(tmp_path / "a.wav")
    memory, durable = _heard(open_audio(path), profile), _heard(path, profile)
    run_dir = tmp_path / "run"
    [(reached, go)] = pause_replaces(monkeypatch, Path(INPUTS_FILE).name, 1)
    first = threading.Thread(target=record_sources, args=(memory, run_dir))
    first.start()
    assert reached.wait(30)
    second = threading.Thread(target=record_sources, args=(durable, run_dir))
    second.start()
    second.join(0.5)
    assert second.is_alive()
    go.set()
    first.join(30)
    second.join(30)
    assert json.loads((run_dir / INPUTS_FILE).read_text())["rows"][0]["audio"] == str(path)


@pytest.mark.contract
def test_a_merge_whose_lease_fails_every_proof_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A merge whose `inputs.json` lease is displaced at each proof is tried `RECORD_ATTEMPTS` times."""
    profile = _profile(**CONTENT)
    displaced = displace_before_proof(monkeypatch, f"{Path(INPUTS_FILE).name}.lease")
    with pytest.raises(LeaseUnavailable):
        record_sources(_heard(_wav(tmp_path / "a.wav"), profile), tmp_path / "run")
    assert len(displaced) == RECORD_ATTEMPTS and not (tmp_path / "run" / INPUTS_FILE).exists()


CRASH_STALE_S = 1.0
CRASH = f"""
import os, sys
from pathlib import Path
from senselab_core.runtime.execution.epoch import Epoch
from senselab_core.runtime.execution_settings import JobLease
lease = JobLease(stale_after_s={CRASH_STALE_S}, renew_fraction=0.2, attribute_delay_s=0)
epochs = Epoch(Path(sys.argv[1]))
with epochs.held(epochs.take(lease), lease):
    os._exit(9)
"""


@pytest.mark.contract
def test_wait_on_a_run_whose_coordinator_died_ends_interrupted(
    tmp_path: Path, lease_clock: Callable[..., ManualClock]
) -> None:
    """A coordinator killed while holding its epoch: `wait(None)` returns once its lease is stale, interrupted."""
    run_dir = tmp_path / "run"
    log = EventLog(run_dir, "a1", 1)
    job = node_fragment(*_admitted(tmp_path)).jobs[0]
    log.record("queued", job.node, job=job)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    crashed = subprocess.run([sys.executable, "-c", CRASH, str(run_dir)], env=env, check=False)
    assert crashed.returncode == 9 and not Epoch(run_dir).stopped()
    clock = lease_clock(storage_origin=newest(run_dir / "epoch" / "1" / "coordinator.lease"))
    poll = 0.1
    status = Run(run_dir, clock=clock).wait(None, poll=poll)
    assert status.interrupted and not status.done and status.counts["queued"] == 1
    assert CRASH_STALE_S < clock.now() <= CRASH_STALE_S + 2 * poll


def _admitted(tmp_path: Path) -> tuple[Plan, Any]:
    profile = _profile(**CONTENT)
    placed, values = _planned(profile, ["one"])
    pool = _pool(tmp_path / "admitting", ResultStore(tmp_path / "store"), _Idle(), profile)
    return placed, pool.admit(placed, values)


@pytest.mark.contract
def test_coordinate_runs_a_plan_on_the_pool_as_the_next_epoch(tmp_path: Path) -> None:
    """`coordinate` takes the next epoch, then its attempt, runs every job through the pool and ends its epoch.

    Status reads claims, not starts.

    The `fail` record is dead after its two content attempts and its descendants are none; the others are done.
    """
    pytest.importorskip("pydra")
    from senselab_core.runtime.execution.pool import coordinate  # noqa: PLC0415 - after the pydra check

    profile = _profile(**CONTENT)
    placed, values, store, workers = _stack(tmp_path, profile, ["one", "fail", "two"])
    run_dir = tmp_path / "run"
    assert prepare(placed, run_dir, store)[0] == 1
    try:
        outcome = coordinate(run_dir, substrate=workers, profile=profile, store=store)
    finally:
        workers.stop()
    failed = [r.job.element for r in outcome.results["a"] if r.failure is not None]
    assert failed == [(1,)] and len(outcome.results["a"]) == 3
    events = read_events(run_dir)
    assert "started" not in {e.kind for e in events} and {(e.epoch, e.ordinal) for e in events} == {(2, 2)}
    status = Run(run_dir).wait(None)
    assert status.counts["done"] == 2 and status.counts["failed"] == 1 and status.done and not status.interrupted
    assert Epoch(run_dir).current() == 2 and Epoch(run_dir).attempt(2) == (events[0].attempt, 2)
    assert Epoch(run_dir).stopped()
    assert cli_main(["coordinate", str(tmp_path / "absent")]) == 1


@pytest.mark.contract
def test_a_stale_coordinator_can_neither_hold_an_env_nor_stop_the_current_one(
    tmp_path: Path, tier_bound: float, lease_clock: Callable[..., ManualClock]
) -> None:
    """An older coordinator whose env breaks after a newer epoch was taken stops; no hold of its epoch counts.

    Its clock is frozen, so every poll may reuse its submit-time epoch read; the outcome that lands must find the newer
    epoch.
    """
    profile = _profile(content_attempts=2, infrastructure_replacements=1, resource_replacements=0)
    placed, values, store, workers = _stack(tmp_path, profile, ["one", "two"])
    stale = _pool(tmp_path, store, _Idle(), profile, clock=lease_clock())
    early = stale.submit(node_fragment(placed, stale.admit(placed, values), elements=[(0,)]))
    workers.act = lambda fragment: LOST if fragment.jobs[0].element == (0,) else None
    queue = stale.queues.queue("cpu")
    claim = queue.claim("w")
    assert claim is not None
    workers._serve(claim)  # noqa: SLF001 - one item served by hand, before the newer epoch
    live = _pool(tmp_path, store, workers, profile)
    late = live.submit(node_fragment(placed, live.admit(placed, values), elements=[(1,)]))
    await_true(lambda: len(workers.served) == 2, tier_bound, profile.pool.poll_s, "serving")
    with pytest.raises(StaleCoordinator):
        list(follow(stale, early))
    queue.hold(stale.epoch, "host")
    assert queue.held("host") is None and not stale.broken
    try:
        [ref] = list(follow(live, late))
    finally:
        workers.stop()
    assert ref.failure is None and ref.job.element == (1,)


@pytest.mark.contract
def test_a_stale_coordinator_writes_no_dead_record(tmp_path: Path, lease_clock: Callable[..., ManualClock]) -> None:
    """A charged fault past the budget lands on a coordinator a newer epoch replaced: it stops and `dead/` stays empty.

    Its clock is frozen, so its polls may reuse its submit-time epoch read.
    """
    profile = _profile(content_attempts=1, infrastructure_replacements=6, resource_replacements=0)
    placed, values, store, workers = _stack(tmp_path, profile, ["one"])
    stale = _pool(tmp_path, store, _Idle(), profile, clock=lease_clock())
    handle = stale.submit(node_fragment(placed, stale.admit(placed, values)))
    workers.act = lambda fragment: CHARGED
    claim = stale.queues.queue("cpu").claim("w")
    assert claim is not None
    workers._serve(claim)  # noqa: SLF001 - one item served by hand, before the newer epoch
    Epoch(tmp_path / "run").take()
    with pytest.raises(StaleCoordinator):
        stale.poll(handle)
    assert not (tmp_path / "run" / "dead").exists()


@pytest.mark.contract
def test_an_env_broken_on_one_queue_is_held_on_another(tmp_path: Path) -> None:
    """An env broken on `cpu` and failing again on `cpu:large` is held there too; nothing is failed or dead."""
    profile = _profile(content_attempts=2, infrastructure_replacements=1, resource_replacements=0)
    placed, values, store, workers = _stack(tmp_path, profile, ["one", "two"])
    pool = _pool(tmp_path, store, _Idle(), profile)
    admission = pool.admit(placed, values)
    small = node_fragment(placed, admission, elements=[(0,)])
    large = node_fragment(placed, admission, elements=[(1,)])
    grant = {"resource_class": "cpu:large", "partition": profile.resource_classes["cpu:large"].partition}
    large = large.model_copy(update={"placement": large.placement.model_copy(update=grant)})
    handles = [pool.submit(small), pool.submit(large)]
    workers.act = lambda fragment: LOST
    for name in ("cpu", "cpu:large"):
        claim = pool.queues.queue(name).claim("w")
        assert claim is not None
        workers._serve(claim)  # noqa: SLF001 - served by hand, so both outcomes exist before either is read
    for handle, name in zip(handles, ("cpu", "cpu:large"), strict=True):
        with pytest.raises(QueueHeld) as raised:
            list(follow(pool, handle))
        assert (raised.value.queue, raised.value.env) == (name, "host")
        assert pool.queues.queue(name).held("host") == pool.epoch
    assert [e.kind for e in read_events(tmp_path / "run") if e.kind == "failed"] == ["failed"]
    assert not any((tmp_path / "run").glob("dead/*"))


@pytest.mark.contract
def test_a_result_under_another_jobs_key_is_never_yielded(tmp_path: Path) -> None:
    """An outcome giving job 0 the key and outputs of job 1 is a content fault of job 0, dead under its own key."""
    profile = _profile(content_attempts=1, infrastructure_replacements=6, resource_replacements=0)
    placed, values, store, workers = _stack(tmp_path, profile, ["one", "two"])
    try:
        first, second = _all(_pool(tmp_path, store, workers, profile), placed, values)
    finally:
        workers.stop()
    pool = _pool(tmp_path / "other", store, _Idle(), profile)
    handle = pool.submit(node_fragment(placed, pool.admit(placed, values), elements=[(0,)]))
    claim = pool.queues.queue("cpu").claim("w")
    assert claim is not None
    forged = second.model_copy(update={"job": first.job})
    claim.commit(claim.items, {"jobs": [forged.model_dump(mode="json")]})
    [got] = list(follow(pool, handle))
    assert got.failure is not None and got.failure.type == "WrongKey" and got.failure.fault == "content"
    assert got.key_digest == first.key_digest != second.key_digest and got.output_identity is None
    assert (tmp_path / "other" / "run" / "dead" / f"{first.key_digest}.json").is_file()


@pytest.mark.contract
def test_a_broken_env_holds_only_its_own_items(tmp_path: Path) -> None:
    """With `host` broken on `cpu`, an item of another env on `cpu` is still claimed and completes."""
    profile = _profile(content_attempts=2, infrastructure_replacements=1, resource_replacements=0)
    placed, values, store, workers = _stack(tmp_path, profile, ["one", "two"])
    workers.act = lambda fragment: LOST if fragment.placement.env == "host" else None
    pool = _pool(tmp_path, store, workers, profile)
    admission = pool.admit(placed, values)
    other = node_fragment(placed, admission, elements=[(1,)])
    other = other.model_copy(update={"placement": other.placement.model_copy(update={"env": "venv:other"})})
    broken = pool.submit(node_fragment(placed, admission, elements=[(0,)]))
    try:
        with pytest.raises(QueueHeld):
            list(follow(pool, broken))
        [ref] = list(follow(pool, pool.submit(other)))
    finally:
        workers.stop()
    assert ref.failure is None and list(pool.broken) == ["host"]
    assert pool.queues.queue("cpu").held("venv:other") is None


@pytest.mark.contract
def test_infrastructure_failures_are_counted_per_epoch_from_disk(tmp_path: Path) -> None:
    """A coordinator restarted within its epoch counts the failures already logged; a new epoch counts afresh."""
    profile = _profile(content_attempts=2, infrastructure_replacements=2, resource_replacements=0)
    placed, values, store, workers = _stack(tmp_path, profile, ["one", "two"])
    workers.act = lambda fragment: LOST if fragment.jobs[0].element == (1,) or fragment.attempt == 1 else None
    first = _pool(tmp_path, store, workers, profile)
    try:
        [done] = _all_of(first, placed, values, (0,))
        restarted = _pool(tmp_path, store, workers, profile, epoch=first.epoch)
        with pytest.raises(QueueHeld) as held:
            _all_of(restarted, placed, values, (1,))
        newer = _pool(tmp_path, store, workers, profile)
        with pytest.raises(QueueHeld) as again:
            _all_of(newer, placed, values, (1,))
    finally:
        workers.stop()
    assert done.failure is None and held.value.failures == 2 and again.value.failures == 2
    failed = [e.epoch for e in read_events(tmp_path / "run") if e.kind == "failed"]
    assert failed == [1, 1, 2, 2]


def _all_of(pool: Pool, placed: Plan, values: Mapping[InputIdentity, object], element: tuple[int, ...]) -> list[Any]:
    fragment = node_fragment(placed, pool.admit(placed, values), elements=[element])
    return list(follow(pool, pool.submit(fragment)))


@pytest.mark.contract
def test_content_and_resource_charges_replay_after_restart(tmp_path: Path) -> None:
    """A restarted coordinator with no worker reads the same dead failures from the queues' outcomes on disk."""
    content = _profile(**CONTENT)
    placed, values, store, workers = _stack(tmp_path, content, ["fail"])
    try:
        [dead] = _all(_pool(tmp_path, store, workers, content), placed, values)
    finally:
        workers.stop()
    assert dead.failure is not None and dead.failure.fault == "content" and len(workers.served) == 2
    assert _all(_pool(tmp_path, store, _Idle(), content), placed, values) == [dead]

    moves = _profile(content_attempts=1, infrastructure_replacements=6, resource_replacements=1)
    placed, values, store, workers = _stack(tmp_path / "moves", moves, ["one"])
    workers.act = lambda fragment: OOM
    try:
        [oom] = _all(_pool(tmp_path / "moves", store, workers, moves), placed, values)
    finally:
        workers.stop()
    assert oom.failure == OOM and [f.placement.resource_class for f in workers.served] == ["cpu", "cpu:large"]
    assert _all(_pool(tmp_path / "moves", store, _Idle(), moves), placed, values) == [oom]
    assert len(workers.served) == 2


@pytest.mark.contract
def test_prepare_takes_the_epoch_before_the_attempt_and_records_it(tmp_path: Path) -> None:
    """Each `prepare` takes the next epoch, then claims an attempt ordinal and records it in that epoch."""
    profile = _profile(**CONTENT)
    placed, _ = _planned(profile, ["one"])
    run_dir, store = tmp_path / "run", ResultStore(tmp_path / "store")
    first, second = prepare(placed, run_dir, store), prepare(placed, run_dir, store)
    assert (first[0], first[2], second[0], second[2]) == (1, 1, 2, 2)
    assert Epoch(run_dir).attempt(2) == second[1:] and json.loads((run_dir / "epoch" / "2" / "lease.json").read_text())


@pytest.mark.contract
def test_wait_never_reports_a_running_run_interrupted(tmp_path: Path, lease_clock: Callable[..., ManualClock]) -> None:
    """A stopped latest epoch does not interrupt `wait` while this `Run`'s thread lives or a claim is live."""
    profile = _profile(**CONTENT)
    placed, values, store, workers = _stack(tmp_path, profile, ["one"])
    run_dir = tmp_path / "run"
    pool = _pool(tmp_path, store, _Idle(), profile)
    job = node_fragment(placed, pool.admit(placed, values)).jobs[0]
    pool.events.record("queued", job.node, job=job)
    pool.submit(node_fragment(placed, pool.admit(placed, values)))
    epochs = Epoch(run_dir)
    (run_dir / "epoch" / str(pool.epoch) / "ended").touch()
    assert epochs.stopped()
    clock, running = lease_clock(waits="spend"), threading.Event()
    run = Run(run_dir, clock=clock)
    run.launch(threading.Thread(target=running.wait, daemon=True))
    try:
        assert not run.wait(0.3, poll=0.05).interrupted
    finally:
        running.set()
    claim = pool.queues.queue("cpu").claim("w")
    assert claim is not None
    assert not Run(run_dir, clock=clock).wait(0.3, poll=0.05).interrupted
    claim.release()
    assert Run(run_dir, clock=clock).wait(None, poll=0.05).interrupted


@pytest.mark.contract
def test_an_epoch_taken_but_never_held_ends_a_wait(tmp_path: Path, lease_clock: Callable[..., ManualClock]) -> None:
    """A coordinator that died between taking its epoch and holding it: `wait(None)` ends one lease bound later."""
    run_dir = tmp_path / "run"
    log = EventLog(run_dir, "a1", 1, epoch=1)
    job = node_fragment(*_admitted(tmp_path)).jobs[0]
    lease = JobLease(stale_after_s=0.5, renew_fraction=0.2, attribute_delay_s=0)
    Epoch(run_dir).take(lease)
    log.record("queued", job.node, job=job)
    clock = lease_clock(storage_origin=os.stat(run_dir / "epoch" / "1").st_mtime)
    poll = 0.05
    assert Run(run_dir, clock=clock).wait(None, poll=poll).interrupted
    assert lease.stale_after_s < clock.now() <= lease.stale_after_s + 2 * poll


@pytest.mark.contract
def test_a_stale_pool_stops_the_runner_without_failing_or_blocking_a_job(tmp_path: Path) -> None:
    """The runner treats `StaleCoordinator` as an interruption: it raises it and logs no terminal event."""
    pytest.importorskip("pydra")
    from senselab_core.runtime.execution.runner import PydraRunner  # noqa: PLC0415 - after the pydra check

    profile = _profile(**CONTENT)
    placed, values, store, _ = _stack(tmp_path, profile, ["one", "two"])
    stale = _pool(tmp_path, store, _Idle(), profile)
    Epoch(tmp_path / "run").take()
    with pytest.raises(StaleCoordinator):
        PydraRunner(profile=profile, work_dir=tmp_path / "pydra").run_plan(
            placed, stale, values=values, events=stale.events
        )
    assert {e.kind for e in read_events(tmp_path / "run")} == {"queued"}


@pytest.mark.contract
def test_plan_run_with_a_pool_binds_it_to_the_runs_own_epoch(tmp_path: Path) -> None:
    """`Plan.run` with a pool built beforehand takes the next epoch, binds the pool to it, and completes."""
    pytest.importorskip("pydra")
    profile = _profile(**CONTENT)
    placed, values = _planned(profile, ["one", "two"])
    run_dir, store = run_dir_of(placed.hash), ResultStore(tmp_path / "store")
    workers = FakeWorkers(run_dir, store, placed, values)
    pool = Pool(
        run_dir,
        queues=QueueSet(run_dir / RUN_QUEUES, profile, epoch=Epoch(run_dir).current),
        store=store,
        events=EventLog(run_dir, "a0", 1, epoch=0),
        profile=profile,
        substrate=workers,
        epoch=0,
        resolver=unresolving(profile),
    )
    try:
        status = placed.run(pool, store=store).wait(60)
    finally:
        workers.stop()
    assert status.done and status.counts["done"] == 2 and pool.epoch == 1
    assert {e.epoch for e in read_events(run_dir)} == {1}


@pytest.mark.contract
def test_a_held_env_leaves_its_jobs_queued_and_the_wait_interrupted(tmp_path: Path) -> None:
    """`coordinate` stopped by a broken env raises `QueueHeld`; its jobs stay queued and `wait` ends interrupted."""
    pytest.importorskip("pydra")
    from senselab_core.runtime.execution.pool import coordinate  # noqa: PLC0415 - after the pydra check

    profile = _profile(content_attempts=2, infrastructure_replacements=1, resource_replacements=0)
    placed, _, store, workers = _stack(tmp_path, profile, ["one", "two"])
    workers.act = lambda fragment: LOST
    cpu = workers.queues.queue("cpu")
    workers.ready = lambda: not workers.served or cpu.held("host") is not None
    run_dir = tmp_path / "run"
    prepare(placed, run_dir, store)
    try:
        with pytest.raises(QueueHeld):
            coordinate(run_dir, substrate=workers, profile=profile, store=store)
    finally:
        workers.stop()
    assert not {"blocked", "cancelled"} & {e.kind for e in read_events(run_dir)}
    status = Run(run_dir).wait(None, poll=0.05)
    assert status.interrupted and status.counts["queued"] == 2 and status.counts["failed"] == 0


@pytest.mark.contract
def test_a_default_queue_set_honours_only_the_current_epochs_holds(tmp_path: Path) -> None:
    """A queue set built with no epoch reads its run's current epoch: a hold an older epoch made is not honoured."""
    profile = _profile(**CONTENT)
    placed, _ = _planned(profile, ["one"])
    run_dir = tmp_path / "run"
    epochs = Epoch(run_dir)
    epochs.take()
    epochs.take()
    queue = QueueSet(run_dir / RUN_QUEUES, profile).queue("cpu")
    queue.put(node_fragment(placed, Admission(id="a", plan_hash=placed.hash)))
    queue.hold(1, "host")
    assert queue.held("host") is None
    claim = queue.claim("w")
    assert claim is not None
    claim.release()
    queue.hold(2, "host")
    assert queue.held("host") == 2 and queue.claim("w") is None


@pytest.mark.contract
def test_the_expected_key_is_the_key_of_the_placement_sent(tmp_path: Path) -> None:
    """A fragment re-placed on another backend is checked against the key under that placement, and is yielded."""
    profile = _profile(**CONTENT)
    placed, values, store, workers = _stack(tmp_path, profile, ["one"])
    pool = _pool(tmp_path, store, workers, profile)
    fragment = node_fragment(placed, pool.admit(placed, values))
    moved = fragment.model_copy(update={"placement": fragment.placement.model_copy(update={"backend": "alt"})})
    try:
        [ref] = list(follow(pool, pool.submit(moved)))
    finally:
        workers.stop()
    planned = job_key(placed, "a", fragment.inputs[0]).digest()
    sent = job_key(placed, "a", fragment.inputs[0], moved.placement).digest()
    assert ref.failure is None and ref.key_digest == sent != planned and len(workers.served) == 1


@pytest.mark.contract
def test_a_resource_fault_never_moves_a_job_to_a_class_the_qos_cap_cannot_start(tmp_path: Path) -> None:
    """With a cap of 4 CPUs, `cpu:large` (8 CPUs) is not a move: the resource fault is dead, and nothing hangs."""
    classes = {
        "cpu": {"device": "cpu", "cpus": 4, "mem_bytes": 2**30, "max_workers": 1},
        "cpu:large": {"device": "cpu", "cpus": 8, "mem_bytes": 2**33, "max_workers": 1},
    }
    profile = ExecutionProfile.load(
        "laptop",
        {
            "devices": ["cpu"],
            "resource_classes": classes,
            "substrate": {**ExecutionProfile.load(CLUSTER).substrate.model_dump(mode="json"), "qos_cpu_cap": 4},
            "pool": {"poll_s": 0.2, "first_poll_s": 0.01},
            "faults": {**CONTENT, "resource_replacements": 1},
        },
    )
    assert larger_class(profile, "cpu") is None
    placed, values, store, workers = _stack(tmp_path, profile, ["one"])
    workers.act = lambda fragment: OOM
    try:
        [failed] = _all(_pool(tmp_path, store, workers, profile), placed, values)
    finally:
        workers.stop()
    assert failed.failure == OOM and [f.placement.resource_class for f in workers.served] == ["cpu"]
    assert (tmp_path / "run" / "dead" / f"{failed.key_digest}.json").is_file()


@pytest.mark.contract
def test_a_resumed_run_logs_prior_outcomes_as_exact_reuse(tmp_path: Path) -> None:
    """Two epochs over one queue: the second's events for the first's items are `cache_hit` under `exact`."""
    pytest.importorskip("pydra")
    from senselab_core.runtime.execution.pool import coordinate  # noqa: PLC0415 - after the pydra check

    profile = _profile(**CONTENT)
    placed, values, store, workers = _stack(tmp_path, profile, ["one", "two", "three"])
    run_dir = tmp_path / "run"
    assert prepare(placed, run_dir, store)[0] == 1
    try:
        coordinate(run_dir, substrate=workers, profile=profile, store=store)
        coordinate(run_dir, substrate=workers, profile=profile, store=store)
    finally:
        workers.stop()
    done = [e for e in read_events(run_dir) if e.kind in ("finished", "cache_hit")]
    first, second = ([e for e in done if e.ordinal == o] for o in sorted({e.ordinal for e in done}))
    assert {e.kind for e in first} == {"finished"} and len(first) == 3
    assert {e.kind for e in second} == {"cache_hit"} and {e.reuse for e in second} == {"exact"} and len(second) == 3


@pytest.mark.contract
def test_report_counts_reused_and_ran_after_resume(tmp_path: Path) -> None:
    """A resumed plan's results count the outcomes already on the queue as reused and the rest as ran."""
    profile = _profile(**CONTENT)
    placed, values, store, workers = _stack(tmp_path, profile, ["one", "two", "three", "four"])
    first = _pool(tmp_path, store, workers, profile)
    head, tail = split(node_fragment(placed, first.admit(placed, values)), 2)
    try:
        assert [r.served for r in follow(first, first.submit(head))] == [None, None]
        second = _pool(tmp_path, store, workers, profile)
        second.admit(placed, values)
        refs = [r for part in (head, tail) for r in follow(second, second.submit(part))]
    finally:
        workers.stop()
    assert [(r.job.element, r.served) for r in refs] == [((0,), "exact"), ((1,), "exact"), ((2,), None), ((3,), None)]


def _late_claim(tmp_path: Path, claimed_in: Callable[[], int]) -> tuple[list[ResultRef], FakeWorkers]:
    """Epoch 1 puts an item and a claim taken under `claimed_in` holds it; epoch 2 puts it; the claim then commits."""
    profile = _profile(**CONTENT)
    placed, values, store, workers = _stack(tmp_path, profile, ["one"])
    workers.ready = lambda: False
    first = _pool(tmp_path, store, workers, profile)
    fragment = node_fragment(placed, first.admit(placed, values))
    first.submit(fragment)
    held = QueueSet(tmp_path / "run" / RUN_QUEUES, profile, epoch=claimed_in).queue("cpu").claim("late")
    assert held is not None
    second = _pool(tmp_path, store, workers, profile)
    second.admit(placed, values)
    handle = second.submit(fragment)
    try:
        report(held, held.items[0], workers._outcomes(fragment))
        return list(follow(second, handle)), workers
    finally:
        workers.stop()


@pytest.mark.contract
def test_a_claim_of_an_earlier_epoch_committing_after_the_next_put_is_a_reuse(tmp_path: Path) -> None:
    """An epoch-1 worker finishing after epoch 2 put its item is served `exact` to epoch 2, not `finished`."""
    refs, _ = _late_claim(tmp_path, lambda: 1)
    assert [r.served for r in refs] == ["exact"]


@pytest.mark.contract
def test_an_outcome_of_a_claim_taken_in_no_epoch_is_a_reuse(tmp_path: Path) -> None:
    """An outcome record whose claim read no epoch counts as another epoch's."""
    refs, _ = _late_claim(tmp_path, lambda: 0)
    assert [r.served for r in refs] == ["exact"]


@pytest.mark.contract
def test_a_claim_of_the_current_epoch_is_not_a_reuse(tmp_path: Path) -> None:
    """An item put and claimed in the current epoch is reported as executed."""
    refs, _ = _late_claim(tmp_path, lambda: 2)
    assert [r.served for r in refs] == [None]


def _coordinator(  # noqa: PLR0913 - the run's parts
    placed: Plan,
    profile: ExecutionProfile,
    workers: FakeWorkers,
    root: Path,
) -> Pool:
    """Return a pool over a coordinator store of its own (the workers' is another), bound by `start`."""
    run_dir = run_dir_of(placed.hash)
    kwargs: dict[str, Any] = {
        "queues": QueueSet(run_dir / RUN_QUEUES, profile, epoch=Epoch(run_dir).current),
        "store": ResultStore(root),
        "events": EventLog(run_dir, "a0", 1, epoch=0),
        "profile": profile,
        "substrate": workers,
        "epoch": 0,
        "resolver": unresolving(profile),
    }
    return Pool(run_dir, **kwargs)


def _activities(run_dir: Path) -> tuple[list[Any], list[Any]]:
    every = ProvStore.read(run_dir).activities()
    return [a for a in every if "senselab:Reuse" not in a.types()], [a for a in every if "senselab:Reuse" in a.types()]


def _logical(records: Sequence[Any]) -> list[str]:
    return sorted(json.dumps(kprov.literal_value(r.node()["senselab:logicalKey"]), sort_keys=True) for r in records)


def _resumed(
    placed: Plan, profile: ExecutionProfile, workers: FakeWorkers, root: Path
) -> tuple[Pool, list[Any], list[Any]]:
    """Resume the run on a pool of its own, close the pool and return it with the run's executions and reuses."""
    pool = _coordinator(placed, profile, workers, root)
    try:
        assert resume(placed.hash, execution=profile, store=pool.store, executor=pool).wait(60).done
    finally:
        pool.close()
    return (pool, *_activities(run_dir_of(placed.hash)))


def _planned_nodes(
    profile: ExecutionProfile, texts: Sequence[str], nodes: Sequence[str]
) -> tuple[Plan, dict[InputIdentity, object]]:
    """Return the plan running `SCORE` over each text at each of `nodes`, every node reading the one input."""
    graph = Graph(
        inputs={"text": "list[json]"},
        nodes=tuple(Node(id=n, task=SCORE.name, task_version="1") for n in nodes),
        edges=tuple(Edge(source="input.text", target=f"{n}.text", kind="map") for n in nodes),
        outputs={f"v{n}": f"{n}.value" for n in nodes},
    )
    rows = [Record(id=f"r{i}", inputs={"text": ValueInput(value=t)}) for i, t in enumerate(texts)]
    placed = plan(graph, RecordSet(rows), config=RunConfig(), execution=profile)
    return placed, {bind_input("x", "json", t)[0]: t for t in texts}


def _stopped(
    tmp_path: Path, tier_bound: float, k: int, texts: Sequence[str], nodes: Sequence[str] = ("a",)
) -> tuple[Plan, ExecutionProfile, FakeWorkers]:
    """Run `texts` through `nodes` one item to a claim and cancel once `k` executions and reuses are recorded."""
    profile = ExecutionProfile.load(
        "laptop",
        {
            "devices": ["cpu"],
            "resource_classes": CLASSES,
            "pool": {"poll_s": 0.2, "first_poll_s": 0.01, "quantum_items": 1},
            "faults": CONTENT,
        },
    )
    placed, values = _planned(profile, texts) if nodes == ("a",) else _planned_nodes(profile, texts, nodes)
    run_dir = run_dir_of(placed.hash)
    workers = FakeWorkers(run_dir, ResultStore(tmp_path / "store"), placed, values)
    workers.ready = lambda: len(workers.served) < k
    pool = _coordinator(placed, profile, workers, tmp_path / "store")
    try:
        run = placed.run(pool, store=pool.store)
        await_true(
            lambda: sum(map(len, _activities(run_dir))) >= k, tier_bound, profile.pool.poll_s, "the first k items"
        )
        run.cancel()
        run.wait(60)
    finally:
        pool.close()
        if sys.exc_info()[0] is not None:
            workers.stop()
    workers.ready = lambda: True
    return placed, profile, workers


@pytest.mark.contract
def test_a_resume_records_a_reuse_for_each_item_it_settles_from_the_stopped_attempt(
    tmp_path: Path, tier_bound: float
) -> None:
    """Stopped after 2 of 4 items: the union holds 4 executions and 2 Reuse, with no context and no agent serving."""
    pytest.importorskip("pydra")
    placed, profile, workers = _stopped(tmp_path, tier_bound, 2, ["one", "two", "three", "four"])
    try:
        _, executed, reused = _resumed(placed, profile, workers, tmp_path / "store")
    finally:
        workers.stop()
    assert len(executed) == 4 and len(set(_logical(executed))) == 4 and len(reused) == 2
    assert set(_logical(reused)) <= set(_logical(executed))
    for reuse in reused:
        node = reuse.node()
        assert "senselab:context" not in node and "senselab:storedContext" in node
        assert "prov:wasAssociatedWith" not in node
        [association] = node["prov:qualifiedAssociation"]
        assert association["prov:hadRole"]["senselab:nodeId"] == "a" and "prov:agent" not in association
        assert association["prov:hadPlan"] == {"@id": f"plan:{placed.hash}"}
        assert kprov.literal_value(node["senselab:reusePolicy"]) == "exact"


@pytest.mark.contract
def test_a_second_resume_of_a_completed_run_records_a_reuse_per_item_and_no_execution(
    tmp_path: Path, tier_bound: float
) -> None:
    """Resuming a completed run of 4: 4 more Reuse, no more executions."""
    pytest.importorskip("pydra")
    placed, profile, workers = _stopped(tmp_path, tier_bound, 4, ["one", "two", "three", "four"])
    try:
        _, executed, reused = _resumed(placed, profile, workers, tmp_path / "store")
    finally:
        workers.stop()
    assert len(executed) == 4 and len(reused) == 4 and _logical(reused) == _logical(executed)


@pytest.mark.contract
def test_two_nodes_running_one_call_on_one_input_settle_one_reuse_each(tmp_path: Path, tier_bound: float) -> None:
    """Nodes `a` and `b` apply one task to the same texts: the resume settles one Reuse per node and item."""
    pytest.importorskip("pydra")
    placed, profile, workers = _stopped(tmp_path, tier_bound, 4, ["one", "two"], nodes=("a", "b"))
    before = len(_activities(run_dir_of(placed.hash))[1])
    try:
        _, _, reused = _resumed(placed, profile, workers, tmp_path / "store")
    finally:
        workers.stop()
    settled = [r for r in reused if "senselab:context" not in r.node()]
    nodes = sorted(r.node()["prov:qualifiedAssociation"][0]["prov:hadRole"]["senselab:nodeId"] for r in settled)
    assert nodes == ["a", "a", "b", "b"] and len(reused) == before + 4


@pytest.mark.contract
def test_a_reuse_settled_from_an_equivalent_outcome_records_that_policy(tmp_path: Path, tier_bound: float) -> None:
    """An outcome the earlier attempt served under `equivalent` is settled as an `equivalent` Reuse."""
    pytest.importorskip("pydra")
    placed, profile, workers = _stopped(tmp_path, tier_bound, 2, ["one", "two"])
    run_dir = run_dir_of(placed.hash)
    marked = 0
    for path in (p for p in (run_dir / RUN_QUEUES).rglob("outcome/*") if p.is_file()):
        document = json.loads(path.read_text())
        if document.get("outcome") == "done":
            for job in document["jobs"]:
                job["served"] = "equivalent"
            path.write_bytes(jcs_dumps(document))
            marked += 1
    assert marked == 2
    try:
        _, _, reused = _resumed(placed, profile, workers, tmp_path / "store")
    finally:
        workers.stop()
    assert len(reused) == 2
    for reuse in reused:
        assert kprov.literal_value(reuse.node()["senselab:reusePolicy"]) == "equivalent"
        assert "senselab:context" not in reuse.node() and "senselab:equivalence" in reuse.node()


@pytest.mark.contract
def test_a_failed_provenance_write_costs_no_settled_result_and_is_retried(tmp_path: Path, tier_bound: float) -> None:
    """A `prov/` that cannot be written when the earlier items settle leaves their results given; close writes it."""
    pytest.importorskip("pydra")
    placed, profile, workers = _stopped(tmp_path, tier_bound, 2, ["one", "two"])
    run_dir = run_dir_of(placed.hash)
    epoch = Epoch(run_dir).take()
    pool = Pool(
        run_dir,
        queues=QueueSet(run_dir / RUN_QUEUES, profile, epoch=Epoch(run_dir).current),
        store=ResultStore(tmp_path / "store"),
        events=EventLog(run_dir, f"a{epoch}", epoch, epoch=epoch),
        profile=profile,
        substrate=workers,
        epoch=epoch,
        resolver=unresolving(profile),
    )
    (run_dir / "prov").chmod(0o500)
    try:
        if os.access(run_dir / "prov", os.W_OK):
            pytest.skip("the directory stays writable (running as root)")
        refs = list(follow(pool, pool.submit(node_fragment(placed, pool.admit(placed, workers.values)))))
        assert [r.served for r in refs] == ["exact", "exact"] and len(_activities(run_dir)[1]) == 0
        (run_dir / "prov").chmod(0o700)
        pool.close()
    finally:
        (run_dir / "prov").chmod(0o700)
        workers.stop()
    assert len(_activities(run_dir)[1]) == 2
