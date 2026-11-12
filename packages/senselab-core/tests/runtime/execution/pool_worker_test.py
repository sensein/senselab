"""Pool workers: quanta across classes, bisection of whole-request failures, renewal, SIGTERM and OOM kills."""

import dataclasses
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from epoch_fixture import prepared
from faulty_workers_fixture import FOREVER, Unloadable, crashes, put_batched, sender
from lease_clock_fixture import ManualClock, age
from pool_stack_fixture import lease_of, packaged, planned, registry, scoring
from senselab_core.kernel.declare import TaskSpec
from senselab_core.runtime import request_reports
from senselab_core.runtime.calibration import FaultKind
from senselab_core.runtime.call import RunConfig, use_packaged
from senselab_core.runtime.call import packaged as current_stack
from senselab_core.runtime.execution.events import read_events, status_from
from senselab_core.runtime.execution.executor import Admission, FaultRecord, JobRef, PlanFragment, ResultRef
from senselab_core.runtime.execution.in_process import job_key, node_fragment
from senselab_core.runtime.execution.job import Emit, Failed, Job, JobOutcome, execute
from senselab_core.runtime.execution.observing import Sent
from senselab_core.runtime.execution.pool import run_plan, split
from senselab_core.runtime.execution.pool_worker import Abandoned, Member, PoolWorker, Send, run_members, serve_pool
from senselab_core.runtime.execution.queue import RUN_QUEUES, Claim, QueueSet, item_digest, settle
from senselab_core.runtime.execution.workers import WorkerSet
from senselab_core.runtime.fake_backend import ECHO, FAKE, EchoParams
from senselab_core.runtime.keys import Call, Placement
from senselab_core.runtime.licenses import LicensePolicy, license_record
from senselab_core.runtime.locks import LeaseTimeout
from senselab_core.runtime.planner import Plan
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.sizing import Sample
from senselab_core.runtime.storage import FileOps, using_file_ops
from senselab_core.runtime.store import ResultStore
from senselab_core.runtime.worker_client import WorkerClient, WorkerLost, WorkerOOMKilled
from senselab_core.testing.interpreter import source_tree_python
from senselab_worker.faults import FaultClass, classify, error_record
from syscall_faults_fixture import await_true

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
    tmp_path: Path,
    profile: ExecutionProfile,
    texts: Sequence[str],
    task: TaskSpec = TASK,
    config: RunConfig | None = None,
) -> tuple[Plan, ResultStore, Path]:
    """Prepare a run of `task`, one job per text, in `tmp_path/run`; return its plan, its store and its queues."""
    placed, _ = planned(task, profile, texts, config)
    store = ResultStore(tmp_path / "store")
    prepared(placed, tmp_path / "run", store)
    return placed, store, tmp_path / "run" / RUN_QUEUES


def _put(placed: Plan, pool_dir: Path, element: int | range, queue: str = "cpu", why: str = "") -> str:
    """Put the item of `element` (or of each in a range) on `queue`, its `why` set when given; return its digest."""
    admission = Admission(id=f"run-{placed.hash}", plan_hash=placed.hash)
    elements = [(e,) for e in element] if isinstance(element, range) else [(element,)]
    fragment = node_fragment(placed, admission, elements=elements)
    update: dict[str, object] = {"resource_class": queue, **({"why": why} if why else {})}
    fragment = fragment.model_copy(update={"placement": fragment.placement.model_copy(update=update)})
    QueueSet(pool_dir, placed.execution, lease=lease_of(placed)).queue(queue).put(fragment)
    return item_digest(fragment)


def _sender(store: ResultStore, profile: ExecutionProfile) -> Send:
    def send(jobs: Sequence[Job], placement: Placement) -> list[JobOutcome]:
        return execute(jobs, placement, store=store, run=RunConfig(), profile=profile, workers=WORKERS)

    return send


def _outcomes(pool_dir: Path, profile: ExecutionProfile, queue: str = "cpu") -> list[dict[str, Any]]:
    directory = QueueSet(pool_dir, profile, lease=profile.lease).queue(queue).path / "outcome"
    return [json.loads(p.read_bytes()) for p in sorted(directory.iterdir()) if not p.name.startswith(".")]


def _text(job: Job) -> object:
    return next(iter(job.values.values()))


def _served(worker: PoolWorker, clock: ManualClock) -> int:
    """Serve `worker` on a thread, advancing `clock` by the pool's poll each time it waits, until it returns."""
    exits: list[int] = []
    thread = threading.Thread(target=lambda: exits.append(worker.serve()), daemon=True)
    thread.start()
    try:
        clock.drive(thread, worker.profile.pool.poll_s)
    finally:
        worker.halt()
        thread.join(clock.hang_s)
    assert not thread.is_alive()
    return exits[0]


@pytest.mark.logic
@pytest.mark.parametrize("polls", [2, 5])
def test_a_worker_exits_after_idle_polls_empty_polls(
    tmp_path: Path, lease_clock: Callable[..., ManualClock], polls: int
) -> None:
    """With nothing to claim a worker waits `pool.idle_polls` polls of `pool.poll_s` on its clock, then exits."""
    profile = _profile(idle_polls=polls)
    _, store, pool_dir = _run(tmp_path, profile, ["a"])
    clock = lease_clock()
    worker = PoolWorker(pool_dir, ["cpu"], profile=profile, store=store, send=_sender(store, profile), clock=clock)
    assert _served(worker, clock) == 0
    assert clock.now() == pytest.approx(polls * profile.pool.poll_s)


@pytest.mark.logic
def test_worker_yields_after_quantum_to_older_queue(tmp_path: Path) -> None:
    """After a quantum of 2 items on `cpu`, the worker claims the older item waiting on `cpu:b`, then goes back."""
    profile = _profile()
    placed, store, pool_dir = _run(tmp_path, profile, ["b", "a1", "a2", "a3", "a4"])
    _put(placed, pool_dir, 0, "cpu:b")
    for element in range(1, 5):
        _put(placed, pool_dir, element)
    other = QueueSet(pool_dir, profile, lease=profile.lease).queue("cpu:b").claim("other")
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
def test_a_worker_gathers_one_job_items_once_warm(tmp_path: Path, lease_clock: Callable[..., ManualClock]) -> None:
    """A cold queue's first item is claimed alone; then one-job items gather into one claim, never a 4-job item."""
    profile = _profile(quantum_items=4)
    placed, store, pool_dir = _run(tmp_path, profile, [f"t{i}" for i in range(11)])
    admission = Admission(id=f"run-{placed.hash}", plan_hash=placed.hash)
    queue = QueueSet(pool_dir, profile, lease=profile.lease).queue("cpu")
    for element in range(3):
        _put(placed, pool_dir, element)
    queue.put_all(split(node_fragment(placed, admission, elements=[(i,) for i in range(3, 11)]), 4))
    send, sizes, clock = _sender(store, profile), [], lease_clock()

    def recorded(jobs: Sequence[Job], placement: Placement) -> list[JobOutcome]:
        sizes.append(len(jobs))
        clock.advance(len(jobs))
        return send(jobs, placement)

    worker = PoolWorker(pool_dir, ["cpu"], profile=profile, store=store, send=recorded, clock=clock)
    assert _served(worker, clock) == 0
    assert sizes == [1, 2, 4, 4] and len(worker.served) == 5 and queue.work() == 0
    samples = [Sample.model_validate(o["measure"]) for o in _outcomes(pool_dir, profile)]
    assert sorted((s.jobs, s.cold) for s in samples) == [(1, False), (1, False), (1, True), (4, False), (4, False)]
    assert all(s.input_s is None and s.wall_s == s.jobs and s.claim_s == 0 for s in samples)


@pytest.mark.logic
def test_samples_leave_the_resident_model_out_of_the_per_item_memory(tmp_path: Path) -> None:
    """A stand-in loading a 1000-byte model and holding 10 bytes a job gives a resident model and 10 bytes an item.

    The model loads on the first request and stays; each job holds 10 bytes while it runs.
    """
    profile = _profile()
    placed, store, pool_dir = _run(tmp_path, profile, [f"t{i}" for i in range(3)])
    for element in range(3):
        _put(placed, pool_dir, element)
    memory = {"current": 100, "peak": 100}
    send = _sender(store, profile)

    def standing_in(jobs: Sequence[Job], placement: Placement) -> list[JobOutcome]:
        if memory["current"] == 100:
            memory["current"] += 1000
        memory["peak"] = max(memory["peak"], memory["current"] + 10)
        return send(jobs, placement)

    worker = PoolWorker(
        pool_dir,
        ["cpu"],
        profile=profile,
        store=store,
        send=standing_in,
        memory=lambda: (memory["current"], memory["peak"]),
        device_memory=lambda: None,
    )
    assert worker.serve() == 0
    samples = [Sample.model_validate(o["measure"]) for o in _outcomes(pool_dir, profile)]
    warm = [s for s in samples if not s.cold]
    assert (min(s.resident_bytes or 0 for s in warm), max(s.peak_memory_bytes for s in samples)) == (1100, 1110)


@pytest.mark.logic
def test_each_gathered_item_is_timed_by_the_requests_it_ran_in(
    tmp_path: Path, lease_clock: Callable[..., ManualClock]
) -> None:
    """Three one-job items gathered into one claim but sent one at a time each read at least their own request."""
    profile = _profile()
    placed, store, pool_dir = _run(tmp_path, profile, [f"t{i}" for i in range(4)])
    _put(placed, pool_dir, 0)
    for element in range(1, 4):
        _put(placed, pool_dir, element, why=f"own request {element}")
    send, took, clock = _sender(store, profile), 0.0625, lease_clock()

    def slow(jobs: Sequence[Job], placement: Placement) -> list[JobOutcome]:
        clock.advance(took * len(jobs))
        return send(jobs, placement)

    worker = PoolWorker(
        pool_dir, ["cpu"], profile=profile, store=store, send=slow, device_memory=lambda: None, clock=clock
    )
    _served(worker, clock)
    warm = [s for s in (Sample.model_validate(o["measure"]) for o in _outcomes(pool_dir, profile)) if not s.cold]
    assert len(warm) == 3 and all(s.wall_s == took for s in warm)


@pytest.mark.logic
def test_device_memory_is_read_once_per_claim_on_a_cuda_class(tmp_path: Path) -> None:
    """Device memory is read once before the first claim and once per claim, but sizes no model backend.

    The stand-in's processes hold nothing before the model loads, 1000 bytes after the cold claim and 1010 after the
    gathered one. The reads reach the samples; a model backend that reported no request peaks is left unmeasured
    on the device (batch 1), since those reads cannot tell its model from its items.
    """
    classes = {**OVERRIDE["resource_classes"], "gpu": {"device": "cuda", "max_workers": 1}}
    pool = {**OVERRIDE["pool"], "quantum_items": 4}
    profile = ExecutionProfile.load("laptop", {**OVERRIDE, "resource_classes": classes, "pool": pool})
    placed, store, pool_dir = _run(tmp_path, profile, [f"t{i}" for i in range(4)])
    for element in range(4):
        _put(placed, pool_dir, element, "gpu")
    reads = iter([0, 1000, 1010])
    worker = PoolWorker(
        pool_dir, ["gpu"], profile=profile, store=store, send=_sender(store, profile), device_memory=lambda: next(reads)
    )
    assert worker.serve() == 0
    samples = [Sample.model_validate(o["measure"]) for o in _outcomes(pool_dir, profile, "gpu")]
    assert next(reads, None) is None and len(samples) == 4 and not any(s.served for s in samples)
    assert {s.device_baseline_bytes for s in samples} == {0} and {s.device_bytes for s in samples} == {1000, 1010}
    assert not any(s.device_item_bytes is not None or s.device_allocated_bytes is not None for s in samples)


@pytest.mark.logic
def test_a_backends_request_peaks_reach_the_samples(tmp_path: Path) -> None:
    """A backend reporting each request's device peak and what stays allocated gives non-zero per-item device bytes."""
    profile = _profile()
    placed, store, pool_dir = _run(tmp_path, profile, [f"t{i}" for i in range(3)])
    for element in range(3):
        _put(placed, pool_dir, element)
    send = _sender(store, profile)

    def reporting(jobs: Sequence[Job], placement: Placement) -> list[JobOutcome]:
        request_reports.record(len(jobs), device=(1000 + 30 * len(jobs), 1000))
        return send(jobs, placement)

    worker = PoolWorker(
        pool_dir,
        ["cpu"],
        profile=profile,
        store=store,
        send=reporting,
        memory=lambda: (100, 200),
        device_memory=lambda: None,
    )
    assert worker.serve() == 0
    samples = [Sample.model_validate(o["measure"]) for o in _outcomes(pool_dir, profile)]
    assert {(s.device_item_bytes, s.device_allocated_bytes, s.batch_size) for s in samples} == {(30, 1000, 1)}


@pytest.mark.logic
def test_an_unreadable_resident_memory_leaves_the_host_memory_unmeasured(tmp_path: Path) -> None:
    """With only a high-water readable (no cgroup v2), no sample has a resident figure."""
    profile = _profile()
    placed, store, pool_dir = _run(tmp_path, profile, [f"t{i}" for i in range(3)])
    for element in range(3):
        _put(placed, pool_dir, element)
    worker = PoolWorker(
        pool_dir,
        ["cpu"],
        profile=profile,
        store=store,
        send=_sender(store, profile),
        memory=lambda: (None, 5000),
        device_memory=lambda: None,
    )
    assert worker.serve() == 0
    samples = [Sample.model_validate(o["measure"]) for o in _outcomes(pool_dir, profile)]
    assert samples and all(s.resident_bytes is None for s in samples)


@pytest.mark.logic
def test_failing_batch_bisected(tmp_path: Path, lease_clock: Callable[..., ManualClock]) -> None:
    """One poisoned job of an 8-job item fails alone, charged to the item; the other 7 commit their results."""
    profile = _profile(quantum_items=8)
    texts = [f"t{i}" for i in range(8)]
    texts[5] = "poison"
    placed, store, pool_dir = _run(tmp_path, profile, texts)
    put_batched(placed, pool_dir, profile, 8)
    workers = crashes({"poison": FOREVER})
    send = sender(store, profile, workers)
    clock = lease_clock()
    worker = PoolWorker(pool_dir, ["cpu"], profile=profile, store=store, send=send, clock=clock)
    assert _served(worker, clock) == 0
    refs = [ResultRef.model_validate(j) for o in _outcomes(pool_dir, profile) for j in o["jobs"]]
    failed = [r for r in refs if r.failure is not None]
    assert len(refs) == 8 and len(failed) == 1 and workers.sent == [8, 4, 4, 2, 1, 1, 1, 2]
    assert failed[0].failure is not None and failed[0].failure.type == "ItemCrashedWorker"
    poison = node_fragment(placed, Admission(id="x", plan_hash=placed.hash), elements=[(5,)])
    assert failed[0].key_digest == job_key(placed, "a", poison.inputs[0], _placement(poison)).digest()
    assert all(store.outputs_of(r.key_digest, r.output_identity) is not None for r in refs if r.output_identity)


def _placement(fragment: PlanFragment) -> Any:  # noqa: ANN401 - the fragment's placement on queue `cpu`
    return fragment.placement.model_copy(update={"resource_class": "cpu"})


@pytest.mark.logic
def test_a_worker_runs_its_items_under_the_license_policy_the_run_was_planned_under(tmp_path: Path) -> None:
    """The run allowed MIT only; the env's lock now says GPL: the worker's job fails `LicenseRefused`, not runs."""
    profile = _profile()
    placed, store, pool_dir = _run(
        tmp_path, profile, ["a"], config=RunConfig(licenses=LicensePolicy(frozenset({"mit"})))
    )
    _put(placed, pool_dir, 0)
    config = run_plan(tmp_path / "run", profile).config
    assert config is not None and config.licenses == LicensePolicy(frozenset({"mit"}))
    gpl = license_record("GPL-3.0-only", source="lock", verified={"date": "2026-11-11", "how": "pep639"}, gated=False)
    previous = use_packaged(dataclasses.replace(current_stack(), lock_records=lambda _: {"p": gpl}))
    try:
        worker = PoolWorker(pool_dir, ["cpu"], profile=profile, store=store, workers=WORKERS)
        assert worker.serve() == 0
    finally:
        use_packaged(previous)
    [outcome] = _outcomes(pool_dir, profile)
    [ref] = [ResultRef.model_validate(j) for j in outcome["jobs"]]
    assert ref.failure is not None and ref.failure.type == "LicenseRefused"


@pytest.mark.logic
def test_a_requests_items_are_settled_before_the_next_request_runs(tmp_path: Path, tier_bound: float) -> None:
    """Four one-job items sent as two requests of two: request 1's items have outcomes when request 2 begins."""
    profile = _profile(quantum_items=4)
    placed, store, pool_dir = _run(tmp_path, profile, [f"t{i}" for i in range(4)])
    for element in range(4):
        _put(placed, pool_dir, element)
    claim = QueueSet(pool_dir, profile, lease=profile.lease).queue("cpu").claim("w", max_jobs=4, singles=True)
    assert claim is not None and len(claim.items) == 4
    settled: list[int] = []

    def two_requests(jobs: Sequence[Job], placement: Placement, emit: Emit) -> list[JobOutcome]:
        run = RunConfig()
        first = execute(jobs[:2], placement, store=store, run=run, profile=profile, workers=WORKERS)
        emit([0, 1], first)
        await_true(lambda: len(_outcomes(pool_dir, profile)) == 2, tier_bound, 0.01, "request 1's items settled")
        settled.append(len(_outcomes(pool_dir, profile)))
        return first + execute(jobs[2:], placement, store=store, run=run, profile=profile, workers=WORKERS)

    PoolWorker(pool_dir, ["cpu"], profile=profile, store=store, send=two_requests).run_claim(claim)
    assert settled == [2]
    found = _outcomes(pool_dir, profile)
    assert len(found) == 4
    assert all(Sample.model_validate(o["measure"]).wall_s > 0 and o.get("observations") for o in found)


class _Proofs(FileOps):
    """Counts the proof directories leases make."""

    def __init__(self) -> None:
        self.count = 0

    def mkdir(self, name: str, mode: int, *, dir_fd: int) -> None:
        """Count `proof-*` and make the directory."""
        self.count += name.startswith("proof-")
        super().mkdir(name, mode, dir_fd=dir_fd)


@pytest.mark.logic
def test_requests_that_emit_inside_one_renewal_interval_prove_each_lease_once_to_settle(
    tmp_path: Path, lease_clock: Callable[..., ManualClock]
) -> None:
    """Four items settled by four emits inside an interval: one proof each, for their settle, and no renewal's."""
    profile = _profile(quantum_items=4)
    placed, store, pool_dir = _run(tmp_path, profile, [f"t{i}" for i in range(4)])
    for element in range(4):
        _put(placed, pool_dir, element)
    clock = lease_clock()
    claim = (
        QueueSet(pool_dir, profile, lease=profile.lease, clock=clock).queue("cpu").claim("w", max_jobs=4, singles=True)
    )
    assert claim is not None and len(claim.items) == 4

    def each_alone(jobs: Sequence[Job], placement: Placement, emit: Emit) -> list[JobOutcome]:
        outcomes: list[JobOutcome] = []
        for i, job in enumerate(jobs):
            found = execute([job], placement, store=store, run=RunConfig(), profile=profile, workers=WORKERS)
            emit([i], found)
            outcomes += found
        return outcomes

    worker = PoolWorker(pool_dir, ["cpu"], profile=profile, store=store, send=each_alone, clock=clock)
    proofs = _Proofs()
    with using_file_ops(proofs):
        worker.run_claim(claim)
    assert len(_outcomes(pool_dir, profile)) == 4 and proofs.count == 4


@pytest.mark.logic
def test_a_claim_that_broke_a_stale_lease_logs_reclaimed_before_its_claimed(tmp_path: Path) -> None:
    """A dead holder's item logs `reclaimed` (age, token) then `claimed` per job; the other item only `claimed`."""
    profile = _profile()
    placed, store, pool_dir = _run(tmp_path, profile, ["a", "b"])
    digests = [_put(placed, pool_dir, element) for element in (0, 1)]
    queue = QueueSet(pool_dir, profile, lease=profile.lease).queue("cpu")
    dead = queue.path / "claims" / f"{digests[0]}.lease" / uuid.uuid4().hex
    dead.mkdir(parents=True)
    for path in (dead, dead.parent):
        os.utime(path, (10**9, 10**9))
    worker = PoolWorker(pool_dir, ["cpu"], profile=profile, store=store, send=_sender(store, profile))
    assert worker.serve() == 0
    events = [e for e in read_events(tmp_path / "run") if e.kind in ("claimed", "reclaimed")]
    assert [(e.kind, e.job.element) for e in events if e.job] == [
        ("reclaimed", (0,)),
        ("claimed", (0,)),
        ("claimed", (1,)),
    ]
    assert events[0].lease is not None and events[0].lease.tokens == (dead.name,)
    assert events[0].lease.age_s > profile.lease.stale_after_s and status_from(events).reclaimed == 1


@pytest.mark.logic
def test_an_item_with_an_outcome_and_an_unmarked_slot_is_resolved_and_never_claimed(tmp_path: Path) -> None:
    """A holder died after linking its outcome, before marking the slot: the next claim marks it, takes nothing."""
    profile = _profile()
    placed, store, pool_dir = _run(tmp_path, profile, ["a"])
    digest = _put(placed, pool_dir, 0)
    queue = QueueSet(pool_dir, profile, lease=profile.lease).queue("cpu")
    held = queue.claim("dead")
    assert held is not None
    with queue.open() as q:
        record = {"outcome": "done", "worker": "dead", "epoch": 1, "finish": "f", "jobs": []}
        assert settle(held._held[digest].lease, q, digest, record) is True  # noqa: SLF001 - the row under test
    held.release()
    assert queue.depth() == 1 and queue.claim("next") is None and queue.depth() == 0


@pytest.mark.logic
def test_claim_renewed_during_long_item(tmp_path: Path, lease_clock: Callable[..., ManualClock]) -> None:
    """An item running past twice the lease's stale bound stays claimed by its worker, which commits it."""
    profile = _profile()
    placed, store, pool_dir = _run(tmp_path, profile, ["slow"])
    _put(placed, pool_dir, 0)
    send, stolen, clock = _sender(store, profile), [], lease_clock()
    claims = QueueSet(pool_dir, profile, lease=profile.lease, clock=clock).queue("cpu")
    beat_s = profile.lease.renew_fraction * profile.lease.stale_after_s

    def slow(jobs: Sequence[Job], placement: Placement) -> list[JobOutcome]:
        clock.settle()
        for _ in range(round(2.5 * profile.lease.stale_after_s / beat_s)):
            age(claims.path / "claims", beat_s)
            clock.advance(beat_s)
            clock.settle()
            stolen.append(claims.claim("thief"))
        return send(jobs, placement)

    worker = PoolWorker(pool_dir, ["cpu"], profile=profile, store=store, send=slow, name="owner", clock=clock)
    assert _served(worker, clock) == 0
    [outcome] = _outcomes(pool_dir, profile)
    assert not any(stolen) and len(stolen) > 5 and (outcome["outcome"], outcome["worker"]) == ("done", "owner")


CHILD = """
import json, sys, threading
from pathlib import Path
from pool_stack_fixture import packaged, registry, scoring
from senselab_core.runtime.call import RunConfig, use_packaged
from senselab_core.runtime.call import packaged as current_stack
from senselab_core.runtime.execution.job import execute
from senselab_core.runtime.execution.pool_worker import serve_pool
from senselab_core.runtime.execution.workers import WorkerSet
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.store import ResultStore

pool_dir, cache, override = Path(sys.argv[1]), Path(sys.argv[2]), json.loads(sys.argv[3])
profile, store, workers = ExecutionProfile.load("laptop", override), ResultStore(Path(sys.argv[4])), WorkerSet(1)

def send(jobs, placement):
    if any(next(iter(j.values.values())) == "slow" for j in jobs):
        print("slow-started", flush=True)
        threading.Event().wait()
    return execute(jobs, placement, store=store, run=RunConfig(), profile=profile, workers=workers)

with packaged(registry(scoring("pool_worker_test.score")), cache):
    sys.exit(serve_pool(pool_dir, ["cpu"], profile=profile, store=store, send=send, join_s=0.2))
"""


@pytest.mark.logic
def test_sigterm_releases_claims(tmp_path: Path, tier_bound: float) -> None:
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
    child = subprocess.Popen([sys.executable, "-c", CHILD, *args], env=env, stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout is not None and child.stdout.readline() == "slow-started\n"
        child.send_signal(signal.SIGTERM)
        assert child.wait(timeout=tier_bound) == 0
    finally:
        child.kill()
        child.wait()
    [outcome] = _outcomes(pool_dir, profile)
    assert outcome["outcome"] == "done"
    queue = QueueSet(pool_dir, profile, lease=profile.lease).queue("cpu")
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
def test_a_load_failure_during_bisection_keeps_the_first_halfs_commits(
    tmp_path: Path, lease_clock: Callable[..., ManualClock]
) -> None:
    """A request lost as a whole is halved; the first half commits, and the second half, whose load fails, fails."""
    profile = _profile(quantum_items=8)
    placed, store, pool_dir = _run(tmp_path, profile, [f"t{i}" for i in range(8)])
    put_batched(placed, pool_dir, profile, 8)
    lost = crashes({"t0": 1})
    unloadable = Unloadable(lost, at=3)
    send = sender(store, profile, unloadable)
    clock = lease_clock()
    assert _served(PoolWorker(pool_dir, ["cpu"], profile=profile, store=store, send=send, clock=clock), clock) == 0
    refs = _refs(pool_dir, profile)
    done = [r for r in refs if r.failure is None]
    assert lost.sent == [8, 4] and unloadable.checkouts == 3 and len(done) == 4 and len(refs) == 8
    assert all(store.outputs_of(r.key_digest, r.output_identity) is not None for r in done if r.output_identity)
    assert {r.failure.type for r in refs if r.failure} == {"WorkerLoadFailed"}


def _members(count: int) -> list[Member]:
    return [
        Member(
            "d", i, Job(JobRef(plan_hash="0" * 64, node="n", element=(i,)), Call(ECHO, EchoParams(), {}), {}), f"k{i}"
        )
        for i in range(count)
    ]


def _failed(member: Member, fault: FaultKind) -> Failed:
    failure = FaultRecord(fault=fault, type="Stand", message="", gap=False, charged=fault == "content")
    return Failed(member.job.ref, member.key, failure)


@pytest.mark.logic
def test_an_abandoned_call_keeps_the_outcomes_it_finished() -> None:
    """`execute` raises `Abandoned` with the jobs it finished; the claim commits them, as the call was recorded."""
    members = _members(8)

    def send(jobs: Sequence[Job], placement: Placement, cancelled: threading.Event) -> list[JobOutcome]:  # noqa: ARG001
        raise Abandoned({i: _failed(members[i], "content") for i in range(4)})

    with pytest.raises(Abandoned) as stopped:
        run_members(
            members,
            Placement(backend=FAKE, backend_params=FAKE.params(), device="cpu"),
            send,
            threading.Event(),
        )
    assert sorted(stopped.value.kept) == [0, 1, 2, 3]
    assert all(isinstance(o, Failed) and o.failure.fault == "content" for o in stopped.value.kept.values())


def _reported(recorded: Sequence[Sent], member: Member) -> list[Any]:
    """Return the request reports the recorded calls hold for `member`'s job."""
    return [r for sent in recorded for r in sent.reports if member.job.ref in r.tags]


@pytest.mark.logic
def test_a_request_the_engine_split_is_recorded_once_beside_a_request_that_emits() -> None:
    """Request A (items 0, 1) was split by `execute`, B (2, 3) emits: A's report is recorded once, as `bisected`."""
    members, handed, recorded = _members(4), [], list[Sent]()

    def send(jobs: Sequence[Job], placement: Placement, emit: Emit) -> list[JobOutcome]:  # noqa: ARG001 - a Send
        with request_reports.request(tuple(m.job.ref for m in members[:2])) as ran:
            request_reports.record(2, device=(900, 500))
        request_reports.mark(ran, "bisected", FaultClass.RESOURCE, 2)
        for m in members[:2]:
            with request_reports.request((m.job.ref,)):
                request_reports.record(1)
        with request_reports.request(tuple(m.job.ref for m in members[2:])):
            request_reports.record(2)
        outcomes: list[JobOutcome] = [_failed(m, "content") for m in members]
        emit([2, 3], outcomes[2:])
        return outcomes

    run_members(
        members,
        Placement(backend=FAKE, backend_params=FAKE.params(), device="cpu"),
        send,
        record=recorded.append,
        handover=lambda found, outs: handed.append(([m.index for m in found], len(outs))),
    )
    assert [len(_reported(recorded, m)) for m in members] == [2, 2, 1, 1]
    assert len([r for sent in recorded for r in sent.reports if r.device == (900, 500)]) == 1
    assert [sent.failure for sent in recorded if (900, 500) in [r.device for r in sent.reports]] == ["bisected"]
    assert handed[0] == ([2, 3], 2) and {i for found, _ in handed for i in found} == {0, 1, 2, 3}


@pytest.mark.logic
def test_a_call_that_raises_after_an_emit_records_only_what_the_emit_did_not() -> None:
    """Items 2 and 3 were emitted and recorded; the call's failure is recorded for items 0 and 1 alone."""
    members, recorded = _members(4), list[Sent]()

    def send(jobs: Sequence[Job], placement: Placement, emit: Emit) -> list[JobOutcome]:  # noqa: ARG001 - a Send
        if len(jobs) == 4:
            with request_reports.request(tuple(m.job.ref for m in members[2:])):
                request_reports.record(2)
            emit([2, 3], [_failed(m, "content") for m in members[2:]])
            raise WorkerLost("died on the next request", returncode=-11)
        return [_failed(next(m for m in members if m.job.ref == j.ref), "content") for j in jobs]

    run_members(
        members,
        Placement(backend=FAKE, backend_params=FAKE.params(), device="cpu"),
        send,
        record=recorded.append,
        handover=lambda found, outs: None,
    )
    first, failed = recorded[0], recorded[1]
    assert first.failure is None and len(first.reports) == 1
    assert failed.failure == "failed" and failed.refs == tuple(m.job.ref for m in members[:2])


class _Abandoning(WorkerSet):
    """A worker set that records `abandon` instead of closing worker processes."""

    def __init__(self) -> None:
        super().__init__(1)
        self.abandoned = threading.Event()

    def abandon(self, thread: int) -> None:
        self.abandoned.set()


@pytest.mark.logic
def test_a_lost_claim_abandons_its_request(tmp_path: Path, tier_bound: float) -> None:
    """A claim broken mid-request: its worker set abandons the request, nothing is committed, nothing more is sent."""
    profile = _profile()
    placed, store, pool_dir = _run(tmp_path, profile, ["one", "two"])
    _put(placed, pool_dir, 0)
    _put(placed, pool_dir, 1, why="second")
    queue = QueueSet(pool_dir, profile, lease=profile.lease).queue("cpu")
    workers, sends, thief, seen = _Abandoning(), [], [], []

    def stolen(jobs: Sequence[Job], placement: Placement) -> list[JobOutcome]:
        sends.append(len(jobs))
        for lease in (queue.path / "claims").iterdir():
            shutil.rmtree(lease)
        thief.append(queue.claim("thief", max_items=2))
        seen.append(workers.abandoned.wait(tier_bound))
        raise WorkerLost("closed by abandon")

    worker = PoolWorker(pool_dir, ["cpu"], profile=profile, store=store, workers=workers, send=stolen)
    assert worker.serve() == 0
    assert seen == [True] and sends == [1] and _outcomes(pool_dir, profile) == [] and thief[0] is not None
    assert len(thief[0].items) == 2
    thief[0].release()


@pytest.mark.logic
def test_halt_commits_finished_groups_and_releases_the_rest(tmp_path: Path, tier_bound: float) -> None:
    """Halted while a claim's second group runs, the worker commits the first group's item and releases the second."""
    profile = _profile()
    placed, store, pool_dir = _run(tmp_path, profile, ["fast", "slow"])
    _put(placed, pool_dir, 0)
    _put(placed, pool_dir, 1, why="second")
    send, release = _sender(store, profile), threading.Event()
    worker: PoolWorker

    def halting(jobs: Sequence[Job], placement: Placement) -> list[JobOutcome]:
        if _text(jobs[0]) == "slow":
            release.wait(tier_bound)
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
    queue = QueueSet(pool_dir, profile, lease=profile.lease).queue("cpu")
    again = queue.claim("next", max_items=2)
    assert outcome["outcome"] == "done" and again is not None and len(again.items) == 1
    again.release()


@pytest.mark.logic
def test_a_lost_claim_releases_its_other_items_before_its_request_ends(
    tmp_path: Path, lease_clock: Callable[..., ManualClock]
) -> None:
    """A request that cannot be abandoned outlives the lease it lost; the claim's other item is claimable at once."""
    profile = _profile()
    placed, store, pool_dir = _run(tmp_path, profile, ["one", "two"])
    lost = _put(placed, pool_dir, 0)
    kept = _put(placed, pool_dir, 1, why="second")
    clock = lease_clock()
    queue = QueueSet(pool_dir, profile, lease=profile.lease, clock=clock).queue("cpu")
    taken: list[Claim] = []
    seen: list[str] = []
    beat_s = profile.lease.renew_fraction * profile.lease.stale_after_s

    def uninterruptible(jobs: Sequence[Job], placement: Placement) -> list[JobOutcome]:
        shutil.rmtree(queue.path / "claims" / f"{lost}.lease")
        clock.settle(0)
        for _ in range(round(2.5 * profile.lease.stale_after_s / beat_s)):
            if kept in seen:
                break
            clock.advance(beat_s)
            clock.settle(0)
            claim = queue.claim("other", max_items=2)
            taken.extend([claim] if claim is not None else [])
            seen.extend(claim.items if claim is not None else ())
        raise WorkerLost("the request ended after its claim was lost")

    worker = PoolWorker(
        pool_dir, ["cpu"], profile=profile, store=store, workers=WORKERS, send=uninterruptible, clock=clock
    )
    try:
        assert _served(worker, clock) == 0
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


IDLE_CHILD = """
import json, signal, sys, threading
from pathlib import Path
from pool_stack_fixture import packaged, registry, scoring
from signal_in_lock_fixture import deliver_in
from senselab_core.runtime.execution.pool_worker import serve_pool
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.store import ResultStore

pool_dir, cache, override = Path(sys.argv[1]), Path(sys.argv[2]), json.loads(sys.argv[3])
profile, store = ExecutionProfile.load("laptop", override), ResultStore(Path(sys.argv[4]))
state = deliver_in(signal.SIGTERM, lambda f: f.f_code.co_name == "wait" and f.f_locals.get("timeout") == 300.0)
with packaged(registry(scoring("pool_worker_test.score")), cache):
    code = serve_pool(pool_dir, ["cpu"], profile=profile, store=store)
sys.settrace(None)
print("fired" if state["fired"] else "not-fired", flush=True)
sys.exit(code)
"""


@pytest.mark.logic
def test_sigterm_landing_inside_the_stop_events_lock_stops_the_worker(tmp_path: Path, tier_bound: float) -> None:
    """A SIGTERM delivered while the claim loop is inside `Event.wait` on `stop`, holding its lock, still stops it."""
    override = {**OVERRIDE, "pool": {**OVERRIDE["pool"], "poll_s": 300.0}}
    profile = ExecutionProfile.load("laptop", override)
    _, _, pool_dir = _run(tmp_path, profile, ["x"])
    args = [str(pool_dir), str(tmp_path / "cache"), json.dumps(override), str(tmp_path / "store")]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    child = subprocess.Popen([sys.executable, "-c", IDLE_CHILD, *args], env=env, stdout=subprocess.PIPE, text=True)
    try:
        assert child.wait(timeout=tier_bound) == 0
        assert child.stdout is not None and child.stdout.read() == "fired\n"
    finally:
        child.kill()
        child.wait()
