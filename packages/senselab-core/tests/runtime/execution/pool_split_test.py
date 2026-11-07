"""`Pool.submit` queues a fragment as items of at most `pool.quantum_items` jobs, and `results` joins them."""

import threading
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from pool_stack_fixture import packaged, planned, registry, scoring
from senselab_core.kernel.declare import BackendSpec, Registry
from senselab_core.kernel.identity import InputIdentity
from senselab_core.runtime.call import RunConfig
from senselab_core.runtime.execution.epoch import Epoch
from senselab_core.runtime.execution.events import EventLog
from senselab_core.runtime.execution.executor import FaultRecord, PlanFragment, ResultRef
from senselab_core.runtime.execution.in_process import InProcess, job_key, node_fragment
from senselab_core.runtime.execution.job import Failed, JobOutcome
from senselab_core.runtime.execution.pool import Pool, report, split
from senselab_core.runtime.execution.queue import RUN_QUEUES, Claim, Queue, QueueSet
from senselab_core.runtime.execution.substrates import WorkerHandle, scale
from senselab_core.runtime.execution.workers import WorkerSet
from senselab_core.runtime.keys import request_size
from senselab_core.runtime.planner import Plan
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.store import ResultStore

SCORE = scoring("pool_split_test.score")
REGISTRY = registry(SCORE)
STRICT = scoring("pool_split_test.strict")
STRICT_REGISTRY = Registry()
STRICT_REGISTRY.add_task(STRICT)
STRICT_REGISTRY.add_backend(
    BackendSpec.model_validate(
        {
            **registry(STRICT).backends(STRICT.name)[0].model_dump(),
            "batch_invariant": False,
            "output_affecting": ("batch_formation",),
        }
    )
)
WORKERS = WorkerSet(2)
CONTENT = FaultRecord(fault="content", type="ValueError", message="bad input", gap=False, charged=True)


def _profile(quantum: int = 10, cap: int | None = None, max_workers: int = 32) -> ExecutionProfile:
    cpu = {"device": "cpu", "cpus": 4, "mem_bytes": 2**33, "time_s": 3600, "max_workers": max_workers}
    override: dict[str, Any] = {
        "devices": ["cpu"],
        "resource_classes": {"cpu": {**cpu, "partition": "cpuq"}},
        "substrate": {"kind": "slurm", "qos_cpu_cap": cap},
        "pool": {"quantum_items": quantum, "first_poll_s": 0.01, "poll_s": 0.2},
        "faults": {"content_attempts": 1},
    }
    return ExecutionProfile.load("orcd", override)


class Recording:
    """A `Substrate` that records its workers and runs nothing."""

    def __init__(self) -> None:
        """Start with no worker."""
        self.handles: list[WorkerHandle] = []

    def start(self, resource_class: str, count: int) -> list[WorkerHandle]:
        """Hold `count` pending workers of the queue."""
        made = [WorkerHandle(resource_class, f"w{len(self.handles) + i}", pending=True) for i in range(count)]
        self.handles.extend(made)
        return made

    def live(self) -> list[WorkerHandle]:
        """Return the workers held."""
        return list(self.handles)

    def running(self, resource_class: str) -> int:
        """Return the workers held for the queue."""
        return sum(1 for h in self.handles if h.queue == resource_class)

    def stop(self, handles: Sequence[WorkerHandle] = ()) -> None:
        """Forget `handles`."""
        gone = {h.id for h in handles}
        self.handles = [h for h in self.handles if h.id not in gone]


@pytest.fixture(autouse=True)
def stack(tmp_path: Path) -> Iterator[None]:
    """This module's registry, every env this interpreter under an MIT lock; the cache under `tmp_path`."""
    with packaged(REGISTRY, tmp_path / "cache"):
        yield
    WORKERS.close()


def _pool(tmp_path: Path, profile: ExecutionProfile, substrate: Recording | None = None) -> Pool:
    run_dir = tmp_path / "run"
    epochs = Epoch(run_dir)
    epoch = epochs.take()
    return Pool(
        run_dir,
        queues=QueueSet(run_dir / RUN_QUEUES, profile, epoch=epochs.current),
        store=ResultStore(tmp_path / "store"),
        events=EventLog(run_dir, f"a{epoch}", epoch, epoch=epoch),
        profile=profile,
        substrate=substrate if substrate is not None else Recording(),
        epoch=epoch,
    )


def _submitted(
    tmp_path: Path, profile: ExecutionProfile, jobs: int, substrate: Recording | None = None
) -> tuple[Plan, Mapping[InputIdentity, object], Pool, PlanFragment]:
    placed, values = planned(SCORE, profile, [f"t{i}" for i in range(jobs)])
    pool = _pool(tmp_path, profile, substrate)
    fragment = node_fragment(placed, pool.admit(placed, values))
    return placed, values, pool, fragment


def _claim_all(queue: Queue) -> list[Claim]:
    claims = []
    while (claim := queue.claim("w", max_items=1)) is not None:
        claims.append(claim)
    return claims


def _executed(
    placed: Plan, values: Mapping[InputIdentity, object], store: ResultStore, fragment: PlanFragment
) -> list[JobOutcome]:
    executor = InProcess(store=store, profile=placed.execution, run=RunConfig(), workers=WORKERS)
    admission = executor.admit(placed, values)
    return list(executor.outcomes(executor.submit(fragment.model_copy(update={"admission": admission.id}))))


def _failed(placed: Plan, fragment: PlanFragment, fault: FaultRecord) -> list[JobOutcome]:
    keys = [job_key(placed, fragment.node, inputs).digest() for inputs in fragment.inputs]
    return [Failed(job, key, fault) for job, key in zip(fragment.jobs, keys, strict=True)]


@pytest.mark.logic
def test_a_fragment_of_200_jobs_is_queued_as_20_items_of_10(tmp_path: Path) -> None:
    """Each item holds ten jobs in job order, with the parent's job keys, and the queue's work is every job."""
    placed, _, pool, fragment = _submitted(tmp_path, _profile(), 200)
    pool.submit(fragment)
    queue = pool.queues.queue("cpu")
    assert queue.depth() == 20 and queue.work() == 200
    items = [f for claim in _claim_all(queue) for f in claim.fragments]
    assert [len(f.jobs) for f in items] == [10] * 20
    assert [j for f in items for j in f.jobs] == list(fragment.jobs)
    keys = [job_key(placed, f.node, i, f.placement).digest() for f in items for i in f.inputs]
    assert keys == [job_key(placed, fragment.node, i, fragment.placement).digest() for i in fragment.inputs]
    assert all(f.placement == fragment.placement for f in items)


@pytest.mark.parametrize(("cap", "expected"), [(None, 20), (16, 4)], ids=["class-max", "qos-cap"])
@pytest.mark.logic
def test_scale_wants_a_worker_per_quantum_of_jobs(tmp_path: Path, cap: int | None, expected: int) -> None:
    """Twenty items of ten jobs want twenty workers, within the QOS cap over four CPUs per worker."""
    profile = _profile(cap=cap)
    recording = Recording()
    _, _, pool, fragment = _submitted(tmp_path, profile, 200, recording)
    pool.submit(fragment)
    recording.handles.clear()
    scale(pool.queues, recording, profile)
    assert len(recording.handles) == expected


@pytest.mark.logic
def test_two_workers_claim_different_items_and_every_job_is_committed_once(tmp_path: Path) -> None:
    """Two claimers bounded by a quantum of jobs each hold one item at once; the items' outcomes cover each job once."""
    placed, _, pool, fragment = _submitted(tmp_path, _profile(), 200)
    pool.submit(fragment)
    queues = [QueueSet(tmp_path / "run" / RUN_QUEUES, _profile()).queue("cpu") for _ in range(2)]
    committed: list[ResultRef] = []
    served: dict[str, list[str]] = {"w0": [], "w1": []}
    lock = threading.Lock()

    def serve(queue: Queue, name: str) -> None:
        while (claim := queue.claim(name, max_items=10, max_jobs=10)) is not None:
            with claim:
                [(digest, item)] = zip(claim.items, claim.fragments, strict=True)
                assert report(claim, digest, _failed(placed, item, CONTENT))
            record = queue.outcome(digest)
            assert record is not None and isinstance(record["jobs"], list)
            with lock:
                served[name].append(digest)
                committed.extend(ResultRef.model_validate(j) for j in record["jobs"])

    threads = [threading.Thread(target=serve, args=(q, f"w{i}")) for i, q in enumerate(queues)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert sorted(r.job.element for r in committed) == sorted(j.element for j in fragment.jobs)
    assert set(served["w0"]).isdisjoint(served["w1"]) and len(served["w0"]) + len(served["w1"]) == 20
    assert pool.queues.queue("cpu").work() == 0


@pytest.mark.logic
def test_two_claims_held_at_once_take_different_items(tmp_path: Path) -> None:
    """While one claim holds an item, a second claimer bounded by a quantum of jobs takes the next one."""
    _, _, pool, fragment = _submitted(tmp_path, _profile(), 200)
    pool.submit(fragment)
    first = QueueSet(tmp_path / "run" / RUN_QUEUES, _profile()).queue("cpu").claim("a", max_items=10, max_jobs=10)
    second = QueueSet(tmp_path / "run" / RUN_QUEUES, _profile()).queue("cpu").claim("b", max_items=10, max_jobs=10)
    assert first is not None and second is not None
    with first, second:
        assert len(first.items) == len(second.items) == 1 and set(first.items).isdisjoint(second.items)
        assert first.fragments[0].jobs == fragment.jobs[:10] and second.fragments[0].jobs == fragment.jobs[10:20]


@pytest.mark.logic
def test_a_fragment_put_again_makes_no_new_item(tmp_path: Path) -> None:
    """A second submit, and a second coordinator's, of one fragment leaves the same twenty items."""
    placed, values, pool, fragment = _submitted(tmp_path, _profile(), 200)
    pool.submit(fragment)
    pool.submit(fragment)
    later = _pool(tmp_path, _profile())
    later.submit(node_fragment(placed, later.admit(placed, values)))
    queue = pool.queues.queue("cpu")
    assert queue.depth() == 20 and len(list((queue.path / "items").iterdir())) == 20


@pytest.mark.contract
def test_results_join_the_items_in_job_order(tmp_path: Path) -> None:
    """Items committed in reverse order are read in one poll and yielded in the parent's job order."""
    placed, values, pool, fragment = _submitted(tmp_path, _profile(quantum=2), 6)
    handle = pool.submit(fragment)
    for claim in reversed(_claim_all(pool.queues.queue("cpu"))):
        with claim:
            report(claim, claim.items[0], _executed(placed, values, pool.store, claim.fragments[0]))
    refs = list(pool.results(handle))
    assert [r.job for r in refs] == list(fragment.jobs) and all(r.failure is None for r in refs)


@pytest.mark.contract
def test_a_content_fault_in_one_item_fails_only_its_jobs(tmp_path: Path) -> None:
    """The second item's jobs are dead after one charge; the other items' jobs are given and nothing is queued again."""
    placed, values, pool, fragment = _submitted(tmp_path, _profile(quantum=2), 6)
    handle = pool.submit(fragment)
    for index, claim in enumerate(_claim_all(pool.queues.queue("cpu"))):
        item = claim.fragments[0]
        outcomes = _failed(placed, item, CONTENT) if index == 1 else _executed(placed, values, pool.store, item)
        with claim:
            report(claim, claim.items[0], outcomes)
    refs = {r.job: r for r in pool.results(handle)}
    assert [j for j, r in refs.items() if r.failure is not None] == list(fragment.jobs[2:4])
    assert all(refs[j].failure == CONTENT for j in fragment.jobs[2:4])
    assert pool.queues.queue("cpu").depth() == 0


@pytest.mark.logic
def test_a_non_invariant_backends_requests_are_never_split(tmp_path: Path) -> None:
    """Its requests are one job each, so every item holds whole requests; each job lands in one item with its key."""
    profile = _profile(quantum=3)
    with packaged(STRICT_REGISTRY, tmp_path / "strict"):
        placed, values = planned(STRICT, profile, [f"t{i}" for i in range(7)])
        pool = _pool(tmp_path, profile)
        fragment = node_fragment(placed, pool.admit(placed, values))
        backend = STRICT_REGISTRY.backends(STRICT.name)[0]
        assert request_size(backend, fragment.placement.batch_size) == 1
        items = split(fragment, 3)
        keys = [job_key(placed, i.node, inputs, i.placement).digest() for i in items for inputs in i.inputs]
        whole = [job_key(placed, fragment.node, inputs, fragment.placement).digest() for inputs in fragment.inputs]
    assert [len(i.jobs) for i in items] == [3, 3, 1] and [j for i in items for j in i.jobs] == list(fragment.jobs)
    assert keys == whole and split(fragment, 7) == [fragment]
