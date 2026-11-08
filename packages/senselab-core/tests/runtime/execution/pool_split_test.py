"""`Pool.submit` queues a fragment as items of its sized quantum of jobs, and `results` joins them."""

import asyncio
import contextlib
import os
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, ClassVar

import pytest
from pool_stack_fixture import packaged, planned, registry, scoring
from senselab_core.kernel.dataset import Record, ValueInput
from senselab_core.kernel.declare import BackendSpec, Registry
from senselab_core.kernel.graph import Edge, Graph, Node
from senselab_core.kernel.identity import InputIdentity
from senselab_core.runtime.call import RunConfig, bind_input
from senselab_core.runtime.datasets import RecordSet
from senselab_core.runtime.execution.epoch import Epoch
from senselab_core.runtime.execution.events import EventLog
from senselab_core.runtime.execution.executor import FaultRecord, PlanFragment, QueueHeld, ResultRef, follow
from senselab_core.runtime.execution.in_process import InProcess, fragment_keys, job_key, node_fragment
from senselab_core.runtime.execution.job import Failed, JobOutcome
from senselab_core.runtime.execution.pool import DEAD_DIR, Pool, report, split
from senselab_core.runtime.execution.queue import RUN_QUEUES, Claim, Queue, QueueSet
from senselab_core.runtime.execution.substrates import WorkerHandle, scale
from senselab_core.runtime.execution.workers import WorkerSet
from senselab_core.runtime.keys import request_size
from senselab_core.runtime.planner import Plan, plan
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.sizing import Sample
from senselab_core.runtime.store import ResultStore
from senselab_core.testing.measures import CEILED, CEILED_SIZING, FixedMeasures

SCORE = scoring("pool_split_test.score")
CHAIN = SCORE.model_copy(update={"name": "pool_split_test.chain", "inputs": {"value": "sha256"}})
REGISTRY = registry(SCORE, CHAIN)
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
OOM = FaultRecord(fault="resource", type="ResourceFault", message="out of memory", gap=False, charged=False)
LOST = FaultRecord(fault="infrastructure", type="WorkerLost", message="gone", gap=False, charged=False)


def _profile(quantum: int = 10, cap: int | None = None, **faults: int) -> ExecutionProfile:
    cpu = {
        "device": "cpu",
        "cpus": 4,
        "mem_bytes": 2**33,
        "time_s": 3600,
        "max_workers": 32,
        "partition": "cpuq",
    }
    override: dict[str, Any] = {
        "devices": ["cpu"],
        "resource_classes": {"cpu": cpu, "cpu:large": {**cpu, "mem_bytes": 2**34, "max_workers": 2}},
        "substrate": {"kind": "slurm", "qos_cpu_cap": cap},
        "pool": {"quantum_items": quantum, "first_poll_s": 0.01, "poll_s": 0.2},
        "sizing": CEILED_SIZING,
        "faults": {"content_attempts": 1, **faults},
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


def _pool(
    tmp_path: Path, profile: ExecutionProfile, substrate: Recording | None = None, measures: FixedMeasures | None = None
) -> Pool:
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
        measures=measures if measures is not None else FixedMeasures(),
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
    batch = fragment.placement.batch_size
    assert all(f.placement.model_copy(update={"batch_size": batch}) == fragment.placement for f in items)
    assert {f.placement.batch_size for f in items} == {10}


def _sample(wall: float, jobs: int = 1, *, cold: bool = False) -> Sample:
    return Sample(
        wall_s=wall,
        claim_s=0.1,
        jobs=jobs,
        batch_size=jobs,
        input_s=None,
        baseline_bytes=10,
        resident_bytes=15,
        peak_memory_bytes=20,
        cold=cold,
        job="7",
        ended_s=0,
    )


@pytest.mark.logic
@pytest.mark.parametrize(("batched_s", "kept"), [(1.5, "batch"), (2.5, "single")], ids=["faster", "slower"])
def test_an_unmeasured_unit_calibrates_alone_then_batched_and_keeps_the_faster(
    tmp_path: Path, batched_s: float, kept: str
) -> None:
    """One-job items run first, then batched and one-job items side by side; the second round's times decide.

    The rest go batched only if batches ran no slower per item than the one-job items beside them on the warm worker.
    """
    profile, measures = _profile(quantum=100), FixedMeasures(None)
    placed, values = planned(SCORE, profile, [f"t{i}" for i in range(16)])
    pool = _pool(tmp_path, profile, measures=measures)
    admission = pool.admit(placed, values)
    first = pool.submit(node_fragment(placed, admission, elements=[(i,) for i in range(13)]))
    pool.submit(node_fragment(placed, admission, elements=[(i,) for i in range(13, 16)]))
    queue = pool.queues.queue("cpu")
    assert queue.work() == profile.sizing.calibration_items == 4 and pool.workers_wanted("cpu", 4) == 1
    for index, claim in enumerate(_claim_all(queue)):
        with claim:
            [item] = claim.fragments
            sample = _sample(4.5 if index == 0 else 4.0, cold=index == 0)
            report(claim, claim.items[0], _executed(placed, values, pool.store, item), sample)
    results, done = pool.poll(first)
    assert len(results) == 4 and not done and measures.puts == []
    pairs = _claim_all(queue)
    digests = {c.items[0] for c in pairs}
    assert [len(c.fragments[0].jobs) for c in pairs] == [2, 1] * (profile.sizing.calibration_items - 1)
    for spread, claim in zip((0.9, 0.9, 1.0, 1.0, 1.4, 1.4), pairs, strict=True):
        [item] = claim.fragments
        jobs = len(item.jobs)
        assert item.placement.batch_size == jobs
        with claim:
            sample = _sample((2.0 if jobs == 1 else jobs * batched_s) * spread, jobs)
            report(claim, claim.items[0], _executed(placed, values, pool.store, item), sample)
    results, done = pool.poll(first)
    [(unit, measure, measured)] = measures.puts
    assert len(results) == 9 and done and (unit.task, unit.resource_class) == (SCORE.name, "cpu")
    assert (measure.claim_s, measure.s_per_input_s) == (0.1, 2.0)
    assert measure.batched is not None and (measure.batched.batch_size, measure.batched.s_per_input_s) == (2, batched_s)
    assert (measure.resident_bytes, measure.peak_memory_bytes) == (15, 20)
    assert measured.sha is None and measured.jobs == ("7",) and "senselab-core" in (measured.run or "")
    assert set(pool.calibrated[unit][0]) >= digests and pool.calibrated[unit][1] == 13
    [rest] = [c.fragments[0] for c in _claim_all(queue)]
    assert len(rest.jobs) == 3 and (rest.placement.batch_size > 1) == (kept == "batch")


@pytest.mark.logic
def test_a_resource_fault_halves_the_batch_later_items_are_sent_in(tmp_path: Path) -> None:
    """A resource fault on an item in batches of 8 writes doubled per-item memory; later items go in batches of 4."""
    room = _profile(quantum=100).sizing.memory_headroom * 2**33
    per_item = int(room // 8)
    measure = CEILED.model_copy(update={"peak_memory_bytes": per_item})
    profile, measures = _profile(quantum=100, resource_replacements=0), FixedMeasures(measure)
    placed, values = planned(SCORE, profile, [f"t{i}" for i in range(32)])
    pool = _pool(tmp_path, profile, measures=measures)
    admission = pool.admit(placed, values)
    handle = pool.submit(node_fragment(placed, admission, elements=[(i,) for i in range(16)]))
    [claim] = _claim_all(pool.queues.queue("cpu"))
    assert claim.fragments[0].placement.batch_size == 8
    with claim:
        report(claim, claim.items[0], _failed(placed, claim.fragments[0], OOM))
    list(follow(pool, handle))
    [(_, grown, derivation)] = measures.puts
    assert grown.peak_memory_bytes == 2 * per_item and derivation.jobs == ("w",)
    pool.submit(node_fragment(placed, admission, elements=[(i,) for i in range(16, 32)]))
    assert {c.fragments[0].placement.batch_size for c in _claim_all(pool.queues.queue("cpu"))} == {4}


@pytest.mark.logic
def test_a_round_without_a_warm_item_measures_nothing(tmp_path: Path) -> None:
    """A one-job round gives only its cold item, so nothing is written; the unit's next fragment starts a round.

    A cold item's load and work cannot be told apart.
    """
    profile, measures = _profile(quantum=100), FixedMeasures(None)
    placed, values = planned(SCORE, profile, [f"t{i}" for i in range(6)])
    pool = _pool(tmp_path, profile, measures=measures)
    admission = pool.admit(placed, values)
    alone = pool.submit(node_fragment(placed, admission, elements=[(0,)]))
    queue = pool.queues.queue("cpu")
    [claim] = _claim_all(queue)
    with claim:
        report(
            claim, claim.items[0], _executed(placed, values, pool.store, claim.fragments[0]), _sample(9.0, cold=True)
        )
    results, done = pool.poll(alone)
    assert len(results) == 1 and done and measures.puts == [] and pool.calibrated == {}
    pool.submit(node_fragment(placed, admission, elements=[(i,) for i in range(1, 6)]))
    assert [len(c.fragments[0].jobs) for c in _claim_all(queue)] == [1] * profile.sizing.calibration_items


@pytest.mark.parametrize(("cap", "expected"), [(None, 20), (16, 4)], ids=["class-max", "qos-cap"])
@pytest.mark.logic
def test_scale_wants_a_worker_per_quantum_of_jobs(tmp_path: Path, cap: int | None, expected: int) -> None:
    """Twenty items of ten jobs want twenty workers, within the QOS cap over four CPUs per worker."""
    profile = _profile(cap=cap)
    recording = Recording()
    _, _, pool, fragment = _submitted(tmp_path, profile, 200, recording)
    pool.submit(fragment)
    pool.close()
    recording.handles.clear()
    scale(pool.queues, recording, profile, workers=pool.workers_wanted)
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
    refs = list(follow(pool, handle))
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
    refs = {r.job: r for r in follow(pool, handle)}
    assert [j for j, r in refs.items() if r.failure is not None] == list(fragment.jobs[2:4])
    assert all(refs[j].failure == CONTENT for j in fragment.jobs[2:4])
    assert pool.queues.queue("cpu").depth() == 0


@pytest.mark.logic
def test_a_bucketed_backends_jobs_keep_their_keys_across_items(tmp_path: Path) -> None:
    """A backend keying its batch formation is split like any other; each job lands in one item with its own key."""
    profile = _profile(quantum=3)
    with packaged(STRICT_REGISTRY, tmp_path / "strict"):
        placed, values = planned(STRICT, profile, [f"t{i}" for i in range(7)])
        pool = _pool(tmp_path, profile)
        fragment = node_fragment(placed, pool.admit(placed, values))
        backend = STRICT_REGISTRY.backends(STRICT.name)[0]
        assert request_size(backend, fragment.placement.batch_size) == fragment.placement.batch_size
        items = split(fragment, 3)
        keys = [k for i in items for k in fragment_keys(i, placed, values)]
        whole = fragment_keys(fragment, placed, values)
    assert [len(i.jobs) for i in items] == [3, 3, 1] and [j for i in items for j in i.jobs] == list(fragment.jobs)
    assert keys == whole and split(fragment, 7) == [fragment]


def _serve_by_item(pool: Pool, placed: Plan, values: Mapping[InputIdentity, object], faulted: FaultRecord) -> None:
    """Commit the first item of the `cpu` queue with `faulted` for each of its jobs and execute every other item."""
    for index, claim in enumerate(_claim_all(pool.queues.queue("cpu"))):
        item = claim.fragments[0]
        outcomes = _failed(placed, item, faulted) if index == 0 else _executed(placed, values, pool.store, item)
        with claim:
            report(claim, claim.items[0], outcomes)


@pytest.mark.contract
def test_a_resource_fault_moves_only_its_items_jobs(tmp_path: Path) -> None:
    """The first item's jobs move to `cpu:large` one by one, its unit's per-item memory doubled; the second's yield."""
    placed, values, pool, fragment = _submitted(tmp_path, _profile(quantum=2, resource_replacements=1), 4)
    handle = pool.submit(fragment)
    _serve_by_item(pool, placed, values, OOM)
    given = follow(pool, handle)
    first = [next(given), next(given)]
    assert [r.job for r in first] == list(fragment.jobs[2:]) and all(r.failure is None for r in first)
    moved = _claim_all(pool.queues.queue("cpu:large"))
    assert [c.fragments[0].jobs for c in moved] == [fragment.jobs[:1], fragment.jobs[1:2]]
    assert pool.queues.queue("cpu").work() == 0
    assert isinstance(pool.measures, FixedMeasures)
    assert [(u.resource_class, m.peak_memory_bytes, d.jobs) for u, m, d in pool.measures.puts] == [("cpu", 2, ("w",))]
    for claim in moved:
        with claim:
            report(claim, claim.items[0], _executed(placed, values, pool.store, claim.fragments[0]))
    rest = list(given)
    assert [r.job for r in rest] == list(fragment.jobs[:2]) and all(r.failure is None for r in rest)


@pytest.mark.contract
def test_a_broken_env_holds_only_the_jobs_of_its_failing_item(tmp_path: Path) -> None:
    """The first item breaks its env: the second item's jobs are yielded, the hold is raised, and nothing is dead."""
    placed, values, pool, fragment = _submitted(tmp_path, _profile(quantum=2, infrastructure_replacements=1), 4)
    handle = pool.submit(fragment)
    _serve_by_item(pool, placed, values, LOST)
    given: list[ResultRef] = []
    with pytest.raises(QueueHeld):
        given.extend(follow(pool, handle))
    assert [r.job for r in given] == list(fragment.jobs[2:]) and all(r.failure is None for r in given)
    assert list(pool.broken) == [fragment.placement.env] and not (tmp_path / "run" / DEAD_DIR).exists()


@pytest.mark.logic
def test_work_counts_from_the_slots_and_reads_an_item_only_without_a_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh queue's work opens no item; a slot without a count is read, an unreadable or missing item counts one."""
    _, _, pool, fragment = _submitted(tmp_path, _profile(), 200)
    pool.submit(fragment)
    path = pool.queues.queue("cpu").path
    segment = path / "segments" / "000000000000"
    unreadable, missing, readable = (path / "items" / f"{os.readlink(segment / f'slot-{n}')}.json" for n in range(3))
    unreadable.write_bytes(b"{")
    opened: list[str] = []
    real_open = os.open

    def counting(name: Any, flags: int, *args: Any, **kwargs: Any) -> int:  # noqa: ANN401
        opened.append(os.path.basename(os.fsdecode(name)))
        return real_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", counting)
    assert QueueSet(tmp_path / "run" / RUN_QUEUES, _profile()).queue("cpu").work() == 200
    assert not [n for n in opened if n.endswith(".json")]
    missing.unlink()
    for n in range(3):
        (segment / f"jobs-{n}-10").unlink()
    assert QueueSet(tmp_path / "run" / RUN_QUEUES, _profile()).queue("cpu").work() == 200 - 30 + 1 + 1 + 10
    assert readable.name in opened and unreadable.name in opened


@pytest.mark.logic
def test_items_end_on_request_boundaries_when_a_request_fits(tmp_path: Path) -> None:
    """Nine jobs at a quantum of 5 are 4 + 4 + 1 under requests of 2, and 5 + 4 under requests of 8."""
    _, _, _, fragment = _submitted(tmp_path, _profile(), 9)
    assert [len(i.jobs) for i in split(fragment, 5, request=2)] == [4, 4, 1]
    assert [len(i.jobs) for i in split(fragment, 5, request=8)] == [5, 4]


class CountingFragment(PlanFragment):
    """A `PlanFragment` that counts the times it is dumped whole."""

    dumps: ClassVar[list[int]] = [0]

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        """Count the dump, then dump."""
        self.dumps[0] += 1
        return super().model_dump(**kwargs)


@pytest.mark.logic
def test_a_fragment_of_ten_thousand_jobs_is_dumped_once_to_split(tmp_path: Path) -> None:
    """Splitting 10**4 jobs into 1000 items dumps the fragment once; each item holds its own ten jobs and inputs."""
    _, _, _, one = _submitted(tmp_path, _profile(), 1)
    document = one.model_dump(mode="json")
    job, inputs = document["jobs"][0], document["inputs"][0]
    document["jobs"] = [{**job, "element": [i]} for i in range(10**4)]
    document["inputs"] = [inputs] * 10**4
    fragment = CountingFragment.model_validate(document)
    CountingFragment.dumps[0] = 0
    items = split(fragment, 10)
    assert CountingFragment.dumps[0] == 1
    assert len(items) == 1000 and all(len(i.jobs) == len(i.inputs) == 10 for i in items)
    assert [j.element for i in items for j in i.jobs] == [(n,) for n in range(10**4)]


class Gated(Recording):
    """A `Recording` whose `serve` thread runs the `cpu` queue's items once its gate opens, or after `wait_s`.

    The gate opens once `want` workers were started (at once for 0). With `hold` set, items of that node are held and
    every other item is served at once; the gate then opens once `want` workers are alive and `hold`'s jobs on the
    queue are `held_jobs`, which `peak` records. Each `start` takes `delay` seconds, as a slow `sbatch` does.
    """

    def __init__(  # noqa: PLR0913 - the gate's settings
        self, want: int, wait_s: float, hold: str | None = None, held_jobs: int = 0, delay: float = 0.0
    ) -> None:
        """Serve once `want` workers were started (and `hold` has `held_jobs` queued), or after `wait_s` seconds."""
        super().__init__()
        self.started = 0
        self.want, self.wait_s, self.hold, self.held_jobs, self.delay = want, wait_s, hold, held_jobs, delay
        self.opened = threading.Event()
        if hold is None and want <= 0:
            self.opened.set()
        self.peak = (0, 0)

    def start(self, resource_class: str, count: int) -> list[WorkerHandle]:
        """Hold `count` pending workers, after `delay` seconds, and count them."""
        time.sleep(self.delay)
        made = super().start(resource_class, count)
        self.started += len(made)
        if self.hold is None and self.started >= self.want:
            self.opened.set()
        return made

    def serve(self, pool: Pool, placed: Plan, values: Mapping[InputIdentity, object], done: threading.Event) -> None:
        """Execute each item of the `cpu` queue until `done`: `hold`'s items only once the gate opens."""
        began = time.monotonic()
        queue = QueueSet(pool.run_dir / RUN_QUEUES, placed.execution).queue("cpu")
        while not done.is_set():
            if not self.opened.is_set() and time.monotonic() - began >= self.wait_s:
                self.opened.set()
            if self.hold is None and not self.opened.is_set():
                done.wait(0.01)
                continue
            claim = queue.claim("gated", max_items=1)
            if claim is not None and claim.fragments[0].node == self.hold and not self.opened.is_set():
                self._watch(queue)
                claim.release()
                claim = None
            if claim is None:
                done.wait(0.01)
                continue
            with claim:
                report(claim, claim.items[0], _executed(placed, values, pool.store, claim.fragments[0]))

    def _watch(self, queue: Queue) -> None:
        """Record the held node's queued jobs and the workers alive; open the gate once both are reached."""
        work, alive = queue.work(), len(self.handles)
        self.peak = max(self.peak, (work, alive))
        if work >= self.held_jobs and alive >= self.want:
            self.opened.set()


def _chained(profile: ExecutionProfile, rows: int) -> tuple[Plan, dict[InputIdentity, object]]:
    """Return the plan `a` (`SCORE`, mapped over `rows` texts) then `b` (`CHAIN`, reading `a.value`), and its values."""
    graph = Graph(
        inputs={"text": "list[json]"},
        nodes=(Node(id="a", task=SCORE.name, task_version="1"), Node(id="b", task=CHAIN.name, task_version="1")),
        edges=(
            Edge(source="input.text", target="a.text", kind="map"),
            Edge(source="a.value", target="b.value", kind="plain"),
        ),
        outputs={"value": "b.value"},
    )
    texts = [f"t{i}" for i in range(rows)]
    records = [Record(id=f"r{i}", inputs={"text": ValueInput(value=t)}) for i, t in enumerate(texts)]
    placed = plan(graph, RecordSet(records), config=RunConfig(), execution=profile)
    return placed, {bind_input("x", "json", t)[0]: t for t in texts}


@contextlib.contextmanager
def _default_threads(count: int) -> Iterator[None]:
    """Run the next pydra loop with a default executor of `count` threads, as a coordinator of few CPUs has."""
    loop = asyncio.new_event_loop()
    loop.set_default_executor(ThreadPoolExecutor(count))
    asyncio.set_event_loop(loop)
    try:
        yield
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def _run_gated(  # noqa: PLR0913 - the plan and its gate, then the run's seams
    tmp_path: Path,
    profile: ExecutionProfile,
    placed: Plan,
    values: Mapping[InputIdentity, object],
    gated: Gated,
    cancel: threading.Event | None = None,
) -> Any:  # noqa: ANN401 - a `RunOutcome`
    """Run `placed` by `PydraRunner` on a pool over `gated`, its serving thread alongside, with 2 default threads."""
    runner = pytest.importorskip("senselab_core.runtime.execution.runner", exc_type=ImportError)
    pool = _pool(tmp_path, profile, gated)
    done = threading.Event()
    server = threading.Thread(target=gated.serve, args=(pool, placed, values, done), daemon=True)
    server.start()
    try:
        with _default_threads(2):
            return runner.PydraRunner(profile=profile, work_dir=tmp_path / "pydra").run_plan(
                placed, pool, values=values, events=pool.events, cancel=cancel
            )
    finally:
        done.set()
        server.join(30)


@pytest.mark.contract
def test_the_runner_queues_a_map_nodes_ready_jobs_so_scale_wants_a_worker_per_quantum(tmp_path: Path) -> None:
    """A 200-row map node run by `PydraRunner` on a pool reaches 20 items of 10 jobs, so 20 workers are started."""
    profile = _profile()
    gated = Gated(want=20, wait_s=10.0)
    placed, values = planned(SCORE, profile, [f"t{i}" for i in range(200)])
    outcome = _run_gated(tmp_path, profile, placed, values, gated)
    assert len(outcome.results["a"]) == 200 and all(r.failure is None for r in outcome.results["a"])
    assert gated.started == 20


@pytest.mark.contract
def test_a_downstream_nodes_jobs_queue_as_their_upstream_lands_so_scale_wants_a_worker_per_quantum(
    tmp_path: Path,
) -> None:
    """With `a`'s items served at once and `b`'s held, every one of `b`'s 200 jobs is queued and 20 workers are alive.

    The pydra loop has two default threads, so no thread may be held per fragment while it waits.
    """
    profile = _profile()
    gated = Gated(want=20, wait_s=10.0, hold="b", held_jobs=200)
    placed, values = _chained(profile, 200)
    outcome = _run_gated(tmp_path, profile, placed, values, gated)
    assert len(outcome.results["b"]) == 200 and all(r.failure is None for r in outcome.results["b"])
    assert gated.peak == (200, 20)


@pytest.mark.contract
def test_a_cancel_stops_following_the_pool_at_once_and_leaves_its_items_queued(tmp_path: Path) -> None:
    """A cancel set while a 200-job node waits on closed gates ends the run cancelled before the gate opens."""
    runner = pytest.importorskip("senselab_core.runtime.execution.runner", exc_type=ImportError)
    profile = _profile()
    gated = Gated(want=10**6, wait_s=30.0)
    placed, values = planned(SCORE, profile, [f"t{i}" for i in range(200)])
    cancel = threading.Event()
    queue = QueueSet(tmp_path / "run" / RUN_QUEUES, profile).queue("cpu")

    def cancel_once_queued() -> None:
        while True:
            with contextlib.suppress(OSError):
                if queue.depth() >= 20:
                    break
            time.sleep(0.01)
        cancel.set()

    threading.Thread(target=cancel_once_queued, daemon=True).start()
    began = time.monotonic()
    with pytest.raises(runner.RunCancelled):
        _run_gated(tmp_path, profile, placed, values, gated, cancel)
    assert time.monotonic() - began < 10 and not gated.opened.is_set()
    assert queue.depth() == 20 and queue.work() == 200


@pytest.mark.contract
def test_a_slow_substrate_never_delays_settling(tmp_path: Path) -> None:
    """With each start taking 8 s, a 60-job node whose items are served at once ends well before the start does."""
    profile = _profile()
    gated = Gated(want=0, wait_s=0.0, delay=8.0)
    placed, values = planned(SCORE, profile, [f"t{i}" for i in range(60)])
    began = time.monotonic()
    outcome = _run_gated(tmp_path, profile, placed, values, gated)
    assert time.monotonic() - began < 6 and gated.started == 0
    assert len(outcome.results["a"]) == 60 and all(r.failure is None for r in outcome.results["a"])


@pytest.mark.contract
def test_a_slow_substrate_never_delays_a_cancel(tmp_path: Path) -> None:
    """With each start taking 20 s, a cancel set once the items are queued ends the run within its bound."""
    runner = pytest.importorskip("senselab_core.runtime.execution.runner", exc_type=ImportError)
    profile = _profile()
    gated = Gated(want=10**6, wait_s=30.0, delay=20.0)
    placed, values = planned(SCORE, profile, [f"t{i}" for i in range(200)])
    cancel = threading.Event()
    queue = QueueSet(tmp_path / "run" / RUN_QUEUES, profile).queue("cpu")

    def cancel_once_queued() -> None:
        while True:
            with contextlib.suppress(OSError):
                if queue.depth() >= 20:
                    break
            time.sleep(0.01)
        cancel.set()

    threading.Thread(target=cancel_once_queued, daemon=True).start()
    began = time.monotonic()
    with pytest.raises(runner.RunCancelled):
        _run_gated(tmp_path, profile, placed, values, gated, cancel)
    assert time.monotonic() - began < 5 and queue.work() == 200


@pytest.mark.logic
def test_concurrent_submits_never_start_more_workers_than_the_target(tmp_path: Path) -> None:
    """48 senders submit a 10-job fragment each at once; the class's 32 workers are started, never more."""
    profile = _profile()
    gated = Gated(want=10**6, wait_s=0.0, delay=0.05)
    placed, values = planned(SCORE, profile, [f"t{i}" for i in range(480)])
    pool = _pool(tmp_path, profile, gated)
    admission = pool.admit(placed, values)
    fragments = [node_fragment(placed, admission, elements=[(i,) for i in range(s, s + 10)]) for s in range(0, 480, 10)]
    barrier = threading.Barrier(len(fragments))

    def send(fragment: PlanFragment) -> None:
        barrier.wait()
        pool.submit(fragment)

    senders = [threading.Thread(target=send, args=(f,)) for f in fragments]
    for sender in senders:
        sender.start()
    for sender in senders:
        sender.join(30)
    deadline = time.monotonic() + 10
    while gated.started < 32 and time.monotonic() < deadline:
        time.sleep(0.05)
    time.sleep(3 * profile.pool.poll_s)
    pool.close()
    assert pool.queues.queue("cpu").work() == 480 and gated.started == 32
