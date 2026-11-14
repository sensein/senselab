"""`Pool.submit` queues a fragment as items of its sized quantum of jobs, and `results` joins them."""

import asyncio
import contextlib
import os
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, ClassVar

import pytest
from epoch_fixture import taken
from lease_clock_fixture import ManualClock
from pool_stack_fixture import lease_of, packaged, planned, registry, scoring
from senselab_core.kernel.dataset import Record, ValueInput
from senselab_core.kernel.declare import BackendSpec, Registry
from senselab_core.kernel.graph import Edge, Graph, Node
from senselab_core.kernel.identity import InputIdentity
from senselab_core.runtime.call import RunConfig, bind_input
from senselab_core.runtime.clock import Clock, SystemClock
from senselab_core.runtime.datasets import RecordSet
from senselab_core.runtime.execution.epoch import Epoch
from senselab_core.runtime.execution.events import EventLog
from senselab_core.runtime.execution.executor import FaultRecord, Handle, PlanFragment, QueueHeld, ResultRef, follow
from senselab_core.runtime.execution.in_process import InProcess, fragment_keys, job_key, node_fragment
from senselab_core.runtime.execution.job import Failed, JobOutcome
from senselab_core.runtime.execution.pool import DEAD_DIR, Pool, report, split
from senselab_core.runtime.execution.queue import RUN_QUEUES, Claim, Queue, QueueSet
from senselab_core.runtime.execution.substrates import WorkerHandle, scale
from senselab_core.runtime.execution.workers import WorkerSet
from senselab_core.runtime.keys import request_size
from senselab_core.runtime.planner import Plan, plan
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.store import ResultStore
from senselab_core.testing.measures import CEILED_SIZING, FixedEstimates, FixedUnit

CLUSTER = (
    next(p for p in Path(__file__).resolve().parents if p.name == "packages")
    / "senselab-core/tests/data/profiles/cluster.yaml"
)

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


def _profile(quantum: int = 10, cap: int | None = None, first_poll_s: float = 0.01, **faults: int) -> ExecutionProfile:
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
        "pool": {"quantum_items": quantum, "first_poll_s": first_poll_s, "poll_s": 0.2},
        "sizing": CEILED_SIZING,
        "faults": {"content_attempts": 1, **faults},
    }
    return ExecutionProfile.load(CLUSTER, override)


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
    tmp_path: Path,
    profile: ExecutionProfile,
    substrate: Recording | None = None,
    clock: Clock = SystemClock(),  # noqa: B008 - stateless
    kind: type[Pool] = Pool,
) -> Pool:
    run_dir = tmp_path / "run"
    epochs = Epoch(run_dir)
    epoch = taken(epochs)
    return kind(
        run_dir,
        queues=QueueSet(run_dir / RUN_QUEUES, profile, lease=profile.lease, epoch=epochs.current),
        store=ResultStore(tmp_path / "store"),
        events=EventLog(run_dir, f"a{epoch}", epoch, epoch=epoch),
        profile=profile,
        substrate=substrate if substrate is not None else Recording(),
        epoch=epoch,
        clock=clock,
        measures=FixedEstimates(),
        resolver=FixedUnit(profile),
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
    assert {f.placement.batch_size for f in items} == {1}


@pytest.mark.logic
def test_workers_wanted_counts_quanta_of_jobs_whatever_the_items(tmp_path: Path) -> None:
    """200 jobs put one to an item are 200 items but 20 quanta of ten jobs, so 20 workers are wanted, not 32."""
    placed, values, pool, _ = _submitted(tmp_path, _profile(), 200)
    admission = pool.admit(placed, values)
    for element in range(200):
        pool.submit(node_fragment(placed, admission, elements=[(element,)]))
    pool.close()
    assert pool.queues.queue("cpu").depth() == 200 and pool.workers_wanted("cpu", 200) == 20


@pytest.mark.logic
def test_a_resume_whose_every_item_is_live_scales_beyond_one_worker(tmp_path: Path) -> None:
    """Epoch 2 follows 200 one-job items epoch 1 queued: its jobs size the scaler as its own puts would."""
    profile = _profile()
    placed, values, first, fragment = _submitted(tmp_path, profile, 200)
    first.queues.queue("cpu").put_all(split(fragment, 1))
    first.close()
    second = _pool(tmp_path, profile)
    second.submit(node_fragment(placed, second.admit(placed, values)))
    second.close()
    assert second.queues.queue("cpu").depth() == 200 and second.workers_wanted("cpu", 200) == 20


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
    queues = [
        QueueSet(tmp_path / "run" / RUN_QUEUES, _profile(), lease=_profile().lease).queue("cpu") for _ in range(2)
    ]
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
    first = (
        QueueSet(tmp_path / "run" / RUN_QUEUES, _profile(), lease=_profile().lease)
        .queue("cpu")
        .claim("a", max_items=10, max_jobs=10)
    )
    second = (
        QueueSet(tmp_path / "run" / RUN_QUEUES, _profile(), lease=_profile().lease)
        .queue("cpu")
        .claim("b", max_items=10, max_jobs=10)
    )
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
    assert QueueSet(tmp_path / "run" / RUN_QUEUES, _profile(), lease=_profile().lease).queue("cpu").work() == 200
    assert not [n for n in opened if n.endswith(".json")]
    missing.unlink()
    for n in range(3):
        (segment / f"jobs-{n}-10").unlink()
    assert (
        QueueSet(tmp_path / "run" / RUN_QUEUES, _profile(), lease=_profile().lease).queue("cpu").work()
        == 200 - 30 + 1 + 1 + 10
    )
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
    """A `Recording` whose `serve` thread runs the `cpu` queue's items once its gate opens.

    The gate opens once `want` workers were started (at once for 0). With `hold` set, items of that node are held and
    every other item is served at once; the gate then opens once `want` workers are alive and `hold`'s jobs on the
    queue are `held_jobs`, which `peak` records. With `stall`, every `start` blocks, as a slow `sbatch` does, until
    `release` is set. With `queued_at`, `serve` sets its event once the `cpu` queue holds that many items. `poll_s` is
    the real seconds `serve` waits between looks at its queue.
    """

    def __init__(  # noqa: PLR0913 - the gate's settings
        self,
        want: int,
        hold: str | None = None,
        held_jobs: int = 0,
        stall: bool = False,
        poll_s: float = 0.0,
        queued_at: tuple[threading.Event, int] | None = None,
    ) -> None:
        """Serve once `want` workers were started (and `hold` has `held_jobs` queued)."""
        super().__init__()
        self.started = 0
        self.want, self.hold, self.held_jobs, self.stall, self.queued_at = want, hold, held_jobs, stall, queued_at
        self.poll_s = poll_s
        self.at_end = 0
        self.opened, self.release, self.reached = threading.Event(), threading.Event(), threading.Event()
        if hold is None and want <= 0:
            self.opened.set()
        self.peak = (0, 0)

    def start(self, resource_class: str, count: int) -> list[WorkerHandle]:
        """Hold `count` pending workers, once `release` is set if the start stalls, and count them."""
        if self.stall:
            self.release.wait()
        made = super().start(resource_class, count)
        self.started += len(made)
        if self.hold is None and self.started >= self.want:
            self.opened.set()
        if self.started >= self.want:
            self.reached.set()
        return made

    def serve(self, pool: Pool, placed: Plan, values: Mapping[InputIdentity, object], done: threading.Event) -> None:
        """Execute each item of the `cpu` queue until `done`: `hold`'s items only once the gate opens."""
        queue = QueueSet(pool.run_dir / RUN_QUEUES, placed.execution, lease=lease_of(placed)).queue("cpu")
        while not done.is_set():
            if self.queued_at is not None and queue.depth() >= self.queued_at[1]:
                self.queued_at[0].set()
            if self.hold is None and not self.opened.is_set():
                done.wait(self.poll_s)
                continue
            claim = queue.claim("gated", max_items=1)
            if claim is not None and claim.fragments[0].node == self.hold and not self.opened.is_set():
                self._watch(queue)
                claim.release()
                claim = None
            if claim is None:
                done.wait(self.poll_s)
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
    clock: Clock = SystemClock(),  # noqa: B008 - stateless
    noticed: threading.Event | None = None,
) -> Any:  # noqa: ANN401 - a `RunOutcome`
    """Run `placed` by `PydraRunner` on a pool over `gated`, its serving thread alongside, with 2 default threads.

    `noticed` is set when the runner cancels a followed handle on the pool.
    """
    runner = pytest.importorskip("senselab_core.runtime.execution.runner", exc_type=ImportError)

    class Noticing(Pool):
        def cancel(self, handle: Handle) -> None:
            if noticed is not None:
                noticed.set()
            super().cancel(handle)

    pool = _pool(tmp_path, profile, gated, clock, Noticing)
    done = threading.Event()
    server = threading.Thread(target=gated.serve, args=(pool, placed, values, done), daemon=True)
    server.start()
    try:
        with _default_threads(2):
            return runner.PydraRunner(profile=profile, work_dir=tmp_path / "pydra", clock=clock).run_plan(
                placed, pool, values=values, events=pool.events, cancel=cancel
            )
    finally:
        done.set()
        gated.at_end = gated.started
        gated.release.set()
        pool.close()
        server.join(30)


def _run_cancelled(  # noqa: PLR0913 - the plan and its gate, then the cancel and the clock that times it
    tmp_path: Path,
    profile: ExecutionProfile,
    placed: Plan,
    values: Mapping[InputIdentity, object],
    gated: Gated,
    clock: ManualClock,
) -> None:
    """Run `placed` until its items are queued and the follower waits, cancel, and time the notice on `clock`."""
    runner = pytest.importorskip("senselab_core.runtime.execution.runner", exc_type=ImportError)
    assert gated.queued_at is not None
    queued, cancel, noticed, ended = gated.queued_at[0], threading.Event(), threading.Event(), []

    def run() -> None:
        try:
            _run_gated(tmp_path, profile, placed, values, gated, cancel, clock, noticed)
        except BaseException as err:  # noqa: BLE001 - what the run ended with, read below
            ended.append(err)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        assert queued.wait(clock.hang_s)
        guard = time.monotonic() + clock.hang_s
        while not (
            waiting := [t for t in threading.enumerate() if t.name == "senselab-follower" and clock.is_parked(t)]
        ):
            assert time.monotonic() < guard, "no follower of this run parked on its clock"
            cancel.wait(clock.poll_s)
        follower = waiting[0]
        cancel.set()
        before = clock.now()
        while not noticed.is_set():
            assert time.monotonic() < guard, "the runner did not notice the cancel"
            if clock.is_parked(follower):
                clock.advance(profile.pool.cancel_check_s)
            noticed.wait(clock.poll_s)
        after = clock.now()
    finally:
        cancel.set()
        thread.join(clock.hang_s)
        gated.release.set()
    assert len(ended) == 1 and isinstance(ended[0], runner.RunCancelled)
    assert after - before <= profile.pool.cancel_check_s + clock.step_s


@pytest.mark.contract
def test_the_runner_queues_a_map_nodes_ready_jobs_so_scale_wants_a_worker_per_quantum(
    tmp_path: Path, fake_poll_s: float
) -> None:
    """A 200-row map node run by `PydraRunner` on a pool reaches 20 items of 10 jobs, so 20 workers are started."""
    profile = _profile()
    gated = Gated(want=20, poll_s=fake_poll_s)
    placed, values = planned(SCORE, profile, [f"t{i}" for i in range(200)])
    outcome = _run_gated(tmp_path, profile, placed, values, gated)
    assert len(outcome.results["a"]) == 200 and all(r.failure is None for r in outcome.results["a"])
    assert gated.started == 20


@pytest.mark.contract
def test_a_downstream_nodes_jobs_queue_as_their_upstream_lands_so_scale_wants_a_worker_per_quantum(
    tmp_path: Path, fake_poll_s: float
) -> None:
    """With `a`'s items served at once and `b`'s held, every one of `b`'s 200 jobs is queued and 20 workers are alive.

    The pydra loop has two default threads, so no thread may be held per fragment while it waits.
    """
    profile = _profile()
    gated = Gated(want=20, poll_s=fake_poll_s, hold="b", held_jobs=200)
    placed, values = _chained(profile, 200)
    outcome = _run_gated(tmp_path, profile, placed, values, gated)
    assert len(outcome.results["b"]) == 200 and all(r.failure is None for r in outcome.results["b"])
    assert gated.peak == (200, 20)


@pytest.mark.contract
def test_a_cancel_stops_following_the_pool_at_once_and_leaves_its_items_queued(
    tmp_path: Path, lease_clock: Callable[..., ManualClock], fake_poll_s: float
) -> None:
    """A cancel set while a 200-job node waits on closed gates ends the run cancelled within one check of it."""
    profile, clock = _profile(first_poll_s=0.2), lease_clock()
    gated = Gated(want=10**6, poll_s=fake_poll_s, queued_at=(threading.Event(), 20))
    placed, values = planned(SCORE, profile, [f"t{i}" for i in range(200)])
    queue = QueueSet(tmp_path / "run" / RUN_QUEUES, profile, lease=profile.lease).queue("cpu")
    _run_cancelled(tmp_path, profile, placed, values, gated, clock)
    assert queue.depth() == 20 and queue.work() == 200


@pytest.mark.contract
def test_a_slow_substrate_never_delays_settling(tmp_path: Path, fake_poll_s: float) -> None:
    """With each start stalled, a 60-job node whose items are served at once ends before any start returns."""
    profile = _profile()
    gated = Gated(want=0, poll_s=fake_poll_s, stall=True)
    placed, values = planned(SCORE, profile, [f"t{i}" for i in range(60)])
    outcome = _run_gated(tmp_path, profile, placed, values, gated)
    assert gated.at_end == 0
    assert len(outcome.results["a"]) == 60 and all(r.failure is None for r in outcome.results["a"])


@pytest.mark.contract
def test_a_slow_substrate_never_delays_a_cancel(
    tmp_path: Path, lease_clock: Callable[..., ManualClock], fake_poll_s: float
) -> None:
    """With each start stalled, a cancel set once the items are queued ends the run within one check of it."""
    profile, clock = _profile(first_poll_s=0.2), lease_clock()
    gated = Gated(want=10**6, poll_s=fake_poll_s, stall=True, queued_at=(threading.Event(), 20))
    placed, values = planned(SCORE, profile, [f"t{i}" for i in range(200)])
    queue = QueueSet(tmp_path / "run" / RUN_QUEUES, profile, lease=profile.lease).queue("cpu")
    _run_cancelled(tmp_path, profile, placed, values, gated, clock)
    assert gated.at_end == 0 and queue.work() == 200


@pytest.mark.logic
def test_concurrent_submits_never_start_more_workers_than_the_target(
    tmp_path: Path, tier_bound: float, lease_clock: Callable[..., ManualClock], fake_poll_s: float
) -> None:
    """48 senders submit a 10-job fragment each at once; the class's 32 workers are started, never more."""
    profile = _profile()
    gated, clock = Gated(want=32, poll_s=fake_poll_s, stall=True), lease_clock()
    placed, values = planned(SCORE, profile, [f"t{i}" for i in range(480)])
    pool = _pool(tmp_path, profile, gated, clock)
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
        sender.join(tier_bound)
    gated.release.set()
    for _ in range(3):
        clock.advance(profile.pool.poll_s)
        clock.settle(0)
    assert gated.reached.wait(tier_bound)
    pool.close()
    assert pool.queues.queue("cpu").work() == 480 and gated.started == 32
