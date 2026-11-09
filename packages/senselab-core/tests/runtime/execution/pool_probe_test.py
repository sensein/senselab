"""The pool's probe state machine: resolve, look up, wait on a peer, probe on a pinned queue, then size by the estimate.

Workers are played by hand: a `Hand` claims items, runs their jobs through `InProcess` and reports the observations a
worker would. Every wait is read on an injected clock; no test sleeps or reads a wall time.
"""

import logging
import os
import platform
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path

import pytest
from lease_clock_fixture import ManualClock
from pool_stack_fixture import host_env, scoring
from senselab_core.kernel.dataset import Record, ValueInput
from senselab_core.kernel.declare import Registry
from senselab_core.kernel.identity import InputIdentity
from senselab_core.runtime.calibration import Cotenancy, HardwareClass, Observation, ProbeRef
from senselab_core.runtime.call import Packaged, RunConfig, bind_input, packaged, use_packaged
from senselab_core.runtime.catalogue import Catalogue
from senselab_core.runtime.datasets import RecordSet
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.execution.epoch import Epoch
from senselab_core.runtime.execution.events import EventLog
from senselab_core.runtime.execution.executor import Handle, PlanFragment, ResultRef
from senselab_core.runtime.execution.in_process import InProcess, fragment_jobs, job_units, node_fragment
from senselab_core.runtime.execution.job import JobOutcome
from senselab_core.runtime.execution.observing import request_units
from senselab_core.runtime.execution.pool import Pool, report
from senselab_core.runtime.execution.pool_parts import placed_backends
from senselab_core.runtime.execution.queue import RUN_QUEUES, QueueSet, item_digest, probe_of, probe_queue
from senselab_core.runtime.execution.run import prepare
from senselab_core.runtime.execution.substrates import QueueReady, Substrate, WorkerHandle, backend_key, mark_ready
from senselab_core.runtime.execution.workers import WorkerSet
from senselab_core.runtime.hardware import HardwareInfo
from senselab_core.runtime.licenses import license_record
from senselab_core.runtime.locks import Lease
from senselab_core.runtime.measures import Build, MeasureUnit, Provenance
from senselab_core.runtime.planner import Plan, plan
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.store import ResultStore
from senselab_core.runtime.units import UnitResolver, Unresolved, mark_entry, measure_roots
from senselab_core.testing.measures import FixedEstimates, FixedUnit

TEXTY = scoring("pool_probe_test.texty").model_copy(update={"inputs": {"text": "Text"}})
BUCKETED = scoring("pool_probe_test.bucketed").model_copy(update={"inputs": {"text": "Audio"}})
HOST = HardwareInfo(
    cpu_count=4, memory_bytes=2**33, gpus=(), apple_silicon=False, platform="linux-x86_64", cpu_model="Test CPU"
)
SEEN = HardwareClass(device="cpu", name="Test CPU", cpus=4)
ELSEWHERE = HardwareClass(device="cpu", name="Another CPU", cpus=4)
JOBS = 60
WORKERS = WorkerSet(2)


class Idle:
    """A `Substrate` that starts no worker."""

    def start(self, resource_class: str, count: int) -> list[WorkerHandle]:  # noqa: ARG002
        """Start nothing."""
        return []

    def live(self) -> list[WorkerHandle]:
        """Return no worker."""
        return []

    def running(self, resource_class: str) -> int:  # noqa: ARG002
        """Return no worker."""
        return 0

    def stop(self, handles: Sequence[WorkerHandle] = ()) -> None:
        """Stop nothing."""


def _handle(wheel_set: str) -> Callable[..., EnvHandle]:
    def read(backend: object, profile: object) -> EnvHandle:
        base = host_env(backend, profile)  # type: ignore[arg-type]
        return EnvHandle(
            python=base.python,
            python_version=platform.python_version(),
            lock_sha=base.lock_sha,
            wheel_set=wheel_set,
            torch_index=base.torch_index,
        )

    return read


@contextmanager
def _env(wheel_set: str, registry: Registry) -> Iterator[None]:
    """Use `registry` with every env this interpreter, built, under the wheel set `wheel_set`."""
    mit = license_record("MIT", source="lock", verified={"date": "2026-11-06", "how": "pep639"}, gated=False)
    opener = _handle(wheel_set)
    previous = use_packaged(
        Packaged(
            Catalogue.load("format: catalogue@2.0\nentries: []\n"),
            opener,
            registry,
            lambda _: {"p": mit},
            env_read=opener,
        )
    )
    try:
        yield
    finally:
        use_packaged(previous)


@pytest.fixture(autouse=True)
def stack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A registry of one text task, its env built, the cache under `tmp_path`."""
    from pool_stack_fixture import registry  # noqa: PLC0415 - one name in this module's namespace

    monkeypatch.setenv("SENSELAB_CACHE", str(tmp_path / "cache"))
    with _env("4" * 64, registry(TEXTY)):
        yield
    WORKERS.close()


def _profile(tmp_path: Path) -> ExecutionProfile:
    override = {
        "devices": ["cpu"],
        "resource_classes": {"cpu": {"device": "cpu", "mem_bytes": 2**33, "max_workers": 2}},
        "pool": {"poll_s": 0.2, "first_poll_s": 0.01, "quantum_items": 10},
        "calibration": {"measures_roots": [str(tmp_path / "measures")], "warmup_requests": 1},
    }
    return ExecutionProfile.load("laptop", override)


def _plan(
    profile: ExecutionProfile, lengths: Sequence[int], model: str | None = None, char: str = "x"
) -> tuple[Plan, dict[InputIdentity, object]]:
    """Return the plan mapping the text task over one record per length, a text of that many `char`s."""
    texts = [char * n for n in lengths]
    graph = {
        "inputs": {"text": "list[Text]"},
        "nodes": [{"id": "a", "task": TEXTY.name, "task_version": "1", "model": model}],
        "edges": [{"source": "input.text", "target": "a.text", "kind": "map"}],
    }
    from senselab_core.kernel.graph import Graph  # noqa: PLC0415 - as `pool_stack_fixture.planned`

    records = [Record(id=f"r{i}", inputs={"text": ValueInput(value=t)}) for i, t in enumerate(texts)]
    placed = plan(Graph.model_validate(graph), RecordSet(records), config=RunConfig(), execution=profile)
    return placed, {bind_input("x", "Text", t)[0]: bind_input("x", "Text", t)[1] for t in texts}


def _seen(profile: ExecutionProfile, *classes: HardwareClass) -> None:
    roots = measure_roots(profile)
    for seen in classes:
        assert roots.note_seen(profile.resource_classes["cpu"], seen) is not None


def _pool(  # noqa: PLR0913 - the run's places, its clock and the seams a test replaces
    tmp_path: Path,
    profile: ExecutionProfile,
    *,
    run: str = "run",
    store: str = "store",
    clock: ManualClock,
    substrate: Substrate | None = None,
    **seams: object,
) -> Pool:
    run_dir = tmp_path / run
    epochs = Epoch(run_dir)
    epoch = epochs.take()
    roots = measure_roots(profile)
    given: dict[str, object] = {"roots": roots, "resolver": UnitResolver(profile, roots, hardware=HOST), **seams}
    return Pool(
        run_dir,
        queues=QueueSet(run_dir / RUN_QUEUES, profile, epoch=epochs.current),
        store=ResultStore(tmp_path / store),
        events=EventLog(run_dir, f"a{epoch}", epoch, epoch=epoch),
        profile=profile,
        substrate=substrate if substrate is not None else Idle(),
        epoch=epoch,
        clock=clock,
        **given,  # type: ignore[arg-type]
    )


def _submit(pool: Pool, placed: Plan, values: Mapping[InputIdentity, object]) -> tuple[PlanFragment, str]:
    fragment = node_fragment(placed, pool.admit(placed, values))
    return fragment, pool.submit(fragment).id


class Hand:
    """Plays a pool worker of class `hardware` by hand: claims each item, runs it, reports its observations.

    A probe queue is taken only where its recorded class is `hardware`, as `PoolWorker` does. `wall` is a request's
    seconds from its units, `load_s` the load the first request reports, `limit` the most probe items it claims, and
    `served` the item (by claim order) whose requests are reported `served`; `max_items` the most items it takes in
    one `serve`.
    """

    def __init__(  # noqa: PLR0913 - the plan and run, then the worker's behaviour
        self,
        pool: Pool,
        placed: Plan,
        values: Mapping[InputIdentity, object],
        hardware: HardwareClass = SEEN,
        wall: Callable[[float], float] = lambda units: 0.05 + 0.01 * units,
        load_s: float = 0.5,
        limit: int | None = None,
        served: int | None = None,
        takes: HardwareClass | None = None,
        max_items: int | None = None,
    ) -> None:
        """Serve `pool`'s queues for `placed`."""
        self.pool, self.placed, self.values, self.hardware = pool, placed, values, hardware
        self.wall, self.load_s, self.limit, self.served = wall, load_s, limit, served
        self.takes = takes if takes is not None else hardware
        self.max_items = max_items
        self.since = 0
        self.items: list[tuple[str, PlanFragment]] = []
        self.probe_items = 0

    def serve(self) -> int:
        """Claim and run every item now claimable; return how many."""
        count = 0
        for name in self.pool.queues.names():
            queue = self.pool.queues.queue(name)
            probe = probe_of(name) is not None
            if probe and (queue.probe_hardware() != self.takes or self.limit == self.probe_items):
                continue
            while (claim := queue.claim("hand", max_items=1)) is not None:
                if (probe and self.limit == self.probe_items) or count == self.max_items:
                    claim.release()
                    break
                [item], [digest] = claim.fragments, claim.items
                with claim:
                    outcomes, seen = self._run(name, digest, item, probe=probe)
                    report(claim, digest, outcomes, observations=seen)
                self.items.append((name, item))
                self.probe_items += probe
                count += 1
        return count

    def _run(
        self, name: str, digest: str, item: PlanFragment, *, probe: bool
    ) -> tuple[list[JobOutcome], list[Observation]]:
        """Return the outcomes of `item` and the observations its worker reports: a probe item's, else none."""
        return self._outcomes(item), self._observations(name, digest, item) if probe else []

    def _outcomes(self, item: PlanFragment) -> list[JobOutcome]:
        executor = InProcess(store=self.pool.store, profile=self.placed.execution, run=RunConfig(), workers=WORKERS)
        admission = executor.admit(self.placed, self.values)
        return list(executor.outcomes(executor.submit(item.model_copy(update={"admission": admission.id}))))

    def _observations(self, name: str, digest: str, item: PlanFragment) -> list[Observation]:
        queue, (_, probe_id) = self.pool.queues.queue(name), probe_of(name) or ("", "")
        phase, stratum = queue.probe_mark(digest)
        jobs, _ = fragment_jobs(item, self.placed, self.values)
        size, found = item.placement.batch_size, []
        for index, start in enumerate(range(0, len(jobs), size)):
            kind, units, padded = request_units(jobs[start : start + size])
            walls = self.wall(units)
            last = start + size >= len(jobs)
            found.append(
                Observation.model_validate(
                    {
                        "probe": ProbeRef.model_validate({"id": probe_id, "phase": phase, "stratum": stratum}),
                        "item": digest,
                        "request_index": index,
                        "jobs": len(jobs[start : start + size]),
                        "batch": len(jobs[start : start + size]),
                        "unit_kind": kind,
                        "units": units,
                        "padded_units": padded,
                        "wall_s": walls,
                        "item_wall_s": walls + 0.01 if last else None,
                        "since_load": self.since,
                        "load_s": self.load_s if self.since == 0 else None,
                        "device_peak": None,
                        "device_after": None,
                        "host_peak": None,
                        "host_after": None,
                        "cotenancy": Cotenancy(device=1, host=1),
                        "outcome": "served" if self.served == self.probe_items else "ok",
                        "hardware": self.hardware,
                    }
                )
            )
            self.since += 1
        return found


def _drive(pool: Pool, handle: str, hand: Hand, rounds: int = 40) -> list[ResultRef]:
    """Poll `handle` with `hand` serving between polls until it ends; return every result, in the order given."""
    given: list[ResultRef] = []
    for _ in range(rounds):
        hand.serve()
        results, done = pool.poll(Handle(id=handle))
        given.extend(results)
        if done:
            return given
    raise AssertionError(f"the fragment did not end in {rounds} rounds; {len(given)} results given")


def _hello(pool: Pool, placed: Plan, values: Mapping[InputIdentity, object], hardware: HardwareClass) -> None:
    """Write the ready mark a worker of class `hardware` writes for the plan's one node: its class and built env."""
    jobs, placement = fragment_jobs(node_fragment(placed, pool.admit(placed, values)), placed, values)
    handle = packaged().open_env(placement.backend, pool.profile)
    entry = mark_entry(pool.profile, placement.backend, handle, HOST, "cpu")
    mark_ready(pool.run_dir, {"cpu": QueueReady(hardware, {backend_key(placement.backend): entry})})
    del jobs


def _probe_queues(pool: Pool) -> list[str]:
    return [n for n in pool.queues.names() if probe_of(n) is not None]


@pytest.mark.logic
def test_probe_results_are_yielded_and_never_rerun(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, clock: ManualClock
) -> None:
    """Every job's result comes once, probe jobs' included; no job is claimed twice; the class is on the queue."""
    profile = _profile(tmp_path)
    _seen(profile, SEEN)
    placed, values = _plan(profile, range(1, JOBS + 1))
    pool = _pool(tmp_path, profile, clock=clock)
    fragment, handle = _submit(pool, placed, values)
    [probe] = _probe_queues(pool)
    assert pool.queues.queue(probe).probe_hardware() == SEEN and pool.queues.queue("cpu").work() == 0
    with caplog.at_level(logging.INFO, logger="senselab_core.runtime.sizing"):
        refs = _drive(pool, handle, Hand(pool, placed, values))
    assert sorted(r.job.element for r in refs) == sorted(j.element for j in fragment.jobs)
    assert all(r.failure is None and pool.store.outputs_of(r.key_digest, str(r.output_identity)) for r in refs)
    [unit] = pool.probes.values()
    assert unit.jobs > 0 and len(unit.items) > 0
    assert len([r for r in caplog.records if "extrapolated" in r.getMessage()]) == 1
    seen = [(n, j.element) for n, item in _claims(pool) for j in item.jobs]
    assert len({e for _, e in seen}) == len(seen) == JOBS


def _claims(pool: Pool) -> list[tuple[str, PlanFragment]]:
    """Return each item of each queue, by name: what the run put, claimed or not."""
    found = []
    for name in pool.queues.names():
        queue = pool.queues.queue(name)
        for digest in sorted(p.name.removesuffix(".json") for p in (queue.path / "items").iterdir()):
            with queue.open() as q, q.child("items") as items:
                data = items.read_regular(f"{digest}.json")
            assert data is not None
            found.append((name, PlanFragment.model_validate_json(data)))
    return found


def _stored(placed: Plan, values: Mapping[InputIdentity, object], store: ResultStore, elements: range) -> None:
    """Run jobs `elements` of the plan into `store`, in the env the stack gives."""
    executor = InProcess(store=store, profile=placed.execution, run=RunConfig(), workers=WORKERS)
    admission = executor.admit(placed, values)
    node = node_fragment(placed, admission, elements=[(i,) for i in elements])
    list(executor.outcomes(executor.submit(node)))


@pytest.mark.logic
def test_servable_jobs_are_put_at_once(tmp_path: Path, clock: ManualClock) -> None:
    """Jobs the store serves are queued before any probe outcome; the computed ones are on the probe queue."""
    profile = _profile(tmp_path)
    _seen(profile, SEEN)
    placed, values = _plan(profile, range(1, JOBS + 1))
    pool = _pool(tmp_path, profile, clock=clock)
    _stored(placed, values, pool.store, range(20))
    _submit(pool, placed, values)
    served = [j.element for n, item in _claims(pool) if n == "cpu" for j in item.jobs]
    probed = [j.element for n, item in _claims(pool) if probe_of(n) for j in item.jobs]
    assert sorted(served) == [(i,) for i in range(20)] and pool.queues.queue("cpu").work() == 20
    assert probed and not set(probed) & set(served)


@pytest.mark.logic
def test_entries_under_another_context_are_probed_not_skipped(tmp_path: Path, clock: ManualClock) -> None:
    """Results stored under an old env's context are not served: their jobs are probed and computed."""
    profile = _profile(tmp_path)
    _seen(profile, SEEN)
    placed, values = _plan(profile, range(1, JOBS + 1))
    pool = _pool(tmp_path, profile, clock=clock)
    with _env("5" * 64, _registry()):
        _stored(placed, values, pool.store, range(JOBS))
    _, handle = _submit(pool, placed, values)
    assert pool.queues.queue("cpu").work() == 0 and len(_probe_queues(pool)) == 1
    refs = _drive(pool, handle, Hand(pool, placed, values))
    assert len(refs) == JOBS and all(r.served is None for r in refs)


def _registry() -> Registry:
    from pool_stack_fixture import registry  # noqa: PLC0415 - as the stack fixture

    return registry(TEXTY)


@pytest.mark.logic
def test_a_served_probe_request_is_replaced_from_its_stratum(tmp_path: Path, clock: ManualClock) -> None:
    """An item whose request came back served is replaced by one of the same phase and stratum over other jobs."""
    profile = _profile(tmp_path)
    _seen(profile, SEEN)
    placed, values = _plan(profile, range(1, JOBS + 1))
    pool = _pool(tmp_path, profile, clock=clock)
    _, handle = _submit(pool, placed, values)
    [unit] = pool.probes.values()
    before = set(unit.items)
    hand = Hand(pool, placed, values, limit=1, served=0)
    hand.serve()
    pool.poll(Handle(id=handle))
    [after] = pool.probes.values()
    [new] = after.items - before
    queue = pool.queues.queue(probe_queue("cpu", after.probe))
    [(_, first)] = hand.items
    items = {item_digest(i): i for _, i in _claims(pool)}
    assert queue.probe_mark(new) == queue.probe_mark(item_digest(first))
    assert not set(items[new].jobs) & set(first.jobs) and len(items[new].jobs) == len(first.jobs)


@pytest.mark.logic
def test_held_work_waits_at_most_probe_s_of_warm_work(tmp_path: Path, clock: ManualClock) -> None:
    """Past `probe_s` of warm requests the probe ends: its held jobs are put while probe items are still unclaimed."""
    profile = _profile(tmp_path)
    _seen(profile, SEEN)
    placed, values = _plan(profile, range(1, JOBS + 1))
    pool = _pool(tmp_path, profile, clock=clock)
    _, handle = _submit(pool, placed, values)
    [unit] = pool.probes.values()
    queue = pool.queues.queue(probe_queue("cpu", unit.probe))
    Hand(pool, placed, values, wall=lambda units: profile.probe_s, limit=3).serve()
    assert pool.queues.queue("cpu").work() == 0
    pool.poll(Handle(id=handle))
    assert pool.queues.queue("cpu").work() > 0 and queue.depth() > 0


@pytest.mark.logic
@pytest.mark.parametrize(("budget", "ended"), [(None, False), (0.01, True)])
def test_a_pool_given_probe_s_ends_its_probe_at_that_budget(
    tmp_path: Path, clock: ManualClock, budget: float | None, ended: bool
) -> None:
    """A budget given to the pool replaces the profile's `probe_s`: the same warm walls end a probe or not."""
    profile = _profile(tmp_path)
    _seen(profile, SEEN)
    placed, values = _plan(profile, range(1, JOBS + 1))
    given = None if budget is None else budget * profile.probe_s
    pool = _pool(tmp_path, profile, clock=clock, probe_s=given)
    _, handle = _submit(pool, placed, values)
    [unit] = pool.probes.values()
    queue = pool.queues.queue(probe_queue("cpu", unit.probe))
    Hand(pool, placed, values, wall=lambda units: 0.01 * profile.probe_s, limit=3).serve()
    pool.poll(Handle(id=handle))
    assert (pool.queues.queue("cpu").work() > 0 and queue.depth() > 0) is ended


@pytest.mark.logic
def test_a_stand_in_with_a_long_load_is_still_measured(tmp_path: Path, clock: ManualClock) -> None:
    """A load many times `probe_s` is not warm work: the probe runs on and its unit is measured."""
    profile = _profile(tmp_path)
    _seen(profile, SEEN)
    placed, values = _plan(profile, range(1, JOBS + 1))
    pool = _pool(tmp_path, profile, clock=clock)
    _, handle = _submit(pool, placed, values)
    _drive(pool, handle, Hand(pool, placed, values, load_s=1000 * profile.probe_s))
    assert pool.measures.get(_unit(pool, placed, values)) is not None


def _unit(pool: Pool, placed: Plan, values: Mapping[InputIdentity, object]) -> MeasureUnit:
    """Return the unit of the plan's one node as the pool resolves it."""
    jobs, placement = fragment_jobs(node_fragment(placed, pool.admit(placed, values)), placed, values)
    unit = pool.resolver.unit(jobs[0].call, placement, None)
    assert isinstance(unit, MeasureUnit)
    return unit


@pytest.mark.logic
def test_a_second_run_reuses_the_first_runs_measure_without_probing(tmp_path: Path, clock: ManualClock) -> None:
    """A second run over the same unit reads the first's written observations: no probe queue, items sized."""
    profile = _profile(tmp_path)
    _seen(profile, SEEN)
    placed, values = _plan(profile, range(1, JOBS + 1))
    first = _pool(tmp_path, profile, clock=clock)
    _, handle = _submit(first, placed, values)
    assert len(_drive(first, handle, Hand(first, placed, values))) == JOBS
    [probed] = first.probes.values()
    second = _pool(tmp_path, profile, run="run2", store="store2", clock=clock)
    _, again = _submit(second, placed, values)
    estimate = second.measures.get(_unit(second, placed, values))
    assert estimate is not None and probed.probe in estimate.source.probe_ids
    assert not _probe_queues(second) and not second.probes and second.queues.queue("cpu").work() == JOBS
    assert len(_drive(second, again, Hand(second, placed, values))) == JOBS


def _peer_observations(count: int, lengths: Sequence[int]) -> list[Observation]:
    found = []
    for i in range(count):
        n = lengths[i % len(lengths)]
        found.append(
            Observation.model_validate(
                {
                    "probe": {"id": "peer", "phase": "SINGLE", "stratum": 0},
                    "item": f"i{i}",
                    "request_index": 0,
                    "jobs": 1,
                    "batch": 1,
                    "unit_kind": "chars",
                    "units": float(n),
                    "padded_units": float(n),
                    "wall_s": 0.05 + 0.01 * n,
                    "item_wall_s": 0.1,
                    "since_load": i + 1,
                    "load_s": None,
                    "device_peak": None,
                    "device_after": None,
                    "host_peak": None,
                    "host_after": None,
                    "cotenancy": {"device": 1, "host": 1},
                    "outcome": "ok",
                    "hardware": SEEN,
                }
            )
        )
    return found


def _peer(pool: Pool, placed: Plan, values: Mapping[InputIdentity, object]) -> ExitStack:
    """Make another process hold the probe lease of the plan's unit; closing the returned stack releases it."""
    lease = pool.roots.probe_lease(_unit(pool, placed, values))
    assert isinstance(lease, Lease)
    stack = ExitStack()
    stack.enter_context(lease)
    return stack


@pytest.mark.logic
def test_a_peer_wait_returns_to_planning_for_uncovered_strata(tmp_path: Path, clock: ManualClock) -> None:
    """While a peer probes, the unit waits; its observations cover stratum 0, so only stratum 1 is probed after."""
    profile = _profile(tmp_path)
    _seen(profile, SEEN)
    placed, values = _plan(profile, range(1, JOBS + 1))
    pool = _pool(tmp_path, profile, clock=clock)
    peer = _peer(pool, placed, values)
    _, handle = _submit(pool, placed, values)
    assert not _probe_queues(pool) and not pool.probes
    unit = _unit(pool, placed, values)
    provenance = Provenance(
        job=None, run="peer", senselab=Build(version="2.0.0"), profile=profile.canonical_digest(), hardware=SEEN
    )
    pool.roots.put_observations(unit, "peer", _peer_observations(8, [3, 5, 8, 10]), provenance)
    peer.close()
    clock.advance(profile.pool.poll_s + 1)
    pool.poll(Handle(id=handle))
    [unit_probe] = pool.probes.values()
    queue = pool.queues.queue(probe_queue("cpu", unit_probe.probe))
    marks = [queue.probe_mark(d) for d in unit_probe.items]
    assert ("SINGLE", 1) in marks and ("SINGLE", 0) not in marks


@pytest.mark.logic
def test_a_peer_that_holds_past_peer_wait_s_is_not_waited_for_again(tmp_path: Path, clock: ManualClock) -> None:
    """A unit waits `peer_wait_s` on its injected clock; then it probes unleased, and waits no more."""
    profile = _profile(tmp_path)
    _seen(profile, SEEN)
    placed, values = _plan(profile, range(1, JOBS + 1))
    pool = _pool(tmp_path, profile, clock=clock)
    peer = _peer(pool, placed, values)
    _, handle = _submit(pool, placed, values)
    clock.advance(profile.peer_wait_s - 1)
    pool.poll(Handle(id=handle))
    assert not _probe_queues(pool)
    clock.advance(2)
    pool.poll(Handle(id=handle))
    assert len(_probe_queues(pool)) == 1
    peer.close()


@pytest.mark.logic
def test_a_point_whose_probe_measured_nothing_ships_no_measure(tmp_path: Path, clock: ManualClock) -> None:
    """A probe whose requests all failed gives no estimate: its jobs run on the safe values and none is written.

    The next run finds no estimate and probes again.
    """
    profile = _profile(tmp_path)
    _seen(profile, SEEN)
    placed, values = _plan(profile, range(1, JOBS + 1))
    pool = _pool(tmp_path, profile, clock=clock)
    _, handle = _submit(pool, placed, values)

    class Broken(Hand):
        def _observations(self, name: str, digest: str, item: PlanFragment) -> list[Observation]:
            return [o.model_copy(update={"outcome": "failed"}) for o in super()._observations(name, digest, item)]

    assert len(_drive(pool, handle, Broken(pool, placed, values))) == JOBS
    assert pool.measures.get(_unit(pool, placed, values)) is None
    again = _pool(tmp_path, profile, run="run2", store="store2", clock=clock)
    _submit(again, placed, values)
    assert len(_probe_queues(again)) == 1


@pytest.mark.logic
def test_a_probe_queue_with_no_eligible_worker_ends_unmeasured_and_its_items_run_as_production(
    tmp_path: Path, clock: ManualClock
) -> None:
    """After the lease's stale window with no worker of the class, the unit is unmeasured and nothing waits on it."""
    profile = _profile(tmp_path)
    _seen(profile, SEEN)
    placed, values = _plan(profile, range(1, JOBS + 1))
    pool = _pool(tmp_path, profile, clock=clock)
    fragment, handle = _submit(pool, placed, values)
    assert len(_probe_queues(pool)) == 1
    hand = Hand(pool, placed, values, hardware=ELSEWHERE)
    pool.poll(Handle(id=handle))
    assert pool.queues.queue("cpu").work() == 0
    clock.advance(profile.lease.stale_after_s + 1)
    pool.poll(Handle(id=handle))
    assert pool.queues.queue("cpu").work() == JOBS
    refs = _drive(pool, handle, hand)
    assert sorted(r.job.element for r in refs) == sorted(j.element for j in fragment.jobs)
    assert all(n == "cpu" for n, _ in hand.items)


@pytest.mark.logic
def test_a_unit_resolves_on_the_first_hello_and_the_hello_wins(tmp_path: Path, clock: ManualClock) -> None:
    """With the host's class not in the index, jobs wait and a worker is wanted; the first hello's class decides."""
    profile = _profile(tmp_path)
    placed, values = _plan(profile, range(1, JOBS + 1))
    pool = _pool(tmp_path, profile, clock=clock)
    _, handle = _submit(pool, placed, values)
    assert not _probe_queues(pool) and pool.queues.queue("cpu").work() == 0
    assert pool.workers_wanted("cpu", 0) == 1
    told = HardwareClass(device="cpu", name="Test CPU", cpus=2)
    _hello(pool, placed, values, told)
    pool.poll(Handle(id=handle))
    [probe] = _probe_queues(pool)
    assert pool.queues.queue(probe).probe_hardware() == told
    assert pool.roots.seen_classes(profile.resource_classes["cpu"]) == (told,)
    assert pool.workers_wanted("cpu", 0) == 1 and pool.workers_wanted(probe, 5) == 0


@pytest.mark.logic
def test_the_pool_and_the_in_process_executor_size_one_unit_alike(tmp_path: Path, clock: ManualClock) -> None:
    """Over one estimate and one unit, the pool's items and the executor's sizing give each job the same batch."""
    from senselab_core.runtime.calibration import (  # noqa: PLC0415 - one estimate, built here
        Estimate,
        MemoryFit,
        RequestFit,
        StratumBatch,
    )

    estimate = Estimate(
        status="measured",
        unit_kind="chars",
        load_s=1.0,
        load_n=3,
        item_s=0.01,
        item_n=3,
        single=RequestFit(batch=1, r=0.1, u=0.01, n=6, spread=0.1, lo=1.0, hi=100.0),
        batched=(RequestFit(batch=4, r=0.1, u=0.001, n=6, spread=0.1, lo=1.0, hi=400.0),),
        batch=(StratumBatch(lo=1.0, hi=100.0, batch=4),),
        host=MemoryFit(resident=2**20, m0=0.0, m1=1.0, n=6),
    )
    profile = _profile(tmp_path)
    lengths = [3, 12, 25, 60, 90, 7, 33, 41]
    placed, values = _plan(profile, lengths)
    pool = _pool(tmp_path, profile, measures=FixedEstimates(estimate), resolver=FixedUnit(profile), clock=clock)
    fragment, _ = _submit(pool, placed, values)
    per_job = {j.element: item.placement.batch_size for _, item in _claims(pool) for j in item.jobs}
    jobs, placement = fragment_jobs(fragment, placed, values)
    executor = InProcess(
        store=pool.store,
        profile=profile,
        run=RunConfig(),
        workers=WORKERS,
        measures=FixedEstimates(estimate),  # type: ignore[arg-type]
        units=lambda call, put, decoder: FixedUnit(profile).unit(call, put, decoder),
    )
    sized = executor.sizing(jobs, placement)
    wanted = {j.ref.element: sized.batch_at(float(n), len(jobs)) for j, n in zip(jobs, lengths, strict=True)}
    assert per_job == wanted and set(per_job.values()) == {4}


def _audio_plan(
    profile: ExecutionProfile, folder: Path, seconds: Sequence[float]
) -> tuple[Plan, dict[InputIdentity, object]]:
    """Return the plan mapping the bucketed audio task over one wav of each duration in `seconds`."""
    import numpy  # noqa: PLC0415 - the one test that writes audio
    import soundfile  # noqa: PLC0415 - as above
    from senselab_core.kernel.graph import Graph  # noqa: PLC0415 - as `_plan`

    paths = []
    for i, n in enumerate(seconds):
        soundfile.write(folder / f"{i}.wav", numpy.zeros(int(16000 * n), dtype="float32"), 16000)
        paths.append(str(folder / f"{i}.wav"))
    graph = {
        "inputs": {"text": "list[Audio]"},
        "nodes": [{"id": "a", "task": BUCKETED.name, "task_version": "1"}],
        "edges": [{"source": "input.text", "target": "a.text", "kind": "map"}],
    }
    records = [Record(id=f"r{i}", inputs={"text": ValueInput(value=p)}) for i, p in enumerate(paths)]
    placed = plan(Graph.model_validate(graph), RecordSet(records), config=RunConfig(), execution=profile)
    return placed, {bind_input("x", "Audio", p)[0]: bind_input("x", "Audio", p)[1] for p in paths}


@pytest.mark.logic
def test_an_unmeasured_units_items_end_where_the_backends_bucket_changes(tmp_path: Path, clock: ManualClock) -> None:
    """With no estimate the backend's bucket edges still cut the items: no item holds jobs of two buckets."""
    from pool_stack_fixture import registry  # noqa: PLC0415 - as the stack fixture
    from senselab_core.kernel.declare import BackendSpec, Registry  # noqa: PLC0415 - one registry

    base = registry(BUCKETED).backends(BUCKETED.name)[0]
    found = Registry()
    found.add_task(BUCKETED)
    found.add_backend(
        BackendSpec.model_validate(
            {**base.model_dump(), "batch_invariant": False, "output_affecting": ("batch_formation",)}
        )
    )
    with _env("4" * 64, found):
        profile = _profile(tmp_path)
        placed, values = _audio_plan(profile, tmp_path, [0.5] * 6 + [3.0] * 6)
        unresolved = FixedUnit(profile, Unresolved("a final cause"))
        pool = _pool(tmp_path, profile, measures=FixedEstimates(None), resolver=unresolved, clock=clock)
        _submit(pool, placed, values)
        items = [item for n, item in _claims(pool) if n == "cpu"]
        spans = [{job_units(j, "audio_s") for j in fragment_jobs(i, placed, values)[0]} for i in items]
    assert all(len(span) == 1 for span in spans) and sum(len(i.jobs) for i in items) == 12


class Live(Idle):
    """A `Substrate` with one live, started worker of `cpu` that never writes a ready mark."""

    def live(self) -> list[WorkerHandle]:
        """Return the worker."""
        return [WorkerHandle("cpu", "w0")]


@pytest.mark.logic
def test_resolving_past_resolve_s_runs_unmeasured_with_its_reason(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, clock: ManualClock
) -> None:
    """From a live worker's first poll, a unit waits `resolve_s` for a mark, then runs the safe values and says why."""
    profile = _profile(tmp_path)
    placed, values = _plan(profile, range(1, JOBS + 1))
    pool = _pool(tmp_path, profile, clock=clock, substrate=Live())
    fragment, handle = _submit(pool, placed, values)
    pool.poll(Handle(id=handle))
    clock.advance(profile.calibration.resolve_s - 1)
    pool.poll(Handle(id=handle))
    assert pool.queues.queue("cpu").work() == 0 and not _probe_queues(pool)
    clock.advance(2)
    with caplog.at_level(logging.WARNING, logger="senselab"):
        pool.poll(Handle(id=handle))
    assert pool.queues.queue("cpu").work() == JOBS
    assert [r for r in caplog.records if "no readable ready mark from a worker of cpu" in r.getMessage()]
    refs = _drive(pool, handle, Hand(pool, placed, values))
    assert sorted(r.job.element for r in refs) == sorted(j.element for j in fragment.jobs)


@pytest.mark.logic
def test_a_broken_class_with_no_hello_fails_its_held_parts(tmp_path: Path, clock: ManualClock) -> None:
    """Workers of a class that die before their mark break it; the jobs a Resolving unit holds are then reported."""
    import sys  # noqa: PLC0415 - one test starts real processes

    from senselab_core.runtime.execution.executor import WorkersBroken, follow  # noqa: PLC0415 - one use
    from senselab_core.runtime.execution.substrates import Local  # noqa: PLC0415 - one use

    profile = ExecutionProfile.load(
        "laptop",
        {
            "devices": ["cpu"],
            "resource_classes": {"cpu": {"device": "cpu", "mem_bytes": 2**33, "max_workers": 2}},
            "calibration": {"measures_roots": [str(tmp_path / "measures")]},
            "pool": {"poll_s": 0.2, "first_poll_s": 0.01},
            "faults": {"infrastructure_replacements": 1},
            "substrate": {"start_backoff_first_s": 0.01, "start_backoff_max_s": 0.01},
        },
    )
    placed, values = _plan(profile, range(1, 5))
    local = Local(2, command=[sys.executable, "-c", "raise SystemExit(1)"])
    pool = _pool(tmp_path, profile, substrate=local, clock=clock)
    handle = pool.submit(node_fragment(placed, pool.admit(placed, values)))
    try:
        with pytest.raises(WorkersBroken):
            list(follow(pool, handle, clock=clock))
    finally:
        pool.close()
    assert not _probe_queues(pool) and "cpu" in pool.broken_workers


@pytest.mark.logic
def test_the_stale_window_runs_from_the_last_claim_and_closes_the_queue_first(
    tmp_path: Path, clock: ManualClock
) -> None:
    """A probe that last progressed less than the window ago is kept; past it the queue is closed, then jobs re-put."""
    profile = _profile(tmp_path)
    _seen(profile, SEEN)
    placed, values = _plan(profile, range(1, JOBS + 1))
    pool = _pool(tmp_path, profile, clock=clock)
    fragment, handle = _submit(pool, placed, values)
    hand = Hand(pool, placed, values, limit=1)
    hand.serve()
    window = profile.lease.stale_after_s
    clock.advance(window - 1)
    given = list(pool.poll(Handle(id=handle))[0])
    [probe] = _probe_queues(pool)
    queue = pool.queues.queue(probe)
    clock.advance(window - 1)
    given += pool.poll(Handle(id=handle))[0]
    assert not queue.probe_closed() and pool.queues.queue("cpu").work() == 0
    clock.advance(2)
    given += pool.poll(Handle(id=handle))[0]
    assert queue.probe_closed() and pool.queues.queue("cpu").work() > 0
    assert hand.serve() > 0 and all(n == "cpu" or probe_of(n) is None for n, _ in hand.items[1:])
    refs = given + _drive(pool, handle, Hand(pool, placed, values, hardware=ELSEWHERE))
    assert sorted(r.job.element for r in refs) == sorted(j.element for j in fragment.jobs)


@pytest.mark.logic
def test_a_disagreeing_hello_resolves_a_probing_unit_again(tmp_path: Path, clock: ManualClock) -> None:
    """With the prediction already in the index, a hello of another class closes the probe built on it and probes anew.

    The in-flight item's observations are written under the class of the worker that made them, and every job's
    result is still given once.
    """
    profile = _profile(tmp_path)
    _seen(profile, SEEN)
    placed, values = _plan(profile, range(1, JOBS + 1))
    pool = _pool(tmp_path, profile, clock=clock)
    fragment, handle = _submit(pool, placed, values)
    [first] = _probe_queues(pool)
    told = HardwareClass(device="cpu", name="Test CPU", cpus=2)
    hand = Hand(pool, placed, values, hardware=told, limit=1, takes=SEEN)
    hand.serve()
    _hello(pool, placed, values, told)
    given = list(pool.poll(Handle(id=handle))[0])
    assert pool.queues.queue(first).probe_closed()
    [second] = [q for q in _probe_queues(pool) if q != first]
    assert pool.queues.queue(second).probe_hardware() == told
    unit = _unit(pool, placed, values).model_copy(update={"hardware": told})
    assert pool.roots.observations(unit)
    refs = given + _drive(pool, handle, Hand(pool, placed, values, hardware=told))
    assert sorted(r.job.element for r in refs) == sorted(j.element for j in fragment.jobs)


def _slurm_profile(tmp_path: Path) -> ExecutionProfile:
    cpu = {"device": "cpu", "cpus": 4, "mem_bytes": 2**33, "time_s": 3600, "max_workers": 2, "partition": "cpuq"}
    override = {
        "devices": ["cpu"],
        "resource_classes": {"cpu": cpu, "cpu:large": {**cpu, "mem_bytes": 2**34}},
        "pool": {"poll_s": 0.2, "first_poll_s": 0.01, "quantum_items": 10},
        "calibration": {"measures_roots": [str(tmp_path / "measures")], "warmup_requests": 1},
    }
    return ExecutionProfile.load("orcd", override)


@pytest.mark.logic
def test_a_slurm_unit_resolves_at_the_first_hello_and_probes(tmp_path: Path, clock: ManualClock) -> None:
    """On a slurm profile no env is readable here; the unit waits and the first hello's class and env resolve it."""
    profile = _slurm_profile(tmp_path)
    placed, values = _plan(profile, range(1, JOBS + 1))
    pool = _pool(tmp_path, profile, clock=clock)
    fragment, handle = _submit(pool, placed, values)
    assert not _probe_queues(pool) and pool.workers_wanted("cpu", 0) == 1
    _hello(pool, placed, values, SEEN)
    given = list(pool.poll(Handle(id=handle))[0])
    [probe] = _probe_queues(pool)
    assert pool.queues.queue(probe).probe_hardware() == SEEN
    refs = given + _drive(pool, handle, Hand(pool, placed, values))
    assert sorted(r.job.element for r in refs) == sorted(j.element for j in fragment.jobs) and pool.probes


@pytest.mark.logic
def test_a_worker_builds_an_unbuilt_env_before_its_mark_and_the_unit_probes(tmp_path: Path, clock: ManualClock) -> None:
    """An env not built when the plan is read leaves the unit pending; the worker builds it, its mark carries it."""
    from senselab_core.runtime.execution.pool_worker import PoolWorker  # noqa: PLC0415 - one use
    from senselab_core.testing.interpreter import source_tree_python  # noqa: PLC0415 - as `pool_stack_fixture`

    del source_tree_python
    profile = _profile(tmp_path)
    placed, values = _plan(profile, range(1, JOBS + 1))
    resolver = UnitResolver(profile, hardware=HOST)
    built: list[str] = []

    def unbuilt(backend: object, profile: object) -> None:  # noqa: ARG001 - an EnvRead
        return None

    previous = use_packaged(
        Packaged(packaged().catalogue, _counting(built), packaged().registry, packaged().lock_records, env_read=unbuilt)
    )
    try:
        prepare(placed, tmp_path / "run", ResultStore(tmp_path / "store"))
        pool = _pool(
            tmp_path,
            profile,
            clock=clock,
            resolver=UnitResolver(profile, measure_roots(profile), hardware=HOST),
        )
        fragment, handle = _submit(pool, placed, values)
        assert not _probe_queues(pool) and not built
        worker = PoolWorker(
            tmp_path / "run" / RUN_QUEUES,
            ["cpu"],
            profile=profile,
            store=pool.store,
            hardware=lambda _: SEEN,
            device_memory=lambda: None,
        )
        ready = worker.ready()
        assert built and set(ready) == {"cpu"} and ready["cpu"].hardware == SEEN
        [entry] = ready["cpu"].envs.values()
        assert {"env_lock", "wheel_set", "driver", "platform", "numerics", "threads"} <= set(entry)
        mark_ready(pool.run_dir, ready)
        given = list(pool.poll(Handle(id=handle))[0])
        assert len(_probe_queues(pool)) == 1
        refs = given + _drive(pool, handle, Hand(pool, placed, values))
    finally:
        use_packaged(previous)
    assert sorted(r.job.element for r in refs) == sorted(j.element for j in fragment.jobs)
    del resolver


def _counting(built: list[str]) -> Callable[..., EnvHandle]:
    opener = _handle("4" * 64)

    def build(backend: object, profile: object) -> EnvHandle:
        built.append("built")
        return opener(backend, profile)

    return build


@pytest.mark.logic
def test_a_final_cause_takes_the_safe_values(tmp_path: Path, clock: ManualClock) -> None:
    """A hello whose env failed to build, or that names no class, is final: the jobs run unmeasured, none probed."""
    profile = _slurm_profile(tmp_path)
    causes: list[Callable[[str], QueueReady]] = [
        lambda b: QueueReady(SEEN, {b: {"unbuilt": "build-failed"}}),
        lambda b: QueueReady(None, {b: {}}),
    ]
    for index, ready in enumerate(causes):
        placed, values = _plan(profile, range(1, 9))
        pool = _pool(tmp_path, profile, run=f"run{index}", store=f"store{index}", clock=clock)
        _, handle = _submit(pool, placed, values)
        jobs, placement = fragment_jobs(node_fragment(placed, pool.admit(placed, values)), placed, values)
        mark_ready(pool.run_dir, {"cpu": ready(backend_key(placement.backend))})
        pool.poll(Handle(id=handle))
        assert not _probe_queues(pool) and pool.queues.queue("cpu").work() == 8
        del jobs


@pytest.mark.logic
def test_a_ready_mark_carries_no_identifier(tmp_path: Path, clock: ManualClock) -> None:
    """A mark built from an env whose torch index is a mirror URL with userinfo names the build tag and nothing else."""
    from senselab_core.runtime.execution.pool_worker import PoolWorker  # noqa: PLC0415 - one use

    mirror = "https://someone:hunter2@mirror.example.org/whl/cu128"

    def opener(backend: object, profile: object) -> EnvHandle:
        base = host_env(backend, profile)  # type: ignore[arg-type]
        return EnvHandle(
            python=base.python,
            python_version=base.python_version,
            lock_sha=base.lock_sha,
            wheel_set=base.wheel_set,
            torch_index=mirror,
        )

    profile = _profile(tmp_path)
    weights = tmp_path / "private-weights-dir" / "m.bin"
    weights.parent.mkdir()
    weights.write_bytes(b"weights")
    placed, values = _plan(profile, range(1, 5), model=f"local:{weights}")
    [queue_name] = placed_backends(placed)
    assert queue_name.startswith("cpu#")
    previous = use_packaged(
        Packaged(packaged().catalogue, opener, packaged().registry, packaged().lock_records, env_read=opener)
    )
    try:
        prepare(placed, tmp_path / "run", ResultStore(tmp_path / "store"))
        pool = _pool(tmp_path, profile, clock=clock)
        _submit(pool, placed, values)
        worker = PoolWorker(
            tmp_path / "run" / RUN_QUEUES,
            [queue_name],
            profile=profile,
            store=pool.store,
            hardware=lambda _: SEEN,
            device_memory=lambda: None,
        )
        mark_ready(pool.run_dir, worker.ready())
    finally:
        use_packaged(previous)
    [path] = [p for p in (pool.run_dir / "workers").iterdir() if not p.name.startswith(".")]
    text = path.read_text()
    assert '"torch_build":"cu128"' in text
    for leaked in (
        "someone",
        "hunter2",
        "mirror.example",
        "private-weights-dir",
        "m.bin",
        os.uname().nodename,
        str(Path.home()),
    ):
        assert leaked not in text
