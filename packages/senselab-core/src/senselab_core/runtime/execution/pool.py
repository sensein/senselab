"""`Pool`: the `Executor` that puts fragments on a run's queues and reads what pool workers made of them.

A fragment goes to the queue of its placement's resource class under `<run dir>/queues/`, named `<class>#<model
identity>` when its node has a resident model, as items of a quantum of jobs each, ending on request boundaries when a
request fits in one (`split`), in job order; a `Substrate` has as many workers serve that queue as its waiting work
wants (`scale`, each round). Quantum and worker count are sized (`size`) from the measure of the fragment's unit (its
task, backend, model and class): a resource-table row's, else a calibration round's. A unit without one first puts
`sizing.calibration_items` of its jobs as one-job items for one worker and holds its other jobs; once those items have
outcomes, their samples give the measure (`calibrated`), which is written back as a resource-table row
(`MeasureSource.put`), and the held jobs are put. A worker records each job's `ResultRef` in the item's outcome, with
what it measured of the item (`report`). The `Pool` yields a result only once the result store holds its outputs, and a
failure only as `dead/<key digest>.json` holds it, so a restarted coordinator reads the same state from disk. Each fault
class has its effect here, with the budgets of the profile's `faults` section:

- **content**, not a gap: charged; the job is queued again until `content_attempts` charges, then it is dead;
- **resource**, not a gap: the job is queued again on the next larger memory class of its device, uncharged, at most
  `resource_replacements` times; then, or with no larger class, it is dead;
- **infrastructure**, or any gap: queued again uncharged (a gap is logged as a classification gap); at
  `infrastructure_replacements` failures of one env in this epoch the env is reported once as broken (`BrokenEnv`) and
  held on each queue it is seen on (`Queue.hold`), so its jobs wait for a later epoch and none is failed.

A retry is a one-job fragment of the next `attempt`, logged `failed` then `queued` by this coordinator; a job held
with its env is logged `queued` again, so it reads as waiting, never as failed.
"""

import contextlib
import itertools
import json
import logging
import os
import threading
import time
import uuid
import weakref
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from senselab_core.kernel.identity import InputIdentity, ReusePolicy
from senselab_core.kernel.placement import NodePlacement
from senselab_core.runtime.call import RunConfig
from senselab_core.runtime.execution.epoch import Epoch, StaleCoordinator
from senselab_core.runtime.execution.events import BrokenEnv, EventLog, read_events
from senselab_core.runtime.execution.executor import (
    Admission,
    FaultRecord,
    Handle,
    JobRef,
    PlanFragment,
    QueueHeld,
    ResultRef,
    Status,
    WorkersBroken,
)
from senselab_core.runtime.execution.in_process import fragment_keys, node_request_size, result_ref
from senselab_core.runtime.execution.job import JobOutcome
from senselab_core.runtime.execution.queue import RUN_QUEUES, Claim, QueueSet, item_digest
from senselab_core.runtime.execution.substrates import (
    WORKERS_DIR,
    BrokenClass,
    Fleet,
    Local,
    Slurm,
    Substrate,
    SubstrateError,
    class_of,
    scale,
)
from senselab_core.runtime.planner import Plan, check_storage
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.resources import Measured, MeasureSource, ResourceMeasure, Unit
from senselab_core.runtime.sizing import (
    Grant,
    Measure,
    Measures,
    Sample,
    Sizing,
    calibrated,
    code_version,
    measure_of,
    size,
)
from senselab_core.runtime.store import ResultStore

if TYPE_CHECKING:
    from senselab_core.runtime.execution.runner import RunOutcome

logger = logging.getLogger("senselab")


class ScalerFailed(RuntimeError):
    """The pool's scaler thread ended on an error that is not a substrate failure; the cause is chained."""


DEAD_DIR = "dead"
"""The directory of a run's dead jobs, one `<key digest>.json` each."""


def report(claim: Claim, digest: str, outcomes: Sequence[JobOutcome], sample: Sample | None = None) -> bool:
    """Commit the claimed item `digest` with each job's `ResultRef` and the `sample` measured; return whether it did."""
    extra: dict[str, object] = {"jobs": [result_ref(outcome).model_dump(mode="json") for outcome in outcomes]}
    if sample is not None:
        extra["measure"] = sample.model_dump(mode="json")
    return digest in claim.commit([digest], extra)


def larger_class(profile: ExecutionProfile, name: str) -> str | None:
    """Return the resource class of `profile` with the least memory above class `name`'s, on its device and GPU type.

    A class whose `cpus` exceed the substrate's `qos_cpu_cap` is skipped: no worker of it could start. None when
    `name` is not a class of the profile, states no memory, or no startable class of its kind has more.
    """
    cap = profile.substrate.qos_cpu_cap
    current = profile.resource_classes.get(name)
    if current is None or current.mem_bytes is None:
        return None
    larger = sorted(
        (c.mem_bytes, n)
        for n, c in profile.resource_classes.items()
        if (c.device, c.gpu_type) == (current.device, current.gpu_type)
        and c.mem_bytes is not None
        and c.mem_bytes > current.mem_bytes
        and (cap is None or c.cpus is None or c.cpus <= cap)
    )
    return larger[0][1] if larger else None


_PER_JOB = frozenset({"jobs", "inputs", "upstream", "gathered"})


@dataclass(frozen=True)
class _Dumped:
    """A fragment dumped once: the fields its parts share, and its jobs, inputs, upstream and gathers to slice from."""

    shared: dict[str, object]
    jobs: list[object]
    inputs: list[dict[str, str]]
    upstream: dict[str, object]
    gathered: dict[str, list[str]]

    @classmethod
    def of(cls, fragment: PlanFragment) -> "_Dumped":
        """Dump `fragment` once."""
        return cls(
            fragment.model_dump(mode="json", exclude=set(_PER_JOB)),
            [j.model_dump(mode="json") for j in fragment.jobs],
            [dict(i) for i in fragment.inputs],
            {i: r.model_dump(mode="json") for i, r in fragment.upstream.items()},
            {g: list(e) for g, e in fragment.gathered.items()},
        )

    def part(self, indices: Sequence[int], **update: object) -> PlanFragment:
        """Return the fragment of jobs `indices`, in that order, with the inputs they read and `update`."""
        named = {i for index in indices for i in self.inputs[index].values()}
        gathered = {g: self.gathered[g] for g in named if g in self.gathered}
        read = {i for i in named if i.startswith("output:")} | {e for es in gathered.values() for e in es}
        return PlanFragment.model_validate(
            {
                **self.shared,
                "jobs": [self.jobs[i] for i in indices],
                "inputs": [self.inputs[i] for i in indices],
                "upstream": {i: self.upstream[i] for i in sorted(read) if i in self.upstream},
                "gathered": gathered,
                **update,
            }
        )


def _jobs_of(fragment: PlanFragment, indices: Sequence[int], **update: object) -> PlanFragment:
    """Return the fragment of jobs `indices` of `fragment`, in that order, with the inputs they read and `update`."""
    return _Dumped.of(fragment).part(indices, **update)


def split(fragment: PlanFragment, quantum: int, request: int = 1) -> list[PlanFragment]:
    """Return `fragment` as items of at most `quantum` jobs each, in job order; `[fragment]` when it fits in one.

    The fragment is dumped once and each item is built from its own slice, so the work is linear in the jobs.

    `request` is the jobs per request of the fragment's backend (`node_request_size`). When it is at most `quantum`,
    every item but the last holds the most whole requests that fit, so no request is cut; a larger request is cut at
    `quantum` jobs.

    Raises:
        ValueError: `quantum` or `request` is below 1.
    """
    if quantum < 1 or request < 1:
        raise ValueError(f"an item and a request hold at least one job, got {quantum} and {request}")
    count = len(fragment.jobs)
    if count <= quantum:
        return [fragment]
    size = quantum - quantum % request if request <= quantum else quantum
    dumped = _Dumped.of(fragment)
    return [dumped.part(range(start, min(start + size, count))) for start in range(0, count, size)]


@dataclass
class _Flight:
    """One item on a queue, by its digest."""

    queue: str
    digest: str
    fragment: PlanFragment


@dataclass
class _Followed:
    """A submitted fragment: its items in flight, its parts waiting on a calibration, each job's charges and moves."""

    flights: list[_Flight]
    waiting: list[tuple[Unit, PlanFragment]] = field(default_factory=list)
    charged: Counter[tuple[int, ...]] = field(default_factory=Counter)
    moved: Counter[tuple[int, ...]] = field(default_factory=Counter)
    held: list[QueueHeld] = field(default_factory=list)
    seen: bool = False
    done: bool = False
    cancelled: bool = False


class Pool:
    """An `Executor` over the queues of the run in `run_dir`, its jobs run by pool workers a `substrate` starts.

    `submit` puts a fragment on its queue unless a newer epoch than `epoch` was taken (`StaleCoordinator`); `poll`
    reads its items' outcomes once, never waiting, and applies each fault's effect. One scaler thread, started by the
    first submit and stopped by `close`, sizes the substrate's workers once per `pool.poll_s` while a fragment is
    followed, and at once after each submit or retry; nothing else scales. An admission holds the plan only: workers
    bind graph inputs from the run's `inputs.json`. `events` is the log of `epoch`'s attempt; an env's infrastructure
    failures are counted from that epoch's events on disk. `measures` gives each unit's measure and keeps what a
    calibration round measured (`Measures` by default).

    Attributes:
        broken: Each env this pool reported broken, once.
        gaps: Each job given a fault with `gap` set, with its fault.
        calibrated: Each unit this pool calibrated, with the digests of its rounds' items and the jobs they ran.
    """

    logs_claims: ClassVar[bool] = True
    splits_fragments: ClassVar[bool] = True

    def __init__(  # noqa: PLR0913 - the run, then keyword-only parts
        self,
        run_dir: Path,
        *,
        queues: QueueSet,
        store: ResultStore,
        events: EventLog,
        profile: ExecutionProfile,
        substrate: Substrate,
        epoch: int,
        clock: Callable[[], float] = time.monotonic,
        measures: MeasureSource | None = None,
    ) -> None:
        """Coordinate the run in `run_dir` as epoch `epoch`; `clock` dates each epoch read for its reuse.

        Raises:
            ValueError: `events` is not a log of `epoch`.
        """
        self.run_dir = Path(run_dir)
        self.queues, self.store, self.profile, self.substrate = queues, store, profile, substrate
        self._fleet = Fleet(
            substrate,
            workers_dir=self.run_dir / WORKERS_DIR,
            budget=profile.faults.infrastructure_replacements,
            first_s=profile.substrate.start_backoff_first_s,
            max_s=profile.substrate.start_backoff_max_s,
        )
        self.broken_workers: dict[str, BrokenClass] = {}
        self._surplus: dict[str, float] = {}
        self._relogged: set[str] = set()
        self._scaler: threading.Thread | None = None
        self._scale_wake = threading.Event()
        self._closing = False
        self._scaling = False
        self._scale_error: Exception | None = None
        self._fenced: tuple[float, int] | None = None
        self._clock = clock
        self.bind_epoch(epoch, events)
        self.broken: dict[str, BrokenEnv] = {}
        self.gaps: list[tuple[JobRef, FaultRecord]] = []
        self._epochs = Epoch(self.run_dir)
        self._plans: dict[str, tuple[Plan, Mapping[InputIdentity, object]]] = {}
        self._admitted: Counter[str] = Counter()
        self._followed: dict[str, _Followed] = {}
        self._ids = itertools.count()
        self._lock = threading.Lock()
        self.measures = measures if measures is not None else Measures()
        self._sized: dict[Unit, Measure] = {}
        self._queue_units: dict[str, set[Unit]] = {}
        self._calibrating: dict[Unit, tuple[str, list[str]]] = {}
        self._first_round: dict[Unit, list[Sample]] = {}
        self.calibrated: dict[Unit, tuple[frozenset[str], int]] = {}
        self._round_digests: dict[Unit, set[str]] = {}

    def bind_epoch(self, epoch: int, events: EventLog) -> None:
        """Coordinate as `epoch` from now on, logging to `events`, a log of that epoch's attempt.

        Raises:
            ValueError: `events` is not a log of `epoch`.
        """
        if events.epoch != epoch:
            raise ValueError(f"the pool coordinates epoch {epoch}; its event log is of epoch {events.epoch}")
        self.epoch, self.events = epoch, events
        if isinstance(self.substrate, Slurm):
            self.substrate.attempt, self.substrate.epoch = events.attempt, epoch
        self._fleet.reset()
        self.broken_workers.clear()
        self._relogged.clear()

    def admit(self, placed: Plan, values: Mapping[InputIdentity, object]) -> Admission:
        """Hold `placed` and its graph-input `values` until released as often as admitted; its id is `run-<plan hash>`.

        Raises:
            ValueError: the plan was placed under another execution profile than this pool's.
        """
        if placed.placement.profile_digest != self.profile.canonical_digest():
            raise ValueError(f"plan {placed.hash} was placed under another execution profile than this pool's")
        admission = Admission(id=f"run-{placed.hash}", plan_hash=placed.hash)
        with self._lock:
            self._plans[admission.id] = (placed, values)
            self._admitted[admission.id] += 1
        return admission

    def release(self, admission: Admission) -> None:
        """Let go of one admission of `admission`'s plan."""
        with self._lock:
            self._admitted[admission.id] -= 1
            if self._admitted[admission.id] <= 0:
                self._plans.pop(admission.id, None)
                del self._admitted[admission.id]

    def _fence(self, *, recent: bool = False) -> None:
        """Raise `StaleCoordinator` once a newer epoch is taken; `recent` reuses a read younger than `first_poll_s`."""
        now, fenced = self._clock(), self._fenced
        if recent and fenced is not None and now - fenced[0] < self.profile.pool.first_poll_s:
            current = fenced[1]
        else:
            current = self._epochs.current()
            self._fenced = (now, current)
        if current > self.epoch:
            raise StaleCoordinator(self.epoch, current)

    def _put(
        self, fragment: PlanFragment, *, retry: bool = False
    ) -> tuple[list[_Flight], list[tuple[Unit, PlanFragment]]]:
        """Put `fragment` on its queue as its items (`split`); return them and the part held for a calibration.

        A unit with a measure is split by its sized quantum, its placement's batch the sized one. One without starts
        a calibration round with the first `sizing.calibration_items` jobs, or, while its round runs, holds the whole
        fragment; a `retry` is put at once, one job per item. The caller wakes the scaler (`_wake_scaler`) once it
        follows the items.

        Raises:
            StaleCoordinator: a newer epoch was taken.
            QueueHeld: the fragment's env is held on its queue by this epoch or a newer one.
            ValueError: the fragment's admission is not held by this pool.
        """
        self._fence()
        name, env = self._queue_name(fragment), fragment.placement.env
        queue = self.queues.queue(name)
        queue.unhold(before=self.epoch)
        if queue.held(env) is not None:
            failures = self._env_failures(env)
            reason = failures[-1].message if failures else "held by an earlier coordinator of this epoch or a newer one"
            raise QueueHeld(name, env, len(failures), reason)
        unit = self._unit(fragment)
        measure = self._measure(unit, name)
        quantum = 1
        if measure is not None:
            sized = self._sizing(name, measure)
            placement = fragment.placement.model_copy(update={"batch_size": sized.batch_size})
            fragment, quantum = fragment.model_copy(update={"placement": placement}), sized.quantum_items
        request = node_request_size(self._admitted_plan(fragment)[0], fragment.node, fragment.placement)
        part: PlanFragment | None = fragment
        waiting: list[tuple[Unit, PlanFragment]] = []
        if measure is None and not retry:
            part, waiting = self._calibration_part(unit, name, fragment)
        items = [] if part is None else split(part, quantum, request)
        reserved = measure is None and not retry and bool(items)
        try:
            queue.put_all(items)
        except BaseException:
            if reserved:
                with self._lock:
                    self._calibrating.pop(unit, None)
            raise
        flights = [_Flight(name, item_digest(item), item) for item in items]
        if reserved:
            with self._lock:
                self._calibrating[unit] = (name, [f.digest for f in flights])
        return flights, waiting

    def _unit(self, fragment: PlanFragment) -> Unit:
        """Return what `fragment`'s jobs are measured as: its node's task, backend, model and resource class."""
        placed = self._admitted_plan(fragment)[0]
        model = placed.logical.models.get(fragment.node)
        task = next(n.task for n in placed.logical.graph.nodes if n.id == fragment.node)
        ident = (None, None) if model is None else (model.ref.id, model.identity)
        return Unit(task, fragment.placement.backend, ident[0], ident[1], fragment.placement.resource_class)

    def _measure(self, unit: Unit, queue: str) -> Measure | None:
        """Return `unit`'s measure, read from `measures` once; note the unit as one of `queue`'s."""
        with self._lock:
            self._queue_units.setdefault(queue, set()).add(unit)
            if unit in self._sized:
                return self._sized[unit]
        found = self.measures.get(unit)
        if found is None:
            return None
        with self._lock:
            return self._sized.setdefault(unit, self._per_item(found, queue))

    def _per_item(self, found: ResourceMeasure, queue: str) -> Measure:
        """Return `found` per item, by device memory on a `cuda` class."""
        return measure_of(found, on_device=self.profile.resource_classes[class_of(queue)].device == "cuda")

    def _grant(self, queue: str, *, on_device: bool = False) -> Grant:
        """Return the grant of `queue`'s resource class; a local substrate's workers share its visible devices."""
        devices = len(self.substrate.devices) if isinstance(self.substrate, Local) else None
        resource = self.profile.resource_classes[class_of(queue)]
        return Grant.of(resource, self.profile.pool, on_device=on_device, devices=devices)

    def _sizing(self, queue: str, measure: Measure, work: int = 0) -> Sizing:
        """Return the sizing of `measure` on `queue`'s resource class for `work` waiting jobs."""
        return size(measure, self._grant(queue, on_device=measure.on_device), self.profile.sizing, work=work)

    def _calibration_part(
        self, unit: Unit, queue: str, fragment: PlanFragment
    ) -> tuple[PlanFragment | None, list[tuple[Unit, PlanFragment]]]:
        """Split `fragment` into the jobs that start `unit`'s calibration round and the part held until it ends.

        None is put while a round of the unit runs; otherwise the round is reserved here and its digests noted once put.
        """
        with self._lock:
            running = unit in self._calibrating
            if not running:
                self._calibrating[unit] = (queue, [])
        count = len(fragment.jobs)
        k = 0 if running else min(self.profile.sizing.calibration_items, count)
        dumped = _Dumped.of(fragment)
        first = dumped.part(range(k)) if k else None
        rest = [(unit, dumped.part(range(k, count)))] if k < count else []
        return first, rest

    def _batch_round(self, unit: Unit, queue: str, found: ResourceMeasure, samples: list[Sample]) -> bool:
        """Start `unit`'s second round, batched and one-job items timed side by side, when its sized batch is above one.

        The round is `sizing.calibration_items - 1` pairs, each an item of the batch sent as one request then a
        one-job item, all on the worker the first round warmed, so both per-item times have a median taken past the
        first requests after the load. The batch is the memory batch, within what lets every pair fit in the unit's
        first held part; the rest of that part waits on. Return whether a round was started.
        """
        with self._lock:
            held = [(f, p) for f in self._followed.values() for u, p in f.waiting if u == unit]
        if not held:
            return False
        followed, part = held[0]
        pairs = self.profile.sizing.calibration_items - 1
        fits = len(part.jobs) // pairs - 1
        batch = min(self._sizing(queue, self._per_item(found, queue)).memory_batch, max(fits, 1))
        placement = part.placement.model_copy(update={"batch_size": batch})
        request = node_request_size(self._admitted_plan(part)[0], part.node, placement)
        if request < 2:  # noqa: PLR2004 - a batch is two items or more
            return False
        dumped = _Dumped.of(part)
        batched, single = (
            placement.model_dump(mode="json"),
            placement.model_copy(update={"batch_size": 1}).model_dump(mode="json"),
        )
        items = []
        for i in range(pairs):
            start = i * (request + 1)
            items += [
                dumped.part(range(start, start + request), placement=batched),
                dumped.part([start + request], placement=single),
            ]
        used = pairs * (request + 1)
        rest = [(unit, dumped.part(range(used, len(part.jobs))))] if used < len(part.jobs) else []
        self.queues.queue(queue).put_all(items)
        with self._lock:
            followed.waiting = [(u, p) for u, p in followed.waiting if p is not part] + rest
            followed.flights.extend(_Flight(queue, item_digest(i), i) for i in items)
            self._first_round[unit] = samples
            self._calibrating[unit] = (queue, [item_digest(i) for i in items])
        self._wake_scaler()
        return True

    def workers_wanted(self, queue: str, work: int) -> int:
        """Return the workers `queue`'s `work` jobs want: by its slowest measured unit, else one, to calibrate.

        A queue holding a measured unit beside a calibrating one is sized by the measured one, so the calibrating
        unit's round may be run by several workers, each giving a cold sample.
        """
        with self._lock:
            measures = [self._sized[u] for u in self._queue_units.get(queue, ()) if u in self._sized]
        if not measures:
            return min(self.profile.sizing.calibration_workers, work, self._grant(queue).worker_bound(None))
        return self._sizing(queue, max(measures, key=lambda m: m.s_per_item), work).workers

    def _calibrate(self) -> None:
        """End each calibration round whose items all have outcomes: record its measure and put its unit's held parts.

        A first round sends items one per request and gives the load, the claim and the memory. When its measure
        sizes a batch above one, a second round times batched and one-job items side by side on the warmed worker
        (`_batch_round`), and the per-item seconds alone and batched are taken from that round alone, so `size` keeps
        the faster of two like regimes. A unit of a model backend whose `cuda` samples reported no
        device figures is logged: its device memory stays unmeasured. A round whose items measured nothing leaves its
        unit unmeasured, so its held parts start another. A round with no warm one-job item measures
        nothing, since its load cannot be told from its items. A part refused when put (`_put`) waits again, with
        those after it.
        """
        with self._lock:
            rounds = list(self._calibrating.items())
        for unit, (queue, digests) in rounds:
            records = [self.queues.queue(queue).outcome(d) for d in digests]
            if not digests or any(r is None for r in records):
                continue
            with self._lock:
                self._round_digests.setdefault(unit, set()).update(digests)
            first = self._first_round.pop(unit, None)
            this = [Sample.model_validate(r["measure"]) for r in records if r is not None and "measure" in r]
            samples = (first or []) + this
            found = _round_measure(first, this)
            if found is not None and found.batched is None and self._batch_round(unit, queue, found, samples):
                continue
            _note_unmeasured_device(unit, queue, found, samples)
            if found is not None:
                jobs = tuple(sorted({s.job for s in samples}))
                self.measures.put(unit, found, self._measured(jobs))
            with self._lock:
                del self._calibrating[unit]
                if found is not None:
                    self._sized[unit] = self._per_item(found, queue)
                    self.calibrated[unit] = (frozenset(self._round_digests[unit]), sum(s.jobs for s in samples))
                held = [(f, p) for f in self._followed.values() for u, p in f.waiting if u == unit]
                for followed in self._followed.values():
                    followed.waiting = [(u, p) for u, p in followed.waiting if u != unit]
            for index, (followed, part) in enumerate(held):
                try:
                    flights, waiting = self._put(part)
                except BaseException:
                    with self._lock:
                        for owner, left in held[index:]:
                            owner.waiting.append((unit, left))
                    raise
                followed.flights.extend(flights)
                followed.waiting.extend(waiting)
            self._wake_scaler()

    def _queue_name(self, fragment: PlanFragment) -> str:
        """Return the queue of `fragment`: its class, with `#<model identity>` for its node's resident model."""
        admitted = self._plans.get(fragment.admission)
        model = admitted[0].logical.models.get(fragment.node) if admitted is not None else None
        name = fragment.placement.resource_class
        return name if model is None else f"{name}#{model.identity}"

    def _wake_scaler(self) -> None:
        """Have the scaler thread size the workers now, starting it the first time."""
        with self._lock:
            if self._scaler is None and not self._closing:
                rounds = (weakref.ref(self), self._scale_wake, self.profile.pool.poll_s)
                self._scaler = threading.Thread(target=_scale_rounds, args=rounds, name="senselab-scaler", daemon=True)
                self._scaler.start()
        self._scale_wake.set()

    def _open(self) -> bool:
        """Return whether any followed fragment is neither ended nor cancelled."""
        with self._lock:
            return any(not (f.done or f.cancelled) for f in self._followed.values())

    def _scale(self) -> None:
        """Have the substrate size its workers to the queues; only the scaler thread calls it, never re-entered.

        A `SubstrateError` is counted (`Fleet.fail_substrate`) and the substrate is not used again until its backoff
        has passed; past `faults.infrastructure_replacements` failures in a row it is broken, the run is held
        (`WorkersBroken`) and nothing is scaled again this epoch. Any other error ends the scaler and is raised by
        `poll`.
        """
        assert threading.current_thread() is self._scaler and not self._scaling, "only the scaler thread scales"
        if not self._fleet.substrate_due():
            return
        self._scaling = True
        try:
            scale(self.queues, self._fleet, self.profile, workers=self.workers_wanted, surplus_since=self._surplus)
        except SubstrateError as err:
            logger.warning("the substrate failed: %s", err)
            self._fleet.fail_substrate(err)
        else:
            self._fleet.substrate_ok()
        finally:
            self._scaling = False

    def _workers_broken(self, flight: _Flight) -> WorkersBroken | None:
        """Return the hold of `flight`'s queue by a broken class of workers, noting the break once."""
        found = self._fleet.broken_for(flight.queue)
        if found is None:
            return None
        if found.resource_class not in self.broken_workers:
            self.broken_workers[found.resource_class] = found
        if (flight.digest) not in self._relogged:
            self._relogged.add(flight.digest)
            keys = self._keys(flight.fragment)
            for job, key in zip(flight.fragment.jobs, keys, strict=True):
                self.events.record("queued", job.node, job=job, key=key, env=flight.fragment.placement.env)
        return WorkersBroken(flight.queue, found.resource_class, found.failures, found.reason)

    def submit(self, fragment: PlanFragment) -> Handle:
        """Put `fragment` on the queue of its resource class, as quantum-sized items, and return its handle.

        Raises:
            StaleCoordinator: a newer epoch than this coordinator's was taken; nothing is put.
            QueueHeld: the fragment's env is held on its queue; nothing is put.
            ValueError: the fragment's admission is not held by this pool; nothing is put.
        """
        flights, waiting = self._put(fragment)
        handle = Handle(id=f"pool-{self.epoch}-{next(self._ids)}")
        with self._lock:
            self._followed[handle.id] = _Followed(flights, waiting)
        self._wake_scaler()
        return handle

    def status(self, handle: Handle) -> Status:
        """Return where the fragment of `handle` is: running once any item of it is claimed or has an outcome.

        Raises:
            KeyError: the handle is not this pool's.
        """
        followed = self._followed[handle.id]
        if followed.cancelled or followed.done:
            return Status.CANCELLED if followed.cancelled else Status.DONE
        if followed.seen or any(self.queues.queue(f.queue).claimed(f.digest) for f in followed.flights):
            return Status.RUNNING
        return Status.PENDING

    def cancel(self, handle: Handle) -> None:
        """Stop following `handle`: its items stay queued, for a later attempt to claim or read."""
        self._followed[handle.id].cancelled = True

    def poll(self, handle: Handle) -> tuple[list[ResultRef], bool]:
        """Read the outcomes of `handle`'s items once, applying each fault's effect, and return the results landed.

        Returns the results not yet given, items read in one poll in job order, and whether the fragment is ended:
        every result given, or the handle cancelled. A hold is raised once the results landed with it are given. The
        epoch is read afresh once outcomes have landed, before any effect is applied.

        Raises:
            KeyError: the handle is not this pool's.
            StaleCoordinator: a newer epoch was taken.
            QueueHeld: a job's env is held, or its class's workers are broken; the jobs not given wait in their queues.
            ScalerFailed: the scaler thread ended on an error, chained as the cause.
        """
        if self._scale_error is not None:
            raise ScalerFailed(f"the pool's scaler failed: {self._scale_error!r}") from self._scale_error
        followed = self._followed[handle.id]
        if followed.cancelled or followed.done:
            return [], True
        self._fence(recent=True)
        self._calibrate()
        given: list[ResultRef] = []
        landed = [(f, r) for f in followed.flights if (r := self.queues.queue(f.queue).outcome(f.digest))]
        if landed:
            self._fence()
        for flight, record in landed:
            followed.flights.remove(flight)
            followed.seen = True
            given.extend(self._settle(followed, flight, record))
        held = followed.held + [self._held(f) for f in followed.flights if f.fragment.placement.env in self.broken]
        stopped = [b for f in followed.flights if (b := self._workers_broken(f)) is not None]
        if (held or stopped) and not given:
            raise (held or stopped)[0]
        followed.done = not followed.flights and not followed.waiting and not held
        return given, followed.done

    def close(self) -> None:
        """Stop the scaler thread, then the workers the substrate started."""
        with self._lock:
            self._closing = True
            scaler = self._scaler
        self._scale_wake.set()
        if scaler is not None:
            scaler.join()
        self._fleet.close()

    def _settle(self, followed: _Followed, flight: _Flight, record: Mapping[str, object]) -> Iterator[ResultRef]:
        """Yield the results of `flight`'s jobs from its outcome `record`, applying each failure's effect.

        A reported result is taken only when it names the job's own key and the store holds its outputs. A resource
        fault doubles the per-item memory of the unit's measure (`_grow`).
        """
        fragment = flight.fragment
        jobs = record.get("jobs") if record.get("outcome") == "done" else None
        given = {r.job: r for r in (ResultRef.model_validate(j) for j in jobs)} if isinstance(jobs, list) else {}
        dead = FaultRecord.model_validate(record["fault"]) if record.get("outcome") == "dead" else None
        keys, grown = self._keys(fragment), False
        for index, (job, key) in enumerate(zip(fragment.jobs, keys, strict=True)):
            ref = given.get(job)
            if ref is not None and ref.key_digest != key:
                fault = FaultRecord(
                    fault="content",
                    type="WrongKey",
                    message=f"the outcome gives {job} the key {ref.key_digest}, not its own {key}",
                    gap=False,
                    charged=True,
                )
            elif ref is not None and ref.failure is None and ref.output_identity is not None:
                if self.store.outputs_of(key, ref.output_identity) is not None:
                    yield ref
                    continue
                fault = _gap("MissingOutput", f"the outputs of {job} are not in the result store")
            elif ref is not None and ref.failure is not None:
                fault = ref.failure
            else:
                fault = dead or _gap("MissingResult", f"the outcome of the item gives no result for {job}")
            if fault.fault == "resource" and not grown:
                grown = True
                self._grow(fragment, record)
            yield from self._effect(followed, flight, index, key, fault)

    def _grow(self, fragment: PlanFragment, record: Mapping[str, object]) -> None:
        """Double the per-item memory of `fragment`'s unit's measure after a resource fault, so its batch halves.

        Both memory pairs the measure holds are grown; the grown measure is written back and sizes every later put.
        """
        unit, worker = self._unit(fragment), record.get("worker")
        found = self.measures.get(unit)
        if found is None or not isinstance(worker, str):
            return
        update: dict[str, int] = {}
        for resident, peak, key in (
            (found.resident_bytes, found.peak_memory_bytes, "peak_memory_bytes"),
            (found.device_resident_bytes, found.device_peak_bytes, "device_peak_bytes"),
        ):
            if resident is not None and peak is not None:
                update[key] = resident + 2 * max(1, peak - resident)
        grown = ResourceMeasure.model_validate({**found.model_dump(exclude={"measured"}), **update})
        self.measures.put(unit, grown, self._measured((worker,)))
        with self._lock:
            self._sized[unit] = self._per_item(grown, fragment.placement.resource_class)

    def _measured(self, jobs: tuple[str, ...]) -> Measured:
        """Return the derivation of a measure this pool writes: the code version and run, and the worker jobs."""
        return Measured(sha=None, run=f"senselab-core {code_version()}, run {self.run_dir.name}", jobs=jobs)

    def _admitted_plan(self, fragment: PlanFragment) -> tuple[Plan, Mapping[InputIdentity, object]]:
        admitted = self._plans.get(fragment.admission)
        if admitted is None:
            raise ValueError(f"admission {fragment.admission} is not held by this pool")
        return admitted

    def _keys(self, fragment: PlanFragment) -> list[str]:
        return fragment_keys(fragment, *self._admitted_plan(fragment))

    def _effect(  # noqa: PLR0913 - the followed fragment, the item and the job's failure
        self, followed: _Followed, flight: _Flight, index: int, key: str, fault: FaultRecord
    ) -> Iterator[ResultRef]:
        """Apply `fault`'s effect to job `index` of `flight`: retry it, hold its env, or yield it dead."""
        fragment, budgets = flight.fragment, self.profile.faults
        job, placement = fragment.jobs[index], fragment.placement
        if fault.gap or fault.fault == "infrastructure":
            if fault.gap:
                self.gaps.append((job, fault))
                logger.warning("classification gap: %s of %s, retried uncharged", fault.type, job)
            if self._infrastructure(flight, job, key, fault):
                self._again(followed, flight, index)
            else:
                self.events.record("queued", job.node, job=job, key=key, env=placement.env)
                followed.held.append(self._held(flight))
            return
        if fault.fault == "resource":
            followed.moved[job.element] += 1
            larger = larger_class(self.profile, placement.resource_class)
            if larger is not None and followed.moved[job.element] <= budgets.resource_replacements:
                self._failed(job, key, fault, placement.env)
                grant = {"resource_class": larger, "partition": self.profile.resource_classes[larger].partition}
                self._again(followed, flight, index, placement.model_copy(update=grant))
                return
        else:
            followed.charged[job.element] += 1
            if followed.charged[job.element] < budgets.content_attempts:
                self._failed(job, key, fault, placement.env)
                self._again(followed, flight, index)
                return
        yield self._dead(job, key, fault)

    def _env_failures(self, env: str) -> list[FaultRecord]:
        """Return the infrastructure and gap failures of `env` this epoch's events on disk record, in order."""
        return [
            e.failure
            for e in read_events(self.run_dir)
            if e.kind == "failed"
            and e.epoch == self.epoch
            and e.env == env
            and e.failure is not None
            and (e.failure.gap or e.failure.fault == "infrastructure")
        ]

    def _held(self, flight: _Flight) -> QueueHeld:
        env = flight.fragment.placement.env
        broken = self.broken[env]
        return QueueHeld(flight.queue, env, broken.failures, broken.fault.message)

    def _infrastructure(self, flight: _Flight, job: JobRef, key: str, fault: FaultRecord) -> bool:
        """Count an infrastructure fault of `flight`'s env this epoch; return False once the env is broken and held.

        A broken env is held on `flight`'s queue (each queue it is seen on); `poll` has fenced afresh before.
        """
        env, budget = flight.fragment.placement.env, self.profile.faults.infrastructure_replacements
        with self._lock:
            failures = len(self._env_failures(env))
            if env not in self.broken and failures < budget:
                self._failed(job, key, fault, env)
                failures += 1
                if failures < budget:
                    return True
            reported = env in self.broken
            if not reported:
                self.broken[env] = BrokenEnv(env=env, fault=fault, failures=failures)
        self.queues.queue(flight.queue).hold(self.epoch, env)
        if not reported:
            logger.error(
                "env %s is broken after %d infrastructure faults; it is held: %s", env, failures, fault.message
            )
        return False

    def _failed(self, job: JobRef, key: str, fault: FaultRecord, env: str) -> None:
        self.events.record("failed", job.node, job=job, key=key, failure=fault, env=env)

    def _again(self, followed: _Followed, flight: _Flight, index: int, placement: NodePlacement | None = None) -> None:
        """Queue job `index` of `flight` again alone, as the next attempt, on `placement` (its own when None)."""
        update: dict[str, object] = {"attempt": flight.fragment.attempt + 1}
        if placement is not None:
            update["placement"] = placement.model_dump(mode="json")
        retry = _jobs_of(flight.fragment, [index], **update)
        followed.flights.extend(self._put(retry, retry=True)[0])
        self._wake_scaler()
        job, placed = retry.jobs[0], retry.placement
        key = self._keys(retry)[0]
        self.events.record(
            "queued", job.node, job=job, key=key, placement_digest=placed.canonical_digest(), env=placed.env
        )

    def _dead(self, job: JobRef, key: str, fault: FaultRecord) -> ResultRef:
        """Write `dead/<key>.json` unless it is there, and return the failure it holds."""
        directory = self.run_dir / DEAD_DIR
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{key}.json"
        staging = directory / f".{key}.{uuid.uuid4().hex}"
        record = {"job": job.model_dump(mode="json"), "key_digest": key, "failure": fault.model_dump(mode="json")}
        staging.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
        try:
            with contextlib.suppress(FileExistsError):
                os.link(staging, path)
        finally:
            staging.unlink()
        held = json.loads(path.read_text(encoding="utf-8"))
        return ResultRef(job=job, key_digest=key, failure=FaultRecord.model_validate(held["failure"]))


def _scale_rounds(ref: "weakref.ref[Pool]", wake: threading.Event, poll_s: float) -> None:
    """Scale the pool `ref` names on each wake, and once per `poll_s` while it follows a fragment.

    While the substrate backs off, the next round comes when its wait ends, if sooner. Ends once the pool is closed or
    collected, or on an error other than a `SubstrateError`, which the pool keeps for `poll` to raise; the pool is held
    only during a round.
    """
    woken = True
    while (pool := ref()) is not None and not pool._closing:  # noqa: SLF001 - the pool's own scaler
        if woken or pool._open():  # noqa: SLF001
            try:
                pool._scale()  # noqa: SLF001
            except Exception as err:  # noqa: BLE001 - carried to the thread that polls
                pool._scale_error = err  # noqa: SLF001
                return
        backoff = pool._fleet.substrate_wait()  # noqa: SLF001
        del pool
        woken = wake.wait(min(poll_s, backoff) if backoff > 0 else poll_s)
        wake.clear()


def _round_measure(first: list[Sample] | None, this: list[Sample]) -> ResourceMeasure | None:
    """Return the measure of a calibration round's samples `this`, after the first round's `first` when it is a second.

    None without a warm one-job sample, whose load cannot be told from its work. A second round's per-item seconds,
    alone and batched, are its own; the load, the claim and the memory are of both rounds.
    """
    samples = (first or []) + this
    found = calibrated(samples) if any(not s.cold and s.batch_size == 1 for s in samples) else None
    steady = calibrated(this) if first is not None else None
    if found is None or steady is None:
        return found
    return found.model_copy(update={k: getattr(steady, k) for k in ("s_per_input_s", "input_s_per_item", "batched")})


def _note_unmeasured_device(unit: Unit, queue: str, found: ResourceMeasure | None, samples: list[Sample]) -> None:
    """Log once that `unit`'s backend ran on a device but reported no device memory per request."""
    if found is not None and found.device_peak_bytes is None and any(s.device_bytes is not None for s in samples):
        logger.warning("%s/%s on %s reported no device memory per request; batch 1", unit.task, unit.backend, queue)


def _gap(kind: str, message: str) -> FaultRecord:
    return FaultRecord(fault="infrastructure", type=kind, message=message, gap=True, charged=False)


def run_plan(run_dir: Path, profile: ExecutionProfile) -> Plan:
    """Return the plan of the run in `run_dir` under its latest placement, with `profile` and the plan's reuse policy.

    Raises:
        ValueError: the plan was placed under another profile than `profile`.
        FileNotFoundError: the run has no plan, or no claimed attempt.
    """
    from senselab_core.runtime.execution.run import load_plan  # noqa: PLC0415

    stored = load_plan(run_dir)
    if stored.placement.profile_digest != profile.canonical_digest():
        raise ValueError(f"plan {stored.hash} was placed under another execution profile; name the one it was")
    return replace(stored, execution=profile, config=RunConfig(reuse=ReusePolicy(stored.logical.reuse)))


def run_store(run_dir: Path) -> ResultStore:
    """Return the result store the run in `run_dir` names in `manifest.json`, else the cache's."""
    from senselab_core.runtime.execution.run import RESULTS_DIR, Run  # noqa: PLC0415
    from senselab_core.runtime.revision import cache_root  # noqa: PLC0415

    named = Run(run_dir).manifest().get("store")
    return ResultStore(Path(named) if isinstance(named, str) else cache_root() / RESULTS_DIR)


def coordinate(
    run_dir: Path,
    *,
    substrate: Substrate,
    profile: ExecutionProfile | None = None,
    store: ResultStore | None = None,
) -> "RunOutcome":
    """Coordinate the run in `run_dir` on a `Pool`: take the next epoch, hold its lease, and run the plan's jobs.

    The plan and its latest placement are read from the run directory, its inputs bound again (`bind_sources`), and a
    new epoch and its attempt prepared (`prepare`); the run's queues are reconciled and every job is sent through the
    pool, whose workers `substrate` starts. `profile` defaults to the packaged one; `store` to the one `manifest.json`
    names.

    Raises:
        ValueError: the plan was placed under another profile than `profile`.
        StaleCoordinator: a newer coordinator took an epoch while this one ran.
        QueueHeld: an env broke and was held.
        InputChanged, InputsUnavailable, ConflictingSources: as `start`.
    """
    from senselab_core.runtime.execution.run import bind_sources, prepare  # noqa: PLC0415
    from senselab_core.runtime.execution.runner import PydraRunner  # noqa: PLC0415 - needs pydra

    run_dir = Path(run_dir)
    profile = profile if profile is not None else ExecutionProfile.load()
    placed = run_plan(run_dir, profile)
    store = store if store is not None else run_store(run_dir)
    check_storage(profile, store.root)
    values = bind_sources(placed)
    epoch, attempt, ordinal = prepare(placed, run_dir, store)
    epochs = Epoch(run_dir)
    with epochs.held(epoch, profile.lease):
        queues = QueueSet(run_dir / RUN_QUEUES, profile, epoch=epochs.current)
        queues.reconcile()
        events = EventLog(run_dir, attempt, ordinal, epoch=epoch)
        pool = Pool(
            run_dir, queues=queues, store=store, events=events, profile=profile, substrate=substrate, epoch=epoch
        )
        try:
            return PydraRunner(profile=profile, work_dir=run_dir / "pydra").run_plan(
                placed, pool, values=values, events=events
            )
        finally:
            pool.close()


__all__ = [
    "DEAD_DIR",
    "Pool",
    "ScalerFailed",
    "coordinate",
    "larger_class",
    "report",
    "run_plan",
    "run_store",
    "split",
]
