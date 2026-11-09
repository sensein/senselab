"""`Pool`: the `Executor` that puts fragments on a run's queues and reads what pool workers made of them.

A fragment goes to the queue of its placement's resource class under `<run dir>/queues/`, named `<class>#<model
identity>` when its node has a resident model. The jobs the store would serve (`ResultStore.admits`) are put at once.
The others are the computed jobs of a unit (`MeasureUnit`, built by `UnitResolver`), and the unit moves through a
`ProbeState`: `Resolving` the hardware class (the index, else the first worker's ready mark), `Lookup` of its estimate,
`PeerWait` while a peer holds the unit's probe lease, `Probing` on a probe queue `<queue>~probe-<id>` served by one
worker of the resolved class, then `Measured` or `Unmeasured`. Computed jobs wait while the unit resolves, waits or
probes; once it ends they are put, cut and batched by `size` from the estimate, else its safe values. Every result of
a probe item is yielded like any other. A worker records each job's `ResultRef` in the item's outcome, with the
`Observation`s of its requests. The `Pool` yields a result only once the result store holds its outputs, and a
failure only as `dead/<key digest>.json` holds it, so a restarted coordinator reads the same state from disk. Each
fault class has its effect here, with the budgets of the profile's `faults` section:

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
import importlib.metadata
import itertools
import json
import logging
import math
import os
import threading
import time
import uuid
import weakref
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

from senselab_core.kernel.identity import InputIdentity, ReusePolicy
from senselab_core.kernel.placement import NodePlacement
from senselab_core.runtime.calibration import (
    Candidate,
    Estimate,
    HardwareClass,
    LengthProfile,
    Observation,
    Prior,
    ProbeItem,
    ProbePlan,
    UnitKind,
)
from senselab_core.runtime.call import RunConfig, host_hardware
from senselab_core.runtime.execution.batching import bucket_edges
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
from senselab_core.runtime.execution.in_process import fragment_jobs, fragment_keys, job_units
from senselab_core.runtime.execution.job import Job
from senselab_core.runtime.execution.observing import job_units as detected_units
from senselab_core.runtime.execution.plan_reads import servable_jobs
from senselab_core.runtime.execution.pool_parts import (
    Dumped,
    Flight,
    Followed,
    Held,
    ProbedUnit,
    Release,
    Run,
    State,
    covers,
    jobs_of,
    larger_class,
    report,
    split,
    worth,
)
from senselab_core.runtime.execution.queue import RUN_QUEUES, QueueSet, item_digest, probe_of, probe_queue
from senselab_core.runtime.execution.substrates import (
    WORKERS_DIR,
    BrokenClass,
    Fleet,
    Local,
    Slurm,
    Substrate,
    SubstrateError,
    class_of,
    read_mark,
    scale,
)
from senselab_core.runtime.keys import Placement
from senselab_core.runtime.locks import Lease, LeaseTimeout, LeaseUnavailable
from senselab_core.runtime.measures import (
    Build,
    Estimates,
    MeasureRoots,
    MeasureUnit,
    PeerProbing,
    Provenance,
    Unleased,
)
from senselab_core.runtime.planner import Plan, check_storage
from senselab_core.runtime.probe import Probe
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.sizing import ExtrapolationLog, Grant, size
from senselab_core.runtime.store import ResultStore
from senselab_core.runtime.units import (
    HardwareUnseen,
    UnitReads,
    UnitResolver,
    Unresolved,
    decoder_of,
    default_measures,
    measure_roots,
)

if TYPE_CHECKING:
    from senselab_core.runtime.execution.runner import RunOutcome

logger = logging.getLogger("senselab")


class ScalerFailed(RuntimeError):
    """The pool's scaler thread ended on an error that is not a substrate failure; the cause is chained."""


DEAD_DIR = "dead"
"""The directory of a run's dead jobs, one `<key digest>.json` each."""


class Pool:
    """An `Executor` over the queues of the run in `run_dir`, its jobs run by pool workers a `substrate` starts.

    `submit` puts a fragment on its queue unless a newer epoch than `epoch` was taken (`StaleCoordinator`); `poll`
    reads its items' outcomes once, never waiting, advances every unit's probe state and applies each fault's effect.
    One scaler thread, started by the first submit and stopped by `close`, sizes the substrate's workers once per
    `pool.poll_s` while a fragment is followed, and at once after each submit or retry; nothing else scales. An
    admission holds the plan only: workers bind graph inputs from the run's `inputs.json`. `events` is the log of
    `epoch`'s attempt; an env's infrastructure failures are counted from that epoch's events on disk. `measures` gives
    each unit's estimate, `roots` keeps its observations, probe lease and hardware index, and `resolver` builds its
    unit; each defaults to the profile's.

    Attributes:
        broken: Each env this pool reported broken, once.
        gaps: Each job given a fault with `gap` set, with its fault.
        probes: Each unit this pool probed, by digest.
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
        measures: Estimates | None = None,
        roots: MeasureRoots | None = None,
        resolver: UnitReads | None = None,
    ) -> None:
        """Coordinate the run in `run_dir` as epoch `epoch`; `clock` dates each epoch read and each probe wait.

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
        self._extrapolated = ExtrapolationLog()
        self.bind_epoch(epoch, events)
        self.broken: dict[str, BrokenEnv] = {}
        self.gaps: list[tuple[JobRef, FaultRecord]] = []
        self._epochs = Epoch(self.run_dir)
        self._plans: dict[str, tuple[Plan, Mapping[InputIdentity, object]]] = {}
        self._admitted: Counter[str] = Counter()
        self._followed: dict[str, Followed] = {}
        self._ids = itertools.count()
        self._lock = threading.Lock()
        self._machine = threading.RLock()
        self.roots = roots if roots is not None else measure_roots(profile)
        self.measures = measures if measures is not None else default_measures(profile, self.roots)
        self.resolver = resolver if resolver is not None else UnitResolver(profile, self.roots)
        self.probes: dict[str, ProbedUnit] = {}
        self._states: dict[str, State] = {}
        self._resolving: list[Held] = []
        self._releasing: list[Release] = []
        self._waited: set[str] = set()
        self._greeted: set[str] = set()
        self._hello: dict[str, HardwareClass] = {}
        self._rates: dict[str, dict[str, tuple[float | None, int]]] = {}

    def bind_epoch(self, epoch: int, events: EventLog) -> None:
        """Coordinate as `epoch` from now on, logging to `events`, a log of that epoch's attempt.

        A run starts its extrapolation log afresh.

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
        self._extrapolated = ExtrapolationLog()

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

    def _check_held(self, name: str, env: str) -> None:
        """Raise `QueueHeld` when `env` is held on queue `name` by this epoch or a newer one."""
        queue = self.queues.queue(name)
        queue.unhold(before=self.epoch)
        if queue.held(env) is not None:
            failures = self._env_failures(env)
            reason = failures[-1].message if failures else "held by an earlier coordinator of this epoch or a newer one"
            raise QueueHeld(name, env, len(failures), reason)

    def _put(self, fragment: PlanFragment, followed: Followed, *, retry: bool = False) -> None:
        """Put `fragment` for `followed`: a retry as one item, else its servable jobs at once and the rest by unit.

        Raises:
            StaleCoordinator: a newer epoch was taken.
            QueueHeld: the fragment's env is held on its queue by this epoch or a newer one.
            ValueError: the fragment's admission is not held by this pool.
        """
        self._fence()
        name = self._queue_name(fragment)
        self._check_held(name, fragment.placement.env)
        if retry:
            self.queues.queue(name).put_all([fragment])
            followed.flights.append(Flight(name, item_digest(fragment), fragment))
            return
        placed, values = self._admitted_plan(fragment)
        jobs, placement = fragment_jobs(fragment, placed, values)
        flags = servable_jobs(
            jobs,
            placement,
            self._keys(fragment),
            self.resolver.envs,
            profile=self.profile,
            store=self.store,
            run=placed.config if placed.config is not None else RunConfig(),
        )
        dumped = Dumped.of(fragment)
        with self._machine:
            for wanted, served in ((True, True), (False, False)):
                picks = [i for i, flag in enumerate(flags) if flag is wanted]
                if not picks:
                    continue
                part = dumped.part(picks)
                if served:
                    self._put_served(followed, name, part, placement)
                else:
                    self._place(Held(followed, name, part))

    def _put_served(self, followed: Followed, name: str, part: PlanFragment, placement: Placement) -> None:
        """Put the jobs the store serves, `quantum_items` to an item, whole requests together."""
        placed = self._admitted_plan(part)[0]
        from senselab_core.runtime.execution.in_process import node_request_size  # noqa: PLC0415 - as `split`'s caller

        request = node_request_size(placed, part.node, part.placement)
        items = split(part, self._grant(name).quantum_items, request)
        self.queues.queue(name).put_all(items)
        followed.flights.extend(Flight(name, item_digest(i), i) for i in items)
        self._wake_scaler()

    def _grant(self, queue: str) -> Grant:
        """Return the grant of `queue`'s resource class: the device pair on a `cuda` class, the host's otherwise."""
        resource = self.profile.resource_classes[class_of(queue)]
        devices = len(self.substrate.devices) if isinstance(self.substrate, Local) else None
        seen = host_hardware() if resource.device == "cuda" or resource.mem_bytes is None else None
        return Grant.seen(resource, self.profile.pool, seen, devices=devices)

    def _place(self, held: Held) -> None:
        """Move `held`'s computed jobs through their unit's `ProbeState`: resolve, look up, and probe or put them."""
        placed, values = self._admitted_plan(held.part)
        jobs, placement = fragment_jobs(held.part, placed, values)
        decoder = decoder_of(jobs)
        hello = self._hello.get(class_of(held.queue))
        unit = (
            decoder if isinstance(decoder, Unresolved) else self.resolver.unit(jobs[0].call, placement, decoder, hello)
        )
        if isinstance(unit, HardwareUnseen):
            held.followed.waiting.append(held.part)
            self._resolving.append(held)
        elif isinstance(unit, Unresolved):
            self._produce(Release(held, None, None), jobs, placement)
        else:
            self._lookup(unit, held, jobs, placement)

    def _lookup(self, unit: MeasureUnit, held: Held, jobs: Sequence[Job], placement: Placement) -> None:
        """Take `held` to its unit's state: an ended unit sizes it, a waiting or probing one holds it, else look up."""
        digest = unit.digest()
        state = self._states.get(digest)
        if state is not None:
            if state.state in ("peer_wait", "probing"):
                held.followed.waiting.append(held.part)
                state.held.append(held)
            else:
                self._produce(Release(held, state.estimate, unit.label()), jobs, placement)
            return
        estimate = self.measures.get(unit)
        plan = None if covers(estimate, jobs) else self._plan_probe(unit, held, jobs, placement)
        if plan is None or not worth(plan, estimate):
            self._states[digest] = State(
                unit, held.queue, "measured" if estimate else "unmeasured", self._clock(), estimate
            )
            self._produce(Release(held, estimate, unit.label()), jobs, placement)
            return
        taken = self._take_lease(unit, digest)
        if taken == "peer":
            held.followed.waiting.append(held.part)
            self._states[digest] = State(unit, held.queue, "peer_wait", self._clock(), estimate, [held])
            return
        self._start_probe(unit, held, plan, taken)

    def _plan_probe(self, unit: MeasureUnit, held: Held, jobs: Sequence[Job], placement: Placement) -> ProbePlan | None:
        """Return `Probe.plan` over `held`'s jobs against what the roots hold of `unit`; None when unplannable."""
        found, kind = [detected_units(j) for j in jobs], detected_kind(jobs)
        keys = self._keys(held.part)
        candidates = [
            Candidate(fragment="part", index=i, key=keys[i], units=u if k == kind != "none" else None)
            for i, (k, u) in enumerate(found)
        ]
        edges = bucket_edges(placement.backend, kind)
        if unit.formation.startswith("bucket:") and not edges:
            return None
        prior = Prior(observations=tuple(self.roots.observations(unit)), records=tuple(self.roots.regime_records(unit)))
        return Probe.plan(
            unit, candidates, prior, self.profile.calibration, unit.formation, self.profile.probe_s, bucket_edges=edges
        )

    def _take_lease(
        self, unit: MeasureUnit, digest: str
    ) -> Literal["peer"] | tuple[Lease | None, contextlib.ExitStack | None]:
        """Take the unit's probe lease; "peer" while an admitted peer holds it and this unit has not yet waited."""
        found = self.roots.probe_lease(unit)
        if isinstance(found, PeerProbing):
            return (None, None) if digest in self._waited else "peer"
        if isinstance(found, Unleased):
            return None, None
        stack = contextlib.ExitStack()
        try:
            stack.enter_context(found)
            stack.enter_context(found.keep_alive(self.profile.lease.renew_fraction))
        except LeaseTimeout:
            stack.close()
            return (None, None) if digest in self._waited else "peer"
        except LeaseUnavailable:
            stack.close()
            return None, None
        return found, stack

    def _start_probe(
        self,
        unit: MeasureUnit,
        held: Held,
        plan: ProbePlan,
        taken: tuple[Lease | None, contextlib.ExitStack | None],
    ) -> None:
        """Put `plan`'s items on a new probe queue pinned to the unit's hardware class; its other jobs wait."""
        probe_id = uuid.uuid4().hex
        name = probe_queue(held.queue, probe_id)
        self.queues.queue(name).set_probe_hardware(unit.hardware)
        prior = tuple(self.roots.observations(unit))
        run = Run(probe_id, name, plan, held, self._clock(), prior, lease=taken[0], stack=taken[1])
        state = State(unit, held.queue, "probing", self._clock(), run=run)
        with self._lock:
            self._states[unit.digest()] = state
        held.followed.waiting.append(held.part)
        self._put_items(state, plan.items)

    def _put_items(self, state: State, items: Sequence[ProbeItem]) -> None:
        """Put probe `items` on the run's queue, one fragment of one request shape each."""
        run, dumped = state.run, Dumped.of(state.run.held.part)  # type: ignore[union-attr]
        assert run is not None
        queue = self.queues.queue(run.queue)
        for item in items:
            indices = [c.index for c in item.jobs]
            placement = run.held.part.placement.model_copy(update={"batch_size": item.request_jobs})
            fragment = dumped.part(indices, placement=placement.model_dump(mode="json"))
            queue.put_probe(fragment, item.phase, item.stratum)
            flight = Flight(run.queue, item_digest(fragment), fragment)
            run.held.followed.flights.append(flight)
            run.awaiting[flight.digest] = (item, flight)
            run.used.update(indices)
            run.put[flight.digest] = len(indices)
        self.probes[state.unit.digest()] = ProbedUnit(run.id, frozenset(run.put), sum(run.put.values()))
        self._wake_scaler()

    def _produce(self, release: Release, jobs: Sequence[Job], placement: Placement) -> None:
        """Put `release`'s jobs sized (`size`) from its estimate, else the safe values: cut, and batched by length."""
        held, estimate = release.held, release.estimate
        kind = detected_kind(jobs) if estimate is None else estimate.unit_kind
        lengths = tuple(job_units(j, kind) for j in jobs)
        sized = size(
            estimate,
            self._grant(held.queue),
            self.profile.sizing,
            LengthProfile(units=lengths),
            calibration=self.profile.calibration,
            bucket_edges=bucket_edges(placement.backend, kind),
        )
        if release.label is not None:
            self._extrapolated.note(release.label, sized)
        self._check_held(held.queue, held.part.placement.env)
        dumped, items = Dumped.of(held.part), []
        for start, stop in sized.cuts:
            for batch, group in itertools.groupby(
                range(start, stop), key=lambda i: sized.batch_at(lengths[i], len(jobs))
            ):
                moved = held.part.placement.model_copy(update={"batch_size": batch})
                items.append(dumped.part(list(group), placement=moved.model_dump(mode="json")))
        self.queues.queue(held.queue).put_all(items)
        held.followed.waiting = [p for p in held.followed.waiting if p is not held.part]
        held.followed.flights.extend(Flight(held.queue, item_digest(i), i) for i in items)
        with self._lock:
            known = sized.measured.time and all(p is not None for p in sized.predicted_s)
            per_job = sum(p or 0.0 for p in sized.predicted_s) / len(jobs) if known else None
            cap = self._grant(held.queue).max_workers
            if self._grant(held.queue).devices is not None:
                cap = min(cap, self._grant(held.queue).devices * sized.workers_per_device)  # type: ignore[operator]
            self._rates.setdefault(held.queue, {})[release.label or "unmeasured"] = (per_job, cap)
        self._wake_scaler()

    def workers_wanted(self, queue: str, work: int) -> int:
        """Return the workers `queue`'s `work` jobs want.

        A probe queue wants none: its unit's one worker is counted on the queue it measures, as is the one that waits
        for a hello. The rest is by the slowest measured unit's seconds per job over `target_wall_s`; an unmeasured
        unit is not scaled (`unmeasured_workers`).
        """
        if probe_of(queue) is not None:
            return 0
        with self._lock:
            rates = list(self._rates.get(queue, {}).values())
            probing = sum(1 for s in self._states.values() if s.queue == queue and s.state == "probing")
            resolving = any(h.queue == queue for h in self._resolving)
        grant = self._grant(queue)
        items = math.ceil(work / grant.quantum_items)
        base = 0
        if work > 0:
            if not rates or any(r is None for r, _ in rates):
                base = min(self.profile.calibration.unmeasured_workers, items, grant.max_workers)
            else:
                slowest = max(r for r, _ in rates if r is not None)
                cap = max(c for _, c in rates)
                base = min(max(1, math.ceil(work * slowest / self.profile.sizing.target_wall_s)), items, cap)
        return min(base + probing + (1 if resolving and not probing else 0), grant.max_workers)

    def _advance(self) -> None:
        """Advance every unit's state: greet new workers, resolve what waited on a hello, tick waits and probes."""
        with self._machine:
            self._greet()
            waiting, self._resolving = self._resolving, []
            for held in waiting:
                held.followed.waiting = [p for p in held.followed.waiting if p is not held.part]
                self._place(held)
            for state in list(self._states.values()):
                if state.state == "peer_wait":
                    self._peer_tick(state)
                elif state.state == "probing":
                    self._probe_tick(state)
            while self._releasing:
                release = self._releasing[0]
                placed, values = self._admitted_plan(release.held.part)
                jobs, placement = fragment_jobs(release.held.part, placed, values)
                self._produce(release, jobs, placement)
                self._releasing.pop(0)

    def _greet(self) -> None:
        """Read each new ready mark: write its class into the hardware index; a class's first hello decides it."""
        directory = self.run_dir / WORKERS_DIR
        if not directory.is_dir():
            return
        for path in sorted(directory.iterdir()):
            if path.name.startswith(".") or path.name in self._greeted or (mark := read_mark(path)) is None:
                continue
            self._greeted.add(path.name)
            for queue, seen in mark.hardware.items():
                resource = self.profile.resource_classes.get(class_of(queue))
                if resource is not None:
                    self.roots.note_seen(resource, seen)
                    self._hello.setdefault(class_of(queue), seen)

    def _peer_tick(self, state: State) -> None:
        """Return a waiting unit to `New` once its peer's lease is gone or `peer_wait_s` passed; else keep waiting."""
        now = self._clock()
        if state.checked is not None and now - state.checked < self.profile.pool.poll_s:
            return
        state.checked = now
        expired = now - state.since >= self.profile.peer_wait_s
        if not expired and self.roots.peer_holds(state.unit):
            return
        if expired:
            self._waited.add(state.unit.digest())
        with self._lock:
            del self._states[state.unit.digest()]
        for held in state.held:
            held.followed.waiting = [p for p in held.followed.waiting if p is not held.part]
            self._place(held)

    def _warm_s(self, observations: Sequence[Observation]) -> float:
        """Return the seconds of warm requests among `observations`: those after the model's load."""
        return sum(o.wall_s for o in observations if o.load_s is None and o.since_load)

    def _probe_tick(self, state: State) -> None:
        """Read the probe's outcomes; replace a served item, plan the batched phase, or end it on its budgets."""
        run, cfg = state.run, self.profile.calibration
        assert run is not None
        queue = self.queues.queue(run.queue)
        for digest, (item, flight) in list(run.awaiting.items()):
            record = queue.outcome(digest)
            if record is None:
                run.touched = run.touched or queue.claimed(digest)
                continue
            run.touched = True
            del run.awaiting[digest]
            reported = record.get("observations") if record.get("outcome") == "done" else None
            found = [Observation.model_validate(o) for o in reported] if isinstance(reported, list) else []
            run.observations.extend(found)
            if any(o.outcome == "served" for o in found):
                old = run.plan
                run.plan = Probe.replace_served(old, item)
                self._put_items(state, [i for i in run.plan.items if i not in old.items])
        spent = self._warm_s(run.observations)
        if run.awaiting and spent < self.profile.probe_s:
            if not run.touched and self._clock() - run.started >= self.profile.lease.stale_after_s:
                self._conclude(state, None, write=False, abandon=True)
            return
        if run.stage == "single" and not run.awaiting and spent < self.profile.probe_s:
            estimate = self._estimate(state, [*run.prior, *run.observations])
            if estimate.status == "measured" and run.plan.batched:
                more = Probe.plan_batched(
                    run.plan,
                    estimate,
                    self._grant(state.queue),
                    cfg,
                    self.profile.probe_s - spent,
                    memory_headroom=self.profile.sizing.memory_headroom,
                )
                if more.items:
                    run.plan, run.stage = more, "batched"
                    self._put_items(state, more.items)
                    return
        final = self._estimate(state, [*run.prior, *run.observations])
        self._conclude(state, final if final.status == "measured" else None, write=True)

    def _estimate(self, state: State, observations: Sequence[Observation]) -> Estimate:
        return Probe.estimate(
            observations,
            self.roots.regime_records(state.unit),
            self.profile.calibration,
            hardware=state.unit.hardware,
            memory_headroom=self.profile.sizing.memory_headroom,
        )

    def _conclude(self, state: State, estimate: Estimate | None, *, write: bool, abandon: bool = False) -> None:
        """End `state`'s probe: write its observations under its proven lease, release the lease and its held jobs."""
        run = state.run
        assert run is not None
        if abandon:
            for digest, (item, flight) in run.awaiting.items():
                run.held.followed.flights.remove(flight)
                run.used.difference_update(c.index for c in item.jobs)
            run.awaiting.clear()
        if write and run.observations and (run.lease is None or run.lease.prove()):
            build = _build()
            provenance = Provenance(
                job=os.environ.get("SLURM_JOB_ID") or None,
                run=self.run_dir.name,
                senselab=build,
                profile=self.profile.canonical_digest(),
                hardware=state.unit.hardware,
            )
            self.roots.put_observations(state.unit, run.id, tuple(run.observations), provenance)
        elif write and run.observations:
            logger.warning("probe %s lost its lease; its observations are not written", run.id[:12])
        if run.stack is not None:
            run.stack.close()
        label = state.unit.label()
        with self._lock:
            state.state, state.estimate = ("measured" if estimate else "unmeasured"), estimate
        part = run.held.part
        left = [i for i in range(len(part.jobs)) if i not in run.used]
        waiting = list(state.held)
        if left:
            waiting.insert(0, Held(run.held.followed, run.held.queue, Dumped.of(part).part(left)))
        run.held.followed.waiting = [p for p in run.held.followed.waiting if p is not part]
        for held in waiting:
            if held.part is not part and held not in state.held:
                held.followed.waiting.append(held.part)
            self._releasing.append(Release(held, estimate, label))
        state.held.clear()

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

    def _workers_broken(self, flight: Flight) -> WorkersBroken | None:
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

    def fragment_size(self, plan: Plan, node: str) -> None:
        """Return None: the pool splits fragments into its own items."""

    def submit(self, fragment: PlanFragment) -> Handle:
        """Put `fragment` on the queue of its resource class, as quantum-sized items, and return its handle.

        Raises:
            StaleCoordinator: a newer epoch than this coordinator's was taken; nothing is put.
            QueueHeld: the fragment's env is held on its queue; nothing is put.
            ValueError: the fragment's admission is not held by this pool; nothing is put.
        """
        followed = Followed([])
        self._put(fragment, followed)
        handle = Handle(id=f"pool-{self.epoch}-{next(self._ids)}")
        with self._lock:
            self._followed[handle.id] = followed
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
        self._advance()
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
        for state in self._states.values():
            if state.run is not None and state.run.stack is not None:
                state.run.stack.close()
        self._fleet.close()

    def _settle(self, followed: Followed, flight: Flight, record: Mapping[str, object]) -> Iterator[ResultRef]:
        """Yield the results of `flight`'s jobs from its outcome `record`, applying each failure's effect.

        A reported result is taken only when it names the job's own key and the store holds its outputs; one whose
        outcome was written by a claim of another epoch, or of none, is given as served under `exact` unless it was
        served under another policy.
        """
        fragment = flight.fragment
        jobs = record.get("jobs") if record.get("outcome") == "done" else None
        given = {r.job: r for r in (ResultRef.model_validate(j) for j in jobs)} if isinstance(jobs, list) else {}
        dead = FaultRecord.model_validate(record["fault"]) if record.get("outcome") == "dead" else None
        keys = self._keys(fragment)
        reused = record.get("epoch") != self.epoch
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
                    yield ref.model_copy(update={"served": "exact"}) if reused and ref.served is None else ref
                    continue
                fault = _gap("MissingOutput", f"the outputs of {job} are not in the result store")
            elif ref is not None and ref.failure is not None:
                fault = ref.failure
            else:
                fault = dead or _gap("MissingResult", f"the outcome of the item gives no result for {job}")
            yield from self._effect(followed, flight, index, key, fault)

    def _admitted_plan(self, fragment: PlanFragment) -> tuple[Plan, Mapping[InputIdentity, object]]:
        admitted = self._plans.get(fragment.admission)
        if admitted is None:
            raise ValueError(f"admission {fragment.admission} is not held by this pool")
        return admitted

    def _keys(self, fragment: PlanFragment) -> list[str]:
        return fragment_keys(fragment, *self._admitted_plan(fragment))

    def _effect(  # noqa: PLR0913 - the followed fragment, the item and the job's failure
        self, followed: Followed, flight: Flight, index: int, key: str, fault: FaultRecord
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

    def _held(self, flight: Flight) -> QueueHeld:
        env = flight.fragment.placement.env
        broken = self.broken[env]
        return QueueHeld(flight.queue, env, broken.failures, broken.fault.message)

    def _infrastructure(self, flight: Flight, job: JobRef, key: str, fault: FaultRecord) -> bool:
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

    def _again(self, followed: Followed, flight: Flight, index: int, placement: NodePlacement | None = None) -> None:
        """Queue job `index` of `flight` again alone, as the next attempt, on `placement` (its own when None)."""
        update: dict[str, object] = {"attempt": flight.fragment.attempt + 1}
        if placement is not None:
            update["placement"] = placement.model_dump(mode="json")
        retry = jobs_of(flight.fragment, [index], **update)
        self._put(retry, followed, retry=True)
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


def detected_kind(jobs: Sequence[Job]) -> UnitKind:
    """Return the kind of input the jobs share (seconds, characters); `none` when they mix or hold neither."""
    kinds = {k for k, _ in (detected_units(j) for j in jobs) if k != "none"}
    return next(iter(kinds)) if len(kinds) == 1 else "none"


def _build() -> Build | None:
    """Return the installed senselab-core release for a record's provenance; None when it is not installed."""
    try:
        return Build(version=importlib.metadata.version("senselab-core"))
    except importlib.metadata.PackageNotFoundError:
        return None


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
    from senselab_core.runtime.execution.run import PYDRA_DIR, bind_sources, prepare  # noqa: PLC0415
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
            return PydraRunner(profile=profile, work_dir=run_dir / PYDRA_DIR).run_plan(
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
