"""The per-unit probe state machine of the `Pool`: resolve a unit, look up its estimate, wait on a peer, probe, end.

`ProbeMachine` holds the state of each unit this run (`State`), the parts waiting on a hello, the worker marks read,
and the stragglers of closed probe queues. The `Pool` supplies what it acts through: the queues and roots, the plan and
keys of a fragment, the grant, the scaler, and `_produce`, which puts a part sized by an estimate.
"""

import contextlib
import importlib.metadata
import logging
import os
import threading
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Literal, Protocol

from senselab_core.kernel.identity import InputIdentity
from senselab_core.runtime.calibration import (
    Candidate,
    Estimate,
    HardwareClass,
    Observation,
    Prior,
    ProbeItem,
    ProbePlan,
    UnitKind,
)
from senselab_core.runtime.execution.batching import bucket_edges
from senselab_core.runtime.execution.executor import (
    PlanFragment,
)
from senselab_core.runtime.execution.in_process import fragment_jobs
from senselab_core.runtime.execution.job import Job
from senselab_core.runtime.execution.observing import job_units as detected_units
from senselab_core.runtime.execution.pool_parts import (
    Dumped,
    Flight,
    Held,
    ProbedUnit,
    Release,
    Run,
    State,
    covers,
    worth,
)
from senselab_core.runtime.execution.queue import item_digest, probe_queue
from senselab_core.runtime.execution.substrates import (
    QueueReady,
    ReadyMark,
    WorkerHandle,
    backend_key,
    class_of,
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
from senselab_core.runtime.planner import Plan
from senselab_core.runtime.probe import Probe
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.sizing import Grant
from senselab_core.runtime.units import (
    Pending,
    UnitReads,
    Unresolved,
    decoder_of,
)

logger = logging.getLogger("senselab")


class Host(Protocol):
    """What the machine acts through, and the `Pool` is: queues, worker marks and liveness, and fragment reads."""

    @property
    def run_name(self) -> str:
        """The run directory's name."""

    def queue(self, name: str) -> "ProbeQueue":
        """Return the queue `name`."""

    def marks(self) -> Iterable[tuple[str, ReadyMark | None]]:
        """Return each worker mark not yet given, by file name; None for one that cannot be read."""

    def live_workers(self) -> list[WorkerHandle]:
        """Return the substrate's workers alive now."""

    def keys(self, part: PlanFragment) -> list[str]:
        """Return the key digest of each job of `part`."""

    def plan_of(self, part: PlanFragment) -> tuple[Plan, Mapping[InputIdentity, object]]:
        """Return the admitted plan `part` belongs to and its graph-input values."""

    def grant(self, queue: str) -> Grant:
        """Return the grant of `queue`'s resource class."""

    def produce(self, release: Release, jobs: Sequence[Job], placement: Placement) -> None:
        """Put `release`'s jobs sized by its estimate, else the safe values."""

    def wake(self) -> None:
        """Have the scaler size the workers now."""


class ProbeQueue(Protocol):
    """The probe queue operations the machine makes (`Queue` has them all)."""

    def set_probe_unit(self, hardware: HardwareClass, backend: str, env: Mapping[str, object]) -> None:
        """Record the class and the backend's env a worker must have to serve this queue."""

    def put_probe(self, fragment: PlanFragment, phase: str, stratum: int) -> None:
        """Put a probe item."""

    def outcome(self, digest: str) -> dict[str, object] | None:
        """Return the outcome record of item `digest`, None while it has none."""

    def claimed(self, digest: str) -> bool:
        """Return whether `digest` is under a live claim."""

    def close_probe(self) -> None:
        """Close the queue to every worker."""


class ProbeMachine:
    """The `ProbeState` of each unit of a run (`calibration-probe.md` §7.1), over a `Host`.

    Its inputs are the clock, the unit resolver, the measures and roots, and `Probe`; what it does to the world it does
    through `host`, so a test drives it with a host of its own.

    Attributes:
        probes: Each unit probed, by digest.
    """

    def __init__(  # noqa: PLR0913 - the profile and its reads, the clock, and the host
        self,
        profile: ExecutionProfile,
        roots: MeasureRoots,
        measures: Estimates,
        resolver: UnitReads,
        clock: Callable[[], float],
        host: Host,
    ) -> None:
        """Run units under `profile`, reading `roots`, `measures`, `resolver`; timed by `clock`, acting by `host`."""
        self.profile, self.roots, self.measures, self.resolver = profile, roots, measures, resolver
        self._clock, self.host = clock, host
        self.probes: dict[str, ProbedUnit] = {}
        self.lock = threading.Lock()
        self._machine = threading.RLock()
        self._states: dict[str, State] = {}
        self._resolving: list[Held] = []
        self._releasing: list[Release] = []
        self._waited: set[str] = set()
        self._greeted: set[str] = set()
        self._unreadable: set[str] = set()
        self._live_since: dict[str, float] = {}
        self._said: set[str] = set()
        self._stragglers: list[tuple[Held, Flight, list[int]]] = []
        self._hello: dict[str, HardwareClass] = {}
        self._marks: dict[str, QueueReady] = {}

    def place(self, held: Held) -> None:
        """Move `held`'s computed jobs through their unit's state: resolve, look up, and probe or put them."""
        with self._machine:
            self._place(held)

    def advance(self) -> None:
        """Read the new marks, place what waited on them, and tick every waiting and probing unit."""
        with self._machine:
            self._advance()

    def counts(self, queue: str) -> tuple[int, bool]:
        """Return the units of `queue` probing, and whether any of its parts is `Resolving`."""
        with self.lock:
            probing = sum(1 for s in self._states.values() if s.queue == queue and s.state == "probing")
            return probing, any(h.queue == queue for h in self._resolving)

    def close(self) -> None:
        """Release every probe lease still held."""
        for state in self._states.values():
            if state.run is not None and state.run.stack is not None:
                state.run.stack.close()

    def _place(self, held: Held) -> None:
        """Move `held`'s computed jobs through their unit's `ProbeState`: resolve, look up, and probe or put them."""
        placed, values = self.host.plan_of(held.part)
        jobs, placement = fragment_jobs(held.part, placed, values)
        decoder = decoder_of(jobs)
        ready, hello = self._marks.get(held.queue), self._hello.get(class_of(held.queue))
        unit: MeasureUnit | Unresolved = (
            decoder if isinstance(decoder, Unresolved) else self.resolver.unit(jobs[0].call, placement, decoder, hello)
        )
        if ready is not None and not isinstance(decoder, Unresolved):
            entry = ready.envs.get(backend_key(placement.backend))
            if ready.hardware is None:
                unit = Unresolved(f"a worker of {held.queue} names no hardware class")
            elif entry is None:
                unit = Pending(f"the ready mark of a worker of {held.queue} holds no env of {placement.backend.name!r}")
            elif "unbuilt" in entry:
                why = entry["unbuilt"]
                unit = Unresolved(f"the env of backend {placement.backend.name!r} could not be built: {why}")
            else:
                unit = self.resolver.unit(jobs[0].call, placement, decoder, hello or ready.hardware, entry)
        if isinstance(unit, Pending):
            held.followed.waiting.append(held.part)
            self._resolving.append(held)
            self.host.wake()
        elif isinstance(unit, Unresolved):
            self.host.produce(Release(held, None, None), jobs, placement)
        else:
            self._lookup(unit, held, jobs, placement)

    def _lookup(self, unit: MeasureUnit, held: Held, jobs: Sequence[Job], placement: Placement) -> None:
        """Take `held` to its unit's state: an ended unit sizes it, a waiting or probing one holds it, else look up."""
        digest = unit.digest()
        predicted = self._hello.get(class_of(held.queue)) is None
        state = self._states.get(digest)
        if state is not None:
            if state.state in ("peer_wait", "probing"):
                held.followed.waiting.append(held.part)
                state.held.append(held)
            else:
                self.host.produce(Release(held, state.estimate, unit.label()), jobs, placement)
            return
        estimate = self.measures.get(unit)
        plan = None if covers(estimate, jobs) else self._plan_probe(unit, held, jobs, placement)
        if plan is None or not worth(plan, estimate):
            self._states[digest] = State(
                unit,
                held.queue,
                "measured" if estimate else "unmeasured",
                self._clock(),
                estimate,
                predicted=predicted,
            )
            self.host.produce(Release(held, estimate, unit.label()), jobs, placement)
            return
        taken = self._take_lease(unit, digest)
        if taken == "peer":
            held.followed.waiting.append(held.part)
            self._states[digest] = State(
                unit, held.queue, "peer_wait", self._clock(), estimate, [held], predicted=predicted
            )
            return
        self._start_probe(unit, held, plan, taken, predicted, placement)

    def _plan_probe(self, unit: MeasureUnit, held: Held, jobs: Sequence[Job], placement: Placement) -> ProbePlan | None:
        """Return `Probe.plan` over `held`'s jobs against what the roots hold of `unit`; None when unplannable."""
        found, kind = [detected_units(j) for j in jobs], detected_kind(jobs)
        keys = self.host.keys(held.part)
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

    def _start_probe(  # noqa: PLR0913 - the unit and its part, the plan, the lease, and what the queue records
        self,
        unit: MeasureUnit,
        held: Held,
        plan: ProbePlan,
        taken: tuple[Lease | None, contextlib.ExitStack | None],
        predicted: bool,
        placement: Placement,
    ) -> None:
        """Put `plan`'s items on a new probe queue pinned to the unit's hardware class; its other jobs wait."""
        probe_id = uuid.uuid4().hex
        name = probe_queue(held.queue, probe_id)
        env = {k: v for k, v in unit.env.model_dump(mode="json").items() if k != "decoder"}
        self.host.queue(name).set_probe_unit(unit.hardware, backend_key(placement.backend), env)
        prior = tuple(self.roots.observations(unit))
        run = Run(probe_id, name, plan, held, self._clock(), prior, last=self._clock(), lease=taken[0], stack=taken[1])
        state = State(unit, held.queue, "probing", self._clock(), run=run, predicted=predicted)
        with self.lock:
            self._states[unit.digest()] = state
        held.followed.waiting.append(held.part)
        self._put_items(state, plan.items)

    def _put_items(self, state: State, items: Sequence[ProbeItem]) -> None:
        """Put probe `items` on the run's queue, one fragment of one request shape each."""
        run, dumped = state.run, Dumped.of(state.run.held.part)  # type: ignore[union-attr]
        assert run is not None
        queue = self.host.queue(run.queue)
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
        self.host.wake()

    def _advance(self) -> None:
        """Advance every unit's state: greet new workers, resolve what waited on a hello, tick waits and probes."""
        fresh = self._greet()
        if fresh and self._resolving:
            waiting, self._resolving = self._resolving, []
            for held in waiting:
                held.followed.waiting = [p for p in held.followed.waiting if p is not held.part]
                self._place(held)
        if self._resolving:
            self._expire_resolving()
        for state in list(self._states.values()):
            if state.state == "peer_wait":
                self._peer_tick(state)
            elif state.state == "probing":
                self._probe_tick(state)
        self._release_stragglers()
        while self._releasing:
            release = self._releasing[0]
            placed, values = self.host.plan_of(release.held.part)
            jobs, placement = fragment_jobs(release.held.part, placed, values)
            self.host.produce(release, jobs, placement)
            self._releasing.pop(0)

    def _release_stragglers(self) -> None:
        """Put as production the jobs of an item of a closed probe queue whose live claim has gone stale."""
        for held, flight, indices in list(self._stragglers):
            queue = self.host.queue(flight.queue)
            if queue.outcome(flight.digest) is None and not queue.claimed(flight.digest):
                self._stragglers.remove((held, flight, indices))
                held.followed.flights.remove(flight)
                part = Held(held.followed, held.queue, Dumped.of(held.part).part(indices))
                held.followed.waiting.append(part.part)
                self._releasing.append(Release(part, None, None))
            elif queue.outcome(flight.digest) is not None:
                self._stragglers.remove((held, flight, indices))

    def _greet(self) -> bool:
        """Read each new ready mark: write its class into the hardware index; a class's first hello decides it.

        Return whether a mark was read that had not been.
        """
        fresh, disagreed = False, []
        for name, mark in self.host.marks():
            if name in self._greeted:
                continue
            if mark is None:
                self._unreadable.add(name)
                continue
            self._unreadable.discard(name)
            self._greeted.add(name)
            fresh = True
            for queue, ready in mark.queues.items():
                self._marks.setdefault(queue, ready)
                resource = self.profile.resource_classes.get(class_of(queue))
                if resource is not None and ready.hardware is not None:
                    self.roots.note_seen(resource, ready.hardware)
                    if class_of(queue) not in self._hello:
                        self._hello[class_of(queue)] = ready.hardware
                        disagreed.append((class_of(queue), ready.hardware))
        for resource_class, seen in disagreed:
            self._reresolve(resource_class, seen)
        return fresh

    def _expire_resolving(self) -> None:
        """Run as production the jobs of a unit still `Resolving` `resolve_s` after its queue's first live worker."""
        now = self._clock()
        for handle in self.host.live_workers():
            if not handle.pending:
                self._live_since.setdefault(handle.queue, now)
        late = [
            h
            for h in self._resolving
            if (since := self._live_since.get(h.queue)) is not None
            and now - since >= self.profile.calibration.resolve_s
        ]
        self._resolving = [h for h in self._resolving if h not in late]
        for held in late:
            if held.queue not in self._said:
                self._said.add(held.queue)
                logger.warning(
                    "no readable ready mark from a worker of %s within resolve_s of starting (%d unreadable); "
                    "its jobs run on the safe values",
                    held.queue,
                    len(self._unreadable),
                )
            placed, values = self.host.plan_of(held.part)
            jobs, placement = fragment_jobs(held.part, placed, values)
            self.host.produce(Release(held, None, None), jobs, placement)

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
        with self.lock:
            del self._states[state.unit.digest()]
        for held in state.held:
            held.followed.waiting = [p for p in held.followed.waiting if p is not held.part]
            self._place(held)

    def _warm_s(self, observations: Sequence[Observation]) -> float:
        """Return the seconds of warm requests among `observations`: those after the model's load."""
        return sum(o.wall_s for o in observations if o.load_s is None and o.since_load)

    def _read_outcomes(self, state: State, *, replace: bool) -> None:
        """Take the observations of every awaited item that has an outcome; `replace` swaps a served item."""
        run = state.run
        assert run is not None
        queue = self.host.queue(run.queue)
        for digest, (item, _) in list(run.awaiting.items()):
            record = queue.outcome(digest)
            if record is None:
                if queue.claimed(digest):
                    run.last = self._clock()
                continue
            run.last = self._clock()
            del run.awaiting[digest]
            reported = record.get("observations") if record.get("outcome") == "done" else None
            found = [Observation.model_validate(o) for o in reported] if isinstance(reported, list) else []
            run.observations.extend(found)
            served = any(o.outcome == "served" for o in found)
            if replace and served and self._warm_s(run.observations) < self.profile.probe_s:
                old = run.plan
                run.plan = Probe.replace_served(old, item)
                self._put_items(state, [i for i in run.plan.items if i not in old.items])

    def _probe_tick(self, state: State) -> None:
        """Read the probe's outcomes; replace a served item, plan the batched phase, or end it on its budgets."""
        run, cfg = state.run, self.profile.calibration
        assert run is not None
        self._read_outcomes(state, replace=True)
        spent = self._warm_s(run.observations)
        if run.awaiting and spent < self.profile.probe_s:
            if self._clock() - run.last >= self.profile.lease.stale_after_s:
                self._conclude(state, None, abandon=True)
            return
        if run.stage == "single" and not run.awaiting and spent < self.profile.probe_s:
            estimate = self._estimate(state, [*run.prior, *run.observations])
            if estimate.status == "measured" and run.plan.batched:
                more = Probe.plan_batched(
                    run.plan,
                    estimate,
                    self.host.grant(state.queue),
                    cfg,
                    self.profile.probe_s - spent,
                    memory_headroom=self.profile.sizing.memory_headroom,
                )
                if more.items:
                    run.plan, run.stage = more, "batched"
                    self._put_items(state, more.items)
                    return
        final = self._estimate(state, [*run.prior, *run.observations])
        self._conclude(state, final if final.status == "measured" else None)

    def _estimate(self, state: State, observations: Sequence[Observation]) -> Estimate:
        return Probe.estimate(
            observations,
            self.roots.regime_records(state.unit),
            self.profile.calibration,
            hardware=state.unit.hardware,
            memory_headroom=self.profile.sizing.memory_headroom,
        )

    def _write_observations(self, state: State) -> None:
        """Write the probe's observations under the unit of the class of the worker that made each, once proven."""
        run = state.run
        assert run is not None
        if not run.observations:
            return
        if run.lease is not None and not run.lease.prove():
            logger.warning("probe %s lost its lease; its observations are not written", state.unit.label())
            return
        for hardware in dict.fromkeys(o.hardware for o in run.observations):
            provenance = Provenance(
                job=os.environ.get("SLURM_JOB_ID") or None,
                run=self.host.run_name,
                senselab=_build(),
                profile=self.profile.canonical_digest(),
                hardware=hardware,
            )
            unit = state.unit.model_copy(update={"hardware": hardware})
            self.roots.put_observations(
                unit, run.id, tuple(o for o in run.observations if o.hardware == hardware), provenance
            )

    def _conclude(self, state: State, estimate: Estimate | None, *, abandon: bool = False, again: bool = False) -> None:
        """End `state`'s probe: write its observations, release the lease and its held jobs.

        `abandon` closes the probe queue first: an item under a live claim finishes on it, the others are given up.
        `again` places the held jobs afresh (the unit was built from a prediction a hello contradicted).
        """
        run = state.run
        assert run is not None
        if abandon:
            self._read_outcomes(state, replace=False)
            queue = self.host.queue(run.queue)
            queue.close_probe()
            for digest, (item, flight) in run.awaiting.items():
                indices = [c.index for c in item.jobs]
                if queue.claimed(digest):
                    self._stragglers.append((run.held, flight, indices))
                    continue
                run.held.followed.flights.remove(flight)
                run.used.difference_update(indices)
            run.awaiting.clear()
        self._write_observations(state)
        if run.stack is not None:
            run.stack.close()
        label = state.unit.label()
        with self.lock:
            state.state, state.estimate = ("measured" if estimate else "unmeasured"), estimate
            if again:
                self._states.pop(state.unit.digest(), None)
        part = run.held.part
        left = [i for i in range(len(part.jobs)) if i not in run.used]
        waiting = list(state.held)
        if left:
            waiting.insert(0, Held(run.held.followed, run.held.queue, Dumped.of(part).part(left)))
        run.held.followed.waiting = [p for p in run.held.followed.waiting if p is not part]
        for held in waiting:
            if held.part is not part and held not in state.held:
                held.followed.waiting.append(held.part)
            if again:
                held.followed.waiting = [p for p in held.followed.waiting if p is not held.part]
                self._place(held)
            else:
                self._releasing.append(Release(held, estimate, label))
        state.held.clear()

    def _reresolve(self, resource_class: str, seen: HardwareClass) -> None:
        """Place again every unit of `resource_class` built from the prediction `seen` contradicts.

        A probe is abandoned (its in-flight items finish and are yielded, their observations written under their
        workers' class); a finished unit is dropped, its put items keeping their sizing; a waiting one is placed afresh.
        """
        for state in list(self._states.values()):
            if not (state.predicted and class_of(state.queue) == resource_class and state.unit.hardware != seen):
                continue
            if state.state == "probing":
                self._conclude(state, None, abandon=True, again=True)
                continue
            with self.lock:
                self._states.pop(state.unit.digest(), None)
            for held in state.held:
                held.followed.waiting = [p for p in held.followed.waiting if p is not held.part]
                self._place(held)


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
