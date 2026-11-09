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
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

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
from senselab_core.runtime.execution.queue import QueueSet, item_digest, probe_queue
from senselab_core.runtime.execution.substrates import (
    WORKERS_DIR,
    Fleet,
    class_of,
    read_mark,
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
    HardwareUnseen,
    UnitReads,
    Unresolved,
    decoder_of,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger("senselab")


class ProbeMachine:
    """The `ProbeState` of each unit of a run (`calibration-probe.md` §7.1); mixed into the `Pool`."""

    run_dir: Path
    profile: ExecutionProfile
    queues: QueueSet
    roots: MeasureRoots
    measures: Estimates
    resolver: UnitReads
    probes: dict[str, ProbedUnit]
    _clock: Callable[[], float]
    _lock: threading.Lock
    _machine: threading.RLock
    _fleet: Fleet
    _states: dict[str, State]
    _resolving: list[Held]
    _releasing: list[Release]
    _waited: set[str]
    _greeted: set[str]
    _unreadable: set[str]
    _live_since: dict[str, float]
    _said: set[str]
    _stragglers: list[tuple[Held, Flight, list[int]]]
    _hello: dict[str, HardwareClass]

    def _keys(self, fragment: PlanFragment) -> list[str]:
        raise NotImplementedError

    def _admitted_plan(self, fragment: PlanFragment) -> tuple[Plan, Mapping[InputIdentity, object]]:
        raise NotImplementedError

    def _grant(self, queue: str) -> Grant:
        raise NotImplementedError

    def _wake_scaler(self) -> None:
        raise NotImplementedError

    def _produce(self, release: Release, jobs: Sequence[Job], placement: Placement) -> None:
        raise NotImplementedError

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
            self._wake_scaler()
        elif isinstance(unit, Unresolved):
            self._produce(Release(held, None, None), jobs, placement)
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
                self._produce(Release(held, state.estimate, unit.label()), jobs, placement)
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
            self._produce(Release(held, estimate, unit.label()), jobs, placement)
            return
        taken = self._take_lease(unit, digest)
        if taken == "peer":
            held.followed.waiting.append(held.part)
            self._states[digest] = State(
                unit, held.queue, "peer_wait", self._clock(), estimate, [held], predicted=predicted
            )
            return
        self._start_probe(unit, held, plan, taken, predicted)

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
        predicted: bool,
    ) -> None:
        """Put `plan`'s items on a new probe queue pinned to the unit's hardware class; its other jobs wait."""
        probe_id = uuid.uuid4().hex
        name = probe_queue(held.queue, probe_id)
        self.queues.queue(name).set_probe_hardware(unit.hardware)
        prior = tuple(self.roots.observations(unit))
        run = Run(probe_id, name, plan, held, self._clock(), prior, last=self._clock(), lease=taken[0], stack=taken[1])
        state = State(unit, held.queue, "probing", self._clock(), run=run, predicted=predicted)
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

    def _advance(self) -> None:
        """Advance every unit's state: greet new workers, resolve what waited on a hello, tick waits and probes."""
        with self._machine:
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
                placed, values = self._admitted_plan(release.held.part)
                jobs, placement = fragment_jobs(release.held.part, placed, values)
                self._produce(release, jobs, placement)
                self._releasing.pop(0)

    def _release_stragglers(self) -> None:
        """Put as production the jobs of an item of a closed probe queue whose live claim has gone stale."""
        for held, flight, indices in list(self._stragglers):
            queue = self.queues.queue(flight.queue)
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
        directory = self.run_dir / WORKERS_DIR
        if not directory.is_dir():
            return False
        fresh, disagreed = False, []
        for path in sorted(directory.iterdir()):
            if path.name.startswith(".") or path.name in self._greeted:
                continue
            if (mark := read_mark(path)) is None:
                self._unreadable.add(path.name)
                continue
            self._unreadable.discard(path.name)
            self._greeted.add(path.name)
            fresh = True
            for queue, seen in mark.hardware.items():
                resource = self.profile.resource_classes.get(class_of(queue))
                if resource is not None:
                    self.roots.note_seen(resource, seen)
                    if class_of(queue) not in self._hello:
                        self._hello[class_of(queue)] = seen
                        disagreed.append((class_of(queue), seen))
        for resource_class, seen in disagreed:
            self._reresolve(resource_class, seen)
        return fresh

    def _expire_resolving(self) -> None:
        """Run as production the jobs of a unit still `Resolving` `resolve_s` after its queue's first live worker."""
        now = self._clock()
        for handle in self._fleet.live():
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
            placed, values = self._admitted_plan(held.part)
            jobs, placement = fragment_jobs(held.part, placed, values)
            self._produce(Release(held, None, None), jobs, placement)

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

    def _read_outcomes(self, state: State, *, replace: bool) -> None:
        """Take the observations of every awaited item that has an outcome; `replace` swaps a served item."""
        run = state.run
        assert run is not None
        queue = self.queues.queue(run.queue)
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
                run=self.run_dir.name,
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
            queue = self.queues.queue(run.queue)
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
        with self._lock:
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
            with self._lock:
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
