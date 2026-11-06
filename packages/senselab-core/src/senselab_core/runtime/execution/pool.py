"""`Pool`: the `Executor` that puts fragments on a run's queues and reads what pool workers made of them.

A fragment goes to the queue of its placement's resource class under `<run dir>/queues/`; a `Substrate` has workers
serve that queue. A worker records each job's `ResultRef` in the item's outcome (`report`). The `Pool` yields a
result only once the result store holds its outputs, and a failure only as `dead/<key digest>.json` holds it, so a
restarted coordinator reads the same state from disk. Each fault class has its effect here, with the budgets of the
profile's `faults` section:

- **content**, not a gap: charged; the job is queued again until `content_attempts` charges, then it is dead;
- **resource**, not a gap: the job is queued again on the next larger memory class of its device, uncharged, at most
  `resource_replacements` times; then, or with no larger class, it is dead;
- **infrastructure**, or any gap: queued again uncharged (a gap is logged as a classification gap); at
  `infrastructure_replacements` failures of one env the env is reported once as broken (`BrokenEnv`) and its queue
  held (`Queue.hold`), so its jobs wait for a later attempt and none is failed.

A retry is a one-job fragment of the next `attempt`, logged `failed` then `queued` by this coordinator.
"""

import contextlib
import itertools
import json
import logging
import os
import threading
import time
import uuid
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Protocol

from senselab_core.kernel.identity import InputIdentity, ReusePolicy
from senselab_core.kernel.placement import NodePlacement
from senselab_core.runtime.call import RunConfig
from senselab_core.runtime.execution.epoch import Epoch, StaleCoordinator
from senselab_core.runtime.execution.events import BrokenEnv, EventLog
from senselab_core.runtime.execution.executor import (
    Admission,
    FaultRecord,
    Handle,
    JobRef,
    PlanFragment,
    QueueHeld,
    ResultRef,
    Status,
)
from senselab_core.runtime.execution.in_process import job_key, result_ref
from senselab_core.runtime.execution.job import JobOutcome
from senselab_core.runtime.execution.queue import RUN_QUEUES, Claim, QueueSet, item_digest
from senselab_core.runtime.planner import Plan
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.store import ResultStore

if TYPE_CHECKING:
    from senselab_core.runtime.execution.runner import RunOutcome

logger = logging.getLogger("senselab")

DEAD_DIR = "dead"
"""The directory of a run's dead jobs, one `<key digest>.json` each."""


class Substrate(Protocol):
    """What starts and stops the pool workers that serve a run's queues."""

    def ensure(self, queue: str) -> None:
        """Have a worker serve the queue `queue` of the run's queues."""

    def stop(self) -> None:
        """Stop the workers this substrate started."""


class ExternalWorkers:
    """A `Substrate` that starts nothing: workers are started on their own (`senselab execute --pool`)."""

    def ensure(self, queue: str) -> None:
        """Do nothing."""

    def stop(self) -> None:
        """Do nothing."""


def report(claim: Claim, digest: str, outcomes: Sequence[JobOutcome]) -> bool:
    """Commit the claimed item `digest` with the `ResultRef` of each of its jobs; return whether this call did."""
    jobs = [result_ref(outcome).model_dump(mode="json") for outcome in outcomes]
    return digest in claim.commit([digest], {"jobs": jobs})


def larger_class(profile: ExecutionProfile, name: str) -> str | None:
    """Return the resource class of `profile` with the least memory above class `name`'s, on its device and GPU type.

    None when `name` is not a class of the profile, states no memory, or no class of its kind has more.
    """
    current = profile.resource_classes.get(name)
    if current is None or current.mem_bytes is None:
        return None
    larger = sorted(
        (c.mem_bytes, n)
        for n, c in profile.resource_classes.items()
        if (c.device, c.gpu_type) == (current.device, current.gpu_type)
        and c.mem_bytes is not None
        and c.mem_bytes > current.mem_bytes
    )
    return larger[0][1] if larger else None


def _one_job(fragment: PlanFragment, index: int, **update: object) -> PlanFragment:
    """Return the fragment of job `index` of `fragment` alone, with the inputs it reads and `update` applied."""
    inputs = fragment.inputs[index]
    gathered = {g: e for g, e in fragment.gathered.items() if g in inputs.values()}
    read = {i for i in inputs.values() if i.startswith("output:")} | {e for es in gathered.values() for e in es}
    return PlanFragment.model_validate(
        {
            **fragment.model_dump(mode="json"),
            "jobs": [fragment.jobs[index].model_dump(mode="json")],
            "inputs": [dict(inputs)],
            "upstream": {i: r.model_dump(mode="json") for i, r in fragment.upstream.items() if i in read},
            "gathered": {g: list(e) for g, e in gathered.items()},
            **update,
        }
    )


@dataclass
class _Flight:
    """One fragment on a queue, by its item digest."""

    queue: str
    digest: str
    fragment: PlanFragment


@dataclass
class _Followed:
    """A submitted fragment: its items in flight, each job's charges and moves, and what became of it."""

    flights: list[_Flight]
    charged: Counter[tuple[int, ...]] = field(default_factory=Counter)
    moved: Counter[tuple[int, ...]] = field(default_factory=Counter)
    held: list[QueueHeld] = field(default_factory=list)
    seen: bool = False
    done: bool = False
    cancelled: bool = False


class Pool:
    """An `Executor` over the queues of the run in `run_dir`, its jobs run by pool workers a `substrate` starts.

    `submit` puts a fragment on its queue unless a newer epoch than `epoch` was taken (`StaleCoordinator`); `results`
    polls the item's outcome every `profile.pool.poll_s` seconds and applies each fault's effect. An admission holds
    the plan only: workers bind graph inputs from the run's `inputs.json`.

    Attributes:
        broken: Each env reported broken, once.
        gaps: Each job given a fault the worker's fallback table classified, with its fault.
    """

    logs_claims: ClassVar[bool] = True
    """Workers log a job `claimed` as they claim it, so the runner does not log it `started` as it is sent."""

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
    ) -> None:
        """Coordinate the run in `run_dir` as epoch `epoch`."""
        self.run_dir = Path(run_dir)
        self.queues, self.store, self.events = queues, store, events
        self.profile, self.substrate, self.epoch = profile, substrate, epoch
        self.broken: dict[str, BrokenEnv] = {}
        self.gaps: list[tuple[JobRef, FaultRecord]] = []
        self._epochs = Epoch(self.run_dir)
        self._plans: dict[str, Plan] = {}
        self._admitted: Counter[str] = Counter()
        self._followed: dict[str, _Followed] = {}
        self._failures: Counter[str] = Counter()
        self._holds: dict[str, QueueHeld] = {}
        self._ids = itertools.count()
        self._lock = threading.Lock()

    def admit(self, placed: Plan, values: Mapping[InputIdentity, object]) -> Admission:  # noqa: ARG002
        """Hold `placed` until it is released as often as it was admitted; its id is the run's, `run-<plan hash>`.

        Every worker binds the graph inputs from the run's records, so every admission of one plan in one run reads
        the same values; one id keeps a fragment one queue item whichever coordinator puts it.

        Raises:
            ValueError: the plan was placed under another execution profile than this pool's.
        """
        if placed.placement.profile_digest != self.profile.canonical_digest():
            raise ValueError(f"plan {placed.hash} was placed under another execution profile than this pool's")
        admission = Admission(id=f"run-{placed.hash}", plan_hash=placed.hash)
        with self._lock:
            self._plans[admission.id] = placed
            self._admitted[admission.id] += 1
        return admission

    def release(self, admission: Admission) -> None:
        """Let go of one admission of `admission`'s plan."""
        with self._lock:
            self._admitted[admission.id] -= 1
            if self._admitted[admission.id] <= 0:
                self._plans.pop(admission.id, None)
                del self._admitted[admission.id]

    def _fence(self) -> None:
        current = self._epochs.current()
        if current > self.epoch:
            raise StaleCoordinator(self.epoch, current)

    def _put(self, fragment: PlanFragment) -> _Flight:
        """Put `fragment` on its queue and have it served; refused by a newer epoch or a hold of this one.

        Raises:
            StaleCoordinator: a newer epoch was taken.
            QueueHeld: this coordinator held the fragment's queue.
        """
        self._fence()
        name = fragment.placement.resource_class
        queue = self.queues.queue(name)
        queue.unhold(before=self.epoch)
        if queue.held() is not None:
            raise self._holds.get(name) or QueueHeld(name, fragment.placement.env, 0)
        queue.put(fragment)
        self.substrate.ensure(name)
        return _Flight(name, item_digest(fragment), fragment)

    def submit(self, fragment: PlanFragment) -> Handle:
        """Put `fragment` on the queue of its resource class and return its handle.

        Raises:
            StaleCoordinator: a newer epoch than this coordinator's was taken; nothing is put.
            QueueHeld: the fragment's queue is held, its env broken; nothing is put.
        """
        flight = self._put(fragment)
        handle = Handle(id=f"pool-{self.epoch}-{next(self._ids)}")
        with self._lock:
            self._followed[handle.id] = _Followed([flight])
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

    def results(self, handle: Handle) -> Iterator[ResultRef]:
        """Yield the result of each job of `handle` as its outcome is read, retrying faults by their class.

        Raises:
            KeyError: the handle is not this pool's.
            StaleCoordinator: a retry was due after a newer epoch was taken.
            QueueHeld: a job's queue is held, its env broken; the jobs not yielded wait in the queue.
        """
        followed = self._followed[handle.id]
        while followed.flights and not followed.cancelled:
            landed = [(f, r) for f in followed.flights if (r := self.queues.queue(f.queue).outcome(f.digest))]
            for flight, record in landed:
                followed.flights.remove(flight)
                followed.seen = True
                yield from self._settle(followed, flight, record)
            held = followed.held + [self._holds[f.queue] for f in followed.flights if f.queue in self._holds]
            if held:
                raise held[0]
            if followed.flights and not landed:
                time.sleep(self.profile.pool.poll_s)
        followed.done = not followed.cancelled

    def close(self) -> None:
        """Stop the workers the substrate started."""
        self.substrate.stop()

    def _settle(self, followed: _Followed, flight: _Flight, record: Mapping[str, object]) -> Iterator[ResultRef]:
        """Yield the results of `flight`'s jobs from its outcome `record`, applying each failure's effect."""
        fragment = flight.fragment
        jobs = record.get("jobs") if record.get("outcome") == "done" else None
        given = {r.job: r for r in (ResultRef.model_validate(j) for j in jobs)} if isinstance(jobs, list) else {}
        dead = FaultRecord.model_validate(record["fault"]) if record.get("outcome") == "dead" else None
        for index, job in enumerate(fragment.jobs):
            ref = given.get(job)
            if ref is not None and ref.failure is None and ref.output_identity is not None:
                if self.store.outputs_of(ref.key_digest, ref.output_identity) is not None:
                    yield ref
                    continue
                fault = _gap("MissingOutput", f"the outputs of {job} are not in the result store")
            elif ref is not None and ref.failure is not None:
                fault = ref.failure
            else:
                fault = dead or _gap("MissingResult", f"the outcome of the item gives no result for {job}")
            key = ref.key_digest if ref is not None else self._key(fragment, index)
            yield from self._effect(followed, flight, index, key, fault)

    def _key(self, fragment: PlanFragment, index: int) -> str:
        placed = self._plans.get(fragment.admission)
        if placed is None:
            raise ValueError(f"admission {fragment.admission} is not held by this pool")
        return job_key(placed, fragment.node, fragment.inputs[index]).digest()

    def _effect(  # noqa: PLR0913 - the followed fragment, the item and the job's failure
        self, followed: _Followed, flight: _Flight, index: int, key: str, fault: FaultRecord
    ) -> Iterator[ResultRef]:
        """Apply `fault`'s effect to job `index` of `flight`: retry it, hold its queue, or yield it dead."""
        fragment, budgets = flight.fragment, self.profile.faults
        job, placement = fragment.jobs[index], fragment.placement
        if fault.gap or fault.fault == "infrastructure":
            if fault.gap:
                self.gaps.append((job, fault))
                logger.warning("classification gap: %s of %s, retried uncharged", fault.type, job)
            if self._infrastructure(flight, job, key, fault):
                self._again(followed, flight, index)
            else:
                followed.held.append(self._holds[flight.queue])
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

    def _infrastructure(self, flight: _Flight, job: JobRef, key: str, fault: FaultRecord) -> bool:
        """Count an infrastructure fault of `flight`'s env; return False once the env is broken and its queue held."""
        env = flight.fragment.placement.env
        with self._lock:
            if env in self.broken:
                return False
            self._failures[env] += 1
            failures = self._failures[env]
            self._failed(job, key, fault, env)
            if failures < self.profile.faults.infrastructure_replacements:
                return True
            self.broken[env] = BrokenEnv(env=env, fault=fault, failures=failures)
            self._holds[flight.queue] = QueueHeld(flight.queue, env, failures)
        self.queues.queue(flight.queue).hold(self.epoch)
        logger.error("env %s is broken after %d infrastructure faults; queue %s is held", env, failures, flight.queue)
        return False

    def _failed(self, job: JobRef, key: str, fault: FaultRecord, env: str) -> None:
        self.events.record("failed", job.node, job=job, key=key, failure=fault, env=env)

    def _again(self, followed: _Followed, flight: _Flight, index: int, placement: NodePlacement | None = None) -> None:
        """Queue job `index` of `flight` again alone, as the next attempt, on `placement` (its own when None)."""
        update: dict[str, object] = {"attempt": flight.fragment.attempt + 1}
        if placement is not None:
            update["placement"] = placement.model_dump(mode="json")
        retry = _one_job(flight.fragment, index, **update)
        followed.flights.append(self._put(retry))
        job = retry.jobs[0]
        self.events.record(
            "queued", job.node, job=job, placement_digest=retry.placement.canonical_digest(), env=retry.placement.env
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


def _gap(kind: str, message: str) -> FaultRecord:
    return FaultRecord(fault="infrastructure", type=kind, message=message, gap=True, charged=False)


def coordinate(
    run_dir: Path,
    *,
    substrate: Substrate,
    profile: ExecutionProfile | None = None,
    store: ResultStore | None = None,
) -> "RunOutcome":
    """Coordinate the run in `run_dir` on a `Pool`: take the next epoch, hold its lease, and run the plan's jobs.

    The plan and its latest placement are read from the run directory, its inputs bound again (`bind_sources`), and a
    new attempt prepared (`prepare`); the run's queues are reconciled and every job is sent through the pool, whose
    workers `substrate` starts. `profile` defaults to the packaged one; `store` to the one `manifest.json` names.

    Raises:
        ValueError: the plan was placed under another profile than `profile`.
        StaleCoordinator: a newer coordinator took an epoch while this one ran.
        QueueHeld: an env broke and its queue was held.
        InputChanged, InputsUnavailable, ConflictingSources: as `start`.
    """
    from senselab_core.runtime.execution.run import RESULTS_DIR, Run, bind_sources, load_plan, prepare  # noqa: PLC0415
    from senselab_core.runtime.execution.runner import PydraRunner  # noqa: PLC0415 - needs pydra
    from senselab_core.runtime.revision import cache_root  # noqa: PLC0415

    run_dir = Path(run_dir)
    profile = profile if profile is not None else ExecutionProfile.load()
    stored = load_plan(run_dir)
    if stored.placement.profile_digest != profile.canonical_digest():
        raise ValueError(f"plan {stored.hash} was placed under another execution profile; name the one it was")
    placed = replace(stored, execution=profile, config=RunConfig(reuse=ReusePolicy(stored.logical.reuse)))
    if store is None:
        named = Run(run_dir).manifest().get("store")
        store = ResultStore(Path(named) if isinstance(named, str) else cache_root() / RESULTS_DIR)
    values = bind_sources(placed)
    attempt, ordinal = prepare(placed, run_dir, store)
    queues = QueueSet(run_dir / RUN_QUEUES, profile)
    queues.reconcile()
    epochs = Epoch(run_dir)
    epoch = epochs.take()
    events = EventLog(run_dir, attempt, ordinal)
    with epochs.held(epoch, profile.lease):
        pool = Pool(
            run_dir, queues=queues, store=store, events=events, profile=profile, substrate=substrate, epoch=epoch
        )
        try:
            return PydraRunner(profile=profile, work_dir=run_dir / "pydra").run_plan(
                placed, pool, values=values, events=events
            )
        finally:
            pool.close()


__all__ = ["DEAD_DIR", "ExternalWorkers", "Pool", "Substrate", "coordinate", "larger_class", "report"]
