"""`Pool`: the `Executor` that puts fragments on a run's queues and reads what pool workers made of them.

A fragment goes to the queue of its placement's resource class under `<run dir>/queues/`, named `<class>#<model
identity>` when its node has a resident model, as items of at most `pool.quantum_items` jobs each, ending on request
boundaries when a request fits in one (`split`), in job order; a `Substrate` has workers serve that queue (`scale`, each
round). A worker records each job's `ResultRef` in the item's outcome (`report`). The `Pool` yields a result only once
the result store holds its outputs, and a failure only as `dead/<key digest>.json` holds it, so a restarted coordinator
reads the same state from disk. Each fault class has its effect here, with the budgets of the profile's `faults`
section:

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
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
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
from senselab_core.runtime.execution.in_process import job_key, node_request_size, result_ref
from senselab_core.runtime.execution.job import JobOutcome
from senselab_core.runtime.execution.queue import RUN_QUEUES, Claim, QueueSet, item_digest
from senselab_core.runtime.execution.substrates import (
    WORKERS_DIR,
    BrokenClass,
    Fleet,
    Slurm,
    SlurmError,
    Substrate,
    scale,
)
from senselab_core.runtime.planner import Plan, check_storage
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.store import ResultStore

if TYPE_CHECKING:
    from senselab_core.runtime.execution.runner import RunOutcome

logger = logging.getLogger("senselab")

DEAD_DIR = "dead"
"""The directory of a run's dead jobs, one `<key digest>.json` each."""


def report(claim: Claim, digest: str, outcomes: Sequence[JobOutcome]) -> bool:
    """Commit the claimed item `digest` with the `ResultRef` of each of its jobs; return whether this call did."""
    jobs = [result_ref(outcome).model_dump(mode="json") for outcome in outcomes]
    return digest in claim.commit([digest], {"jobs": jobs})


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
    `quantum` jobs, which only a batch-invariant backend can have.

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
    polls the item's outcome, from `profile.pool.first_poll_s` doubling up to `poll_s`, and applies each fault's
    effect. An admission holds the plan only: workers bind graph inputs from the run's `inputs.json`. `events` is the
    log of `epoch`'s attempt; an env's infrastructure failures are counted from that epoch's events on disk.

    Attributes:
        broken: Each env this pool reported broken, once.
        gaps: Each job given a fault with `gap` set, with its fault.
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
    ) -> None:
        """Coordinate the run in `run_dir` as epoch `epoch`.

        Raises:
            ValueError: `events` is not a log of `epoch`.
        """
        self.run_dir = Path(run_dir)
        self.queues, self.store, self.profile, self.substrate = queues, store, profile, substrate
        pool_settings = profile.pool
        self._fleet = Fleet(
            substrate,
            workers_dir=self.run_dir / WORKERS_DIR,
            budget=profile.faults.infrastructure_replacements,
            first_s=pool_settings.first_poll_s,
            max_s=pool_settings.poll_s,
        )
        self.broken_workers: dict[str, BrokenClass] = {}
        self._surplus: dict[str, float] = {}
        self._relogged: set[str] = set()
        self._scaled = float("-inf")
        self.bind_epoch(epoch, events)
        self.broken: dict[str, BrokenEnv] = {}
        self.gaps: list[tuple[JobRef, FaultRecord]] = []
        self._epochs = Epoch(self.run_dir)
        self._plans: dict[str, Plan] = {}
        self._admitted: Counter[str] = Counter()
        self._followed: dict[str, _Followed] = {}
        self._ids = itertools.count()
        self._lock = threading.Lock()

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

    def admit(self, placed: Plan, values: Mapping[InputIdentity, object]) -> Admission:  # noqa: ARG002
        """Hold `placed` until it is released as often as it was admitted; its id is the run's, `run-<plan hash>`.

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

    def _put(self, fragment: PlanFragment) -> list[_Flight]:
        """Put `fragment` on its queue as its items (`split`) and have them served; refused by a newer epoch or a hold.

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
            raise QueueHeld(name, env, self._env_failures(env))
        request = node_request_size(self._admitted_plan(fragment), fragment.node, fragment.placement)
        items = split(fragment, self.profile.pool.quantum_items, request)
        queue.put_all(items)
        self._scale()
        return [_Flight(name, item_digest(item), item) for item in items]

    def _queue_name(self, fragment: PlanFragment) -> str:
        """Return the queue of `fragment`: its class, with `#<model identity>` for its node's resident model."""
        placed = self._plans.get(fragment.admission)
        model = placed.logical.models.get(fragment.node) if placed is not None else None
        name = fragment.placement.resource_class
        return name if model is None else f"{name}#{model.identity}"

    def _scale(self) -> None:
        """Have the substrate size its workers to the queues: at once the first time, then once per `pool.poll_s`.

        A failing substrate is retried next round and counted (`Fleet.fail_substrate`); once it keeps failing the
        run is held (`WorkersBroken`), never failed.
        """
        now = time.monotonic()
        if self._scaled != float("-inf") and now - self._scaled < self.profile.pool.poll_s:
            return
        self._scaled = now
        try:
            scale(self.queues, self._fleet, self.profile, surplus_since=self._surplus)
        except SlurmError as err:
            logger.warning("the substrate failed: %s", err)
            self._fleet.fail_substrate(err)
        else:
            self._fleet.substrate_ok()

    def _workers_broken(self, flight: _Flight) -> WorkersBroken | None:
        """Return the hold of `flight`'s queue by a broken class of workers, noting the break once."""
        found = self._fleet.broken_for(flight.queue)
        if found is None:
            return None
        if found.resource_class not in self.broken_workers:
            self.broken_workers[found.resource_class] = found
        if (flight.digest) not in self._relogged:
            self._relogged.add(flight.digest)
            for index, job in enumerate(flight.fragment.jobs):
                key = self._key(flight.fragment, index)
                self.events.record("queued", job.node, job=job, key=key, env=flight.fragment.placement.env)
        return WorkersBroken(flight.queue, found.resource_class, found.failures, found.reason)

    def submit(self, fragment: PlanFragment) -> Handle:
        """Put `fragment` on the queue of its resource class, as quantum-sized items, and return its handle.

        Raises:
            StaleCoordinator: a newer epoch than this coordinator's was taken; nothing is put.
            QueueHeld: the fragment's env is held on its queue; nothing is put.
            ValueError: the fragment's admission is not held by this pool; nothing is put.
        """
        flights = self._put(fragment)
        handle = Handle(id=f"pool-{self.epoch}-{next(self._ids)}")
        with self._lock:
            self._followed[handle.id] = _Followed(flights)
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
        """Yield the result of each job of `handle` as its item's outcome is read, retrying faults by their class.

        Items whose outcomes are read in one poll are taken in job order.

        Raises:
            KeyError: the handle is not this pool's.
            StaleCoordinator: a newer epoch was taken while this pool waited.
            QueueHeld: a job's env is held; the jobs not yielded wait in their queues.
        """
        pause = self.profile.pool.first_poll_s
        while True:
            given, done = self.poll(handle)
            yield from given
            if done:
                return
            if given:
                pause = self.profile.pool.first_poll_s
            else:
                time.sleep(pause)
                pause = min(2 * pause, self.profile.pool.poll_s)

    def poll(self, handle: Handle) -> tuple[list[ResultRef], bool]:
        """Read the outcomes of `handle`'s items once, applying each fault's effect, and return the results landed.

        Returns the results not yet given, items read in one poll in job order, and whether the fragment is ended:
        every result given, or the handle cancelled. A hold is raised once the results landed with it are given.

        Raises:
            KeyError: the handle is not this pool's.
            StaleCoordinator: a newer epoch was taken.
            QueueHeld: a job's env is held, or its class's workers are broken; the jobs not given wait in their queues.
        """
        followed = self._followed[handle.id]
        if followed.cancelled or followed.done:
            return [], True
        self._fence()
        self._scale()
        given: list[ResultRef] = []
        landed = [(f, r) for f in followed.flights if (r := self.queues.queue(f.queue).outcome(f.digest))]
        for flight, record in landed:
            followed.flights.remove(flight)
            followed.seen = True
            given.extend(self._settle(followed, flight, record))
        held = followed.held + [self._held(f) for f in followed.flights if f.fragment.placement.env in self.broken]
        stopped = [b for f in followed.flights if (b := self._workers_broken(f)) is not None]
        if (held or stopped) and not given:
            raise (held or stopped)[0]
        followed.done = not followed.flights and not held
        return given, followed.done

    def close(self) -> None:
        """Stop the workers the substrate started."""
        self._fleet.close()

    def _settle(self, followed: _Followed, flight: _Flight, record: Mapping[str, object]) -> Iterator[ResultRef]:
        """Yield the results of `flight`'s jobs from its outcome `record`, applying each failure's effect.

        A reported result is taken only when it names the job's own key and the store holds its outputs.
        """
        fragment = flight.fragment
        jobs = record.get("jobs") if record.get("outcome") == "done" else None
        given = {r.job: r for r in (ResultRef.model_validate(j) for j in jobs)} if isinstance(jobs, list) else {}
        dead = FaultRecord.model_validate(record["fault"]) if record.get("outcome") == "dead" else None
        for index, job in enumerate(fragment.jobs):
            key = self._key(fragment, index)
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
            yield from self._effect(followed, flight, index, key, fault)

    def _admitted_plan(self, fragment: PlanFragment) -> Plan:
        placed = self._plans.get(fragment.admission)
        if placed is None:
            raise ValueError(f"admission {fragment.admission} is not held by this pool")
        return placed

    def _key(self, fragment: PlanFragment, index: int) -> str:
        placed = self._admitted_plan(fragment)
        return job_key(placed, fragment.node, fragment.inputs[index], fragment.placement).digest()

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

    def _env_failures(self, env: str) -> int:
        """Return the infrastructure and gap failures of `env` this epoch's events on disk record."""
        return sum(
            1
            for e in read_events(self.run_dir)
            if e.kind == "failed"
            and e.epoch == self.epoch
            and e.env == env
            and e.failure is not None
            and (e.failure.gap or e.failure.fault == "infrastructure")
        )

    def _held(self, flight: _Flight) -> QueueHeld:
        env = flight.fragment.placement.env
        return QueueHeld(flight.queue, env, self.broken[env].failures)

    def _infrastructure(self, flight: _Flight, job: JobRef, key: str, fault: FaultRecord) -> bool:
        """Count an infrastructure fault of `flight`'s env this epoch; return False once the env is broken and held.

        A broken env is held on `flight`'s queue (each queue it is seen on), after the fence.

        Raises:
            StaleCoordinator: a newer epoch was taken.
        """
        env, budget = flight.fragment.placement.env, self.profile.faults.infrastructure_replacements
        with self._lock:
            failures = self._env_failures(env)
            if env not in self.broken and failures < budget:
                self._failed(job, key, fault, env)
                failures += 1
                if failures < budget:
                    return True
            reported = env in self.broken
            if not reported:
                self.broken[env] = BrokenEnv(env=env, fault=fault, failures=failures)
        self._fence()
        self.queues.queue(flight.queue).hold(self.epoch, env)
        if not reported:
            logger.error("env %s is broken after %d infrastructure faults; it is held", env, failures)
        return False

    def _failed(self, job: JobRef, key: str, fault: FaultRecord, env: str) -> None:
        self.events.record("failed", job.node, job=job, key=key, failure=fault, env=env)

    def _again(self, followed: _Followed, flight: _Flight, index: int, placement: NodePlacement | None = None) -> None:
        """Queue job `index` of `flight` again alone, as the next attempt, on `placement` (its own when None)."""
        update: dict[str, object] = {"attempt": flight.fragment.attempt + 1}
        if placement is not None:
            update["placement"] = placement.model_dump(mode="json")
        retry = _jobs_of(flight.fragment, [index], **update)
        followed.flights.extend(self._put(retry))
        job, placed = retry.jobs[0], retry.placement
        key = self._key(retry, 0)
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
    "coordinate",
    "larger_class",
    "report",
    "run_plan",
    "run_store",
    "split",
]
