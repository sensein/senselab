"""Pool workers: a host process serving a run's queues of some resource classes (`senselab execute --pool`).

A pool worker claims one item at a time from the queue of the class whose oldest waiting item is oldest, logs each job
`claimed` under the run's current epoch and attempt, runs the jobs through the process's `WorkerSet` (`execute`), and
commits each item with one `ResultRef` per job and a `Sample` of what the item took (`report`); the `Pool` applies each
failure's effect. It stays on a class for at most `pool.quantum_items` jobs or `pool.quantum_s` seconds, and then claims
from the class with the oldest work. While a claim runs, its leases are proven every renewal interval; an item whose
lease was lost is not reported. A request that fails as a whole (`whole`) is split in halves within the claim until the
failing job is alone, which is reported with that failure; an item that fails inside a request is reported as it failed.
The worker exits once its queues have had nothing to claim for `IDLE_POLLS` polls of `pool.poll_s`, or on SIGTERM, after
committing the items that finished and releasing the rest. A worker of `svc:<id>` classes registers each server it
starts (`services`); once it holds one, its queues being empty decide nothing: it exits when no consumer lease of any
service it holds has been live for `services.idle_exit_s`, the jobs it runs itself counting as consumption.
"""

import contextlib
import functools
import logging
import queue
import signal
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

from senselab_worker.faults import ProtocolError, classify

from senselab_core.kernel.identity import InputIdentity, ReusePolicy
from senselab_core.runtime import request_reports
from senselab_core.runtime.calibration import Cotenancy, HardwareClass, Observation
from senselab_core.runtime.call import RunConfig, packaged
from senselab_core.runtime.clock import Clock, SystemClock
from senselab_core.runtime.execution.epoch import Epoch
from senselab_core.runtime.execution.events import EventLog
from senselab_core.runtime.execution.executor import FaultRecord, JobRef, PlanFragment
from senselab_core.runtime.execution.in_process import fragment_jobs, fragment_keys, process_workers
from senselab_core.runtime.execution.job import Failed, Job, JobOutcome, execute, ran_out, splits
from senselab_core.runtime.execution.observing import (
    Presence,
    RequestRecord,
    Seen,
    Sent,
    hardware_of,
    host_report,
    job_units,
    observations,
    records_of,
)
from senselab_core.runtime.execution.pool import report, run_plan
from senselab_core.runtime.execution.pool_parts import placed_backends
from senselab_core.runtime.execution.queue import Claim, Queue, QueueSet, ServeLease, UnmarkedProbeItem, probe_of
from senselab_core.runtime.execution.records import add_plan_entity
from senselab_core.runtime.execution.services import service_registry
from senselab_core.runtime.execution.substrates import (
    EXIT_UNREACHABLE,
    QueueReady,
    StoreUnreachable,
    class_of,
    mark_ready,
    worker_id,
)
from senselab_core.runtime.execution.supervisor import Supervisor
from senselab_core.runtime.execution.workers import WorkerSet
from senselab_core.runtime.execution_settings import is_service_class
from senselab_core.runtime.hardware import device_memory_bytes, memory_bytes, process_tree
from senselab_core.runtime.keys import Placement, request_size
from senselab_core.runtime.locks import LeaseTimeout
from senselab_core.runtime.measures import env_identity
from senselab_core.runtime.planner import Plan
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.request_reports import RequestReport
from senselab_core.runtime.sizing import Sample
from senselab_core.runtime.storage import StorageUnsupported, probe_capabilities
from senselab_core.runtime.store import ResultStore
from senselab_core.runtime.units import mark_entry
from senselab_core.runtime.worker_client import DEFAULT_GRACES, WorkerFatal, WorkerLost

logger = logging.getLogger("senselab")

IDLE_POLLS = 3
"""The polls of `pool.poll_s` with nothing to claim after which a pool worker exits."""

Send = Callable[[Sequence[Job], Placement], list[JobOutcome]]
"""Run jobs of one call under a placement and return one outcome per job, in order (`execute`)."""


class Abandoned(Exception):
    """The claim whose jobs were running was lost or halted; nothing more is sent for it.

    `kept` holds the outcomes of the jobs that had finished, by index among the members of the call that raised it.
    """

    def __init__(self, kept: Mapping[int, JobOutcome] | None = None) -> None:
        """Hold `kept`, the outcomes finished before the claim was given up."""
        super().__init__()
        self.kept = dict(kept or {})


def whole(exc: BaseException) -> bool:
    """Whether `exc` fails a request of a loaded worker as a whole: it was lost, fatal, broke the protocol, or ran out.

    A worker that failed to start or load (`WorkerLoadFailed`) is never one, whatever its class.
    """
    return isinstance(exc, (WorkerLost, WorkerFatal, ProtocolError)) or splits(exc)


def _ran_out(members: "Sequence[Member]", outcomes: Sequence[JobOutcome], report: RequestReport) -> list[int]:
    """Return the indices of the members of the request `report` names, when it has several and every one ran out."""
    named = {t for t in report.tags if isinstance(t, JobRef)}
    at = [i for i, m in enumerate(members) if m.job.ref in named]
    if len(at) < 2 or len(at) != len(named):  # noqa: PLR2004 - a request of one has none to split
        return []
    return at if ran_out([outcomes[i] for i in at]) else []


def failure_of(exc: BaseException) -> FaultRecord:
    """Return the fault record of `exc`, classified as a worker classifies it (`error_record`)."""
    return FaultRecord.from_exception(exc)


@dataclass(frozen=True)
class Member:
    """One job of a claimed item: the item's digest, the job's index in its fragment, the job and its key's digest."""

    digest: str
    index: int
    job: Job
    key: str


def bisect(  # noqa: PLR0913 - the members and their placement, the request seam, then the cancel flag and the log
    members: Sequence[Member],
    placement: Placement,
    send: Send,
    cancelled: threading.Event | None = None,
    record: Callable[[Sent], None] | None = None,
    clock: Clock = SystemClock(),  # noqa: B008 - stateless
) -> list[JobOutcome]:
    """Return one outcome per member, sent as one request; one that fails as a whole is split in halves.

    A request of a call that did not raise, which holds several jobs and gave a resource fault for each, is split too;
    the call's other requests keep their outcomes and reports. A member alone whose request fails as a whole gets that
    failure. Any other exception is the failure of every member of the request that raised it, so a half that
    completed keeps its outcomes. Each call of `send` is given to `record` as a `Sent`, timed by itself on `clock`,
    with the requests it made, each timed on `clock` too.

    Raises:
        Abandoned: `cancelled` was set before a request, or while one ran.
        LeaseTimeout: a start lease the request needs is busy; nothing failed.
    """
    if cancelled is not None and cancelled.is_set():
        raise Abandoned
    refs = tuple(m.job.ref for m in members)
    began = clock.now()
    try:
        with request_reports.collected(clock) as reported:
            outcomes = send([m.job for m in members], placement)
    except Exception as exc:
        wall = clock.now() - began
        if cancelled is not None and cancelled.is_set():
            raise Abandoned from exc
        if isinstance(exc, LeaseTimeout):
            raise
        split = whole(exc) and len(members) > 1
        if record is not None:
            record(Sent(refs, wall, (), "bisected" if split else "failed", classify(exc).value))
        if not split:
            if len(members) == 1:
                logger.warning("job %s fails its request alone: %s", members[0].job.ref, type(exc).__name__)
            return [Failed(m.job.ref, m.key, failure_of(exc)) for m in members]
        return _halves(members, placement, send, cancelled, record, clock)
    return _split_requests(members, outcomes, reported, began, (placement, send, cancelled, record, clock))


def _split_requests(
    members: Sequence[Member],
    outcomes: list[JobOutcome],
    reported: Sequence[RequestReport],
    began: float,
    again: tuple[Placement, Send, threading.Event | None, Callable[[Sent], None] | None, Clock],
) -> list[JobOutcome]:
    """Return `outcomes`, where each request of several jobs that all ran out is replaced by its halves' outcomes.

    The requests that did not run out keep their outcomes and reports; each that did is given to `record` as a
    `bisected` resource fault with its own report, and its jobs are sent again in halves.
    """
    placement, send, cancelled, record, clock = again
    faulted = [(r, at) for r in reported if (at := _ran_out(members, outcomes, r))]
    gone = {i for _, at in faulted for i in at}
    merged = list(outcomes)
    if record is not None:
        kept = tuple(r for r in reported if all(r is not f for f, _ in faulted))
        refs = tuple(m.job.ref for i, m in enumerate(members) if i not in gone)
        wall = clock.now() - began - sum(f.wall_s or 0.0 for f, _ in faulted)
        if refs:
            record(Sent(refs, max(0.0, wall), kept, None))
    finished = {i: o for i, o in enumerate(outcomes) if i not in gone}
    for ran, at in faulted:
        sub = [members[i] for i in at]
        if record is not None:
            record(Sent(tuple(m.job.ref for m in sub), ran.wall_s or 0.0, (ran,), "bisected", "resource"))
        try:
            halves = _halves(sub, placement, send, cancelled, record, clock)
        except Abandoned as stopped:
            raise Abandoned({**finished, **{at[i]: o for i, o in stopped.kept.items()}}) from stopped
        for i, outcome in zip(at, halves, strict=True):
            merged[i] = finished[i] = outcome
    return merged


def _halves(  # noqa: PLR0913 - `bisect`'s own arguments
    members: Sequence[Member],
    placement: Placement,
    send: Send,
    cancelled: threading.Event | None,
    record: Callable[[Sent], None] | None,
    clock: Clock,
) -> list[JobOutcome]:
    """Return the outcomes of `members` sent as two requests, each half through `bisect`."""
    half = len(members) // 2
    first = bisect(members[:half], placement, send, cancelled, record, clock)
    try:
        second = bisect(members[half:], placement, send, cancelled, record, clock)
    except Abandoned as stopped:
        raise Abandoned({**dict(enumerate(first)), **{half + i: o for i, o in stopped.kept.items()}}) from stopped
    return first + second


@dataclass
class _Spans:
    """What each item of a claim ran in, by digest: its wall seconds and the requests its jobs were in.

    An item's wall is its jobs' share of each request they were in, each timed by itself, and of the time a call of
    `send` took beyond its requests. Its requests are those the calls made.
    """

    walls: dict[str, float] = field(default_factory=dict)
    requests: dict[str, list[RequestRecord]] = field(default_factory=dict)

    def add(self, sent: Sent, owners: Mapping[JobRef, str]) -> None:
        """Take one call of `send`: charge each job's item its share of each request and of the rest of the call."""
        records = records_of(sent)
        rest = max(0.0, sent.wall_s - sum(r.wall_s for r in records)) / len(sent.refs)
        for ref in sent.refs:
            self._charge(owners[ref], rest)
        for record in records:
            for ref in record.refs:
                if ref in owners:
                    self._charge(owners[ref], record.wall_s / len(record.refs))
            for digest in dict.fromkeys(owners[r] for r in record.refs if r in owners):
                self.requests.setdefault(digest, []).append(record)

    def _charge(self, digest: str, seconds: float) -> None:
        self.walls[digest] = self.walls.get(digest, 0.0) + seconds


@dataclass
class _Group:
    """Members of one claim of one plan, node and placement, so one request may carry them."""

    placement: Placement
    run: RunConfig
    members: list[Member]


Finished: TypeAlias = "queue.SimpleQueue[list[tuple[Member, JobOutcome | None]] | None]"
"""The outcomes of a claim's groups as they finish, then None; an empty batch only wakes the wait, and a member's
outcome None gives its item back unreported (a busy start lease)."""


class PoolWorker:
    """A pool worker over the queues in `pool_dir` (a run's `queues` directory) of the resource classes `classes`.

    Attributes:
        served: The queue name and item digest of each item claimed, in claim order.
        stop: Set (`halt`) to make the worker commit what finished, release the rest and return.
        join_s: How long a halted claim waits for its abandoned request to stop; the graces' bound of a worker
            client by default, within which a closed worker process is stopped.
    """

    def __init__(  # noqa: PLR0913 - the queues and classes, then keyword-only parts and seams
        self,
        pool_dir: Path,
        classes: Sequence[str],
        *,
        profile: ExecutionProfile,
        store: ResultStore,
        workers: WorkerSet | None = None,
        send: Send | None = None,
        name: str | None = None,
        join_s: float = DEFAULT_GRACES.bound,
        memory: Callable[[], tuple[int | None, int]] = memory_bytes,
        device_memory: Callable[[], int | None] | None = None,
        hardware: Callable[[str], HardwareClass] | None = None,
        cotenancy: Callable[[str], Cotenancy] | None = None,
        clock: Clock = SystemClock(),  # noqa: B008 - stateless
    ) -> None:
        """Serve `classes` under `profile`, storing into `store`; jobs go through `send` (`execute` when None).

        `hardware` gives the `HardwareClass` a resource class runs on and `cotenancy` its device and host
        co-tenancy; they default to what this host shows (`hardware_of`, `Presence`). The worker's name is its
        `worker_id()` unless given, so no run record names its host. `clock` times every measurement the worker
        records (a claim, each call of `send` and each request it makes, an item's claim-to-commit wall) and the reuse
        of what it read of its peers and of the run's probe queues (`pool.presence_refresh_s`); the quantum and idle
        deadlines and every wait too.

        Raises:
            ValueError: no class is named.
        """
        if not classes:
            raise ValueError("a pool worker serves at least one resource class")
        self.run_dir = Path(pool_dir).parent
        self.epochs = Epoch(self.run_dir, clock)
        self.queues = QueueSet(Path(pool_dir), profile, epoch=self.epochs.current, clock=clock)
        self.classes = tuple(classes)
        self.profile, self.store = profile, store
        self.name = name if name is not None else worker_id()
        self.served: list[tuple[str, str]] = []
        self.stop = threading.Event()
        self._workers = workers
        self._send = send
        self._plans: dict[str, tuple[Plan, Mapping[InputIdentity, object]]] = {}
        self._finished: Finished = queue.SimpleQueue()
        self.join_s = join_s
        self._memory = memory
        self._device_memory = device_memory or (
            lambda: device_memory_bytes(process_tree(), timeout_s=profile.sizing.device_query_s)
        )
        self._baseline = memory()[0] or memory()[1]
        self._device_baseline: int | None = None
        self._warm: set[str] = set()
        self._serving: dict[str, ServeLease] = {}
        self._hardware = hardware or functools.partial(hardware_of, profile)
        self._cotenancy = cotenancy or Presence(profile, self.run_dir, clock=clock).cotenancy
        self._clock = clock
        self._listed: float | None = None
        self._entries: dict[tuple[str, str], dict[str, Any]] = {}

    def ready(self) -> dict[str, QueueReady]:
        """Return what this worker says of each queue it serves: its class, and the env of each backend placed on it.

        The backends come from the run's placement of its latest attempt (`private/placements/<attempt>.json`). Each
        env is built here, or read when built, so the mark that carries it is written after the builds; a build that
        fails is recorded in the mark, not raised.
        """
        try:
            placed = placed_backends(run_plan(self.run_dir, self.profile))
        except (FileNotFoundError, ValueError):
            placed = {}
        found = {}
        for name in self.classes:
            try:
                hardware: HardwareClass | None = self._hardware(class_of(name))
            except ValueError as err:
                logger.warning("pool worker %s cannot describe the hardware of %s: %s", self.name, name, err)
                hardware = None
            envs = {f"{task}~{backend}": self._env(name, task, backend) for task, backend in placed.get(name, ())}
            found[name] = QueueReady(hardware, envs)
        return found

    def _env(self, queue: str, task: str, backend_name: str) -> dict[str, Any]:
        """Return the mark's entry for the env of `task`'s backend `backend_name`, built here; else its cause."""
        registry = packaged().registry
        backend = next((b for b in registry.backends(task) if b.name == backend_name), None)
        if backend is None:
            return {"unbuilt": "unreadable"}
        try:
            handle = packaged().open_env(backend, self.profile)
        except Exception:  # noqa: BLE001 - a build fails in many ways; the mark records it, the Pool holds the env
            logger.exception("pool worker %s could not build the env of %s/%s", self.name, task, backend_name)
            return {"unbuilt": "build-failed"}
        device = self.profile.resource_classes[class_of(queue)].device
        try:
            return mark_entry(self.profile, backend, handle, host_report(), device)
        except ValueError:
            return {"unbuilt": "unreadable"}

    def halt(self) -> None:
        """Set `stop` and wake the wait for a running claim's outcomes; safe in a signal handler."""
        self.stop.set()
        self._finished.put([])

    def serve(self) -> int:
        """Claim and run items until the queues stay empty for `IDLE_POLLS` polls or `stop` is set; return 0.

        A probe queue the worker serves (`_pinned_probes`) is waited on from `pool.first_poll_s`, doubling to
        `pool.poll_s`, so a probe's next phase costs no full poll; its serve leases are released on return.
        """
        failed = True
        try:
            result = self._claim_loop()
            failed = False
            return result
        finally:
            for lease in self._serving.values():
                lease.release()
            self._serving.clear()
            self._write_provenance(raising=not failed)

    def _write_provenance(self, *, raising: bool) -> None:
        """Replace this worker's provenance document; an `OSError` is raised, or logged where `raising` is False."""
        try:
            self.store.provenance.write(self.run_dir)
        except OSError:
            if raising:
                raise
            logger.exception(
                "pool worker %s could not write its provenance; the error that ended it propagates", self.name
            )

    def _claim_loop(self) -> int:
        pool = self.profile.pool
        current: str | None = None
        spent, began, idle, waited = 0, self._clock.now(), None, 0.0
        while not self.stop.is_set():
            ended = current is not None and (spent >= pool.quantum_items or self._clock.now() - began >= pool.quantum_s)
            prefer = None if ended else current
            asked = self._clock.now()
            claim = self._next(prefer, pool.quantum_items - spent)
            if claim is None:
                current, idle = None, idle if idle is not None else self._clock.now()
                provided = self._workers.provided() if self._workers is not None else []
                if provided:
                    if all(p.idle_s() >= self.profile.services.idle_exit_s for p in provided):
                        break
                    self._clock.wait(self.stop, pool.poll_s)
                    continue
                left = IDLE_POLLS * pool.poll_s - (self._clock.now() - idle)
                if left <= 0:
                    break
                pause = pool.poll_s
                if self._serving:
                    waited = min(pool.poll_s, max(waited * 2, pool.first_poll_s))
                    pause = waited
                self._clock.wait(self.stop, min(pause, left))
                continue
            idle, waited = None, 0.0
            if claim.queue.name != current or ended:
                current, spent, began = claim.queue.name, 0, self._clock.now()
            spent += sum(len(f.jobs) for f in claim.fragments)
            self.served.extend((claim.queue.name, d) for d in claim.items)
            self.run_claim(claim, claim_s=self._clock.now() - asked)
            for service in self._workers.provided() if self._workers is not None else []:
                service.touch()
        return 0

    def _by_age(self) -> list[str]:
        """Return the classes with waiting work, the one whose oldest waiting item is oldest first."""
        ages = [(age, n) for n in self.classes if (age := self.queues.queue(n).oldest_age_s()) is not None]
        return [n for _, n in sorted(ages, key=lambda a: -a[0])]

    def _pinned_probes(self) -> list[str]:
        """Return the probe queues of this worker's queues it serves, oldest first.

        A worker serves a probe queue while it holds that queue's serve lease; it takes the lease of one with waiting
        work whose recorded hardware class is its own (none recorded, or the queue closed: none takes it), and skips
        one whose lease is live under another worker. A lease it lost, or on a queue since closed, is dropped. The
        run's queues are listed at most once per `pool.presence_refresh_s`.
        """
        for name, kept in list(self._serving.items()):
            if not kept.held or self.queues.queue(name).probe_closed():
                kept.release()
                del self._serving[name]
        now = self._clock.now()
        if self._listed is None or now - self._listed >= self.profile.pool.presence_refresh_s:
            self._listed = now
            for name in self.queues.names():
                found = probe_of(name)
                if found is not None and found[0] in self.classes and name not in self._serving:
                    queue = self.queues.queue(name)
                    if queue.oldest_age_s() is not None and self._serves(queue):
                        if (lease := queue.serve_lease()) is not None:
                            self._serving[name] = lease
        ages = {n: self.queues.queue(n).oldest_age_s() or 0.0 for n in self._serving}
        return sorted(ages, key=lambda n: -ages[n])

    def _serves(self, queue: Queue) -> bool:
        """Return whether this worker is what the probe queue names: its class and its env for the probe's backend.

        False while the queue names none, or is closed; the lease is never taken on a guess.
        """
        if queue.probe_closed():
            return False
        wanted, env = queue.probe_hardware(), queue.probe_env()
        if wanted is None or env is None:
            return False
        try:
            if self._hardware(class_of(queue.name)) != wanted:
                return False
        except ValueError:
            return False
        production = probe_of(queue.name)
        if production is None or "~" not in env[0]:
            return False
        task, backend = env[0].split("~", 1)
        mine = self._entries.get((production[0], env[0]))
        if mine is None:
            mine = self._entries[production[0], env[0]] = self._env(production[0], task, backend)
        return "unbuilt" not in mine and env_identity(mine) == env_identity(env[1])

    def _next(self, prefer: str | None, left: int) -> Claim | None:
        """Claim up to `left` jobs from `prefer`, else from the class with the oldest work; None if none.

        A claim holds one item, or one-job items up to the bound; on a queue this worker has not run, one item, so
        the model load falls on one item alone. A probe queue this worker serves comes first, and its items are
        claimed one per claim, never gathered.
        """
        order = [prefer] if prefer is not None else []
        order += [n for n in self._by_age() if n != prefer]
        probes = self._pinned_probes()
        for name in [*probes, *(n for n in order if n not in probes)]:
            size = left if name == prefer else self.profile.pool.quantum_items
            queue = self.queues.queue(name)
            if name in probes:
                claim = queue.claim(self.name, max_items=1)
            elif name in self._warm:
                claim = queue.claim(self.name, max_jobs=max(size, 1), singles=True)
            else:
                claim = queue.claim(self.name, max_items=1, max_jobs=max(size, 1))
            if claim is not None:
                return claim
        return None

    def run_claim(self, claim: Claim, claim_s: float = 0.0) -> None:
        """Run every item of `claim`, committing each as its jobs finish; release what is held at the end.

        An item whose jobs cannot be built is given the outcome `dead` with that failure. The jobs run on a thread
        while this one waits, proving the claim's leases every renewal interval. Once a lease is lost, the claim's
        other items are released at once, the request in flight is abandoned (`WorkerSet.abandon`), nothing more is
        sent, and this returns only after the thread stopped. Once `stop` is set, the request is abandoned too, the
        thread is given `join_s`, and the items whose jobs finished by then are committed. The worker's provenance
        document in the run's `prov/` is replaced when the claim ends, however it ends; a failed write is raised, or
        logged where an error already ends the claim.
        """
        failed = True
        try:
            self._run_claim(claim, claim_s)
            failed = False
        finally:
            self._write_provenance(raising=not failed)

    def _run_claim(self, claim: Claim, claim_s: float) -> None:
        spans = _Spans()
        with claim:
            self._log_claimed(claim)
            groups, dead = self._groups(claim)
            sample = self._sampler(claim, groups, spans, claim_s)
            observe = self._observer(claim, groups, spans)
            for digest, failure in dead.items():
                claim.dead(failure, [digest])
            pending = {d: len(f.jobs) for d, f in zip(claim.items, claim.fragments, strict=True)}
            outcomes: dict[str, dict[int, JobOutcome]] = {d: {} for d in claim.items}
            finished: Finished = queue.SimpleQueue()
            self._finished, cancelled = finished, threading.Event()
            runner = threading.Thread(target=self._run_groups, args=(groups, finished, cancelled, spans), daemon=True)
            runner.start()
            interval = self.profile.lease.renew_fraction * self.profile.lease.stale_after_s
            running = True
            while running and claim.items and not self.stop.is_set():
                with contextlib.suppress(queue.Empty):
                    batch = finished.get(timeout=interval)
                    running = batch is not None
                    self._commit(claim, batch or [], outcomes, pending, (sample, observe))
                if not claim.renew():
                    logger.warning(
                        "pool worker %s lost a claim on queue %s; its request is abandoned", self.name, claim.queue.name
                    )
                    claim.release()
                    break
            self._stop_runner(runner, cancelled)
            while not finished.empty():
                self._commit(claim, finished.get_nowait() or [], outcomes, pending, (sample, observe))
        self._warm.add(claim.queue.name)

    def _sampler(
        self, claim: Claim, groups: list["_Group"], spans: "_Spans", claim_s: float
    ) -> Callable[[str], Sample]:
        """Return what measures an item of `claim` once its jobs ran.

        Its wall (by the group spans it ran in), the items per request it was sent in, its inputs and host memory; the
        device memory its requests' backend allocated, per item beyond what stays allocated, where the backend
        reported it; and on a `cuda` class the device memory of this worker's processes, read once per claim, its
        baseline before the worker's first `cuda` claim.
        """
        batches = {
            m.digest: request_size(g.placement.backend, g.placement.batch_size) for g in groups for m in g.members
        }
        inputs: dict[str, float | None] = {}
        chars: dict[str, int | None] = {}
        for member in (m for g in groups for m in g.members):
            kind, amount = job_units(member.job)
            known, so_far = inputs.get(member.digest, 0.0), chars.get(member.digest, 0)
            inputs[member.digest] = amount + known if kind == "audio_s" and known is not None else None
            chars[member.digest] = int(amount) + so_far if kind == "chars" and so_far is not None else None
        cold = claim.queue.name not in self._warm
        device: list[int | None] = []
        grant = self.profile.resource_classes.get(class_of(claim.queue.name))
        on_device = grant is not None and grant.device == "cuda"
        if on_device and self._device_baseline is None:
            self._device_baseline = self._device_memory()

        def sample(digest: str) -> Sample:
            jobs = len(claim.fragments[claim.items.index(digest)].jobs)
            current, peak = self._memory()
            if on_device and self._device_baseline is not None and not device:
                device.append(self._device_memory())
            used = device[0] if device else None
            requests = [(r.device[0], r.device[1], r.items) for r in spans.requests.get(digest, []) if r.device]
            return Sample(
                wall_s=spans.walls.get(digest, 0.0),
                claim_s=claim_s,
                jobs=jobs,
                batch_size=min(batches.get(digest, 1), jobs),
                input_s=inputs.get(digest),
                input_chars=chars.get(digest),
                baseline_bytes=self._baseline,
                resident_bytes=current,
                peak_memory_bytes=peak,
                device_baseline_bytes=None if used is None else self._device_baseline,
                device_bytes=used,
                served=is_service_class(claim.queue.name),
                device_item_bytes=max(((p - a) // n for p, a, n in requests), default=None),
                device_allocated_bytes=min((a for _, a, _ in requests), default=None),
                cold=cold,
                job=worker_id(),
                ended_s=time.time(),
            )

        return sample

    def _observer(
        self, claim: Claim, groups: list["_Group"], spans: "_Spans"
    ) -> Callable[[str, Sequence[JobOutcome]], list[Observation]]:
        """Return what gives an item of `claim` its `Observation`s once its jobs ran.

        The hardware class and co-tenancy are read once per claim, when its first item is reported.
        """
        jobs = {m.job.ref: m.job for g in groups for m in g.members}
        began = self._clock.now()
        seen: list[Seen | None] = []

        def observe(digest: str, outcomes: Sequence[JobOutcome]) -> list[Observation]:
            if not seen:
                seen.append(self._seen(claim))
            found = seen[0]
            if found is None:
                return []
            try:
                mark = claim.queue.probe_mark(digest) if found.probe is not None else None
            except UnmarkedProbeItem as err:
                logger.error("pool worker %s reports no observations of an unmarked probe item: %s", self.name, err)
                return []
            return observations(
                digest,
                spans.requests.get(digest, []),
                jobs,
                outcomes,
                found,
                mark=mark,
                item_wall_s=self._clock.now() - began,
            )

        return observe

    def _seen(self, claim: Claim) -> Seen | None:
        """Return the hardware class and co-tenancy `claim` ran under; None, logged, where the hardware is unknown."""
        name = class_of(claim.queue.name)
        try:
            hardware = self._hardware(name)
        except ValueError as err:
            logger.warning("pool worker %s reports no observations of %s: %s", self.name, name, err)
            return None
        return Seen(hardware, self._cotenancy(name), probe_of(claim.queue.name), is_service_class(claim.queue.name))

    def _stop_runner(self, runner: threading.Thread, cancelled: threading.Event) -> None:
        """Abandon what `runner` still runs and wait for it: `join_s` once halted, else until it stops."""
        if not runner.is_alive():
            return
        cancelled.set()
        if runner.ident is not None:
            (self._workers if self._workers is not None else process_workers(self.profile)).abandon(runner.ident)
        if self.stop.is_set():
            runner.join(self.join_s)
            return
        while runner.is_alive() and not self.stop.is_set():
            runner.join(self.profile.pool.poll_s)
        if runner.is_alive():
            runner.join(self.join_s)

    def _commit(
        self,
        claim: Claim,
        batch: list[tuple[Member, JobOutcome | None]],
        outcomes: dict[str, dict[int, JobOutcome]],
        pending: dict[str, int],
        measure: tuple[Callable[[str], Sample], Callable[[str, Sequence[JobOutcome]], list[Observation]]],
    ) -> None:
        """Record `batch`'s outcomes; commit each item still held whose every job has one, with what `measure` gives.

        `measure` is the item's `Sample` and its `Observation`s, given its outcomes.

        An item with a job whose outcome is None (a busy start lease) is released, unreported.
        """
        busy = {m.digest for m, outcome in batch if outcome is None}
        claim.release([d for d in busy if d in claim.items])
        for member, outcome in batch:
            if outcome is not None:
                outcomes[member.digest][member.index] = outcome
        for digest in {m.digest for m, _ in batch} - busy:
            if len(outcomes[digest]) == pending[digest] and digest in claim.items:
                done = [outcomes[digest][i] for i in range(pending[digest])]
                sample, observe = measure
                report(claim, digest, done, sample(digest), observe(digest, done))

    def _run_groups(
        self, groups: list[_Group], finished: Finished, cancelled: threading.Event, spans: "_Spans"
    ) -> None:
        """Run each group until `cancelled`, putting its members' outcomes on `finished`; then put None.

        Each call of a group's send is given in `spans`, with the requests it made, before the outcomes are put.
        """
        try:
            for group in groups:
                send = self._send or functools.partial(self._execute, run=group.run)
                owners = {m.job.ref: m.digest for m in group.members}
                try:
                    outcomes: list[JobOutcome | None] = list(
                        bisect(
                            group.members,
                            group.placement,
                            send,
                            cancelled,
                            lambda sent: spans.add(sent, owners),
                            self._clock,
                        )
                    )
                except LeaseTimeout as busy:
                    logger.info("pool worker %s gives its items back: %s", self.name, busy)
                    outcomes = [None] * len(group.members)
                except Abandoned as stopped:
                    if stopped.kept:
                        finished.put([(group.members[i], o) for i, o in sorted(stopped.kept.items())])
                    raise
                finished.put(list(zip(group.members, outcomes, strict=True)))
        except Abandoned:
            logger.info("pool worker %s abandoned the rest of its claim", self.name)
        finally:
            finished.put(None)

    def _execute(self, jobs: Sequence[Job], placement: Placement, *, run: RunConfig) -> list[JobOutcome]:
        workers = self._workers if self._workers is not None else process_workers(self.profile)
        return execute(jobs, placement, store=self.store, run=run, profile=self.profile, workers=workers)

    def _groups(self, claim: Claim) -> tuple[list[_Group], dict[str, FaultRecord]]:
        """Return the claim's jobs grouped by plan, node and placement, and each item that cannot be built."""
        groups: dict[tuple[str, str, str], _Group] = {}
        dead: dict[str, FaultRecord] = {}
        for digest, fragment in zip(claim.items, claim.fragments, strict=True):
            try:
                placed, values = self._plan(fragment)
                jobs, placement = fragment_jobs(fragment, placed, values)
            except Exception as exc:  # noqa: BLE001 - an item that cannot be built fails with what it raised
                dead[digest] = failure_of(exc)
                continue
            keys = fragment_keys(fragment, placed, values)
            label = (fragment.plan_hash, fragment.node, fragment.placement.canonical_digest())
            run = placed.config or RunConfig(reuse=ReusePolicy(placed.logical.reuse))
            group = groups.setdefault(label, _Group(placement, run, []))
            group.members.extend(Member(digest, i, j, k) for i, (j, k) in enumerate(zip(jobs, keys, strict=True)))
        return list(groups.values()), dead

    def _plan(self, fragment: PlanFragment) -> tuple[Plan, Mapping[InputIdentity, object]]:
        """Return the run's plan and its bound graph inputs, read once per plan hash.

        Raises:
            ValueError: the run's plan is not the fragment's.
        """
        from senselab_core.runtime.execution.run import bind_sources  # noqa: PLC0415

        if fragment.plan_hash not in self._plans:
            placed = run_plan(self.run_dir, self.profile)
            if placed.hash != fragment.plan_hash:
                raise ValueError(f"the run's plan is {placed.hash}; the item is of plan {fragment.plan_hash}")
            self._plans[placed.hash] = (placed, bind_sources(placed))
            add_plan_entity(self.store.provenance, placed.hash)
        return self._plans[fragment.plan_hash]

    def _log_claimed(self, claim: Claim) -> None:
        """Log each job of `claim` `claimed` under the current epoch's attempt; nothing before an attempt exists."""
        epoch = self.epochs.current()
        attempt = self.epochs.attempt(epoch)
        if attempt is None:
            return
        log = EventLog(self.run_dir, *attempt, epoch=epoch)
        for fragment in claim.fragments:
            for job in fragment.jobs:
                log.record("claimed", job.node, job=job)


def check_reachable(run_dir: Path, store: ResultStore) -> None:
    """Probe the run directory and the result store from this host.

    Raises:
        StoreUnreachable: either root cannot be read and written here, or lacks a primitive the leases need.
    """
    for root in (run_dir, store.root):
        try:
            probe_capabilities(root).require(root)
        except (OSError, StorageUnsupported) as err:
            raise StoreUnreachable(f"{root} is not reachable from this host: {err}") from err


def serve_pool(  # noqa: PLR0913 - the queues and classes, then keyword-only parts and seams
    pool_dir: Path,
    classes: Sequence[str],
    *,
    profile: ExecutionProfile,
    store: ResultStore,
    workers: WorkerSet | None = None,
    send: Send | None = None,
    join_s: float = DEFAULT_GRACES.bound,
    clock: Clock = SystemClock(),  # noqa: B008 - stateless
) -> int:
    """Serve the queues of `classes` in `pool_dir` (a run's `queues` directory) as one `PoolWorker`; return 0.

    The worker runs under a `Supervisor`, so every process it launches, and every orphan they leave, ends with it.
    Without `workers`, a worker of service classes runs on a set that provides its servers to this host's registry
    (`service_registry`), closed when it returns.
    A worker that cannot reach the run directory or the store returns `EXIT_UNREACHABLE` and marks nothing; one that can
    marks itself ready (`mark_ready`, with each queue's `HardwareClass` and the env of each backend placed on it, built
    first) once its worker is built and supervised, just before it serves. On the main
    thread SIGTERM halts the worker (`PoolWorker.halt`); the previous handler is restored after.
    """
    try:
        check_reachable(Path(pool_dir).parent, store)
    except StoreUnreachable as err:
        logger.error("pool worker cannot start: %s", err)
        return EXIT_UNREACHABLE
    owned = None
    if workers is None and any(is_service_class(c) for c in classes):
        bound = sum(profile.resource_classes[class_of(c)].max_workers for c in classes)
        workers = owned = WorkerSet(bound, registry=service_registry(profile, clock=clock), provide=True)
    worker = PoolWorker(
        pool_dir, classes, profile=profile, store=store, workers=workers, send=send, join_s=join_s, clock=clock
    )
    with Supervisor(settings=profile.supervisor), contextlib.ExitStack() as stack:
        if owned is not None:
            stack.callback(owned.close)
        if threading.current_thread() is not threading.main_thread():
            mark_ready(Path(pool_dir).parent, worker.ready())
            return worker.serve()
        previous = signal.signal(signal.SIGTERM, lambda *_: worker.halt())
        try:
            mark_ready(Path(pool_dir).parent, worker.ready())
            return worker.serve()
        finally:
            signal.signal(signal.SIGTERM, previous)


__all__ = ["IDLE_POLLS", "Member", "PoolWorker", "Send", "bisect", "failure_of", "serve_pool", "whole"]
