"""Pool workers: a host process serving a run's queues of some resource classes (`senselab execute --pool`).

A pool worker claims items from the queue of the class whose oldest waiting item is oldest, logs each job `claimed`
under the run's current epoch and attempt, runs the jobs through the process's `WorkerSet` (`execute`), and commits
each item with one `ResultRef` per job (`report`); the `Pool` applies each failure's effect. It stays on a class for
a quantum, `pool.quantum_items` jobs or `pool.quantum_s` seconds, and then claims from the class with the oldest
work. While a claim runs, its leases are proven every renewal interval; an item whose lease was lost is not
reported. A request that fails as a whole (`whole`) is split in halves within the claim until the failing job is
alone, which is reported with that failure; an item that fails inside a request is reported as it failed. The worker
exits once its queues have had nothing to claim for `IDLE_POLLS` polls of `pool.poll_s`, or on SIGTERM, after
committing the items that finished and releasing the rest.
"""

import contextlib
import functools
import logging
import os
import queue
import signal
import socket
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from senselab_worker.faults import FaultClass, ProtocolError, classify, error_record

from senselab_core.kernel.identity import InputIdentity, ReusePolicy
from senselab_core.runtime.call import RunConfig
from senselab_core.runtime.execution.epoch import Epoch
from senselab_core.runtime.execution.events import EventLog
from senselab_core.runtime.execution.executor import FaultRecord, PlanFragment
from senselab_core.runtime.execution.in_process import fragment_jobs, job_key, process_workers
from senselab_core.runtime.execution.job import Failed, Job, JobOutcome, execute
from senselab_core.runtime.execution.pool import report, run_plan
from senselab_core.runtime.execution.queue import Claim, QueueSet
from senselab_core.runtime.execution.substrates import EXIT_UNREACHABLE, StoreUnreachable, mark_ready
from senselab_core.runtime.execution.supervisor import Supervisor
from senselab_core.runtime.execution.workers import WorkerSet
from senselab_core.runtime.keys import Placement
from senselab_core.runtime.locks import LeaseTimeout
from senselab_core.runtime.planner import Plan
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.storage import StorageUnsupported, probe_capabilities
from senselab_core.runtime.store import ResultStore
from senselab_core.runtime.worker_client import DEFAULT_GRACES, WorkerFatal, WorkerLost

logger = logging.getLogger("senselab")

IDLE_POLLS = 3
"""The polls of `pool.poll_s` with nothing to claim after which a pool worker exits."""

Send = Callable[[Sequence[Job], Placement], list[JobOutcome]]
"""Run jobs of one call under a placement and return one outcome per job, in order (`execute`)."""


class Abandoned(Exception):
    """The claim whose jobs were running was lost or halted; nothing more is sent for it."""


def whole(exc: BaseException) -> bool:
    """Whether `exc` fails a request of a loaded worker as a whole: it was lost, fatal, broke the protocol, or ran out.

    A worker that failed to start or load (`WorkerLoadFailed`) is an infrastructure fault of its env and is never one.
    """
    return isinstance(exc, (WorkerLost, WorkerFatal, ProtocolError)) or classify(exc) is FaultClass.RESOURCE


def failure_of(exc: BaseException) -> FaultRecord:
    """Return the fault record of `exc`, classified as a worker classifies it (`error_record`)."""
    return FaultRecord.of(error_record(exc))


@dataclass(frozen=True)
class Member:
    """One job of a claimed item: the item's digest, the job's index in its fragment, the job and its key's digest."""

    digest: str
    index: int
    job: Job
    key: str


def bisect(
    members: Sequence[Member], placement: Placement, send: Send, cancelled: threading.Event | None = None
) -> list[JobOutcome]:
    """Return one outcome per member, sent as one request; one that fails as a whole is split in halves.

    A member alone whose request fails as a whole gets that failure. Any other exception is the failure of every
    member of the request that raised it, so a half that completed keeps its outcomes.

    Raises:
        Abandoned: `cancelled` was set before a request, or while one ran.
        LeaseTimeout: a start lease the request needs is busy; nothing failed.
    """
    if cancelled is not None and cancelled.is_set():
        raise Abandoned
    try:
        return send([m.job for m in members], placement)
    except Exception as exc:
        if cancelled is not None and cancelled.is_set():
            raise Abandoned from exc
        if isinstance(exc, LeaseTimeout):
            raise
        if not whole(exc) or len(members) == 1:
            if len(members) == 1:
                logger.warning("job %s fails its request alone: %s", members[0].job.ref, type(exc).__name__)
            return [Failed(m.job.ref, m.key, failure_of(exc)) for m in members]
        half = len(members) // 2
        return bisect(members[:half], placement, send, cancelled) + bisect(members[half:], placement, send, cancelled)


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
    ) -> None:
        """Serve `classes` under `profile`, storing into `store`; jobs go through `send` (`execute` when None).

        Raises:
            ValueError: no class is named.
        """
        if not classes:
            raise ValueError("a pool worker serves at least one resource class")
        self.run_dir = Path(pool_dir).parent
        self.epochs = Epoch(self.run_dir)
        self.queues = QueueSet(Path(pool_dir), profile, epoch=self.epochs.current)
        self.classes = tuple(classes)
        self.profile, self.store = profile, store
        self.name = name if name is not None else f"{socket.gethostname()}:{os.getpid()}"
        self.served: list[tuple[str, str]] = []
        self.stop = threading.Event()
        self._workers = workers
        self._send = send
        self._plans: dict[str, tuple[Plan, Mapping[InputIdentity, object]]] = {}
        self._finished: Finished = queue.SimpleQueue()
        self.join_s = join_s

    def halt(self) -> None:
        """Set `stop` and wake the wait for a running claim's outcomes; safe in a signal handler."""
        self.stop.set()
        self._finished.put([])

    def serve(self) -> int:
        """Claim and run items until the queues stay empty for `IDLE_POLLS` polls or `stop` is set; return 0."""
        pool = self.profile.pool
        current: str | None = None
        spent, began, idle = 0, time.monotonic(), None
        while not self.stop.is_set():
            ended = current is not None and (spent >= pool.quantum_items or time.monotonic() - began >= pool.quantum_s)
            prefer = None if ended else current
            claim = self._next(prefer, pool.quantum_items - spent)
            if claim is None:
                current, idle = None, idle if idle is not None else time.monotonic()
                left = IDLE_POLLS * pool.poll_s - (time.monotonic() - idle)
                if left <= 0:
                    break
                self.stop.wait(min(pool.poll_s, left))
                continue
            idle = None
            if claim.queue.name != current or ended:
                current, spent, began = claim.queue.name, 0, time.monotonic()
            spent += sum(len(f.jobs) for f in claim.fragments)
            self.served.extend((claim.queue.name, d) for d in claim.items)
            self.run_claim(claim)
        return 0

    def _by_age(self) -> list[str]:
        """Return the classes with waiting work, the one whose oldest waiting item is oldest first."""
        ages = [(age, n) for n in self.classes if (age := self.queues.queue(n).oldest_age_s()) is not None]
        return [n for _, n in sorted(ages, key=lambda a: -a[0])]

    def _next(self, prefer: str | None, left: int) -> Claim | None:
        """Claim up to `left` jobs from `prefer`, else a quantum from the class with the oldest work; None if none."""
        order = [prefer] if prefer is not None else []
        order += [n for n in self._by_age() if n != prefer]
        for name in order:
            size = left if name == prefer else self.profile.pool.quantum_items
            claim = self.queues.queue(name).claim(self.name, max_items=max(size, 1), max_jobs=max(size, 1))
            if claim is not None:
                return claim
        return None

    def run_claim(self, claim: Claim) -> None:
        """Run every item of `claim`, committing each as its jobs finish; release what is held at the end.

        An item whose jobs cannot be built is given the outcome `dead` with that failure. The jobs run on a thread
        while this one waits, proving the claim's leases every renewal interval. Once a lease is lost, the claim's
        other items are released at once, the request in flight is abandoned (`WorkerSet.abandon`), nothing more is
        sent, and this returns only after the thread stopped. Once `stop` is set, the request is abandoned too, the
        thread is given `join_s`, and the items whose jobs finished by then are committed.
        """
        with claim:
            self._log_claimed(claim)
            groups, dead = self._groups(claim)
            for digest, failure in dead.items():
                claim.dead(failure, [digest])
            pending = {d: len(f.jobs) for d, f in zip(claim.items, claim.fragments, strict=True)}
            outcomes: dict[str, dict[int, JobOutcome]] = {d: {} for d in claim.items}
            finished: Finished = queue.SimpleQueue()
            self._finished, cancelled = finished, threading.Event()
            runner = threading.Thread(target=self._run_groups, args=(groups, finished, cancelled), daemon=True)
            runner.start()
            interval = self.profile.lease.renew_fraction * self.profile.lease.stale_after_s
            running = True
            while running and claim.items and not self.stop.is_set():
                with contextlib.suppress(queue.Empty):
                    batch = finished.get(timeout=interval)
                    running = batch is not None
                    self._commit(claim, batch or [], outcomes, pending)
                if not claim.renew():
                    logger.warning(
                        "pool worker %s lost a claim on queue %s; its request is abandoned", self.name, claim.queue.name
                    )
                    claim.release()
                    break
            self._stop_runner(runner, cancelled)
            while not finished.empty():
                self._commit(claim, finished.get_nowait() or [], outcomes, pending)

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
    ) -> None:
        """Record `batch`'s outcomes; commit each item still held whose every job has one.

        An item with a job whose outcome is None (a busy start lease) is released, unreported.
        """
        busy = {m.digest for m, outcome in batch if outcome is None}
        claim.release([d for d in busy if d in claim.items])
        for member, outcome in batch:
            if outcome is not None:
                outcomes[member.digest][member.index] = outcome
        for digest in {m.digest for m, _ in batch} - busy:
            if len(outcomes[digest]) == pending[digest] and digest in claim.items:
                report(claim, digest, [outcomes[digest][i] for i in range(pending[digest])])

    def _run_groups(self, groups: list[_Group], finished: Finished, cancelled: threading.Event) -> None:
        """Run each group until `cancelled`, putting its members' outcomes on `finished`; then put None."""
        try:
            for group in groups:
                send = self._send or functools.partial(self._execute, run=group.run)
                try:
                    outcomes: list[JobOutcome | None] = list(bisect(group.members, group.placement, send, cancelled))
                except LeaseTimeout as busy:
                    logger.info("pool worker %s gives its items back: %s", self.name, busy)
                    outcomes = [None] * len(group.members)
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
            keys = [job_key(placed, fragment.node, i, fragment.placement).digest() for i in fragment.inputs]
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
) -> int:
    """Serve the queues of `classes` in `pool_dir` (a run's `queues` directory) as one `PoolWorker`; return 0.

    The worker runs under a `Supervisor`, so every process it launches, and every orphan they leave, ends with it.
    A worker that cannot reach the run directory or the store returns `EXIT_UNREACHABLE` and marks nothing; one that can
    marks itself ready (`mark_ready`) once its worker is built and supervised, just before it serves. On the main
    thread SIGTERM halts the worker (`PoolWorker.halt`); the previous handler is restored after.
    """
    try:
        check_reachable(Path(pool_dir).parent, store)
    except StoreUnreachable as err:
        logger.error("pool worker cannot start: %s", err)
        return EXIT_UNREACHABLE
    worker = PoolWorker(pool_dir, classes, profile=profile, store=store, workers=workers, send=send, join_s=join_s)
    with Supervisor():
        if threading.current_thread() is not threading.main_thread():
            mark_ready(Path(pool_dir).parent)
            return worker.serve()
        previous = signal.signal(signal.SIGTERM, lambda *_: worker.halt())
        try:
            mark_ready(Path(pool_dir).parent)
            return worker.serve()
        finally:
            signal.signal(signal.SIGTERM, previous)


__all__ = ["IDLE_POLLS", "Member", "PoolWorker", "Send", "bisect", "failure_of", "serve_pool", "whole"]
