"""Pool workers: a host process serving a run's queues of some resource classes (`senselab execute --pool`).

A pool worker claims items from the queue of the class whose oldest waiting item is oldest, logs each job `claimed`
under the run's current epoch and attempt, runs the jobs through the process's `WorkerSet` (`execute`), and commits
each item with one `ResultRef` per job (`report`); the `Pool` applies each failure's effect. It stays on a class for
a quantum, `pool.quantum_items` items or `pool.quantum_s` seconds, and then claims from the class with the oldest
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
from senselab_core.runtime.execution.workers import WorkerSet
from senselab_core.runtime.keys import Placement
from senselab_core.runtime.planner import Plan
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.store import ResultStore
from senselab_core.runtime.worker_client import WorkerFatal, WorkerLost

logger = logging.getLogger("senselab")

IDLE_POLLS = 3
"""The polls of `pool.poll_s` with nothing to claim after which a pool worker exits."""

Send = Callable[[Sequence[Job], Placement], list[JobOutcome]]
"""Run jobs of one call under a placement and return one outcome per job, in order (`execute`)."""


def whole(exc: BaseException) -> bool:
    """Whether `exc` fails a request as a whole: its worker was lost, fatal or broke the protocol, or ran out."""
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


def bisect(members: Sequence[Member], placement: Placement, send: Send) -> list[JobOutcome]:
    """Return one outcome per member, sent as one request; one that fails as a whole is split in halves.

    A member alone whose request fails as a whole gets that failure; any other exception propagates.
    """
    try:
        return send([m.job for m in members], placement)
    except Exception as exc:
        if not whole(exc):
            raise
        if len(members) == 1:
            logger.warning("job %s fails its request alone: %s", members[0].job.ref, type(exc).__name__)
            return [Failed(members[0].job.ref, members[0].key, failure_of(exc))]
        half = len(members) // 2
        return bisect(members[:half], placement, send) + bisect(members[half:], placement, send)


@dataclass
class _Group:
    """Members of one claim of one plan, node and placement, so one request may carry them."""

    placement: Placement
    run: RunConfig
    members: list[Member]


Finished: TypeAlias = "queue.SimpleQueue[list[tuple[Member, JobOutcome]] | None]"
"""The outcomes of a claim's groups as they finish, then None; an empty batch only wakes the wait."""


class PoolWorker:
    """A pool worker over the queues in `pool_dir` (a run's `queues` directory) of the resource classes `classes`.

    Attributes:
        served: The queue name and item digest of each item claimed, in claim order.
        stop: Set (`halt`) to make the worker commit what finished, release the rest and return.
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
            spent += len(claim.items)
            self.served.extend((claim.queue.name, d) for d in claim.items)
            self.run_claim(claim)
        return 0

    def _by_age(self) -> list[str]:
        """Return the classes with waiting work, the one whose oldest waiting item is oldest first."""
        ages = [(age, n) for n in self.classes if (age := self.queues.queue(n).oldest_age_s()) is not None]
        return [n for _, n in sorted(ages, key=lambda a: -a[0])]

    def _next(self, prefer: str | None, left: int) -> Claim | None:
        """Claim up to `left` items from `prefer`, else a quantum from the class with the oldest work; None if none."""
        order = [prefer] if prefer is not None else []
        order += [n for n in self._by_age() if n != prefer]
        for name in order:
            size = left if name == prefer else self.profile.pool.quantum_items
            claim = self.queues.queue(name).claim(self.name, max_items=max(size, 1))
            if claim is not None:
                return claim
        return None

    def run_claim(self, claim: Claim) -> None:
        """Run every item of `claim`, committing each as its jobs finish; release what is held at the end.

        An item whose jobs cannot be built is given the outcome `dead` with that failure. The jobs run on a thread
        while this one waits, proving the claim's leases every renewal interval; once `stop` is set, what finished
        is committed and the rest released.
        """
        with claim:
            self._log_claimed(claim)
            groups, dead = self._groups(claim)
            for digest, failure in dead.items():
                claim.dead(failure, [digest])
            pending = {d: len(f.jobs) for d, f in zip(claim.items, claim.fragments, strict=True)}
            outcomes: dict[str, dict[int, JobOutcome]] = {d: {} for d in claim.items}
            finished: Finished = queue.SimpleQueue()
            self._finished = finished
            threading.Thread(target=self._run_groups, args=(groups, finished), daemon=True).start()
            interval = self.profile.lease.renew_fraction * self.profile.lease.stale_after_s
            running = True
            while running and claim.items and not self.stop.is_set():
                with contextlib.suppress(queue.Empty):
                    batch = finished.get(timeout=interval)
                    running = batch is not None
                    self._commit(claim, batch or [], outcomes, pending)
                claim.renew()
            while running and not finished.empty():
                batch = finished.get_nowait()
                running = batch is not None
                self._commit(claim, batch or [], outcomes, pending)

    def _commit(
        self,
        claim: Claim,
        batch: list[tuple[Member, JobOutcome]],
        outcomes: dict[str, dict[int, JobOutcome]],
        pending: dict[str, int],
    ) -> None:
        """Record `batch`'s outcomes; commit each item still held whose every job has one."""
        for member, outcome in batch:
            outcomes[member.digest][member.index] = outcome
        for digest in {m.digest for m, _ in batch}:
            if len(outcomes[digest]) == pending[digest] and digest in claim.items:
                report(claim, digest, [outcomes[digest][i] for i in range(pending[digest])])

    def _run_groups(self, groups: list[_Group], finished: "Finished") -> None:
        """Run each group, putting its members' outcomes on `finished`; then put None.

        An exception that is not bisected is the failure of every job of its group.
        """
        try:
            for group in groups:
                send = self._send or functools.partial(self._execute, run=group.run)
                try:
                    outcomes = bisect(group.members, group.placement, send)
                except Exception as exc:  # noqa: BLE001 - a group's failure is each of its jobs' failure
                    outcomes = [Failed(m.job.ref, m.key, failure_of(exc)) for m in group.members]
                finished.put(list(zip(group.members, outcomes, strict=True)))
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


def serve_pool(  # noqa: PLR0913 - the queues and classes, then keyword-only parts and seams
    pool_dir: Path,
    classes: Sequence[str],
    *,
    profile: ExecutionProfile,
    store: ResultStore,
    workers: WorkerSet | None = None,
    send: Send | None = None,
) -> int:
    """Serve the queues of `classes` in `pool_dir` (a run's `queues` directory) as one `PoolWorker`; return 0.

    On the main thread SIGTERM halts the worker (`PoolWorker.halt`); the previous handler is restored after.
    """
    worker = PoolWorker(pool_dir, classes, profile=profile, store=store, workers=workers, send=send)
    if threading.current_thread() is not threading.main_thread():
        return worker.serve()
    previous = signal.signal(signal.SIGTERM, lambda *_: worker.halt())
    try:
        return worker.serve()
    finally:
        signal.signal(signal.SIGTERM, previous)


__all__ = ["IDLE_POLLS", "Member", "PoolWorker", "Send", "bisect", "failure_of", "serve_pool", "whole"]
