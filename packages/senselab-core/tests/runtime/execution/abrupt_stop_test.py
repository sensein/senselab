"""A pool worker stopped without warning at each boundary of its commit path leaves a run a resume completes.

The commit path is claim, execute, provenance write, commit, outcome. A counting pass lists, in order, every edge of
what the worker does through the seams it offers: the request it sends (`send`), the provenance document it writes
(`ProvStore.write`) and, inside a commit (`Claim._finish`), the writes to the queue: the staged outcome, its link at
`outcome/<digest>` and the mark that resolves the slot (`Queue.open` yields a `Watched` directory that reports them;
the lease is released right after the mark). A case stops the worker at one by taking the disk as it is at that instant,
letting the worker end, and putting that disk back, so nothing the worker did afterwards is there, as after a `SIGKILL`.
Its claim leases stay held. A resume then serves the run from a new process, or from another host, once the injected
clock has run past the leases' stale bound. Every worker runs on a `ManualClock`, which is not advanced while the
worker works, so no lease is beaten or judged on wall time and no beat is written while the disk is copied.
"""

import contextlib
import shutil
import sys
import threading
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, cast

import pytest
from host_fixture import HostShape
from lease_clock_fixture import ManualClock
from pool_run_fixture import ONE_ITEM_CLAIMS, TASK, Run
from pool_stack_fixture import packaged, registry
from senselab_core.runtime.call import RunConfig
from senselab_core.runtime.execution.job import Job, JobOutcome, execute
from senselab_core.runtime.execution.pool_worker import PoolWorker
from senselab_core.runtime.execution.provenance_check import check_provenance, committed_keys
from senselab_core.runtime.execution.queue import Queue, QueueSet
from senselab_core.runtime.execution.workers import WorkerSet
from senselab_core.runtime.keys import Placement
from senselab_core.runtime.provenance import ProvStore
from senselab_core.runtime.storage import StorageDir
from senselab_core.runtime.store import ResultStore

pytestmark = pytest.mark.contract
PROFILE = ONE_ITEM_CLAIMS
TEXTS = ["t0", "t1", "t2"]
COUNTS = {
    "send:enter": 3,
    "send:exit": 3,
    "write:enter": 7,
    "write:exit": 7,
    "outcome:staged": 3,
    "outcome:link:enter": 3,
    "outcome:link:exit": 3,
    "slot:mark:enter": 3,
    "slot:mark:exit": 3,
}
"""How often a worker of three one-item claims crosses each boundary. Each claim sends once, writes before its commit,
stages, links and marks, then writes when it ends; the worker writes once more as it ends. The commit's boundaries are
the staged outcome (proven next), its link at `outcome/<digest>` and the slot mark; the lease is released after it."""
BOUNDARIES = [(b, n) for b, count in COUNTS.items() for n in range(count)]


@pytest.fixture(autouse=True)
def stack(tmp_path: Path) -> Iterator[None]:
    """This module's registry, every env this interpreter under an MIT lock; the cache under `tmp_path`."""
    with packaged(registry(TASK), tmp_path / "cache"):
        yield


class Crossings:
    """Counts each boundary a worker crosses and stops it, as a kill does, at the one named (`at`)."""

    def __init__(self, at: tuple[str, int] | None, take: Callable[[], None]) -> None:
        """Stop at the `at[1]`-th crossing (from 0) of `at[0]`, calling `take` as the disk is at that instant."""
        self.at, self.take = at, take
        self.seen: dict[str, int] = {}
        self.order: list[tuple[str, int]] = []
        self.stopped = False
        self.worker: PoolWorker | None = None
        self.guard = threading.Lock()

    def cross(self, boundary: str) -> None:
        """Count one crossing of `boundary`; at the stopping one, take the disk and halt the worker."""
        with self.guard:
            n = self.seen.get(boundary, 0)
            self.seen[boundary] = n + 1
            self.order.append((boundary, n))
            if self.at == (boundary, n) and not self.stopped:
                self.stopped = True
                self.take()
                assert self.worker is not None
                self.worker.halt()


class Marks(ProvStore):
    """A worker's provenance that reports `write:enter` and `write:exit` to a `Crossings`."""

    def __init__(self, crossings: Crossings) -> None:
        """Report to `crossings`."""
        super().__init__()
        self.crossings = crossings

    def write(self, run: str | Path) -> None:
        """Write the document, reporting the boundary before and after."""
        self.crossings.cross("write:enter")
        super().write(run)
        self.crossings.cross("write:exit")


class Watched:
    """A queue directory that reports the writes of a commit to `crossings`, and passes everything else through."""

    def __init__(self, inner: StorageDir, crossings: Crossings, names: tuple[str, ...] = ()) -> None:
        """Wrap `inner`, which is the directory at `names` below the queue's."""
        self._inner, self._crossings, self._names = inner, crossings, names

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 - whatever `StorageDir` offers
        """Pass every other attribute to the wrapped directory."""
        return getattr(self._inner, name)

    def __enter__(self) -> "Watched":
        """Enter the wrapped directory."""
        self._inner.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        """Leave the wrapped directory."""
        self._inner.__exit__(*exc)

    def child(self, name: str, **kwargs: Any) -> "Watched":  # noqa: ANN401 - `StorageDir.child`'s keywords
        """Open the subdirectory `name`, watched."""
        return Watched(self._inner.child(name, **kwargs), self._crossings, (*self._names, name))

    def write_new(self, name: str, data: bytes, mode: int, *, grammar: Any) -> None:  # noqa: ANN401 - a pattern
        """Create the file, reporting a staged outcome and a slot mark."""
        mark = name.startswith("res-")
        if mark:
            self._crossings.cross("slot:mark:enter")
        self._inner.write_new(name, data, mode, grammar=grammar)
        if mark:
            self._crossings.cross("slot:mark:exit")
        elif self._names[-1:] == ("outcome",):
            self._crossings.cross("outcome:staged")

    def link(self, name: str, dst: "Watched", dst_name: str, *, grammar: Any) -> None:  # noqa: ANN401 - a pattern
        """Link the file, reporting the outcome's link before and after."""
        linked = dst._names[-1:] == ("outcome",)  # noqa: SLF001 - this wrapper's own
        if linked:
            self._crossings.cross("outcome:link:enter")
        self._inner.link(name, dst._inner, dst_name, grammar=grammar)  # noqa: SLF001 - this wrapper's own
        if linked:
            self._crossings.cross("outcome:link:exit")


class WatchedQueue(Queue):
    """A queue whose directory reports the commit's writes."""

    crossings: Crossings

    @contextlib.contextmanager
    def open(self) -> Iterator[StorageDir]:
        """Yield the queue's directory, watched."""
        with super().open() as directory:
            yield cast(StorageDir, Watched(directory, self.crossings))


class WatchedQueues(QueueSet):
    """The queues of a run, each a `WatchedQueue`."""

    crossings: Crossings
    watched: dict[str, Queue]

    def queue(self, name: str) -> Queue:
        """Return the watched queue `name`."""
        if name not in self.watched:
            made = WatchedQueue(self.root, name, self.profile, self.lease, self.epoch, self._clock)
            made.crossings = self.crossings
            self.watched[name] = made
        return self.watched[name]


def _serve(
    run: Run, store: ResultStore, clock: ManualClock, workers: WorkerSet, crossings: Crossings | None = None
) -> None:
    def send(jobs: Sequence[Job], placement: Placement) -> list[JobOutcome]:
        if crossings is not None:
            crossings.cross("send:enter")
        done = execute(jobs, placement, store=store, run=RunConfig(), profile=PROFILE, workers=workers)
        if crossings is not None:
            crossings.cross("send:exit")
        return done

    worker = PoolWorker(run.pool_dir, ["cpu"], profile=PROFILE, store=store, send=send, workers=workers, clock=clock)
    if crossings is not None:
        queues = WatchedQueues.for_run(run.run_dir, PROFILE, own=PROFILE.lease, clock=clock)
        assert isinstance(queues, WatchedQueues)
        queues.crossings, queues.watched = crossings, {}
        worker.queues, crossings.worker = queues, worker
    ended: list[int] = []
    thread = threading.Thread(target=lambda: ended.append(worker.serve()), daemon=True)
    thread.start()
    clock.drive(thread, PROFILE.pool.poll_s)
    assert ended == [0]


def _stored(root: Path) -> set[str]:
    """Return the key of every entry the result store at `root` holds."""
    return {d.name for d in root.glob("store@*/*/*")}


def _replace(live: Path, kept: Path) -> None:
    """Put the copy `kept` (none: nothing was there) where `live` is; read-only entries are made writable to go."""

    def writable(function: Any, name: str, error: Any) -> None:  # noqa: ANN401, ARG001 - `shutil`'s handler
        Path(name).parent.chmod(0o700)
        function(name)

    # `onexc` replaced `onerror` in 3.12; the handler takes the same three arguments under either keyword.
    if sys.version_info >= (3, 12):
        shutil.rmtree(live, onexc=writable)
    else:
        shutil.rmtree(live, onerror=writable)
    if kept.exists():
        shutil.copytree(kept, live, symlinks=True)


def _stopped_at(
    tmp_path: Path, lease_clock: Callable[..., ManualClock], boundary: tuple[str, int] | None
) -> tuple[Run, Crossings]:
    """Serve a run of `TEXTS` as a worker that is stopped without warning at `boundary` (None: not stopped)."""
    run = Run(tmp_path, "run", TEXTS, profile=PROFILE)
    snapshot = tmp_path / "snapshot"

    def take() -> None:
        shutil.copytree(run.run_dir, snapshot / "run", symlinks=True)
        if run.root.exists():
            shutil.copytree(run.root, snapshot / "store", symlinks=True)

    crossings = Crossings(boundary, take)
    workers = WorkerSet(1)
    try:
        _serve(run, ResultStore(run.root, Marks(crossings)), lease_clock(start=1000.0), workers, crossings)
    finally:
        workers.close()
    if boundary is not None:
        assert crossings.stopped, f"the worker never crossed {boundary}: {crossings.order}"
        _replace(run.run_dir, snapshot / "run")
        _replace(run.root, snapshot / "store")
    return run, crossings


def test_the_boundaries_are_those_a_worker_of_three_one_item_claims_crosses(
    tmp_path: Path, lease_clock: Callable[..., ManualClock]
) -> None:
    """The counting pass crosses exactly `BOUNDARIES`, so a boundary the commit path gains is a case here."""
    run, crossings = _stopped_at(tmp_path, lease_clock, None)
    assert sorted(crossings.order) == sorted(BOUNDARIES)
    first = [b for b, _ in crossings.order[:10]]
    assert first == [
        "send:enter",
        "send:exit",
        "write:enter",
        "write:exit",
        "outcome:staged",
        "outcome:link:enter",
        "outcome:link:exit",
        "slot:mark:enter",
        "slot:mark:exit",
        "write:enter",
    ]
    keys = committed_keys(run.run_dir)
    assert len(keys) == len(TEXTS) and _stored(run.root) == keys
    check_provenance(run.run_dir, reuses=0).require()


@pytest.mark.parametrize(("boundary", "occurrence"), BOUNDARIES, ids=[f"{b}#{n}" for b, n in BOUNDARIES])
def test_a_worker_stopped_at_a_boundary_leaves_a_run_a_resume_completes(
    tmp_path: Path, lease_clock: Callable[..., ManualClock], host_shape: HostShape, boundary: str, occurrence: int
) -> None:
    """Whatever the stop left: its committed keys each have an execution or a Reuse, and a resume commits the rest.

    The resume (a new process, or one of another host) serves the stored results of uncommitted items as Reuse, so the
    run's Reuse activities are the stored results it found uncommitted; no lease or lock is left.
    """
    run, _ = _stopped_at(tmp_path, lease_clock, (boundary, occurrence))
    kept = committed_keys(run.run_dir)
    check_provenance(run.run_dir).require()
    uncommitted = _stored(run.root) - kept
    clock = lease_clock(start=1000.0, storage_root=tmp_path / "clock")
    clock.advance(2 * PROFILE.lease.stale_after_s)
    with host_shape.moved():
        workers = WorkerSet(1)
        try:
            _serve(run, ResultStore(run.root), clock, workers)
        finally:
            workers.close()
    assert len(committed_keys(run.run_dir)) == len(TEXTS)
    check_provenance(run.run_dir, reuses=len(uncommitted)).require()
    assert not QueueSet(run.pool_dir, PROFILE, lease=PROFILE.lease, clock=clock).claimed_any()
    assert not list(run.run_dir.rglob("*.lock"))
