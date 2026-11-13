"""A pool worker stopped without warning at each boundary of its commit path leaves a run a resume completes.

The boundaries are the edges of the two calls the commit path makes through seams the worker offers: the request it
sends (`send`) and the provenance document it writes (`ProvStore.write`). A counting pass lists them in order; a case
stops the worker at one by taking the disk as it is at that instant, letting the worker end, and putting that disk
back, so nothing the worker did afterwards is there, as after a `SIGKILL`. Its claim leases are left held, stamped
as they were. A resume then serves the run from a new process, or from another host, once those leases are stale.
"""

import os
import shutil
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from host_fixture import HostShape
from lease_clock_fixture import age
from pool_run_fixture import ONE_ITEM_CLAIMS, TASK, Run
from pool_stack_fixture import packaged, registry
from senselab_core.runtime.call import RunConfig
from senselab_core.runtime.execution.job import Job, JobOutcome, execute
from senselab_core.runtime.execution.pool_worker import PoolWorker
from senselab_core.runtime.execution.provenance_check import check_provenance, committed_keys
from senselab_core.runtime.execution.queue import RUN_QUEUES, QueueSet
from senselab_core.runtime.execution.workers import WorkerSet
from senselab_core.runtime.keys import Placement
from senselab_core.runtime.provenance import ProvStore
from senselab_core.runtime.store import ResultStore

pytestmark = pytest.mark.contract
PROFILE = ONE_ITEM_CLAIMS
TEXTS = ["t0", "t1", "t2"]
BOUNDARIES = (
    [("send:enter", n) for n in range(3)]
    + [("send:exit", n) for n in range(3)]
    + [("write:enter", n) for n in range(7)]
    + [("write:exit", n) for n in range(7)]
)
"""Each boundary a worker of three one-item claims crosses, with its occurrence: three claims make a request and two
writes each (one before its commit, one when it ends), and the worker writes once more as it ends. The write that ends
the last claim comes right after the last commit."""


@pytest.fixture(autouse=True)
def stack(tmp_path: Path) -> Any:  # noqa: ANN401 - a generator fixture
    """This module's registry, every env this interpreter under an MIT lock; the cache under `tmp_path`."""
    with packaged(registry(TASK), tmp_path / "cache"):
        yield


class Marks(ProvStore):
    """A worker's provenance that reports `write:enter` and `write:exit` to a `Crossings`."""

    def __init__(self, crossings: "Crossings") -> None:
        """Report to `crossings`."""
        super().__init__()
        self.crossings = crossings

    def write(self, run: str | Path) -> None:
        """Write the document, reporting the boundary before and after."""
        self.crossings.cross("write:enter")
        super().write(run)
        self.crossings.cross("write:exit")


class Crossings:
    """Counts each boundary a worker crosses and stops it, as a kill does, at the one named (`at`)."""

    def __init__(self, at: tuple[str, int] | None, take: Any) -> None:  # noqa: ANN401 - the disk-taking callable
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


def _serve(run: Run, store: ResultStore, workers: WorkerSet, crossings: Crossings | None = None) -> PoolWorker:
    def send(jobs: Sequence[Job], placement: Placement) -> list[JobOutcome]:
        if crossings is not None:
            crossings.cross("send:enter")
        done = execute(jobs, placement, store=store, run=RunConfig(), profile=PROFILE, workers=workers)
        if crossings is not None:
            crossings.cross("send:exit")
        return done

    worker = PoolWorker(run.pool_dir, ["cpu"], profile=PROFILE, store=store, send=send, workers=workers)
    if crossings is not None:
        crossings.worker = worker
    assert worker.serve() == 0
    return worker


def _stored(root: Path) -> set[str]:
    """Return the key of every entry the result store at `root` holds."""
    return {d.name for d in root.glob("store@*/*/*")}


def _replace(live: Path, kept: Path) -> None:
    """Put the copy `kept` (none: nothing was there) where `live` is; read-only entries are made writable to go."""

    def writable(function: Any, name: str, error: BaseException) -> None:  # noqa: ANN401 - `shutil`'s handler
        os.chmod(Path(name).parent, 0o700)
        function(name)

    shutil.rmtree(live, onexc=writable)
    if kept.exists():
        shutil.copytree(kept, live, symlinks=True)


def _stopped_at(tmp_path: Path, boundary: tuple[str, int] | None) -> tuple[Run, Crossings]:
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
        _serve(run, ResultStore(run.root, Marks(crossings)), workers, crossings)
    finally:
        workers.close()
    if boundary is not None:
        assert crossings.stopped, f"the worker never crossed {boundary}: {crossings.order}"
        _replace(run.run_dir, snapshot / "run")
        _replace(run.root, snapshot / "store")
        for claims in (run.run_dir / RUN_QUEUES).glob("*/claims"):
            age(claims, 2 * PROFILE.lease.stale_after_s)
    return run, crossings


def test_the_boundaries_are_those_a_worker_of_three_one_item_claims_crosses(tmp_path: Path) -> None:
    """The counting pass crosses exactly `BOUNDARIES`, so a boundary the commit path gains is a case here."""
    run, crossings = _stopped_at(tmp_path, None)
    assert sorted(crossings.order) == sorted(BOUNDARIES)
    keys = committed_keys(run.run_dir)
    assert len(keys) == len(TEXTS) and _stored(run.root) == keys
    check_provenance(run.run_dir, reuses=0).require()


@pytest.mark.parametrize(("boundary", "occurrence"), BOUNDARIES, ids=[f"{b}#{n}" for b, n in BOUNDARIES])
def test_a_worker_stopped_at_a_boundary_leaves_a_run_a_resume_completes(
    tmp_path: Path, host_shape: HostShape, boundary: str, occurrence: int
) -> None:
    """Whatever the stop left: its committed keys each have one execution or a Reuse, and a resume commits the rest.

    The resume (a new process, or one of another host) serves the stored results of uncommitted items as Reuse, so the
    run's Reuse activities are the stored results it found uncommitted; no lease or lock is left, and no key is
    executed twice.
    """
    run, _ = _stopped_at(tmp_path, (boundary, occurrence))
    kept = committed_keys(run.run_dir)
    check_provenance(run.run_dir).require()
    uncommitted = _stored(run.root) - kept
    with host_shape.moved():
        workers = WorkerSet(1)
        try:
            _serve(run, ResultStore(run.root), workers)
        finally:
            workers.close()
    assert len(committed_keys(run.run_dir)) == len(TEXTS)
    check_provenance(run.run_dir, reuses=len(uncommitted)).require()
    assert not QueueSet(run.pool_dir, PROFILE, lease=PROFILE.lease).claimed_any()
    assert not list(run.run_dir.rglob("*.lock"))
