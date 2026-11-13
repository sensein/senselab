"""A pool worker writes its provenance into the run's `prov/` (ADR-010): per claim, one document per writer."""

import errno
import os
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, cast

import pytest
from lease_clock_fixture import ManualClock
from pool_run_fixture import PROFILE, TASK, WORKERS, Run, executions_in, keys_of, reuses_in
from pool_stack_fixture import packaged, registry
from senselab_core.kernel import prov as kprov
from senselab_core.kernel.identity import LogicalKey
from senselab_core.runtime.call import RunConfig
from senselab_core.runtime.clock import SystemClock
from senselab_core.runtime.execution.job import Abandoned, Emit, Job, JobOutcome, execute
from senselab_core.runtime.execution.pool_worker import PoolWorker
from senselab_core.runtime.execution.queue import QueueSet, probe_queue
from senselab_core.runtime.execution.workers import WorkerSet
from senselab_core.runtime.keys import Placement
from senselab_core.runtime.provenance import PROV_DIR, ProvStore, RunIOError
from senselab_core.runtime.store import ResultStore
from senselab_worker.canonical import jcs_dumps


@pytest.fixture(autouse=True)
def stack(tmp_path: Path) -> Iterator[None]:
    """This module's registry, every env this interpreter under an MIT lock; the cache under `tmp_path`."""
    with packaged(registry(TASK), tmp_path / "cache"):
        yield
    WORKERS.close()


@pytest.mark.contract
def test_a_worker_with_its_own_store_leaves_one_execution_per_item_and_a_reuse_on_rerun(
    tmp_path: Path, clock: ManualClock
) -> None:
    """N items served by a worker whose store is not the coordinator's: N keyed executions; the rerun, N reuses."""
    texts = [f"t{i}" for i in range(4)]
    first = Run(tmp_path, "first", texts)
    first.serve(first.worker_store(), clock)
    read = ProvStore.read(first.run_dir)
    executions = executions_in(read)
    assert len(read.plans()) == 1 and len(executions) == 4 and not reuses_in(read)
    assert len(set(keys_of(executions))) == 4

    again = Run(tmp_path, "again", texts)
    again.serve(again.worker_store(), clock)
    rerun = ProvStore.read(again.run_dir)
    assert len(rerun.plans()) == 1 and len(reuses_in(rerun)) == 4 and not executions_in(rerun)
    assert keys_of(reuses_in(rerun)) == keys_of(executions)


@pytest.mark.contract
def test_a_worker_killed_after_k_written_claims_leaves_those_executions_on_disk(
    tmp_path: Path, clock: ManualClock
) -> None:
    """A kill leaves the disk as an item starts: the first claim's execution is there by the time the next runs."""
    run = Run(tmp_path, "run", [f"t{i}" for i in range(5)])
    on_disk: dict[int, int] = {}

    def killed_before(n: int) -> None:
        on_disk[n] = len(executions_in(ProvStore.read(run.run_dir))) if (run.run_dir / PROV_DIR).exists() else 0

    run.serve(run.worker_store(), clock, before=killed_before)
    assert on_disk[0] == 0 and on_disk[1] == 1 and all(on_disk[n] >= 1 for n in range(1, 5))
    assert len(executions_in(ProvStore.read(run.run_dir))) == 5 and len(ProvStore.read(run.run_dir).plans()) == 1


@pytest.mark.contract
def test_two_workers_write_a_document_per_claim_whose_union_is_every_item_once(
    tmp_path: Path, clock: ManualClock
) -> None:
    """Two workers with their own stores serve one run: a writer document per claim, each item's execution once."""
    run = Run(tmp_path, "run", [f"t{i}" for i in range(5)])
    stores = [run.worker_store(), run.worker_store()]
    workers: list[PoolWorker] = []

    def send(jobs: Sequence[Job], placement: Placement) -> list[JobOutcome]:
        outcomes = execute(jobs, placement, store=stores[0], run=RunConfig(), profile=PROFILE, workers=WORKERS)
        if len(executions_in(stores[0].provenance)) >= 2:
            workers[0].halt()
        return outcomes

    workers.append(PoolWorker(run.pool_dir, ["cpu"], profile=PROFILE, store=stores[0], send=send))
    assert workers[0].serve() == 0
    run.serve(stores[1], clock)
    documents = {p.name for p in (run.run_dir / PROV_DIR).iterdir()}
    assert {f"{s.provenance.writer}.jsonld" for s in stores} <= documents and len(documents) >= len(stores)
    union = ProvStore.read(run.run_dir)
    executions = executions_in(union)
    assert len(union.plans()) == 1 and len(executions) == 5 and len(set(keys_of(executions))) == 5


class _FailingWait(SystemClock):
    """A clock whose wait raises, so a worker with nothing to claim ends in an error of the claim loop."""

    def wait(self, event: threading.Event, timeout: float) -> bool:
        raise RuntimeError("the claim loop failed")


def _held_probe_lease(run: Run, worker: PoolWorker) -> None:
    """Give `worker` a serve lease on a probe queue, as a worker serving one holds it."""
    lease = QueueSet(run.pool_dir, PROFILE, lease=PROFILE.lease).queue(probe_queue("cpu", "p1")).serve_lease()
    assert lease is not None
    worker._serving["cpu" + probe_queue("", "p1")] = lease  # noqa: SLF001 - what serving a probe queue holds


def _lease_released(run: Run) -> bool:
    lease = QueueSet(run.pool_dir, PROFILE, lease=PROFILE.lease).queue(probe_queue("cpu", "p1")).serve_lease()
    if lease is not None:
        lease.release()
    return lease is not None


@pytest.fixture
def unwritable(tmp_path: Path) -> Iterator[Path]:
    """A run whose `prov/` directory refuses a new file."""
    run_dir = tmp_path / "run"
    (run_dir / PROV_DIR).mkdir(parents=True)
    (run_dir / PROV_DIR).chmod(0o500)
    if os.access(run_dir / PROV_DIR, os.W_OK):
        pytest.skip("the directory stays writable (running as root)")
    yield tmp_path
    (run_dir / PROV_DIR).chmod(0o700)


@pytest.mark.contract
def test_a_failed_provenance_write_still_releases_the_serve_leases(unwritable: Path) -> None:
    """The write at serve end follows the lease release, so its failure leaves no lease pinned."""
    run = Run(unwritable, "run", ["t0"], serve_items=False)
    worker = PoolWorker(run.pool_dir, ["cpu"], profile=PROFILE, store=run.worker_store())
    _held_probe_lease(run, worker)
    with pytest.raises(OSError):
        worker.serve()
    assert _lease_released(run)


@pytest.mark.contract
def test_a_failed_provenance_write_does_not_replace_the_claim_loops_error(unwritable: Path) -> None:
    """When the claim loop raised, that error propagates and the write failure is logged; the leases are released."""
    run = Run(unwritable, "run", ["t0"], serve_items=False)
    worker = PoolWorker(run.pool_dir, ["cpu"], profile=PROFILE, store=run.worker_store(), clock=_FailingWait())
    _held_probe_lease(run, worker)
    with pytest.raises(RuntimeError, match="the claim loop failed"):
        worker.serve()
    assert _lease_released(run)


class _KilledAtRunEnd(ProvStore):
    """A worker's provenance whose process its coordinator kills once every item of the run has an outcome.

    The kill comes at once, as at the end of a run: a write made from then on never lands, and the worker is halted.
    """

    def __init__(self, run: Run, digests: Sequence[str]) -> None:
        """Watch the `cpu` queue of `run` for the outcomes of `digests`."""
        super().__init__()
        self.queue = QueueSet(run.pool_dir, PROFILE, lease=PROFILE.lease).queue("cpu")
        self.digests = list(digests)
        self.worker: PoolWorker | None = None
        self.killed = False

    def write(self, run: str | Path) -> None:
        """Write the document unless every item has an outcome; then drop it and halt the worker."""
        if self.killed or all(self.queue.outcome(d) is not None for d in self.digests):
            self.killed = True
            if self.worker is not None:
                self.worker.halt()
            return
        super().write(run)


def _items(run: Run) -> list[str]:
    queue = QueueSet(run.pool_dir, PROFILE, lease=PROFILE.lease).queue("cpu")
    return [digest for digest, _ in queue.live_items()]


def _committed_keys(run: Run, digests: Sequence[str]) -> list[str]:
    queue = QueueSet(run.pool_dir, PROFILE, lease=PROFILE.lease).queue("cpu")
    jobs = [j for d in digests if (record := queue.outcome(d)) is not None for j in cast(list[Any], record["jobs"])]
    return sorted(str(j["key_digest"]) for j in jobs)


def _activity_keys(read: ProvStore) -> list[str]:
    return sorted(
        LogicalKey.model_validate(kprov.literal_value(a.node()["senselab:logicalKey"])).digest()
        for a in read.activities()
    )


@pytest.mark.contract
@pytest.mark.parametrize("stored", [False, True], ids=["executed", "served"])
def test_a_worker_killed_at_run_end_leaves_an_activity_for_every_committed_item(
    tmp_path: Path, clock: ManualClock, stored: bool
) -> None:
    """The coordinator stops the worker as the last outcome lands: each committed key has one execution or Reuse.

    The run's items are executed, or (`stored`) served from results an earlier run left in the store.
    """
    texts = [f"t{i}" for i in range(4)]
    if stored:
        earlier = Run(tmp_path, "earlier", texts)
        earlier.serve(earlier.worker_store(), clock)
    run = Run(tmp_path, "run", texts)
    digests = _items(run)
    provenance = _KilledAtRunEnd(run, digests)
    store = ResultStore(run.root, provenance)

    def send(jobs: Sequence[Job], placement: Placement) -> list[JobOutcome]:
        return execute(jobs, placement, store=store, run=RunConfig(), profile=PROFILE, workers=WORKERS)

    provenance.worker = PoolWorker(run.pool_dir, ["cpu"], profile=PROFILE, store=store, send=send, clock=clock)
    assert provenance.worker.serve() == 0
    committed = _committed_keys(run, digests)
    assert provenance.killed and len(committed) == len(texts)
    assert _activity_keys(ProvStore.read(run.run_dir)) == committed


class _Counting(ProvStore):
    """A worker's provenance that records the activities and bytes of each document it writes."""

    def __init__(self) -> None:
        """Hold nothing yet."""
        super().__init__()
        self.written: list[tuple[int, int]] = []

    def write(self, run: str | Path) -> None:
        """Record the document's activities and size, then write it."""
        self.written.append((len(self.activities()), len(jcs_dumps(self.document()))))
        super().write(run)


def _largest_write(tmp_path: Path, items: int, clock: ManualClock) -> tuple[int, int]:
    run = Run(tmp_path, f"run{items}", [f"t{i:03d}" for i in range(items)])
    provenance = _Counting()
    store = ResultStore(run.root, provenance)

    def send(jobs: Sequence[Job], placement: Placement) -> list[JobOutcome]:
        return execute(jobs, placement, store=store, run=RunConfig(), profile=PROFILE, workers=WORKERS)

    worker = PoolWorker(run.pool_dir, ["cpu"], profile=PROFILE, store=store, send=send, clock=clock)
    served = threading.Thread(target=worker.serve, daemon=True)
    served.start()
    clock.drive(served, PROFILE.pool.poll_s)
    assert len(ProvStore.read(run.run_dir).activities()) == items
    return max(a for a, _ in provenance.written), max(b for _, b in provenance.written)


@pytest.mark.contract
def test_a_workers_provenance_write_does_not_grow_with_the_items_it_committed(
    tmp_path: Path, clock: ManualClock
) -> None:
    """Each claim is its own writer: a document holds at most one claim's activities, however many items committed.

    The whole worker's document would hold every activity so far: 32 at the end of 32 items, four times the 8.
    """
    quantum = PROFILE.pool.quantum_items
    few, many = _largest_write(tmp_path / "few", quantum, clock), _largest_write(tmp_path / "many", 4 * quantum, clock)
    assert few[0] <= quantum and many[0] <= quantum and many[1] <= 1.25 * few[1]


class _FailingWrite(ProvStore):
    """A worker's provenance whose every write fails, as on a full `prov/`."""

    def write(self, run: str | Path) -> None:
        """Fail."""
        raise RunIOError(OSError(errno.ENOSPC, "no space left"))


class _Abandons(WorkerSet):
    """A worker set that records each thread whose request is abandoned."""

    def __init__(self) -> None:
        """Hold one worker."""
        super().__init__(1)
        self.abandoned: list[int] = []

    def abandon(self, thread: int) -> None:
        """Record `thread`, then abandon its request."""
        self.abandoned.append(thread)
        super().abandon(thread)


@pytest.mark.contract
def test_a_failed_write_before_a_commit_stops_the_running_call_before_its_item_is_released(
    tmp_path: Path, clock: ManualClock, hang_s: float
) -> None:
    """A failed provenance write before a commit commits nothing and stops the call still running.

    The call emits its job's outcome and keeps running; the commit's write fails, and the call is cancelled and its
    request abandoned while the item is still under the claim's lease.
    """
    run = Run(tmp_path, "run", ["t0"])
    [digest] = _items(run)
    queue = QueueSet(run.pool_dir, PROFILE, lease=PROFILE.lease).queue("cpu")
    store, workers = ResultStore(run.root, _FailingWrite()), _Abandons()
    seen: dict[str, bool] = {}

    def send(jobs: Sequence[Job], placement: Placement, *, emit: Emit, cancelled: threading.Event) -> list[JobOutcome]:
        done = execute(jobs, placement, store=store, run=RunConfig(), profile=PROFILE, workers=WORKERS)
        emit(range(len(jobs)), done)
        seen["cancelled"] = cancelled.wait(hang_s)
        seen["held"] = (queue.path / "claims" / f"{digest}.lease").exists()
        raise Abandoned(dict(enumerate(done)))

    worker = PoolWorker(run.pool_dir, ["cpu"], profile=PROFILE, store=store, workers=workers, send=send, clock=clock)
    try:
        with pytest.raises(RunIOError):
            worker.serve()
    finally:
        workers.close()
    assert seen == {"cancelled": True, "held": True} and len(workers.abandoned) == 1
    assert queue.outcome(digest) is None and not (queue.path / "claims" / f"{digest}.lease").exists()
