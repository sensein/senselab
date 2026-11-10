"""A pool worker writes its provenance into the run's `prov/` (ADR-010): per claim, one document per writer."""

import json
import os
import threading
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import pytest
from pool_stack_fixture import packaged, planned, registry, scoring
from senselab_core.kernel import prov as kprov
from senselab_core.runtime.call import RunConfig
from senselab_core.runtime.clock import SystemClock
from senselab_core.runtime.execution.executor import Admission
from senselab_core.runtime.execution.in_process import node_fragment
from senselab_core.runtime.execution.job import Job, JobOutcome, execute
from senselab_core.runtime.execution.pool_worker import PoolWorker
from senselab_core.runtime.execution.queue import RUN_QUEUES, QueueSet, probe_queue
from senselab_core.runtime.execution.run import prepare
from senselab_core.runtime.execution.workers import WorkerSet
from senselab_core.runtime.keys import Placement
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.provenance import PROV_DIR, ProvStore
from senselab_core.runtime.store import ResultStore

TASK = scoring("pool_provenance_test.score")
WORKERS = WorkerSet(2)
PROFILE = ExecutionProfile.load(
    "laptop",
    {
        "devices": ["cpu"],
        "resource_classes": {"cpu": {"device": "cpu", "mem_bytes": 2**30, "max_workers": 1}},
        "pool": {"poll_s": 0.05, "first_poll_s": 0.01, "quantum_items": 8, "quantum_s": 60},
        "lease": {"stale_after_s": 0.4, "renew_fraction": 0.2, "attribute_delay_s": 0},
    },
)


@pytest.fixture(autouse=True)
def stack(tmp_path: Path) -> Iterator[None]:
    """This module's registry, every env this interpreter under an MIT lock; the cache under `tmp_path`."""
    with packaged(registry(TASK), tmp_path / "cache"):
        yield
    WORKERS.close()


class Run:
    """A run of one `TASK` item per text, its items on the `cpu` queue; its workers each own a `ResultStore`."""

    def __init__(self, tmp_path: Path, name: str, texts: Sequence[str], *, serve_items: bool = True) -> None:
        """Prepare run `name` over `texts`, its results in `tmp_path/store` (shared by every run of a test)."""
        self.placed, _ = planned(TASK, PROFILE, texts)
        self.root = tmp_path / "store"
        self.run_dir = tmp_path / name
        prepare(self.placed, self.run_dir, ResultStore(self.root))
        self.pool_dir = self.run_dir / RUN_QUEUES
        admission = Admission(id=f"run-{self.placed.hash}", plan_hash=self.placed.hash)
        for element in range(len(texts) if serve_items else 0):
            fragment = node_fragment(self.placed, admission, elements=[(element,)])
            placement = fragment.placement.model_copy(update={"resource_class": "cpu", "why": f"item {element}"})
            QueueSet(self.pool_dir, PROFILE).queue("cpu").put(fragment.model_copy(update={"placement": placement}))

    def serve(self, store: ResultStore, before: Callable[[int], None] | None = None) -> None:
        """Serve the queue with a worker of `store`; `before(n)` runs as the n-th item (from 0) is about to execute."""
        seen: list[int] = []

        def send(jobs: Sequence[Job], placement: Placement) -> list[JobOutcome]:
            if before is not None:
                before(len(seen))
            seen.append(len(jobs))
            return execute(jobs, placement, store=store, run=RunConfig(), profile=PROFILE, workers=WORKERS)

        assert PoolWorker(self.pool_dir, ["cpu"], profile=PROFILE, store=store, send=send).serve() == 0

    def worker_store(self) -> ResultStore:
        """Return a store over the run's results with its own provenance, as a worker process builds."""
        return ResultStore(self.root)


def _executions(read: ProvStore) -> list[kprov.Record]:
    return [a for a in read.activities() if "senselab:Reuse" not in a.types()]


def _reuses(read: ProvStore) -> list[kprov.Record]:
    return [a for a in read.activities() if "senselab:Reuse" in a.types()]


def _keys(records: Sequence[kprov.Record]) -> list[str]:
    return sorted(json.dumps(kprov.literal_value(r.node()["senselab:logicalKey"]), sort_keys=True) for r in records)


@pytest.mark.contract
def test_a_worker_with_its_own_store_leaves_one_execution_per_item_and_a_reuse_on_rerun(tmp_path: Path) -> None:
    """N items served by a worker whose store is not the coordinator's: N keyed executions; the rerun, N reuses."""
    texts = [f"t{i}" for i in range(4)]
    first = Run(tmp_path, "first", texts)
    first.serve(first.worker_store())
    read = ProvStore.read(first.run_dir)
    executions = _executions(read)
    assert len(read.plans()) == 1 and len(executions) == 4 and not _reuses(read)
    assert len(set(_keys(executions))) == 4

    again = Run(tmp_path, "again", texts)
    again.serve(again.worker_store())
    rerun = ProvStore.read(again.run_dir)
    assert len(rerun.plans()) == 1 and len(_reuses(rerun)) == 4 and not _executions(rerun)
    assert _keys(_reuses(rerun)) == _keys(executions)


@pytest.mark.contract
def test_a_worker_killed_after_k_written_claims_leaves_those_executions_on_disk(tmp_path: Path) -> None:
    """A kill leaves the disk as an item starts: the first claim's execution is there by the time the next runs."""
    run = Run(tmp_path, "run", [f"t{i}" for i in range(5)])
    on_disk: dict[int, int] = {}

    def killed_before(n: int) -> None:
        on_disk[n] = len(_executions(ProvStore.read(run.run_dir))) if (run.run_dir / PROV_DIR).exists() else 0

    run.serve(run.worker_store(), before=killed_before)
    assert on_disk[0] == 0 and on_disk[1] == 1 and all(on_disk[n] >= 1 for n in range(1, 5))
    assert len(_executions(ProvStore.read(run.run_dir))) == 5 and len(ProvStore.read(run.run_dir).plans()) == 1


@pytest.mark.contract
def test_two_workers_write_two_documents_whose_union_is_every_item_once(tmp_path: Path) -> None:
    """Two workers with their own stores serve one run: two writer documents, each item's execution once."""
    run = Run(tmp_path, "run", [f"t{i}" for i in range(5)])
    stores = [run.worker_store(), run.worker_store()]
    workers: list[PoolWorker] = []

    def send(jobs: Sequence[Job], placement: Placement) -> list[JobOutcome]:
        outcomes = execute(jobs, placement, store=stores[0], run=RunConfig(), profile=PROFILE, workers=WORKERS)
        if len(_executions(stores[0].provenance)) >= 2:
            workers[0].halt()
        return outcomes

    workers.append(PoolWorker(run.pool_dir, ["cpu"], profile=PROFILE, store=stores[0], send=send))
    assert workers[0].serve() == 0
    run.serve(stores[1])
    documents = sorted(p.name for p in (run.run_dir / PROV_DIR).iterdir())
    assert documents == sorted(f"{s.provenance.writer}.jsonld" for s in stores)
    union = ProvStore.read(run.run_dir)
    executions = _executions(union)
    assert len(union.plans()) == 1 and len(executions) == 5 and len(set(_keys(executions))) == 5


class _FailingWait(SystemClock):
    """A clock whose wait raises, so a worker with nothing to claim ends in an error of the claim loop."""

    def wait(self, event: threading.Event, timeout: float) -> bool:
        raise RuntimeError("the claim loop failed")


def _held_probe_lease(run: Run, worker: PoolWorker) -> None:
    """Give `worker` a serve lease on a probe queue, as a worker serving one holds it."""
    lease = QueueSet(run.pool_dir, PROFILE).queue(probe_queue("cpu", "p1")).serve_lease()
    assert lease is not None
    worker._serving["cpu" + probe_queue("", "p1")] = lease  # noqa: SLF001 - what serving a probe queue holds


def _lease_released(run: Run) -> bool:
    lease = QueueSet(run.pool_dir, PROFILE).queue(probe_queue("cpu", "p1")).serve_lease()
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
