"""A run of `TASK` items served by real `PoolWorker`s over one run directory: the pool provenance tests' scaffold."""

import json
import threading
from collections.abc import Callable, Sequence
from pathlib import Path

from epoch_fixture import prepared
from lease_clock_fixture import ManualClock
from pool_stack_fixture import planned, scoring
from senselab_core.kernel import prov as kprov
from senselab_core.runtime.call import RunConfig
from senselab_core.runtime.execution.executor import Admission
from senselab_core.runtime.execution.in_process import node_fragment
from senselab_core.runtime.execution.job import Job, JobOutcome, execute
from senselab_core.runtime.execution.pool_worker import PoolWorker
from senselab_core.runtime.execution.queue import RUN_QUEUES, QueueSet
from senselab_core.runtime.execution.workers import WorkerSet
from senselab_core.runtime.keys import Placement
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.provenance import ProvStore
from senselab_core.runtime.store import ResultStore

TASK = scoring("pool_run_fixture.score")
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


ONE_ITEM_CLAIMS = ExecutionProfile.load(
    "laptop",
    {
        "devices": ["cpu"],
        "resource_classes": {"cpu": {"device": "cpu", "mem_bytes": 2**30, "max_workers": 1}},
        "pool": {"poll_s": 0.05, "first_poll_s": 0.01, "quantum_items": 1, "quantum_s": 60},
        "lease": {"stale_after_s": 0.4, "renew_fraction": 0.2, "attribute_delay_s": 0},
    },
)
"""`PROFILE` with a claim of one item, so each item has its own claim, execution, write and commit."""


class Run:
    """A run of one `TASK` item per text, its items on the `cpu` queue; its workers each own a `ResultStore`."""

    def __init__(
        self,
        tmp_path: Path,
        name: str,
        texts: Sequence[str],
        *,
        serve_items: bool = True,
        profile: ExecutionProfile = PROFILE,
    ) -> None:
        """Prepare run `name` over `texts` under `profile`; results are in `tmp_path/store`."""
        self.profile = profile
        self.placed, _ = planned(TASK, profile, texts)
        self.root = tmp_path / "store"
        self.run_dir = tmp_path / name
        prepared(self.placed, self.run_dir, ResultStore(self.root))
        self.pool_dir = self.run_dir / RUN_QUEUES
        admission = Admission(id=f"run-{self.placed.hash}", plan_hash=self.placed.hash)
        for element in range(len(texts) if serve_items else 0):
            fragment = node_fragment(self.placed, admission, elements=[(element,)])
            placement = fragment.placement.model_copy(update={"resource_class": "cpu", "why": f"item {element}"})
            QueueSet(self.pool_dir, profile, lease=profile.lease).queue("cpu").put(
                fragment.model_copy(update={"placement": placement})
            )

    def serve(self, store: ResultStore, clock: ManualClock, before: Callable[[int], None] | None = None) -> None:
        """Serve the queue with a worker of `store` on `clock`; `before(n)` runs as item n starts."""
        seen: list[int] = []

        def send(jobs: Sequence[Job], placement: Placement) -> list[JobOutcome]:
            if before is not None:
                before(len(seen))
            seen.append(len(jobs))
            return execute(jobs, placement, store=store, run=RunConfig(), profile=self.profile, workers=WORKERS)

        worker = PoolWorker(self.pool_dir, ["cpu"], profile=self.profile, store=store, send=send, clock=clock)
        ended: list[int] = []
        thread = threading.Thread(target=lambda: ended.append(worker.serve()), daemon=True)
        thread.start()
        clock.drive(thread, self.profile.pool.poll_s)
        assert ended == [0]

    def worker_store(self) -> ResultStore:
        """Return a store over the run's results with its own provenance, as a worker process builds."""
        return ResultStore(self.root)


def executions_in(read: ProvStore) -> list[kprov.Record]:
    """Return the execution activities of `read`."""
    return [a for a in read.activities() if "senselab:Reuse" not in a.types()]


def reuses_in(read: ProvStore) -> list[kprov.Record]:
    """Return the Reuse activities of `read`."""
    return [a for a in read.activities() if "senselab:Reuse" in a.types()]


def activities_of(run_dir: Path) -> tuple[list[kprov.Record], list[kprov.Record]]:
    """Return the executions and the Reuse of the run at `run_dir`."""
    read = ProvStore.read(run_dir)
    return executions_in(read), reuses_in(read)


def keys_of(records: Sequence[kprov.Record]) -> list[str]:
    """Return the sorted logical keys of `records`, one string each."""
    return sorted(json.dumps(kprov.literal_value(r.node()["senselab:logicalKey"]), sort_keys=True) for r in records)
