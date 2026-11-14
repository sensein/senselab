"""A run's lease: every judge of a run's leases and every holder reads the bound the run's epochs wrote."""

import ast
import json
import os
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from epoch_fixture import prepared
from pool_stack_fixture import packaged, planned, registry, scoring
from queue_race_fixture import fragment
from senselab_core.runtime.call import RunConfig
from senselab_core.runtime.execution.epoch import (
    Epoch,
    LeaseMismatch,
    RunLeaseUnreadable,
    combined,
    run_lease,
)
from senselab_core.runtime.execution.executor import Admission
from senselab_core.runtime.execution.in_process import node_fragment
from senselab_core.runtime.execution.job import Job, JobOutcome, execute
from senselab_core.runtime.execution.pool_worker import PoolWorker
from senselab_core.runtime.execution.queue import RUN_QUEUES, Queue, QueueSet
from senselab_core.runtime.execution.run import Run
from senselab_core.runtime.execution.workers import WorkerSet
from senselab_core.runtime.execution_settings import JobLease
from senselab_core.runtime.keys import Placement
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.store import ResultStore
from senselab_core.testing.leases import packaged_lease

pytestmark = pytest.mark.logic

SLOW = packaged_lease(3600.0, 0.3, 0.0)
FAST = packaged_lease(300.0, 0.2, 1.0)
AGE_S = 400.0
TASK = scoring("run_lease_test.score")
WORKERS = WorkerSet(1)


def _epochs(run_dir: Path, *leases: JobLease) -> None:
    """Take and bind one epoch per lease section, oldest first."""
    epochs = Epoch(run_dir)
    for lease in leases:
        n = epochs.take(lease)
        epochs.bind(n, f"a{n}", n)
    (run_dir / RUN_QUEUES).mkdir(parents=True, exist_ok=True)


def _dead(lease: Path) -> Path:
    """Plant a lease whose holder stopped beating `AGE_S` seconds ago; return its token."""
    token = lease / uuid.uuid4().hex
    token.mkdir(parents=True)
    stamp = time.time() - AGE_S
    for path in (token, lease):
        os.utime(path, (stamp, stamp))
    return token


def test_every_judge_reads_the_runs_bound_not_the_processs(tmp_path: Path) -> None:
    """A claim 400 s old under a run bound of 3600 s is live to the queues, the worker's claim and the epoch."""
    run_dir = tmp_path / "run"
    _epochs(run_dir, SLOW)
    queues = QueueSet.for_run(run_dir)
    assert ExecutionProfile.load().lease.stale_after_s < AGE_S < queues.lease.stale_after_s
    queue = queues.queue("cpu")
    item = fragment("x")
    queue.put(item)
    (digest,) = (d for d, _ in queue.live_items())
    _dead(queue.path / "claims" / f"{digest}.lease")
    _dead(run_dir / "epoch" / "1" / "coordinator.lease")
    fresh = QueueSet.for_run(run_dir)
    assert fresh.claimed_any() and fresh.queue("cpu").claimed(digest)
    assert fresh.queue("cpu").oldest_age_s() is None and fresh.queue("cpu").claim("worker") is None
    assert not Epoch(run_dir).stopped()


def test_a_run_whose_coordinator_ended_is_not_interrupted_while_a_claim_is_live(tmp_path: Path) -> None:
    """`Run.wait` does not give up on a run a worker still holds a claim of, by the run's bound."""
    run_dir = tmp_path / "run"
    _epochs(run_dir, SLOW)
    queue = QueueSet.for_run(run_dir).queue("cpu")
    queue.put(fragment("x"))
    (digest, _) = queue.live_items()[0]
    _dead(queue.path / "claims" / f"{digest}.lease")
    (run_dir / "epoch" / "1" / "ended").write_bytes(b"")
    assert not Run(run_dir).wait(timeout=0.05, poll=0.01).interrupted


def test_the_run_lease_combines_by_the_longest_bound_and_the_shortest_renewal(tmp_path: Path) -> None:
    """Epochs bound at 3600 s and at 300 s judge by 3600 s; renewal is the smaller fraction; `own` is one more epoch."""
    run_dir = tmp_path / "run"
    _epochs(run_dir, SLOW, FAST)
    found = run_lease(run_dir)
    assert (found.stale_after_s, found.renew_fraction, found.attribute_delay_s) == (3600.0, 0.2, 1.0)
    longer = packaged_lease(7200.0, 0.1, 0.0)
    assert run_lease(run_dir, own=longer).stale_after_s == 7200.0
    queue = QueueSet.for_run(run_dir).queue("cpu")
    queue.put(fragment("x"))
    (digest, _) = queue.live_items()[0]
    _dead(queue.path / "claims" / f"{digest}.lease")
    assert queue.claim("w") is None


def test_an_epoch_that_never_bound_is_not_part_of_the_runs_lease(tmp_path: Path) -> None:
    """A taken epoch with no `attempt.json` leaves the run's bound as it was."""
    run_dir = tmp_path / "run"
    _epochs(run_dir, FAST)
    Epoch(run_dir).take(SLOW)
    assert run_lease(run_dir).stale_after_s == FAST.stale_after_s


def test_a_lease_section_of_another_build_is_refused(tmp_path: Path) -> None:
    """A bound epoch whose `lease.json` holds two keys, or no format, is unreadable: the run is not resumable."""
    run_dir = tmp_path / "run"
    _epochs(run_dir, SLOW)
    path = run_dir / "epoch" / "1" / "lease.json"
    path.write_text(json.dumps({"stale_after_s": 300.0, "attribute_delay_s": 0.0}))
    with pytest.raises(RunLeaseUnreadable, match="lease.json"):
        run_lease(run_dir)
    path.write_text(json.dumps({"format": "run-lease@2", "lease": {"stale_after_s": 300.0}}))
    with pytest.raises(RunLeaseUnreadable):
        QueueSet.for_run(run_dir)


def test_a_run_lease_at_the_old_tag_is_refused_naming_both_tags(tmp_path: Path) -> None:
    """A `lease.json` written as `run-lease@1` is refused with the tag it has and the one this build reads."""
    run_dir = tmp_path / "run"
    _epochs(run_dir, SLOW)
    path = run_dir / "epoch" / "1" / "lease.json"
    path.write_text(json.dumps({"format": "run-lease@1", "lease": SLOW.model_dump(mode="json")}))
    with pytest.raises(RunLeaseUnreadable, match=r"run-lease@1.*run-lease@2"):
        run_lease(run_dir)


def _worker(
    tmp_path: Path, profile: ExecutionProfile, send: Callable[[], None] | None = None
) -> tuple[PoolWorker, Path, Queue]:
    placed, _ = planned(TASK, profile, ["a", "b"])
    store = ResultStore(tmp_path / "store")
    prepared(placed, tmp_path / "run", store)
    pool_dir = tmp_path / "run" / RUN_QUEUES
    admission = Admission(id=f"run-{placed.hash}", plan_hash=placed.hash)
    queue = QueueSet.for_run(tmp_path / "run", profile).queue("cpu")
    for element in range(2):
        queue.put(node_fragment(placed, admission, elements=[(element,)]))

    def run(jobs: Sequence[Job], placement: Placement) -> list[JobOutcome]:
        if send is not None:
            send()
        return execute(jobs, placement, store=store, run=RunConfig(), profile=profile, workers=WORKERS)

    return PoolWorker(pool_dir, ["cpu"], profile=profile, store=store, send=run), tmp_path / "run", queue


@pytest.fixture(autouse=True)
def stack(tmp_path: Path) -> object:
    """This module's registry under `tmp_path`'s cache."""
    with packaged(registry(TASK), tmp_path / "cache"):
        yield
    WORKERS.close()


def test_a_worker_whose_lease_section_is_not_the_runs_refuses_to_start(tmp_path: Path) -> None:
    """The run was bound at 3600 s; a worker under the 300 s profile raises `LeaseMismatch` naming the key."""
    profile = ExecutionProfile.load("laptop", {"devices": ["cpu"]})
    placed, _ = planned(TASK, profile, ["a"])
    prepared(placed, tmp_path / "run", ResultStore(tmp_path / "store"))
    section = tmp_path / "run" / "epoch" / "1" / "lease.json"
    document = json.loads(section.read_text())
    document["lease"].update({"stale_after_s": 3600.0, "wait_s": 7200.0})
    section.write_text(json.dumps(document))
    worker = PoolWorker(tmp_path / "run" / RUN_QUEUES, ["cpu"], profile=profile, store=ResultStore(tmp_path / "store"))
    with pytest.raises(LeaseMismatch) as raised:
        worker.serve()
    assert raised.value.keys == ("stale_after_s", "wait_s")


def test_a_worker_stops_claiming_when_a_newer_epoch_binds_another_section(tmp_path: Path) -> None:
    """Epoch 2 binds a different section while item 1 runs: item 1 commits, item 2 is not claimed, `LeaseMismatch`."""
    profile = ExecutionProfile.load("laptop", {"devices": ["cpu"]})
    run_dir = tmp_path / "run"

    def newer() -> None:
        epochs = Epoch(run_dir)
        if epochs.current() == 1:
            n = epochs.take(SLOW)
            epochs.bind(n, "a2", 2)

    worker, _, queue = _worker(tmp_path, profile, newer)
    with pytest.raises(LeaseMismatch):
        worker.serve()
    assert len(worker.served) == 1 and queue.depth() == 1


def test_no_queue_set_is_built_outside_for_run() -> None:
    """Source builds queues only through `QueueSet.for_run`: no call of the plain constructor."""
    root = next(p for p in Path(__file__).resolve().parents if p.name == "packages")
    found = []
    for path in root.glob("*/src/**/*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "QueueSet":
                found.append(f"{path.name}:{node.lineno}")
    assert found == []


def test_a_resume_under_a_shorter_profile_keeps_the_longest_wait(tmp_path: Path) -> None:
    """The run's lease takes the longest `wait_s` of its epochs, and never less than its stale bound plus delay."""
    run_dir = tmp_path / "run"
    long = packaged_lease(1200.0, 0.2, 60.0, wait_s=2400.0)
    short = packaged_lease(300.0, 0.2, 60.0, wait_s=600.0)
    _epochs(run_dir, long)
    found = run_lease(run_dir, own=short)
    assert found.stale_after_s == 1200.0 and found.wait_s == 2400.0
    assert (
        combined(
            [packaged_lease(1200.0, 0.2, 0.0, wait_s=1200.0), packaged_lease(300.0, 0.2, 60.0, wait_s=400.0)]
        ).wait_s
        == 1260.0
    )
