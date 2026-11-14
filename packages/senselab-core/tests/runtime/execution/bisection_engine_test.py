"""One bisection engine for both executors: a job that keeps crashing fresh workers is charged to the item.

The workers are fakes over the real host worker (`faulty_workers_fixture`): a request carrying a poisoned text raises
`WorkerLost` and discards that worker, so the next checkout starts another, and each start is counted. Nothing is
timed.
"""

import ast
import threading
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from epoch_fixture import prepared
from faulty_workers_fixture import FOREVER, crashes, put_batched, sender
from in_process_bisect_test import BATCH, _planned, _run, _texts
from lease_clock_fixture import ManualClock
from pool_probe_test import TEXTY
from pool_stack_fixture import packaged, registry
from pool_worker_test import _outcomes
from pool_worker_test import _served as _clocked
from senselab_core.runtime.call import RunConfig
from senselab_core.runtime.execution.executor import ItemCrashedWorker, ResultRef
from senselab_core.runtime.execution.in_process import InProcess, fragment_jobs, node_fragment
from senselab_core.runtime.execution.job import Abandoned, Completed, Failed, JobOutcome, execute
from senselab_core.runtime.execution.pool_worker import PoolWorker
from senselab_core.runtime.execution.queue import RUN_QUEUES
from senselab_core.runtime.execution.workers import WorkerSet
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.store import ResultStore
from senselab_core.runtime.worker_client import WorkerFatal, WorkerLost
from senselab_worker.faults import ProtocolError

pytestmark = pytest.mark.logic


@pytest.fixture(autouse=True)
def stack(tmp_path: Path) -> Iterator[None]:
    """The text task served by the default scoring backend, which accepts any batch; its cache under `tmp_path`."""
    with packaged(registry(TEXTY), tmp_path / "cache"):
        yield


def _profile(tmp_path: Path, **faults: int) -> ExecutionProfile:
    override = {
        "devices": ["cpu"],
        "resource_classes": {"cpu": {"device": "cpu", "mem_bytes": 2**33, "max_workers": 2}},
        "calibration": {"measures_roots": [str(tmp_path / "measures")], "warmup_requests": 1},
        "pool": {"poll_s": 0.05, "first_poll_s": 0.01, "quantum_items": BATCH, "quantum_s": 60},
        "lease": {"stale_after_s": 0.4, "renew_fraction": 0.2, "attribute_delay_s": 0},
        "faults": faults,
    }
    return ExecutionProfile.load("laptop", override)


def _failed(outcomes: Sequence[JobOutcome]) -> list[Failed]:
    return [o for o in outcomes if isinstance(o, Failed)]


def test_an_item_that_crashes_every_worker_is_charged_after_two_and_the_rest_succeed(tmp_path: Path) -> None:
    """In-process: the poisoned job is `ItemCrashedWorker`, content, charged; its eleven batch-mates complete."""
    workers = crashes({"p07": FOREVER})
    outcomes = _run(tmp_path, _profile(tmp_path), _texts(BATCH, at={7: "p07"}), workers)
    [dead] = _failed(outcomes)
    assert (dead.failure.fault, dead.failure.type) == ("content", "ItemCrashedWorker")
    assert dead.failure.charged and not dead.failure.gap
    assert sum(isinstance(o, Completed) for o in outcomes) == BATCH - 1
    assert workers.failed[0] == BATCH and workers.failed.count(1) == 2
    assert workers.starts == len(workers.failed) + 1


@pytest.mark.parametrize("charged", [1, 3])
def test_the_count_is_the_profiles_key(tmp_path: Path, charged: int) -> None:
    """A job alone is lost exactly `faults.crashes_charged` times before it is charged."""
    workers = crashes({"p00": FOREVER})
    [dead] = _run(tmp_path, _profile(tmp_path, crashes_charged=charged), ["p00"], workers)
    assert isinstance(dead, Failed) and dead.failure.type == "ItemCrashedWorker"
    assert workers.sent == [1] * charged and workers.starts == charged


@pytest.mark.parametrize("error", [WorkerLost, ProtocolError])
def test_a_crash_another_request_survives_is_not_charged(tmp_path: Path, error: type[Exception]) -> None:
    """The first request is lost once; its halves and every job alone complete on fresh workers: no failure at all."""
    workers = crashes({"p03": 1}, lambda: error("died"))
    outcomes = _run(tmp_path, _profile(tmp_path), _texts(BATCH, at={3: "p03"}), workers)
    assert all(isinstance(o, Completed) for o in outcomes) and len(outcomes) == BATCH
    assert len(workers.failed) == 1 and workers.starts == 2 and workers.sent.count(BATCH) == 1


def test_a_lone_job_lost_once_and_then_completing_is_not_charged(tmp_path: Path) -> None:
    """One job lost on its worker is sent again to a fresh worker, completes there, and is never a failure."""
    workers = crashes({"p00": 1})
    [done] = _run(tmp_path, _profile(tmp_path), ["p00"], workers)
    assert isinstance(done, Completed) and workers.sent == [1, 1] and workers.starts == 2


def test_a_worker_fatal_for_a_request_is_a_crash_and_one_for_a_load_is_not(tmp_path: Path) -> None:
    """A fatal reported for a loaded worker's request is split and charged; a fatal of a reload is never split."""
    record = {"type": "Fatal", "message": "unreadable line", "fault": "infrastructure"}
    crash = crashes({"p05": FOREVER}, lambda: WorkerFatal(record, from_load=False))
    outcomes = _run(tmp_path, _profile(tmp_path), _texts(BATCH, at={5: "p05"}), crash)
    assert [f.failure.type for f in _failed(outcomes)] == ["ItemCrashedWorker"]

    reload = crashes({"p05": FOREVER}, lambda: WorkerFatal(record, from_load=True))
    with pytest.raises(WorkerFatal):
        _run(tmp_path / "again", _profile(tmp_path), _texts(BATCH, at={5: "p05"}), reload)
    assert reload.sent == [BATCH]


def test_in_process_no_longer_aborts_the_run_on_one_items_crash(tmp_path: Path) -> None:
    """`submit` returns every job's outcome instead of raising the first `WorkerLost`."""
    outcomes = _run(tmp_path, _profile(tmp_path), _texts(BATCH, at={0: "p00"}), crashes({"p00": FOREVER}))
    assert len(outcomes) == BATCH and len(_failed(outcomes)) == 1


def test_a_cancel_is_the_claim_stopping_not_a_crash_to_retry(tmp_path: Path) -> None:
    """The loss `abandon` causes ends the call with `Abandoned`; nothing is split or retried."""
    cancelled = threading.Event()

    def abandoned() -> Exception:
        cancelled.set()
        return WorkerLost("closed by abandon")

    workers = crashes({"p07": FOREVER}, abandoned)
    profile = _profile(tmp_path)
    placed, values = _planned(profile, _texts(BATCH, at={7: "p07"}))
    executor = InProcess(store=ResultStore(tmp_path / "store"), profile=profile, workers=workers)
    jobs, placement = fragment_jobs(node_fragment(placed, executor.admit(placed, values)), placed, values)
    try:
        with pytest.raises(Abandoned):
            execute(
                jobs,
                replace(placement, batch_size=BATCH),
                store=ResultStore(tmp_path / "store"),
                run=RunConfig(),
                profile=profile,
                workers=workers,
                cancelled=cancelled,
            )
    finally:
        workers.close()
    assert workers.sent == [BATCH] and workers.starts == 1


def _served(  # noqa: PLR0913 - the run's directory, clock and workers, then its texts and fault budgets
    tmp_path: Path,
    clock: ManualClock,
    workers: WorkerSet,
    texts: Sequence[str],
    **faults: int,
) -> list[dict[str, Any]]:
    """Serve one claim of every text, one request of them all, through a pool worker over `workers`; return the refs."""
    profile = _profile(tmp_path, **faults)
    placed, _ = _planned(profile, texts)
    store = ResultStore(tmp_path / "store")
    prepared(placed, tmp_path / "run", store)
    pool_dir = tmp_path / "run" / RUN_QUEUES
    put_batched(placed, pool_dir, profile, len(texts))
    send = sender(store, profile, workers)
    worker = PoolWorker(pool_dir, ["cpu"], profile=profile, store=store, send=send, clock=clock)
    assert _clocked(worker, clock) == 0
    return _outcomes(pool_dir, profile)


def _refs(outcomes: Sequence[dict[str, Any]]) -> list[ResultRef]:
    return [ResultRef.model_validate(j) for o in outcomes for j in o["jobs"]]


def test_the_pool_charges_the_item_too_and_commits_the_others(
    tmp_path: Path, lease_clock: Callable[..., ManualClock]
) -> None:
    """The pool worker, through the same engine: the poisoned job is charged, the other eleven are committed."""
    workers = crashes({"p07": FOREVER})
    refs = _refs(_served(tmp_path, lease_clock(), workers, _texts(BATCH, at={7: "p07"})))
    failed = [r for r in refs if r.failure is not None]
    assert len(refs) == BATCH and len(failed) == 1
    assert failed[0].failure is not None and failed[0].failure.type == ItemCrashedWorker.__name__
    assert (failed[0].failure.fault, failed[0].failure.charged) == ("content", True)
    assert workers.starts == len(workers.failed) + 1


def test_the_pool_and_in_process_agree_over_the_same_crash(
    tmp_path: Path, lease_clock: Callable[..., ManualClock]
) -> None:
    """Identical outcomes and identical requests for the same poisoned job: one engine, not two."""
    inproc, pooled = crashes({"p07": FOREVER}), crashes({"p07": FOREVER})
    texts = _texts(BATCH, at={7: "p07"})
    outcomes = _run(tmp_path / "a", _profile(tmp_path / "a"), texts, inproc)
    refs = _refs(_served(tmp_path / "b", lease_clock(), pooled, texts))
    assert [isinstance(o, Failed) for o in outcomes] == [r.failure is not None for r in refs]
    assert inproc.sent == pooled.sent and inproc.starts == pooled.starts


def test_a_crash_another_item_survives_is_infrastructure_in_the_pool(
    tmp_path: Path, lease_clock: Callable[..., ManualClock]
) -> None:
    """A crash lost once and survived is no job's failure: all twelve commit, and the item reports the loss as infra."""
    [outcome] = _served(tmp_path, lease_clock(), crashes({"p03": 1}), _texts(BATCH, at={3: "p03"}))
    assert all(r.failure is None for r in _refs([outcome])) and len(outcome["jobs"]) == BATCH
    [lost] = outcome["survived"]
    assert (lost["fault"], lost["type"], lost["charged"]) == ("infrastructure", "WorkerLost", False)


def test_a_charged_item_reports_no_survived_loss(tmp_path: Path, lease_clock: Callable[..., ManualClock]) -> None:
    """The losses a poisoned item caused are its own: none is reported against the env."""
    [outcome] = _served(tmp_path, lease_clock(), crashes({"p07": FOREVER}), _texts(BATCH, at={7: "p07"}))
    assert "survived" not in outcome


def test_a_request_fatal_typed_resource_is_a_resource_fault_never_charged(tmp_path: Path) -> None:
    """The worker's verdict wins: the lone job fails `resource` once, to be moved to a larger class, not retried."""
    record = {"type": "MemoryError", "message": "out of memory", "fault": "resource"}
    workers = crashes({"p05": FOREVER}, lambda: WorkerFatal(record, from_load=False))
    outcomes = _run(tmp_path, _profile(tmp_path), _texts(BATCH, at={5: "p05"}), workers)
    [dead] = _failed(outcomes)
    assert (dead.failure.fault, dead.failure.charged) == ("resource", False)
    assert dead.failure.type != "ItemCrashedWorker" and workers.failed.count(1) == 1


def test_bisect_resource_is_gone() -> None:
    """The flag that only `InProcess` passed, and the Pool's own split, no longer exist in the sources."""
    root = Path(__file__).parents[3] / "src" / "senselab_core"
    names: set[str] = set()
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.arg):
                names.add(node.arg)
            elif isinstance(node, ast.keyword) and node.arg is not None:
                names.add(node.arg)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
    assert not {"bisect_resource", "whole", "splits", "_split_requests", "_ran_out", "bisect"} & names
