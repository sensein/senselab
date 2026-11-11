"""`InProcess` splits a request whose jobs all ran out of memory, as the pool does, and writes no sizing record."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from backends import recording_oom_backend_fixture as backend
from pool_backend_oom_test import BATCHED
from pool_probe_test import TEXTY
from pool_stack_fixture import packaged, registry
from senselab_core.kernel.dataset import Record, ValueInput
from senselab_core.kernel.graph import Graph
from senselab_core.kernel.identity import InputIdentity
from senselab_core.runtime.call import RunConfig, bind_input
from senselab_core.runtime.datasets import RecordSet
from senselab_core.runtime.execution.executor import Handle
from senselab_core.runtime.execution.in_process import InProcess, fragment_jobs, node_fragment
from senselab_core.runtime.execution.job import Completed, Failed, JobOutcome, execute
from senselab_core.runtime.execution.workers import WorkerSet
from senselab_core.runtime.locks import LeaseTimeout
from senselab_core.runtime.planner import Plan, plan
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.store import ResultStore
from senselab_core.testing.measures import FixedEstimates, FixedUnit
from senselab_worker.faults import ResourceFault, error_record

pytestmark = pytest.mark.logic

BATCH = 12
"""The batch the estimate sizes every job at."""


@pytest.fixture
def profile(tmp_path: Path) -> Iterator[ExecutionProfile]:
    """The laptop profile on one cpu class, with the text task served by the recording backend."""
    override = {
        "devices": ["cpu"],
        "resource_classes": {"cpu": {"device": "cpu", "mem_bytes": 2**33, "max_workers": 2}},
        "calibration": {"measures_roots": [str(tmp_path / "measures")], "warmup_requests": 1},
    }
    backend.BATCHES.clear()
    with packaged(registry(TEXTY, worker="backends.recording_oom_backend_fixture"), tmp_path / "cache"):
        yield ExecutionProfile.load("laptop", override)


def _planned(profile: ExecutionProfile, texts: Sequence[str]) -> tuple[Plan, dict[InputIdentity, object]]:
    """Return the plan mapping the text task over one record per text, and its input values."""
    graph = {
        "inputs": {"text": "list[Text]"},
        "nodes": [{"id": "a", "task": TEXTY.name, "task_version": "1"}],
        "edges": [{"source": "input.text", "target": "a.text", "kind": "map"}],
    }
    records = [Record(id=f"r{i}", inputs={"text": ValueInput(value=t)}) for i, t in enumerate(texts)]
    placed = plan(Graph.model_validate(graph), RecordSet(records), config=RunConfig(), execution=profile)
    return placed, {bind_input("x", "Text", t)[0]: bind_input("x", "Text", t)[1] for t in texts}


def _run(
    tmp_path: Path, profile: ExecutionProfile, texts: Sequence[str], workers: WorkerSet | None = None
) -> list[JobOutcome]:
    """Submit one fragment mapping the text task over `texts` to an `InProcess` that batches at 12; return outcomes."""
    placed, values = _planned(profile, texts)
    held = workers if workers is not None else WorkerSet(1)
    try:
        executor = InProcess(
            store=ResultStore(tmp_path / "store"),
            profile=profile,
            workers=held,
            measures=FixedEstimates(BATCHED),  # type: ignore[arg-type]
            units=FixedUnit(profile).unit,
        )
        handle = executor.submit(node_fragment(placed, executor.admit(placed, values)))
        return executor.outcomes(Handle(id=handle.id))
    finally:
        held.close()


def _texts(n: int, *, at: dict[int, str] | None = None) -> list[str]:
    """Return `n` distinct texts of 30 characters, `at` replacing some by index."""
    return [(at or {}).get(i, f"t{i:02d}").ljust(30, "x") for i in range(n)]


def test_a_batched_oom_above_five_bisects_twelve_to_six_to_three(tmp_path: Path, profile: ExecutionProfile) -> None:
    """Twelve jobs run out as one request; each half that is still over five splits again; all succeed, none twice."""
    outcomes = _run(tmp_path, profile, _texts(BATCH))
    assert all(isinstance(o, Completed) for o in outcomes) and len(outcomes) == BATCH
    assert backend.BATCHES == [12, 6, 3, 3, 6, 3, 3]


def test_an_oversize_item_alone_is_the_only_failure(tmp_path: Path, profile: ExecutionProfile) -> None:
    """One text no request holds: every request with it runs out until it is alone, and only it fails."""
    texts = _texts(BATCH, at={7: "h07" + "y" * 80})
    outcomes = _run(tmp_path, profile, texts)
    [dead] = [o for o in outcomes if isinstance(o, Failed)]
    assert (dead.failure.fault, dead.failure.type) == ("resource", "OutOfMemoryError")
    assert sum(isinstance(o, Completed) for o in outcomes) == BATCH - 1
    assert backend.BATCHES[0] == 12 and backend.BATCHES[-1] <= 3 and 1 in backend.BATCHES


def test_a_content_fault_item_beside_a_batched_oom_keeps_its_own_outcome(
    tmp_path: Path, profile: ExecutionProfile
) -> None:
    """The request runs out as a whole; once split, the item the backend rejects fails as content, the rest succeed."""
    outcomes = _run(tmp_path, profile, _texts(BATCH, at={4: "b04"}))
    assert backend.BATCHES == [12, 6, 3, 3, 6, 3, 3]
    [bad] = [o for o in outcomes if isinstance(o, Failed)]
    assert (bad.failure.fault, bad.failure.type) == ("content", "ContentFault")
    assert sum(isinstance(o, Completed) for o in outcomes) == BATCH - 1


def test_a_request_that_fits_is_sent_once(tmp_path: Path, profile: ExecutionProfile) -> None:
    """Five jobs fit: one request, no split."""
    outcomes = _run(tmp_path, profile, _texts(5))
    assert all(isinstance(o, Completed) for o in outcomes) and backend.BATCHES == [5]


def test_a_one_job_resource_fault_is_that_jobs_failure(tmp_path: Path, profile: ExecutionProfile) -> None:
    """A single job that runs out is sent once and fails with the fault; nothing is split or retried."""
    [dead] = _run(tmp_path, profile, ["h00" + "y" * 80])
    assert isinstance(dead, Failed) and dead.failure.fault == "resource"
    assert backend.BATCHES == [1]


class Raising(WorkerSet):
    """A worker set whose loaded workers raise `exc` for a request of more than `above` jobs, and count each request."""

    def __init__(self, exc: Exception, *, above: int = 0) -> None:
        """Raise `exc` above `above` jobs."""
        super().__init__(1)
        self.exc, self.above, self.sent = exc, above, []  # type: ignore[var-annotated]

    @contextmanager
    def checkout(self, *args: Any, **kwargs: Any) -> Iterator[Any]:  # noqa: ANN401 - `checkout`'s own arguments
        """Hold the real worker, wrapped so that its `run` raises."""
        with super().checkout(*args, **kwargs) as worker:
            yield _Loaded(worker, self)


class _Loaded:
    def __init__(self, worker: Any, owner: Raising) -> None:  # noqa: ANN401 - the real worker
        self.worker, self.owner = worker, owner

    def run(self, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
        self.owner.sent.append(len(items))
        if len(items) > self.owner.above:
            raise self.owner.exc
        return self.worker.run(items, params)  # type: ignore[no-any-return]


class Unloadable(WorkerSet):
    """A worker set whose checkout raises `exc`, as a start or load that failed or a start lease that is busy."""

    def __init__(self, exc: Exception) -> None:
        """Raise `exc` from every checkout."""
        super().__init__(1)
        self.exc, self.checkouts = exc, 0

    @contextmanager
    def checkout(self, *args: Any, **kwargs: Any) -> Iterator[Any]:  # noqa: ANN401, ARG002 - `checkout`'s own
        """Raise."""
        self.checkouts += 1
        raise self.exc
        yield


def test_a_request_that_raises_a_resource_fault_is_split_too(tmp_path: Path, profile: ExecutionProfile) -> None:
    """A worker that raises out of its request, rather than answering each item, is split by the same predicate."""
    workers = Raising(backend.OutOfMemoryError("out of memory"), above=backend.FITS)
    outcomes = _run(tmp_path, profile, _texts(BATCH), workers)
    assert all(isinstance(o, Completed) for o in outcomes) and workers.sent == [12, 6, 3, 3, 6, 3, 3]


def test_a_lone_job_that_raises_a_resource_fault_fails_with_it(tmp_path: Path, profile: ExecutionProfile) -> None:
    """A request of one whose worker raised out of memory is sent once and that job fails; nothing is split."""
    workers = Raising(backend.OutOfMemoryError("out of memory"))
    [dead] = _run(tmp_path, profile, _texts(1), workers)
    assert isinstance(dead, Failed) and (dead.failure.fault, dead.failure.type) == ("resource", "OutOfMemoryError")
    assert workers.sent == [1]


def test_a_busy_start_lease_propagates_from_the_request_that_waited_on_it(
    tmp_path: Path, profile: ExecutionProfile
) -> None:
    """A lease wait is a `TimeoutError` of the resource class, but it is raised once and nothing is split."""
    workers = Raising(LeaseTimeout("start lease busy"))
    with pytest.raises(LeaseTimeout):
        _run(tmp_path, profile, _texts(BATCH), workers)
    assert workers.sent == [12]


@pytest.mark.parametrize("exc", [LeaseTimeout("start lease busy"), ResourceFault("the model does not fit at load")])
def test_a_checkout_that_raises_is_never_split(tmp_path: Path, profile: ExecutionProfile, exc: Exception) -> None:
    """A lease wait or a resource fault raised by checkout, start or load is not a request running out."""
    workers = Unloadable(exc)
    with pytest.raises(type(exc)):
        _run(tmp_path, profile, _texts(BATCH), workers)
    assert workers.checkouts == 1


def test_what_a_request_raises_keeps_its_cause_chain(tmp_path: Path, profile: ExecutionProfile) -> None:
    """An exception that is not split reaches the caller as raised: its cause is the OSError and the record names it."""
    cause = OSError("pipe closed")
    failure = ValueError("the request broke")
    failure.__cause__ = cause
    placed, values = _planned(profile, _texts(3))
    workers = Raising(failure)
    executor = InProcess(store=ResultStore(tmp_path / "store"), profile=profile, workers=workers)
    jobs, placement = fragment_jobs(node_fragment(placed, executor.admit(placed, values)), placed, values)
    try:
        with pytest.raises(ValueError, match="the request broke") as caught:
            execute(
                jobs,
                replace(placement, batch_size=3),
                store=ResultStore(tmp_path / "store"),
                run=RunConfig(),
                profile=profile,
                workers=workers,
            )
    finally:
        workers.close()
    assert caught.value is failure and caught.value.__cause__ is cause
    assert "OSError" in str(error_record(caught.value)["traceback"])
