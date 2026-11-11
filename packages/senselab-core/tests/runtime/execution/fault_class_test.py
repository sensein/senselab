"""A fault's class survives every wrapper (ADR-009): a load failure keeps its cause's class and is never split."""

import ast
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from pool_stack_fixture import host_env, packaged, registry
from pool_worker_test import _profile, _put, _refs, _run, _sender
from senselab_core.kernel.declare import BackendSpec
from senselab_core.runtime.execution.executor import FaultRecord
from senselab_core.runtime.execution.job import Job, JobOutcome
from senselab_core.runtime.execution.pool_worker import PoolWorker
from senselab_core.runtime.execution.workers import WorkerLoadFailed, WorkerSet
from senselab_core.runtime.keys import Placement
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.worker_client import WorkerFatal
from senselab_worker.faults import ContentFault, InfrastructureFault, ResourceFault, error_record
from worker_set_test import _attestation, _model

from pool_worker_test import BROKEN  # isort: skip

pytestmark = pytest.mark.logic

OOM_BACKEND = "backends.oom_load_backend_fixture"
TYPED: dict[str, type[Exception]] = {
    "content": ContentFault,
    "resource": ResourceFault,
    "infrastructure": InfrastructureFault,
}
SRC = next(p for p in Path(__file__).resolve().parents if p.name == "senselab-core") / "src"


def _backend() -> BackendSpec:
    return registry(BROKEN, worker=OOM_BACKEND, name="broken").backends(BROKEN.name)[0]


def host_load_failure(raised: Exception) -> WorkerLoadFailed:
    """Return the `WorkerLoadFailed` a real host checkout raises when its backend's load raises `raised`."""
    from backends import oom_load_backend_fixture as fixture  # noqa: PLC0415 - the module holds what load raises

    fixture.LOADS.clear()
    fixture.RAISE[:] = [raised]
    backend, profile = _backend(), ExecutionProfile.load()
    with pytest.raises(WorkerLoadFailed) as caught:
        with WorkerSet(1).checkout(backend, host_env(backend, profile), None, device="cpu", profile=profile):
            pass
    return caught.value


def _server_load_failure(raised: Exception) -> WorkerLoadFailed:
    from senselab_core.kernel.declare import Env, ServerConfig  # noqa: PLC0415

    def start(*_: object) -> object:
        raise raised

    backend = _backend().model_copy(
        update={"env": Env.server("s"), "server": ServerConfig(settings={}, startup_timeout_s=1.0)}
    )
    model, profile = _model("b" * 40), ExecutionProfile.load()
    with pytest.raises(WorkerLoadFailed) as caught:
        with WorkerSet(1, start_server=start).checkout(  # type: ignore[arg-type]
            backend, host_env(backend, profile), model, device="cpu", profile=profile, expected=_attestation(model)
        ):
            pass
    return caught.value


def _fatal_load_failure(raised: Exception) -> WorkerLoadFailed:
    failed = WorkerLoadFailed("WorkerFatal")
    failed.__cause__ = WorkerFatal(error_record(raised))
    return failed


KINDS: dict[str, Callable[[Exception], WorkerLoadFailed]] = {
    "host": host_load_failure,
    "server": _server_load_failure,
    "client": _fatal_load_failure,
}


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("fault", TYPED)
def test_a_load_failure_keeps_its_causes_class_through_every_wrapper(fault: str, kind: str) -> None:
    """Whatever class the backend states at start or load, the wrapper's record carries it, as a typed fault."""
    wrapped = KINDS[kind](TYPED[fault]("stated by the backend"))
    record = FaultRecord.from_exception(wrapped)
    assert (record.fault, record.gap) == (fault, False)
    assert record.type == "WorkerLoadFailed" or kind == "client"


def test_a_load_failure_the_table_classified_is_a_gap_of_that_class() -> None:
    """A `MemoryError` at load is a resource gap, as the table gives it: the class survives and so does the gap."""
    record = FaultRecord.from_exception(_fatal_load_failure(MemoryError("out of memory")))
    assert (record.fault, record.gap, record.charged) == ("resource", True, False)
    assert FaultRecord.from_exception(_fatal_load_failure(RuntimeError("env"))).fault == "infrastructure"


@pytest.mark.parametrize(
    "raised", [ResourceFault("out of memory"), MemoryError("out of memory")], ids=["typed", "table"]
)
def test_a_load_oom_is_a_resource_fault_costing_one_load_and_never_split(tmp_path: Path, raised: Exception) -> None:
    """A claim of 8 whose backend runs out of memory at load is one request and one load, each job `resource`."""
    from backends import oom_load_backend_fixture as fixture  # noqa: PLC0415 - the module records loads

    fixture.LOADS.clear()
    fixture.RAISE[:] = [raised]
    profile = _profile(quantum_items=8)
    with packaged(registry(BROKEN, worker=OOM_BACKEND, name="broken"), tmp_path / "cache"):
        placed, store, pool_dir = _run(tmp_path, profile, [f"t{i}" for i in range(8)], BROKEN)
        _put(placed, pool_dir, range(8))
        send, sizes = _sender(store, profile), []

        def counted(jobs: Sequence[Job], placement: Placement) -> list[JobOutcome]:
            sizes.append(len(jobs))
            return send(jobs, placement)

        assert PoolWorker(pool_dir, ["cpu"], profile=profile, store=store, send=counted).serve() == 0
    refs = _refs(pool_dir, profile)
    assert sizes == [8] and len(fixture.LOADS) == 1 and len(refs) == 8
    assert {(r.failure.type, r.failure.fault, r.failure.charged) for r in refs if r.failure} == {
        ("WorkerLoadFailed", "resource", False)
    }


def test_no_fault_record_is_built_from_a_literal_class_outside_the_one_constructor() -> None:
    """Every `FaultRecord` comes from `of`, `from_exception` or `gap_of`: no `fault=` keyword at a call site."""
    found = []
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and any(k.arg == "fault" for k in node.keywords):
                name = ast.unparse(node.func)
                if name in {"FaultRecord", "cls"} and not (path.name == "executor.py" and name == "cls"):
                    found.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert found == []
