"""A fault's class survives every wrapper (ADR-009): a load failure keeps its cause's class and is never split."""

import ast
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from pool_stack_fixture import host_env, packaged, registry
from pool_worker_test import _profile, _put, _refs, _run, _sender
from senselab_core.kernel.declare import BackendSpec
from senselab_core.runtime.execution.executor import FaultRecord, JobRef
from senselab_core.runtime.execution.job import Failed, Job, JobOutcome
from senselab_core.runtime.execution.pool_worker import PoolWorker
from senselab_core.runtime.execution.workers import WorkerLoadFailed, WorkerSet
from senselab_core.runtime.fake_backend import ECHO, FAKE, EchoParams
from senselab_core.runtime.keys import Call, Placement
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.worker_client import WorkerFatal
from senselab_worker.faults import MAX_TEXT_CHARS, ContentFault, InfrastructureFault, ResourceFault, error_record
from worker_set_test import _attestation, _model

from pool_worker_test import BROKEN  # isort: skip

pytestmark = pytest.mark.logic

OOM_BACKEND = "backends.oom_load_backend_fixture"
TYPED: dict[str, type[Exception]] = {
    "content": ContentFault,
    "resource": ResourceFault,
    "infrastructure": InfrastructureFault,
}
PACKAGES = next(p for p in Path(__file__).resolve().parents if p.name == "packages")


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

    backend = _backend().model_copy(update={"env": Env.server("s"), "server": ServerConfig(settings={})})
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
    stands = "infrastructure" if fault == "content" else fault
    assert (record.fault, record.gap, record.load) == (stands, False, True)
    assert record.type == "WorkerLoadFailed" or kind == "client"
    assert not record.charged


@pytest.mark.parametrize("fault", TYPED)
def test_a_worker_fatal_carries_its_records_class_as_a_load(fault: str) -> None:
    """`WorkerFatal` is its record's class; a content record is the environment's, since a load is not an item."""
    fatal = WorkerFatal(error_record(TYPED[fault]("stated")))
    record = FaultRecord.from_exception(fatal)
    assert (record.fault, record.load) == ("infrastructure" if fault == "content" else fault, True)


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
    fixture.SUCCEED[:] = [0]
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


def test_a_load_is_serialised_only_when_true() -> None:
    """A record that is no load's has no `load` key, so a reader of the earlier record shape still reads it."""
    assert "load" not in FaultRecord.gap_of("X", "m").model_dump(mode="json")
    assert "load" not in FaultRecord.from_exception(ContentFault("m")).model_dump_json()
    loaded = FaultRecord.from_exception(_fatal_load_failure(MemoryError("m")))
    assert loaded.model_dump(mode="json")["load"] is True
    assert FaultRecord.model_validate(loaded.model_dump(mode="json")) == loaded


def test_the_message_of_a_record_with_notes_stays_within_the_text_cap() -> None:
    """The notes `from_exception` appends are cut with the message at `MAX_TEXT_CHARS`."""
    exc = RuntimeError("m" * MAX_TEXT_CHARS)
    exc.add_note("n" * 1000)
    assert len(FaultRecord.from_exception(exc).message) <= MAX_TEXT_CHARS


def test_no_fault_record_is_built_from_a_literal_class_outside_the_one_constructor() -> None:
    """Every `FaultRecord` comes from `of`, `from_exception` or `gap_of`: no literal class at a call site.

    A call is a violation when it passes `fault=` to `FaultRecord` (however it is reached) or `cls`, or hands `of` a
    dict with a literal `"fault"` key; only `executor.py`'s own constructors are exempt.
    """
    found = []
    for path in sorted(PACKAGES.glob("*/src/**/*.py")):
        own = path.name == "executor.py" and path.parent.name == "execution"
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            name = ast.unparse(node.func).rpartition(".")[2]
            keyword = any(k.arg == "fault" for k in node.keywords) and name in {"FaultRecord", "cls"}
            literal = name == "of" and any(
                isinstance(key, ast.Constant) and key.value == "fault"
                for arg in node.args
                if isinstance(arg, ast.Dict)
                for key in arg.keys
            )
            if (keyword or literal) and not own:
                found.append(f"{path.relative_to(PACKAGES)}:{node.lineno}")
    assert found == []


def test_a_reload_that_runs_out_of_memory_is_a_load_never_split(tmp_path: Path) -> None:
    """A state the cache evicted reloads inside a request; its OOM fails the claim as one request, a resource load."""
    from backends import oom_load_backend_fixture as fixture  # noqa: PLC0415 - the module holds what load raises
    from senselab_core.runtime.execution.pool_worker import Member, run_members  # noqa: PLC0415
    from senselab_core.runtime.host_worker import HostWorker  # noqa: PLC0415
    from senselab_worker.model_cache import ModelCache  # noqa: PLC0415

    fixture.LOADS.clear()
    fixture.SUCCEED[:] = [1]
    fixture.RAISE[:] = [MemoryError("CUDA out of memory")]
    cache = ModelCache(2**20)
    worker = HostWorker(
        OOM_BACKEND, init={"device": "cpu"}, key=("oom", "a" * 40, "cpu", "fp32"), cache=cache, size_of=lambda _: 1
    )
    cache.clear()
    sizes: list[int] = []

    def send(jobs: Sequence[Job], placement: Placement) -> list[JobOutcome]:  # noqa: ARG001 - a Send
        sizes.append(len(jobs))
        worker.run([{"id": str(i), "inputs": {}} for i in range(len(jobs))], {})
        raise AssertionError("the reload did not fail")

    members = [
        Member(
            "d", i, Job(JobRef(plan_hash="0" * 64, node="n", element=(i,)), Call(ECHO, EchoParams(), {}), {}), f"k{i}"
        )
        for i in range(8)
    ]
    outcomes = run_members(members, Placement(backend=FAKE, backend_params=FAKE.params(), device="cpu"), send)
    fixture.SUCCEED[:] = [0]
    assert sizes == [8] and len(fixture.LOADS) == 2
    failures = {(o.failure.fault, o.failure.load, o.failure.charged) for o in outcomes if isinstance(o, Failed)}
    assert failures == {("resource", True, False)}
