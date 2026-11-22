"""A fault's class survives every wrapper (ADR-009): a load failure keeps its cause's class and is never split."""

import ast
from collections.abc import Callable
from pathlib import Path

import pytest
from pool_stack_fixture import host_env, registry, scoring
from senselab_core.kernel.declare import BackendSpec
from senselab_core.runtime.execution.executor import FaultRecord
from senselab_core.runtime.execution.providers import Loader, SenselabHost
from senselab_core.runtime.execution.workers import WorkerLoadFailed, loaded
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.worker_client import WorkerFatal
from senselab_worker.faults import MAX_TEXT_CHARS, ContentFault, InfrastructureFault, ResourceFault, error_record

pytestmark = pytest.mark.logic

OOM_BACKEND = "backends.oom_load_backend_fixture"
BROKEN = scoring("fault_class_test.broken")
TYPED: dict[str, type[Exception]] = {
    "content": ContentFault,
    "resource": ResourceFault,
    "infrastructure": InfrastructureFault,
}
PACKAGES = next(p for p in Path(__file__).resolve().parents if p.name == "packages")


def _backend() -> BackendSpec:
    return registry(BROKEN, worker=OOM_BACKEND, name="broken").backends(BROKEN.name)[0]


def host_load_failure(raised: Exception) -> WorkerLoadFailed:
    """Return the `WorkerLoadFailed` the host provider raises when its backend's load raises `raised`."""
    from backends import oom_load_backend_fixture as fixture  # noqa: PLC0415 - the module holds what load raises

    fixture.LOADS.clear()
    fixture.RAISE[:] = [raised]
    backend, profile = _backend(), ExecutionProfile.load()
    provider = SenselabHost()
    spec = Loader(backend, host_env(backend, profile), None, "cpu", profile).spec
    with pytest.raises(WorkerLoadFailed) as caught:
        provider.load(spec, None)
    return caught.value


def _server_load_failure(raised: Exception) -> WorkerLoadFailed:
    """Return the `WorkerLoadFailed` a server's start raises when the start raises `raised`."""

    def start() -> object:
        raise raised

    with pytest.raises(WorkerLoadFailed) as caught:
        loaded(start)
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
