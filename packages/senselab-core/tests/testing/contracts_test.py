"""Generated contract tests over the registry (testing.md §3), and checks that each contract can fail."""

import hashlib
import os
import subprocess
import sys
import unittest
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import senselab_core.runtime.fake_backend  # noqa: F401  registers fake.echo and its backend
from pydantic import BaseModel, ConfigDict
from senselab_core.kernel import declare
from senselab_core.kernel.declare import REGISTRY, BackendParams, BackendSpec, Registry, TaskParams, TaskSpec
from senselab_core.kernel.identity import ExecutionContext, LogicalKey
from senselab_core.kernel.license import License, MissingLicense
from senselab_core.runtime.fake_backend import FAKE
from senselab_core.runtime.keys import Call, Placement, build_key
from senselab_core.runtime.licenses import license_record
from senselab_core.runtime.worker_client import WorkerFatal
from senselab_core.testing.contracts import PLACEMENT_ONLY, contract_cases


@pytest.mark.contract
@pytest.mark.parametrize("case", [pytest.param(case, id=name) for name, case in contract_cases(REGISTRY)])
def test_contract(case: Callable[[], None]) -> None:
    """Every declared task and backend satisfies the generated contract."""
    case()


@pytest.mark.logic
def test_fake_cases_are_listed_by_name() -> None:
    """Ids are the task or backend name and the case; key cases are one per identity and context field."""
    ids = {name for name, _ in contract_cases(REGISTRY)}
    expected = {
        "fake.echo::params_round_trip",
        "fake.echo::params_reject_unknown",
        "fake::key_deterministic",
        "fake::output_affecting_produced",
        "fake::key_independent_of_hash_seed",
        "fake::content_fault_isolated",
    }
    undeclared = {"batch_formation", "chunk_grid"}
    expected |= {f"fake::key_changes_with_{f}" for f in set(LogicalKey.model_fields) - undeclared}
    expected |= {f"fake::key_ignores_{f}" for f in {*ExecutionContext.model_fields, *PLACEMENT_ONLY, *undeclared}}
    assert expected <= ids
    assert not {f"fake::key_changes_with_{f}" for f in undeclared} & ids


class _Echo(TaskParams):
    salt: str = ""


class _Knobs(BackendParams):
    beam: int = 1


def _registry(task: dict[str, Any], backend: dict[str, Any] | None = None) -> Registry:
    fields: dict[str, Any] = {
        "name": "t.bad",
        "version": "1",
        "kind": "transform",
        "inputs": {"payload": "json"},
        "outputs": {"value": "sha256"},
        "params": _Echo,
    }
    fields.update(task)
    registry = Registry()
    registry.add_task(TaskSpec(**fields))
    registry.add_backend(BackendSpec(**{**FAKE.model_dump(), "task": "t.bad", "params": _Knobs, **(backend or {})}))
    return registry


def _case(registry: Registry, name: str, **kwargs: Any) -> Callable[[], None]:  # noqa: ANN401
    return dict(contract_cases(registry, **kwargs))[name]


@pytest.mark.logic
def test_a_good_declaration_passes_every_offline_case() -> None:
    """The fixture declaration is valid, so the failures below come from the one defect each introduces."""
    for name, case in contract_cases(_registry({})):
        if name.endswith(("content_fault_isolated", "key_independent_of_hash_seed", "backend_license")):
            continue
        case()


class _Lax(BaseModel):
    salt: str = ""


class _Silent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    salt: str = ""


class _NoDefault(TaskParams):
    salt: str


@pytest.mark.logic
@pytest.mark.parametrize(
    "task,case",
    [
        ({"params": _Lax}, "t.bad::params_reject_unknown"),
        ({"params": _Silent}, "t.bad::params_reject_unknown"),
        ({"params": _NoDefault}, "t.bad::params_round_trip"),
    ],
    ids=["unknown_key_accepted", "unknown_key_without_suggestion", "defaults_invalid"],
)
def test_a_defective_task_fails_its_case(task: dict[str, Any], case: str) -> None:
    """Each task-level contract fails on the defect it guards."""
    with pytest.raises((AssertionError, ValueError)):
        _case(_registry(task), case)()


@pytest.mark.logic
@pytest.mark.parametrize(
    "backend",
    [
        {"batch_invariant": False},
        {"batch_invariant": False, "output_affecting": ("batch_formation", "chunk_grid")},
    ],
    ids=["batch_dependent_undeclared", "chunk_grid_without_producer"],
)
def test_an_output_affecting_component_needs_a_producer(backend: dict[str, Any]) -> None:
    """A batch-dependent backend declares its batch formation, and a declared chunk grid has a producer."""
    case = _case(_registry({}, backend), "fake::output_affecting_produced")
    with pytest.raises(AssertionError):
        case()
    _case(
        _registry({}, {**backend, "output_affecting": ("batch_formation", "chunk_grid"), "chunk_grid": "30s"}),
        "fake::output_affecting_produced",
    )()


@pytest.mark.logic
def test_a_backend_shared_by_name_across_tasks_is_qualified() -> None:
    """Two tasks with a backend of the same name get ids `<task>/<backend>::<case>`, never a number."""
    registry = _registry({})
    registry.add_task(TaskSpec(**{**registry.tasks()[0].model_dump(), "name": "t.two", "kind": "fit"}))
    registry.add_backend(BackendSpec(**{**FAKE.model_dump(), "task": "t.two"}))
    ids = {name for name, _ in contract_cases(registry)}
    assert {"t.bad/fake::key_deterministic", "t.two/fake::key_deterministic"} <= ids
    assert "fake::key_deterministic" not in ids


def _drops_params(call: Call, placement: Placement, context: ExecutionContext) -> LogicalKey:
    return build_key(replace(call, params=call.task.params()), placement, context)


def _drops_backend_params(call: Call, placement: Placement, context: ExecutionContext) -> LogicalKey:
    return build_key(call, replace(placement, backend_params=placement.backend.params()), context)


def _drops_backend_version(call: Call, placement: Placement, context: ExecutionContext) -> LogicalKey:
    return build_key(call, replace(placement, backend=placement.backend.model_copy(update={"version": "1"})), context)


def _drops_inputs(call: Call, placement: Placement, context: ExecutionContext) -> LogicalKey:
    inputs = {port: "content:" + "0" * 64 for port in call.task.inputs}
    return build_key(replace(call, inputs=inputs), placement, context)


def _drops_model(call: Call, placement: Placement, context: ExecutionContext) -> LogicalKey:
    return build_key(replace(call, model=None), placement, context)


def _drops_profile_slices(call: Call, placement: Placement, context: ExecutionContext) -> LogicalKey:
    return build_key(replace(call, profile_slices={}), placement, context)


@pytest.mark.logic
@pytest.mark.parametrize(
    "builder,field",
    [
        (_drops_params, "params"),
        (_drops_backend_params, "backend_params"),
        (_drops_backend_version, "backend_version"),
        (_drops_inputs, "inputs"),
        (_drops_model, "model"),
        (_drops_profile_slices, "profile_slices"),
    ],
    ids=lambda v: getattr(v, "__name__", v),
)
def test_a_key_builder_that_drops_a_component_fails_its_case(builder: Callable[..., LogicalKey], field: str) -> None:
    """The key cases go through the key builder, so one that ignores a call component fails that case only."""
    registry = _registry({})
    with pytest.raises(AssertionError, match=field):
        _case(registry, f"fake::key_changes_with_{field}", key_builder=builder)()
    _case(registry, "fake::key_changes_with_task", key_builder=builder)()


def _drops_chunk_grid(call: Call, placement: Placement, context: ExecutionContext) -> LogicalKey:
    return build_key(call, replace(placement, chunk_grid="fixed"), context)


def _leaks_batch_formation(call: Call, placement: Placement, context: ExecutionContext) -> LogicalKey:
    key = build_key(call, placement, context)
    return LogicalKey.model_validate({**key.model_dump(), "batch_formation": placement.batch_formation})


@pytest.mark.logic
def test_output_affecting_cases_follow_the_declaration() -> None:
    """A declared chunk grid gets `key_changes_with_chunk_grid`, an undeclared batch formation `key_ignores_*`."""
    registry = _registry({}, {"output_affecting": ("chunk_grid",)})
    cases = dict(contract_cases(registry))
    assert "fake::key_changes_with_batch_formation" not in cases
    cases["fake::key_changes_with_chunk_grid"]()
    cases["fake::key_ignores_batch_formation"]()
    with pytest.raises(AssertionError, match="chunk_grid"):
        _case(registry, "fake::key_changes_with_chunk_grid", key_builder=_drops_chunk_grid)()
    with pytest.raises(AssertionError, match="batch_formation reached the key"):
        _case(registry, "fake::key_ignores_batch_formation", key_builder=_leaks_batch_formation)()


def _keys_the_context(call: Call, placement: Placement, context: ExecutionContext) -> LogicalKey:
    key = build_key(call, placement, context)
    return LogicalKey.model_validate({**key.model_dump(), "chunk_grid": context.host})


def _keys_the_device(call: Call, placement: Placement, context: ExecutionContext) -> LogicalKey:
    key = build_key(call, placement, context)
    return LogicalKey.model_validate({**key.model_dump(), "chunk_grid": placement.device})


@pytest.mark.logic
def test_ignore_cases_catch_a_builder_that_keys_context_or_placement() -> None:
    """`key_ignores_<field>` builds the key under two contexts or placements; a builder that keys one fails it."""
    registry = _registry({})
    _case(registry, "fake::key_ignores_host")()
    _case(registry, "fake::key_ignores_device")()
    with pytest.raises(AssertionError, match="host reached the key"):
        _case(registry, "fake::key_ignores_host", key_builder=_keys_the_context)()
    with pytest.raises(AssertionError, match="device reached the key"):
        _case(registry, "fake::key_ignores_device", key_builder=_keys_the_device)()


@pytest.mark.contract
def test_worker_backend_gets_contract_cases() -> None:
    """A backend on any stack gets `content_fault_isolated`, which runs its worker; no stack is refused."""
    registry = _registry({}, {"stack": "torch", "env": {"kind": "isolated", "name": "torch-env"}})
    _case(registry, "fake::content_fault_isolated")()


def _lic(expression: str, **overrides: object) -> License:
    return license_record(
        expression,
        source="a lock",
        verified={"date": "2026-11-01", "how": "pep639"},
        gated=False,
        **overrides,  # type: ignore[arg-type]
    )


@pytest.mark.logic
def test_backend_license_case_derives_from_supplied_lock_records() -> None:
    """The case passes on consistent lock records, fails on a record whose flags contradict it or is absent."""
    case = _case(
        _registry({}), "fake::backend_license", lock_records=lambda b: {"a": _lic("MIT"), "b": _lic("MPL-2.0")}
    )
    case()
    contradiction = License.model_validate({**_lic("GPL-3.0-only").model_dump(), "copyleft": "none"})
    with pytest.raises(ValueError, match="contradict"):
        _case(_registry({}), "fake::backend_license", lock_records=lambda b: {"a": contradiction})()
    with pytest.raises(MissingLicense, match="package a"):
        _case(_registry({}), "fake::backend_license", lock_records=lambda b: {"a": None})()
    with pytest.raises(unittest.SkipTest):
        _case(_registry({}), "fake::backend_license")()


@pytest.fixture
def stub_worker_importable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let worker processes import `stub_worker_fixture` from this directory."""
    here = str(Path(__file__).parent)
    existing = os.environ.get("PYTHONPATH")
    monkeypatch.setenv("PYTHONPATH", here if not existing else f"{here}{os.pathsep}{existing}")


@pytest.mark.contract
@pytest.mark.usefixtures("stub_worker_importable")
def test_contract_cases_load_the_declared_stub() -> None:
    """`content_fault_isolated` passes the declared load stub to `load`; a stub the worker cannot load fails it."""
    worker = {"worker": "stub_worker_fixture", "stack": "torch"}
    model = {"source": "stub", "id": "stub", "identity": "0" * 40}
    stub = {"model": model, "device": "cpu", "dtype": "default"}
    _case(_registry({}, {**worker, "load_stub": stub}), "fake::content_fault_isolated")()
    weights = {**stub, "model": {**model, "id": "weights"}}
    with pytest.raises(WorkerFatal, match="weights"):
        _case(_registry({}, {**worker, "load_stub": weights}), "fake::content_fault_isolated")()


@pytest.mark.contract
@pytest.mark.usefixtures("stub_worker_importable")
def test_fault_injection_per_request_one_load(tmp_path: Path) -> None:
    """`content_fault_isolated` runs its clean and its faulted request on one load of the worker."""
    log = tmp_path / "loads"
    registry = _registry(
        {}, {"worker": "load_log_worker_fixture", "stack": "torch", "load_stub": {"load_log": str(log)}}
    )
    _case(registry, "fake::content_fault_isolated")()
    assert log.read_text().splitlines() == ["load"]


@pytest.mark.contract
def test_a_declared_minimum_the_worker_does_not_enforce_fails_its_case() -> None:
    """A backend declaring `min_input_ns` whose worker accepts shorter audio fails `input_below_declared_minimum`."""
    case = "fake::input_below_declared_minimum_is_content_fault"
    registry = _registry({"inputs": {"audio": "Audio"}}, {"min_input_ns": 1_000_000})
    with pytest.raises(AssertionError, match="were not refused"):
        _case(registry, case)()
    with pytest.raises(AssertionError, match="no Audio input"):
        _case(_registry({}, {"min_input_ns": 1_000_000}), case)()
    assert case not in dict(contract_cases(_registry({})))


@pytest.mark.logic
def test_importing_the_contracts_module_registers_nothing() -> None:
    """Only importing the fake stack declares `fake.echo`; the contracts module alone leaves REGISTRY empty."""
    code = (
        "import senselab_core.testing.contracts\n"
        "from senselab_core.kernel.declare import REGISTRY\n"
        "assert REGISTRY.tasks() == [], REGISTRY.tasks()\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


class _Tagged(TaskParams):
    tags: set[str] = {f"tag-{i}" for i in range(16)}


def _seed_dependent(call: Call, placement: Placement, context: ExecutionContext) -> LogicalKey:
    salted = {port: "content:" + hashlib.sha256(str(hash(port)).encode()).hexdigest() for port in call.inputs}
    return build_key(replace(call, inputs=salted), placement, context)


def _seed_dependent_model(call: Call, placement: Placement, context: ExecutionContext) -> LogicalKey:
    model = call.model
    assert model is not None, "the base call carries a model"
    salted = model.model_copy(update={"identity": hashlib.sha256(str(hash(model.ref.id)).encode()).hexdigest()[:40]})
    return build_key(replace(call, model=salted), placement, context)


@pytest.mark.contract
def test_hash_seed_case_catches_a_seed_dependent_model_identity() -> None:
    """The base call carries a model, so a model identity that reads `hash()` fails the case."""
    case = _case(_registry({}), "fake::key_independent_of_hash_seed", key_builder=_seed_dependent_model)
    with pytest.raises(AssertionError, match="hash seed"):
        case()


@pytest.mark.logic
def test_hash_seed_case_reports_a_timeout() -> None:
    """An interpreter that does not finish within the bound fails the case instead of stalling the suite."""
    with pytest.raises(AssertionError, match="took over 0.001 s"):
        _case(_registry({}), "fake::key_independent_of_hash_seed", hash_seed_timeout_s=0.001)()


@pytest.mark.contract
def test_hash_seed_case_passes_for_the_good_declaration() -> None:
    """The fixture declaration's key is the same under both hash seeds."""
    _case(_registry({}), "fake::key_independent_of_hash_seed")()


@pytest.mark.contract
def test_hash_seed_case_catches_a_seed_dependent_builder() -> None:
    """A key builder that reads `hash()` gives two digests under two seeds, and the case says so."""
    case = _case(_registry({}), "fake::key_independent_of_hash_seed", key_builder=_seed_dependent)
    with pytest.raises(AssertionError, match="hash seed"):
        case()


class _Unchecked(Registry):
    """One task and backend, held as a registry that skipped the canonical-params check would hold them."""

    def __init__(self, task: TaskSpec, backend: BackendSpec) -> None:
        super().__init__()
        self._task, self._backend = task, backend

    def tasks(self) -> list[TaskSpec]:
        return [self._task]

    def backends(self, task: str) -> list[BackendSpec]:
        return [self._backend]


@pytest.mark.contract
def test_set_params_are_refused_and_caught_if_declared_anyway() -> None:
    """A `set[str]` param is keyed differently under each hash seed.

    The registry refuses the type; a registry that skipped that check would still fail the generated case.
    """
    with pytest.raises(declare.NonCanonicalParams, match="tags"):
        _registry({"params": _Tagged})
    good = _registry({})
    (task,) = good.tasks()
    (backend,) = good.backends(task.name)
    unchecked = _Unchecked(task.model_copy(update={"params": _Tagged}), backend)
    case = _case(unchecked, "fake::key_independent_of_hash_seed")
    with pytest.raises(AssertionError, match="hash seed"):
        case()


@pytest.mark.logic
def test_hash_seed_case_needs_importable_declarations() -> None:
    """A params model defined inside a function cannot be imported by name, and the case says so."""

    class Local(TaskParams):
        salt: str = ""

    with pytest.raises(AssertionError, match="module level"):
        _case(_registry({"params": Local}), "fake::key_independent_of_hash_seed")()
