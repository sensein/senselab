"""Generated contract tests over the registry (testing.md §3), and checks that each contract can fail."""

import hashlib
import subprocess
import sys
import unittest
from collections.abc import Callable
from typing import Any

import pytest
import senselab_core.runtime.fake_backend  # noqa: F401  registers fake.echo and its backend
from pydantic import BaseModel, ConfigDict
from senselab_core.kernel import declare
from senselab_core.kernel.declare import REGISTRY, BackendSpec, Registry, TaskParams, TaskSpec, key_for
from senselab_core.kernel.identity import ExecutionContext, LogicalKey, ResolvedModel
from senselab_core.runtime.fake_backend import FAKE
from senselab_core.testing.contracts import NOT_YET_WIRED, NoContractRunner, contract_cases


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
        "fake.echo::measure_has_no_verdict",
        "fake::license_present",
        "fake::key_deterministic",
        "fake::key_independent_of_hash_seed",
        "fake::content_fault_isolated",
    }
    expected |= {f"fake::key_changes_with_{f}" for f in LogicalKey.model_fields}
    expected |= {f"fake::key_ignores_{f}" for f in ExecutionContext.model_fields}
    assert expected <= ids


class _Echo(TaskParams):
    salt: str = ""


def _registry(task: dict[str, Any], backend: dict[str, Any] | None = None) -> Registry:
    fields: dict[str, Any] = {
        "name": "t.bad",
        "version": "1",
        "kind": "measure",
        "inputs": {"payload": "json"},
        "outputs": {"value": "sha256"},
        "params": _Echo,
    }
    fields.update(task)
    registry = Registry()
    registry.add_task(TaskSpec(**fields))
    registry.add_backend(BackendSpec(**{**FAKE.model_dump(), "task": "t.bad", **(backend or {})}))
    return registry


def _case(registry: Registry, name: str, **kwargs: Any) -> Callable[[], None]:  # noqa: ANN401
    return dict(contract_cases(registry, **kwargs))[name]


@pytest.mark.logic
def test_a_good_declaration_passes_every_offline_case() -> None:
    """The fixture declaration is valid, so the failures below come from the one defect each introduces."""
    for name, case in contract_cases(_registry({})):
        if name.endswith(("content_fault_isolated", "key_independent_of_hash_seed")):
            continue
        try:
            case()
        except unittest.SkipTest:
            assert name.removeprefix("fake::key_changes_with_") in NOT_YET_WIRED


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
        ({"outputs": {"value": "sha256", "keep": "Verdict"}}, "t.bad::measure_has_no_verdict"),
    ],
    ids=["unknown_key_accepted", "unknown_key_without_suggestion", "defaults_invalid", "measure_emits_verdict"],
)
def test_a_defective_task_fails_its_case(task: dict[str, Any], case: str) -> None:
    """Each task-level contract fails on the defect it guards."""
    with pytest.raises((AssertionError, ValueError)):
        _case(_registry(task), case)()


@pytest.mark.logic
def test_verdict_check_applies_to_measure_only() -> None:
    """A `decide` task may output a verdict; it gets no measure case."""
    registry = _registry({"kind": "decide", "outputs": {"keep": "Verdict"}})
    assert not any(name.endswith("measure_has_no_verdict") for name, _ in contract_cases(registry))


@pytest.mark.logic
def test_a_backend_shared_by_name_across_tasks_is_qualified() -> None:
    """Two tasks with a backend of the same name get ids `<task>/<backend>::<case>`, never a number."""
    registry = _registry({})
    registry.add_task(TaskSpec(**{**registry.tasks()[0].model_dump(), "name": "t.two", "kind": "transform"}))
    registry.add_backend(BackendSpec(**{**FAKE.model_dump(), "task": "t.two"}))
    ids = {name for name, _ in contract_cases(registry)}
    assert {"t.bad/fake::license_present", "t.two/fake::license_present"} <= ids
    assert "fake::license_present" not in ids


def _drops_params(task: TaskSpec, backend: BackendSpec, *, params: BaseModel, **call: Any) -> LogicalKey:  # noqa: ANN401
    return key_for(task, backend, params=task.params(), **call)


def _drops_inputs(task: TaskSpec, backend: BackendSpec, *, inputs: dict[str, str], **call: Any) -> LogicalKey:  # noqa: ANN401
    return key_for(task, backend, inputs={port: "content:" + "0" * 64 for port in task.inputs}, **call)


def _drops_model(task: TaskSpec, backend: BackendSpec, *, model: ResolvedModel | None, **call: Any) -> LogicalKey:  # noqa: ANN401
    return key_for(task, backend, **call)


def _drops_profile_slices(task: TaskSpec, backend: BackendSpec, *, profile_slices: Any, **call: Any) -> LogicalKey:  # noqa: ANN401
    return key_for(task, backend, **call)


@pytest.mark.logic
@pytest.mark.parametrize(
    "builder,field",
    [
        (_drops_params, "params"),
        (_drops_inputs, "inputs"),
        (_drops_model, "model"),
        (_drops_profile_slices, "profile_slices"),
    ],
    ids=["drops_params", "drops_inputs", "drops_model", "drops_profile_slices"],
)
def test_a_key_builder_that_drops_a_component_fails_its_case(builder: Callable[..., LogicalKey], field: str) -> None:
    """The key cases go through the key builder, so one that ignores a call component fails that case only."""
    registry = _registry({})
    with pytest.raises(AssertionError, match=field):
        _case(registry, f"fake::key_changes_with_{field}", key_builder=builder)()
    _case(registry, "fake::key_changes_with_task", key_builder=builder)()


@pytest.mark.logic
@pytest.mark.parametrize("field", sorted(NOT_YET_WIRED))
def test_unwired_key_fields_skip_naming_their_owner(field: str) -> None:
    """A key field no call can set yet is listed, and its case skips with the owner rather than passing."""
    assert field in LogicalKey.model_fields
    with pytest.raises(unittest.SkipTest, match="owner"):
        _case(_registry({}), f"fake::key_changes_with_{field}")()


@pytest.mark.logic
def test_a_stack_without_a_runner_is_refused() -> None:
    """A backend on a stack with no contract runner fails case generation instead of getting fewer cases (P-34)."""
    with pytest.raises(NoContractRunner, match="'venv'"):
        contract_cases(_registry({}, {"stack": "venv"}))


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


def _seed_dependent(task: TaskSpec, backend: BackendSpec, *, inputs: dict[str, str], **call: Any) -> LogicalKey:  # noqa: ANN401
    salted = {port: "content:" + hashlib.sha256(str(hash(port)).encode()).hexdigest() for port in inputs}
    return key_for(task, backend, inputs=salted, **call)


def _seed_dependent_model(
    task: TaskSpec,
    backend: BackendSpec,
    *,
    model: ResolvedModel | None,
    **call: Any,  # noqa: ANN401
) -> LogicalKey:
    assert model is not None, "the base call carries a model"
    salted = model.model_copy(update={"identity": hashlib.sha256(str(hash(model.ref.id)).encode()).hexdigest()[:40]})
    return key_for(task, backend, model=salted, **call)


@pytest.mark.contract
def test_hash_seed_case_catches_a_seed_dependent_model_identity() -> None:
    """The base call carries a model, so a model identity that reads `hash()` fails the case."""
    case = _case(_registry({}), "fake::key_independent_of_hash_seed", key_builder=_seed_dependent_model)
    with pytest.raises(AssertionError, match="hash seed"):
        case()


@pytest.mark.logic
def test_hash_seed_case_reports_a_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """An interpreter that does not finish within the bound fails the case instead of stalling the suite."""
    from senselab_core.testing import contracts

    def hang(spec: Any, seed: str) -> Any:  # noqa: ANN401
        raise subprocess.TimeoutExpired(cmd="probe", timeout=contracts.HASH_SEED_TIMEOUT_S)

    monkeypatch.setattr(contracts, "_run_seed_probe", hang)
    with pytest.raises(AssertionError, match="took over"):
        _case(_registry({}), "fake::key_independent_of_hash_seed")()


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


@pytest.mark.contract
def test_a6_set_params_are_refused_and_caught_if_declared_anyway(monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe: a `set[str]` param keyed differently under each hash seed.

    The registry refuses the type; a registry that skipped that check would still fail the generated case.
    """
    with pytest.raises(declare.NonCanonicalParams, match="tags"):
        _registry({"params": _Tagged})
    monkeypatch.setattr(declare, "check_canonical_params", lambda model: None)
    case = _case(_registry({"params": _Tagged}), "fake::key_independent_of_hash_seed")
    with pytest.raises(AssertionError, match="hash seed"):
        case()


@pytest.mark.logic
def test_hash_seed_case_needs_importable_declarations() -> None:
    """A params model defined inside a function cannot be imported by name, and the case says so."""

    class Local(TaskParams):
        salt: str = ""

    with pytest.raises(AssertionError, match="module level"):
        _case(_registry({"params": Local}), "fake::key_independent_of_hash_seed")()
