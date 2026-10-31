"""Generated contract tests over the registry (testing.md §3), and checks that each contract can fail."""

from collections.abc import Callable
from typing import Any

import pytest
import senselab_core.runtime.fake_backend  # noqa: F401  registers fake.echo and its backend
from pydantic import BaseModel, ConfigDict
from senselab_core.kernel.declare import REGISTRY, BackendSpec, Registry, TaskParams, TaskSpec
from senselab_core.kernel.identity import ExecutionContext, LogicalKey
from senselab_core.runtime.fake_backend import FAKE
from senselab_core.testing.contracts import contract_cases


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


def _case(registry: Registry, name: str) -> Callable[[], None]:
    return dict(contract_cases(registry))[name]


@pytest.mark.logic
def test_a_good_declaration_passes_every_offline_case() -> None:
    """The fixture declaration is valid, so the failures below come from the one defect each introduces."""
    for name, case in contract_cases(_registry({})):
        if not name.endswith("content_fault_isolated"):
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
