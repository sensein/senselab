"""The contract suite every declared task and backend gets by being declared (testing.md §3).

`contract_cases(registry)` returns `(id, case)` pairs; a case is a no-argument callable that raises
`AssertionError` (or the validation error it provokes) when the declaration breaks the contract.
Ids are `<task>::<case>` and `<backend>::<case>`; a backend name declared under several tasks is
written `<task>/<backend>`.

Task cases: `params_round_trip`, `params_reject_unknown`, and for `measure` tasks
`measure_has_no_verdict` (P-20). Backend cases: `license_present`, `key_deterministic`,
`key_changes_with_<field>` for every `LogicalKey` field, `key_ignores_<field>` for every
`ExecutionContext` field, and for the `fake` stack `content_fault_isolated`.
"""

import functools
import inspect
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from senselab_worker.fake import FakeFaults
from senselab_worker.faults import FaultClass
from senselab_worker.hashing import sha256_bytes

from senselab_core.kernel.declare import VERDICT_TYPES, BackendSpec, Registry, TaskSpec, key_for
from senselab_core.kernel.identity import ExecutionContext, LogicalKey
from senselab_core.kernel.license import License, require_license
from senselab_core.runtime.fake_backend import run_fake

Case = Callable[[], None]

BASE_CONTEXT: Mapping[str, Any] = {
    "env_lock": "env-lock",
    "platform": "linux-x86_64",
    "wheel_set": "wheel-set",
    "numerics": {"class": "fp32"},
    "decoder": None,
    "driver": {"cuda": "12.8"},
    "host": "host",
    "protocol": "protocol",
}
"""An execution context whose fields the context cases vary one at a time."""


def contract_cases(registry: Registry) -> list[tuple[str, Case]]:
    """Return the contract cases of every task and backend in `registry`, with their ids."""
    names = Counter(b.name for t in registry.tasks() for b in registry.backends(t.name))
    cases: list[tuple[str, Case]] = []
    for task in registry.tasks():
        cases += [(f"{task.name}::{case}", functools.partial(fn, task)) for case, fn in _task_cases(task)]
        for backend in registry.backends(task.name):
            label = backend.name if names[backend.name] == 1 else f"{task.name}/{backend.name}"
            cases += [
                (f"{label}::{case}", functools.partial(fn, task, backend)) for case, fn in _backend_cases(backend)
            ]
    return cases


def _task_cases(task: TaskSpec) -> list[tuple[str, Callable[[TaskSpec], None]]]:
    cases: list[tuple[str, Callable[[TaskSpec], None]]] = [
        ("params_round_trip", _params_round_trip),
        ("params_reject_unknown", _params_reject_unknown),
    ]
    if task.kind == "measure":
        cases.append(("measure_has_no_verdict", _measure_has_no_verdict))
    return cases


def _backend_cases(backend: BackendSpec) -> list[tuple[str, Callable[[TaskSpec, BackendSpec], None]]]:
    cases: list[tuple[str, Callable[[TaskSpec, BackendSpec], None]]] = [
        ("license_present", _license_present),
        ("key_deterministic", _key_deterministic),
    ]
    cases += [(f"key_changes_with_{f}", functools.partial(_key_changes_with, f)) for f in LogicalKey.model_fields]
    cases += [(f"key_ignores_{f}", functools.partial(_key_ignores, f)) for f in ExecutionContext.model_fields]
    if backend.stack == "fake":
        cases.append(("content_fault_isolated", _content_fault_isolated))
    return cases


def _params_round_trip(task: TaskSpec) -> None:
    params = task.params()
    assert task.params.model_validate_json(params.model_dump_json()) == params, "params do not survive JSON"
    assert task.params.model_validate(params.model_dump()) == params, "params do not survive a dict"
    assert task.params.model_json_schema()["type"] == "object"


def _params_reject_unknown(task: TaskSpec) -> None:
    known = list(task.params.model_fields)
    probe = f"{known[0]}_" if known else "unknown_param"
    try:
        task.params.model_validate({**task.params().model_dump(), probe: 0})
    except ValidationError as e:
        message = str(e)
    else:
        raise AssertionError(f"{task.params.__name__} accepted the unknown key {probe!r}")
    assert probe in message, f"the error does not name the unknown key {probe!r}: {message}"
    if known:
        assert "did you mean" in message and repr(known[0]) in message, f"no suggestion of {known[0]!r}: {message}"


def _measure_has_no_verdict(task: TaskSpec) -> None:
    verdicts = {port: t for port, t in task.outputs.items() if t in VERDICT_TYPES}
    assert not verdicts, f"measure task {task.name!r} outputs verdicts {verdicts}; declare a decide task (P-20)"


def _license_present(task: TaskSpec, backend: BackendSpec) -> None:
    assert isinstance(require_license(backend.license, f"backend {backend.name!r}"), License)


def _base_key(task: TaskSpec, backend: BackendSpec) -> LogicalKey:
    inputs = {port: sha256_bytes(port.encode()) for port in task.inputs}
    return key_for(task, backend, params=task.params(), inputs=inputs)


def _key_deterministic(task: TaskSpec, backend: BackendSpec) -> None:
    key = _base_key(task, backend)
    assert _base_key(task, backend).digest() == key.digest(), "the same call gave two keys"
    reordered = dict(reversed(list(key.inputs.items())))
    again = key_for(
        task, backend, params=task.params.model_validate_json(task.params().model_dump_json()), inputs=reordered
    )
    assert again.digest() == key.digest(), "input order or a params JSON round trip changed the key"


def _perturbed(field: str, value: object) -> object:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, str):
        return value + "~"
    if value is None:
        return "perturbed"
    if isinstance(value, Mapping):
        return {**value, "~": "perturbed"}
    raise TypeError(f"no perturbation for field {field!r} holding {type(value).__name__}; extend _perturbed")


def _key_changes_with(field: str, task: TaskSpec, backend: BackendSpec) -> None:
    key = _base_key(task, backend)
    fields = key.model_dump()
    changed = LogicalKey.model_validate({**fields, field: _perturbed(field, fields[field])})
    assert changed.digest() != key.digest(), f"changing LogicalKey.{field} left the key unchanged"


def _key_ignores(field: str, task: TaskSpec, backend: BackendSpec) -> None:
    """No path from a context field to the key: not a key field, and not an argument of `key_for`."""
    assert field not in LogicalKey.model_fields, f"context field {field!r} is also a key field"
    assert field not in inspect.signature(key_for).parameters, f"key_for takes context field {field!r}"
    base = ExecutionContext.model_validate(BASE_CONTEXT)
    other = ExecutionContext.model_validate({**BASE_CONTEXT, field: _perturbed(field, base.model_dump()[field])})
    assert other != base, f"ExecutionContext.{field} did not vary"


def _content_fault_isolated(task: TaskSpec, backend: BackendSpec) -> None:
    items = [{"id": i, "inputs": {port: f"{i}:{port}" for port in task.inputs}} for i in ("a", "b", "c")]
    params = task.params().model_dump(mode="json")
    with tempfile.TemporaryDirectory(prefix="senselab-contract-") as scratch:
        clean_dir, faulted_dir = Path(scratch) / "clean", Path(scratch) / "faulted"
        clean = run_fake(items, params, FakeFaults(), array_dir=clean_dir)
        faulted = run_fake(items, params, FakeFaults(fail_items=frozenset({"b"})), array_dir=faulted_dir)
        assert not clean_dir.exists() and not faulted_dir.exists(), "the worker's array directory was left behind"
    assert all("outputs" in r for r in clean), f"a fault-free run failed: {clean}"
    assert faulted[1].get("error", {}).get("fault") is FaultClass.CONTENT, f"item b did not fail as content: {faulted}"
    assert (faulted[0], faulted[2]) == (clean[0], clean[2]), "a content fault on b changed its batch-mates"
