"""The contract suite every declared task and backend gets by being declared (testing.md §3).

`contract_cases(registry)` returns `(id, case)` pairs; a case is a no-argument callable that raises
`AssertionError` (or the validation error it provokes) when the declaration breaks the contract, or
`unittest.SkipTest` naming its owner for a key component no call can set yet (`NOT_YET_WIRED`).
Ids are `<task>::<case>` and `<backend>::<case>`; a backend name declared under several tasks is
written `<task>/<backend>`.

Task cases: `params_round_trip`, `params_reject_unknown`, and for `measure` tasks
`measure_has_no_verdict` (P-20). Backend cases: `license_present`, `key_deterministic`,
`key_changes_with_<field>` for every `LogicalKey` field, `key_ignores_<field>` for every
`ExecutionContext` field, `key_independent_of_hash_seed`, and `content_fault_isolated` by the
stack's runner in `FAULT_RUNNERS`. The key cases build keys with `key_builder` (default `key_for`).

`key_independent_of_hash_seed` builds the base call's key in two interpreters started with
`PYTHONHASHSEED=1` and `PYTHONHASHSEED=2`. Each imports the params model and the key builder by
`__module__` and `__qualname__`, so both must be defined at module level.
"""

import functools
import importlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError
from senselab_worker.fake import FakeFaults
from senselab_worker.faults import FaultClass
from senselab_worker.hashing import sha256_bytes

from senselab_core.kernel.declare import VERDICT_TYPES, BackendSpec, Registry, TaskSpec, key_for
from senselab_core.kernel.identity import ExecutionContext, LogicalKey, ModelRef, ResolvedModel
from senselab_core.kernel.license import License, require_license

Case = Callable[[], None]
KeyBuilder = Callable[..., LogicalKey]
"""`key_for`'s call shape: `(task, backend, *, params, inputs, model=None, profile_slices=None)`."""

HASH_SEEDS: tuple[str, str] = ("1", "2")
"""The `PYTHONHASHSEED` values `key_independent_of_hash_seed` builds a key under."""

HASH_SEED_TIMEOUT_S = 120
"""How long one `key_independent_of_hash_seed` interpreter may take to import and build a key."""

BASE_MODEL = ResolvedModel(ref=ModelRef.parse("contract/model@main"), identity="0" * 40)
"""The pinned model every key case's base call carries."""

VARIED_MODEL = ResolvedModel(ref=ModelRef.parse("contract/model@main"), identity="1" * 40)
"""The pinned model `key_changes_with_model` puts in place of `BASE_MODEL`."""

KEY_FORMAT_FIELDS: frozenset[str] = frozenset({"schema_version"})
"""Key fields fixed by the key format rather than by a call; their case varies the key directly."""

NOT_YET_WIRED: Mapping[str, str] = {
    "batch_formation": "BackendSpec.output_affecting does not reach key_for yet; owner: Phase 2 planner (ADR-004)",
    "chunk_grid": "BackendSpec.output_affecting does not reach key_for yet; owner: Phase 2 planner (ADR-004)",
}
"""Key fields no call can set yet, with the reason and the owner; their cases skip with that text."""


class NoContractRunner(LookupError):
    """A backend's stack has no runner for the runtime contract cases."""


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


def contract_cases(registry: Registry, *, key_builder: KeyBuilder = key_for) -> list[tuple[str, Case]]:
    """Return the contract cases of every task and backend in `registry`, with their ids.

    Raises:
        NoContractRunner: if a backend's stack has no entry in `FAULT_RUNNERS`.
    """
    for task in registry.tasks():
        for backend in registry.backends(task.name):
            if backend.stack not in FAULT_RUNNERS:
                raise NoContractRunner(
                    f"backend {backend.name!r} of {task.name!r} has stack {backend.stack!r}, which has no contract "
                    f"runner; runners exist for {sorted(FAULT_RUNNERS)}"
                )
    names = Counter(b.name for t in registry.tasks() for b in registry.backends(t.name))
    cases: list[tuple[str, Case]] = []
    for task in registry.tasks():
        cases += [(f"{task.name}::{case}", functools.partial(fn, task)) for case, fn in _task_cases(task)]
        for backend in registry.backends(task.name):
            label = backend.name if names[backend.name] == 1 else f"{task.name}/{backend.name}"
            cases += [
                (f"{label}::{case}", functools.partial(fn, task, backend))
                for case, fn in _backend_cases(backend, key_builder)
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


def _backend_cases(
    backend: BackendSpec, build: KeyBuilder
) -> list[tuple[str, Callable[[TaskSpec, BackendSpec], None]]]:
    cases: list[tuple[str, Callable[[TaskSpec, BackendSpec], None]]] = [
        ("license_present", _license_present),
        ("key_deterministic", functools.partial(_key_deterministic, build)),
    ]
    cases += [
        (f"key_changes_with_{f}", functools.partial(_key_changes_with, build, f)) for f in LogicalKey.model_fields
    ]
    cases += [(f"key_ignores_{f}", functools.partial(_key_ignores, build, f)) for f in ExecutionContext.model_fields]
    cases.append(("key_independent_of_hash_seed", functools.partial(_key_independent_of_hash_seed, build)))
    cases.append(("content_fault_isolated", FAULT_RUNNERS[backend.stack]))
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


def base_call(task: TaskSpec) -> dict[str, Any]:
    """Return the key builder arguments every key case starts from: default params and `BASE_MODEL`."""
    return {
        "params": task.params(),
        "inputs": {port: "content:" + sha256_bytes(port.encode()) for port in task.inputs},
        "model": BASE_MODEL,
        "profile_slices": {},
    }


def _key_deterministic(build: KeyBuilder, task: TaskSpec, backend: BackendSpec) -> None:
    call = base_call(task)
    key = build(task, backend, **call)
    assert build(task, backend, **base_call(task)).digest() == key.digest(), "the same call gave two keys"
    again = build(
        task,
        backend,
        **{
            **call,
            "params": task.params.model_validate_json(call["params"].model_dump_json()),
            "inputs": dict(reversed(list(call["inputs"].items()))),
        },
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


def _varied_params(task: TaskSpec, params: BaseModel) -> BaseModel:
    dumped = params.model_dump(mode="json")
    for name, value in dumped.items():
        try:
            varied = task.params.model_validate({**dumped, name: _perturbed(name, value)})
        except (ValidationError, TypeError):
            continue
        if varied != params:
            return varied
    raise unittest.SkipTest(f"no field of {task.params.__name__} can be varied generically; nothing to vary")


def _varied_call(
    field: str, task: TaskSpec, backend: BackendSpec, call: dict[str, Any]
) -> tuple[TaskSpec, BackendSpec, dict[str, Any]]:
    """Return the call with the one declaration or argument behind key field `field` changed."""
    if field in ("task", "task_version"):
        attr = "name" if field == "task" else "version"
        changed = task.model_copy(update={attr: getattr(task, attr) + "~"})
        return changed, backend.model_copy(update={"task": changed.name}), call
    if field in ("backend", "backend_version"):
        attr = "name" if field == "backend" else "version"
        return task, backend.model_copy(update={attr: getattr(backend, attr) + "~"}), call
    if field == "params":
        return task, backend, {**call, "params": _varied_params(task, call["params"])}
    if field == "inputs":
        if not call["inputs"]:
            raise unittest.SkipTest(f"{task.name!r} declares no inputs; nothing to vary")
        port = sorted(call["inputs"])[0]
        return task, backend, {**call, "inputs": {**call["inputs"], port: "content:" + sha256_bytes(b"varied")}}
    if field == "model":
        return task, backend, {**call, "model": VARIED_MODEL}
    if field == "profile_slices":
        return task, backend, {**call, "profile_slices": {"profile": "profile:" + sha256_bytes(b"slice")}}
    raise AssertionError(
        f"LogicalKey.{field} has no contract variation: add it to _varied_call, KEY_FORMAT_FIELDS or NOT_YET_WIRED"
    )


def _key_changes_with(build: KeyBuilder, field: str, task: TaskSpec, backend: BackendSpec) -> None:
    if field in NOT_YET_WIRED:
        raise unittest.SkipTest(f"LogicalKey.{field} not yet wired: {NOT_YET_WIRED[field]}")
    call = base_call(task)
    key = build(task, backend, **call)
    if field in KEY_FORMAT_FIELDS:
        fields = key.model_dump()
        changed = LogicalKey.model_validate({**fields, field: _perturbed(field, fields[field])})
    else:
        varied_task, varied_backend, varied = _varied_call(field, task, backend, call)
        changed = build(varied_task, varied_backend, **varied)
    assert changed.digest() != key.digest(), f"varying the {field} of a call left its key unchanged"


def _key_ignores(build: KeyBuilder, field: str, task: TaskSpec, backend: BackendSpec) -> None:
    """No path from a context field to the key: not a key field, and not an argument of the key builder."""
    assert field not in LogicalKey.model_fields, f"context field {field!r} is also a key field"
    assert field not in inspect.signature(build).parameters, f"the key builder takes context field {field!r}"
    base = ExecutionContext.model_validate(BASE_CONTEXT)
    other = ExecutionContext.model_validate({**BASE_CONTEXT, field: _perturbed(field, base.model_dump()[field])})
    assert other != base, f"ExecutionContext.{field} did not vary"


_SEED_PROBE = """
import importlib, json, sys
spec = json.loads(sys.stdin.read())
sys.path[:] = spec["path"]
from senselab_core.kernel.declare import BackendSpec, TaskSpec
from senselab_core.testing.contracts import base_call, import_qualified
task = TaskSpec(**spec["task"], params=import_qualified(*spec["params"]))
backend = BackendSpec.model_validate(spec["backend"])
print(import_qualified(*spec["builder"])(task, backend, **base_call(task)).digest())
"""


def import_qualified(module: str, qualname: str) -> Any:  # noqa: ANN401
    """Return the object named `qualname` in `module`, importing the module."""
    obj: Any = importlib.import_module(module)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _qualified(obj: Any, what: str) -> list[str]:  # noqa: ANN401
    module, qualname = getattr(obj, "__module__", None), getattr(obj, "__qualname__", None)
    if not module or not qualname or "<" in qualname:
        raise AssertionError(f"the {what} {obj!r} is not importable by name; define it at module level")
    return [module, qualname]


def _key_independent_of_hash_seed(build: KeyBuilder, task: TaskSpec, backend: BackendSpec) -> None:
    spec = {
        "path": sys.path,
        "task": task.model_dump(mode="json", exclude={"params"}),
        "params": _qualified(task.params, "params model"),
        "backend": backend.model_dump(mode="json"),
        "builder": _qualified(build, "key builder"),
    }
    digests = {}
    for seed in HASH_SEEDS:
        try:
            proc = _run_seed_probe(spec, seed)
        except subprocess.TimeoutExpired as err:
            raise AssertionError(
                f"building the key under PYTHONHASHSEED={seed} took over {HASH_SEED_TIMEOUT_S} s"
            ) from err
        assert proc.returncode == 0, f"building the key under PYTHONHASHSEED={seed} failed:\n{proc.stderr}"
        digests[seed] = proc.stdout.strip()
    assert len(set(digests.values())) == 1, f"the key depends on the hash seed: {digests}"


def _run_seed_probe(spec: dict[str, Any], seed: str) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(
        [sys.executable, "-c", _SEED_PROBE],
        input=json.dumps(spec),
        env={**os.environ, "PYTHONHASHSEED": seed},
        capture_output=True,
        text=True,
        check=False,
        timeout=HASH_SEED_TIMEOUT_S,
    )


def _content_fault_isolated(task: TaskSpec, backend: BackendSpec) -> None:
    from senselab_core.runtime.fake_backend import run_fake

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


FAULT_RUNNERS: Mapping[str, Callable[[TaskSpec, BackendSpec], None]] = {"fake": _content_fault_isolated}
"""Per stack, the case that runs a backend's worker and checks per-item fault isolation."""
