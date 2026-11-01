"""The contract suite every declared task and backend gets by being declared (testing.md §3).

`contract_cases(registry)` returns `(id, case)` pairs; a case is a no-argument callable that raises
`AssertionError` (or the validation error it provokes) when the declaration breaks the contract.
Ids are `<task>::<case>` and `<backend>::<case>`; a backend name declared under several tasks is
written `<task>/<backend>`.

Task cases: `params_round_trip` and `params_reject_unknown`. Backend cases, each building keys
with `key_builder` (default `build_key`): `key_deterministic`; `key_changes_with_<field>` for every
`LogicalKey` field, except that a batch formation or chunk grid the backend does not declare output-affecting gets
`key_ignores_<field>`; `key_ignores_<field>` for every other `Placement` field and every
`ExecutionContext` field; `key_independent_of_hash_seed`; and `content_fault_isolated`, which runs
the backend's worker on its load stub.

`backend_license` derives the backend's code license from `lock_records(backend)` and checks every package record
and the result against the license table; it is skipped when no `lock_records` is given.

`key_independent_of_hash_seed` builds the base call's key in two interpreters started with
`PYTHONHASHSEED=1` and `PYTHONHASHSEED=2`. Each imports the params models and the key builder by
`__module__` and `__qualname__`, so they must be defined at module level.
"""

import functools
import importlib
import json
import os
import subprocess
import sys
import tempfile
import typing
import unittest
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, TypeVar

import numpy
from pydantic import BaseModel, ValidationError
from senselab_worker.digest import identity_digest, npy_digest
from senselab_worker.faults import FaultClass

from senselab_core.kernel._base import thawed
from senselab_core.kernel.declare import OUTPUT_AFFECTING, BackendSpec, Registry, TaskParams, TaskSpec
from senselab_core.kernel.identity import ExecutionContext, LogicalKey, ModelRef, ResolvedModel
from senselab_core.kernel.license import License
from senselab_core.runtime.keys import Call, Placement, build_key
from senselab_core.runtime.licenses import backend_license, check_license
from senselab_core.runtime.worker_client import WorkerClient

Case = Callable[[], None]
P = TypeVar("P", bound=BaseModel)
KeyBuilder = Callable[[Call, Placement, ExecutionContext], LogicalKey]
"""`build_key`'s call shape."""
LockRecords = Callable[[BackendSpec], Mapping[str, License | None]]
"""A backend's env-lock package records, `package name -> license record` (`None` for a package without one)."""

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

PLACEMENT_ONLY: tuple[str, ...] = tuple(
    f.name for f in fields(Placement) if f.name not in {"backend", "backend_params"} | OUTPUT_AFFECTING
)
"""`Placement` fields that never reach a key."""

WORKER_TIMEOUT_S = 60
"""How long `content_fault_isolated` gives a worker to load its stub, and then to answer one request."""


def _regular_series(item: str, port: str) -> dict[str, Any]:
    """Return the wire form of a one-channel regular group of 800 samples, its phase set by the item and port."""
    phase = int(identity_digest("contract-probe@1", [item, port])[:8], 16) % 1000
    data = numpy.sin(2.0 * numpy.pi * (numpy.arange(800) + phase) / 40.0).astype(numpy.float32)[None, :]
    return {
        "format": "regular-group@2.0",
        "channels": [{"name": "ch0", "type": "audio", "units": "FS", "placement": None, "reference": None}],
        "sampling_rate": 16000.0,
        "data": data,
        "source": None,
        "source_stamp": None,
    }


WIRE_INPUTS: Mapping[str, Callable[[str, str], object]] = {
    "json": lambda item, port: f"{item}:{port}",
    "RegularSeries": _regular_series,
}
"""Per input port type, the wire value `content_fault_isolated` sends for an item id and port."""

FAULTED_ITEM = "b"
"""The item `content_fault_isolated` makes the stub model fail, through `init["faults"]`."""


BASE_CONTEXT: Mapping[str, Any] = {
    "env_lock": "env-lock",
    "platform": "linux-x86_64",
    "wheel_set": "wheel-set",
    "numerics": {"class": "fp32"},
    "decoder": None,
    "driver": {"cuda": "12.8"},
    "host": "host",
    "protocol": "protocol",
    "model_source": "hf",
}
"""An execution context whose fields the context cases vary one at a time."""


def contract_cases(
    registry: Registry,
    *,
    key_builder: KeyBuilder = build_key,
    hash_seed_timeout_s: float = HASH_SEED_TIMEOUT_S,
    lock_records: LockRecords | None = None,
) -> list[tuple[str, Case]]:
    """Return the contract cases of every task and backend in `registry`, with their ids.

    Each `key_independent_of_hash_seed` interpreter may take `hash_seed_timeout_s` seconds.
    """
    names = Counter(b.name for t in registry.tasks() for b in registry.backends(t.name))
    cases: list[tuple[str, Case]] = []
    for task in registry.tasks():
        cases += [(f"{task.name}::{case}", functools.partial(fn, task)) for case, fn in _task_cases(task)]
        for backend in registry.backends(task.name):
            label = backend.name if names[backend.name] == 1 else f"{task.name}/{backend.name}"
            cases += [
                (f"{label}::{case}", functools.partial(fn, task, backend))
                for case, fn in _backend_cases(backend, key_builder, hash_seed_timeout_s, lock_records)
            ]
    return cases


PROBE_KEY = "contract_probe"
"""The `json_schema_extra` key under which a required param declares the sample value contract cases build it with."""


def probe_params(task: TaskSpec) -> TaskParams:
    """Return the task's params built from defaults, each required field taking its declared `PROBE_KEY` sample.

    Raises:
        AssertionError: if a required field declares no sample.
    """
    probes: dict[str, Any] = {}
    for name, field in task.params.model_fields.items():
        if field.is_required():
            extra = field.json_schema_extra if isinstance(field.json_schema_extra, dict) else {}
            assert PROBE_KEY in extra, f"required param {name!r} of {task.name!r} declares no {PROBE_KEY!r} sample"
            probes[name] = extra[PROBE_KEY]
    return task.params.model_validate(probes)


def _task_cases(task: TaskSpec) -> list[tuple[str, Callable[[TaskSpec], None]]]:
    return [("params_round_trip", _params_round_trip), ("params_reject_unknown", _params_reject_unknown)]


def _backend_cases(
    backend: BackendSpec, build: KeyBuilder, hash_seed_timeout_s: float, lock_records: LockRecords | None
) -> list[tuple[str, Callable[[TaskSpec, BackendSpec], None]]]:
    cases: list[tuple[str, Callable[[TaskSpec, BackendSpec], None]]] = [
        ("key_deterministic", functools.partial(_key_deterministic, build)),
    ]
    for f in LogicalKey.model_fields:
        if f in OUTPUT_AFFECTING and f not in backend.output_affecting:
            cases.append((f"key_ignores_{f}", functools.partial(_key_ignores_placement, build, f)))
        else:
            cases.append((f"key_changes_with_{f}", functools.partial(_key_changes_with, build, f)))
    cases += [(f"key_ignores_{f}", functools.partial(_key_ignores_placement, build, f)) for f in PLACEMENT_ONLY]
    cases += [
        (f"key_ignores_{f}", functools.partial(_key_ignores_context, build, f)) for f in ExecutionContext.model_fields
    ]
    cases.append(
        ("key_independent_of_hash_seed", functools.partial(_key_independent_of_hash_seed, build, hash_seed_timeout_s))
    )
    cases.append(("content_fault_isolated", _content_fault_isolated))
    cases.append(("backend_license", functools.partial(_backend_license, lock_records)))
    return cases


def _backend_license(lock_records: LockRecords | None, task: TaskSpec, backend: BackendSpec) -> None:
    if lock_records is None:
        raise unittest.SkipTest("no lock records supplied; nothing to derive a license from")
    records = lock_records(backend)
    for name, record in records.items():
        if record is not None:
            check_license(record)
    derived = backend_license(backend, records)
    assert check_license(derived) == derived, "the derived license does not match its expression"


def _params_round_trip(task: TaskSpec) -> None:
    params = probe_params(task)
    assert task.params.model_validate_json(params.model_dump_json()) == params, "params do not survive JSON"
    assert task.params.model_validate(params.model_dump()) == params, "params do not survive a dict"
    assert task.params.model_json_schema()["type"] == "object"


def _params_reject_unknown(task: TaskSpec) -> None:
    known = list(task.params.model_fields)
    probe = f"{known[0]}_" if known else "unknown_param"
    try:
        task.params.model_validate({**probe_params(task).model_dump(), probe: 0})
    except ValidationError as e:
        message = str(e)
    else:
        raise AssertionError(f"{task.params.__name__} accepted the unknown key {probe!r}")
    assert probe in message, f"the error does not name the unknown key {probe!r}: {message}"
    if known:
        assert "did you mean" in message and repr(known[0]) in message, f"no suggestion of {known[0]!r}: {message}"


def base_call(task: TaskSpec, backend: BackendSpec) -> tuple[Call, Placement]:
    """Return the call and placement every key case starts from: default params, `BASE_MODEL`, every component set."""
    call = Call(
        task=task,
        params=probe_params(task),
        inputs={port: "content:" + identity_digest("contract-probe@1", ["input", port]) for port in task.inputs},
        model=BASE_MODEL,
    )
    placement = Placement(
        backend=backend,
        backend_params=backend.params(),
        device=backend.devices[0],
        batch_formation="contract-batches",
        chunk_grid="contract-grid",
    )
    return call, placement


BASE = ExecutionContext.model_validate(BASE_CONTEXT)
"""`BASE_CONTEXT` as the context every key case builds under."""


def _key_deterministic(build: KeyBuilder, task: TaskSpec, backend: BackendSpec) -> None:
    call, placement = base_call(task, backend)
    key = build(call, placement, BASE)
    assert build(*base_call(task, backend), BASE).digest() == key.digest(), "the same call gave two keys"
    again = build(
        replace(
            call,
            params=task.params.model_validate_json(call.params.model_dump_json()),
            inputs=dict(reversed(list(call.inputs.items()))),
        ),
        replace(
            placement, backend_params=backend.params.model_validate_json(placement.backend_params.model_dump_json())
        ),
        BASE,
    )
    assert again.digest() == key.digest(), "input order or a params JSON round trip changed the key"


def _literal_choices(annotation: object) -> list[object]:
    """Return the values the `Literal`s in `annotation` admit, through unions; empty for any other type."""
    if typing.get_origin(annotation) is typing.Literal:
        return list(typing.get_args(annotation))
    return [c for arg in typing.get_args(annotation) for c in _literal_choices(arg)]


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


def _varied_params(model: type[P], params: P) -> P:
    dumped = params.model_dump(mode="json")
    for name, value in dumped.items():
        try:
            varied = model.model_validate({**dumped, name: _perturbed(name, value)})
        except (ValidationError, TypeError):
            continue
        if varied != params:
            return varied
    raise unittest.SkipTest(f"no field of {model.__name__} can be varied generically; nothing to vary")


def _varied_call(field: str, call: Call, placement: Placement) -> tuple[Call, Placement]:  # noqa: PLR0911 - one return per key field
    """Return the call and placement with the one declaration or argument behind key field `field` changed."""
    task, backend = call.task, placement.backend
    if field in ("task", "task_version"):
        attr = "name" if field == "task" else "version"
        changed = task.model_copy(update={attr: getattr(task, attr) + "~"})
        return replace(call, task=changed), replace(
            placement, backend=backend.model_copy(update={"task": changed.name})
        )
    if field in ("backend", "backend_version"):
        attr = "name" if field == "backend" else "version"
        return call, replace(placement, backend=backend.model_copy(update={attr: getattr(backend, attr) + "~"}))
    if field == "params":
        return replace(call, params=_varied_params(task.params, call.params)), placement
    if field == "backend_params":
        return call, replace(placement, backend_params=_varied_params(backend.params, placement.backend_params))
    if field == "inputs":
        if not call.inputs:
            raise unittest.SkipTest(f"{task.name!r} declares no inputs; nothing to vary")
        port = sorted(call.inputs)[0]
        varied = "content:" + identity_digest("contract-probe@1", ["varied"])
        return replace(call, inputs={**call.inputs, port: varied}), placement
    if field == "model":
        return replace(call, model=VARIED_MODEL), placement
    if field == "profile_slices":
        return replace(
            call, profile_slices={"profile": "profile:" + identity_digest("contract-probe@1", ["slice"])}
        ), placement
    if field in OUTPUT_AFFECTING:
        return call, _placed(placement, field, f"{getattr(placement, field)}~")
    raise AssertionError(f"LogicalKey.{field} has no contract variation: add it to _varied_call or KEY_FORMAT_FIELDS")


def _placed(placement: Placement, field: str, value: object) -> Placement:
    changes: dict[str, Any] = {field: value}
    return replace(placement, **changes)


def _key_changes_with(build: KeyBuilder, field: str, task: TaskSpec, backend: BackendSpec) -> None:
    call, placement = base_call(task, backend)
    key = build(call, placement, BASE)
    if field in KEY_FORMAT_FIELDS:
        dumped = key.model_dump()
        changed = LogicalKey.model_validate({**dumped, field: _perturbed(field, dumped[field])})
    else:
        changed = build(*_varied_call(field, call, placement), BASE)
    assert changed.digest() != key.digest(), f"varying the {field} of a call left its key unchanged"


def _key_ignores_placement(build: KeyBuilder, field: str, task: TaskSpec, backend: BackendSpec) -> None:
    call, placement = base_call(task, backend)
    varied = _placed(placement, field, _perturbed(field, getattr(placement, field)))
    assert build(call, varied, BASE).digest() == build(call, placement, BASE).digest(), (
        f"the placement's {field} reached the key"
    )


def _key_ignores_context(build: KeyBuilder, field: str, task: TaskSpec, backend: BackendSpec) -> None:
    assert field not in LogicalKey.model_fields, f"context field {field!r} is also a key field"
    value = BASE.model_dump()[field]
    choices = _literal_choices(ExecutionContext.model_fields[field].annotation)
    varied = next(c for c in choices if c != value) if choices else _perturbed(field, value)
    other = ExecutionContext.model_validate({**BASE_CONTEXT, field: varied})
    assert other != BASE, f"ExecutionContext.{field} did not vary"
    call, placement = base_call(task, backend)
    assert build(call, placement, other).digest() == build(call, placement, BASE).digest(), (
        f"the execution context's {field} reached the key"
    )


_SEED_PROBE = """
import importlib, json, sys
spec = json.loads(sys.stdin.read())
sys.path[:] = spec["path"]
from senselab_core.kernel.declare import BackendSpec, TaskSpec
from senselab_core.testing.contracts import BASE, base_call, import_qualified
task = TaskSpec(**spec["task"], params=import_qualified(*spec["params"]))
backend = BackendSpec(**spec["backend"], params=import_qualified(*spec["backend_params"]))
print(import_qualified(*spec["builder"])(*base_call(task, backend), BASE).digest())
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


def _key_independent_of_hash_seed(build: KeyBuilder, timeout: float, task: TaskSpec, backend: BackendSpec) -> None:
    spec = {
        "path": sys.path,
        "task": task.model_dump(mode="json", exclude={"params"}),
        "params": _qualified(task.params, "params model"),
        "backend": backend.model_dump(mode="json", exclude={"params"}),
        "backend_params": _qualified(backend.params, "backend params model"),
        "builder": _qualified(build, "key builder"),
    }
    digests = {}
    for seed in HASH_SEEDS:
        try:
            proc = _run_seed_probe(spec, seed, timeout)
        except subprocess.TimeoutExpired as err:
            raise AssertionError(f"building the key under PYTHONHASHSEED={seed} took over {timeout} s") from err
        assert proc.returncode == 0, f"building the key under PYTHONHASHSEED={seed} failed:\n{proc.stderr}"
        digests[seed] = proc.stdout.strip()
    assert len(set(digests.values())) == 1, f"the key depends on the hash seed: {digests}"


def _run_seed_probe(spec: dict[str, Any], seed: str, timeout: float) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(
        [sys.executable, "-c", _SEED_PROBE],
        input=json.dumps(spec),
        env={**os.environ, "PYTHONHASHSEED": seed},
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def _content_fault_isolated(task: TaskSpec, backend: BackendSpec) -> None:
    """Run the backend's worker on its load stub; a content fault on one item leaves its batch-mates unchanged."""
    missing = sorted(set(task.inputs.values()) - set(WIRE_INPUTS))
    assert not missing, f"input port types {missing} of {task.name!r} have no contract input; add them to WIRE_INPUTS"
    items = [
        {"id": i, "inputs": {port: WIRE_INPUTS[t](i, port) for port, t in task.inputs.items()}} for i in ("a", "b", "c")
    ]
    params = probe_params(task).model_dump(mode="json")
    stub = thawed(backend.load_stub)
    with tempfile.TemporaryDirectory(prefix="senselab-contract-") as scratch:
        clean = _run_worker(backend, {**stub, "faults": {}}, items, params, Path(scratch))
        faulted = _run_worker(backend, {**stub, "faults": {"fail_items": [FAULTED_ITEM]}}, items, params, Path(scratch))
        assert not any(Path(scratch).iterdir()), "the worker's array directory was left behind"
    assert all("outputs" in r for r in clean), f"a fault-free run failed: {clean}"
    assert faulted[1].get("error", {}).get("fault") is FaultClass.CONTENT, f"item b did not fail as content: {faulted}"
    assert _comparable((faulted[0], faulted[2])) == _comparable((clean[0], clean[2])), (
        "a content fault on b changed its batch-mates"
    )


def _comparable(value: object) -> object:
    """Return `value` with each array replaced by its `npy_digest`, so results holding arrays compare with `==`."""
    if isinstance(value, numpy.ndarray):
        return npy_digest("npy@1", value)
    if isinstance(value, Mapping):
        return {k: _comparable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_comparable(v) for v in value]
    return value


def _run_worker(
    backend: BackendSpec, init: dict[str, Any], items: list[dict[str, Any]], params: dict[str, Any], scratch: Path
) -> list[dict[str, Any]]:
    with WorkerClient(
        sys.executable,
        backend.worker,
        init=init,
        array_parent=scratch,
        load_timeout=WORKER_TIMEOUT_S,
        request_timeout=WORKER_TIMEOUT_S,
    ) as worker:
        return worker.run(items, params)
