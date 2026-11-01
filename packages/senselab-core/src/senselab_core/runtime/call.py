"""`run_call`: one task call resolved, placed, keyed, served from the result store or executed, and recorded.

A declaring package hands `run_call` its catalogue and how its backends' envs open with `use_packaged`, as it
declares its tasks in `REGISTRY` on import. Until one does, the catalogue is empty and no env opens.
"""

import atexit
import functools
import importlib
import platform
import sysconfig
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from os import PathLike
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, PlainSerializer, model_validator
from senselab_worker import WIRE_VERSION
from senselab_worker.canonical import jcs_dumps
from senselab_worker.model_cache import ModelCache, ModelKey

import senselab_core.kernel.results  # noqa: F401 - importing it defines the result types
import senselab_core.kernel.time  # noqa: F401 - importing it defines the time types
from senselab_core.kernel import prov as kprov
from senselab_core.kernel._base import Frozen, FrozenMapping, NonEmptyStr, value_digest
from senselab_core.kernel.declare import REGISTRY, BackendSpec, TaskParams, TaskSpec
from senselab_core.kernel.formats import Formatted
from senselab_core.kernel.identity import (
    DEFAULT_REUSE_POLICY,
    ExecutionContext,
    InputIdentity,
    LogicalKey,
    ModelRef,
    OutputIdentity,
    ResolvedModel,
    ReusePolicy,
)
from senselab_core.kernel.signals import Recording
from senselab_core.runtime.audio_io import open_audio
from senselab_core.runtime.catalogue import Catalogue, CatalogueEntry, UnknownModel
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.hardware import HardwareInfo, numerics_class, probe_hardware
from senselab_core.runtime.in_process import InProcessWorker
from senselab_core.runtime.keys import Call, Placement, build_key
from senselab_core.runtime.models import resolve
from senselab_core.runtime.profiles import ExecutionProfile, resolve_execution
from senselab_core.runtime.revision import cache_root
from senselab_core.runtime.store import ResultStore, context_record, key_record
from senselab_core.runtime.vocabulary import core_vocabulary
from senselab_core.runtime.worker_client import WorkerClient

PRECISION = "default"
"""The precision a call records: the backend's own; no backend declares another yet."""
ITEM_ID = "call"
_READERS: Mapping[str, Callable[[str | PathLike[str]], Recording]] = {"Audio": open_audio}


def kernel_types() -> dict[str, type]:
    """Return the kernel value types a port may name, by name: every `Recording` and `Formatted` subclass."""
    found: dict[str, type] = {Recording.__name__: Recording}
    pending: list[type] = [Recording, Formatted]
    while pending:
        for sub in pending.pop().__subclasses__():
            found[sub.__name__] = sub
            pending.append(sub)
    return found


Inputs = Annotated[Mapping[str, Any], AfterValidator(FrozenMapping), PlainSerializer(dict)]


class TaskCall(BaseModel):
    """A validated call: the task and its version, one input per declared port, validated params, the model.

    Built from a mapping (`{task, inputs, params, model, backend_params}`, the one call shape), `params` is
    validated against the declared task's params, `model` is parsed from its reference string and an absent
    `task_version` is the declared one. `backend_params` are held as given and validated against the backend
    `run_call` chooses.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    task: NonEmptyStr
    task_version: NonEmptyStr
    inputs: Inputs
    params: TaskParams
    model: ModelRef | None = None
    backend_params: Inputs = FrozenMapping({})

    @model_validator(mode="before")
    @classmethod
    def _from_call_shape(cls, data: Any) -> Any:  # noqa: ANN401 - pydantic's hook signature
        if not isinstance(data, Mapping) or not isinstance(data.get("task"), str):
            return data
        out = dict(data)
        if "task_version" not in out or not isinstance(out.get("params"), TaskParams):
            spec = REGISTRY.spec(out["task"])
            out.setdefault("task_version", spec.version)
            if not isinstance(out.get("params"), TaskParams):
                out["params"] = spec.params.model_validate(out.get("params") or {})
        if isinstance(out.get("model"), str):
            out["model"] = ModelRef.parse(out["model"])
        return out


@dataclass(frozen=True)
class RunConfig:
    """A run's settings: `reuse`, the widest policy under which a stored result serves a call."""

    reuse: ReusePolicy = DEFAULT_REUSE_POLICY


@dataclass(frozen=True)
class Result:
    """One call's result: its outputs, their identity, the key, the context that computed them and the placement.

    `reused` is the policy that admitted a stored result, or None when the call executed.
    """

    outputs: dict[str, object]
    output_identity: OutputIdentity
    key: LogicalKey
    context: ExecutionContext
    placement: Placement
    reused: ReusePolicy | None


class CallFailed(RuntimeError):
    """The backend failed the call's item; `error` is the worker's error object, with its `fault` class."""

    def __init__(self, error: dict[str, Any]) -> None:
        """Wrap the error object."""
        super().__init__(f"{error.get('type')} ({error.get('fault')}): {error.get('message')}")
        self.error = error


EnvOpener = Callable[[BackendSpec, ExecutionProfile], EnvHandle]
"""Open the env a backend declares under a profile, building it if needed."""


@dataclass(frozen=True)
class Packaged:
    """What a declaring package hands `run_call`: its catalogue and how its backends' envs open."""

    catalogue: Catalogue
    open_env: EnvOpener


def _no_env(backend: BackendSpec, profile: ExecutionProfile) -> EnvHandle:  # noqa: ARG001 - an EnvOpener
    raise LookupError(f"no package has said how to open env {backend.env.kind!r} of backend {backend.name!r}")


_PACKAGED = Packaged(Catalogue(entries=()), _no_env)


def use_packaged(packaged: Packaged) -> Packaged:
    """Make `packaged` what `run_call` reads, and return what it read before."""
    global _PACKAGED  # noqa: PLW0603 - the process-wide counterpart of REGISTRY
    previous, _PACKAGED = _PACKAGED, packaged
    return previous


def packaged() -> Packaged:
    """Return what `run_call` reads: the catalogue and env opener last given to `use_packaged`."""
    return _PACKAGED


def run_call(
    call: TaskCall,
    *,
    execution: ExecutionProfile | None = None,
    run: RunConfig | None = None,
    store: ResultStore | None = None,
) -> Result:
    """Run `call` once: serve it from `store` under `run.reuse`, else execute it, store the outputs and record it.

    The model is the catalogue's pin for a catalogue model at that revision (the task's catalogue default when the
    call names none), else `resolve()`d. `resolve_execution` places the call under `execution`; the backend's env
    opens through the packaged opener and its worker is reused across calls (one model per worker process; host
    backends share one `ModelCache` per budget). An execution writes one activity, associated with an agent carrying
    `senselab:modelSource`, recording the key, the context and the profile's digest; a store hit writes none (the
    store records its reuse). `None` means the packaged default: the `laptop` profile, `RunConfig()` and the store
    under `cache_root()/results`.

    Raises:
        UnknownTask: the call's task is not declared.
        UnlistedSource, NoBackend, NoDevice: as `resolve_execution`.
        NotImplementedError: the backend runs in an `Env.server` env.
        ValueError: an input has no reader for its port's type, or the backend params do not validate.
        CallFailed: the backend failed the item.
        WorkerFatal, WorkerLost, ProtocolError: as the worker reports them.
    """
    profile = execution if execution is not None else ExecutionProfile.load()
    run = run if run is not None else RunConfig()
    store = store if store is not None else ResultStore(cache_root() / "results")
    spec = REGISTRY.spec(call.task)
    if call.task_version != spec.version:
        raise ValueError(f"{call.task!r} is declared at version {spec.version!r}, the call names {call.task_version!r}")
    model, entry = _resolve_model(spec, call.model, profile, _PACKAGED.catalogue)
    hardware = _hardware()
    placement = resolve_execution(spec, entry, profile, hardware, REGISTRY, _PACKAGED.catalogue)
    backend = placement.backend
    placement = replace(placement, backend_params=backend.params.model_validate(dict(call.backend_params)))
    if backend.env.kind == "server":
        raise NotImplementedError(f"backend {backend.name!r} runs in a server env, which run_call does not open")
    bound = {port: _bind(port, spec.inputs[port], value) for port, value in call.inputs.items()}
    env = _PACKAGED.open_env(backend, profile)
    context = ExecutionContext(
        **env.context_fields(),
        platform=sysconfig.get_platform(),
        numerics=numerics_class(hardware, placement.device, PRECISION),
        decoder=None,
        host=platform.node(),
        protocol=f"wire@{WIRE_VERSION}",
        model_source=None if model is None else model.ref.source,
    )
    identities = {port: identity for port, (identity, _) in bound.items()}
    key = build_key(Call(task=spec, params=call.params, inputs=identities, model=model), placement, context)
    hit = store.get(key, context, run.reuse)
    if hit is not None:
        return Result(hit.outputs, hit.output_identity, key, hit.context, placement, hit.policy_used)
    started = datetime.now(timezone.utc)
    outputs = _execute(call, placement, env, model, {port: value for port, (_, value) in bound.items()}, profile)
    output_identity = store.put(key, outputs, context)
    store.provenance.add(
        *(kprov.entity(i, core_vocabulary()) for i in identities.values()),
        *_execution(key, context, profile, identities, output_identity, model, started),
    )
    return Result(outputs, output_identity, key, context, placement, None)


def _resolve_model(
    spec: TaskSpec, ref: ModelRef | None, profile: ExecutionProfile, catalogue: Catalogue
) -> tuple[ResolvedModel | None, CatalogueEntry | None]:
    if ref is None:
        try:
            default = catalogue.default(spec.name)
        except UnknownModel:
            return None, None
        ref = ModelRef.parse(f"{default.id}@{default.revision}")
    entry = next((e for e in catalogue.entries if e.id == ref.id), None) if ref.source == "hf" else None
    if entry is not None and ref.ref == entry.revision:
        return ResolvedModel(ref=ref, identity=entry.revision, license=entry.license), entry
    if ref.source == "hf":
        profile.check_source(ref.id, catalogue)
    resolved = resolve(ref)
    return (resolved if entry is None else resolved.model_copy(update={"license": entry.license})), entry


def _bind(port: str, port_type: str, value: object) -> tuple[InputIdentity, object]:
    """Return an input's identity and the value its item carries: a kernel value as it is, a path read first."""
    if isinstance(value, str | PathLike) and port_type in kernel_types():
        reader = _READERS.get(port_type)
        if reader is None:
            raise ValueError(f"input {port!r} is a {port_type}, which is not read from a path; pass the value")
        value = reader(value)
    if isinstance(value, Recording):
        return value.identity, value
    if isinstance(value, Frozen):
        return "content:" + value.canonical_digest(), value
    return "content:" + value_digest("json", {"value": value}), value


@functools.cache
def _hardware() -> HardwareInfo:
    return probe_hardware()


_POOL: dict[tuple[str, ...], InProcessWorker | WorkerClient] = {}
_CACHES: dict[int | None, ModelCache] = {}
_POOL_LOCK = threading.Lock()


def _execute(  # noqa: PLR0913 - the call and everything placing it
    call: TaskCall,
    placement: Placement,
    env: EnvHandle,
    model: ResolvedModel | None,
    inputs: dict[str, object],
    profile: ExecutionProfile,
) -> dict[str, object]:
    """Run the call's one item on the backend's worker with the task params and `backend_params` beside them."""
    backend = placement.backend
    init: dict[str, Any] = {"device": placement.device}
    if model is not None:
        init["model"] = {"source": model.ref.source, "id": model.ref.id, "identity": model.identity}
    params = {**call.params.model_dump(mode="json"), "backend_params": placement.backend_params.model_dump(mode="json")}
    worker = _worker(backend, env, init, model, profile.model_cache_budget_bytes)
    [result] = worker.run([{"id": ITEM_ID, "inputs": inputs}], params)
    if "error" in result:
        raise CallFailed(result["error"])
    return dict(result["outputs"])


def _worker(
    backend: BackendSpec, env: EnvHandle, init: dict[str, Any], model: ResolvedModel | None, budget: int | None
) -> InProcessWorker | WorkerClient:
    """Return the pooled worker for this backend, env and init, starting one if none is alive."""
    pool_key = (backend.env.kind, env.python, backend.worker, jcs_dumps(init).decode())
    with _POOL_LOCK:
        worker = _POOL.get(pool_key)
        if isinstance(worker, InProcessWorker) or (worker is not None and worker.alive):
            return worker
        if backend.env.kind == "host":
            identity = model.identity if model is not None else value_digest("backend", {"name": backend.name})
            key: ModelKey = (backend.name, identity, init["device"], PRECISION)
            worker = InProcessWorker(
                backend.worker, init=init, key=key, cache=_cache(budget), size_of=_size_of(backend.worker, budget)
            )
        else:
            worker = WorkerClient(env.python, backend.worker, init=init)
        _POOL[pool_key] = worker
        return worker


def _cache(budget: int | None) -> ModelCache:
    if budget not in _CACHES:
        _CACHES[budget] = ModelCache(budget if budget is not None else 2**63 - 1)
    return _CACHES[budget]


def _size_of(module: str, budget: int | None) -> Callable[[object], int]:
    """Return how a backend's state is sized: 0 bytes with no budget, else by its module's `state_bytes`."""
    if budget is None:
        return lambda _: 0
    size_of = getattr(importlib.import_module(module), "state_bytes", None)
    if size_of is None:
        raise ValueError(f"backend module {module!r} defines no state_bytes, so a model-cache budget cannot hold it")
    return size_of  # type: ignore[no-any-return]


@atexit.register
def _close_workers() -> None:
    for worker in _POOL.values():
        if isinstance(worker, WorkerClient):
            worker.close()
    _POOL.clear()


def _execution(  # noqa: PLR0913 - one execution's records
    key: LogicalKey,
    context: ExecutionContext,
    profile: ExecutionProfile,
    inputs: Mapping[str, InputIdentity],
    output: OutputIdentity,
    model: ResolvedModel | None,
    started: datetime,
) -> tuple[kprov.Record, kprov.Record]:
    """Return the execution's agent and its activity, which used the inputs and generated the output."""
    vocabulary = core_vocabulary()
    source: str | None = None if model is None else model.ref.source
    if model is not None and model.ref.source == "hf":
        source = f"hf:{model.ref.id}@{model.identity}"
    agent = kprov.agent(
        vocabulary,
        types=["senselab:SoftwareAgent"],
        attributes={} if source is None else {"senselab:modelSource": source},
    )
    activity = kprov.activity(
        vocabulary,
        attributes={
            "senselab:logicalKey": key_record(key),
            "senselab:context": context_record(context),
            "senselab:executionProfile": profile.canonical_digest(),
        },
        used=[kprov.Usage(identity, port=port) for port, identity in inputs.items()],
        generated=[kprov.Generation(output)],
        associations=[kprov.Association(agent.id, node_id=key.task)],
        started=started,
        ended=datetime.now(timezone.utc),
    )
    return agent, activity
