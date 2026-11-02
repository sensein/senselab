"""`run_call`: one task call resolved, placed, keyed, served from the result store or executed, and recorded.

A declaring package hands `run_call` its catalogue, how its backends' envs open and the registry of its tasks
(`REGISTRY`) with `use_packaged`, on import. Until one does, the catalogue is empty and no env opens.
"""

import atexit
import functools
import importlib
import logging
import platform
import sys
import sysconfig
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from importlib.metadata import entry_points
from importlib.resources.abc import Traversable
from os import PathLike
from pathlib import Path
from typing import Annotated, Any, Literal, get_type_hints

from pydantic import AfterValidator, BaseModel, ConfigDict, PlainSerializer, ValidationInfo, model_validator
from senselab_worker import WIRE_VERSION
from senselab_worker.canonical import jcs_dumps
from senselab_worker.model_cache import ModelCache, ModelKey

from senselab_core.kernel import prov as kprov
from senselab_core.kernel._base import Frozen, FrozenMapping, NonEmptyStr, value_digest
from senselab_core.kernel.declare import (
    REGISTRY,
    BackendSpec,
    DuplicateDeclaration,
    Registry,
    TaskParams,
    TaskSpec,
    kernel_types,
)
from senselab_core.kernel.identity import (
    DEFAULT_REUSE_POLICY,
    ExecutionContext,
    InputIdentity,
    LogicalKey,
    ModelRef,
    OutputIdentity,
    ResolvedModel,
    ReusePolicy,
    numerics_key,
    text_identity_fields,
)
from senselab_core.kernel.signals import PlainText, Recording, text_identity
from senselab_core.kernel.time import TimeAnchor, TimeMap, rebind
from senselab_core.kernel.wire import WireFormatRetired, from_wire_form, wire_form
from senselab_core.runtime.audio_io import DECODER, READER, open_audio
from senselab_core.runtime.catalogue import Catalogue, CatalogueEntry, UnknownModel
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.equivalence import SAMPLING, EquivalenceEvidence, EvidenceSide, sampled
from senselab_core.runtime.hardware import HardwareInfo, numerics_class, probe_hardware
from senselab_core.runtime.in_process import InProcessWorker
from senselab_core.runtime.keys import Call, Placement, build_key
from senselab_core.runtime.models import resolve
from senselab_core.runtime.profiles import ExecutionProfile, resolve_execution
from senselab_core.runtime.revision import cache_root, run_id
from senselab_core.runtime.store import ResultStore, StoredResult, context_record, key_record
from senselab_core.runtime.store import output_identity as store_output_identity
from senselab_core.runtime.tolerance import NoToleranceProfile, ToleranceProfile, flatten, read_profile, within
from senselab_core.runtime.vllm_server import (
    Attestation,
    AttestationMismatch,
    StaleAttestation,
    VllmServer,
    attested_fields,
    expected_attestation,
)
from senselab_core.runtime.vocabulary import core_vocabulary
from senselab_core.runtime.worker_client import WorkerClient

logger = logging.getLogger("senselab")

PRECISION = "default"
"""The precision a call records: the backend's own; no backend declares another yet."""
ITEM_ID = "call"
_READERS: Mapping[str, tuple[Callable[[str | PathLike[str]], Recording], str]] = {
    "Audio": (open_audio, DECODER),
    "RegularSeries": (open_audio, DECODER),
}


Inputs = Annotated[Mapping[str, Any], AfterValidator(FrozenMapping), PlainSerializer(dict)]


class TaskCall(BaseModel):
    """A validated call: the task and its version, one input per declared port, validated params, the model.

    Built from a mapping (`{task, inputs, params, model, backend_params}`, the one call shape), `params` is
    validated against the declared task's params, `model` is parsed from its reference string and an absent
    `task_version` is the declared one; tasks are looked up in the validation context's `registry`, else the
    packaged one. `backend_params` are held as given and validated against the backend `run_call` chooses.
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
    def _from_call_shape(cls, data: Any, info: ValidationInfo) -> Any:  # noqa: ANN401 - pydantic's hook signature
        if not isinstance(data, Mapping) or not isinstance(data.get("task"), str):
            return data
        out = dict(data)
        if "task_version" not in out or not isinstance(out.get("params"), TaskParams):
            registry = (info.context or {}).get("registry")
            out.setdefault("task_version", (registry or packaged().registry).spec(out["task"]).version)
            if not isinstance(out.get("params"), TaskParams):
                out["params"] = cls.parse_params(out["task"], out.get("params") or {}, registry)
        out["model"] = cls.parse_model(out.get("model"))
        return out

    @staticmethod
    def parse_params(task: str, params: Mapping[str, Any], registry: Registry | None = None) -> TaskParams:
        """Return `params` validated against the params of `task` in `registry` (the packaged one when None).

        Raises:
            UnknownTask: `task` is not declared.
            pydantic.ValidationError: the params do not validate.
            InconsistentInputs: the params contradict a constraint the task declares.
        """
        spec = (registry or packaged().registry).spec(task)
        validated = spec.params.model_validate(dict(params))
        spec.check_params(validated)
        return validated

    @staticmethod
    def parse_model(model: ModelRef | str | None) -> ModelRef | None:
        """Return `model` as a reference: a string is parsed with `ModelRef.parse`.

        Raises:
            ValueError: the string is not a model reference.
        """
        return ModelRef.parse(model) if isinstance(model, str) else model


@dataclass(frozen=True)
class RunConfig:
    """A run's settings: `reuse`, the widest policy under which a stored result serves a call, and `verify`.

    `verify` is the fraction of cross-context hits recomputed and compared within the backend's tolerance profile.
    """

    reuse: ReusePolicy = DEFAULT_REUSE_POLICY
    verify: float = 0.0

    def __post_init__(self) -> None:
        """Refuse a `verify` outside [0, 1]."""
        if not 0.0 <= self.verify <= 1.0:
            raise ValueError(f"verify is a fraction in [0, 1], got {self.verify}")


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


class AmbiguousClock(ValueError):
    """Two recording inputs of one content carry different anchors: a stored output has no one clock to re-bind to."""


class OutputTypeMismatch(ValueError):
    """A backend's output for a kernel-typed port is not a wire form of the port's declared type."""


EnvOpener = Callable[[BackendSpec, ExecutionProfile], EnvHandle]
"""Open the env a backend declares under a profile, building it if needed."""


@dataclass(frozen=True)
class Packaged:
    """What a declaring package hands `run_call`: its catalogue, env opener, registry and tolerance profiles' folder."""

    catalogue: Catalogue
    open_env: EnvOpener
    registry: Registry = REGISTRY
    tolerance: Path | Traversable | None = None


def _no_env(backend: BackendSpec, profile: ExecutionProfile) -> EnvHandle:  # noqa: ARG001 - an EnvOpener
    raise LookupError(f"no package has said how to open env {backend.env.kind!r} of backend {backend.name!r}")


_PACKAGED = Packaged(Catalogue(entries=()), _no_env)


PACKAGES_GROUP = "senselab.packages"
"""The entry-point group of declaring packages; each entry names a module that calls `use_packaged` on import."""

_STATE = {"registered": False, "discovered": False, "loading": False, "attempted": False, "uses": 0}
_DISCOVERY = threading.RLock()


def use_packaged(packaged: Packaged) -> Packaged:
    """Make `packaged` what `run_call` reads, and return what it read before."""
    global _PACKAGED  # noqa: PLW0603 - the process-wide counterpart of REGISTRY
    previous, _PACKAGED = _PACKAGED, packaged
    _STATE["registered"] = True
    _STATE["uses"] += 1
    return previous


def packaged() -> Packaged:
    """Return what `run_call` reads: the catalogue and env opener last given to `use_packaged`.

    When no package has been registered, the first call loads every entry point of `PACKAGES_GROUP` under a lock,
    and every concurrent first call waits for it. A load that fails raises `PackageLoadError` naming the entry
    point, leaves discovery undone and is retried by the next call.

    Raises:
        PackageLoadError: an entry point fails to load, or two of them declare the same task or backend or both
            register packaged declarations.
    """
    if _STATE["discovered"]:
        return _PACKAGED
    with _DISCOVERY:
        if _STATE["discovered"] or _STATE["loading"]:
            return _PACKAGED
        if not _STATE["registered"] or _STATE["attempted"]:
            _STATE["attempted"] = True
            _STATE["loading"] = True
            try:
                _load_entry_points()
            finally:
                _STATE["loading"] = False
        _STATE["discovered"] = True
    return _PACKAGED


class PackageLoadError(RuntimeError):
    """An entry point of `PACKAGES_GROUP` failed to load, or collides with another."""


def _declared() -> dict[tuple[str, ...], None]:
    registry = REGISTRY
    keys: dict[tuple[str, ...], None] = {(t.name,): None for t in registry.tasks()}
    keys.update({(t.name, b.name): None for t in registry.tasks() for b in registry.backends(t.name)})
    return keys


def _load_entry_points() -> None:
    owners: dict[tuple[str, ...], str] = {k: "a declaration made before discovery" for k in _declared()}
    registering: str | None = None
    for entry in entry_points(group=PACKAGES_GROUP):
        uses = _STATE["uses"]
        name = f"{entry.name} = {entry.value}"
        before = (REGISTRY.snapshot(), _PACKAGED, bool(_STATE["registered"]), set(sys.modules))
        try:
            entry.load()
        except Exception as error:
            _roll_back(entry.value, before)
            raise _load_error(name, error, owners) from error
        owners.update({k: name for k in _declared() if k not in owners})
        if _STATE["uses"] > uses:
            if registering is not None:
                raise PackageLoadError(f"entry points {registering!r} and {name!r} both register packaged declarations")
            registering = name


def _roll_back(value: str, before: tuple[Any, Packaged, bool, set[str]]) -> None:
    """Undo a failed load: its declarations, its `use_packaged` and the modules of its top-level package."""
    global _PACKAGED  # noqa: PLW0603 - the process-wide counterpart of REGISTRY
    snapshot, previous, registered, modules = before
    REGISTRY.restore(snapshot)
    _PACKAGED, _STATE["registered"] = previous, int(registered)
    top = value.split(":", maxsplit=1)[0].split(".", maxsplit=1)[0]
    for module in [m for m in sys.modules if m not in modules and (m == top or m.startswith(top + "."))]:
        del sys.modules[module]


def _load_error(name: str, error: Exception, owners: dict[tuple[str, ...], str]) -> PackageLoadError:
    if isinstance(error, DuplicateDeclaration):
        existing = error.existing
        key = (existing.name,) if isinstance(existing, TaskSpec) else (existing.task, existing.name)
        return PackageLoadError(
            f"entry point {name!r} redeclares {'/'.join(key)!r}, declared by {owners.get(key, 'another package')}"
        )
    return PackageLoadError(f"entry point {name!r} failed to load: {type(error).__name__}: {error}")


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
    backends share one `ModelCache` per budget). A server backend's context carries the attestation its lock,
    model and settings declare; its `VllmServer` starts on the first call the store does not serve, must attest
    exactly that, serves every later call of that env and model, and stops at exit.
    An execution writes one activity, associated with an agent carrying `senselab:modelSource`, recording the key,
    the context and the profile's digest; a store hit writes none (the store records its reuse). An execution whose
    `put` finds different outputs another writer committed first is served those, as a reuse, and writes no
    execution activity. An output on a port declared as a kernel type is stored as its wire form and returned as the
    kernel value, read by `from_wire_form`. Outputs are stored on the
    clocks their recording inputs' identities name and served re-bound (`rebind`) to the caller's anchors, so the key
    holds no anchor and one entry serves one content under every anchor. A cross-context hit among the fraction
    `run.verify` (`sampled`) is not served: the backend's tolerance profile is read, the call executes, and the
    outputs are compared with the stored result before anything is stored; the comparison is recorded as
    `EquivalenceEvidence` (a disagreement splits the store's class and is logged), then the outputs are stored and
    the execution activity names the evidence (`senselab:verifies`). Every activity a call writes records
    `senselab:verifyFraction` and `senselab:verifySampling`. `None` means the packaged
    default: the `laptop` profile, `RunConfig()` and the store under `cache_root()/results`.

    Raises:
        UnknownTask, InconsistentInputs: the task is not declared; the inputs contradict a constraint it declares.
        AmbiguousClock: two recording inputs of one content carry different anchors.
        WireFormatRetired: a stored output is in a retired wire format.
        UnlistedSource, NoBackend, NoDevice: as `resolve_execution`.
        ValueError: an input has no reader for its port's type or, for an isolated or server backend, no wire form;
            the backend params do not validate; or a server backend got no model.
        VllmStartError, LeaseTimeout, LeaseUnavailable: a server backend's server did not start.
        AttestationMismatch: a started server attests other than its declared attestation.
        CallFailed: the backend failed the item.
        OutputTypeMismatch: an output of a kernel-typed port is not a wire form of its declared type.
        WorkerFatal, WorkerLost, ProtocolError: as the worker reports them.
        NoToleranceProfile: a sampled hit's backend has no tolerance profile.
    """
    profile = execution if execution is not None else ExecutionProfile.load()
    run = run if run is not None else RunConfig()
    store = store if store is not None else ResultStore(cache_root() / "results")
    spec = packaged().registry.spec(call.task)
    if call.task_version != spec.version:
        raise ValueError(f"{call.task!r} is declared at version {spec.version!r}, the call names {call.task_version!r}")
    model, hardware, placement = _place(spec, call.model, profile)
    backend = placement.backend
    placement = replace(placement, backend_params=backend.params.model_validate(dict(call.backend_params)))
    bound = _bound(spec, call)
    env = packaged().open_env(backend, profile)
    expected = _expected(backend, env, model) if backend.env.kind == "server" else None
    context = ExecutionContext(
        **(env.context_fields() if expected is None else attested_fields(env, expected)),
        platform=sysconfig.get_platform(),
        numerics=numerics_class(hardware, placement.device, PRECISION),
        decoder=next((d for _, _, d in bound.values() if d is not None), None),
        host=platform.node(),
        protocol=f"wire@{WIRE_VERSION}",
        model_source=None if model is None else model.ref.source,
    )
    identities = {port: identity for port, (identity, _, _) in bound.items()}
    key = build_key(Call(task=spec, params=call.params, inputs=identities, model=model), placement, context)
    anchors = _anchors(value for _, value, _ in bound.values())
    looked = datetime.now(timezone.utc)
    hit = store.get(key, context, run.reuse, record=False)
    check = (
        hit is not None
        and hit.policy_used is not ReusePolicy.EXACT
        and sampled(run.verify, key.digest(), hit.context, context)
    )
    if hit is not None and not check:
        store.record_reuse(key, context, hit, looked, _verify_attributes(run))
        return _served(spec, hit, key, placement, anchors)
    tolerance = _tolerance(key) if check else None
    started = datetime.now(timezone.utc)
    server = _server(backend, env, model, profile, expected) if expected is not None else None
    computed = _execute(
        call, placement, env, model, {port: value for port, (_, value, _) in bound.items()}, profile, server
    )
    decoded, outputs = _as_stored(spec, computed, anchors)
    attributes = _verify_attributes(run)
    if hit is not None and tolerance is not None:
        attributes["senselab:verifies"] = _verify(store, key, hit, context, outputs, tolerance)
    output_identity = store.put(key, outputs, context)
    if output_identity != store_output_identity(outputs):
        looked = datetime.now(timezone.utc)
        stored = store.get(key, context, ReusePolicy.EXACT, record=False)
        if stored is None:
            raise RuntimeError(f"the entry another writer committed for {key.task!r} could not be read back")
        store.record_reuse(key, context, stored, looked, attributes)
        return _served(spec, stored, key, placement, anchors)
    store.provenance.add(
        *(kprov.entity(i, core_vocabulary()) for i in identities.values()),
        *_execution(key, context, profile, identities, output_identity, model, started, attributes),
    )
    return Result(decoded, output_identity, key, context, placement, None)


def _bound(spec: TaskSpec, call: TaskCall) -> dict[str, tuple[InputIdentity, object, str | None]]:
    """Return the call's inputs bound by port, once they satisfy the constraints the task declares."""
    bound = {port: bind_input(port, spec.inputs[port], value) for port, value in call.inputs.items()}
    spec.check_inputs({port: value for port, (_, value, _) in bound.items()})
    return bound


def _as_stored(
    spec: TaskSpec, computed: dict[str, object], anchors: list[tuple[TimeAnchor, TimeAnchor]]
) -> tuple[dict[str, object], dict[str, object]]:
    """Return computed outputs decoded, and as the store holds them: on their identity clocks, as wire forms."""
    decoded = _decoded(spec, computed)
    on_identity_clocks = _rebound(decoded, [(reader, stored) for stored, reader in anchors])
    outputs = {
        p: wire_form(v, READER) if isinstance(v, Frozen) and v is not decoded[p] else computed[p]
        for p, v in on_identity_clocks.items()
    }
    return decoded, outputs


def _served(
    spec: TaskSpec,
    hit: StoredResult,
    key: LogicalKey,
    placement: Placement,
    anchors: list[tuple[TimeAnchor, TimeAnchor]],
) -> Result:
    """Return a stored result as the call's result, its outputs decoded and re-bound to the caller's anchors."""
    outputs = _rebound(_decoded(spec, hit.outputs), anchors)
    return Result(outputs, hit.output_identity, key, hit.context, placement, hit.policy_used)


def _verify_attributes(run: RunConfig) -> dict[str, object]:
    """Return the verify settings a call's activity records: the fraction and how hits are sampled."""
    return {"senselab:verifyFraction": float(run.verify), "senselab:verifySampling": SAMPLING}


def _tolerance(key: LogicalKey) -> ToleranceProfile:
    """Return the tolerance profile of the key's backend.

    Raises:
        NoToleranceProfile: the backend has none, or no package names a folder of profiles.
    """
    folder = packaged().tolerance
    if folder is None:
        raise NoToleranceProfile(f"no package names a folder of tolerance profiles for {key.backend!r}")
    return read_profile(folder, key.task, key.backend)


def _verify(  # noqa: PLR0913 - the hit, its recomputation and the profile judging them
    store: ResultStore,
    key: LogicalKey,
    hit: StoredResult,
    context: ExecutionContext,
    outputs: dict[str, object],
    profile: ToleranceProfile,
) -> str:
    """Compare a sampled hit with its recomputation within `profile`, record the evidence and return its digest.

    A disagreement splits the store's class of the two numerics classes (`ResultStore.add_evidence`) and is logged.
    """
    mismatched = within(flatten(hit.outputs), flatten(outputs), profile)
    stored, current = EvidenceSide.of(hit.context), EvidenceSide.of(context)
    evidence = EquivalenceEvidence(
        task=key.task,
        backend=key.backend,
        key=key.digest(),
        stored=stored,
        current=current,
        mismatched=tuple(mismatched),
        tolerance=profile.canonical_digest(),
        run=run_id(),
    )
    store.add_evidence(evidence)
    if mismatched:
        logger.warning(
            "verify: %s on %s disagrees with its result from %s beyond its tolerance in %s",
            key.backend,
            numerics_key(current.numerics),
            numerics_key(stored.numerics),
            ", ".join(mismatched),
        )
    return evidence.canonical_digest()


def _anchors(values: Iterable[object]) -> list[tuple[TimeAnchor, TimeAnchor]]:
    """Return, for each recording input not on the clock its identity names, that clock's anchor and its own.

    A store holds outputs on the identity clocks of their inputs, so one entry serves every anchor of one content.

    Raises:
        AmbiguousClock: two recording inputs of one content carry different anchors.
    """
    readers: dict[str, TimeAnchor] = {}
    for v in values:
        if isinstance(v, Recording) and readers.setdefault(v.identity, v.anchor) != v.anchor:
            raise AmbiguousClock(f"recording inputs of {v.identity} carry anchors on two clocks; pass one anchor")
    return [
        (TimeAnchor(clock_id=identity, absolute_start_ns=None), anchor)
        for identity, anchor in readers.items()
        if anchor.clock_id != identity
    ]


def _rebound(outputs: dict[str, object], anchors: list[tuple[TimeAnchor, TimeAnchor]]) -> dict[str, object]:
    """Return `outputs` re-bound from each pair's first anchor to its second, the two naming one content's start."""
    for stored, reader in anchors:
        timemap = TimeMap(source=stored.clock_id, target=reader.clock_id, offset_ns=0)
        outputs = {port: rebind(value, stored, reader, timemap) for port, value in outputs.items()}
    return outputs


def _place(
    spec: TaskSpec, ref: ModelRef | None, profile: ExecutionProfile
) -> tuple[ResolvedModel | None, HardwareInfo, Placement]:
    model, entry = _resolve_model(spec, ref, profile, packaged().catalogue)
    hardware = _hardware()
    return model, hardware, resolve_execution(spec, entry, profile, hardware, packaged().registry, packaged().catalogue)


def placement_of(task: str, model: ModelRef | None, execution: ExecutionProfile) -> Placement:
    """Return where `run_call` would run `task` on `model` under `execution` on this host, starting nothing.

    Raises:
        UnknownTask: the task is not declared.
        UnlistedSource, NoBackend, NoDevice: as `resolve_execution`.
    """
    return _place(packaged().registry.spec(task), model, execution)[2]


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


def input_binding(port_type: str) -> Literal["path", "text", "value"]:
    """Return how `bind_input` takes a `str` for `port_type`: a `path` it reads, the `text` itself or a `value`.

    A `value` port takes the object, or for a type that is not a kernel type its JSON value; a path is refused.
    """
    if port_type == "Text":
        return "text"
    return "path" if port_type in _READERS else "value"


def _with_identities(model: type[BaseModel], value: Mapping[str, Any]) -> dict[str, Any]:
    """Return `value` with each `TextIdentity` field of `model` it gives replaced by the identity (`text_identity`)."""
    return {k: text_identity(k, v) if k in text_identity_fields(model) else v for k, v in value.items()}


def input_model(port_type: str) -> type[BaseModel] | None:
    """Return the model `bind_input` holds a value of `port_type` as: the type itself, or what a protocol port binds to.

    `Text` binds to `PlainText` and a type with a reader to what the reader returns; None for a type that is not a
    kernel type.
    """
    kernel_type = kernel_types().get(port_type)
    if kernel_type is None:
        return None
    if issubclass(kernel_type, BaseModel):
        return kernel_type
    if port_type == "Text":
        return PlainText
    returned: type[BaseModel] = get_type_hints(_READERS[port_type][0])["return"]
    return returned


def bind_input(port: str, port_type: str, value: object) -> tuple[InputIdentity, object, str | None]:
    """Return an input's identity, the value its item carries and the decoder that read it from a path, if one did.

    A `str` for a `Text` port is held as `PlainText`.
    """
    decoder = None
    model = input_model(port_type)
    if isinstance(value, Mapping) and model is not None:
        value = model.model_validate(_with_identities(model, value))
    if port_type == "Text" and isinstance(value, str):
        value = PlainText(content=value)
    if isinstance(value, str | PathLike) and port_type in kernel_types():
        if port_type not in _READERS:
            raise ValueError(f"input {port!r} is a {port_type}, which is not read from a path; pass the value")
        reader, decoder = _READERS[port_type]
        value = reader(value)
    if isinstance(value, Recording | PlainText):
        return value.identity, value, decoder
    if isinstance(value, Frozen):
        return "content:" + value.canonical_digest(), value, decoder
    return "content:" + value_digest("json", {"value": value}), value, decoder


@functools.cache
def _hardware() -> HardwareInfo:
    return probe_hardware()


_POOL: dict[tuple[str, ...], InProcessWorker | WorkerClient] = {}
_CACHES: dict[int | None, ModelCache] = {}
_POOL_LOCK = threading.Lock()
_SERVERS: dict[tuple[str, str, str], VllmServer] = {}
_SERVER_LOCK = threading.Lock()


def _expected(backend: BackendSpec, env: EnvHandle, model: ResolvedModel | None) -> Attestation:
    """Return the attestation a server of this backend must give, computed without starting it.

    Raises:
        ValueError: the call has no model.
        VllmStartError: the env's lock pins no vllm.
    """
    if model is None or backend.server is None:
        raise ValueError(f"backend {backend.name!r} serves a model through a server, and the call resolved none")
    return expected_attestation(env, model, backend.server.settings)


def _server(
    backend: BackendSpec, env: EnvHandle, model: ResolvedModel | None, profile: ExecutionProfile, expected: Attestation
) -> VllmServer:
    """Return the running server of this backend, env and model, starting it (or replacing a dead one) if needed.

    Raises:
        AttestationMismatch: the server attests something other than `expected`.
    """
    assert model is not None and backend.server is not None
    key = (env.python, backend.name, model.identity)
    root = profile.env_root("server")
    with _SERVER_LOCK:
        server = _SERVERS.get(key)
        if server is not None:
            try:
                server.context_fields()
                return server
            except StaleAttestation:
                server.stop()
        server = VllmServer(
            env,
            model,
            backend.server.settings,
            start_lock_dir=root / "locks",
            startup_timeout_s=backend.server.startup_timeout_s,
            start_margin_s=profile.envs.server_start_margin_s,
            log_dir=root / "logs",
        )
        server.__enter__()
        if server.attestation != expected:
            attested = server.attestation
            server.stop()
            raise AttestationMismatch(f"the server of {server.label} attests {attested}, the lock declares {expected}")
        _SERVERS[key] = server
        return server


def _server_worker(backend: BackendSpec, init: dict[str, Any], server: VllmServer) -> InProcessWorker:
    """Return the in-process worker of a server backend, given `server` in its init."""
    return InProcessWorker(
        backend.worker,
        init={**init, "server": server},
        key=(backend.name, init["model"]["identity"], init["device"], PRECISION),
        cache=ModelCache(2**63 - 1),
        size_of=lambda _: 0,
    )


def start_server_worker(
    backend: BackendSpec, env: EnvHandle, model: ResolvedModel, profile: ExecutionProfile, init: dict[str, Any]
) -> InProcessWorker:
    """Return the worker `run_call` would run a server backend on: in this process, over its attested running server.

    Raises:
        ValueError, VllmStartError, LeaseTimeout, LeaseUnavailable, AttestationMismatch: as `run_call`.
    """
    return _server_worker(backend, init, _server(backend, env, model, profile, _expected(backend, env, model)))


def _execute(  # noqa: PLR0913 - the call and everything placing it
    call: TaskCall,
    placement: Placement,
    env: EnvHandle,
    model: ResolvedModel | None,
    inputs: dict[str, object],
    profile: ExecutionProfile,
    server: VllmServer | None,
) -> dict[str, object]:
    """Run the call's one item on the backend's worker with the task params and `backend_params` beside them.

    A host worker is handed kernel values as they are; an isolated or server one, their wire forms (`wire_form`).
    A server backend's worker runs in this process and is given `server` as `init["server"]`. The outputs
    are returned as the store holds them: a kernel value a host backend returns, as its wire form.
    """
    backend = placement.backend
    init: dict[str, Any] = {"device": placement.device}
    if model is not None:
        init["model"] = {"source": model.ref.source, "id": model.ref.id, "identity": model.identity}
    params = {**call.params.model_dump(mode="json"), "backend_params": placement.backend_params.model_dump(mode="json")}
    if server is not None:
        worker: InProcessWorker | WorkerClient = _server_worker(backend, init, server)
    else:
        worker = _worker(backend, env, init, model, profile.model_cache_budget_bytes)
    if backend.env.kind != "host":
        inputs = {port: wire_form(v, READER) if isinstance(v, Frozen) else v for port, v in inputs.items()}
    [result] = worker.run([{"id": ITEM_ID, "inputs": inputs}], params)
    if "error" in result:
        raise CallFailed(result["error"])
    return {port: wire_form(v, READER) if isinstance(v, Frozen) else v for port, v in result["outputs"].items()}


def _decoded(spec: TaskSpec, outputs: Mapping[str, object]) -> dict[str, object]:
    """Return `outputs` with each port declared as a kernel type read from its wire form (`from_wire_form`).

    Raises:
        OutputTypeMismatch: an output of a kernel-typed port is not a wire form of its declared type.
    """
    types = kernel_types()
    decoded = dict(outputs)
    for port, declared in spec.outputs.items():
        if declared not in types or port not in outputs:
            continue
        form = outputs[port]
        try:
            value = from_wire_form(form) if isinstance(form, Mapping) else form
        except WireFormatRetired:
            raise
        except ValueError as exc:
            raise OutputTypeMismatch(f"output {port!r} of {spec.name!r} is not a wire form: {exc}") from exc
        if not isinstance(value, types[declared]):
            found = form.get("format") if isinstance(form, Mapping) else type(form).__name__
            raise OutputTypeMismatch(f"output {port!r} of {spec.name!r} is declared {declared}, got {found}")
        decoded[port] = value
    return decoded


def _worker(
    backend: BackendSpec, env: EnvHandle, init: dict[str, Any], model: ResolvedModel | None, budget: int | None
) -> InProcessWorker | WorkerClient:
    """Return the pooled worker for this backend, env, init and (in process) budget; a dead one is closed, replaced."""
    host = backend.env.kind == "host"
    pool_key = (backend.env.kind, env.python, backend.worker, jcs_dumps(init).decode(), str(budget) if host else "")
    with _POOL_LOCK:
        worker = _POOL.get(pool_key)
        if isinstance(worker, InProcessWorker) or (worker is not None and worker.alive):
            return worker
        if worker is not None:
            worker.close()
        if host:
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
def close_workers() -> None:
    """Stop every worker and server `run_call` started in this process; a later call starts its own again."""
    for worker in _POOL.values():
        if isinstance(worker, WorkerClient):
            worker.close()
    _POOL.clear()
    stop_servers()


def stop_servers() -> None:
    """Stop every server `run_call` or `start_server_worker` started in this process."""
    for server in _SERVERS.values():
        server.stop()
    _SERVERS.clear()


def _execution(  # noqa: PLR0913 - one execution's records
    key: LogicalKey,
    context: ExecutionContext,
    profile: ExecutionProfile,
    inputs: Mapping[str, InputIdentity],
    output: OutputIdentity,
    model: ResolvedModel | None,
    started: datetime,
    attributes: Mapping[str, object],
) -> tuple[kprov.Record, kprov.Record]:
    """Return the execution's agent and its activity, which used the inputs and generated the output.

    `attributes` are recorded on the activity beside the key, the context and the profile's digest.
    """
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
            **attributes,
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
