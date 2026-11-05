"""`run_call`: one task call resolved, placed, keyed, served from the result store or executed, and recorded.

A declaring package hands `run_call` its catalogue, how its backends' envs open and its tasks and backends through
the `register` function its `PACKAGES_GROUP` entry point names. Until one does, the catalogue is empty and no env
opens.
"""

import atexit
import contextlib
import functools
import importlib
import logging
import platform
import sysconfig
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from importlib.metadata import entry_points
from os import PathLike
from typing import Annotated, Any, Literal, get_type_hints

from pydantic import AfterValidator, BaseModel, ConfigDict, PlainSerializer, ValidationInfo, model_validator
from senselab_worker import WIRE_VERSION
from senselab_worker.canonical import jcs_dumps
from senselab_worker.model_cache import ModelCache, ModelKey

from senselab_core.kernel import prov as kprov
from senselab_core.kernel._base import Frozen, FrozenMapping, NonEmptyStr, thawed, value_digest
from senselab_core.kernel.declare import (
    REGISTRY,
    BackendParams,
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
    Params,
    ResolvedModel,
    ReusePolicy,
    text_identity_fields,
)
from senselab_core.kernel.license import License
from senselab_core.kernel.signals import PlainText, Recording, text_identity
from senselab_core.kernel.time import TimeAnchor, TimeMap, rebind
from senselab_core.kernel.wire import WireFormatRetired, from_wire_form, wire_form
from senselab_core.runtime.audio_io import DECODER, READER, open_audio
from senselab_core.runtime.catalogue import Catalogue, CatalogueEntry, UnknownModel
from senselab_core.runtime.compare import Thresholds, compare
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.equivalence import SAMPLING, EquivalenceEvidence, EvidenceSide, sampled
from senselab_core.runtime.hardware import HardwareInfo, numerics_class, probe_hardware
from senselab_core.runtime.in_process import InProcessWorker
from senselab_core.runtime.keys import Call, Placement, build_key, profile_slices, request_components
from senselab_core.runtime.launch import launch
from senselab_core.runtime.licenses import LicensePolicy, LockRecords, backend_license
from senselab_core.runtime.models import resolve
from senselab_core.runtime.network import network_variables
from senselab_core.runtime.profiles import ExecutionProfile, resolve_execution
from senselab_core.runtime.revision import PROCESS_RESOLVER, Resolver, cache_root, run_id
from senselab_core.runtime.store import ResultStore, StoredResult, agent_record, context_record, key_record
from senselab_core.runtime.store import output_identity as store_output_identity
from senselab_core.runtime.vllm_server import (
    Attestation,
    AttestationMismatch,
    StaleAttestation,
    VllmServer,
    attested_fields,
    expected_attestation,
    server_launch,
    server_variables,
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
    packaged one. `backend_params` maps a backend name to that backend's params, held as given; only the entry of the
    backend `run_call` chooses is read, and validated against it (no entry is that backend's defaults).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    task: NonEmptyStr
    task_version: NonEmptyStr
    inputs: Inputs
    params: TaskParams
    model: ModelRef | None = None
    backend_params: Annotated[Mapping[str, Params], AfterValidator(FrozenMapping), PlainSerializer(thawed)] = (
        FrozenMapping({})
    )

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
    """A run's settings: `reuse`, the widest policy under which a stored result serves a call, `verify`, `licenses`.

    `verify` is the fraction of cross-context hits recomputed and compared under `thresholds` (`Thresholds.load`,
    the packaged run config's unless a user's is merged over it). `licenses` is the user's license policy; with None
    no call is refused for its license. `settings` holds the values tasks read, from which a task's `run.<path>`
    profile slices are cut (`keys.profile_slices`).
    """

    reuse: ReusePolicy = DEFAULT_REUSE_POLICY
    verify: float = 0.0
    licenses: LicensePolicy | None = None
    thresholds: Thresholds = field(default_factory=Thresholds.load)
    settings: Mapping[str, Any] = field(default_factory=lambda: FrozenMapping({}))

    def __post_init__(self) -> None:
        """Refuse a `verify` outside [0, 1], and hold `settings` read-only."""
        if not 0.0 <= self.verify <= 1.0:
            raise ValueError(f"verify is a fraction in [0, 1], got {self.verify}")
        object.__setattr__(self, "settings", FrozenMapping(dict(self.settings)))


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
EnvProbe = Callable[[BackendSpec, ExecutionProfile], bool]
"""Whether the env a backend declares is built under a profile, building nothing."""


def _unbuilt(backend: BackendSpec, profile: ExecutionProfile) -> bool:  # noqa: ARG001 - an EnvProbe
    return False


@dataclass(frozen=True)
class Packaged:
    """What a declaring package hands `run_call`: catalogue, env opener, registry, lock records and env probe.

    `lock_records` gives each backend's env-lock package license records, from which `env_license` derives; without
    them no stored result is served unless the run's license policy allows unlicensed hits. `env_built` says whether
    a backend's env is built (by default none is).
    """

    catalogue: Catalogue
    open_env: EnvOpener
    registry: Registry = REGISTRY
    lock_records: LockRecords | None = None
    env_built: EnvProbe = _unbuilt


def env_license(backend: BackendSpec) -> License | None:
    """Return the code license of `backend`'s env, `backend_license` over the packaged lock records.

    None when the declaring package hands no lock records.

    Raises:
        ValueError, MissingLicense: as the lock records provider and `backend_license`.
    """
    records = packaged().lock_records
    return None if records is None else _derived_license(records, backend)


@functools.cache
def _derived_license(records: LockRecords, backend: BackendSpec) -> License:
    return backend_license(backend, records(backend))


def _no_env(backend: BackendSpec, profile: ExecutionProfile) -> EnvHandle:  # noqa: ARG001 - an EnvOpener
    raise LookupError(f"no package has said how to open env {backend.env.kind!r} of backend {backend.name!r}")


_PACKAGED = Packaged(Catalogue(entries=()), _no_env)


PACKAGES_GROUP = "senselab.packages"
"""The entry-point group of declaring packages; each entry names a `register(registry) -> Packaged | None` function.

`register` declares the package's tasks and backends in `registry` and returns its `Packaged`, or None when it
declares only. Importing the package declares nothing.
"""

_STATE = {"registered": False, "discovered": False, "loading": False, "attempted": False}
_DISCOVERY = threading.RLock()


def use_packaged(packaged: Packaged) -> Packaged:
    """Make `packaged` what `run_call` reads, and return what it read before."""
    global _PACKAGED  # noqa: PLW0603 - the process-wide counterpart of REGISTRY
    previous, _PACKAGED = _PACKAGED, packaged
    _STATE["registered"] = True
    return previous


def packaged() -> Packaged:
    """Return what `run_call` reads: the catalogue and env opener last given to `use_packaged`.

    When no package has been registered, the first call calls every entry point's `register` under a lock, each
    over a staging copy of `REGISTRY`, and every concurrent first call waits for it. The declarations and the
    `Packaged` are merged only when every entry point succeeds; a failure raises `PackageLoadError` naming the
    entry point, changes nothing and is retried by the next call.

    Raises:
        PackageLoadError: an entry point fails to load or register, or two of them declare the same task or backend
            or both return a `Packaged`.
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


def _declared(registry: Registry) -> dict[tuple[str, ...], None]:
    keys: dict[tuple[str, ...], None] = {(t.name,): None for t in registry.tasks()}
    keys.update({(t.name, b.name): None for t in registry.tasks() for b in registry.backends(t.name)})
    return keys


def _load_entry_points() -> None:
    staging = Registry()
    staging.restore(REGISTRY.snapshot())
    owners: dict[tuple[str, ...], str] = {k: "a declaration made before discovery" for k in _declared(staging)}
    found: tuple[str, Packaged] | None = None
    for entry in entry_points(group=PACKAGES_GROUP):
        name = f"{entry.name} = {entry.value}"
        try:
            result = entry.load()(staging)
        except Exception as error:
            raise _load_error(name, error, owners) from error
        owners.update({k: name for k in _declared(staging) if k not in owners})
        if result is not None:
            if found is not None:
                raise PackageLoadError(f"entry points {found[0]!r} and {name!r} both return a Packaged")
            found = (name, result)
    REGISTRY.restore(staging.snapshot())
    if found is not None:
        use_packaged(found[1])


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

    The model is the catalogue's pin for a catalogue model at that revision (the task's catalogue default when the call
    names none), else `resolve()`d. `resolve_execution` places the call under `execution`; the backend's env opens
    through the packaged opener and its worker is reused across calls (one model per worker process; host backends share
    one `ModelCache` per budget). A server backend's context carries the attestation its lock, model and settings
    declare; its `VllmServer` starts on the first call the store does not serve, must attest exactly that, serves every
    later call of that env and model, and stops at exit. An execution writes one activity, associated with its agent
    (`agent_record`, env lock included), recording the key, the context and the profile's digest; a store hit writes
    none (the store records its reuse). An execution whose `put` finds different outputs another writer committed first
    is served those, as a reuse, and writes no execution activity. An output on a port declared as a kernel type is
    stored as its wire form and returned as the kernel value, read by `from_wire_form`. Outputs are stored on the clocks
    their recording inputs' identities name and served re-bound (`rebind`) to the caller's anchors, so the key holds no
    anchor and one entry serves one content under every anchor. A cross-context hit among the fraction `run.verify`
    (`sampled`) is not served: the call executes, and the outputs are compared with the stored result under
    `run.thresholds` (`compare`) before anything is stored; the comparison is recorded as `EquivalenceEvidence` (a
    disagreement splits the store's class and is logged), then the outputs are stored and the execution activity names
    the evidence (`senselab:verifies`). The context and its agent record the env's code license (`env_license`). Only
    `run.licenses` refuses a license: a refused env or model license raises before anything opens, and a result from a
    refused env is not served. A result whose context records no env license is recomputed unless `run.licenses`
    allows unlicensed hits; a reuse so allowed records `senselab:unlicensedReuse`. Every activity a call writes
    records `senselab:verifyFraction`, `senselab:verifySampling` and `senselab:verifyThresholds`. `None` means the
    packaged default: the `laptop` profile, `RunConfig()` and the store under `cache_root()/results`.

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
        LicenseRefused: `run.licenses` refuses the license of the backend's env or of the model.
    """
    profile = execution if execution is not None else ExecutionProfile.load()
    run = run if run is not None else RunConfig()
    store = store if store is not None else ResultStore(cache_root() / "results")
    spec = packaged().registry.spec(call.task)
    if call.task_version != spec.version:
        raise ValueError(f"{call.task!r} is declared at version {spec.version!r}, the call names {call.task_version!r}")
    model, hardware, placement = _place(spec, call.model, profile)
    backend = placement.backend
    code_license = _code_license(run, backend, model)
    placement = replace(placement, backend_params=_placed_params(call, backend))
    bound = _bound(spec, call)
    env = packaged().open_env(backend, profile)
    expected = _expected(backend, env, model) if backend.env.kind == "server" else None
    context = _context(placement, env, expected, profile, hardware, bound, model, code_license)
    identities = {port: identity for port, (identity, _, _) in bound.items()}
    keyed = Call(spec, call.params, identities, model, profile_slices(spec, run.settings))
    key = build_key(keyed, placement, context)
    anchors = _anchors(value for _, value, _ in bound.values())
    looked = datetime.now(timezone.utc)
    hit = _hit(store, key, context, run)
    check = (
        hit is not None
        and hit.policy_used is not ReusePolicy.EXACT
        and sampled(run.verify, key.digest(), hit.context, context)
    )
    if hit is not None and not check:
        store.record_reuse(key, context, hit, looked, _reuse_attributes(run, hit))
        return _served(spec, hit, key, placement, anchors)
    started = datetime.now(timezone.utc)
    server = _server(backend, env, model, profile, expected) if expected is not None else None
    computed = _execute(
        call, placement, env, model, {port: value for port, (_, value, _) in bound.items()}, profile, server
    )
    decoded, outputs = _as_stored(spec, computed, anchors)
    attributes = _verify_attributes(run)
    if hit is not None and check:
        attributes["senselab:verifies"] = _verify(store, key, hit, context, outputs, run.thresholds)
    output_identity = store.put(key, outputs, context)
    if output_identity != store_output_identity(outputs):
        lost = _lost_race(spec, store, key, context, run, attributes, placement, anchors)
        if lost is not None:
            return lost
        output_identity = store_output_identity(outputs)
    store.provenance.add(
        *(kprov.entity(i, core_vocabulary()) for i in identities.values()),
        *_execution(key, context, profile, identities, output_identity, started, attributes),
    )
    return Result(decoded, output_identity, key, context, placement, None)


def _context(  # noqa: PLR0913 - every input a context records
    placement: Placement,
    env: EnvHandle,
    expected: Attestation | None,
    profile: ExecutionProfile,
    hardware: HardwareInfo,
    bound: Mapping[str, tuple[InputIdentity, object, str | None]],
    model: ResolvedModel | None,
    code_license: License | None,
) -> ExecutionContext:
    """Return the context a call runs in: its env, attestation, network, host, decoder, model source and license."""
    return ExecutionContext(
        **_context_fields(placement.backend, env, expected, network_variables(profile.network)),
        **_host_fields(hardware, placement, bound),
        model_source=None if model is None else model.ref.source,
        env_license=code_license,
    )


def _lost_race(  # noqa: PLR0913 - the call's store, key, context and run, and what serving needs
    spec: TaskSpec,
    store: ResultStore,
    key: LogicalKey,
    context: ExecutionContext,
    run: RunConfig,
    attributes: Mapping[str, object],
    placement: Placement,
    anchors: list[tuple[TimeAnchor, TimeAnchor]],
) -> Result | None:
    """Serve the entry another writer committed first, recording the reuse; None when its license refuses it."""
    committed = _committed(store, key, context, run)
    if committed is None:
        return None
    store.record_reuse(key, context, committed, datetime.now(timezone.utc), {**attributes, **_unlicensed(committed)})
    return _served(spec, committed, key, placement, anchors)


def _placed_params(call: TaskCall, backend: BackendSpec) -> BackendParams:
    """Return the call's entry for `backend` validated against its params; no entry is its defaults."""
    return backend.params.model_validate(dict(call.backend_params.get(backend.name, {})))


def _host_fields(
    hardware: HardwareInfo, placement: Placement, bound: Mapping[str, tuple[InputIdentity, object, str | None]]
) -> dict[str, Any]:
    """Return the context fields this host and the bound inputs decide: platform, numerics, decoder, host, protocol."""
    return {
        "platform": sysconfig.get_platform(),
        "numerics": numerics_class(hardware, placement.device, PRECISION),
        "decoder": next((d for _, _, d in bound.values() if d is not None), None),
        "host": platform.node(),
        "protocol": f"wire@{WIRE_VERSION}",
    }


def _code_license(run: RunConfig, backend: BackendSpec, model: ResolvedModel | None) -> License | None:
    """Return the code license of `backend`'s env, once `run.licenses`, when given, accepts it and the model's.

    Raises:
        LicenseRefused: the policy refuses the env's or the model's license.
    """
    code = env_license(backend)
    if run.licenses is not None:
        run.licenses.check(f"the env of backend {backend.name!r}", code)
        if model is not None:
            run.licenses.check(f"model {model.ref}", model.license)
    return code


def _hit(store: ResultStore, key: LogicalKey, context: ExecutionContext, run: RunConfig) -> StoredResult | None:
    """Return the stored result `run.reuse` admits for `key` in `context`, unless its env's license refuses it."""
    hit = store.get(key, context, run.reuse, record=False)
    return hit if hit is not None and _license_admits(run, hit) else None


def _license_admits(run: RunConfig, hit: StoredResult) -> bool:
    """Whether `hit` may be served: its context records an env license `run.licenses` accepts.

    A result whose context records no env license is refused unless `run.licenses` allows unlicensed hits.
    """
    recorded = hit.context.env_license
    if recorded is None:
        return run.licenses is not None and run.licenses.allow_unlicensed_hits
    return run.licenses is None or run.licenses.refusal("", recorded) is None


def _committed(store: ResultStore, key: LogicalKey, context: ExecutionContext, run: RunConfig) -> StoredResult | None:
    """Return the entry another writer committed first for `key` in `context`, or None when its license refuses it.

    Raises:
        RuntimeError: the committed entry cannot be read back.
    """
    stored = store.get(key, context, ReusePolicy.EXACT, record=False)
    if stored is None:
        raise RuntimeError(f"the entry another writer committed for {key.task!r} could not be read back")
    return stored if _license_admits(run, stored) else None


def _reuse_attributes(run: RunConfig, hit: StoredResult) -> dict[str, object]:
    """Return what a reuse activity records: the verify settings and, for a hit with no env license, its allowance."""
    return {**_verify_attributes(run), **_unlicensed(hit)}


def _unlicensed(hit: StoredResult) -> dict[str, object]:
    """Return `senselab:unlicensedReuse` for a served result whose context records no env license, else nothing."""
    return {"senselab:unlicensedReuse": True} if hit.context.env_license is None else {}


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
    """Return the verify settings a call's activity records: the fraction, how hits are sampled, the thresholds."""
    return {
        "senselab:verifyFraction": float(run.verify),
        "senselab:verifySampling": SAMPLING,
        "senselab:verifyThresholds": run.thresholds.canonical_digest(),
    }


def _verify(  # noqa: PLR0913 - the hit, its recomputation and the thresholds judging them
    store: ResultStore,
    key: LogicalKey,
    hit: StoredResult,
    context: ExecutionContext,
    outputs: dict[str, object],
    thresholds: Thresholds,
) -> str:
    """Compare a sampled hit with its recomputation under `thresholds`, record the evidence and return its digest.

    A disagreement splits the store's class of the two members (`ResultStore.add_evidence`) and is logged with the
    paths that differ, never their values.
    """
    comparison = compare(hit.outputs, outputs, thresholds)
    stored, current = EvidenceSide.of(hit.context), EvidenceSide.of(context)
    evidence = EquivalenceEvidence(
        task=key.task,
        backend=key.backend,
        key=key.digest(),
        stored=stored,
        current=current,
        mismatched=comparison.differences,
        thresholds=comparison.thresholds,
        run=run_id(),
    )
    store.add_evidence(evidence)
    if not comparison.agree:
        logger.warning(
            "verify: %s on %s disagrees with its result from %s in %s",
            key.backend,
            current.member(),
            stored.member(),
            ", ".join(comparison.paths),
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
    hardware = host_hardware()
    placement = resolve_execution(spec, entry, profile, hardware, packaged().registry, packaged().catalogue)
    return model, hardware, replace(placement, **request_components(placement.backend))


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
    """Return `pin_model` under the profile's network policy, its source allowlist checked before resolving."""
    return pin_model(
        spec.name,
        ref,
        catalogue,
        resolver=replace(PROCESS_RESOLVER, network=profile.network),
        admit=lambda model_id: profile.check_source(model_id, catalogue),
    )


def pin_model(
    task: str,
    ref: ModelRef | None,
    catalogue: Catalogue,
    *,
    resolver: Resolver = PROCESS_RESOLVER,
    admit: Callable[[str], None] | None = None,
) -> tuple[ResolvedModel | None, CatalogueEntry | None]:
    """Return the model `ref` resolves to and the catalogue entry of that `(id, revision)`, None off every pin.

    A catalogue model at its catalogue revision is the catalogue's pin, with no lookup; any other is `resolve()`d
    through `resolver`, and a hub model's id is first given to `admit`, which raises to refuse it. With `ref` None the
    task's catalogue default is used, and a task without one has no model.
    """
    if ref is None:
        try:
            default = catalogue.default(task)
        except UnknownModel:
            return None, None
        ref = default.ref
    entry = next((e for e in catalogue.entries if e.id == ref.id), None) if ref.source == "hf" else None
    if entry is not None and ref.ref == entry.revision:
        return ResolvedModel(ref=ref, identity=entry.revision, license=entry.license), entry
    if ref.source == "hf" and admit is not None:
        admit(ref.id)
    resolved = resolve(ref, resolver=resolver)
    if entry is None or resolved.identity != entry.revision:
        return resolved, None
    return resolved.model_copy(update={"license": entry.license}), entry


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
def host_hardware() -> HardwareInfo:
    """Return this host's hardware, probed once per process."""
    return probe_hardware()


_POOL: dict[tuple[str, ...], InProcessWorker | WorkerClient] = {}
_HELD: dict[tuple[str, ...], threading.Lock] = {}
_CACHES: dict[int | None, ModelCache] = {}
_POOL_LOCK = threading.Lock()
_SERVERS: dict[tuple[str, str, str], VllmServer] = {}
_SERVER_LOCK = threading.Lock()


def _context_fields(
    backend: BackendSpec, env: EnvHandle, expected: Attestation | None, declared: Mapping[str, str]
) -> dict[str, Any]:
    """Return the env's context fields: with the attestation of a server, the launch digest of any worker process.

    `declared` are the variables the profile sets on every process it starts.
    """
    if expected is not None:
        assert backend.server is not None
        variables = {**server_variables(backend.server.settings), **declared}
        return attested_fields(env, expected, server_launch(env, variables))
    fields = env.context_fields()
    if backend.env.kind == "host":
        return fields
    return {**fields, "driver": {**fields["driver"], "launch_env": launch(env.python, declared).digest}}


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
            network=profile.network,
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
    if backend.env.kind != "host":
        inputs = {port: wire_form(v, READER) if isinstance(v, Frozen) else v for port, v in inputs.items()}
    held: contextlib.AbstractContextManager[InProcessWorker | WorkerClient] = (
        contextlib.nullcontext(_server_worker(backend, init, server))
        if server is not None
        else _checked_out(backend, env, init, model, profile)
    )
    with held as worker:
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


@contextlib.contextmanager
def _checked_out(
    backend: BackendSpec, env: EnvHandle, init: dict[str, Any], model: ResolvedModel | None, profile: ExecutionProfile
) -> Iterator[InProcessWorker | WorkerClient]:
    """Hold the pooled worker for this backend, env, init and (in process) budget or (isolated) network; others wait.

    A dead pooled worker is closed and replaced.
    """
    host = backend.env.kind == "host"
    budget = profile.model_cache_budget_bytes
    variant = str(budget) if host else profile.network
    pool_key = (backend.env.kind, env.python, backend.worker, jcs_dumps(init).decode(), variant)
    with _POOL_LOCK:
        held = _HELD.setdefault(pool_key, threading.Lock())
    with held:
        worker = _POOL.get(pool_key)
        if not (isinstance(worker, InProcessWorker) or (worker is not None and worker.alive)):
            if worker is not None:
                worker.close()
            worker = _new_worker(backend, env, init, model, profile)
            with _POOL_LOCK:
                _POOL[pool_key] = worker
        yield worker


def _new_worker(
    backend: BackendSpec, env: EnvHandle, init: dict[str, Any], model: ResolvedModel | None, profile: ExecutionProfile
) -> InProcessWorker | WorkerClient:
    if backend.env.kind != "host":
        return WorkerClient(
            env.python,
            backend.worker,
            init=init,
            variables=network_variables(profile.network),
            load_timeout=profile.timeouts.load_s,
            request_timeout=profile.timeouts.request_s,
        )
    budget = profile.model_cache_budget_bytes
    identity = model.identity if model is not None else value_digest("backend", {"name": backend.name})
    key: ModelKey = (backend.name, identity, init["device"], PRECISION)
    return InProcessWorker(
        backend.worker, init=init, key=key, cache=_cache(budget), size_of=_size_of(backend.worker, budget)
    )


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
    started: datetime,
    attributes: Mapping[str, object],
) -> tuple[kprov.Record, kprov.Record]:
    """Return the execution's agent and its activity, which used the inputs and generated the output.

    `attributes` are recorded on the activity beside the key, the context and the profile's digest.
    """
    agent = agent_record(key, context)
    activity = kprov.activity(
        core_vocabulary(),
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
