"""A task call, a run's settings and a call's result, and what a declaring package hands the runtime.

A declaring package hands the runtime its catalogue, how its backends' envs open and its tasks and backends through
the `register` function its `PACKAGES_GROUP` entry point names (`packaged`). Until one does, the catalogue is empty
and no env opens.
"""

import functools
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from importlib.metadata import entry_points
from os import PathLike
from pathlib import Path
from typing import Annotated, Any, Literal, get_type_hints

from pydantic import AfterValidator, BaseModel, ConfigDict, PlainSerializer, ValidationInfo, model_validator

from senselab_core.kernel._base import Frozen, FrozenMapping, NonEmptyStr, thawed, value_digest
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
    Params,
    ResolvedModel,
    ReusePolicy,
    text_identity_fields,
)
from senselab_core.kernel.license import License
from senselab_core.kernel.signals import PlainText, Recording, StaleSource, UndecodableSource, text_identity
from senselab_core.runtime.audio_io import DECODER, open_audio
from senselab_core.runtime.calibration import RunCalibration
from senselab_core.runtime.catalogue import Catalogue, CatalogueEntry, UnknownModel
from senselab_core.runtime.compare import RunConfigFile, Thresholds
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.hardware import HardwareInfo, probe_hardware
from senselab_core.runtime.keys import Placement, request_components
from senselab_core.runtime.licenses import LicensePolicy, LockRecords, backend_license
from senselab_core.runtime.models import resolve
from senselab_core.runtime.profiles import ExecutionProfile, resolve_execution
from senselab_core.runtime.revision import PROCESS_RESOLVER, Resolver

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
    backend a plan places is read, and validated against it (no entry is that backend's defaults).
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

    `config` is the run config file (`RunConfigFile.load`, the packaged one unless a user's is merged over it), whose
    sections are `thresholds` and `calibration`. `verify` is the fraction of cross-context hits recomputed and
    compared under `thresholds`. `licenses` is the user's license policy; with None no call is refused for its
    license. `settings` holds the values tasks read, from which a task's `run.<path>` profile slices are cut
    (`keys.profile_slices`).
    """

    reuse: ReusePolicy = DEFAULT_REUSE_POLICY
    verify: float = 0.0
    licenses: LicensePolicy | None = None
    config: RunConfigFile = field(default_factory=RunConfigFile.load)
    settings: Mapping[str, Any] = field(default_factory=lambda: FrozenMapping({}))

    def __post_init__(self) -> None:
        """Refuse a `verify` outside [0, 1], and hold `settings` read-only."""
        if not 0.0 <= self.verify <= 1.0:
            raise ValueError(f"verify is a fraction in [0, 1], got {self.verify}")
        object.__setattr__(self, "settings", FrozenMapping(dict(self.settings)))

    @property
    def thresholds(self) -> Thresholds:
        """The run config's comparison thresholds."""
        return self.config.thresholds

    @property
    def calibration(self) -> RunCalibration:
        """What the run overrides of its profile's `calibration` section (`RunCalibration.apply`)."""
        return self.config.calibration


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


EnvOpener = Callable[[BackendSpec, ExecutionProfile], EnvHandle]
"""Open the env a backend declares under a profile, building it if needed."""
EnvProbe = Callable[[BackendSpec, ExecutionProfile], bool]
"""Whether the env a backend declares is built under a profile, building nothing."""


class EnvUnreadable(Exception):  # noqa: N818 - a state, not an error of the caller
    """The env a backend declares cannot be read without building it; the message says why."""


EnvRead = Callable[[BackendSpec, ExecutionProfile], EnvHandle | None]
"""The handle of the env a backend declares if it is built under a profile, else None; builds and locks nothing.

Raises `EnvUnreadable` for any cause it cannot read it.
"""
EnvLock = Callable[[BackendSpec], Path]
"""The lock file the isolated or server env a backend declares is built from, reading nothing."""


def _unbuilt(backend: BackendSpec, profile: ExecutionProfile) -> bool:  # noqa: ARG001 - an EnvProbe
    return False


def _unread(backend: BackendSpec, profile: ExecutionProfile) -> EnvHandle | None:  # noqa: ARG001 - an EnvRead
    return None


def _no_lock(backend: BackendSpec) -> Path:
    raise LookupError(f"no package has said which lock the env of backend {backend.name!r} is built from")


class InputUnreadable(ValueError):
    """An input given as a path whose file its port type's reader cannot read or decode."""


@dataclass(frozen=True)
class Packaged:
    """What a declaring package hands the runtime: catalogue, env opener, registry, lock records, env probe and locks.

    `lock_records` gives each backend's env-lock package license records, from which `env_license` derives; without
    them no stored result is served unless the run's license policy allows unlicensed hits. `env_built` says whether
    a backend's env is built (by default none is), and `env_read` gives the built env's handle (by default none).
    `env_lock` names the lock a non-host env is built from (by default none is known).
    """

    catalogue: Catalogue
    open_env: EnvOpener
    registry: Registry = REGISTRY
    lock_records: LockRecords | None = None
    env_built: EnvProbe = _unbuilt
    env_read: EnvRead = _unread
    env_lock: EnvLock = _no_lock


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
    """Make `packaged` what the runtime reads, and return what it read before."""
    global _PACKAGED  # noqa: PLW0603 - the process-wide counterpart of REGISTRY
    previous, _PACKAGED = _PACKAGED, packaged
    _STATE["registered"] = True
    return previous


def packaged() -> Packaged:
    """Return what the runtime reads: the catalogue and env opener last given to `use_packaged`.

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


def placement_of(task: str, model: ModelRef | None, execution: ExecutionProfile) -> Placement:
    """Return where `task` on `model` would run under `execution` on this host, starting nothing.

    Raises:
        UnknownTask: the task is not declared.
        UnlistedSource, NoBackend, NoDevice: as `resolve_execution`.
    """
    spec = packaged().registry.spec(task)
    _, entry = _resolve_model(spec, model, execution, packaged().catalogue)
    placement = resolve_execution(
        spec,
        entry,
        execution,
        host_hardware(execution.sizing.device_query_s),
        packaged().registry,
        packaged().catalogue,
    )
    return replace(placement, **request_components(placement.backend))


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


def input_decoder(port_type: str) -> str | None:
    """Return the decoder a path given for `port_type` is read by: its reader's, None for a type without one."""
    return _READERS[port_type][1] if port_type in _READERS else None


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

    Raises:
        InputUnreadable: a path's file cannot be read or decoded by its port type's reader, for any cause.
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
        try:
            value = reader(value)
        except (OSError, StaleSource, UndecodableSource) as err:
            raise InputUnreadable(f"input {port!r} cannot be read: {err}") from err
    if isinstance(value, Recording | PlainText):
        return value.identity, value, decoder
    if isinstance(value, Frozen):
        return "content:" + value.canonical_digest(), value, decoder
    return "content:" + value_digest("json", {"value": value}), value, decoder


@functools.cache
def host_hardware(timeout_s: float | None = None) -> HardwareInfo:
    """Return this host's hardware, probed once per process and bound, within `timeout_s` (`sizing.device_query_s`)."""
    return probe_hardware(timeout_s=timeout_s)
