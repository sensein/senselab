"""The execution profile (`execution-profile@4`) and `resolve_execution`, which places a task call under it.

A profile holds stack preferences per task, device preference, env roots, the torch index override, the model-source
allowlist, the model-cache budget, the retry policy, the network policy, the resource classes, the substrate, the
pool, calibration, job leases, fault budgets, worker timeouts and service settings. `ExecutionProfile.load` deep-merges
a user mapping or YAML file over a packaged profile (`laptop`, `slurm`) or a profile file of the user's own. A profile
is recorded with each result through `canonical_digest` and never reaches a key.
"""

import difflib
import os
import platform
import re
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from typing import Annotated, Any, Final, Literal

from pydantic import Field, model_validator

from senselab_core.kernel._base import Frozen, NonEmptyStr
from senselab_core.kernel.declare import BackendSpec, Registry, TaskSpec
from senselab_core.kernel.formats import (
    EXECUTION_PROFILE_FORMAT,
    RETIRED_EXECUTION_PROFILE_FORMATS,
    FormatTag,
    Formatted,
    FormatVerdict,
    check,
)
from senselab_core.runtime.catalogue import Catalogue, CatalogueEntry
from senselab_core.runtime.config import expand_variables, load_yaml, merge_over
from senselab_core.runtime.execution_settings import (
    CalibrationSettings,
    FaultBudgets,
    JobLease,
    PoolSettings,
    ResourceClass,
    ServiceSettings,
    SizingSettings,
    SubstrateSpec,
    SupervisorSettings,
    Timeouts,
    is_service_class,
)
from senselab_core.runtime.hardware import HardwareInfo
from senselab_core.runtime.keys import Placement
from senselab_core.runtime.locks import TIMEOUT_S as BUILD_LOCK_TIMEOUT_S
from senselab_core.runtime.network import Network
from senselab_core.runtime.retry import RetryPolicy

DEFAULT_PROFILE = "laptop"
_PROFILE_SUFFIXES = (".yaml", ".yml")
_UNFILLED = re.compile(r"^<fill: (.+)>$")


class UnknownProfileKey(ValueError):
    """A profile key that the format does not define."""


class UnlistedSource(ValueError):
    """A model whose hub organisation is not in the profile's allowlist."""


class NoBackend(LookupError):
    """No declared backend serves the task and model under the profile's stack preferences."""


class NoDevice(RuntimeError):
    """The chosen backend runs on no device that the profile lists and the node has."""


_DROPPED_KEYS: Final = {
    "execution-profile@3": "; drop calibration.probe_items and calibration.resolve_s, which are now derived",
}


class ProfileFormatRetired(ValueError):
    """A profile in a retired format; it is rewritten as `execution-profile@4.0`."""

    def __init__(self, found: str) -> None:
        """Name the declared format, and the keys a newer major dropped from it."""
        dropped = _DROPPED_KEYS.get(found.split(".", maxsplit=1)[0], "")
        super().__init__(
            f"the profile declares the retired format {found}; it is not read. Rewrite it as"
            f" {EXECUTION_PROFILE_FORMAT}, whose keys `ExecutionProfile` documents{dropped}"
        )
        self.found = found


class UnfilledProfile(ValueError):
    """A profile that still holds `<fill: ...>` placeholders, which a user's own values replace."""

    def __init__(self, name: str, unfilled: Mapping[str, str]) -> None:
        """Name the profile and each unfilled key with what goes there."""
        listed = "; ".join(f"{key} ({what})" for key, what in unfilled.items())
        super().__init__(
            f"profile {name!r} has values to fill before it loads: {listed}. Copy the packaged profile to a file of"
            " your own, replace each `<fill: ...>`, and pass that file, or give the values as an override"
        )
        self.name, self.unfilled = name, dict(unfilled)


class IncompleteProfile(ValueError):
    """A profile file that omits keys: a file passed as a profile is a whole one."""

    def __init__(self, source: str, missing: list[str]) -> None:
        """Name the file and the keys it lacks."""
        super().__init__(
            f"{source}: a profile file must be complete, and this one lacks {', '.join(missing)}. To change a few"
            " values of a packaged profile, pass them as an override: `ExecutionProfile.load(name, file)`,"
            " or `--override FILE` where a command takes one"
        )
        self.source, self.missing = source, missing


class QosCapTooSmall(ValueError):
    """A resource class whose workers need more CPUs than the substrate's QOS CPU cap allows at all."""

    def __init__(self, resource_class: str, cpus: int, cap: int, node: str | None = None) -> None:
        """Name the class, its CPUs per worker, the cap and the node placed there."""
        where = "" if node is None else f" (node {node!r})"
        super().__init__(
            f"resource class {resource_class!r}{where} needs {cpus} CPUs per worker; the QOS CPU cap is {cap}"
        )
        self.resource_class, self.cpus, self.cap, self.node = resource_class, cpus, cap, node


class Envs(Frozen):
    """Env settings.

    Attributes:
        server_start_margin_s: Seconds a waiter on a server start lease waits beyond the start-up timeout.
        torch_index: Replaces the probed CUDA wheel index when set.
    """

    server_start_margin_s: Annotated[float, Field(gt=0)]
    torch_index: NonEmptyStr | None = None


class ExecutionProfile(Formatted):
    """Where and how calls run on one site.

    Attributes:
        name: The packaged profile this was loaded from, or the file stem of a user profile.
        stack_preferences: Per task, its stacks best first.
        devices: Device kinds best first (`cuda`, `mps`, `cpu`).
        env_roots: Per env kind (`isolated`, `server`), the directory env trees are built under;
            `${VAR:-fallback}`, then `~` and `$VAR`, are expanded where the root is used.
        envs: Env settings.
        model_sources: Allowed hub organisations; None allows those of the catalogue entries.
        model_cache_budget_bytes: Cap on the model cache; None measures it (`hardware.measured_budget`).
        retry: How a transient fault is retried.
        network: `online`, or `offline` to forbid any fetch.
        resource_classes: The worker classes a node is placed on, by name (`cpu`, `gpu:<type>`, `mps`, `svc:<id>`).
        substrate: What starts workers: `local` processes or `slurm` jobs.
        pool: How pool workers claim and yield, and the quantum ceilings.
        sizing: The fractions and targets a unit's batch, quantum and worker count are derived from (`size`).
        calibration: How a unit without a measure is probed, where measures are kept and whose are trusted.
        lease: How a lease is kept alive: its stale bound and renewal share, for job and pool leases alike.
        faults: The retry budget of each fault class.
        timeouts: A worker's load and request timeouts.
        services: When an idle service exits.
        supervisor: How a supervisor reads a launched process's start time.
    """

    format: FormatTag = EXECUTION_PROFILE_FORMAT
    name: NonEmptyStr
    stack_preferences: Mapping[str, tuple[NonEmptyStr, ...]]
    devices: tuple[Literal["cuda", "mps", "cpu"], ...]
    env_roots: Mapping[str, NonEmptyStr]
    envs: Envs
    model_sources: tuple[NonEmptyStr, ...] | None
    model_cache_budget_bytes: int | None
    retry: RetryPolicy
    network: Network
    resource_classes: Mapping[NonEmptyStr, ResourceClass]
    substrate: SubstrateSpec
    pool: PoolSettings
    sizing: SizingSettings
    calibration: CalibrationSettings
    lease: JobLease
    faults: FaultBudgets
    timeouts: Timeouts
    services: ServiceSettings
    supervisor: SupervisorSettings

    @model_validator(mode="after")
    def _check_classes(self) -> "ExecutionProfile":
        wrong = sorted(name for name, c in self.resource_classes.items() if not c.matches(name))
        if wrong:
            raise ValueError(f"resource classes {wrong} are not named for their device and GPU type")
        return self

    def check_qos_cap(self, placed: Mapping[str, str]) -> None:
        """Raise `QosCapTooSmall` for a placed class whose `cpus` exceed the substrate's `qos_cpu_cap`.

        `placed` maps each node to the resource class it is placed on; the error names the first such node.
        """
        cap = self.substrate.qos_cpu_cap
        for node, name in placed.items():
            grant = self.resource_classes.get(name)
            if cap is not None and grant is not None and grant.cpus is not None and grant.cpus > cap:
                raise QosCapTooSmall(name, grant.cpus, cap, node)

    @classmethod
    def load(
        cls, source: str | Path = DEFAULT_PROFILE, override: Mapping[str, Any] | Path | None = None
    ) -> "ExecutionProfile":
        """Return the profile `source` names with `override` (a mapping or a YAML file) deep-merged over it.

        `source` is a packaged profile name (`laptop`, `slurm`), or a path to a profile file holding a whole profile.

        Raises:
            FileNotFoundError: `source` is neither a packaged profile nor a file.
            IncompleteProfile: the merged mapping lacks keys, as a partial file passed as a profile does.
            ProfileFormatRetired: the merged mapping declares a retired format.
            UnfilledProfile: a value is still a `<fill: ...>` placeholder.
            UnknownProfileKey: a key of `override` that the format does not define, naming the closest one.
            LeaseBoundError: two of the lease's beat intervals plus its attribute delay are not below its stale bound.
            ValueError: the merged mapping is invalid.
        """
        packaged = _read(_profile_source(source))
        extra = override if isinstance(override, Mapping) else _read(override) if override is not None else {}
        merged = merge_over(packaged, extra)
        if check(EXECUTION_PROFILE_FORMAT, merged.get("format"), RETIRED_EXECUTION_PROFILE_FORMATS) is (
            FormatVerdict.RETIRED
        ):
            raise ProfileFormatRetired(str(merged.get("format")))
        missing = [k for k, f in cls.model_fields.items() if f.is_required() and k not in merged]
        if missing:
            raise IncompleteProfile(os.fspath(source), missing)
        unfilled = _unfilled(merged)
        if unfilled:
            raise UnfilledProfile(str(merged.get("name", source)), unfilled)
        _check_keys(merged, cls.model_fields)
        profile = cls.model_validate(merged)
        profile.lease.check_bound()
        return profile

    @property
    def probe_s(self) -> float:
        """Seconds of warm probe requests a probe may take: `sizing.load_share × sizing.target_wall_s`."""
        return self.sizing.load_share * self.sizing.target_wall_s

    @property
    def peer_wait_s(self) -> float:
        """Seconds a coordinator waits for a peer's probe of the same unit: `probe_s + lease.stale_after_s`."""
        return self.probe_s + self.lease.stale_after_s

    @property
    def resolve_s(self) -> float:
        """Seconds a unit waits for a worker's ready mark: `2 × (lease.stale_after_s + the build lock timeout)`."""
        return 2 * (self.lease.stale_after_s + BUILD_LOCK_TIMEOUT_S)

    def env_root(self, kind: str) -> Path:
        """Return the root for env `kind` with `${VAR:-fallback}`, then `~` and `$VAR`, expanded.

        `${VAR:-fallback}` is VAR's value, or `fallback` when VAR is unset or empty.

        Raises:
            KeyError: the profile has no root for `kind`.
            ValueError: the root names an unset variable.
        """
        raw = self.env_roots[kind]
        text = expand_variables(raw)
        if "$" in text:
            raise ValueError(f"env root {raw!r} names a variable that is not set")
        return Path(text)

    def check_source(self, model_id: str, catalogue: Catalogue) -> None:
        """Raise `UnlistedSource` unless `model_id`'s organisation is allowed.

        The allowlist is `model_sources`, or the organisations of `catalogue` when that is None.
        """
        allowed = self.model_sources if self.model_sources is not None else catalogue.sources()
        organisation = model_id.split("/", maxsplit=1)[0] if "/" in model_id else ""
        if organisation not in allowed:
            raise UnlistedSource(
                f"model {model_id!r} is from {organisation!r}, not in the allowed sources {list(allowed)}"
            )


def packaged_profiles() -> tuple[str, ...]:
    """Return the names of the packaged profiles, sorted."""
    directory = resources.files("senselab_core.runtime").joinpath("data", "profiles")
    return tuple(sorted(p.name.removesuffix(".yaml") for p in directory.iterdir() if p.name.endswith(".yaml")))


def _profile_source(source: str | Path) -> Any:  # noqa: ANN401 - a Path or a package resource
    """Return where profile `source` is read from: a path when it looks like one, else a packaged profile."""
    text = os.fspath(source)
    if isinstance(source, Path) or os.sep in text or text.endswith(_PROFILE_SUFFIXES):
        if not Path(text).is_file():
            raise FileNotFoundError(f"{text}: not a profile file")
        return Path(text)
    if text not in packaged_profiles():
        raise FileNotFoundError(f"{text!r} is neither a packaged profile {list(packaged_profiles())} nor a file path")
    return resources.files("senselab_core.runtime").joinpath("data", "profiles", f"{text}.yaml")


def _unfilled(node: Any, path: str = "") -> dict[str, str]:  # noqa: ANN401 - any YAML node
    """Return each `<fill: what>` string under `node` by its key path, with `what`."""
    if isinstance(node, str):
        found = _UNFILLED.match(node)
        return {path: found[1]} if found else {}
    if isinstance(node, Mapping):
        return {
            k: v
            for key, item in node.items()
            for k, v in _unfilled(item, f"{path}.{key}" if path else str(key)).items()
        }
    if isinstance(node, list):
        return {k: v for i, item in enumerate(node) for k, v in _unfilled(item, f"{path}[{i}]").items()}
    return {}


def _read(source: Any) -> dict[str, Any]:  # noqa: ANN401 - a Path or a package resource
    return load_yaml(source, "profile")


_SECTIONS: Mapping[str, type[Frozen]] = {
    "envs": Envs,
    "retry": RetryPolicy,
    "substrate": SubstrateSpec,
    "pool": PoolSettings,
    "calibration": CalibrationSettings,
    "lease": JobLease,
    "faults": FaultBudgets,
    "timeouts": Timeouts,
    "services": ServiceSettings,
    "supervisor": SupervisorSettings,
}


def _check_keys(mapping: Mapping[str, Any], fields: Mapping[str, object], path: str = "") -> None:
    known = list(fields)
    for key, value in mapping.items():
        if key not in known:
            close = difflib.get_close_matches(str(key), known, n=1)
            hint = f"did you mean {close[0]!r}?" if close else f"known keys: {known}"
            raise UnknownProfileKey(f"unknown profile key {path + str(key)!r}; {hint}")
        if path or not isinstance(value, Mapping):
            continue
        if key in _SECTIONS:
            _check_keys(value, _SECTIONS[key].model_fields, f"{key}.")
        elif key == "resource_classes":
            for name, given in value.items():
                if isinstance(given, Mapping):
                    _check_keys(given, ResourceClass.model_fields, f"resource_classes.{name}.")


def _has(resource: ResourceClass, hardware: HardwareInfo) -> bool:
    if resource.device == "cpu":
        return True
    if resource.device == "mps":
        return hardware.apple_silicon
    wanted = resource.gpu_type
    return any(wanted is None or wanted.lower() in g.name.lower() for g in hardware.gpus)


def resource_class(
    backend: BackendSpec, profile: ExecutionProfile, hardware: HardwareInfo | None
) -> tuple[str, ResourceClass] | None:
    """Return the first resource class of `profile` that runs `backend`, by name, or None.

    Devices are taken in `profile.devices` order and classes in profile order; a class runs the backend when the
    backend declares its device and, with `hardware` given, this host has that device (and GPU type). With
    `hardware` None the class is not judged against this host, as when a scheduler places the worker. A server
    backend takes a service class (`svc:<id>`) that runs it before any other class, so its consumers share one
    registered server; no other backend is placed on a service class.
    """
    server = backend.env.kind == "server"
    for services in (True, False) if server else (False,):
        for device in profile.devices:
            if device not in backend.devices:
                continue
            for name, resource in profile.resource_classes.items():
                runs = resource.device == device and (hardware is None or _has(resource, hardware))
                if runs and is_service_class(name) == services:
                    return name, resource
    return None


def _host_devices(hardware: HardwareInfo) -> str:
    return f"{'cuda ' if hardware.gpus else ''}{'mps ' if hardware.apple_silicon else ''}cpu"


def _chosen(
    task: TaskSpec, model_entry: CatalogueEntry | None, profile: ExecutionProfile, registry: Registry
) -> tuple[BackendSpec, bool]:
    served = [
        b
        for b in registry.backends(task.name)
        if model_entry is None
        or (
            b.name in model_entry.backends
            and b.stack in model_entry.stacks
            and model_entry.id in (b.models or (model_entry.id,))
        )
    ]
    wanted = profile.stack_preferences.get(task.name, ())
    candidates = [b for s in wanted for b in served if b.stack == s] if wanted else served
    if not candidates:
        raise NoBackend(
            f"no backend of task {task.name!r} serves model {model_entry.id if model_entry else None!r} on stacks "
            f"{list(wanted) or 'any'}; available: {sorted({b.stack for b in served})}"
        )
    return candidates[0], bool(wanted)


def resolve_execution(  # noqa: PLR0913 - the brief fixes the signature and the allowlist needs the catalogue
    task: TaskSpec,
    model_entry: CatalogueEntry | None,
    profile: ExecutionProfile,
    hardware: HardwareInfo | None,
    registry: Registry,
    catalogue: Catalogue,
) -> Placement:
    """Return where `task` runs under `profile`: backend (hence stack and env), resource class, device and partition.

    The first stack of `profile.stack_preferences[task.name]` with a backend serving `model_entry` wins, or the
    first declared backend when the profile lists none. The resource class is `resource_class`'s, judged against
    `hardware` (None when a scheduler places the worker); the device is the class's. The batch size is 1: `size`
    sizes it from an estimate where the work runs.

    Raises:
        UnlistedSource: the model's organisation is neither in `model_sources` nor, when that is None, in `catalogue`.
        NoBackend: no backend serves the task and model under the preferences.
        NoDevice: no resource class runs the chosen backend here; names the host, never falls back to another device.
    """
    if model_entry is not None:
        profile.check_source(model_entry.id, catalogue)
    backend, preferred = _chosen(task, model_entry, profile, registry)
    chosen = resource_class(backend, profile, hardware)
    if chosen is None:
        here = "" if hardware is None else f" and node {platform.node()!r} has {_host_devices(hardware)}"
        raise NoDevice(
            f"backend {backend.name!r} of {task.name!r} runs on {list(backend.devices)}; profile {profile.name!r} "
            f"lists devices {list(profile.devices)} and resource classes {sorted(profile.resource_classes)}{here}"
        )
    name, resource = chosen
    why = f"stack {backend.stack!r} " + (f"preferred by profile {profile.name!r}" if preferred else "first declared")
    return Placement(
        backend=backend,
        backend_params=backend.params(),
        device=resource.device,
        why=f"{why}; resource class {name!r} is the first that runs it",
        resource_class=name,
        partition=resource.partition,
    )
