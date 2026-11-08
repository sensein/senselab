"""The execution profile (`execution-profile@3`) and `resolve_execution`, which places a task call under it.

A profile holds stack preferences per task, device preference, env roots, the torch index override, the model-source
allowlist, the model-cache budget, the retry policy, the network policy, the resource classes, the substrate, the
pool, calibration, job leases, fault budgets, worker timeouts and service settings. `ExecutionProfile.load` deep-merges
a user mapping or YAML file over a packaged profile (`laptop`, `orcd`). A profile is recorded with each result through
`canonical_digest` and never reaches a key.
"""

import difflib
import os
import platform
import re
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from typing import Annotated, Any, Literal

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
from senselab_core.runtime.config import load_yaml
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
from senselab_core.runtime.keys import SINGLE, Placement, fixed_formation
from senselab_core.runtime.network import Network
from senselab_core.runtime.retry import RetryPolicy
from senselab_core.runtime.sizing import Grant, measure_of, size

DEFAULT_PROFILE = "laptop"


class UnknownProfileKey(ValueError):
    """A profile key that the format does not define."""


class UnlistedSource(ValueError):
    """A model whose hub organisation is not in the profile's allowlist."""


class NoBackend(LookupError):
    """No declared backend serves the task and model under the profile's stack preferences."""


class NoDevice(RuntimeError):
    """The chosen backend runs on no device that the profile lists and the node has."""


class ProfileFormatRetired(ValueError):
    """A profile in a retired format; it is rewritten as `execution-profile@3.0`."""

    def __init__(self, found: str) -> None:
        """Name the declared format."""
        super().__init__(
            f"the profile declares the retired format {found}; it is not read. Rewrite it as"
            f" {EXECUTION_PROFILE_FORMAT}, whose keys `ExecutionProfile` documents"
        )
        self.found = found


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


_DEFAULTED = re.compile(r"\$\{(\w+):-([^}]*)\}")


def _expanded(raw: str) -> str:
    chosen = _DEFAULTED.sub(lambda m: os.environ.get(m[1]) or m[2], raw)
    return os.path.expanduser(os.path.expandvars(chosen))


def expand(raw: str) -> str:
    """Return `raw` with `${VAR:-fallback}`, then `~` and `$VAR`, expanded from the environment.

    Raises:
        ValueError: `raw` names a variable that is not set.
    """
    expanded = _expanded(raw)
    if "$" in expanded:
        raise ValueError(f"{raw!r} names a variable that is not set")
    return expanded


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
    def load(cls, name: str = DEFAULT_PROFILE, override: Mapping[str, Any] | Path | None = None) -> "ExecutionProfile":
        """Return packaged profile `name` with `override` (a mapping or a YAML file) deep-merged over it.

        Raises:
            ProfileFormatRetired: `override` declares a retired format.
            UnknownProfileKey: a key of `override` that the format does not define, naming the closest one.
            LeaseBoundError: two of the lease's beat intervals plus its attribute delay are not below its stale bound.
            ValueError: the merged mapping is invalid.
        """
        packaged = _read(resources.files("senselab_core.runtime").joinpath("data", "profiles", f"{name}.yaml"))
        extra = override if isinstance(override, Mapping) else _read(override) if override is not None else {}
        merged = _merge(packaged, extra)
        if check(EXECUTION_PROFILE_FORMAT, merged.get("format"), RETIRED_EXECUTION_PROFILE_FORMATS) is (
            FormatVerdict.RETIRED
        ):
            raise ProfileFormatRetired(str(merged.get("format")))
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

    def env_root(self, kind: str) -> Path:
        """Return the root for env `kind` with `${VAR:-fallback}`, then `~` and `$VAR`, expanded.

        `${VAR:-fallback}` is VAR's value, or `fallback` when VAR is unset or empty.

        Raises:
            KeyError: the profile has no root for `kind`.
            ValueError: the root names an unset variable.
        """
        raw = self.env_roots[kind]
        expanded = _expanded(raw)
        if "$" in expanded:
            raise ValueError(f"env root {raw!r} names a variable that is not set")
        return Path(expanded)

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


def _read(source: Any) -> dict[str, Any]:  # noqa: ANN401 - a Path or a package resource
    return load_yaml(source, "profile")


def _merge(base: Mapping[str, Any], top: Mapping[str, Any]) -> dict[str, Any]:
    return {
        k: _merge(base[k], top[k])
        if isinstance(base.get(k), Mapping) and isinstance(top.get(k), Mapping)
        else top[k]
        if k in top
        else base[k]
        for k in {**base, **top}
    }


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
    `hardware` (None when a scheduler places the worker); the device is the class's, and the batch size is sized
    (`size`) from the backend and model's resource table on that class, 1 when it is unmeasured.

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
    root = None if backend.env.kind == "host" else str(profile.env_root(backend.env.kind))
    why = f"stack {backend.stack!r} " + (f"preferred by profile {profile.name!r}" if preferred else "first declared")
    return Placement(
        backend=backend,
        backend_params=backend.params(),
        device=resource.device,
        env_root=root,
        why=f"{why}; resource class {name!r} is the first that runs it",
        resource_class=name,
        batch_size=_batch_size(backend, model_entry, (name, resource), profile, hardware, catalogue),
        partition=resource.partition,
    )


def _batch_size(  # noqa: PLR0913 - the backend, its model, the class it is placed on, and what sizes it
    backend: BackendSpec,
    model_entry: CatalogueEntry | None,
    placed: tuple[str, ResourceClass],
    profile: ExecutionProfile,
    hardware: HardwareInfo | None,
    catalogue: Catalogue,
) -> int:
    """Return the batch size `size` derives from `backend` and its model's measure on class `placed`; 1 unmeasured."""
    model = (None, None) if model_entry is None else (model_entry.id, model_entry.revision)
    row = catalogue.resources(backend.task, backend.name, *model)
    measured = None if row is None else row.classes.get(placed[0])
    if measured is None:
        return 1
    grant = Grant.of(placed[1], profile.pool, None if hardware is None else hardware.memory_bytes)
    return size(measure_of(measured), grant, profile.sizing, batchable=fixed_formation(backend) != SINGLE).batch_size
