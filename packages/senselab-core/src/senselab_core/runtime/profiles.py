"""The execution profile (`execution-profile@1`) and `resolve_execution`, which places a task call under it.

A profile holds stack preferences per task, device preference, env roots, the torch index override, the model-source
allowlist, the model-cache budget, the retry policy and the network policy. `ExecutionProfile.load` deep-merges a
user mapping or YAML file over a packaged profile (`laptop`, `orcd`). A profile is recorded with each result through
`canonical_digest` and never reaches a key.
"""

import difflib
import os
import platform
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml

from senselab_core.kernel._base import Frozen, NonEmptyStr
from senselab_core.kernel.declare import BackendSpec, Registry, TaskSpec
from senselab_core.kernel.formats import FormatTag, Formatted
from senselab_core.runtime.catalogue import Catalogue, CatalogueEntry
from senselab_core.runtime.hardware import HardwareInfo
from senselab_core.runtime.keys import Placement
from senselab_core.runtime.retry import RetryPolicy

EXECUTION_PROFILE_FORMAT = "execution-profile@1.0"
DEFAULT_PROFILE = "laptop"
_DERIVATIONS = "derivations"


class UnknownProfileKey(ValueError):
    """A profile key that the format does not define."""


class UnlistedSource(ValueError):
    """A model whose hub organisation is not in the profile's allowlist."""


class NoBackend(LookupError):
    """No declared backend serves the task and model under the profile's stack preferences."""


class NoDevice(RuntimeError):
    """The chosen backend runs on no device that the profile lists and the node has."""


class Envs(Frozen):
    """Env settings: `torch_index` replaces the probed CUDA wheel index when set."""

    torch_index: NonEmptyStr | None = None


class ExecutionProfile(Formatted):
    """Where and how calls run on one site.

    Attributes:
        name: The packaged profile this was loaded from, or the file stem of a user profile.
        stack_preferences: Per task, its stacks best first.
        devices: Device kinds best first (`cuda`, `mps`, `cpu`).
        env_roots: Per env kind (`isolated`, `server`), the directory env trees are built under;
            `~` and `$VAR` are expanded where the root is used.
        envs: Env settings.
        model_sources: Allowed hub organisations; None allows those of the catalogue entries.
        model_cache_budget_bytes: Cap on the model cache; None for none.
        retry: How a transient fault is retried.
        network: `online`, or `offline` to forbid any fetch.
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
    network: Literal["online", "offline"]

    @classmethod
    def load(cls, name: str = DEFAULT_PROFILE, override: Mapping[str, Any] | Path | None = None) -> "ExecutionProfile":
        """Return packaged profile `name` with `override` (a mapping or a YAML file) deep-merged over it.

        Raises:
            UnknownProfileKey: a key of `override` that the format does not define, naming the closest one.
            ValueError: the packaged profile lacks a derivation for a key, or the merged mapping is invalid.
        """
        packaged = _read(resources.files("senselab_core.runtime").joinpath("data", "profiles", f"{name}.yaml"))
        derived = packaged.get(_DERIVATIONS, {})
        missing = [
            k for k in packaged if k not in (_DERIVATIONS, "format", "name") and not str(derived.get(k, "")).strip()
        ]
        if missing:
            raise ValueError(f"profile {name!r} has no derivation for {missing}")
        packaged.pop(_DERIVATIONS)
        extra = override if isinstance(override, Mapping) else _read(override) if override is not None else {}
        merged = _merge(packaged, {k: v for k, v in extra.items() if k != _DERIVATIONS})
        _check_keys(merged, cls.model_fields)
        return cls.model_validate(merged)

    def env_root(self, kind: str) -> Path:
        """Return the root for env `kind` with `~` and `$VAR` expanded.

        Raises:
            KeyError: the profile has no root for `kind`.
            ValueError: the root names an unset variable.
        """
        raw = self.env_roots[kind]
        expanded = os.path.expanduser(os.path.expandvars(raw))
        if "$" in expanded:
            raise ValueError(f"env root {raw!r} names a variable that is not set")
        return Path(expanded)

    def check_source(self, model_id: str, catalogue: Catalogue | None = None) -> None:
        """Raise `UnlistedSource` unless `model_id`'s organisation is allowed.

        The allowlist is `model_sources`, or the organisations of `catalogue` when that is None.
        """
        allowed = self.model_sources if self.model_sources is not None else catalogue.sources() if catalogue else None
        organisation = model_id.split("/", maxsplit=1)[0] if "/" in model_id else ""
        if allowed is not None and organisation not in allowed:
            raise UnlistedSource(
                f"model {model_id!r} is from {organisation!r}, not in the allowed sources {list(allowed)}"
            )


def _read(source: Any) -> dict[str, Any]:  # noqa: ANN401 - a Path or a package resource
    raw = yaml.safe_load(source.read_text("utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("a profile is a mapping")
    return raw


def _merge(base: Mapping[str, Any], top: Mapping[str, Any]) -> dict[str, Any]:
    return {
        k: _merge(base[k], top[k])
        if isinstance(base.get(k), Mapping) and isinstance(top.get(k), Mapping)
        else top[k]
        if k in top
        else base[k]
        for k in {**base, **top}
    }


def _check_keys(mapping: Mapping[str, Any], fields: Mapping[str, object], path: str = "") -> None:
    known = list(fields)
    for key, value in mapping.items():
        if key not in known:
            close = difflib.get_close_matches(str(key), known, n=1)
            hint = f"did you mean {close[0]!r}?" if close else f"known keys: {known}"
            raise UnknownProfileKey(f"unknown profile key {path + str(key)!r}; {hint}")
        nested = {"envs": Envs.model_fields, "retry": RetryPolicy.model_fields}.get(key) if not path else None
        if nested is not None and isinstance(value, Mapping):
            _check_keys(value, nested, f"{key}.")


def _backend_device(backend: BackendSpec, profile: ExecutionProfile, hardware: HardwareInfo) -> str | None:
    have = {"cpu"} | ({"cuda"} if hardware.gpus else set()) | ({"mps"} if hardware.apple_silicon else set())
    return next((d for d in profile.devices if d in have and d in backend.devices), None)


def resolve_execution(
    task: TaskSpec,
    model_entry: CatalogueEntry | None,
    profile: ExecutionProfile,
    hardware: HardwareInfo,
    registry: Registry,
) -> Placement:
    """Return where `task` runs under `profile` on `hardware`: backend (hence stack and env) and device.

    The first stack of `profile.stack_preferences[task.name]` with a backend serving `model_entry` wins, or the
    first declared backend when the profile lists none. A device is the first of `profile.devices` that the host
    has and the backend declares.

    Raises:
        UnlistedSource: the model's organisation is not allowed.
        NoBackend: no backend serves the task and model under the preferences.
        NoDevice: the chosen backend has no device here; names the node, never falls back to another device.
    """
    if model_entry is not None:
        profile.check_source(model_entry.id)
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
    backend = candidates[0]
    device = _backend_device(backend, profile, hardware)
    if device is None:
        raise NoDevice(
            f"backend {backend.name!r} of {task.name!r} runs on {list(backend.devices)}; profile {profile.name!r} "
            f"lists {list(profile.devices)} and node {platform.node()!r} has "
            f"{'cuda ' if hardware.gpus else ''}{'mps ' if hardware.apple_silicon else ''}cpu"
        )
    root = None if backend.env.kind == "host" else str(profile.env_root(backend.env.kind))
    why = f"stack {backend.stack!r} " + (f"preferred by profile {profile.name!r}" if wanted else "first declared")
    return Placement(backend=backend, backend_params=backend.params(), device=device, env_root=root, why=why)
