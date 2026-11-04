"""The one key builder: a call and its placement to the call's logical key (ADR-002, ADR-004)."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, TypedDict

from senselab_core.kernel.declare import BackendParams, BackendSpec, TaskParams, TaskSpec, key_for
from senselab_core.kernel.identity import (
    ExecutionContext,
    InputIdentity,
    LogicalKey,
    ProfileSliceIdentity,
    ResolvedModel,
)
from senselab_core.kernel.plan import slice_identity


@dataclass(frozen=True)
class Call:
    """A task call: the task, its params, input identities, resolved model and profile slices."""

    task: TaskSpec
    params: TaskParams
    inputs: Mapping[str, InputIdentity]
    model: ResolvedModel | None = None
    profile_slices: Mapping[str, ProfileSliceIdentity] = field(default_factory=dict)


class MissingProfileSlice(LookupError):
    """A task declares a profile slice that neither the run config's settings nor the dataset profile holds."""


def profile_slices(
    task: TaskSpec, settings: Mapping[str, Any], dataset_profile: Mapping[str, Any] | None = None
) -> dict[str, ProfileSliceIdentity]:
    """Return each profile slice `task` declares, by name, keyed by the identity of its value (`slice_identity`).

    A `run.<path>` slice is cut from `settings`, a `dataset.<path>` one from `dataset_profile`, following the dotted
    path one mapping key per part. Nothing else of either mapping is read.

    Raises:
        MissingProfileSlice: a declared slice's path is absent.
    """
    sources: dict[str, Mapping[str, Any] | None] = {"run": settings, "dataset": dataset_profile}
    cut: dict[str, ProfileSliceIdentity] = {}
    for name in task.profile_slices:
        source, *path = name.split(".")
        value: Any = sources[source]
        for part in path:
            if not isinstance(value, Mapping) or part not in value:
                given = "no dataset profile" if sources[source] is None else f"no {part!r} under {source}"
                raise MissingProfileSlice(f"{task.name!r} reads profile slice {name!r}; there is {given}")
            value = value[part]
        cut[name] = slice_identity(value)
    return cut


SINGLE_ITEM = "single"
"""The batch formation of a request holding one item, the only request `run_call` sends."""


class RequestComponents(TypedDict):
    """The batch formation and chunk grid of the requests `run_call` sends a backend."""

    batch_formation: str
    chunk_grid: str | None


def request_components(backend: BackendSpec) -> RequestComponents:
    """Return the batch formation and chunk grid of the requests `run_call` sends `backend`."""
    return {"batch_formation": SINGLE_ITEM, "chunk_grid": backend.chunk_grid}


@dataclass(frozen=True)
class Placement:
    """Where a call runs: the backend and its params, the device, and how batches and chunks are formed.

    `batch_formation` and `chunk_grid` describe the placement whether or not the backend declares them
    output-affecting. `env_root` is the directory the backend's env is built under (None for a host env) and
    `why` the sentence `explain` returns; neither reaches a key.
    """

    backend: BackendSpec
    backend_params: BackendParams
    device: str
    batch_formation: str | None = None
    chunk_grid: str | None = None
    env_root: str | None = None
    why: str = ""

    def explain(self) -> str:
        """Return the backend, stack, env, device and the reason for the choice, as one line."""
        env = self.backend.env
        where = env.kind if env.name is None else f"{env.kind}:{env.name}"
        return (
            f"{self.backend.task}: backend {self.backend.name!r} (stack {self.backend.stack!r}, env {where}, "
            f"device {self.device}); {self.why}"
        )


def build_key(call: Call, placement: Placement, context: ExecutionContext) -> LogicalKey:
    """Return the logical key of running `call` under `placement`; `context` never reaches it.

    The key holds the backend's name and version, its params, and the batch formation and chunk grid
    the backend declares output-affecting; no other placement field.

    Raises:
        TypeError: if `context` is not an `ExecutionContext`.
        ValueError, UnpinnedModel, pydantic.ValidationError: as `key_for`.
    """
    if not isinstance(context, ExecutionContext):
        raise TypeError(f"context must be an ExecutionContext, got {type(context).__name__}")
    declared = placement.backend.output_affecting
    return key_for(
        call.task,
        placement.backend,
        params=call.params,
        backend_params=placement.backend_params,
        inputs=call.inputs,
        model=call.model,
        profile_slices=call.profile_slices,
        batch_formation=placement.batch_formation if "batch_formation" in declared else None,
        chunk_grid=placement.chunk_grid if "chunk_grid" in declared else None,
    )
