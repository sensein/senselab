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
from senselab_core.kernel.placement import NodePlacement
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
"""The batch formation of a request holding one item."""


class RequestComponents(TypedDict):
    """The batch formation and chunk grid of a request sent to a backend."""

    batch_formation: str
    chunk_grid: str | None


def request_size(backend: BackendSpec, batch_size: int) -> int:
    """Return the items per request `backend` is sent under a placement of `batch_size`: 1 unless it is batch-invariant.

    Raises:
        ValueError: `batch_size` is below 1.
    """
    if batch_size < 1:
        raise ValueError(f"a request holds at least one item, got batch size {batch_size}")
    return batch_size if backend.batch_invariant else 1


def request_components(backend: BackendSpec, request_size: int) -> RequestComponents:
    """Return the batch formation and chunk grid of a request of `request_size` items sent to `backend`.

    One item is `single`, `n` items `items:<n>`; the chunk grid is the backend's.

    Raises:
        ValueError: `request_size` is below 1, or above 1 for a backend that is not batch-invariant.
    """
    if request_size < 1:
        raise ValueError(f"a request holds at least one item, got {request_size}")
    if request_size > 1 and not backend.batch_invariant:
        raise ValueError(f"backend {backend.name!r} is not batch-invariant and is sent one item per request")
    formation = SINGLE_ITEM if request_size == 1 else f"items:{request_size}"
    return {"batch_formation": formation, "chunk_grid": backend.chunk_grid}


@dataclass(frozen=True)
class Placement:
    """Where a call runs: the backend and its params, the device, the resource class, and how batches are formed.

    `batch_formation` and `chunk_grid` describe the placement whether or not the backend declares them
    output-affecting. `env_root` is the directory the backend's env is built under (None for a host env),
    `resource_class` the class of worker it runs on, `batch_size` the items per request, `partition` the scheduler
    partition as the profile names it, and `why` the reason `explain` gives; none of these reaches a key.
    """

    backend: BackendSpec
    backend_params: BackendParams
    device: str
    batch_formation: str | None = None
    chunk_grid: str | None = None
    env_root: str | None = None
    why: str = ""
    resource_class: str = "cpu"
    batch_size: int = 1
    partition: str | None = None

    def env(self) -> str:
        """Return the backend's env as `host` or `<kind>:<name>`."""
        env = self.backend.env
        return env.kind if env.name is None else f"{env.kind}:{env.name}"

    def record(self) -> NodePlacement:
        """Return this placement as the `NodePlacement` a plan's placement document holds."""
        return NodePlacement(
            backend=self.backend.name,
            backend_version=self.backend.version,
            backend_params=self.backend_params.model_dump(mode="json"),
            stack=self.backend.stack,
            env=self.env(),
            device=self.device,
            batch_size=self.batch_size,
            resource_class=self.resource_class,
            partition=self.partition,
            env_root=self.env_root,
            why=self.why,
        )

    def explain(self) -> str:
        """Return the backend, stack, env, device, batch, resource class, partition and the reason, as one line."""
        return (
            f"{self.backend.task}: backend {self.backend.name!r} (stack {self.backend.stack!r}, env {self.env()}, "
            f"device {self.device}, batch {self.batch_size}, resource class {self.resource_class}, "
            f"partition {self.partition or 'none'}); {self.why}"
        )


def build_key(call: Call, placement: Placement, context: ExecutionContext) -> LogicalKey:
    """Return the logical key of running `call` under `placement` (`call_key`); `context` never reaches it.

    Raises:
        TypeError: if `context` is not an `ExecutionContext`.
        ValueError, UnpinnedModel, pydantic.ValidationError: as `key_for`.
    """
    if not isinstance(context, ExecutionContext):
        raise TypeError(f"context must be an ExecutionContext, got {type(context).__name__}")
    return call_key(call, placement)


def call_key(call: Call, placement: Placement) -> LogicalKey:
    """Return the logical key of running `call` under `placement`.

    The key holds the backend's name and version, its params, and the batch formation and chunk grid
    the backend declares output-affecting; no other placement field: never the device, env root, reason, resource
    class, batch size or partition.

    Raises:
        ValueError, UnpinnedModel, pydantic.ValidationError: as `key_for`.
    """
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
