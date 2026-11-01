"""The one key builder: a call and its placement to the call's logical key (ADR-002, ADR-004)."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from senselab_core.kernel.declare import BackendParams, BackendSpec, TaskParams, TaskSpec, key_for
from senselab_core.kernel.identity import (
    ExecutionContext,
    InputIdentity,
    LogicalKey,
    ProfileSliceIdentity,
    ResolvedModel,
)


@dataclass(frozen=True)
class Call:
    """A task call: the task, its params, input identities, resolved model and profile slices."""

    task: TaskSpec
    params: TaskParams
    inputs: Mapping[str, InputIdentity]
    model: ResolvedModel | None = None
    profile_slices: Mapping[str, ProfileSliceIdentity] = field(default_factory=dict)


@dataclass(frozen=True)
class Placement:
    """Where a call runs: the backend and its params, the device, and how batches and chunks are formed.

    `batch_formation` and `chunk_grid` describe the placement whether or not the backend declares them
    output-affecting.
    """

    backend: BackendSpec
    backend_params: BackendParams
    device: str
    batch_formation: str | None = None
    chunk_grid: str | None = None


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
