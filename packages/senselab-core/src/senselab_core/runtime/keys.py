"""The one key builder: a call and its placement to the call's logical key (ADR-002, ADR-004)."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, TypedDict

from senselab_core.kernel.declare import BackendParams, BackendSpec, TaskParams, TaskSpec, key_for
from senselab_core.kernel.identity import (
    Attestation,
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
    """A task call: the task, its params, input identities, resolved model and profile slices.

    `service` is the attestation of the service answering a server backend's call, None for any other backend.
    """

    task: TaskSpec
    params: TaskParams
    inputs: Mapping[str, InputIdentity]
    model: ResolvedModel | None = None
    profile_slices: Mapping[str, ProfileSliceIdentity] = field(default_factory=dict)
    service: Attestation | None = None


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


SINGLE = "single"
"""The batch formation of a request to a backend that is not batch-invariant and keys no bucket: one item."""

MIXED = "mixed"
"""The batch formation of a request to a batch-invariant backend: any items, never keyed."""

BUCKET = "bucket"
"""The prefix of a bucketed batch formation, `bucket:<grid id>:<index>`."""


class FormationUnknown(ValueError):
    """A job's key was asked for without its batch formation, which its inputs' length picks."""


class RequestComponents(TypedDict):
    """The batch formation and chunk grid of a request sent to a backend."""

    batch_formation: str | None
    chunk_grid: str | None


def keys_bucket(backend: BackendSpec) -> bool:
    """Whether `backend` is not batch-invariant and declares its batch formation output-affecting."""
    return not backend.batch_invariant and "batch_formation" in backend.output_affecting


def fixed_formation(backend: BackendSpec) -> str | None:
    """Return the formation of every request to `backend`: `mixed`, `single`, or None when each job's length picks."""
    if backend.batch_invariant:
        return MIXED
    return None if keys_bucket(backend) else SINGLE


def request_size(backend: BackendSpec, batch_size: int) -> int:
    """Return the most items per request to `backend` under a placement of `batch_size`: 1 for formation `single`.

    Raises:
        ValueError: `batch_size` is below 1.
    """
    if batch_size < 1:
        raise ValueError(f"a request holds at least one item, got batch size {batch_size}")
    return 1 if fixed_formation(backend) == SINGLE else batch_size


def request_components(backend: BackendSpec, formation: str | None = None) -> RequestComponents:
    """Return the batch formation and chunk grid of a request to `backend` of a batch of `formation` (`form_batches`).

    `formation` omitted is the backend's `fixed_formation`. The chunk grid is the backend's.

    Raises:
        ValueError: `formation` is not one `backend` is sent in: `mixed` exactly when it is batch-invariant, a bucket
            only when it keys one.
    """
    if formation is None:
        formation = fixed_formation(backend)
    elif backend.batch_invariant != (formation == MIXED) or (
        formation.startswith(f"{BUCKET}:") and not keys_bucket(backend)
    ):
        raise ValueError(f"backend {backend.name!r} is not sent batches of formation {formation!r}")
    return {"batch_formation": formation, "chunk_grid": backend.chunk_grid}


@dataclass(frozen=True)
class Placement:
    """Where a call runs: the backend and its params, the device, the resource class, and how batches are formed.

    `batch_formation` and `chunk_grid` describe the placement whether or not the backend declares them
    output-affecting; `batch_formation` is None while each job's length picks it. `resource_class` is the class of
    worker it runs on, `batch_size` the most items per request, `partition` the scheduler partition as the profile
    names it, and `why` the reason `explain` gives; none of these reaches a key.
    """

    backend: BackendSpec
    backend_params: BackendParams
    device: str
    batch_formation: str | None = None
    chunk_grid: str | None = None
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
        service=call.service,
    )
