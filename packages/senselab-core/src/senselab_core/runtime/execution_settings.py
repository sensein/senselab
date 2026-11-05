"""The sections of an execution profile that size and pace a run: classes, substrate, pool, leases, faults, timeouts.

Each is a `Frozen` value the profile holds; `ExecutionProfile` checks that each resource class's name matches its
device.
"""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from senselab_core.kernel._base import Frozen, NonEmptyStr

PositiveInt = Annotated[int, Field(ge=1)]
PositiveSeconds = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class ResourceClass(Frozen):
    """A class of worker grants: the device and, for a GPU, its type; what each worker is granted; the caps.

    Attributes:
        device: `cpu`, `cuda` or `mps`.
        gpu_type: The GPU type a `cuda` class is granted, such as `h100`; None for any GPU and for other devices.
        cpus: CPUs per worker; None for the host's.
        mem_bytes: Memory per worker; None for the host's.
        time_s: Wall-time limit per worker; None for none.
        partition: The scheduler partition, `${VAR}` expanded where a worker is submitted; None off a scheduler.
        gres: The scheduler's generic-resource request, such as `gpu:h100:1`; None for none.
        max_workers: Workers of the class alive at once.
        memory_cap_bytes: The static memory the models resident at once on one worker may hold; None for no cap.
    """

    device: Literal["cpu", "cuda", "mps"]
    gpu_type: NonEmptyStr | None = None
    cpus: PositiveInt | None = None
    mem_bytes: PositiveInt | None = None
    time_s: PositiveSeconds | None = None
    partition: NonEmptyStr | None = None
    gres: NonEmptyStr | None = None
    max_workers: PositiveInt
    memory_cap_bytes: PositiveInt | None = None

    @model_validator(mode="after")
    def _check_gpu(self) -> "ResourceClass":
        if self.gpu_type is not None and self.device != "cuda":
            raise ValueError(f"a {self.device} class has no gpu_type")
        return self

    def matches(self, name: str) -> bool:
        """Whether `name` fits this device: `cpu[:v]`, `mps[:v]`, `gpu` or `gpu:<gpu_type>[:v]`, or `svc:<id>`."""
        kind, _, rest = name.partition(":")
        if kind == "svc":
            return True
        if kind == "gpu":
            if self.device != "cuda":
                return False
            return rest == "" if self.gpu_type is None else rest.split(":")[0] == self.gpu_type
        return {"cpu": "cpu", "mps": "mps"}.get(kind) == self.device


class SubstrateSpec(Frozen):
    """What starts pool workers: `local` processes, or `slurm` jobs within a QOS CPU cap, never on excluded nodes."""

    kind: Literal["local", "slurm"]
    qos_cpu_cap: PositiveInt | None = None
    exclude_nodes: tuple[NonEmptyStr, ...] = ()

    @model_validator(mode="after")
    def _check_local(self) -> "SubstrateSpec":
        if self.kind == "local" and (self.qos_cpu_cap is not None or self.exclude_nodes):
            raise ValueError("a local substrate has no QOS cap and no excluded nodes")
        return self


class PoolSettings(Frozen):
    """How pool workers claim and yield: the quantum, the empty-queue poll, the lease renewal and the index segment."""

    quantum_items: PositiveInt
    quantum_s: PositiveSeconds
    poll_s: PositiveSeconds
    lease_renew_fraction: Annotated[float, Field(gt=0, lt=1)]
    segment_items: PositiveInt


class JobLease(Frozen):
    """How a running job's lease is kept: its stale bound in seconds, and the share of that between renewals."""

    stale_after_s: PositiveSeconds
    renew_fraction: Annotated[float, Field(gt=0, lt=1)]


class FaultBudgets(Frozen):
    """How many times an item is retried per fault class before it fails, or its env is reported broken."""

    content_attempts: PositiveInt
    infrastructure_replacements: Annotated[int, Field(ge=0)]
    resource_replacements: Annotated[int, Field(ge=0)]


class Timeouts(Frozen):
    """Seconds a worker may take to load its model and to answer one request; None for no bound."""

    load_s: PositiveSeconds | None = None
    request_s: PositiveSeconds | None = None


class ServiceSettings(Frozen):
    """How long a service stays up after its last consumer lease lapsed."""

    idle_exit_s: PositiveSeconds
