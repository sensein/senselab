"""The sections of an execution profile that size and pace a run.

Classes, substrate, pool, calibration, leases, faults and timeouts. Each is a `Frozen` value the profile holds;
`ExecutionProfile` checks that each resource class's name matches its device.
"""

import math
from typing import Annotated, Literal

from pydantic import Field, model_validator

from senselab_core.kernel._base import Frozen, NonEmptyStr

PositiveInt = Annotated[int, Field(ge=1)]
PositiveSeconds = Annotated[float, Field(gt=0, allow_inf_nan=False)]
PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]


SERVICE_CLASS = "svc"
"""The kind of a resource class whose workers hold services: `svc:<id>`."""


def is_service_class(name: str) -> bool:
    """Whether the resource class or queue `name` is a service class, `svc:<id>[#<model identity>]`."""
    return name.partition(":")[0] == SERVICE_CLASS


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
        if kind == SERVICE_CLASS:
            return True
        if kind == "gpu":
            if self.device != "cuda":
                return False
            return rest == "" if self.gpu_type is None else rest.split(":")[0] == self.gpu_type
        return {"cpu": "cpu", "mps": "mps"}.get(kind) == self.device


class SlurmCommands(Frozen):
    """How each Slurm command is run: its timeout, its tries, and the first wait between tries, which doubles."""

    timeout_s: PositiveSeconds
    attempts: PositiveInt
    retry_first_s: PositiveSeconds


class SubstrateSpec(Frozen):
    """What starts pool workers: `local` processes, or `slurm` jobs within a QOS CPU cap, never on excluded nodes.

    Attributes:
        start_backoff_first_s: Seconds a class, or the substrate itself, waits after its first failure in a row.
        start_backoff_max_s: Most seconds that wait doubles to.
        commands: How each Slurm command is run; required for `slurm`.
    """

    kind: Literal["local", "slurm"]
    qos_cpu_cap: PositiveInt | None = None
    exclude_nodes: tuple[NonEmptyStr, ...] = ()
    start_backoff_first_s: PositiveSeconds
    start_backoff_max_s: PositiveSeconds
    commands: SlurmCommands | None = None

    @model_validator(mode="after")
    def _check_kind(self) -> "SubstrateSpec":
        if self.kind == "local" and (self.qos_cpu_cap is not None or self.exclude_nodes):
            raise ValueError("a local substrate has no QOS cap and no excluded nodes")
        if self.kind == "slurm" and self.commands is None:
            raise ValueError("a slurm substrate needs its commands settings")
        return self


class PoolSettings(Frozen):
    """How pool workers claim and yield, and how a coordinator polls: the quantum ceilings, the polls and the segment.

    `quantum_items` and `quantum_s` bound the quantum `size` derives; a worker stays on a class for at most that many
    jobs or seconds. `poll_s` is a worker's empty-queue poll and the longest wait between a coordinator's reads of an
    outcome, which start at `first_poll_s` and double.
    """

    quantum_items: PositiveInt
    quantum_s: PositiveSeconds
    poll_s: PositiveSeconds
    first_poll_s: PositiveSeconds
    segment_items: PositiveInt


class SizingSettings(Frozen):
    """The fractions and targets `size` derives a unit's batch, quantum and worker count from.

    Attributes:
        load_share: The largest share of a quantum's time its model load may take.
        memory_headroom: The share of a worker's memory a batch may fill.
        target_wall_s: The wall time the workers of one queue aim to finish its waiting work in.
        calibration_items: The one-job items a unit without a measure runs first; the first carries the model load
            and the rest give the median per item.
        calibration_workers: The workers a queue whose units are all unmeasured is given for their rounds.
        device_query_s: Seconds a pool worker's per-claim read of its processes' device memory may take.
    """

    load_share: Annotated[float, Field(gt=0, lt=1)]
    memory_headroom: Annotated[float, Field(gt=0, le=1)]
    target_wall_s: PositiveSeconds
    calibration_items: Annotated[int, Field(ge=2)]
    calibration_workers: PositiveInt
    device_query_s: PositiveSeconds


Trust = Literal["group", "own"]
"""Whose measures a reader trusts: every file of every root (`group`), or only its own (`own`)."""


class CalibrationRuleError(ValueError):
    """A `calibration` value that is not what the rule deriving it from other keys of the profile gives."""


class CalibrationSettings(Frozen):
    """How a unit without a measure is probed, where measures are kept, and how a measure is judged stale.

    Attributes:
        measures_roots: Roots read in order and pooled; writes go to the first writable one.
        trust: `group` reads every root's files, `own` only the reader's own.
        warmup_requests: Requests a model state answers before one counts as warm.
        length_strata: Strata of equal count a probe cuts its candidate jobs into.
        min_requests: Warm requests a stratum needs for a quantity to be measured.
        single_item_jobs: Jobs of the one item per probe that separates the per-item cost from the per-job cost.
        batched_sizes: Batch sizes the batched phase measures.
        probe_items: Jobs a probe plans at most.
        probe_s: Seconds of warm probe requests a probe may take.
        confidence: The one-sided z of the batch decision and of the item cut.
        drift_window: Production requests of a regime judged for drift together.
        drift_confidence: The two-sided z of the drift test.
        peer_wait_s: Seconds a coordinator waits for a peer's probe of the same unit.
        observations_kept: Superseded probes' observation files kept for audit.
        unmeasured_workers: Workers a unit without a measure runs on.
    """

    measures_roots: Annotated[tuple[NonEmptyStr, ...], Field(min_length=1)]
    trust: Trust
    warmup_requests: Annotated[int, Field(ge=0)]
    length_strata: PositiveInt
    min_requests: PositiveInt
    single_item_jobs: Annotated[int, Field(ge=2)]
    batched_sizes: Annotated[int, Field(ge=1, le=2)]
    probe_items: PositiveInt
    probe_s: PositiveSeconds
    confidence: PositiveFloat
    drift_window: PositiveInt
    drift_confidence: PositiveFloat
    peer_wait_s: PositiveSeconds
    observations_kept: PositiveInt
    unmeasured_workers: PositiveInt

    def check_rules(self, sizing: SizingSettings, lease: "JobLease") -> None:
        """Raise `CalibrationRuleError` unless each key a rule derives equals what the rule gives.

        `probe_s` is `sizing.load_share × sizing.target_wall_s`; `peer_wait_s` is `probe_s + lease.stale_after_s`;
        `drift_window` is `length_strata × min_requests`; `probe_items` is `warmup_requests + length_strata ×
        min_requests × (1 + B) + min_requests × single_item_jobs` for a whole target batch `B` of at least 1.
        """
        if not math.isclose(self.probe_s, sizing.load_share * sizing.target_wall_s):
            raise CalibrationRuleError(
                f"calibration.probe_s {self.probe_s} is not sizing.load_share {sizing.load_share} × "
                f"sizing.target_wall_s {sizing.target_wall_s}"
            )
        if not math.isclose(self.peer_wait_s, self.probe_s + lease.stale_after_s):
            raise CalibrationRuleError(
                f"calibration.peer_wait_s {self.peer_wait_s} is not calibration.probe_s {self.probe_s} + "
                f"lease.stale_after_s {lease.stale_after_s}"
            )
        if self.drift_window != self.length_strata * self.min_requests:
            raise CalibrationRuleError(
                f"calibration.drift_window {self.drift_window} is not calibration.length_strata "
                f"{self.length_strata} × calibration.min_requests {self.min_requests}"
            )
        per_batch = self.length_strata * self.min_requests
        spare = self.probe_items - self.warmup_requests - self.min_requests * self.single_item_jobs
        if spare % per_batch or spare // per_batch - 1 < 1:
            raise CalibrationRuleError(
                f"calibration.probe_items {self.probe_items} is not warmup_requests {self.warmup_requests} + "
                f"{per_batch} × (1 + B) + min_requests × single_item_jobs {self.min_requests * self.single_item_jobs} "
                "for a whole B of at least 1"
            )


class LeaseBoundError(ValueError):
    """Two renewal intervals plus the storage's attribute delay do not stay below a lease's stale bound."""


class JobLease(Frozen):
    """How every lease is kept (`Lease.keep_alive`).

    Attributes:
        stale_after_s: Seconds on the storage clock after which another process breaks the lease.
        renew_fraction: The share of `stale_after_s` between two beats.
        attribute_delay_s: Seconds a client of the storage may show a directory's stale mtime.
    """

    stale_after_s: PositiveSeconds
    renew_fraction: Annotated[float, Field(gt=0, lt=1)]
    attribute_delay_s: Annotated[float, Field(ge=0, allow_inf_nan=False)]

    def check_bound(self) -> None:
        """Raise `LeaseBoundError` unless `2 × renew_fraction × stale_after_s + attribute_delay_s < stale_after_s`.

        The factor 2 allows one missed beat.
        """
        if 2 * self.renew_fraction * self.stale_after_s + self.attribute_delay_s >= self.stale_after_s:
            raise LeaseBoundError(
                f"2 × lease.renew_fraction {self.renew_fraction} × lease.stale_after_s {self.stale_after_s} + "
                f"lease.attribute_delay_s {self.attribute_delay_s} is not below lease.stale_after_s "
                f"{self.stale_after_s}: a beating lease that missed one beat could be seen as stale and broken"
            )


class FaultBudgets(Frozen):
    """How many times an item is retried per fault class before it fails, or its env is reported broken."""

    content_attempts: PositiveInt
    infrastructure_replacements: Annotated[int, Field(ge=0)]
    resource_replacements: Annotated[int, Field(ge=0)]


class Timeouts(Frozen):
    """Seconds a worker may take to load its model and to answer one request; None for no bound."""

    load_s: PositiveSeconds | None = None
    request_s: PositiveSeconds | None = None


class SupervisorSettings(Frozen):
    """How a supervisor reads a launched process's start time.

    Attributes:
        start_time_probe_s: Seconds `ps -o lstart=` may take off Linux; past it the start time is unknown.
    """

    start_time_probe_s: PositiveSeconds


class ServiceSettings(Frozen):
    """How long a service stays up after its last consumer lease lapsed."""

    idle_exit_s: PositiveSeconds
