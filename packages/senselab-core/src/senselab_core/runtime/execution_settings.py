"""The sections of an execution profile that size and pace a run.

Classes, substrate, pool, calibration, leases, faults and timeouts. Each is a `Frozen` value the profile holds;
`ExecutionProfile` checks that each resource class's name matches its device.
"""

from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

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
        placement_hook: The `module:callable` of the site's placement hook; None for none.
        placement_hook_s: Seconds the hook may take per call, past which it has failed.
        stop_grace_s: Seconds a local worker is given after SIGTERM before it is killed.
        commands: How each Slurm command is run; required for `slurm`.
    """

    kind: Literal["local", "slurm"]
    qos_cpu_cap: PositiveInt | None = None
    exclude_nodes: tuple[NonEmptyStr, ...] = ()
    start_backoff_first_s: PositiveSeconds
    start_backoff_max_s: PositiveSeconds
    placement_hook: Annotated[str, StringConstraints(pattern=r"^[\w.]+:[\w.]+$")] | None = None
    placement_hook_s: PositiveSeconds
    stop_grace_s: PositiveSeconds
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
    outcome, which start at `first_poll_s` and double. `presence_refresh_s` is the longest a worker reuses what it read
    of the workers on its host, the processes on its device and the probe queues of its run. `admit_threads` bounds the
    threads that ask the result store which jobs it would serve, for a plan's counts. A worker exits once its queues
    stayed empty for `idle_polls` polls of `poll_s`; a following coordinator checks its cancel event every
    `cancel_check_s`; `Run.wait` reads the event log of a run another process drives every `event_poll_s`.
    """

    quantum_items: PositiveInt
    quantum_s: PositiveSeconds
    poll_s: PositiveSeconds
    first_poll_s: PositiveSeconds
    presence_refresh_s: PositiveSeconds
    segment_items: PositiveInt
    admit_threads: PositiveInt
    idle_polls: PositiveInt
    cancel_check_s: PositiveSeconds
    event_poll_s: PositiveSeconds


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


class CalibrationSettings(Frozen):
    """How a unit without a measure is probed, where measures are kept, and how a measure is judged stale.

    Attributes:
        measures_roots: Roots read in order and pooled; writes go to the first writable one.
        trust: `group` reads every root's files, `own` only the reader's own.
        warmup_requests: Requests a model state answers before one counts as warm (the floor is 1: the first
            request after a load is cold).
        length_strata: Strata of equal count a probe cuts its candidate jobs into.
        min_requests: Warm requests a stratum needs for a quantity to be measured.
        single_item_jobs: Jobs of the one item per probe that separates the per-item cost from the per-job cost.
        batched_sizes: Batch sizes the batched phase measures.
        confidence: The one-sided z of the batch choice and of the item cut.
        drift_confidence: The two-sided z of the drift test.
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
    confidence: PositiveFloat
    drift_confidence: PositiveFloat
    observations_kept: PositiveInt
    unmeasured_workers: PositiveInt

    @model_validator(mode="after")
    def _check_roots(self) -> "CalibrationSettings":
        if blank := [r for r in self.measures_roots if not r.strip()]:
            raise ValueError(f"measures_roots holds a blank root {blank[0]!r}")
        if relative := [r for r in self.measures_roots if r.strip() and r.strip()[0] not in "/~$"]:
            raise ValueError(f"measures_roots holds {relative[0]!r}, which is not an absolute path")
        if repeated := sorted({r for r in self.measures_roots if self.measures_roots.count(r) > 1}):
            raise ValueError(f"measures_roots holds {repeated[0]!r} twice")
        return self

    @property
    def drift_window(self) -> int:
        """Production requests of a regime judged for drift together: `length_strata × min_requests`."""
        return self.length_strata * self.min_requests


class LeaseBoundError(ValueError):
    """`missed_beats` renewal intervals plus the storage's attribute delay do not stay below a lease's stale bound."""


class JobLease(Frozen):
    """How every lease is kept and taken (`Lease`, `Lease.keep_alive`).

    Attributes:
        stale_after_s: Seconds on the storage clock after which another process breaks the lease.
        renew_fraction: The share of `stale_after_s` between two beats.
        attribute_delay_s: Seconds a client of the storage may show a directory's stale mtime.
        missed_beats: Renewal intervals a beating lease's mtime may be seen unrefreshed without being stale; 2 allows
            one missed beat.
        wait_s: Seconds a wait for a lease may take by default.
        poll_s: Seconds between two attempts to take a lease.
        remove_attempts: Times a lease moved aside is removed while a taker that reached it before the move races
            the removal.
        record_attempts: Times a holder displaced before its write takes the lease again and merges, for a run's
            `inputs.json` and a run manifest.
        dir_mode: The octal mode of a lease directory and its token directory, such as `2770`.
    """

    stale_after_s: PositiveSeconds
    renew_fraction: Annotated[float, Field(gt=0, lt=1)]
    attribute_delay_s: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    missed_beats: PositiveInt
    wait_s: PositiveSeconds
    poll_s: PositiveSeconds
    remove_attempts: PositiveInt
    record_attempts: PositiveInt
    dir_mode: Annotated[str, StringConstraints(pattern=r"^[0-7]{3,4}$")]

    @property
    def mode(self) -> int:
        """`dir_mode` as the integer mode bits."""
        return int(self.dir_mode, 8)

    def check_bound(self) -> None:
        """Raise `LeaseBoundError` unless the bound holds.

        The bound is `missed_beats × renew_fraction × stale_after_s + attribute_delay_s < stale_after_s`.
        """
        if self.missed_beats * self.renew_fraction * self.stale_after_s + self.attribute_delay_s >= self.stale_after_s:
            raise LeaseBoundError(
                f"{self.missed_beats} × lease.renew_fraction {self.renew_fraction} × lease.stale_after_s "
                f"{self.stale_after_s} + lease.attribute_delay_s {self.attribute_delay_s} is not below "
                f"lease.stale_after_s {self.stale_after_s}: a beating lease that missed "
                f"{self.missed_beats - 1} beat(s) could be seen as stale and broken"
            )


class FaultBudgets(Frozen):
    """How many times an item is retried per fault class before it fails, or its env is reported broken.

    `crashes_charged` is the number of fresh workers in a row a job alone must be lost on to be charged to the item.
    """

    content_attempts: PositiveInt
    infrastructure_replacements: Annotated[int, Field(ge=0)]
    resource_replacements: Annotated[int, Field(ge=0)]
    crashes_charged: PositiveInt


class Timeouts(Frozen):
    """Seconds a worker may take to load its model and to answer one request (None for no bound), and to fetch.

    Attributes:
        load_s: A worker's model-load bound; None for none.
        request_s: A worker's per-request bound; None for none.
        fetch_s: Seconds one network request to fetch a file or read an index may take before it fails.
    """

    load_s: PositiveSeconds | None = None
    request_s: PositiveSeconds | None = None
    fetch_s: PositiveSeconds


class SupervisorSettings(Frozen):
    """How a supervisor reads a launched process's start time.

    Attributes:
        start_time_probe_s: Seconds `ps -o lstart=` may take off Linux; past it the start time is unknown.
        reap_poll_s: Seconds between the reaper's passes over the orphans.
        exit_bound_s: Seconds leaving a supervisor waits for its killed orphans, leaders and keeper to be reaped.
    """

    start_time_probe_s: PositiveSeconds
    reap_poll_s: PositiveSeconds
    exit_bound_s: PositiveSeconds


class ServiceSettings(Frozen):
    """How long a service stays up after its last consumer lease lapsed."""

    idle_exit_s: PositiveSeconds


class StoreSettings(Frozen):
    """How a result store commits and how a storage root's clock is probed.

    Attributes:
        commit_attempts: Times `put` stages and renames before giving up on a target that keeps turning out corrupt.
        probe_bound_s: Seconds the capability probe waits for a directory's mtime to show a change before it judges
            the clock.
        probe_poll_s: Seconds between the probe's reads of a directory's mtime.
        clock_tolerance_s: Seconds the storage clock may differ from the host's before the root is judged skewed.
    """

    commit_attempts: PositiveInt
    probe_bound_s: PositiveSeconds
    probe_poll_s: PositiveSeconds
    clock_tolerance_s: PositiveSeconds


class VllmSettings(Frozen):
    """How a `vllm serve` process is started, polled, asked and stopped.

    Attributes:
        startup_s: Seconds the engine has to become healthy once the start lease is held.
        poll_s: Seconds between reads of the server's health while it starts.
        request_s: Seconds one request to the server may take.
        stop_s: Seconds the server is given to exit after SIGTERM before it is killed.
        health_s: Seconds one read of the server's `/health` may take.
    """

    startup_s: PositiveSeconds
    poll_s: PositiveSeconds
    request_s: PositiveSeconds
    stop_s: PositiveSeconds
    health_s: PositiveSeconds


class WorkerSettings(Frozen):
    """How a worker client starts and watches its worker subprocess.

    Attributes:
        hello_s: Seconds a started worker has to send its hello line.
        leader_probe_s: Seconds a probe of whether a worker's process group leader is held may take.
        group_poll_max_s: The longest wait, in seconds, between two reads of a stopping process group.
    """

    hello_s: PositiveSeconds
    leader_probe_s: PositiveSeconds
    group_poll_max_s: PositiveSeconds
