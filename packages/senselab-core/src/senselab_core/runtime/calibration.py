"""The data of execution calibration: what a worker reports per request, and what is estimated from it.

`Observation` is one request as its worker measured it. `Estimate` is what `Probe.estimate` derives from the warm
observations of one unit. `HardwareClass` is the normalised report of the hardware an observation was made on.
`ProbePhase` names the three kinds of probe item, and `LengthProfile` the input units of the jobs waiting to be sized.
`RunCalibration` is the run config's `calibration` section.
"""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from senselab_core.kernel._base import Frozen, NonEmptyStr
from senselab_core.runtime.execution_settings import CalibrationSettings, Trust

Seconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Units = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Bytes = Annotated[int, Field(ge=0)]
Count = Annotated[int, Field(ge=0)]
Positive = Annotated[int, Field(ge=1)]

ProbePhase = Literal["WARMUP", "SINGLE", "BATCHED"]
"""The intent of a probe item: `WARMUP` loads and warms, `SINGLE` fits one job per request, `BATCHED` fits batches."""
Outcome = Literal["ok", "failed", "bisected", "served"]
UnitKind = Literal["audio_s", "chars", "none"]


class RunCalibration(Frozen):
    """The run config's `calibration` section: what a run overrides of its profile's.

    Attributes:
        trust: `group` or `own`, replacing the profile's `calibration.trust`; None keeps the profile's.
    """

    trust: Trust | None = None

    def apply(self, settings: CalibrationSettings) -> CalibrationSettings:
        """Return `settings` with this run's overrides."""
        return settings if self.trust is None else settings.model_copy(update={"trust": self.trust})


class HardwareClass(Frozen):
    """The normalised report of the hardware a worker runs on; the only form measures are looked up and written by.

    Attributes:
        device: `cuda`, `mps` or `cpu`.
        name: The driver's product name (`cuda`), the chip string (`mps`), or the CPU model (`cpu`).
        memory_mib: `memory.total` in MiB (`cuda`), the unified memory in MiB (`mps`); None for `cpu`.
        compute_capability: `<major>.<minor>` for `cuda` where the driver reports it, else None.
        cpus: The CPUs the worker is granted.
    """

    device: Literal["cuda", "mps", "cpu"]
    name: NonEmptyStr
    memory_mib: Positive | None = None
    compute_capability: NonEmptyStr | None = None
    cpus: Positive

    @model_validator(mode="after")
    def _check_device(self) -> Self:
        if self.device != "cpu" and self.memory_mib is None:
            raise ValueError(f"a {self.device} class carries its memory")
        if self.device == "cpu" and self.memory_mib is not None:
            raise ValueError("a cpu class carries no device memory")
        if self.device != "cuda" and self.compute_capability is not None:
            raise ValueError(f"a {self.device} class has no compute capability")
        return self


class ProbeRef(Frozen):
    """The probe item a request belongs to: the probe's id, its phase and its length stratum."""

    id: NonEmptyStr
    phase: ProbePhase
    stratum: Count


class Cotenancy(Frozen):
    """Compute processes on the request's device and live pool workers on its host, the request itself counted."""

    device: Positive
    host: Positive


class Observation(Frozen):
    """What a worker reports of one request.

    Attributes:
        probe: The probe item the request belongs to; None for a production request.
        item: The item's id.
        request_index: The request's index within its item.
        jobs: Jobs in the request.
        batch: The request's batch size.
        units: The sum of its jobs' input units.
        padded_units: `batch` times the longest job's units.
        wall_s: Request start to end inside the worker.
        item_wall_s: On an item's last request, claim taken to commit; else None.
        since_load: Requests the model state answered before this one.
        load_s: On a request that loaded the model, the load's seconds; else None.
        device_peak: The torch peak device allocation, else None.
        device_after: The device allocation after the request, else None.
        host_peak: The model process's peak host memory, else None.
        host_after: Its host memory after the request, else None.
        cotenancy: The co-tenancy the request ran under.
        outcome: `ok`, `failed`, `bisected` or `served`.
        hardware: The hardware as the worker reads it.
    """

    probe: ProbeRef | None
    item: NonEmptyStr
    request_index: Count
    jobs: Positive
    batch: Positive
    units: Units
    padded_units: Units
    wall_s: Seconds
    item_wall_s: Seconds | None
    since_load: Count
    load_s: Seconds | None
    device_peak: Bytes | None
    device_after: Bytes | None
    host_peak: Bytes | None
    host_after: Bytes | None
    cotenancy: Cotenancy
    outcome: Outcome
    hardware: HardwareClass

    @model_validator(mode="after")
    def _check_request(self) -> Self:
        if self.padded_units < self.units:
            raise ValueError(f"padded_units {self.padded_units} is below units {self.units}")
        if self.load_s is not None and self.since_load:
            raise ValueError(f"load_s is set on the request that loaded, not one with since_load {self.since_load}")
        for name, peak, after in (
            ("device", self.device_peak, self.device_after),
            ("host", self.host_peak, self.host_after),
        ):
            if peak is not None and after is not None and peak < after:
                raise ValueError(f"{name}_peak {peak} is below {name}_after {after}")
        return self


class RequestFit(Frozen):
    """A fit of per-request seconds `r + u·units` at one batch size, over the units in `[lo, hi]`.

    `u` and `spread` are None when unmeasured: fewer than two strata fit a slope, and a median has no spread there.
    """

    batch: Positive
    r: Seconds
    u: Annotated[float, Field(allow_inf_nan=False)] | None
    n: Positive
    spread: Seconds | None
    lo: Units
    hi: Units

    @model_validator(mode="after")
    def _check_range(self) -> Self:
        if self.hi < self.lo:
            raise ValueError(f"range hi {self.hi} is below lo {self.lo}")
        return self


class MemoryFit(Frozen):
    """Memory of one kind: the resident bytes, and the peak beyond them as `m0 + m1·padded_units`."""

    resident: Bytes
    m0: Annotated[float, Field(allow_inf_nan=False)]
    m1: Annotated[float, Field(allow_inf_nan=False)]
    n: Positive


class StratumBatch(Frozen):
    """The batch chosen at one length stratum; 1 when no measured batch is faster beyond the margin."""

    stratum: Count
    batch: Positive


class RegimeFactor(Frozen):
    """The factor `rho` that scales predictions under co-tenancy `k`, from `n` requests."""

    k: Positive
    rho: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    n: Positive


class OomRecord(Frozen):
    """A resource fault at batch `batch` and length `length`, under co-tenancy `cotenancy`."""

    resource: Literal["device", "host"]
    batch: Positive
    length: Units
    cotenancy: Positive


class EstimateSource(Frozen):
    """Where an estimate came from: probe ids and observation count, or a packaged row's commit and probe ids."""

    probe_ids: tuple[NonEmptyStr, ...] = ()
    observations: Count = 0
    commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")] | None = None


def _refuse_repeat(field: str, label: str, keys: list[int]) -> None:
    repeated = sorted({k for k in keys if keys.count(k) > 1})
    if repeated:
        raise ValueError(f"{field} holds {label} {repeated[0]} twice")


class Estimate(Frozen):
    """What `Probe.estimate` derives from the observations of one unit.

    Attributes:
        status: `measured` when the single fit's `r` is measured, else `unmeasured`.
        reasons: Why unmeasured, and what each None figure lacks.
        unit_kind: What the unit's input units are: recording seconds, characters, or none.
        load_s: The model load's seconds, None when unmeasured.
        load_n: Loads it is the median of.
        item_s: The per-item cost beyond its requests (claim, bind, commit), None when unmeasured.
        item_n: Items it is the median of.
        single: The fit at batch 1.
        batched: The fit at each measured batch above 1.
        batch: The chosen batch per length stratum.
        device: Device memory; None when unmeasured, never zero.
        host: Host memory; None when unmeasured.
        regimes: The factor per measured co-tenancy.
        oom: The resource faults that raise the fit.
        source: Where it came from.
        estimator: The estimator's format tag.
    """

    status: Literal["measured", "unmeasured"]
    reasons: tuple[NonEmptyStr, ...] = ()
    unit_kind: UnitKind
    load_s: Seconds | None = None
    load_n: Count = 0
    item_s: Seconds | None = None
    item_n: Count = 0
    single: RequestFit | None = None
    batched: tuple[RequestFit, ...] = ()
    batch: tuple[StratumBatch, ...] = ()
    device: MemoryFit | None = None
    host: MemoryFit | None = None
    regimes: tuple[RegimeFactor, ...] = ()
    oom: tuple[OomRecord, ...] = ()
    source: EstimateSource = EstimateSource()
    estimator: Literal["estimate@1"] = "estimate@1"

    @model_validator(mode="after")
    def _check_status(self) -> Self:
        if self.status == "measured" and self.single is None:
            raise ValueError("a measured estimate carries its single fit")
        if self.status == "unmeasured" and not self.reasons:
            raise ValueError("an unmeasured estimate names its reasons")
        if self.single is not None and self.single.batch != 1:
            raise ValueError("the single fit is at batch 1")
        if any(f.batch == 1 for f in self.batched):
            raise ValueError("a batched fit is at a batch above 1")
        _refuse_repeat("batched", "batch", [f.batch for f in self.batched])
        _refuse_repeat("batch", "stratum", [b.stratum for b in self.batch])
        _refuse_repeat("regimes", "k", [r.k for r in self.regimes])
        return self


class LengthProfile(Frozen):
    """The input units of each waiting job in put order; None for a job whose length is not yet known."""

    units: tuple[Units | None, ...]
