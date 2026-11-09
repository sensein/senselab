"""Sizing: a unit's batch, quantum and worker count, derived from a measure of it and the grant it runs under.

`size` is pure. Its measure is a resource-table row's (`measure_of`), or one a calibration round wrote: one-job items
run first on one worker, the first carrying the model load (`calibrated`, from each item's `Sample`). `Measures` reads
the packaged tables, then the ones calibration rounds wrote under the cache.
"""

import functools
import importlib.metadata
import math
import statistics
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated

from pydantic import Field, model_validator
from senselab_worker.digest import file_digest, identity_digest

from senselab_core.kernel._base import Frozen, NonEmptyStr
from senselab_core.runtime.calibration import (
    Estimate,
    LengthProfile,
    MemoryFit,
    RegimeFactor,
    RequestFit,
    StratumBatch,
)
from senselab_core.runtime.execution_settings import CalibrationSettings, PoolSettings, ResourceClass, SizingSettings
from senselab_core.runtime.hardware import HardwareInfo
from senselab_core.runtime.probe import memory_batch, peak
from senselab_core.runtime.resources import (
    Batched,
    Measured,
    ResourceMeasure,
    ResourceTable,
    Unit,
    load_resource_tables,
    write_row,
)

Seconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Bytes = Annotated[int, Field(ge=0)]
Count = Annotated[int, Field(ge=1)]


class Measure(Frozen):
    """What one item costs on one resource class: seconds to load and to claim, memory resident and per item, seconds.

    `on_device` says the memory pair is the device's, sized against the class's device memory cap; the pair is None
    when unmeasured. `batched_s_per_item` is the seconds per item measured in batches of `batched_batch_size`, both
    None where unmeasured.
    """

    load_s: Seconds
    claim_s: Seconds
    resident_bytes: Bytes | None
    bytes_per_item: Bytes | None
    s_per_item: Seconds
    on_device: bool = False
    batched_s_per_item: Seconds | None = None
    batched_batch_size: Count | None = None

    @model_validator(mode="after")
    def _check_batched(self) -> "Measure":
        if (self.batched_s_per_item is None) != (self.batched_batch_size is None):
            raise ValueError("a batched measure names both its seconds per item and the batch it was measured at")
        return self


class Grant(Frozen):
    """What a unit's workers are granted: the most alive at once, memory per worker (None unknown), quantum ceilings.

    `devices` is the count of devices a `cuda` class's workers share; None where each worker is granted its own device
    or runs on none. `on_device` says `memory_bytes` is a device's, so sizing reads the device memory fit.
    """

    max_workers: Count
    memory_bytes: Count | None
    quantum_items: Count
    quantum_s: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    devices: Annotated[int, Field(ge=0)] | None = None
    on_device: bool = False

    @classmethod
    def of(  # noqa: PLR0913 - the class and pool, then what the host measured
        cls,
        resource: ResourceClass,
        pool: PoolSettings,
        host_memory_bytes: int | None = None,
        *,
        on_device: bool = False,
        devices: int | None = None,
        device_memory_bytes: int | None = None,
    ) -> "Grant":
        """Return the grant of a worker of `resource`: its device memory cap `on_device`, else its memory or the host's.

        A device grant of a class with no cap takes `device_memory_bytes`, the memory of the device seen.

        The quantum ceilings are the pool's, `quantum_s` within the class's wall-time limit. `devices`, the devices the
        class's workers share, is kept for a `cuda` class only.
        """
        memory = (
            resource.memory_cap_bytes or device_memory_bytes
            if on_device
            else resource.mem_bytes or host_memory_bytes or None
        )
        time_s = pool.quantum_s if resource.time_s is None else min(pool.quantum_s, resource.time_s)
        return cls(
            max_workers=resource.max_workers,
            memory_bytes=memory,
            quantum_items=pool.quantum_items,
            quantum_s=time_s,
            devices=devices if resource.device == "cuda" else None,
            on_device=on_device,
        )

    @classmethod
    def seen(
        cls, resource: ResourceClass, pool: PoolSettings, hardware: HardwareInfo | None, *, devices: int | None = None
    ) -> "Grant":
        """Return the grant of a worker of `resource` on `hardware` (None unknown), by its device's or host's memory."""
        cuda = resource.device == "cuda"
        host = hardware.memory_bytes if hardware is not None and not cuda else None
        seen = min((g.memory_bytes for g in hardware.gpus), default=None) if hardware is not None and cuda else None
        return cls.of(resource, pool, host, on_device=cuda, devices=devices, device_memory_bytes=seen)

    def worker_bound(self, measure: "Measure | None") -> int:
        """Return the most workers of a unit measured as `measure` (None unmeasured) this grant runs at once.

        On shared devices each holds `floor(device memory cap / measured device peak)` workers, at least one, and
        one when the device peak or the cap is unmeasured; the bound is that times `devices`, within `max_workers`.
        """
        if self.devices is None:
            return self.max_workers
        peak = None
        if measure is not None and measure.on_device and measure.resident_bytes is not None:
            peak = measure.resident_bytes + (measure.bytes_per_item or 0)
        each = 1 if not peak or self.memory_bytes is None else max(1, self.memory_bytes // peak)
        return min(self.max_workers, self.devices * each)


class RowSizing(Frozen):
    """Items per request, the most memory allows, jobs and seconds per quantum, and the workers a queue's work wants."""

    batch_size: Count
    memory_batch: Count
    quantum_items: Count
    quantum_s: Seconds
    workers: Annotated[int, Field(ge=0)]


def size_row(
    measure: Measure, grant: Grant, config: SizingSettings, *, work: int = 0, batchable: bool = True
) -> RowSizing:
    """Return the sizing of a unit measured as `measure` under `grant`, for `work` waiting jobs; the Pool's until 7a.

    - `quantum_items`: the fewest jobs whose time keeps the load and the claim within `load_share` of the quantum
      (the ceiling when both are 0), within the ceilings: `quantum_items`, and the jobs that fit in `quantum_s` after
      them;
    - `quantum_s`: the quantum's jobs plus the load and claim, within `quantum_s`;
    - `memory_batch`: the most items whose memory beside the resident memory fits `memory_headroom` of the grant's,
      at most a quantum; 1 when the grant's or the unit's memory is unmeasured or `batchable` is False;
    - `batch_size`: `memory_batch`, at most the batch measured, where items sent in that batch were measured no slower
      per item than items sent alone, else 1 (batched throughput unmeasured counts as not shown);
    - `workers`: as many as end `work` within `target_wall_s`, at most a quantum each and `Grant.worker_bound`; 0 for
      none.
    """
    per, load, ceiling = measure.s_per_item, measure.load_s + measure.claim_s, grant.quantum_items
    paid = config.load_share * per
    wanted = _held(load / paid, ceiling, math.ceil) if paid and load else ceiling
    fits = _held((grant.quantum_s - load) / per, ceiling, math.floor) if per else ceiling
    quantum = max(1, min(wanted, fits))
    memory_batch = 1
    if batchable and grant.memory_bytes is not None and measure.resident_bytes is not None:
        room = config.memory_headroom * grant.memory_bytes - measure.resident_bytes
        per_bytes = measure.bytes_per_item or 0
        memory_batch = (
            1 if room < 0 else quantum if per_bytes == 0 else max(1, _held(room / per_bytes, quantum, math.floor))
        )
    faster = measure.batched_s_per_item is not None and measure.batched_s_per_item <= per
    batch = min(memory_batch, measure.batched_batch_size or 1) if faster else 1
    workers = 0
    if work > 0:
        timely = max(1, _held(work * per / config.target_wall_s, grant.max_workers, math.ceil))
        workers = min(timely, math.ceil(work / quantum), grant.worker_bound(measure))
    return RowSizing(
        batch_size=batch,
        memory_batch=memory_batch,
        quantum_items=quantum,
        quantum_s=min(quantum * per + load, grant.quantum_s),
        workers=workers,
    )


def _held(x: float, bound: int, rounded: Callable[[float], int]) -> int:
    """Return `rounded(x)` held within `[0, bound]`, for an infinite `x` too."""
    return bound if x >= bound else rounded(max(x, 0.0))


class Basis(Frozen):
    """Which figures of a `Sizing` rest on a measure: `time` (per-job seconds), `memory`, `batch` and `workers`."""

    time: bool
    memory: bool
    batch: bool
    workers: bool


class Sizing(Frozen):
    """What `size` derives for a unit's waiting jobs: item cuts, batch per length, workers and predicted seconds.

    Attributes:
        cuts: Half-open job index ranges `[start, stop)` of the items, in put order, covering every job.
        strata: The estimate's batch per length range.
        measured_range: The `(lo, hi)` units the single fit measured; None when unmeasured.
        memory: The device or host memory fit the batches are sized against; None when unmeasured.
        budget_bytes: The memory a batch may fill, `memory_headroom` of one worker's share of the grant.
        ceiling: The most jobs of one item.
        workers: Workers the work wants.
        workers_per_device: Workers that share one card; 1 where each holds its own.
        predicted_s: Each job's median predicted seconds; None for every job unless all are of known length inside
            the measured range.
        wall_s: Predicted wall seconds of the work on `workers`; None unless every job's time is measured.
        batch_one_peak: The memory of a batch-1 request at the longest job (`peak`); None unmeasured.
        batch_peak: The memory of a request at the longest job and its batch (`peak`); None unmeasured.
        measured: Which figures rest on a measure.
    """

    cuts: tuple[tuple[int, int], ...]
    strata: tuple[StratumBatch, ...] = ()
    measured_range: tuple[float, float] | None = None
    memory: MemoryFit | None = None
    budget_bytes: float | None = None
    ceiling: Count = 1
    workers: Annotated[int, Field(ge=0)]
    workers_per_device: Count = 1
    predicted_s: tuple[Seconds | None, ...] = ()
    wall_s: Seconds | None = None
    batch_one_peak: Bytes | None = None
    batch_peak: Bytes | None = None
    measured: Basis

    def batch_at(self, length: float | None, jobs: int | None = None) -> int:
        """Return the batch for jobs of `length` units in an item of `jobs` jobs.

        It is the least of the estimate's batch for that length, the memory batch and `jobs`; 1 for an unknown length,
        one above the measured range, or unmeasured memory.
        """
        return _batch(self, length, self.ceiling if jobs is None else min(jobs, self.ceiling))


def _star(strata: Sequence[StratumBatch], length: float) -> int:
    """Return the estimate's batch at `length`: its stratum's, the smaller of the two beside a gap, else the nearest."""
    holding = [s.batch for s in strata if s.lo <= length <= s.hi]
    if holding:
        return min(holding)
    below = [s for s in strata if s.hi < length]
    above = [s for s in strata if s.lo > length]
    near = ([max(below, key=lambda s: s.hi)] if below else []) + ([min(above, key=lambda s: s.lo)] if above else [])
    return min((s.batch for s in near), default=1)


def _batch(sized: Sizing, length: float | None, jobs: int) -> int:
    span = sized.measured_range
    if length is None or span is None or sized.memory is None or sized.budget_bytes is None or length > span[1]:
        return 1
    room = memory_batch(sized.memory, sized.budget_bytes, length, max(jobs, 1))
    return max(1, min(_star(sized.strata, length), room, jobs))


def _regime_workers(regimes: Sequence[RegimeFactor], bound: int) -> int:
    """Return the most workers up to `bound` whose throughput `k/ρ_k` keeps increasing, where `ρ_k` is measured."""
    rho = {r.k: r.rho for r in regimes}
    best, throughput = 1, 1.0
    for k in range(2, bound + 1):
        if k in rho and k / rho[k] <= throughput:
            break
        throughput = k / rho.get(k, 1.0)
        best = k
    return best


def estimate_of(row: ResourceMeasure, *, on_device: bool = False) -> Estimate:
    """Return a resource-table row as an estimate over recording seconds, with no length range, spread or oom.

    Time is `s_per_input_s` per input second, memory the row's pair per input second of its median item, the batch the
    measured one where its per-second time is no slower than a single's. The spread is unmeasured, so items are cut by
    count.
    """
    scale = row.input_s_per_item or 1.0
    top = sys.float_info.max

    def fit(resident: int | None, peak_bytes: int | None) -> MemoryFit | None:
        if resident is None or peak_bytes is None:
            return None
        return MemoryFit(resident=resident, m0=0.0, m1=max(0, peak_bytes - resident) / scale, n=1)

    def request(batch: int, per_s: float) -> RequestFit:
        return RequestFit(batch=batch, r=0.0, u=per_s, n=1, spread=None, lo=0.0, hi=top)

    found = row.batched
    wide = found is not None and found.batch_size > 1
    chosen = found.batch_size if found is not None and wide and found.s_per_input_s <= row.s_per_input_s else 1
    return Estimate(
        status="measured",
        unit_kind="audio_s",
        load_s=row.load_s,
        load_n=1,
        item_s=row.claim_s,
        item_n=1,
        single=request(1, row.s_per_input_s),
        batched=(request(found.batch_size, found.s_per_input_s),) if found is not None and wide else (),
        batch=(StratumBatch(lo=0.0, hi=top, batch=chosen),),
        device=fit(row.device_resident_bytes, row.device_peak_bytes) if on_device else None,
        host=fit(row.resident_bytes, row.peak_memory_bytes) if not on_device else None,
    )


def _by_count(grant: Grant, jobs: int) -> tuple[tuple[int, int], ...]:
    return tuple((a, min(a + grant.quantum_items, jobs)) for a in range(0, jobs, grant.quantum_items))


def _unmeasured(grant: Grant, calibration: CalibrationSettings, jobs: int) -> Sizing:
    cuts = _by_count(grant, jobs)
    workers = 0 if jobs == 0 else min(calibration.unmeasured_workers, len(cuts), grant.max_workers)
    return Sizing(
        cuts=cuts,
        ceiling=grant.quantum_items,
        workers=workers,
        predicted_s=(None,) * jobs,
        measured=Basis(time=False, memory=False, batch=False, workers=False),
    )


def size(  # noqa: PLR0913 - the estimate, grant, config and work, then the calibration settings
    estimate: Estimate | None,
    grant: Grant,
    config: SizingSettings,
    work: LengthProfile,
    *,
    calibration: CalibrationSettings,
) -> Sizing:
    """Return the sizing of `work` under `grant` from `estimate` (None: unmeasured, so the safe values), purely.

    A job's seconds at batch 1 are `r + u·L`, at its length's batch B `(r_B + u_B·B·L)/B`, times the regime factor of
    the co-tenancy the card's workers make. Items are cut in put order at the upper prediction
    `t·(1 + confidence·spread)`: an item closes before its upper seconds plus the per-item cost pass the grant's
    `quantum_s`, when its seconds reach `(load_s + a)/load_share`, or at `quantum_items` jobs; with `u` or `spread`
    unmeasured they are cut by `quantum_items` alone. Workers are the median
    seconds over `target_wall_s`, at most the item count, `max_workers` and, on shared devices, `devices × w_d`.

    On shared devices `w_d = ⌊memory_headroom × cap / P⌋` for the batch peak `P` at the longest job, lowered where a
    measured regime says another worker adds no throughput, and each worker's memory is `cap / w_d`. The fit read is
    the device's on a device grant, the host's otherwise, never the other. A length above the measured range, or an
    unmeasured memory, gives batch 1; so does an unknown length. Lengths are `0` for an estimate of unit kind `none`.
    """
    jobs = len(work.units)
    single = None if estimate is None else estimate.single
    if estimate is None or single is None:
        return _unmeasured(grant, calibration, jobs)
    fit = estimate.device if grant.on_device else estimate.host
    span = (single.lo, single.hi)
    lengths = [single.hi if length is None else length for length in work.units]
    longest = max(lengths, default=0.0)
    share = grant.memory_bytes if fit is not None else None
    per_device = 1
    if share is not None and fit is not None and grant.devices is not None and grant.on_device:
        star = _star(estimate.batch, longest) if longest <= span[1] else 1
        batch_peak = peak(fit, star, longest)
        fits = grant.max_workers if batch_peak <= 0 else max(1, math.floor(config.memory_headroom * share / batch_peak))
        per_device = _regime_workers(estimate.regimes, fits)
        share //= per_device
    budget = None if share is None else config.memory_headroom * share
    shell = Sizing(
        cuts=(),
        strata=estimate.batch,
        measured_range=span,
        memory=fit,
        budget_bytes=budget,
        ceiling=grant.quantum_items,
        workers=0,
        measured=Basis(time=False, memory=False, batch=False, workers=False),
    )
    rho = {r.k: r.rho for r in estimate.regimes}.get(per_device, 1.0)
    medians = [_seconds(estimate, shell.batch_at(length), length) * rho for length in lengths]
    spread = 1.0 + calibration.confidence * (single.spread or 0.0)  # read only where the spread is measured
    per_item, load = estimate.item_s or 0.0, estimate.load_s or 0.0
    timed = all(length is not None and span[0] <= length <= span[1] for length in work.units)
    by_count = single.u is None or single.spread is None
    cuts = (
        _by_count(grant, jobs)
        if by_count
        else _cut(medians, [m * spread for m in medians], per_item, (load + per_item) / config.load_share, grant)
    )
    bound = grant.max_workers if grant.devices is None else grant.devices * per_device
    untimed = single.u is None and not timed
    wanted = calibration.unmeasured_workers if untimed else max(1, math.ceil(sum(medians) / config.target_wall_s))
    workers = 0 if jobs == 0 else min(wanted, len(cuts), grant.max_workers, bound)
    wall = load + (sum(medians) + per_item * len(cuts)) / workers if timed and workers else None
    return shell.model_copy(
        update={
            "cuts": cuts,
            "workers": workers,
            "workers_per_device": per_device,
            "predicted_s": tuple(m if timed else None for m in medians),
            "wall_s": wall,
            "batch_one_peak": None if fit is None else round(peak(fit, 1, longest)),
            "batch_peak": None if fit is None else round(peak(fit, shell.batch_at(longest), longest)),
            "measured": Basis(
                time=timed, memory=budget is not None, batch=bool(estimate.batch) and budget is not None, workers=timed
            ),
        }
    )


def _seconds(estimate: Estimate, batch: int, length: float) -> float:
    """Return the median seconds per job at `length` units sent `batch` to a request."""
    single = estimate.single
    assert single is not None
    fit = next((f for f in estimate.batched if f.batch == batch), None)
    if batch > 1 and fit is not None:
        return max(0.0, (fit.r + (fit.u or 0.0) * batch * length) / batch)
    return max(0.0, single.r + (single.u or 0.0) * length)


def _cut(
    medians: Sequence[float], uppers: Sequence[float], per_item: float, floor_s: float, grant: Grant
) -> tuple[tuple[int, int], ...]:
    """Return the item cuts: walk the jobs, closing an item at the grant's quantum or once it earns its load."""
    cuts: list[tuple[int, int]] = []
    start, upper, median = 0, 0.0, 0.0
    for i, (m, u) in enumerate(zip(medians, uppers, strict=True)):
        full = i - start >= grant.quantum_items or upper + u + per_item > grant.quantum_s or median >= floor_s > 0
        if i > start and full:
            cuts.append((start, i))
            start, upper, median = i, 0.0, 0.0
        upper, median = upper + u, median + m
    return (*cuts, (start, len(medians))) if medians else ()


def measure_of(row: ResourceMeasure, *, on_device: bool = False) -> Measure:
    """Return the per-item measure of a resource-table measure, at its median item's input seconds.

    `on_device` takes the device memory pair when the row measured one, else the host pair.
    """
    scale = 1.0 if row.input_s_per_item is None else row.input_s_per_item
    seconds = row.s_per_input_s * scale
    device = on_device and row.device_resident_bytes is not None and row.device_peak_bytes is not None
    resident, peak = (
        (row.device_resident_bytes, row.device_peak_bytes) if device else (row.resident_bytes, row.peak_memory_bytes)
    )
    return Measure(
        load_s=row.load_s,
        claim_s=row.claim_s,
        resident_bytes=resident,
        bytes_per_item=None if resident is None or peak is None else peak - resident,
        s_per_item=seconds,
        on_device=device,
        batched_s_per_item=None if row.batched is None else row.batched.s_per_input_s * scale,
        batched_batch_size=None if row.batched is None else row.batched.batch_size,
    )


class Sample(Frozen):
    """What a pool worker measured of one item it ran.

    Attributes:
        wall_s: The item's share, by jobs, of the seconds from its claim to its claim's outcomes.
        claim_s: Seconds the claim holding the item took to take.
        jobs: The item's jobs.
        input_s: Their recording seconds; None when a job has no recording input or its length is unrecorded.
        input_chars: The characters of their held text inputs; None when a job holds no text.
        baseline_bytes: The worker's host memory before it ran anything.
        resident_bytes: Its host memory once the item's outcomes were in, the model loaded and the item freed; None
            where only a high-water is readable.
        peak_memory_bytes: Its host high-water memory once the item ran.
        device_baseline_bytes: The device memory this worker's processes used before it ran anything; None off a
            device.
        device_bytes: The device memory they used once the item's claim had its outcomes, the model loaded and
            nothing in flight; None off a device.
        device_item_bytes: The most device memory a request of the item's allocated per item beyond what stayed
            allocated after it, as its backend reported it; None where none reported (a server, or no CUDA in use).
        device_allocated_bytes: The least device memory the backend held allocated after one of those requests.
        batch_size: The most items of the item's jobs one request held.
        served: Whether the item's class is a service class, whose server holds a fixed device reservation.
        cold: Whether the worker had run nothing of the item's queue before, so the item carried the model load.
        job: The scheduler job id or process id of the worker.
        ended_s: The worker's wall clock (`time.time`) when the item's sample was taken, its outcomes in;
            comparable across workers only on one host or synchronised clocks.
    """

    wall_s: Seconds
    claim_s: Seconds
    jobs: Count
    batch_size: Count
    input_s: Seconds | None
    input_chars: Annotated[int, Field(ge=0)] | None = None
    baseline_bytes: Bytes
    resident_bytes: Bytes | None
    peak_memory_bytes: Bytes
    device_baseline_bytes: Bytes | None = None
    device_bytes: Bytes | None = None
    device_item_bytes: Bytes | None = None
    device_allocated_bytes: Bytes | None = None
    served: bool = False
    cold: bool
    job: NonEmptyStr
    ended_s: Seconds


def calibrated(samples: Sequence[Sample]) -> ResourceMeasure | None:
    """Return the measure a calibration round's `samples` give; None for none.

    Seconds per item are the median over warm items sent one per request (over cold ones when none is warm), the load
    the median of what a cold item took beyond its jobs, the claim the median claim; items sent in batches give the
    per-item seconds at their largest batch (`batched`). The resident host memory is the lowest after a warm item
    (after any item when none is warm) and the peak the highest; with no resident figure the host pair is unmeasured.
    The device pair is, where the backend reported its requests, the least allocation after a request and that plus
    the most a request allocated per item; for a server (`served`) the lowest and the highest device use of the
    worker's processes read after items, its fixed reservation; else unmeasured.
    """
    if not samples:
        return None
    single = [s for s in samples if s.batch_size == 1] or list(samples)
    warm = [s.wall_s / s.jobs for s in single if not s.cold]
    cold = [s for s in samples if s.cold]
    per = statistics.median(warm or [s.wall_s / s.jobs for s in cold] or [s.wall_s / s.jobs for s in single])
    load = max(0.0, statistics.median([s.wall_s - per * s.jobs for s in cold])) if cold else 0.0
    inputs = [s.input_s / s.jobs for s in samples if s.input_s is not None]
    item_s = statistics.median(inputs) if len(inputs) == len(samples) and all(inputs) else None
    after = [s.resident_bytes for s in samples if s.resident_bytes is not None and not s.cold]
    after = after or [s.resident_bytes for s in samples if s.resident_bytes is not None]
    resident = min(after) if after else None
    reported = [s for s in samples if s.device_item_bytes is not None and s.device_allocated_bytes is not None]
    used = [s.device_bytes for s in samples if s.device_bytes is not None and s.served]
    device: tuple[int | None, int | None] = (min(used), max(used)) if used else (None, None)
    if reported:
        held = min(s.device_allocated_bytes or 0 for s in reported)
        device = (held, held + max(s.device_item_bytes or 0 for s in reported))
    return ResourceMeasure(
        load_s=load,
        claim_s=statistics.median([s.claim_s for s in samples]),
        resident_bytes=resident,
        peak_memory_bytes=None if resident is None else max(1, resident, *(s.peak_memory_bytes for s in samples)),
        device_resident_bytes=device[0],
        device_peak_bytes=None if device[1] is None else max(1, device[0] or 0, device[1]),
        s_per_input_s=per if item_s is None else per / item_s,
        input_s_per_item=item_s,
        batched=_batched(samples, item_s),
    )


def _batched(samples: Sequence[Sample], item_s: float | None) -> Batched | None:
    """Return the per-item seconds of the items sent in the largest batch, warm ones preferred; None for none."""
    top = max((s.batch_size for s in samples), default=1)
    if top == 1:
        return None
    at = [s for s in samples if s.batch_size == top]
    per = statistics.median(
        [s.wall_s / s.jobs for s in (s for s in at if not s.cold)] or [s.wall_s / s.jobs for s in at]
    )
    return Batched(batch_size=top, s_per_input_s=per if item_s is None else per / item_s)


@functools.cache
def code_version() -> str:
    """Return the installed senselab-core version and the digest of its source, `<version>+<12 hex>`.

    The digest is `identity_digest("core-runtime@1", files)` over the package's `*.py` files by relative path, so a
    checkout whose version did not move with its code still names different code differently. It is computed once per
    process.
    """
    root = Path(__file__).resolve().parents[1]
    files = sorted((p.relative_to(root).as_posix(), p) for p in root.rglob("*.py"))
    source = identity_digest("core-runtime@1", [[rel, file_digest("file@1", path)] for rel, path in files])
    return f"{importlib.metadata.version('senselab-core')}+{source[:12]}"


class Measures:
    """The measures calibration rounds of this code version wrote under `root`, then the packaged resource tables'.

    `root` defaults to `<cache root>/2x/resources/<code version>`, so a row written by other code is never read;
    `packaged` defaults to the packaged catalogue's tables, read on first use.
    """

    def __init__(self, root: Path | None = None, packaged: Sequence[ResourceTable] | None = None) -> None:
        """Read the tables under `root` first, then `packaged`; write under `root`."""
        if root is None:
            from senselab_core.runtime.revision import (  # noqa: PLC0415 - revision imports profiles
                NAMESPACE,
                cache_root,
            )

            root = cache_root() / NAMESPACE / "resources" / code_version()
        self.root = Path(root)
        self._packaged = packaged

    def get(self, unit: Unit) -> ResourceMeasure | None:
        """Return the measure of `unit`: a written table's, which is newer, else a packaged one's, else None."""
        if self._packaged is None:
            from senselab_core.runtime.call import packaged  # noqa: PLC0415 - call imports the planner

            self._packaged = packaged().catalogue.resource_tables
        for tables in (load_resource_tables(self.root), self._packaged):
            table = next((t for t in tables if (t.task, t.backend) == (unit.task, unit.backend)), None)
            row = None if table is None else table.row(unit.model_id, unit.revision)
            if row is not None and unit.resource_class in row.classes:
                return row.classes[unit.resource_class]
        return None

    def put(self, unit: Unit, measure: ResourceMeasure, measured: Measured) -> None:
        """Write `measure` of `unit` under `root` (`write_row`)."""
        write_row(self.root, unit, measure, measured)
