"""Sizing: a unit's batch, quantum and worker count, derived from an estimate of it and the grant it runs under.

`size` is pure; `Sample` is what a pool worker reports of an item it ran.
"""

import bisect
import logging
import math
import threading
from collections.abc import Sequence
from typing import Annotated

from pydantic import Field

from senselab_core.kernel._base import Frozen, NonEmptyStr
from senselab_core.runtime.calibration import Estimate, LengthProfile, MemoryFit, RegimeFactor, StratumBatch
from senselab_core.runtime.execution_settings import CalibrationSettings, PoolSettings, ResourceClass, SizingSettings
from senselab_core.runtime.hardware import HardwareInfo
from senselab_core.runtime.probe import memory_batch, peak, raised

Seconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Bytes = Annotated[int, Field(ge=0)]
Count = Annotated[int, Field(ge=1)]


class Grant(Frozen):
    """What a unit's workers are granted: the most alive at once, memory per worker (None unknown), quantum ceilings.

    `devices` is the count of devices a `cuda` class's workers share; None where each worker is granted its own device
    or runs on none. `on_device` says `memory_bytes` is a device's, so sizing reads the device memory fit; `shared_host`
    says it is the host's, shared by every worker of the class on it.
    """

    max_workers: Count
    memory_bytes: Count | None
    quantum_items: Count
    quantum_s: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    devices: Annotated[int, Field(ge=0)] | None = None
    on_device: bool = False
    shared_host: bool = False

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
            shared_host=not on_device and not resource.mem_bytes and memory is not None,
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


class RowSizing(Frozen):
    """Items per request, the most memory allows, jobs and seconds per quantum, and the workers a queue's work wants."""

    batch_size: Count
    memory_batch: Count
    quantum_items: Count
    quantum_s: Seconds
    workers: Annotated[int, Field(ge=0)]


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
        extrapolated: Whether a job's known length lies outside `measured_range`, so its time is extrapolated.
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
    extrapolated: bool = False
    measured: Basis

    def batch_at(self, length: float | None, jobs: int | None = None) -> int:
        """Return the batch for jobs of `length` units in an item of `jobs` jobs.

        It is the least of the estimate's batch for that length, the memory batch and `jobs`; 1 for an unknown length,
        one above the measured range, or unmeasured memory.
        """
        return _batch(self, length, self.ceiling if jobs is None else min(jobs, self.ceiling))


logger = logging.getLogger(__name__)


class ExtrapolationLog:
    """Says once per unit per run that `size` extrapolated beyond the unit's measured lengths."""

    def __init__(self) -> None:
        """Start a run with no unit said."""
        self._said: set[str] = set()
        self._lock = threading.Lock()

    def note(self, label: str, sized: Sizing) -> bool:
        """Log, the first time for `label` (a unit's `MeasureUnit.label`), the measured range `sized` extrapolated past.

        Returns whether a line was logged.
        """
        if not sized.extrapolated or sized.measured_range is None:
            return False
        with self._lock:
            if label in self._said:
                return False
            self._said.add(label)
        lo, hi = sized.measured_range
        logger.info("unit %s: lengths outside the measured range [%g, %g] are extrapolated linearly", label, lo, hi)
        return True


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


def _by_count(grant: Grant, jobs: int) -> tuple[tuple[int, int], ...]:
    return tuple((a, min(a + grant.quantum_items, jobs)) for a in range(0, jobs, grant.quantum_items))


def _bucket(length: float | None, edges: Sequence[float]) -> int:
    """Return the index of the bucket holding `length` (`bucket i` is `[edges[i-1], edges[i])`), -1 when unknown."""
    return -1 if length is None else bisect.bisect_right(edges, length)


def _on_edges(cuts: tuple[tuple[int, int], ...], buckets: Sequence[int]) -> tuple[tuple[int, int], ...]:
    """Return `cuts` split wherever the bucket of consecutive jobs changes; unchanged without buckets."""
    if not buckets:
        return cuts
    split: list[tuple[int, int]] = []
    for start, stop in cuts:
        at = start
        for i in range(start + 1, stop):
            if buckets[i] != buckets[i - 1]:
                split.append((at, i))
                at = i
        split.append((at, stop))
    return tuple(split)


def _unmeasured(grant: Grant, calibration: CalibrationSettings, jobs: int, buckets: Sequence[int]) -> Sizing:
    cuts = _on_edges(_by_count(grant, jobs), buckets)
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
    bucket_edges: Sequence[float] = (),
) -> Sizing:
    """Return the sizing of `work` under `grant` from `estimate` (None: unmeasured, so the safe values), purely.

    A job's seconds at batch 1 are `r + u·L`, at its length's batch B `(r_B + u_B·B·L)/B`, times the regime factor of
    the co-tenancy the card's workers make. Items are cut in put order at the upper prediction
    `t·(1 + confidence·spread)`: an item closes before its upper seconds plus the per-item cost pass the grant's
    `quantum_s`, when its seconds reach `(load_s + a)/load_share`, or at `quantum_items` jobs; with `u` or `spread`
    unmeasured they are cut by `quantum_items` alone; with `bucket_edges` (a bucketed formation's grid, in input units)
    every item is also cut where the bucket changes, so no item spans two. Workers are the median
    seconds over `target_wall_s`, at most the item count, `max_workers` and, on shared devices, `devices × w_d`.

    On shared devices `w_d = ⌊memory_headroom × cap / P⌋` for the batch peak `P` at the longest job, lowered where a
    measured regime says another worker adds no throughput, and each worker's memory is `cap / w_d`; on a shared host
    `w` is that count at most `max_workers`, and each worker's memory is `host / w`. Either way the `oom` records of
    co-tenancy 2 to `w` raise the fit (`raised`). The fit read is
    the device's on a device grant, the host's otherwise, never the other. A length above the measured range, or an
    unmeasured memory, gives batch 1; so does an unknown length. Lengths are `0` for an estimate of unit kind `none`.
    """
    jobs = len(work.units)
    single = None if estimate is None else estimate.single
    buckets = [_bucket(length, bucket_edges) for length in work.units] if bucket_edges else []
    if estimate is None or single is None:
        return _unmeasured(grant, calibration, jobs, buckets)
    fit = estimate.device if grant.on_device else estimate.host
    span = (single.lo, single.hi)
    lengths = [single.hi if length is None else length for length in work.units]
    longest = max(lengths, default=0.0)
    share = grant.memory_bytes if fit is not None else None
    per_device = 1
    if share is not None and fit is not None and (grant.shared_host or grant.devices is not None and grant.on_device):
        star = _star(estimate.batch, longest) if longest <= span[1] else 1
        batch_peak = peak(fit, star, longest)
        fits = grant.max_workers if batch_peak <= 0 else max(1, math.floor(config.memory_headroom * share / batch_peak))
        sharing = _regime_workers(estimate.regimes, fits) if grant.on_device else min(fits, grant.max_workers)
        resource = "device" if grant.on_device else "host"
        shared = [r for r in estimate.oom if r.resource == resource and 2 <= r.cotenancy <= sharing]  # noqa: PLR2004
        fit = raised(fit, shared, config.memory_headroom)
        share //= sharing
        per_device = sharing if grant.on_device else 1
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
    cuts = _on_edges(cuts, buckets)
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
            "extrapolated": any(length is not None and not span[0] <= length <= span[1] for length in work.units),
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
