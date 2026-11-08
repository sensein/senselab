"""Sizing: a unit's batch, quantum and worker count, derived from a measure of it and the grant it runs under.

`size` is pure. Its measure is a resource-table row's (`measure_of`), or one a calibration round wrote: one-job items
run first on one worker, the first carrying the model load (`calibrated`, from each item's `Sample`). `Measures` reads
the packaged tables, then the ones calibration rounds wrote under the cache.
"""

import importlib.metadata
import math
import statistics
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated

from pydantic import Field

from senselab_core.kernel._base import Frozen, NonEmptyStr
from senselab_core.runtime.execution_settings import PoolSettings, ResourceClass, SizingSettings
from senselab_core.runtime.resources import (
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
    when unmeasured.
    """

    load_s: Seconds
    claim_s: Seconds
    resident_bytes: Bytes | None
    bytes_per_item: Bytes | None
    s_per_item: Seconds
    on_device: bool = False


class Grant(Frozen):
    """What a unit's workers are granted: the most alive at once, memory per worker (None unknown), quantum ceilings.

    `devices` is the count of devices a `cuda` class's workers share; None where each worker is granted its own device
    or runs on none.
    """

    max_workers: Count
    memory_bytes: Count | None
    quantum_items: Count
    quantum_s: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    devices: Annotated[int, Field(ge=0)] | None = None

    @classmethod
    def of(  # noqa: PLR0913 - the class and pool, then what the host measured
        cls,
        resource: ResourceClass,
        pool: PoolSettings,
        host_memory_bytes: int | None = None,
        *,
        on_device: bool = False,
        devices: int | None = None,
    ) -> "Grant":
        """Return the grant of a worker of `resource`: its device memory cap `on_device`, else its memory or the host's.

        The quantum ceilings are the pool's, `quantum_s` within the class's wall-time limit. `devices`, the devices the
        class's workers share, is kept for a `cuda` class only.
        """
        memory = resource.memory_cap_bytes if on_device else resource.mem_bytes or host_memory_bytes or None
        time_s = pool.quantum_s if resource.time_s is None else min(pool.quantum_s, resource.time_s)
        return cls(
            max_workers=resource.max_workers,
            memory_bytes=memory,
            quantum_items=pool.quantum_items,
            quantum_s=time_s,
            devices=devices if resource.device == "cuda" else None,
        )

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


class Sizing(Frozen):
    """Items per request, jobs and seconds per quantum, and the workers a queue's waiting work wants."""

    batch_size: Count
    quantum_items: Count
    quantum_s: Seconds
    workers: Annotated[int, Field(ge=0)]


def size(measure: Measure, grant: Grant, config: SizingSettings, *, work: int = 0, batchable: bool = True) -> Sizing:
    """Return the sizing of a unit measured as `measure` under `grant`, for `work` waiting jobs.

    - `quantum_items`: the fewest jobs whose time keeps the load and the claim within `load_share` of the quantum
      (the ceiling when both are 0), within the ceilings: `quantum_items`, and the jobs that fit in `quantum_s` after
      them;
    - `quantum_s`: the quantum's jobs plus the load and claim, within `quantum_s`;
    - `batch_size`: the most items whose memory beside the resident memory fits `memory_headroom` of the grant's,
      at most a quantum; 1 when the grant's or the unit's memory is unmeasured or `batchable` is False;
    - `workers`: as many as end `work` within `target_wall_s`, at most a quantum each and `Grant.worker_bound`; 0 for
      none.
    """
    per, load, ceiling = measure.s_per_item, measure.load_s + measure.claim_s, grant.quantum_items
    paid = config.load_share * per
    wanted = _held(load / paid, ceiling, math.ceil) if paid and load else ceiling
    fits = _held((grant.quantum_s - load) / per, ceiling, math.floor) if per else ceiling
    quantum = max(1, min(wanted, fits))
    batch = 1
    if batchable and grant.memory_bytes is not None and measure.resident_bytes is not None:
        room = config.memory_headroom * grant.memory_bytes - measure.resident_bytes
        per_bytes = measure.bytes_per_item or 0
        batch = 1 if room < 0 else quantum if per_bytes == 0 else max(1, _held(room / per_bytes, quantum, math.floor))
    workers = 0
    if work > 0:
        timely = max(1, _held(work * per / config.target_wall_s, grant.max_workers, math.ceil))
        workers = min(timely, math.ceil(work / quantum), grant.worker_bound(measure))
    return Sizing(
        batch_size=batch,
        quantum_items=quantum,
        quantum_s=min(quantum * per + load, grant.quantum_s),
        workers=workers,
    )


def _held(x: float, bound: int, rounded: Callable[[float], int]) -> int:
    """Return `rounded(x)` held within `[0, bound]`, for an infinite `x` too."""
    return bound if x >= bound else rounded(max(x, 0.0))


def measure_of(row: ResourceMeasure, *, on_device: bool = False) -> Measure:
    """Return the per-item measure of a resource-table measure, at its median item's input seconds.

    `on_device` takes the device memory pair when the row measured one, else the host pair.
    """
    seconds = row.s_per_input_s if row.input_s_per_item is None else row.s_per_input_s * row.input_s_per_item
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
    )


class Sample(Frozen):
    """What a pool worker measured of one item it ran.

    Attributes:
        wall_s: The item's share, by jobs, of the seconds from its claim to its claim's outcomes.
        claim_s: Seconds the claim holding the item took to take.
        jobs: The item's jobs.
        input_s: Their recording seconds; None when a job has no recording input or its length is unrecorded.
        baseline_bytes: The worker's host memory before it ran anything.
        resident_bytes: Its host memory once the item's outcomes were in, the model loaded and the item freed; None
            where only a high-water is readable.
        peak_memory_bytes: Its host high-water memory once the item ran.
        device_baseline_bytes: The device memory this worker's processes used before it ran anything; None off a
            device.
        device_bytes: The device memory they used once the item's claim had its outcomes, the model loaded and
            nothing in flight; None off a device.
        cold: Whether the worker had run nothing of the item's queue before, so the item carried the model load.
        job: The scheduler job id or process id of the worker.
    """

    wall_s: Seconds
    claim_s: Seconds
    jobs: Count
    input_s: Seconds | None
    baseline_bytes: Bytes
    resident_bytes: Bytes | None
    peak_memory_bytes: Bytes
    device_baseline_bytes: Bytes | None = None
    device_bytes: Bytes | None = None
    cold: bool
    job: NonEmptyStr


def calibrated(samples: Sequence[Sample]) -> ResourceMeasure | None:
    """Return the measure a calibration round's `samples` give; None for none.

    Seconds per item are the median over warm items (over cold ones when none is warm), the load the median of what
    a cold item took beyond its jobs, the claim the median claim. The resident host memory is the lowest after a warm
    item (after any item when none is warm) and the peak the highest; with no resident figure the host pair is
    unmeasured. The device pair is the lowest and the highest device use read after items, when the items ran on a
    device: the lowest holds the loaded model alone, so per-item device bytes exclude it.
    """
    if not samples:
        return None
    warm = [s.wall_s / s.jobs for s in samples if not s.cold]
    cold = [s for s in samples if s.cold]
    per = statistics.median(warm or [s.wall_s / s.jobs for s in cold])
    load = max(0.0, statistics.median([s.wall_s - per * s.jobs for s in cold])) if cold else 0.0
    inputs = [s.input_s / s.jobs for s in samples if s.input_s is not None]
    item_s = statistics.median(inputs) if len(inputs) == len(samples) and all(inputs) else None
    after = [s.resident_bytes for s in samples if s.resident_bytes is not None and not s.cold]
    after = after or [s.resident_bytes for s in samples if s.resident_bytes is not None]
    resident = min(after) if after else None
    used = [s.device_bytes for s in samples if s.device_bytes is not None]
    device = (min(used), max(used)) if used else (None, None)
    return ResourceMeasure(
        load_s=load,
        claim_s=statistics.median([s.claim_s for s in samples]),
        resident_bytes=resident,
        peak_memory_bytes=None if resident is None else max(1, resident, *(s.peak_memory_bytes for s in samples)),
        device_resident_bytes=device[0],
        device_peak_bytes=None if device[1] is None else max(1, device[0] or 0, device[1]),
        s_per_input_s=per if item_s is None else per / item_s,
        input_s_per_item=item_s,
    )


def code_version() -> str:
    """Return the installed senselab-core version, which names the commit it was built from."""
    return importlib.metadata.version("senselab-core")


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
