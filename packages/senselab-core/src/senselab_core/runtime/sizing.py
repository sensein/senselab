"""Sizing: a unit's batch, quantum and worker count, derived from a measure of it and the grant it runs under.

`size` is pure. Its measure is a resource-table row's (`measure_of`), or one a calibration round wrote: one-job items
run first on one worker, the first carrying the model load (`calibrated`, from each item's `Sample`). `Measures` reads
the packaged tables, then the ones calibration rounds wrote under the cache.
"""

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
    """What one item costs on one resource class: seconds to load, bytes resident after it, bytes and seconds each."""

    load_s: Seconds
    resident_bytes: Bytes
    bytes_per_item: Bytes
    s_per_item: Seconds


class Grant(Frozen):
    """What a unit's workers are granted: the most alive at once, memory per worker (None unknown), quantum ceilings."""

    max_workers: Count
    memory_bytes: Count | None
    quantum_items: Count
    quantum_s: Annotated[float, Field(gt=0, allow_inf_nan=False)]

    @classmethod
    def of(cls, resource: ResourceClass, pool: PoolSettings, host_memory_bytes: int | None = None) -> "Grant":
        """Return the grant of a worker of `resource`: its memory cap, else its memory, else `host_memory_bytes`.

        The quantum ceilings are the pool's, `quantum_s` within the class's wall-time limit.
        """
        memory = resource.memory_cap_bytes or resource.mem_bytes or host_memory_bytes or None
        time_s = pool.quantum_s if resource.time_s is None else min(pool.quantum_s, resource.time_s)
        return cls(
            max_workers=resource.max_workers, memory_bytes=memory, quantum_items=pool.quantum_items, quantum_s=time_s
        )


class Sizing(Frozen):
    """Items per request, jobs and seconds per quantum, and the workers a queue's waiting work wants."""

    batch_size: Count
    quantum_items: Count
    quantum_s: Seconds
    workers: Annotated[int, Field(ge=0)]


def size(measure: Measure, grant: Grant, config: SizingSettings, *, work: int = 0, batchable: bool = True) -> Sizing:
    """Return the sizing of a unit measured as `measure` under `grant`, for `work` waiting jobs.

    - `quantum_items`: the fewest jobs whose time keeps the load within `load_share` of the quantum, within the
      ceilings: `quantum_items`, and as many jobs as fit in `quantum_s` after the load;
    - `quantum_s`: the quantum's jobs plus the load, within `quantum_s`;
    - `batch_size`: the most items whose memory beside the resident memory fits `memory_headroom` of the grant's,
      at most a quantum; 1 when the grant's memory is unknown or `batchable` is False;
    - `workers`: as many as end `work` within `target_wall_s`, at most a quantum each and `max_workers`; 0 for none.
    """
    per, load, ceiling = measure.s_per_item, measure.load_s, grant.quantum_items
    paid = config.load_share * per
    wanted = 1 if load == 0 else _held(load / paid, ceiling, math.ceil) if paid else ceiling
    fits = _held((grant.quantum_s - load) / per, ceiling, math.floor) if per else ceiling
    quantum = max(1, min(wanted, fits))
    batch = 1
    if batchable and grant.memory_bytes is not None:
        room = config.memory_headroom * grant.memory_bytes - measure.resident_bytes
        per_bytes = measure.bytes_per_item
        batch = 1 if room < 0 else quantum if per_bytes == 0 else max(1, _held(room / per_bytes, quantum, math.floor))
    workers = 0
    if work > 0:
        timely = max(1, _held(work * per / config.target_wall_s, grant.max_workers, math.ceil))
        workers = min(timely, math.ceil(work / quantum))
    return Sizing(
        batch_size=batch,
        quantum_items=quantum,
        quantum_s=min(quantum * per + load, grant.quantum_s),
        workers=workers,
    )


def _held(x: float, bound: int, rounded: Callable[[float], int]) -> int:
    """Return `rounded(x)` held within `[0, bound]`, for an infinite `x` too."""
    return bound if x >= bound else rounded(max(x, 0.0))


def measure_of(row: ResourceMeasure) -> Measure:
    """Return the per-item measure of a resource-table measure, at its median item's input seconds."""
    seconds = row.s_per_input_s if row.input_s_per_item is None else row.s_per_input_s * row.input_s_per_item
    return Measure(
        load_s=row.load_s,
        resident_bytes=row.resident_bytes,
        bytes_per_item=row.peak_memory_bytes - row.resident_bytes,
        s_per_item=seconds,
    )


class Sample(Frozen):
    """What a pool worker measured of one item it ran.

    Attributes:
        wall_s: Seconds from the claim to the item's outcomes.
        jobs: The item's jobs.
        input_s: Their recording seconds; None when a job has no recording input or its length is unrecorded.
        baseline_bytes: The worker's memory before it ran anything.
        peak_memory_bytes: Its high-water memory once the item ran.
        cold: Whether the worker had run nothing of the item's queue before, so the item carried the model load.
        job: The scheduler job id or process id of the worker.
    """

    wall_s: Seconds
    jobs: Count
    input_s: Seconds | None
    baseline_bytes: Bytes
    peak_memory_bytes: Count
    cold: bool
    job: NonEmptyStr


def calibrated(samples: Sequence[Sample]) -> ResourceMeasure | None:
    """Return the measure a calibration round's `samples` give; None for none.

    Seconds per item are the median over warm items (over cold ones when none is warm), the load is the median of
    what a cold item took beyond its jobs, the resident memory is the lowest baseline and the peak the highest.
    """
    if not samples:
        return None
    warm = [s.wall_s / s.jobs for s in samples if not s.cold]
    cold = [s for s in samples if s.cold]
    per = statistics.median(warm or [s.wall_s / s.jobs for s in cold])
    load = max(0.0, statistics.median([s.wall_s - per * s.jobs for s in cold])) if cold else 0.0
    inputs = [s.input_s / s.jobs for s in samples if s.input_s is not None]
    item_s = statistics.median(inputs) if len(inputs) == len(samples) and all(inputs) else None
    resident = min(s.baseline_bytes for s in samples)
    return ResourceMeasure(
        load_s=load,
        resident_bytes=resident,
        peak_memory_bytes=max(resident, *(s.peak_memory_bytes for s in samples)),
        s_per_input_s=per if item_s is None else per / item_s,
        input_s_per_item=item_s,
    )


class Measures:
    """The measures of the packaged resource tables, then of those calibration rounds wrote under `root`.

    `root` defaults to `<cache root>/2x/resources`; `packaged` to the packaged catalogue's tables, read on first use.
    """

    def __init__(self, root: Path | None = None, packaged: Sequence[ResourceTable] | None = None) -> None:
        """Read `packaged` tables first, then the tables under `root`; write under `root`."""
        if root is None:
            from senselab_core.runtime.revision import (  # noqa: PLC0415 - revision imports profiles
                NAMESPACE,
                cache_root,
            )

            root = cache_root() / NAMESPACE / "resources"
        self.root = Path(root)
        self._packaged = packaged

    def get(self, unit: Unit) -> ResourceMeasure | None:
        """Return the measure of `unit`: a packaged table's, else a written one's, else None."""
        if self._packaged is None:
            from senselab_core.runtime.call import packaged  # noqa: PLC0415 - call imports the planner

            self._packaged = packaged().catalogue.resource_tables
        for tables in (self._packaged, load_resource_tables(self.root)):
            table = next((t for t in tables if (t.task, t.backend) == (unit.task, unit.backend)), None)
            row = None if table is None else table.row(unit.model_id, unit.revision)
            if row is not None and unit.resource_class in row.classes:
                return row.classes[unit.resource_class]
        return None

    def put(self, unit: Unit, measure: ResourceMeasure, measured: Measured) -> None:
        """Write `measure` of `unit` under `root` (`write_row`)."""
        write_row(self.root, unit, measure, measured)
