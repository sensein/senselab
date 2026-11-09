"""The calibration probe: `Probe.plan` chooses what to run, `Probe.estimate` fits what the observations say."""

import itertools
import math
import statistics
import struct
import sys
from bisect import bisect_left, bisect_right
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

import numpy as np

from senselab_core.runtime.calibration import (
    Candidate,
    Estimate,
    EstimateReason,
    EstimateSource,
    HardwareClass,
    MemoryFit,
    Observation,
    OomRecord,
    PlanReason,
    Prior,
    ProbeItem,
    ProbePhase,
    ProbePlan,
    RegimeFactor,
    RegimeRecord,
    RequestFit,
    Stratum,
    StratumBatch,
    UnitKind,
    Unplanned,
)
from senselab_core.runtime.execution_settings import CalibrationSettings

if TYPE_CHECKING:
    from senselab_core.runtime.measures import MeasureUnit
    from senselab_core.runtime.sizing import Grant

MAD_TO_SIGMA: Final = 1 / statistics.NormalDist().inv_cdf(
    0.75
)  # invariant: a median absolute deviation to a normal's σ
MEDIAN_SE: Final = math.sqrt(math.pi / 2)  # invariant: the standard error of a median relative to that of a mean


def _median(values: Sequence[float]) -> float:
    return statistics.median(sorted(values))


def _inversions(z: np.ndarray) -> int:
    """Count the pairs `i < j` with `z[i] > z[j]`, by merging runs of doubling length level by level."""
    ranks = np.unique(z, return_inverse=True)[1].ravel().astype(np.int64)
    n, span, total = len(ranks), int(ranks.max(initial=0)) + 1, 0
    index, run = np.arange(n), 1
    while run < n:
        block, right = index // (2 * run), (index // run) % 2 == 1
        keys = block * span + ranks
        left = np.sort(keys[~right])
        at_most = np.searchsorted(left, keys[right], side="right") - np.searchsorted(left, block[right] * span)
        total += int((run - at_most).sum())
        run *= 2
    return total


def _inverted_pairs(z: Sequence[float]) -> list[tuple[int, int]]:
    """List the pairs `i < j` with `z[i] > z[j]`, in time linear in their number beyond n log n."""

    def merge(run: list[int]) -> list[int]:
        if len(run) <= 1:
            return run
        half = len(run) // 2
        left, right = merge(run[:half]), merge(run[half:])
        keys = [z[i] for i in left]
        pairs.extend((i, j) for j in right for i in left[bisect_right(keys, z[j]) :])
        return sorted(left + right, key=z.__getitem__)

    pairs: list[tuple[int, int]] = []
    merge(list(range(len(z))))
    return pairs


def _ordered(value: float) -> int:
    """Map a double to an integer in the same order, so a bisection over integers halves the doubles."""
    bits: int = struct.unpack("<q", struct.pack("<d", value))[0]
    return bits if bits >= 0 else -(2**63) - bits


def _unordered(bits: int) -> float:
    value: float = struct.unpack("<d", struct.pack("<q", bits if bits >= 0 else -(2**63) - bits))[0]
    return value


def _rank_bound(points: Sequence[tuple[float, float]], slope: float) -> float:
    """Return `2ε·(max|y − t·x̃| + |t|·max|x̃|) / min Δx` at `t = slope`: `x̃` is x centred at its median, `ε` epsilon.

    `_kth_slope` compares points by `y − t·x̃`, which rounds once in the product and once in the subtraction; a pair
    compares two of them, so it is ranked wrongly only if its slope is within this distance of `t`.
    """
    xs, ys = np.array([x for x, _ in points]), np.array([y for _, y in points])
    centred = xs - float(np.median(xs))
    spread = float(np.abs(ys - slope * centred).max()) + abs(slope) * float(np.abs(centred).max())
    return 2 * sys.float_info.epsilon * spread / float(np.diff(np.unique(xs)).min())


def _kth_slope(xs: np.ndarray, ys: np.ndarray, k: int, total: int) -> float:
    """Return the `k`-th smallest (from 0) of the `total` slopes between points of distinct `x`, sorted by `(x, y)`.

    A pair's slope is below `t` exactly when the pair is inverted in `y − t·x`, so the slopes below `t` are counted by
    inversion counting and bisected over the doubles until at most n lie between the bounds, which are then listed.
    Slopes tied beyond that are the one value of the pairs adjacent in `y − t·x` at the bound. The comparisons use `x`
    centred at its median, which leaves every slope unchanged and makes differences of nearby values exact; a pair is
    then ranked wrongly only if its slope is within `_rank_bound` of `t`, and the slopes returned are computed from the
    original `x`.
    """
    centred = xs - float(np.median(xs))
    bound = max(float(np.ptp(ys)) / float(np.diff(np.unique(xs)).min()), 1.0)
    low, high, below, above = _ordered(-2 * bound), _ordered(2 * bound), 0, total
    while above - below > len(xs) and high - low > 1:
        middle = (low + high) // 2
        count = _inversions(ys - _unordered(middle) * centred)
        low, below = (middle, count) if count <= k else (low, below)
        high, above = (high, above) if count <= k else (middle, count)
    if above - below > len(xs):
        pivot = _unordered(low)
        near = np.argsort(ys - pivot * centred, kind="stable")
        tied = [(ys[b] - ys[a]) / (xs[b] - xs[a]) for a, b in itertools.pairwise(near.tolist()) if xs[a] != xs[b]]
        return float(min(tied, key=lambda slope: abs(slope - pivot), default=pivot))
    lower, upper = ys - _unordered(low) * centred, ys - _unordered(high) * centred
    order = sorted(range(len(xs)), key=lambda i: (lower[i], i))
    slopes = sorted(
        (ys[order[j]] - ys[order[i]]) / (xs[order[j]] - xs[order[i]])
        for i, j in _inverted_pairs([upper[o] for o in order])
    )
    return float(slopes[min(max(k - below, 0), len(slopes) - 1)])


def _theil_sen(points: Sequence[tuple[float, float]]) -> tuple[float, float] | None:
    """Return `(intercept, slope)`: the median of the pairwise slopes, and the median intercept at that slope.

    The slope is the median of the n(n−1)/2 pairwise slopes, found in O(n log² n) time and O(n) memory; it equals the
    reference's except for slopes closer than `_rank_bound` to it.
    """
    ordered = sorted(points)
    xs, ys = np.array([x for x, _ in ordered]), np.array([y for _, y in ordered])
    ties = sum(c * (c - 1) // 2 for c in Counter(xs.tolist()).values())
    total = len(ordered) * (len(ordered) - 1) // 2 - ties
    if total == 0:
        return None
    middle = (total - 1) // 2
    slope = _kth_slope(xs, ys, middle, total)
    if total % 2 == 0:
        slope = statistics.median([slope, _kth_slope(xs, ys, middle + 1, total)])
    return _median([y - slope * x for x, y in ordered]), slope


def _length(o: Observation) -> float:
    return o.padded_units / o.batch


def _cuts(units: Sequence[float], count: int) -> list[tuple[int, int]]:
    """Return `[start, stop)` runs cutting ascending `units` into at most `count` of equal count, no value in two."""
    n, parts = len(units), min(count, len(units))
    bounds = set()
    for i in range(1, parts):
        b = i * n // parts
        moves = [b] if units[b - 1] != units[b] else [bisect_left(units, units[b]), bisect_right(units, units[b])]
        if inside := [p for p in moves if 0 < p < n]:
            bounds.add(min(inside, key=lambda p: (abs(p - b), p)))
    edges = [0, *sorted(bounds), n]
    return [(a, z) for a, z in itertools.pairwise(edges) if a < z]


def _by_stratum(items: Sequence[Observation], ranges: Sequence[tuple[float, float]]) -> dict[int, list[Observation]]:
    """Group `items` by the length range holding each one's per-job length."""
    grouped: dict[int, list[Observation]] = {}
    los = [lo for lo, _ in ranges]
    for o in items:
        i = bisect_right(los, _length(o)) - 1
        if i >= 0 and _length(o) <= ranges[i][1]:
            grouped.setdefault(i, []).append(o)
    return grouped


@dataclass(frozen=True)
class _Kept:
    """The observations that count, split by what they give."""

    single: list[Observation]
    batched: list[Observation]
    loads: list[Observation]
    items: list[float]
    warm: list[Observation]
    ranges: list[tuple[float, float]]


def _unit_kind(observations: Sequence[Observation], hardware: HardwareClass) -> tuple[UnitKind, Sequence[Observation]]:
    """Return the kind of input the unit's probe observations share, and the observations measured by it.

    A unit whose observations mix kinds is of kind `none`, and its observations are returned with units 0, so each
    request is measured by itself.
    """
    kinds = {o.unit_kind for o in observations if o.probe is not None and o.hardware == hardware}
    if len(kinds) <= 1:
        return (next(iter(kinds)) if kinds else "none"), observations
    flat = {"unit_kind": "none", "units": 0.0, "padded_units": 0.0}
    return "none", [o.model_copy(update=flat) for o in observations]


def _select(observations: Sequence[Observation], hardware: HardwareClass, config: CalibrationSettings) -> _Kept:
    """Keep `ok` probe requests of the unit's hardware in the regime each probe ran under (§5.4 steps 1-2).

    The strata are the warm requests' per-job lengths cut into `length_strata` ranges of equal count.
    """
    bisected = {(o.probe.id, o.item) for o in observations if o.probe is not None and o.outcome == "bisected"}
    ok = [
        o
        for o in observations
        if o.outcome == "ok"
        and o.probe is not None
        and o.hardware == hardware
        and (o.probe.id, o.item) not in bisected
        and o.cotenancy.device == 1
    ]
    quietest: dict[str, int] = {}
    for o in ok:
        if o.probe is not None:
            quietest[o.probe.id] = min(o.cotenancy.host, quietest.get(o.probe.id, o.cotenancy.host))
    kept = [o for o in ok if o.probe is not None and o.cotenancy.host == quietest[o.probe.id]]
    warm = [o for o in kept if o.load_s is None and o.since_load is not None and o.since_load >= config.warmup_requests]
    admitted = {id(o) for o in warm}
    lengths = sorted(_length(o) for o in warm)
    whole: dict[tuple[str, str], list[Observation]] = {}
    for o in observations:
        if o.probe is not None and o.hardware == hardware:
            whole.setdefault((o.probe.id, o.item), []).append(o)
    extras: list[float] = []
    for requests in whole.values():
        last = [o.item_wall_s for o in requests if o.item_wall_s is not None]
        if last and all(id(o) in admitted for o in requests):
            extras.append(max(0.0, last[0] - sum(o.wall_s for o in requests)))
    return _Kept(
        single=[o for o in warm if o.batch == 1],
        batched=[o for o in warm if o.batch > 1],
        loads=[o for o in kept if o.load_s is not None],
        items=extras,
        warm=warm,
        ranges=[(lengths[a], lengths[z - 1]) for a, z in _cuts(lengths, config.length_strata)],
    )


def _request_fit(
    observations: Sequence[Observation],
    batch: int,
    units: Callable[[Observation], float],
    config: CalibrationSettings,
    ranges: Sequence[tuple[float, float]],
) -> RequestFit | None:
    """Fit `wall_s = r + u·units` over the strata with `min_requests` observations; None when none has them."""
    strata = {s: o for s, o in _by_stratum(observations, ranges).items() if len(o) >= config.min_requests}
    points = [(units(o), o.wall_s) for group in strata.values() for o in group]
    if not points:
        return None
    xs = [x for x, _ in points]
    fit = _theil_sen(points) if len(strata) >= 2 else None  # noqa: PLR2004 - a slope needs two points in length
    if fit is not None and fit[1] >= 0:
        r, u = max(fit[0], 0.0), fit[1]
        predicted = [r + u * x for x in xs]
        spread = _relative_spread([y - p for (_, y), p in zip(points, predicted, strict=True)], predicted)
        return RequestFit(batch=batch, r=r, u=u, n=len(points), spread=spread, lo=min(xs), hi=max(xs))
    walls = [y for _, y in points]
    centre = _median(walls)
    spread = _relative_spread([y - centre for y in walls], [centre] * len(walls))
    return RequestFit(batch=batch, r=centre, u=None, n=len(points), spread=spread, lo=min(xs), hi=max(xs))


def _relative_spread(residuals: Sequence[float], predicted: Sequence[float]) -> float | None:
    centre = _median(predicted)
    if centre <= 0:
        return None
    middle = _median(residuals)
    return MAD_TO_SIGMA * _median([abs(e - middle) for e in residuals]) / centre


def _memory_fit(
    kept: _Kept, kind: Literal["device", "host"], config: CalibrationSettings
) -> tuple[MemoryFit | None, EstimateReason | None]:
    """Return the memory fit of `kind` over warm requests, and the reason it is None."""
    device = kind == "device"
    rows = {
        id(o): (peak, after)
        for o in kept.warm
        if (peak := o.device_peak if device else o.host_peak) is not None
        and (after := o.device_after if device else o.host_after) is not None
    }
    if not rows:
        return None, "device_unmeasured" if device else "host_unmeasured"
    groups = _by_stratum([o for o in kept.warm if id(o) in rows], kept.ranges).values()
    points = [
        (o.padded_units, float(rows[id(o)][0] - rows[id(o)][1]))
        for group in groups
        if len(group) >= config.min_requests
        for o in group
    ]
    fit = _theil_sen(points)
    if fit is None:
        return None, "device_below_min_requests" if device else "host_below_min_requests"
    if fit[1] < 0:
        return None, "device_fit_nonphysical" if device else "host_fit_nonphysical"
    resident = min(after for _, after in rows.values())
    return MemoryFit(resident=resident, m0=fit[0], m1=fit[1], n=len(points)), None


def peak(fit: MemoryFit, batch: int, length: float) -> float:
    """Return the memory of one request of `batch` jobs of `length` units: `resident + m0 + m1·batch·length`."""
    return fit.resident + fit.m0 + fit.m1 * batch * length


def memory_batch(fit: MemoryFit, budget_bytes: float, length: float, ceiling: int) -> int:
    """Return the most jobs of `length` units whose batch peak fits `budget_bytes`, within `[1, ceiling]`."""
    room = budget_bytes - fit.resident - fit.m0
    per_job = fit.m1 * length
    if room < 0:
        return 1
    if per_job <= 0:
        return ceiling
    return max(1, math.floor(min(ceiling, room / per_job)))


def _holding(spans: Sequence[tuple[float, float]], length: float) -> tuple[float, float] | None:
    """Return the span holding `length`, else the nearest one, the one at or above `length` on a tie."""
    inside = [s for s in spans if s[0] <= length <= s[1]]
    if inside or not spans:
        return inside[0] if inside else None
    return min(spans, key=lambda s: (min(abs(length - s[0]), abs(length - s[1])), s[0] < length, s))


def _probe_regime(record: OomRecord) -> bool:
    """Whether a probe can measure the fault: one process on the device, at a batch of two or more."""
    return record.cotenancy == 1 and record.batch >= 2  # noqa: PLR2004 - a batch of one is re-placed, not raised


def _in_force(records: Sequence[RegimeRecord], kept: _Kept, minimum: int) -> tuple[OomRecord, ...]:
    """Return the distinct `oom` records in force, oldest first (§7.4).

    A probe measures and lifts in the probe regime only: a record there is lifted once a probe outside its `basis`
    has `minimum` warm batched observations in the stratum holding its length; any other record stays in force.
    """
    live = []
    for r in dict.fromkeys(x for x in records if isinstance(x, OomRecord)):
        span = _holding(kept.ranges, r.length)
        later = Counter(
            o.probe.id
            for o in kept.batched
            if span is not None
            and o.probe is not None
            and o.probe.id not in r.basis
            and span[0] <= _length(o) <= span[1]
        )
        if not (_probe_regime(r) and any(n >= minimum for n in later.values())):
            live.append(r)
    return tuple(live)


def _raised(fit: MemoryFit | None, records: Sequence[OomRecord], memory_headroom: float) -> MemoryFit | None:
    """Return `fit` with `m1` raised so that every unlifted faulting pair is infeasible (§7.4)."""
    if fit is None:
        return None
    m1 = fit.m1
    for r in records:
        if r.length > 0:
            m1 = max(m1, (memory_headroom * r.grant_bytes - fit.resident - fit.m0) / ((r.batch - 1) * r.length))
    return fit.model_copy(update={"m1": m1})


def _batch_choice(
    single: RequestFit, batched: Sequence[RequestFit], lengths: Mapping[tuple[float, float], float], z: float
) -> tuple[StratumBatch, ...]:
    """Return per stratum the best measured batch that beats one beyond the margin, else 1 (§5.4 step 6)."""
    chosen = []
    for (lo, hi), length in sorted(lengths.items()):
        t1 = single.r + (single.u or 0.0) * length
        best, best_t = 1, t1
        for fit in sorted(batched, key=lambda f: f.batch):
            if single.spread is None or fit.spread is None or t1 <= 0:
                continue
            t_b = (fit.r + (fit.u or 0.0) * fit.batch * length) / fit.batch
            sigma = MEDIAN_SE * math.sqrt(single.spread**2 / single.n + fit.spread**2 / fit.n)
            if (t1 - t_b) / t1 > z * sigma and t_b < best_t:
                best, best_t = fit.batch, t_b
        chosen.append(StratumBatch(lo=lo, hi=hi, batch=best))
    return tuple(chosen)


def _spread_idx(size: int, count: int) -> list[int]:
    """Return `count` distinct positions of `size`, spread evenly by rank."""
    return [(2 * i + 1) * size // (2 * count) for i in range(count)]


def _order(c: Candidate) -> tuple[bool, float, str, int]:
    return (c.units is None, c.units or 0.0, c.fragment, c.index)


def _strata(
    candidates: Sequence[Candidate], count: int
) -> tuple[tuple[Stratum, ...], tuple[tuple[Candidate, ...], ...]]:
    """Return one job of each key cut into `count` strata of equal count by units; lengthless ones form one more."""
    unique = sorted({c.key: c for c in sorted(candidates, key=_order, reverse=True)}.values(), key=_order)
    sized = [c for c in unique if c.units is not None]
    blank = [c for c in unique if c.units is None]
    pieces = [sized[a:z] for a, z in _cuts([c.units or 0.0 for c in sized], count)]
    if blank:
        pieces.append(blank)
    spans = tuple(
        Stratum(index=i, lo=p[0].units, hi=p[-1].units if p[0].units is not None else None)
        for i, p in enumerate(pieces)
    )
    return spans, tuple(tuple(p) for p in pieces)


def _covered(
    observations: Sequence[Observation], stratum: Stratum, minimum: int, units: Callable[[Observation], float]
) -> bool:
    """Whether `minimum` of `observations` lie in the stratum's units (zero units for a lengthless one)."""
    if stratum.lo is None or stratum.hi is None:
        return sum(units(o) == 0 for o in observations) >= minimum
    return sum(stratum.lo <= units(o) <= stratum.hi for o in observations) >= minimum


def _reopened(strata: Sequence[Stratum], records: Sequence[OomRecord]) -> set[int]:
    """Return the strata holding the length of a probe-regime `oom` record in force (the nearest one beyond them)."""
    measured = [s for s in strata if s.lo is not None and s.hi is not None]
    spans = [(s.lo or 0.0, s.hi or 0.0) for s in measured]
    found: set[int] = set()
    for r in filter(_probe_regime, records):
        span = _holding(spans, r.length)
        found |= {s.index for s in measured if span == (s.lo, s.hi)}
    return found


def _unplanned(
    plan: ProbePlan, reason: PlanReason, stratum: int | None = None, phase: ProbePhase = "BATCHED"
) -> ProbePlan:
    return plan.model_copy(
        update={"unplanned": (*plan.unplanned, Unplanned(phase=phase, stratum=stratum, reason=reason))}
    )


def _single_items(
    free: list[list[Candidate]],
    wanted: Sequence[int],
    used: int,
    config: CalibrationSettings,
    unplanned: list[Unplanned],
) -> tuple[list[ProbeItem], int]:
    """Take the `SINGLE` items of the `wanted` strata from `free` (shortest first); return them and the jobs used."""
    singles: list[ProbeItem] = []
    for index in wanted:
        counts = [(config.min_requests, 1)]
        if index == len(free) // 2:
            counts.append((config.min_requests * config.single_item_jobs, config.single_item_jobs))
        for count, per_item in counts:
            if len(free[index]) < count:
                unplanned.append(Unplanned(phase="SINGLE", stratum=index, reason="too_few_candidates"))
            elif used + count > config.probe_items:
                unplanned.append(Unplanned(phase="SINGLE", stratum=index, reason="over_budget"))
            else:
                used += count
                taken = [free[index].pop(0) for _ in range(count)]
                singles += [
                    ProbeItem(phase="SINGLE", stratum=index, request_jobs=1, jobs=tuple(taken[i : i + per_item]))
                    for i in range(0, count, per_item)
                ]
    return singles, used


def _second(first: int, config: CalibrationSettings) -> int | None:
    """Return `B₂`: the integer nearest √`first`, when `batched_sizes` is 2 and it is at least 2 and not `first`."""
    second = math.floor(math.sqrt(first) + 0.5)
    return second if config.batched_sizes == 2 and 2 <= second != first else None  # noqa: PLR2004


def _per_set(first: int, config: CalibrationSettings) -> tuple[int, int]:
    """Return the jobs and the items of one repetition of a stratum's batched sizes."""
    second = _second(first, config)
    return (first, 1) if second is None else (first + second, 2)


def _widths(first: int, config: CalibrationSettings) -> list[int]:
    """Return the job count of each batched item of a stratum at `B₁` = `first`, widest first."""
    second = _second(first, config)
    return [first] * config.min_requests + ([second] * config.min_requests if second else [])


def _buckets(edges: Sequence[float], pool: Sequence[Candidate]) -> list[list[int]]:
    """Return the positions of `pool` grouped by bucket of `edges`, ascending (one group when there are no edges)."""
    groups: dict[int, list[int]] = {}
    for i, c in enumerate(pool):
        groups.setdefault(bisect_right(edges, c.units or 0.0), []).append(i)
    return [groups[k] for k in sorted(groups)]


def _pack(sizes: Sequence[int], widths: Sequence[int]) -> list[list[int]] | None:
    """Return per bucket the widths of the items it supplies, each item from one bucket; None when they do not fit.

    Each item, widest first, goes to the bucket with the most jobs left, so one width is supplied while
    `Σ_b ⌊n_b / w⌋` reaches the items asked.
    """
    left = list(sizes)
    packed: list[list[int]] = [[] for _ in sizes]
    for w in widths:
        b = max(range(len(left)), key=lambda i: (left[i], -i), default=None)
        if b is None or left[b] < w:
            return None
        left[b] -= w
        packed[b].append(w)
    return packed


def _supply(first: int, pool: Sequence[Candidate], edges: Sequence[float], config: CalibrationSettings) -> int:
    """Return the largest `B₁` within `first` whose items the `pool`'s buckets supply, one bucket per item."""
    sizes = [len(g) for g in _buckets(edges, pool)]
    while first >= 2 and _pack(sizes, _widths(first, config)) is None:  # noqa: PLR2004 - a batch of one is single
        first -= 1
    return first


def _blocked(plan: ProbePlan, single_estimate: Estimate, grant: "Grant") -> PlanReason | None:
    """Return why no batched phase can be planned from `single_estimate` under `grant`, None when it can."""
    if plan.formation == "single":
        return "single_formation"
    if single_estimate.single is None:
        return "single_unmeasured"
    if single_estimate.device is None:
        return "device_unmeasured"
    return "grant_unbounded" if grant.memory_bytes is None else None


def _deal(jobs: Sequence[Candidate], widths: Sequence[int]) -> list[list[Candidate]]:
    """Deal `jobs` in turn to items of `widths`, skipping full ones, so each item spans the jobs' range."""
    items: list[list[Candidate]] = [[] for _ in widths]
    slot = 0
    for job in jobs:
        while len(items[slot % len(widths)]) >= widths[slot % len(widths)]:
            slot += 1
        items[slot % len(widths)].append(job)
        slot += 1
    return items


def _batched_items(
    plan: ProbePlan, sizes: Mapping[int, int], none: ProbePlan, config: CalibrationSettings, jobs: int
) -> ProbePlan:
    """Return `none` with the batched items of `sizes` (`B₁` per stratum), their jobs spread by rank."""
    items: list[ProbeItem] = []
    remaining = [list(r) for r in plan.remaining]
    for index, first in sizes.items():
        groups = _buckets(plan.edges, remaining[index])
        packed = _pack([len(g) for g in groups], _widths(first, config))
        assert packed is not None, "_supply sized this batch to fit its buckets"
        gone: set[int] = set()
        for group, widths in zip(groups, packed, strict=True):
            if not widths:
                continue
            picks = [group[i] for i in _spread_idx(len(group), sum(widths))]
            gone |= set(picks)
            dealt = _deal([remaining[index][i] for i in picks], widths)
            items += [ProbeItem(phase="BATCHED", stratum=index, request_jobs=len(b), jobs=tuple(b)) for b in dealt]
        remaining[index] = [c for i, c in enumerate(remaining[index]) if i not in gone]
    return none.model_copy(
        update={
            "items": tuple(items),
            "batched": tuple(sizes),
            "remaining": tuple(tuple(r) for r in remaining),
            "used_jobs": plan.used_jobs + jobs,
        }
    )


class Probe:
    """The calibration probe: what to run (`plan`), and what the observations say (`estimate`). Pure; no I/O."""

    @staticmethod
    def estimate(  # noqa: PLR0913 - the observations and records, then the unit's hardware, kind and headroom
        observations: Sequence[Observation],
        records: Sequence[RegimeRecord],
        config: CalibrationSettings,
        *,
        hardware: HardwareClass,
        memory_headroom: float,
    ) -> Estimate:
        """Return the estimate the observations of one unit give (§5.4).

        Only `ok` probe requests of `hardware`, in the regime their probe ran under, count; cold or reloaded ones give
        a load and nothing else. The unit's kind of input is its observations' when they share one; when they mix
        kinds each request is measured by itself (units 0), so `u` is unmeasured. `records` are the unit's regime and
        `oom` records, oldest first; a later regime record of one `k` replaces an earlier one. An `oom` record raises
        the memory fit by `memory_headroom` of its grant (§7.4) unless other processes shared its device, and only
        until a probe outside its `basis` measures its stratum.
        """
        unit_kind, observations = _unit_kind(observations, hardware)
        kept = _select(observations, hardware, config)
        reasons: list[EstimateReason] = []
        single = _request_fit(kept.single, 1, lambda o: o.units, config, kept.ranges)
        if single is None:
            reasons.append("single_below_min_requests" if kept.warm else "no_usable_observations")
        elif single.u is None:
            reasons.append("single_slope_unmeasured")
        sizes = sorted({o.batch for o in kept.batched})
        fits = [
            _request_fit([o for o in kept.batched if o.batch == b], b, lambda o: o.padded_units, config, kept.ranges)
            for b in sizes
        ]
        batched = tuple(f for f in fits if f is not None)
        if len(batched) < len(sizes):
            reasons.append("batched_below_min_requests")
        if not kept.loads:
            reasons.append("load_unmeasured")
        if len(kept.items) < config.min_requests:
            reasons.append("item_below_min_requests")
        memory = {}
        for kind in ("device", "host"):
            fit, why = _memory_fit(kept, kind, config)
            memory[kind] = fit
            if why is not None:
                reasons.append(why)
        ooms = _in_force(records, kept, config.min_requests)
        probed = list(filter(_probe_regime, ooms))
        device = _raised(memory["device"], [r for r in probed if r.resource == "device"], memory_headroom)
        host = _raised(memory["host"], [r for r in probed if r.resource == "host"], memory_headroom)
        lengths = {
            kept.ranges[s]: _median([o.units for o in group])
            for s, group in _by_stratum(kept.single, kept.ranges).items()
            if len(group) >= config.min_requests
        }
        regimes = {r.k: r for r in records if isinstance(r, RegimeFactor)}
        return Estimate(
            status="measured" if single is not None else "unmeasured",
            reasons=tuple(reasons),
            unit_kind=unit_kind,
            load_s=_median([o.load_s or 0.0 for o in kept.loads]) if kept.loads else None,
            load_n=len(kept.loads),
            item_s=_median(kept.items) if len(kept.items) >= config.min_requests else None,
            item_n=len(kept.items),
            single=single,
            batched=batched,
            batch=() if single is None else _batch_choice(single, batched, lengths, config.confidence),
            device=device,
            host=host,
            regimes=tuple(sorted(regimes.values(), key=lambda r: r.k)),
            oom=ooms,
            source=EstimateSource(
                probe_ids=tuple(sorted({o.probe.id for o in kept.warm + kept.loads if o.probe is not None})),
                observations=len(kept.warm) + len(kept.loads),
            ),
        )

    @staticmethod
    def estimator(
        config: CalibrationSettings,
        *,
        records_of: Callable[["MeasureUnit"], Sequence[RegimeRecord]],
        memory_headroom: float,
    ) -> Callable[["MeasureUnit", Sequence[Observation]], Estimate]:
        """Return `Probe.estimate` as `MeasureSource`'s `estimate=` callable: `(unit, observations) -> Estimate`.

        The records and the hardware are each resolved for the unit asked about.
        """
        return lambda unit, observations: Probe.estimate(
            observations,
            records_of(unit),
            config,
            hardware=unit.hardware,
            memory_headroom=memory_headroom,
        )

    @staticmethod
    def plan(  # noqa: PLR0913 - the protocol's inputs (§4.1)
        unit: "MeasureUnit",
        candidates: Sequence[Candidate],
        prior: Prior,
        config: CalibrationSettings,
        formation: str,
        seconds: float,
        *,
        bucket_edges: Sequence[float],
    ) -> ProbePlan:
        """Return the `WARMUP` and `SINGLE` items of a probe over `candidates`, and what a batched phase may use.

        `candidates` are the held jobs the store would not serve. One job of each key is kept, the rest cut into
        `length_strata` strata of equal count, and each (phase, stratum) `prior` already covers with `min_requests`
        steady observations is left out, a batched stratum an `oom` record of `prior` re-opens excepted. `seconds`
        is the warm-work budget the plan carries for the Pool. `bucket_edges` are the input-unit edges of a `bucket:`
        formation's grid (empty for any other): a batched item draws its jobs from one bucket.

        Raises:
            ValueError: the formation is bucketed and `bucket_edges` is empty.
        """
        if formation.startswith("bucket:") and not bucket_edges:
            raise ValueError("a bucketed formation plans over its bucket edges; none were given")
        kept = _select(prior.observations, unit.hardware, config)
        strata, pools = _strata(candidates, config.length_strata)
        plan = ProbePlan(
            items=(),
            strata=strata,
            remaining=pools,
            batched=(),
            unplanned=(),
            formation=formation,
            seconds=seconds,
            probe_items=config.probe_items,
            used_jobs=0,
            edges=tuple(sorted(bucket_edges)),
        )
        if not strata:
            return plan
        middle = len(strata) // 2
        unplanned: list[Unplanned] = []
        wanted = []
        for st in strata:
            covered = _covered(kept.single, st, config.min_requests, lambda o: o.units)
            if st.index == middle:
                covered = covered and len(kept.items) >= config.min_requests
            if covered:
                unplanned.append(Unplanned(phase="SINGLE", stratum=st.index, reason="covered"))
            else:
                wanted.append(st.index)
        reopened = _reopened(strata, _in_force(prior.records, kept, config.min_requests))
        batched = tuple(
            st.index
            for st in strata
            if formation != "single"
            and (
                st.index in reopened
                or not _covered(kept.batched, st, config.min_requests, lambda o: o.padded_units / o.batch)
            )
        )
        plan = plan.model_copy(update={"batched": batched})
        free = [list(p) for p in pools]
        warmup = config.warmup_requests
        if warmup > len(free[0]) or warmup > config.probe_items:
            wanted = []
            unplanned.append(Unplanned(phase="WARMUP", stratum=None, reason="too_few_candidates"))
            warmups: list[ProbeItem] = []
        else:
            warmups = [
                ProbeItem(phase="WARMUP", stratum=0, request_jobs=1, jobs=(free[0].pop(0),)) for _ in range(warmup)
            ]
        singles, used = _single_items(free, wanted, warmup, config, unplanned)
        if not singles and not (batched and warmups):
            return plan.model_copy(update={"unplanned": tuple(unplanned)})
        return plan.model_copy(
            update={
                "items": (*warmups, *singles),
                "remaining": tuple(tuple(f) for f in free),
                "unplanned": tuple(unplanned),
                "used_jobs": used,
            }
        )

    @staticmethod
    def plan_batched(  # noqa: PLR0913 - the batched phase's inputs (§4.3)
        plan: ProbePlan,
        single_estimate: Estimate,
        grant: "Grant",
        config: CalibrationSettings,
        seconds_left: float,
        *,
        memory_headroom: float,
    ) -> ProbePlan:
        """Return `plan` with the `BATCHED` items its `SINGLE` estimate allows (§4.3).

        `B₁` per stratum is the least of the memory batch at its longest candidate under `memory_headroom` of the
        grant, the batch the item budget leaves, the batch whose items the single fit's upper prediction fits in
        `seconds_left`, `grant.quantum_items`, and what the stratum's remaining candidates supply. `B₂` is the integer
        nearest √B₁ when `batched_sizes` is 2. The result's `items` are the batched ones only.
        """
        none = plan.model_copy(update={"items": (), "batched": ()})
        single, device = single_estimate.single, single_estimate.device
        if plan.formation != "single" and plan.strata and not plan.batched:
            return _unplanned(none, "covered")
        blocked = _blocked(plan, single_estimate, grant)
        if blocked is not None or single is None or device is None or grant.memory_bytes is None:
            return _unplanned(none, blocked or "single_unmeasured")
        spans = {st.index: st for st in plan.strata}
        caps: dict[int, int] = {}
        times: dict[int, float] = {}
        for index in plan.batched:
            hi = spans[index].hi
            if hi is None:
                none = _unplanned(none, "no_lengths", index)
                continue
            budget = memory_headroom * grant.memory_bytes
            caps[index] = memory_batch(device, budget, hi, grant.quantum_items)
            times[index] = (single.r + (single.u or 0.0) * hi) * (1 + config.confidence * (single.spread or 0.0))
        left = plan.probe_items - plan.used_jobs
        item_s = single_estimate.item_s or 0.0
        supplied = fits_jobs = False
        for top in range(max(caps.values(), default=0), 1, -1):
            sizes = {i: _supply(min(c, top), plan.remaining[i], plan.edges, config) for i, c in caps.items()}
            sizes = {i: b for i, b in sizes.items() if b >= 2}  # noqa: PLR2004 - a batch of one is the single phase
            if not sizes:
                continue
            supplied = True
            jobs = sum(config.min_requests * _per_set(b, config)[0] for b in sizes.values())
            wall = sum(
                config.min_requests * (_per_set(b, config)[0] * times[i] + item_s * _per_set(b, config)[1])
                for i, b in sizes.items()
            )
            fits_jobs = fits_jobs or jobs <= left
            if jobs <= left and wall <= seconds_left:
                return _batched_items(plan, sizes, none, config, jobs)
        if not caps:
            return none
        if max(caps.values()) < 2:  # noqa: PLR2004 - a batch of one is the single phase
            return _unplanned(none, "memory_below_two")
        return _unplanned(
            none, "too_few_candidates" if not supplied else "over_seconds" if fits_jobs else "over_budget"
        )

    @staticmethod
    def replace_served(plan: ProbePlan, served_item: ProbeItem) -> ProbePlan:
        """Return `plan` with `served_item` (a probe request that came back `served`) replaced from its stratum.

        The replacement has the item's phase, stratum and shape: the shortest remaining candidates for a `WARMUP` or
        `SINGLE` item, candidates spread by rank for a `BATCHED` one. With too few left the item is dropped and the
        plan says so.

        Raises:
            ValueError: `served_item` is not an item of `plan`.
        """
        if served_item not in plan.items:
            raise ValueError("the served item is not an item of the plan")
        at = plan.items.index(served_item)
        pool = list(plan.remaining[served_item.stratum])
        need = len(served_item.jobs)
        bucket = bisect_right(plan.edges, served_item.jobs[0].units or 0.0)
        usable = [
            i
            for i, c in enumerate(pool)
            if served_item.phase != "BATCHED" or bisect_right(plan.edges, c.units or 0.0) == bucket
        ]
        others = (*plan.items[:at], *plan.items[at + 1 :])
        if len(usable) < need:
            dropped = _unplanned(plan, "too_few_candidates", served_item.stratum, served_item.phase)
            return dropped.model_copy(update={"items": others})
        picks = [usable[i] for i in (range(need) if served_item.phase != "BATCHED" else _spread_idx(len(usable), need))]
        jobs = tuple(pool[i] for i in picks)
        gone = set(picks)
        pool = [c for i, c in enumerate(pool) if i not in gone]
        remaining = tuple(tuple(pool) if i == served_item.stratum else r for i, r in enumerate(plan.remaining))
        item = served_item.model_copy(update={"jobs": jobs})
        return plan.model_copy(
            update={"items": (*plan.items[:at], item, *plan.items[at + 1 :]), "remaining": remaining}
        )
