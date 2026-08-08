"""Region proposal from belief rows (FR-010, contracts/region-reprocessing.md §crop)."""

from __future__ import annotations

from typing import Any

from senselab.audio.workflows.audio_analysis.adaptive.types import AxisName, Region
from senselab.audio.workflows.audio_analysis.estimates import control_doubt


def propose_regions(
    rows: list[dict[str, Any]],
    *,
    axis: AxisName,
    policy: dict[str, Any],
    round_idx: int,
    duration_s: float,
) -> list[Region]:
    """Seed at ≥ θ_high, expand while ≥ θ_low, merge small gaps, pad, rank by mass.

    Rows must be time-ordered on one axis, and there is one set of them: a region is a span of the
    recording the run is unsure about, not a span of one pass. Proposing per (pass, axis) produced
    two overlapping regions for one ambiguity, each spending budget separately, and made the
    intervention catalogue's target a property of which pass happened to look worse.

    Only ``status == "open"`` rows can seed; closed rows still participate in expansion so a region
    keeps its natural extent. Region ids are deterministic: ``r<round>_<axis>_<idx>`` with idx
    assigned in start order (FR-025).
    """
    th = policy["thresholds"]
    rg = policy["regions"]
    theta_high, theta_low = float(th["theta_high"]), float(th["theta_low"])

    def _u(row: dict[str, Any]) -> float:
        # Doubt, not entropy: ``theta_high`` / ``theta_low`` are doubt-scaled, and comparing them
        # against the entropy column meant "seed above 17% doubt" (see ``estimates.control_doubt``).
        # ``-1.0`` for an unmeasured bucket, so it can never seed or extend a region.
        v = control_doubt(row)
        return -1.0 if v is None else float(v)

    # 1. seed + bidirectional expansion over contiguous indices.
    spans: list[tuple[int, int]] = []  # inclusive index ranges
    i = 0
    n = len(rows)
    while i < n:
        if rows[i].get("status") == "open" and _u(rows[i]) >= theta_high:
            lo = i
            while lo > 0 and _u(rows[lo - 1]) >= theta_low:
                lo -= 1
            hi = i
            while hi + 1 < n and _u(rows[hi + 1]) >= theta_low:
                hi += 1
            if spans and lo <= spans[-1][1] + 1:
                spans[-1] = (spans[-1][0], max(spans[-1][1], hi))
            else:
                spans.append((lo, hi))
            i = hi + 1
        else:
            i += 1

    # 2. time-domain gap merge.
    merged: list[tuple[float, float, list[int]]] = []
    for lo, hi in spans:
        start, end = float(rows[lo]["start"]), float(rows[hi]["end"])
        idxs = list(range(lo, hi + 1))
        if merged and start - merged[-1][1] < float(rg["gap_merge_s"]):
            pstart, _, pidx = merged[-1]
            merged[-1] = (pstart, end, pidx + idxs)
        else:
            merged.append((start, end, idxs))

    # 3. build regions: mass, pad, quantized core (rows are already on-grid).
    regions: list[Region] = []
    pad = float(rg["pad_s"])
    for start, end, idxs in merged:
        mass = 0.0
        for j in idxs:
            u = _u(rows[j])
            if u >= theta_low:
                mass += (u - theta_low) * (float(rows[j]["end"]) - float(rows[j]["start"]))
        regions.append(
            {
                "axis": axis,
                "core_start": start,
                "core_end": end,
                "crop_start": max(0.0, start - pad),
                "crop_end": min(duration_s, end + pad),
                "uncertainty_mass": round(mass, 9),
                "n_buckets": len(idxs),
                "status": "open",
                "region_id": "",  # assigned in start order below, after ranking
            }
        )

    # 4. rank by mass (desc, start asc tiebreak), cap top-N, then assign ids in start order.
    regions.sort(key=lambda r: (-r["uncertainty_mass"], r["core_start"]))
    regions = regions[: int(rg["top_n_per_round"])]
    regions.sort(key=lambda r: r["core_start"])
    for idx, r in enumerate(regions):
        r["region_id"] = f"r{round_idx}_{axis}_{idx}"
    return regions


def region_buckets(region: Region, rows: list[dict[str, Any]]) -> set[tuple[float, float]]:
    """Bucket keys of ``rows`` whose midpoint lies in the region core (merge-back rule)."""
    out: set[tuple[float, float]] = set()
    for row in rows:
        mid = (float(row["start"]) + float(row["end"])) / 2.0
        if region["core_start"] <= mid < region["core_end"]:
            out.add((round(float(row["start"]), 6), round(float(row["end"]), 6)))
    return out
