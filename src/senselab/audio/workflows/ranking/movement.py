"""Compare two rankings: per-item ordinal shift + coarse band-region movement.

Accounts for 100% of items across both versions (FR-023/SC-006): moved /
unchanged / added / removed / became-unscorable. Band movement is a coarse
summary consistent with the per-item band transitions (SC-007), not an exact
boundary ledger. Both rankings must share the same unit.
"""

from __future__ import annotations

from senselab.audio.workflows.ranking.types import (
    Annotation,
    ItemDelta,
    MovementEntry,
    MovementReport,
    Ranking,
)


def _index(ranking: Ranking) -> dict[str, object]:
    return {it.item_id: it for it in ranking.items}


def compute_movement(
    from_ranking: Ranking,
    to_ranking: Ranking,
    annotations: list[Annotation] | None = None,
) -> MovementReport:
    """Build a :class:`MovementReport` comparing ``from_ranking`` → ``to_ranking``."""
    if from_ranking.unit != to_ranking.unit:
        raise ValueError(f"cannot compare rankings across units {from_ranking.unit!r} vs {to_ranking.unit!r}")

    active = {a.item_id: a for a in (annotations or []) if a.resolution == "active"}
    a_items = _index(from_ranking)
    b_items = _index(to_ranking)
    all_ids = sorted(set(a_items) | set(b_items))

    entries: list[MovementEntry] = []
    added: list[str] = []
    removed: list[str] = []
    became_unscorable: list[str] = []
    band_summary = {"entered_top": 0, "left_top": 0, "entered_bottom": 0, "left_bottom": 0}

    for iid in all_ids:
        a = a_items.get(iid)
        b = b_items.get(iid)
        a_scored = a is not None and a.status == "scored"  # type: ignore[attr-defined]
        b_scored = b is not None and b.status == "scored"  # type: ignore[attr-defined]

        from_rank = a.rank if a_scored else None  # type: ignore[attr-defined]
        to_rank = b.rank if b_scored else None  # type: ignore[attr-defined]
        from_band = a.band if a_scored else None  # type: ignore[attr-defined]
        to_band = b.band if b_scored else None  # type: ignore[attr-defined]

        if a is None:
            delta_kind: ItemDelta = "added"
            added.append(iid)
        elif b is None:
            delta_kind = "removed"
            removed.append(iid)
        elif a_scored and not b_scored:
            delta_kind = "became_unscorable"
            became_unscorable.append(iid)
        elif not a_scored and not b_scored:
            delta_kind = "unchanged"
        elif a_scored and b_scored:
            delta_kind = "unchanged" if from_rank == to_rank else "moved"
        else:  # unscorable in from, scored in to — entered the ranking
            delta_kind = "moved"

        position_delta = (from_rank - to_rank) if (from_rank is not None and to_rank is not None) else None
        from_pct = a.percentile if a_scored else None  # type: ignore[attr-defined]
        to_pct = b.percentile if b_scored else None  # type: ignore[attr-defined]
        percentile_delta = (from_pct - to_pct) if (from_pct is not None and to_pct is not None) else None

        if to_band == "top" and from_band != "top":
            band_summary["entered_top"] += 1
        if from_band == "top" and to_band != "top":
            band_summary["left_top"] += 1
        if to_band == "bottom" and from_band != "bottom":
            band_summary["entered_bottom"] += 1
        if from_band == "bottom" and to_band != "bottom":
            band_summary["left_bottom"] += 1

        ann = active.get(iid)
        entries.append(
            MovementEntry(
                item_id=iid,
                from_rank=from_rank,
                to_rank=to_rank,
                position_delta=position_delta,
                percentile_delta=percentile_delta,
                from_band=from_band,
                to_band=to_band,
                delta_kind=delta_kind,
                annotated=ann is not None,
                annotation_label=ann.label if ann is not None else None,
            )
        )

    return MovementReport(
        from_version=from_ranking.version_id,
        to_version=to_ranking.version_id,
        unit=from_ranking.unit,
        band_fraction=to_ranking.band_fraction,
        entries=entries,
        band_summary=band_summary,
        added=added,
        removed=removed,
        became_unscorable=became_unscorable,
    )
