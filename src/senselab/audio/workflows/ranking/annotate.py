"""Annotation store (latest-wins) and spot-check sampling.

Annotations are corpus-scoped and version-independent: every later metric
version sees the full active set (FR-014). Repeat annotations on the same item
are resolved by **latest-wins** — exactly one active annotation per item; the
prior active one is marked ``superseded`` and retained for history (FR-013).

The store holds only item ids, quality labels/scores, and short notes — never
raw audio/transcripts/PII content (spec Assumptions: low-sensitivity store).
"""

from __future__ import annotations

from dataclasses import asdict

from senselab.audio.workflows.ranking import io
from senselab.audio.workflows.ranking.store import RankingStore
from senselab.audio.workflows.ranking.types import Annotation, Ranking


def load_annotations(store: RankingStore) -> list[Annotation]:
    """Load all annotations (active and superseded) for the store."""
    if not store.annotations_path.exists():
        return []
    data = io.load_json(store.annotations_path)
    return [
        Annotation(
            item_id=a["item_id"],
            label=a.get("label"),
            score=a.get("score"),
            unit=a["unit"],
            reviewed_under_version=a.get("reviewed_under_version"),
            reviewer=a.get("reviewer"),
            created_at=a.get("created_at", ""),
            note=a.get("note", ""),
            resolution=a.get("resolution", "active"),
        )
        for a in data.get("annotations", [])
    ]


def load_active_annotations(store: RankingStore) -> list[Annotation]:
    """Load only the active (current) annotation per item."""
    return [a for a in load_annotations(store) if a.resolution == "active"]


def _save_annotations(store: RankingStore, unit: str, annotations: list[Annotation]) -> None:
    io.save_json(store.annotations_path, {"unit": unit, "annotations": [asdict(a) for a in annotations]})


def add_annotation(store: RankingStore, annotation: Annotation) -> None:
    """Add an annotation, superseding any prior active one for the same item (latest-wins)."""
    if annotation.label is None and annotation.score is None:
        raise ValueError("annotation must carry at least one of label / score")
    existing = load_annotations(store)
    for a in existing:
        if a.item_id == annotation.item_id and a.resolution == "active":
            a.resolution = "superseded"
    annotation.resolution = "active"
    existing.append(annotation)
    _save_annotations(store, annotation.unit, existing)


def add_annotations_batch(store: RankingStore, annotations: list[Annotation]) -> None:
    """Add several annotations, applying latest-wins per item in order."""
    for annotation in annotations:
        add_annotation(store, annotation)


def sample_items(
    ranking: Ranking,
    n: int,
    *,
    strategy: str = "spread",
    threshold_rank: int | None = None,
) -> list[str]:
    """Select up to ``n`` item ids to spot-check (FR-011).

    Strategies: ``spread`` (evenly across the ranking), ``near-threshold``
    (closest ranks to ``threshold_rank``), ``disagreement`` (around the
    top/middle and middle/bottom band boundaries).
    """
    scored = [it for it in ranking.items if it.status == "scored"]
    scored.sort(key=lambda it: it.rank or 0)
    total = len(scored)
    if total == 0 or n <= 0:
        return []
    n = min(n, total)

    if strategy == "spread":
        idxs = [round(i * (total - 1) / max(n - 1, 1)) for i in range(n)]
    elif strategy == "near-threshold":
        if threshold_rank is None:
            raise ValueError("near-threshold strategy requires threshold_rank")
        order = sorted(range(total), key=lambda i: abs((scored[i].rank or 0) - threshold_rank))
        idxs = sorted(order[:n])
    elif strategy == "disagreement":
        boundaries = [i for i in range(1, total) if scored[i].band != scored[i - 1].band]
        anchors = boundaries or [total // 2]
        idxs = []
        a = 0
        while len(idxs) < n:
            anchor = anchors[a % len(anchors)]
            for cand in (anchor, anchor - 1, anchor + 1):
                if 0 <= cand < total and cand not in idxs:
                    idxs.append(cand)
                    break
            a += 1
            if a > total * 2:
                break
        idxs = sorted(set(idxs))[:n]
    else:
        raise ValueError(f"unknown sampling strategy {strategy!r}")

    seen: list[str] = []
    for i in idxs:
        iid = scored[i].item_id
        if iid not in seen:
            seen.append(iid)
    return seen
