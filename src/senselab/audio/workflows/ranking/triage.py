"""Triage threshold: partition a ranking into auto-accept vs. human-review.

Items above the cut are treated as confidently good (release-ready); the rest go
to human review (FR-010b/c). **Unscorable items are auto-fail** — never
auto-accepted, always routed to human review regardless of the cut, and counted
as ``n_unscorable_routed`` (spec FR-010b / Clarifications 2026-06-04).
"""

from __future__ import annotations

from senselab.audio.workflows.ranking.constants import QUALITY_LABELS
from senselab.audio.workflows.ranking.types import Annotation, CutKind, Ranking, TriageThreshold


def _label_counts(labels: list[str | None]) -> dict[str, int]:
    counts = {lab: 0 for lab in QUALITY_LABELS}
    for lab in labels:
        if lab in counts:
            counts[lab] += 1
    return counts


def apply_triage_threshold(
    ranking: Ranking,
    annotations: list[Annotation],
    *,
    cut: float,
    cut_kind: CutKind = "percentile",
) -> TriageThreshold:
    """Compute the auto-accept / human-review split and its annotation readout."""
    active = {a.item_id: a for a in annotations if a.resolution == "active"}

    above_labels: list[str | None] = []
    below_labels: list[str | None] = []
    n_auto_accept = 0
    n_human_review = 0
    n_unscorable_routed = 0

    for it in ranking.items:
        if it.status != "scored":
            n_human_review += 1
            n_unscorable_routed += 1
            if it.item_id in active:
                below_labels.append(active[it.item_id].label)
            continue
        if cut_kind == "rank":
            accept = (it.rank or 0) <= cut
        else:  # percentile: top portion has the smallest percentile
            accept = (it.percentile if it.percentile is not None else 1.0) <= cut
        label = active[it.item_id].label if it.item_id in active else None
        if accept:
            n_auto_accept += 1
            above_labels.append(label)
        else:
            n_human_review += 1
            below_labels.append(label)

    above_counts = _label_counts(above_labels)
    annotated_above = sum(above_counts.values())
    poor_rate = (above_counts["poor"] / annotated_above) if annotated_above else None

    return TriageThreshold(
        version_id=ranking.version_id,
        cut=float(cut),
        cut_kind=cut_kind,
        n_auto_accept=n_auto_accept,
        n_human_review=n_human_review,
        n_unscorable_routed=n_unscorable_routed,
        above_counts=above_counts,
        below_counts=_label_counts(below_labels),
        auto_accept_poor_rate=poor_rate,
    )
