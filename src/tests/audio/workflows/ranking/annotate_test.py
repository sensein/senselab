"""Tests for the annotation store (latest-wins) and sampling (FR-011/013/014)."""

from __future__ import annotations

from pathlib import Path

import pytest

from senselab.audio.workflows.ranking import annotate, io
from senselab.audio.workflows.ranking.metric import score_items
from senselab.audio.workflows.ranking.rank import build_ranking
from senselab.audio.workflows.ranking.store import RankingStore
from senselab.audio.workflows.ranking.types import Annotation, MetricDefinition, QualityLabel, Ranking, SignalTerm


def _ann(item_id: str, label: QualityLabel, ts: str) -> Annotation:
    """Build a file-unit ordinal annotation."""
    return Annotation(item_id=item_id, label=label, score=None, unit="file", created_at=ts)


def _aligned_ranking(aligned_signals: Path) -> Ranking:
    """Build a ranking from the aligned-signals fixture."""
    table = io.load_signal_table(aligned_signals)
    items = score_items(table, MetricDefinition(name="m", terms=[SignalTerm("q", 1.0)]))
    return build_ranking(
        items, version_id="v1", unit=table.unit, direction="higher_is_better", band_fraction=0.20, provenance={}
    )


def test_latest_wins_supersedes_prior(store: RankingStore) -> None:
    """A newer annotation supersedes the prior active one; history is retained."""
    annotate.add_annotation(store, _ann("x", "poor", "t0"))
    annotate.add_annotation(store, _ann("x", "good", "t1"))
    active = annotate.load_active_annotations(store)
    assert len(active) == 1
    assert active[0].label == "good"
    all_anns = annotate.load_annotations(store)
    assert len(all_anns) == 2
    assert sum(a.resolution == "superseded" for a in all_anns) == 1


def test_full_set_available_across_versions(store: RankingStore) -> None:
    """All active annotations are visible regardless of version."""
    annotate.add_annotation(store, _ann("x", "good", "t0"))
    annotate.add_annotation(store, _ann("y", "poor", "t0"))
    assert {a.item_id for a in annotate.load_active_annotations(store)} == {"x", "y"}


def test_batch_add(store: RankingStore) -> None:
    """Batch ingest adds several annotations."""
    annotate.add_annotations_batch(store, [_ann("a", "good", "t0"), _ann("b", "poor", "t0")])
    assert len(annotate.load_active_annotations(store)) == 2


def test_annotation_requires_label_or_score(store: RankingStore) -> None:
    """An annotation must carry at least a label or a score."""
    with pytest.raises(ValueError, match="at least one"):
        annotate.add_annotation(store, Annotation(item_id="x", label=None, score=None, unit="file"))


def test_sample_spread_covers_range(aligned_signals: Path) -> None:
    """Spread sampling returns distinct items including both extremes."""
    ranking = _aligned_ranking(aligned_signals)
    ids = annotate.sample_items(ranking, 5, strategy="spread")
    assert len(ids) == 5
    assert len(set(ids)) == 5
    assert "item019" in ids and "item000" in ids


def test_sample_near_threshold(aligned_signals: Path) -> None:
    """Near-threshold sampling returns items close to the given rank."""
    ranking = _aligned_ranking(aligned_signals)
    ids = annotate.sample_items(ranking, 3, strategy="near-threshold", threshold_rank=10)
    ranks = {it.item_id: it.rank for it in ranking.items}
    assert all(abs((ranks[i] or 0) - 10) <= 2 for i in ids)
