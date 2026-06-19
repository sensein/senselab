"""Tests for triage thresholding (FR-010b/c, SC-009)."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

from senselab.audio.workflows.ranking import io
from senselab.audio.workflows.ranking.metric import score_items
from senselab.audio.workflows.ranking.rank import build_ranking
from senselab.audio.workflows.ranking.triage import apply_triage_threshold
from senselab.audio.workflows.ranking.types import Annotation, MetricDefinition, Ranking, SignalTerm

from .conftest import quality_for_index


def _ranking(path: Path) -> Ranking:
    """Build a higher-is-better ranking from a one-signal table."""
    table = io.load_signal_table(path)
    items = score_items(table, MetricDefinition(name="m", terms=[SignalTerm("q", 1.0)]))
    return build_ranking(
        items, version_id="v1", unit=table.unit, direction="higher_is_better", band_fraction=0.20, provenance={}
    )


def test_rank_cut_partition(aligned_signals: Path) -> None:
    """A rank cut splits items into auto-accept and human-review counts."""
    ranking = _ranking(aligned_signals)
    result = apply_triage_threshold(ranking, [], cut=5, cut_kind="rank")
    assert result.n_auto_accept == 5
    assert result.n_human_review == 15


def test_percentile_cut_partition(aligned_signals: Path) -> None:
    """A percentile cut partitions all scored items."""
    ranking = _ranking(aligned_signals)
    result = apply_triage_threshold(ranking, [], cut=0.5, cut_kind="percentile")
    assert result.n_auto_accept + result.n_human_review == 20
    assert result.n_auto_accept > 0


def test_annotation_counts_and_poor_rate(aligned_signals: Path) -> None:
    """The auto-accept region reports annotated counts and poor-rate (SC-009)."""
    ranking = _ranking(aligned_signals)
    anns = [
        Annotation(item_id=f"item{i:03d}", label=quality_for_index(i, 20), score=None, unit="file") for i in range(20)
    ]
    result = apply_triage_threshold(ranking, anns, cut=7, cut_kind="rank")
    # Top 7 by q-desc are items 19..13; only 14..19 (6) are "good", item13 is "acceptable".
    assert result.above_counts["good"] == 6
    assert result.above_counts["acceptable"] == 1
    assert result.above_counts["poor"] == 0
    assert result.auto_accept_poor_rate == 0.0


def test_unscorable_auto_fail(make_signal_table: Callable[..., Path]) -> None:
    """Unscorable items are never auto-accepted; they route to human review (FR-010b)."""
    path = make_signal_table({"q": [10.0, math.nan, 5.0, 1.0]})
    ranking = _ranking(path)
    result = apply_triage_threshold(ranking, [], cut=1.0, cut_kind="percentile")
    assert result.n_unscorable_routed == 1
    assert result.n_auto_accept == 3
    assert result.n_human_review == 1
