"""Tests for ranking-quality evaluation (FR-008/010/010a, SC-001/008)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from senselab.audio.workflows.ranking import io
from senselab.audio.workflows.ranking.evaluate import evaluate_ranking
from senselab.audio.workflows.ranking.metric import score_items
from senselab.audio.workflows.ranking.rank import build_ranking
from senselab.audio.workflows.ranking.types import Annotation, MetricDefinition, Ranking, SignalTerm

from .conftest import quality_for_index


def _ranking(path: Path) -> Ranking:
    """Build a higher-is-better ranking from a one-signal table."""
    table = io.load_signal_table(path)
    defn = MetricDefinition(name="m", terms=[SignalTerm("q", 1.0)])
    items = score_items(table, defn)
    return build_ranking(
        items, version_id="v1", unit=table.unit, direction="higher_is_better", band_fraction=0.20, provenance={}
    )


def _annotate_aligned(n: int) -> list[Annotation]:
    """Annotate items by tertile so quality rises with item index."""
    return [
        Annotation(item_id=f"item{i:03d}", label=quality_for_index(i, n), score=None, unit="file") for i in range(n)
    ]


def test_good_ranking_high_agreement(aligned_signals: Path) -> None:
    """A well-aligned ranking yields high rank agreement and full band separation."""
    ranking = _ranking(aligned_signals)
    result = evaluate_ranking(ranking, _annotate_aligned(20))
    assert result.evaluable
    assert result.rank_agreement_spearman is not None and result.rank_agreement_spearman > 0.8
    assert result.band_pairwise_agreement == 1.0
    assert result.meets_separation_target is True


def test_not_evaluable_without_annotations(aligned_signals: Path) -> None:
    """With no annotations the check reports not-evaluable (FR-010)."""
    ranking = _ranking(aligned_signals)
    result = evaluate_ranking(ranking, [])
    assert result.evaluable is False
    assert result.reason is not None


def test_band_not_evaluable_few_annotations(aligned_signals: Path) -> None:
    """Band separation is skipped when a band lacks enough annotations."""
    ranking = _ranking(aligned_signals)
    anns = [
        Annotation(item_id="item019", label="good", score=None, unit="file"),
        Annotation(item_id="item018", label="good", score=None, unit="file"),
    ]
    result = evaluate_ranking(ranking, anns)
    assert result.band_pairwise_agreement is None
    assert result.n_annotated_bottom == 0


def test_inverted_metric_negative_agreement(make_signal_table: Callable[..., Path]) -> None:
    """A metric anti-aligned with quality yields negative rank agreement."""
    n = 12
    path = make_signal_table({"q": [float(i) for i in range(n)]})
    ranking = _ranking(path)
    anns = [
        Annotation(item_id=f"item{i:03d}", label=quality_for_index(n - 1 - i, n), score=None, unit="file")
        for i in range(n)
    ]
    result = evaluate_ranking(ranking, anns)
    assert result.evaluable
    assert result.rank_agreement_spearman is not None and result.rank_agreement_spearman < 0
