"""Tests for movement tracking between versions (FR-020-023, SC-006/007)."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import pytest

from senselab.audio.workflows.ranking import io
from senselab.audio.workflows.ranking.metric import score_items
from senselab.audio.workflows.ranking.movement import compute_movement
from senselab.audio.workflows.ranking.rank import build_ranking
from senselab.audio.workflows.ranking.types import Annotation, Direction, MetricDefinition, Ranking, SignalTerm


def _ranking(path: Path, signal: str, version_id: str, direction: Direction = "higher_is_better") -> Ranking:
    """Build a ranking over ``signal`` for the given version id."""
    table = io.load_signal_table(path)
    items = score_items(table, MetricDefinition(name="m", terms=[SignalTerm(signal, 1.0)], direction=direction))
    return build_ranking(
        items, version_id=version_id, unit=table.unit, direction=direction, band_fraction=0.20, provenance={}
    )


def test_full_coverage_and_deltas(make_signal_table: Callable[..., Path]) -> None:
    """Movement accounts for 100% of items with correct rank deltas (SC-006)."""
    n = 10
    path = make_signal_table({"a": [float(i) for i in range(n)], "b": [float(n - i) for i in range(n)]})
    v1 = _ranking(path, "a", "v1")
    v2 = _ranking(path, "b", "v2")
    report = compute_movement(v1, v2)
    assert len(report.entries) == n
    assert all(e.delta_kind in ("moved", "unchanged") for e in report.entries)
    e0 = next(e for e in report.entries if e.item_id == "item000")
    assert e0.from_rank == n and e0.to_rank == 1


def test_band_summary_consistent_with_transitions(make_signal_table: Callable[..., Path]) -> None:
    """Band summary counts match the per-entry band transitions (SC-007)."""
    n = 10
    path = make_signal_table({"a": [float(i) for i in range(n)], "b": [float(n - i) for i in range(n)]})
    v1 = _ranking(path, "a", "v1")
    v2 = _ranking(path, "b", "v2")
    report = compute_movement(v1, v2)
    entered_top = sum(1 for e in report.entries if e.to_band == "top" and e.from_band != "top")
    assert report.band_summary["entered_top"] == entered_top


def test_became_unscorable_and_added_removed(make_signal_table: Callable[..., Path]) -> None:
    """Added / removed / became-unscorable items are all accounted for (FR-023)."""
    p1 = make_signal_table({"q": [1.0, 2.0, 3.0]}, item_ids=["a", "b", "c"], name="p1.parquet")
    p2 = make_signal_table({"q": [math.nan, 3.0, 4.0]}, item_ids=["b", "c", "d"], name="p2.parquet")
    v1 = _ranking(p1, "q", "v1")
    v2 = _ranking(p2, "q", "v2")
    report = compute_movement(v1, v2)
    assert "d" in report.added
    assert "a" in report.removed
    assert "b" in report.became_unscorable
    assert len(report.entries) == 4


def test_annotation_highlight(make_signal_table: Callable[..., Path]) -> None:
    """Annotated items are highlighted in the movement report (FR-022)."""
    path = make_signal_table({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]}, item_ids=["a", "b", "c"])
    v1 = _ranking(path, "a", "v1")
    v2 = _ranking(path, "b", "v2")
    anns = [Annotation(item_id="a", label="poor", score=None, unit="file")]
    report = compute_movement(v1, v2, anns)
    entry = next(e for e in report.entries if e.item_id == "a")
    assert entry.annotated is True and entry.annotation_label == "poor"


def test_unit_mismatch_rejected(make_signal_table: Callable[..., Path]) -> None:
    """Comparing rankings across units is rejected."""
    pf = make_signal_table({"q": [1.0, 2.0]}, unit="file", name="f.parquet")
    ps = make_signal_table({"q": [1.0, 2.0]}, unit="segment", name="s.parquet")
    v1 = _ranking(pf, "q", "v1")
    v2 = _ranking(ps, "q", "v2")
    with pytest.raises(ValueError, match="across units"):
        compute_movement(v1, v2)
