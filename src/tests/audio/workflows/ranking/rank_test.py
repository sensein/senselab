"""Tests for ranking assembly: coverage, deterministic ties, bands (FR-002/004/005)."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

from senselab.audio.workflows.ranking import io
from senselab.audio.workflows.ranking.metric import score_items
from senselab.audio.workflows.ranking.rank import assign_bands, build_ranking
from senselab.audio.workflows.ranking.types import Direction, MetricDefinition, Ranking, SignalTerm


def _rank(path: Path, direction: Direction = "higher_is_better", band_fraction: float = 0.20) -> Ranking:
    """Build a ranking from a one-signal table at ``path``."""
    table = io.load_signal_table(path)
    defn = MetricDefinition(name="m", terms=[SignalTerm("q", 1.0)], direction=direction)
    items = score_items(table, defn)
    return build_ranking(
        items, version_id="v1", unit=table.unit, direction=direction, band_fraction=band_fraction, provenance={}
    )


def test_full_coverage_and_dense_ranks(aligned_signals: Path) -> None:
    """Every item appears once; scored ranks are dense and unique (SC-002)."""
    ranking = _rank(aligned_signals)
    assert len(ranking.items) == 20
    ranks = [it.rank for it in ranking.items if it.rank is not None]
    assert sorted(ranks) == list(range(1, 21))


def test_higher_is_better_orders_top_first(aligned_signals: Path) -> None:
    """For higher-is-better, the highest signal is rank 1."""
    ranking = _rank(aligned_signals, direction="higher_is_better")
    top = next(it for it in ranking.items if it.rank == 1)
    assert top.item_id == "item019"


def test_lower_is_better_inverts(aligned_signals: Path) -> None:
    """For lower-is-better, the lowest signal is rank 1."""
    ranking = _rank(aligned_signals, direction="lower_is_better")
    top = next(it for it in ranking.items if it.rank == 1)
    assert top.item_id == "item000"


def test_deterministic_tiebreak_by_item_id(make_signal_table: Callable[..., Path]) -> None:
    """Equal scores are ordered by item_id ascending, deterministically."""
    path = make_signal_table({"q": [5.0, 5.0, 5.0]}, item_ids=["c", "a", "b"])
    ranking = _rank(path)
    ordered = [it.item_id for it in ranking.items if it.status == "scored"]
    assert ordered == ["a", "b", "c"]


def test_band_assignment_disjoint() -> None:
    """Top/bottom bands are disjoint with a non-empty middle at moderate N."""
    bands = assign_bands(10, 0.20)
    assert bands[:2] == ["top", "top"]
    assert bands[-2:] == ["bottom", "bottom"]
    assert "middle" in bands
    assert bands.count("top") == 2 and bands.count("bottom") == 2


def test_band_small_n_no_overlap() -> None:
    """Small N keeps bands disjoint; single item is middle."""
    assert assign_bands(3, 0.20) == ["top", "middle", "bottom"]
    assert assign_bands(1, 0.20) == ["middle"]


def test_unscorable_items_have_no_rank(make_signal_table: Callable[..., Path]) -> None:
    """Unscorable items carry no rank/band but are still present (FR-006)."""
    path = make_signal_table({"q": [1.0, math.nan, 3.0]})
    ranking = _rank(path)
    unscorable = [it for it in ranking.items if it.status == "unscorable"]
    assert len(unscorable) == 1
    assert unscorable[0].rank is None and unscorable[0].band is None
    assert ranking.n_scored == 2 and ranking.n_unscorable == 1
