"""Tests for metric scoring (FR-001/002/006/019)."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import pytest

from senselab.audio.workflows.ranking import io
from senselab.audio.workflows.ranking.metric import MetricError, score_items
from senselab.audio.workflows.ranking.types import Direction, MetricDefinition, SignalTerm


def _defn(terms: list[SignalTerm], direction: Direction = "higher_is_better") -> MetricDefinition:
    """Build a metric definition from terms."""
    return MetricDefinition(name="m", terms=terms, direction=direction)


def test_weighted_sum_combination(make_signal_table: Callable[..., Path]) -> None:
    """Combined score is the weighted sum of (transformed) signals."""
    path = make_signal_table({"a": [1.0, 2.0, 3.0], "b": [10.0, 0.0, 5.0]})
    table = io.load_signal_table(path)
    items = score_items(table, _defn([SignalTerm("a", 1.0), SignalTerm("b", 0.5)]))
    scores = {it.item_id: it.score for it in items}
    assert scores["item000"] == pytest.approx(1.0 + 5.0)
    assert all(it.status == "scored" for it in items)


def test_threshold_transform(make_signal_table: Callable[..., Path]) -> None:
    """Threshold transform maps a signal to 0/1 before weighting."""
    path = make_signal_table({"pii": [0.1, 0.9, 0.5]})
    table = io.load_signal_table(path)
    items = score_items(table, _defn([SignalTerm("pii", -1.0, transform="threshold", transform_params={"at": 0.5})]))
    scores = [next(it.score for it in items if it.item_id == f"item{n:03d}") for n in range(3)]
    assert scores == [0.0, -1.0, -1.0]


def test_missing_default_unscorable(make_signal_table: Callable[..., Path]) -> None:
    """A missing required signal makes the item unscorable by default (FR-006)."""
    path = make_signal_table({"a": [1.0, math.nan, 3.0]})
    table = io.load_signal_table(path)
    items = score_items(table, _defn([SignalTerm("a", 1.0)]))
    by_id = {it.item_id: it for it in items}
    assert by_id["item001"].status == "unscorable"
    assert by_id["item001"].reason == "missing:a"
    assert by_id["item000"].status == "scored"
    assert len(items) == 3


def test_missing_neutral_and_fill(make_signal_table: Callable[..., Path]) -> None:
    """`neutral` contributes 0; `fill:` substitutes a value before transform."""
    path = make_signal_table({"a": [math.nan, 2.0], "b": [5.0, math.nan]})
    table = io.load_signal_table(path)
    items = score_items(
        table, _defn([SignalTerm("a", 1.0, missing="neutral"), SignalTerm("b", 1.0, missing="fill:0.0")])
    )
    by_id = {it.item_id: it for it in items}
    assert by_id["item000"].status == "scored"
    assert by_id["item000"].score == pytest.approx(5.0)
    assert by_id["item001"].score == pytest.approx(2.0)


def test_unknown_signal_rejected(make_signal_table: Callable[..., Path]) -> None:
    """A metric referencing an absent signal is rejected (FR-019)."""
    path = make_signal_table({"a": [1.0, 2.0]})
    table = io.load_signal_table(path)
    with pytest.raises(MetricError, match="unknown signal"):
        score_items(table, _defn([SignalTerm("nonexistent", 1.0)]))


def test_empty_terms_rejected(make_signal_table: Callable[..., Path]) -> None:
    """A metric with no terms is rejected."""
    path = make_signal_table({"a": [1.0]})
    table = io.load_signal_table(path)
    with pytest.raises(MetricError):
        score_items(table, _defn([]))
