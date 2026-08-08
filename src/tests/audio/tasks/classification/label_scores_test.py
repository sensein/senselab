"""Scene-classifier windows carry `[{label: score}, ...]`, not parallel arrays.

Two aligned lists are one `[:top_k]` slice away from disagreeing, and nothing in the shape says
they must stay the same length or the same order. A list of single-entry dicts cannot drift: each
score is attached to the label it belongs to.
"""

from __future__ import annotations

import pytest

from senselab.audio.tasks.classification.label_scores import label_scores, top_label


def test_a_window_reports_each_label_with_its_own_score() -> None:
    """The pairing is structural rather than positional."""
    window = {"label_scores": [{"Speech": 0.9}, {"Music": 0.05}]}
    assert label_scores(window) == [{"Speech": 0.9}, {"Music": 0.05}]


def test_order_is_preserved_because_rank_is_information() -> None:
    """These arrive sorted by score; re-sorting or de-duplicating would discard the ranking."""
    window = {"label_scores": [{"Speech": 0.9}, {"Music": 0.5}, {"Noise": 0.1}]}
    assert [next(iter(d)) for d in label_scores(window)] == ["Speech", "Music", "Noise"]


def test_a_window_with_no_classification_yields_no_labels() -> None:
    """Absent is not empty-with-confidence: a window nothing classified makes no claim."""
    assert label_scores({}) == []


def test_top_label_returns_the_highest_scoring_pair() -> None:
    """The common consumer question, asked once rather than re-derived at every call site."""
    window = {"label_scores": [{"Speech": 0.9}, {"Music": 0.5}]}
    assert top_label(window) == ("Speech", 0.9)


def test_top_label_of_an_unclassified_window_is_none() -> None:
    """No label is a legitimate answer; a caller must be able to tell it from a low-scoring one."""
    assert top_label({}) is None


def test_malformed_entries_are_skipped_rather_than_guessed_at() -> None:
    """A multi-key or empty dict is not a label/score pair, and inventing one would be worse."""
    window = {"label_scores": [{"Speech": 0.9}, {}, {"a": 1.0, "b": 2.0}, "nonsense"]}
    assert label_scores(window) == [{"Speech": 0.9}]


def test_scores_are_floats_so_downstream_arithmetic_is_safe() -> None:
    """Parquet round-trips can hand back numpy scalars; consumers sum and compare these."""
    import numpy as np

    window = {"label_scores": [{"Speech": np.float32(0.25)}]}
    assert label_scores(window)[0]["Speech"] == pytest.approx(0.25)
    assert isinstance(label_scores(window)[0]["Speech"], float)
