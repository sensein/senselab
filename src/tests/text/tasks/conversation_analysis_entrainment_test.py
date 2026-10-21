"""Entrainment on constructed series with known proximity, convergence and synchrony."""

import numpy as np
import pandas as pd
import pytest

from senselab.text.tasks.conversation_analysis import analyze_conversation, compute_entrainment, turn_entrainment


def alternating(a_values: list, b_values: list) -> pd.DataFrame:
    """A at t = 0, 2, 4, ...; B at t = 1, 3, 5, ...; each B turn answers the A turn before it."""
    rows = []
    for i, (a, b) in enumerate(zip(a_values, b_values)):
        rows.append({"speaker": "A", "start": 2.0 * i, "f0": a})
        rows.append({"speaker": "B", "start": 2.0 * i + 1, "f0": b})
    return pd.DataFrame(rows)


def by_target(results: list, target: str):  # noqa: ANN201
    """The result for one responding speaker."""
    return next(r for r in results if r.target_speaker == target)


def test_perfect_mimicry_proximity_and_synchrony() -> None:
    """B repeats A's preceding value: adjacent diff 0, non-adjacent mean 2.0, synchrony r = 1."""
    results = compute_entrainment(alternating([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]), ["f0"], n_surrogates=500)
    b = by_target(results, "B")
    assert b.n_pairs == 5
    assert b.adjacent_diff_mean == pytest.approx(0.0)
    assert b.nonadjacent_diff_mean == pytest.approx(2.0)
    assert b.proximity == pytest.approx(2.0)
    assert b.synchrony_r == pytest.approx(1.0)
    assert b.convergence_r is None  # constant |diff|
    assert b.proximity_p is not None and b.proximity_p < 0.05
    assert b.synchrony_p is not None and b.synchrony_p < 0.05
    assert b.synchrony_surrogate_mean is not None and abs(b.synchrony_surrogate_mean) < 0.3
    assert {r.target_speaker for r in results} == {"all", "A", "B"}
    assert by_target(results, "all").n_pairs == 9


def test_linear_convergence() -> None:
    """|diff| shrinking linearly over time gives convergence r = -1."""
    results = compute_entrainment(alternating([0, 0, 0, 0, 0], [4, 3, 2, 1, 0]), ["f0"], n_surrogates=200)
    b = by_target(results, "B")
    assert b.convergence_r == pytest.approx(-1.0)
    assert b.adjacent_diff_mean == pytest.approx(2.0)
    assert b.synchrony_r is None  # partner series is constant


def test_anti_synchrony() -> None:
    """B mirrors A around 6: synchrony r = -1."""
    results = compute_entrainment(alternating([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]), ["f0"], n_surrogates=0)
    b = by_target(results, "B")
    assert b.synchrony_r == pytest.approx(-1.0)
    assert b.synchrony_p is None  # no surrogates requested


def test_known_correlation_matches_numpy() -> None:
    """Synchrony equals numpy's Pearson r on the adjacent pairs."""
    rng = np.random.default_rng(3)
    a = rng.normal(size=20)
    b = 0.6 * a + rng.normal(scale=0.5, size=20)
    results = compute_entrainment(alternating(list(a), list(b)), ["f0"], n_surrogates=0)
    assert by_target(results, "B").synchrony_r == pytest.approx(float(np.corrcoef(a, b)[0, 1]))


def test_missing_values_dropped() -> None:
    """A NaN turn is dropped before pairing."""
    frame = alternating([1, 2, 3], [1, 2, 3])
    frame.loc[2, "f0"] = np.nan
    results = compute_entrainment(frame, ["f0"], n_surrogates=0)
    assert by_target(results, "all").n_pairs == 3


def test_entrainment_through_analyze_conversation() -> None:
    """Per-turn features keyed by turn id feed entrainment over floor-holding turns."""
    words = []
    for i in range(6):
        speaker = "A" if i % 2 == 0 else "B"
        for k, tok in enumerate(["some", "words", "here", "now"]):
            words.append({"speaker": speaker, "start": 2.0 * i + 0.3 * k, "end": 2.0 * i + 0.3 * (k + 1), "text": tok})
    result = analyze_conversation(words)
    assert len(result.turns) == 6
    features = pd.DataFrame({"turn_id": [t.id for t in result.turns], "f0": [100, 100, 120, 120, 140, 140]})
    with_features = analyze_conversation(words, turn_features=features, entrainment_surrogates=50)
    assert with_features.entrainment is not None
    b = by_target(with_features.entrainment, "B")
    assert b.synchrony_r == pytest.approx(1.0)
    assert b.adjacent_diff_mean == pytest.approx(0.0)
    assert "entrainment" in with_features.to_dataframes()
    with pytest.raises(ValueError, match="turn_id"):
        analyze_conversation(words, turn_features=features.drop(columns="turn_id"))


def test_public_turn_entrainment_matches_analyze_conversation() -> None:
    """``turn_entrainment`` on an analysis' turns equals the entrainment ``analyze_conversation`` reports."""
    words = []
    for i in range(6):
        speaker = "A" if i % 2 == 0 else "B"
        for k, tok in enumerate(["some", "words", "here", "now"]):
            words.append({"speaker": speaker, "start": 2.0 * i + 0.3 * k, "end": 2.0 * i + 0.3 * (k + 1), "text": tok})
    result = analyze_conversation(words)
    features = pd.DataFrame({"turn_id": [t.id for t in result.turns], "f0": [100, 100, 120, 120, 140, 140]})
    direct = turn_entrainment(result.turns, features, n_surrogates=50, seed=3)
    via_api = analyze_conversation(words, turn_features=features, entrainment_surrogates=50, seed=3).entrainment
    assert via_api is not None
    assert [r.model_dump() for r in direct] == [r.model_dump() for r in via_api]
    assert by_target(direct, "B").synchrony_r == pytest.approx(1.0)
    with pytest.raises(ValueError, match="turn_id"):
        turn_entrainment(result.turns, features.drop(columns="turn_id"))
