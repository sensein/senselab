"""Confidence, variability, and uncertainty are three quantities with three estimators.

The codebase had been using "uncertainty" for all of them, which is why a max-doubt fold and a
Shannon entropy and a max-minus-min spread all ended up in a column of that name. They are not
interchangeable: only one is a probability, only one is a dispersion, and only one decomposes.
"""

from __future__ import annotations

import math

import pytest

from senselab.audio.workflows.audio_analysis.statistics import (
    confidence,
    entropy_uncertainty,
    epistemic_uncertainty,
    variability,
)

# ── confidence: a probability-like belief in a proposition ─────────────


def test_confidence_is_the_weighted_share_of_signals_asserting_the_proposition() -> None:
    """P(proposition) estimated from votes, so it lives in [0, 1] and is calibratable."""
    assert confidence([True, True, False, False]) == pytest.approx(0.5)
    assert confidence([True, True, True]) == pytest.approx(1.0)


def test_confidence_respects_signal_weights() -> None:
    """A weighted vote is still a probability; the weights renormalize."""
    assert confidence([True, False], weights=[3.0, 1.0]) == pytest.approx(0.75)


def test_confidence_of_no_votes_is_undefined_not_zero() -> None:
    """Zero is the confident claim "definitely not"; no votes is "we did not look"."""
    assert confidence([]) is None


# ── variability: dispersion of repeated measurements ──────────────────


def test_variability_is_a_dispersion_not_a_probability() -> None:
    """Standard deviation of the sample, so identical measurements give exactly zero."""
    assert variability([0.5, 0.5, 0.5]) == pytest.approx(0.0)
    assert variability([0.0, 1.0]) == pytest.approx(0.5)


def test_variability_of_one_measurement_is_undefined() -> None:
    """Dispersion needs at least two measurements; zero would claim perfect agreement."""
    assert variability([0.5]) is None


def test_variability_is_not_bounded_by_one_in_general() -> None:
    """A dispersion carries the units of the quantity.

    Forcing it into [0, 1] would make it a different statistic and invite reading it as a
    probability.
    """
    assert variability([0.0, 10.0]) == pytest.approx(5.0)


# ── uncertainty: entropy of a distribution over outcomes ──────────────


def test_uncertainty_is_normalized_shannon_entropy() -> None:
    """An even split over two outcomes is maximal uncertainty; a certain outcome is zero."""
    assert entropy_uncertainty({"a": 0.5, "b": 0.5}) == pytest.approx(1.0)
    assert entropy_uncertainty({"a": 1.0}) == pytest.approx(0.0)


def test_uncertainty_normalization_accounts_for_the_number_of_outcomes() -> None:
    """Raw entropy in nats is unbounded above, so it cannot be compared across axes.

    Dividing by log(k) puts an even split at 1.0 whether there are two outcomes or five.
    """
    assert entropy_uncertainty({k: 0.25 for k in "abcd"}) == pytest.approx(1.0)
    assert entropy_uncertainty({k: 0.2 for k in "abcde"}) == pytest.approx(1.0)


def test_uncertainty_of_an_unnormalized_distribution_normalizes_first() -> None:
    """Vote masses are not probabilities until divided by their total."""
    assert entropy_uncertainty({"a": 3.0, "b": 3.0}) == pytest.approx(1.0)


def test_uncertainty_of_a_single_outcome_space_is_zero_not_undefined() -> None:
    """With one possible answer there is nothing to be uncertain between."""
    assert entropy_uncertainty({"only": 1.0}) == pytest.approx(0.0)


def test_uncertainty_of_an_empty_distribution_is_undefined() -> None:
    """Nothing was observed, so no entropy is defined over it."""
    assert entropy_uncertainty({}) is None


# ── decomposition: which part more data could remove ──────────────────


def test_epistemic_uncertainty_is_total_minus_mean_individual() -> None:
    """The standard mutual-information decomposition.

    Total entropy of the ensemble mean minus the mean of each signal's own entropy. What is
    left is disagreement *between* signals — the reducible part, which is what tells the
    adaptive loop that another measurement could help.
    """
    # Two signals, each internally certain, disagreeing with each other: all epistemic.
    total, epistemic = epistemic_uncertainty([{"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}])
    assert total == pytest.approx(1.0)
    assert epistemic == pytest.approx(1.0)


def test_shared_internal_doubt_is_not_epistemic() -> None:
    """Signals that agree but are each unsure face irreducible noise, not disagreement.

    Reporting that as reducible would send the loop off to gather evidence that cannot help.
    """
    total, epistemic = epistemic_uncertainty([{"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}])
    assert total == pytest.approx(1.0)
    assert epistemic == pytest.approx(0.0)


def test_epistemic_uncertainty_never_exceeds_total() -> None:
    """A mathematical property of the decomposition; violating it means a sign error."""
    total, epistemic = epistemic_uncertainty([{"a": 0.9, "b": 0.1}, {"a": 0.6, "b": 0.4}])
    assert 0.0 <= epistemic <= total <= 1.0


def test_a_single_signal_has_no_epistemic_uncertainty() -> None:
    """Disagreement needs at least two parties."""
    _total, epistemic = epistemic_uncertainty([{"a": 0.5, "b": 0.5}])
    assert epistemic == pytest.approx(0.0)


def test_no_signals_yields_no_decomposition() -> None:
    """Nothing observed, so neither part of the decomposition is defined."""
    assert epistemic_uncertainty([]) == (None, None)
