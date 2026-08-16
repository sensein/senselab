"""Tests for the Estimate type."""

import math

import pytest
from pydantic import ValidationError

from senselab.utils.data_structures import Estimate


def _est(**kw: object) -> Estimate:
    base = dict(raw=1.0, n_evidence=1, prior=0.5, prior_key="k", prior_weight=1.0, population="p")
    base.update(kw)
    return Estimate(**base)  # type: ignore[arg-type]


def test_no_evidence_collapses_to_the_prior() -> None:
    """With nothing observed, the published value must be exactly the named prior, not a blend."""
    e = Estimate(raw=None, n_evidence=0, prior=0.42, prior_key="k", prior_weight=2.0, population="p")
    assert e.value == 0.42


def test_a_raw_value_without_evidence_is_rejected() -> None:
    """F-156: a fabricated number and a measured one must not be the same object."""
    with pytest.raises(ValidationError):
        _est(raw=0.5, n_evidence=0)


def test_evidence_without_a_raw_value_is_rejected() -> None:
    """Evidence with nothing to shrink is the same defect as raw with no evidence, mirrored."""
    with pytest.raises(ValidationError):
        _est(raw=None, n_evidence=3)


def test_negative_evidence_is_rejected() -> None:
    """Evidence count is a count; a negative one is a bug upstream, not a valid estimate."""
    with pytest.raises(ValidationError):
        _est(n_evidence=-1)


def test_a_non_positive_prior_weight_is_rejected() -> None:
    """A zero or negative pseudo-count would make the prior inert or sign-flipping in the blend."""
    with pytest.raises(ValidationError):
        _est(prior_weight=0.0)


def test_an_empty_population_is_rejected() -> None:
    """An unstated population is how an adult-derived threshold reaches a child recording."""
    with pytest.raises(ValidationError):
        _est(population="  ")


def test_four_and_twenty_unanimous_sources_differ() -> None:
    """Statistical review N3: both published P = 1.000 before this type existed."""
    four = _est(raw=1.0, n_evidence=4, prior=0.5, prior_weight=1.0)
    twenty = _est(raw=1.0, n_evidence=20, prior=0.5, prior_weight=1.0)
    assert four.value != twenty.value
    assert four.value < twenty.value < 1.0


def test_more_evidence_moves_the_value_toward_raw() -> None:
    """Value must move monotonically toward raw as evidence accumulates, never overshoot it."""
    values = [_est(raw=1.0, n_evidence=n, prior=0.0, prior_weight=1.0).value for n in (1, 2, 8, 64)]
    assert values == sorted(values)
    assert values[-1] < 1.0


def test_shrinkage_reports_how_much_of_the_value_is_prior() -> None:
    """Shrinkage is the prior's literal share of the blend, not a free-floating confidence score."""
    assert _est(n_evidence=1, prior_weight=1.0).shrinkage == pytest.approx(0.5)
    assert _est(n_evidence=9, prior_weight=1.0).shrinkage == pytest.approx(0.1)
    assert Estimate(raw=None, n_evidence=0, prior=0.1, prior_key="k", prior_weight=3.0, population="p").shrinkage == 1.0


def test_value_is_not_settable() -> None:
    """A published value that disagrees with its own evidence is the defect this type prevents."""
    with pytest.raises((ValidationError, AttributeError, TypeError)):
        Estimate(  # type: ignore[call-arg]
            raw=1.0,
            n_evidence=1,
            prior=0.0,
            prior_key="k",
            prior_weight=1.0,
            population="p",
            value=0.999,
        )


def test_model_copy_with_update_cannot_bypass_the_raw_evidence_invariant() -> None:
    """Pydantic's own model_copy(update=...) skips validators; this type cannot allow that gap."""
    good = _est(raw=1.0, n_evidence=1)
    with pytest.raises(ValidationError):
        good.model_copy(update={"n_evidence": 0})


def test_model_copy_with_a_valid_update_still_works() -> None:
    """Overriding model_copy to re-validate must not break the ordinary, invariant-preserving case."""
    original = _est(population="adult-read-speech")
    copy = original.model_copy(update={"population": "child-read-speech"})
    assert copy.population == "child-read-speech"
    assert copy.value == original.value


def test_non_finite_raw_is_rejected() -> None:
    """A NaN or infinite raw would poison value silently; reject it at construction instead."""
    with pytest.raises(ValidationError):
        _est(raw=math.nan)
    with pytest.raises(ValidationError):
        _est(raw=math.inf)


def test_non_finite_prior_is_rejected() -> None:
    """The prior is what value collapses to at zero evidence; it must be an ordinary finite number."""
    with pytest.raises(ValidationError):
        _est(prior=math.nan, n_evidence=0, raw=None)
