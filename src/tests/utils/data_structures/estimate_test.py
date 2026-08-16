"""Tests for the Estimate type."""

import pytest
from pydantic import ValidationError

from senselab.utils.data_structures import Estimate


def _est(**kw: object) -> Estimate:
    base = dict(raw=1.0, n_evidence=1, prior=0.5, prior_key="k", prior_weight=1.0, population="p")
    base.update(kw)
    return Estimate(**base)  # type: ignore[arg-type]


def test_no_evidence_collapses_to_the_prior() -> None:
    e = Estimate(raw=None, n_evidence=0, prior=0.42, prior_key="k", prior_weight=2.0, population="p")
    assert e.value == 0.42


def test_a_raw_value_without_evidence_is_rejected() -> None:
    """F-156: a fabricated number and a measured one must not be the same object."""
    with pytest.raises(ValidationError):
        _est(raw=0.5, n_evidence=0)


def test_evidence_without_a_raw_value_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _est(raw=None, n_evidence=3)


def test_negative_evidence_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _est(n_evidence=-1)


def test_a_non_positive_prior_weight_is_rejected() -> None:
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
    values = [_est(raw=1.0, n_evidence=n, prior=0.0, prior_weight=1.0).value for n in (1, 2, 8, 64)]
    assert values == sorted(values)
    assert values[-1] < 1.0


def test_shrinkage_reports_how_much_of_the_value_is_prior() -> None:
    assert _est(n_evidence=1, prior_weight=1.0).shrinkage == pytest.approx(0.5)
    assert _est(n_evidence=9, prior_weight=1.0).shrinkage == pytest.approx(0.1)
    assert Estimate(
        raw=None, n_evidence=0, prior=0.1, prior_key="k", prior_weight=3.0, population="p"
    ).shrinkage == 1.0


def test_value_is_not_settable() -> None:
    """A published value that disagrees with its own evidence is the defect this type prevents."""
    with pytest.raises((ValidationError, AttributeError, TypeError)):
        Estimate(  # type: ignore[call-arg]
            raw=1.0, n_evidence=1, prior=0.0, prior_key="k",
            prior_weight=1.0, population="p", value=0.999,
        )
