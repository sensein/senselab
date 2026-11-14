"""Tests for Holm-Bonferroni adjustment."""

import pytest

from senselab.utils.evaluation.multiplicity import holm


def test_holm_textbook() -> None:
    """The textbook case gives adjusted d 0.02, a 0.03, c 0.06, b 0.06 at alpha 0.05; d and a are rejected only."""
    result = holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005}, 0.05)
    assert {k: round(v.adjusted_p, 12) for k, v in result.items()} == {"d": 0.02, "a": 0.03, "c": 0.06, "b": 0.06}
    assert {k for k, v in result.items() if v.reject} == {"d", "a"}
    assert {k: v.rank for k, v in result.items()} == {"d": 1, "a": 2, "c": 3, "b": 4}


def test_holm_adjusted_p_is_monotone_and_capped_at_one() -> None:
    """Adjusted values never decrease with rank and never exceed 1."""
    result = holm({"x": 0.9, "y": 0.5, "z": 0.001}, 0.05)
    ordered = [result[k].adjusted_p for k in sorted(result, key=lambda k: result[k].rank)]
    assert ordered == sorted(ordered) and max(ordered) == 1.0 and result["z"].reject


def test_holm_rejects_bad_input() -> None:
    """An empty set, a p outside [0, 1] and an alpha outside (0, 1) are refused."""
    for bad, alpha in (({}, 0.05), ({"a": 1.2}, 0.05), ({"a": 0.1}, 0.0)):
        with pytest.raises(ValueError):
            holm(bad, alpha)
