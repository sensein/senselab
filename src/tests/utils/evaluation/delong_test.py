"""Tests for DeLong's test of two correlated ROC-AUCs."""

import json

import numpy as np
import pytest
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from senselab.utils.evaluation.delong import DeLongResult, delong_test


def _case(seed: int, n: int = 120, tie_step: float = 0.05) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Binary labels and two correlated scorers, rounded so ties occur."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.35).astype(int)
    shared = rng.normal(0, 1, n)
    a = np.round((0.9 * y + shared + rng.normal(0, 0.6, n)) / tie_step) * tie_step
    b = np.round((0.5 * y + shared + rng.normal(0, 0.9, n)) / tie_step) * tie_step
    return a, b, y


def _structural_reference(a: np.ndarray, b: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Variances and covariance from O(m*n) structural components (DeLong et al. 1988)."""

    def components(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, z = s[y == 1], s[y == 0]
        psi = (x[:, None] > z[None, :]) + 0.5 * (x[:, None] == z[None, :])
        return psi.mean(axis=1), psi.mean(axis=0)

    pa, na = components(a)
    pb, nb = components(b)
    m, n = len(pa), len(na)
    var_a = np.var(pa, ddof=1) / m + np.var(na, ddof=1) / n
    var_b = np.var(pb, ddof=1) / m + np.var(nb, ddof=1) / n
    cov = np.cov(pa, pb, ddof=1)[0, 1] / m + np.cov(na, nb, ddof=1)[0, 1] / n
    return float(var_a), float(var_b), float(cov)


def test_aurocs_match_scikit_learn_with_ties() -> None:
    """Midrank AUROCs equal roc_auc_score on 20 seeded cases with ties."""
    for seed in range(20):
        a, b, y = _case(seed)
        result = delong_test(a, b, y)
        assert abs(result.roc_auc_a - roc_auc_score(y, a)) <= 1e-12
        assert abs(result.roc_auc_b - roc_auc_score(y, b)) <= 1e-12


def test_variances_match_the_structural_components_reference() -> None:
    """The fast algorithm's variances and covariance equal the quadratic-time definition."""
    for seed in range(20):
        a, b, y = _case(seed)
        result = delong_test(a, b, y)
        var_a, var_b, cov = _structural_reference(a, b, y)
        assert abs(result.var_a - var_a) <= 1e-12
        assert abs(result.var_b - var_b) <= 1e-12
        assert abs(result.cov_ab - cov) <= 1e-12


def test_identical_scores_have_no_test_statistic() -> None:
    """With zero variance of the difference, z and p are undefined (None), never NaN."""
    a, _, y = _case(0)
    result = delong_test(a, a.copy(), y)
    assert result.z is None and result.p_value is None
    json.dumps(result.to_dict(), allow_nan=False)


def test_a_clearly_better_scorer_is_significant() -> None:
    """At n = 500 a strong scorer beats a weak one with p < 0.01."""
    rng = np.random.default_rng(1)
    y = (rng.random(500) < 0.4).astype(int)
    strong = 2.0 * y + rng.normal(0, 1, 500)
    weak = 0.3 * y + rng.normal(0, 1, 500)
    result = delong_test(strong, weak, y)
    assert result.z is not None and result.p_value is not None
    assert result.z > 0 and result.p_value < 0.01


def _origin_reimplementation(a: np.ndarray, b: np.ndarray, y: np.ndarray) -> tuple[float, ...]:
    """The earlier lab DeLong, rewritten here from its algorithm (loop midranks, Sun & Xu formulas)."""

    def midrank(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="stable")
        ordered = x[order]
        ranks = np.zeros(len(x))
        i = 0
        while i < len(x):
            j = i
            while j < len(x) and ordered[j] == ordered[i]:
                j += 1
            ranks[order[i:j]] = 0.5 * (i + j - 1) + 1
            i = j
        return ranks

    pos, neg = y == 1, y == 0
    m, n = int(pos.sum()), int(neg.sum())
    aucs, parts = [], []
    for s in (a, b):
        x, z = s[pos], s[neg]
        tx, tz_neg = midrank(x), midrank(z)
        tz = midrank(np.concatenate([x, z]))
        aucs.append((tz[:m].sum() / m - (m + 1) / 2.0) / n)
        parts.append(((tz[:m] - tx) / n, 1 - (tz[m:] - tz_neg) / m))
    sx = np.cov(np.vstack([parts[0][0], parts[1][0]]), bias=False)
    sy = np.cov(np.vstack([parts[0][1], parts[1][1]]), bias=False)
    var_a = sx[0, 0] / m + sy[0, 0] / n
    var_b = sx[1, 1] / m + sy[1, 1] / n
    cov = sx[0, 1] / m + sy[0, 1] / n
    z = (aucs[0] - aucs[1]) / np.sqrt(var_a + var_b - 2 * cov)
    return aucs[0], aucs[1], var_a, var_b, cov, z, 2.0 * (1 - norm.cdf(abs(z)))


def test_matches_the_earlier_implementation() -> None:
    """Every output equals the earlier implementation's within 1e-9."""
    for seed in range(20):
        a, b, y = _case(seed, n=200)
        ours = delong_test(a, b, y)
        theirs = _origin_reimplementation(a, b, y)
        got = (ours.roc_auc_a, ours.roc_auc_b, ours.var_a, ours.var_b, ours.cov_ab, ours.z, ours.p_value)
        for value, expected in zip(got, theirs):
            assert value is not None and abs(value - expected) <= 1e-9


def test_string_labels_with_a_positive_label() -> None:
    """Labels other than 0/1 work with an explicit positive label."""
    a, b, y = _case(2)
    named = np.where(y == 1, "pos", "neg")
    assert delong_test(a, b, named, positive_label="pos") == delong_test(a, b, y)


@pytest.mark.parametrize(
    ("a", "b", "y", "message"),
    [
        ([0.1, 0.2, 0.3], [0.1, 0.2], [0, 1, 0], "same length"),
        ([0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0, 0, 0, 1], "at least 2"),
        ([0.1, np.nan, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0, 1, 0, 1], "finite"),
    ],
    ids=["length", "one-positive", "nan"],
)
def test_inputs_are_validated(a: list, b: list, y: list, message: str) -> None:
    """Mismatched lengths, fewer than two of a class, and non-finite scores are refused."""
    with pytest.raises(ValueError, match=message):
        delong_test(np.array(a), np.array(b), np.array(y))


def test_result_is_a_plain_record() -> None:
    """The result serialises to strict JSON for the CLI's metrics file."""
    a, b, y = _case(3)
    result = delong_test(a, b, y)
    assert isinstance(result, DeLongResult)
    data = result.to_dict()
    assert set(data) == {
        "roc_auc_a",
        "roc_auc_b",
        "var_a",
        "var_b",
        "cov_ab",
        "z",
        "p_value",
        "n_positive",
        "n_negative",
    }
    json.dumps(data, allow_nan=False)


def test_no_variance_estimate_even_when_aucs_differ() -> None:
    """A perfect scorer against a constant one: AUCs 1 and 0.5, but both have zero variance, so z is None."""
    y = np.array([0, 0, 0, 1, 1, 1])
    result = delong_test(np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]), np.full(6, 0.5), y)
    assert (result.roc_auc_a, result.roc_auc_b) == (1.0, 0.5)
    assert result.z is None and result.p_value is None


@pytest.mark.parametrize(
    ("labels", "positive", "message"),
    [
        ([0, 1, 0, None, 1, 0], 1, "label id"),
        ([0, 1, 2, 1, 0, 1], 1, "exactly two"),
        (["a", "b", "a", "b", "a", "b"], 1, "positive_label 1 is not one of the labels"),
    ],
    ids=["missing", "three-values", "positive-absent"],
)
def test_labels_are_checked(labels: list, positive: object, message: str) -> None:
    """Missing labels, other than two values, or a positive label not among them are refused."""
    a = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3])
    with pytest.raises(ValueError, match=message):
        delong_test(a, a[::-1].copy(), np.array(labels, dtype=object), positive_label=positive)  # type: ignore[arg-type]
