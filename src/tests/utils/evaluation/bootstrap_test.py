"""Tests for the cluster bootstrap."""

import inspect
import json
from dataclasses import dataclass

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score

from senselab.utils.evaluation import bootstrap
from senselab.utils.evaluation.bootstrap import (
    cluster_bootstrap_ci,
    cluster_bootstrap_indices,
    legacy_cluster_bootstrap,
    paired_cluster_bootstrap_delta,
)
from senselab.utils.evaluation.classification import score_classification
from senselab.utils.evaluation.metrics import balanced_accuracy


def _cohort(seed: int = 0, n_groups: int = 30) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grouped binary data: 3-12 rows per group, a group-level positive rate, informative scores."""
    rng = np.random.default_rng(seed)
    y, s, g = [], [], []
    for i in range(n_groups):
        n = int(rng.integers(3, 13))
        rate = rng.choice([0.1, 0.3, 0.6])
        labels = (rng.random(n) < rate).astype(int)
        y += labels.tolist()
        s += np.clip(0.35 + 0.3 * labels + rng.normal(0, 0.15, n), 0, 1).tolist()
        g += [f"g{(i * 7) % n_groups:02d}"] * n
    return np.array(y), np.array(s), np.array(g)


def _ba(y: np.ndarray, pred: np.ndarray) -> float:
    return balanced_accuracy(y, pred)


def test_deterministic_for_a_fixed_seed() -> None:
    """The same seed gives the same interval; another seed gives another."""
    y, s, g = _cohort()
    pred = (s >= 0.5).astype(int)
    a = cluster_bootstrap_ci({"ba": _ba}, y, pred, g, n_resamples=300, seed=1)["ba"]
    b = cluster_bootstrap_ci({"ba": _ba}, y, pred, g, n_resamples=300, seed=1)["ba"]
    c = cluster_bootstrap_ci({"ba": _ba}, y, pred, g, n_resamples=300, seed=2)["ba"]
    assert (a.low, a.high) == (b.low, b.high) and (a.low, a.high) != (c.low, c.high)
    assert a.method == "cluster_bootstrap_percentile" and a.n_groups == 30 and a.seed == 1


def test_interval_contains_the_point_estimate() -> None:
    """The percentile interval brackets the full-data value."""
    y, s, g = _cohort()
    pred = (s >= 0.5).astype(int)
    ci = cluster_bootstrap_ci({"ba": _ba}, y, pred, g, n_resamples=500, seed=0)["ba"]
    assert ci.estimate == _ba(y, pred)
    assert ci.low is not None and ci.high is not None
    assert ci.low <= ci.estimate <= ci.high


def test_single_class_resamples_are_skipped_and_counted() -> None:
    """A resample whose truth holds one class is skipped; skipped + used = requested."""
    y = np.array([0, 0, 0, 0, 1, 1])
    g = np.array(["a", "a", "b", "b", "c", "c"])
    ci = cluster_bootstrap_ci({"ba": _ba}, y, y, g, n_resamples=400, seed=0)["ba"]
    assert ci.n_skipped > 0 and ci.n_used + ci.n_skipped == 400


def test_one_row_per_group_matches_a_row_bootstrap() -> None:
    """With one row per group, the resampled rows are the row bootstrap's index draws."""
    groups = np.array([f"r{i}" for i in range(9)])
    rng = np.random.default_rng(5)
    expected = [rng.integers(0, 9, size=9) for _ in range(4)]
    got = list(cluster_bootstrap_indices(groups, n_resamples=4, seed=5))
    assert all(np.array_equal(a, b) for a, b in zip(got, expected))


def test_groups_are_ordered_by_first_appearance() -> None:
    """Group order is first appearance, rows within a group stay in input order."""
    groups = np.array(["b", "a", "b", "c", "a"])
    first = next(cluster_bootstrap_indices(groups, n_resamples=1, seed=0))
    draws = np.random.default_rng(0).integers(0, 3, size=3)
    members = {0: [0, 2], 1: [1, 4], 2: [3]}
    assert first.tolist() == [row for d in draws for row in members[int(d)]]


class _RecordingRng:
    """Wraps a Generator and counts calls to ``integers``."""

    def __init__(self, seed: int) -> None:
        self.inner = np.random.default_rng(seed)
        self.calls = 0

    def integers(self, *args: object, **kwargs: object) -> np.ndarray:
        self.calls += 1
        return self.inner.integers(*args, **kwargs)  # type: ignore[call-overload]


def test_legacy_mode_draws_exactly_once_per_resample(monkeypatch: pytest.MonkeyPatch) -> None:
    """One integers() call per requested resample, skipped ones included."""
    recorders: list[_RecordingRng] = []

    def make(seed: int) -> _RecordingRng:
        recorders.append(_RecordingRng(seed))
        return recorders[-1]

    monkeypatch.setattr(bootstrap, "_make_rng", make)
    y = np.array([0, 0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
    g = np.array(["a", "a", "b", "b", "c", "c"])
    result = legacy_cluster_bootstrap(y, s, g, threshold=0.5, n_resamples=250, seed=42)
    assert len(recorders) == 1 and recorders[0].calls == 250
    assert result["balanced_accuracy"].n_skipped > 0


def test_paired_delta_of_a_model_with_itself_is_zero() -> None:
    """Comparing a model with itself gives a zero difference on every resample."""
    y, s, g = _cohort()
    pred = (s >= 0.5).astype(int)
    delta = paired_cluster_bootstrap_delta(_ba, y, pred, pred, g, n_resamples=200, seed=0)
    assert (delta.estimate, delta.low, delta.high) == (0.0, 0.0, 0.0)


def test_paired_delta_sign_follows_the_better_model() -> None:
    """A clearly better model has a positive difference interval against a worse one."""
    y, s, g = _cohort()
    good = (s >= 0.5).astype(int)
    rng = np.random.default_rng(1)
    poor = np.where(rng.random(len(y)) < 0.4, 1 - good, good)
    delta = paired_cluster_bootstrap_delta(_ba, y, good, poor, g, n_resamples=300, seed=0)
    assert delta.low is not None and delta.estimate is not None
    assert delta.estimate > 0 and delta.low > 0


def _origin_reimplementation(y: np.ndarray, s: np.ndarray, g: np.ndarray, threshold: float, b: int, seed: int) -> dict:
    """The earlier lab bootstrap, rewritten here from its algorithm for the parity check."""
    rng = np.random.default_rng(seed)
    keys = g.astype(str)
    order = list(dict.fromkeys(keys.tolist()))
    labels = [y[keys == k] for k in order]
    scores = [s[keys == k] for k in order]
    values: dict[str, list[float]] = {"f1": [], "balanced_accuracy": [], "auroc": [], "auprc": []}
    for _ in range(b):
        pick = rng.integers(0, len(order), size=len(order))
        yy = np.concatenate([labels[i] for i in pick])
        ss = np.concatenate([scores[i] for i in pick])
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        pp = (ss >= threshold).astype(int)
        values["f1"].append(f1_score(yy, pp, zero_division=0))
        values["balanced_accuracy"].append(balanced_accuracy_score(yy, pp))
        values["auroc"].append(roc_auc_score(yy, ss))
        values["auprc"].append(average_precision_score(yy, ss))
    return {k: (np.percentile(v, 2.5), np.percentile(v, 97.5)) for k, v in values.items()}


@pytest.mark.parametrize("n_resamples", [300, pytest.param(2000, marks=pytest.mark.slow)])
def test_legacy_mode_matches_the_earlier_bootstrap_exactly(n_resamples: int) -> None:
    """The legacy mode reproduces the earlier bootstrap's intervals exactly."""
    y, s, g = _cohort(seed=3, n_groups=40)
    ours = legacy_cluster_bootstrap(y, s, g, threshold=0.5, n_resamples=n_resamples, seed=42)
    theirs = _origin_reimplementation(y, s, g, 0.5, n_resamples, 42)
    for metric, (low, high) in theirs.items():
        assert (ours[metric].low, ours[metric].high) == (low, high)


# ── score_classification(groups=...) ───────────────────────────────────


def _report(with_groups: bool, **kwargs: object) -> tuple:
    y, s, g = _cohort()
    labels = np.where(y == 1, "pos", "neg")
    fields: dict = {
        "labels": ["neg", "pos"],
        "protocol": "unseen_subject",
        "split_name": "test",
        "n_dropped": 0,
        "split_report": None,
        "y_score": s,
        "positive_label": "pos",
    }
    fields.update(kwargs)
    pred = np.where(s >= 0.5, "pos", "neg")
    return score_classification(labels, pred, groups=g if with_groups else None, **fields), labels, pred


def test_report_intervals_with_groups() -> None:
    """Groups give intervals for BA, macro-F1, MCC, ROC-AUC and AP, each recording how it was made."""
    report, labels, pred = _report(True, n_resamples=300, seed=7)
    assert report.ci is not None
    assert set(report.ci) == {"balanced_accuracy", "f1_macro", "mcc", "roc_auc", "average_precision"}
    ba = report.ci["balanced_accuracy"]
    assert ba.estimate == report.headline_value == balanced_accuracy(labels, pred)
    assert (ba.n_resamples, ba.seed, ba.n_groups, ba.method) == (300, 7, 30, "cluster_bootstrap_percentile")
    assert ba.low <= ba.estimate <= ba.high
    assert report.to_dict()["ci"]["balanced_accuracy"]["n_resamples"] == 300


def test_report_without_groups_has_no_interval() -> None:
    """No groups, no interval."""
    report, _, _ = _report(False)
    assert report.ci is None


def test_default_is_two_thousand_resamples() -> None:
    """B = 2000 by default, for the report and the bootstrap functions."""
    for fn in (score_classification, cluster_bootstrap_ci, legacy_cluster_bootstrap, paired_cluster_bootstrap_delta):
        assert inspect.signature(fn).parameters["n_resamples"].default == 2000


@pytest.mark.slow
def test_report_with_default_resamples() -> None:
    """A report with groups and the default B carries 2000-resample intervals."""
    report, _, _ = _report(True)
    assert report.ci is not None and report.ci["mcc"].n_resamples == 2000


def test_resamples_missing_a_class_average_over_present_classes() -> None:
    """Perfect three-class predictions keep BA at 1 in every resample, even resamples missing a class."""
    labels = ["a", "b", "c"]
    y = np.array(["a"] * 6 + ["b"] * 6 + ["c"] * 2)
    g = np.array([f"g{i}" for i in range(14)])
    report = score_classification(
        y,
        y,
        labels=labels,
        protocol="unseen_subject",
        split_name="test",
        n_dropped=0,
        split_report=None,
        groups=g,
        n_resamples=500,
        seed=0,
    )
    assert report.ci is not None
    assert (report.ci["balanced_accuracy"].low, report.ci["balanced_accuracy"].high) == (1.0, 1.0)


def test_groups_are_validated() -> None:
    """Groups need one id per row, none missing, and at least two groups."""
    y = np.array(["neg", "pos", "neg", "pos"])
    common: dict = {
        "labels": ["neg", "pos"],
        "protocol": "unseen_subject",
        "split_name": "test",
        "n_dropped": 0,
        "split_report": None,
    }
    with pytest.raises(ValueError, match="one entry per row"):
        score_classification(y, y, groups=np.array(["a", "b"]), **common)
    with pytest.raises(ValueError, match="group id"):
        score_classification(y, y, groups=["a", None, "b", "b"], **common)
    with pytest.raises(ValueError, match="at least 2 groups"):
        score_classification(y, y, groups=np.array(["a"] * 4), **common)


@dataclass(frozen=True)
class _Threshold:
    threshold: float
    objective: str
    split_name: str


def test_threshold_stays_fixed_inside_resamples() -> None:
    """With a tuned threshold, resamples score the same fixed predictions; nothing is re-tuned."""
    y, s, g = _cohort()
    labels = np.where(y == 1, "pos", "neg")
    report = score_classification(
        labels,
        None,
        labels=["neg", "pos"],
        protocol="unseen_subject",
        split_name="test",
        n_dropped=0,
        split_report=None,
        y_score=s,
        positive_label="pos",
        threshold=_Threshold(0.6, "balanced_accuracy", "val"),
        groups=g,
        n_resamples=300,
        seed=0,
    )
    fixed = np.where(s >= 0.6, "pos", "neg")
    expected = cluster_bootstrap_ci({"ba": balanced_accuracy}, labels, fixed, g, n_resamples=300, seed=0)["ba"]
    assert report.ci is not None
    assert (report.ci["balanced_accuracy"].low, report.ci["balanced_accuracy"].high) == (expected.low, expected.high)


# ── Review fixes ──────────────────────────────────────────────────────


def test_numpy_stream_golden_values() -> None:
    """Pins the generator stream the resampling depends on; NEP 19 does not promise it across releases."""
    assert np.random.default_rng(42).integers(0, 40, 40)[:8].tolist() == [3, 30, 26, 17, 17, 34, 3, 27]


def test_heavy_skipping_flags_the_interval_unreliable() -> None:
    """More than 5 % skipped resamples marks the interval unreliable; to_dict carries the flag."""
    y = np.array([0, 0, 0, 0, 1, 1])
    g = np.array(["a", "a", "b", "b", "c", "c"])
    skip_heavy = cluster_bootstrap_ci({"ba": _ba}, y, y, g, n_resamples=400, seed=0)["ba"]
    assert skip_heavy.n_skipped / 400 > 0.05 and skip_heavy.unreliable
    y2, s2, g2 = _cohort()
    calm = cluster_bootstrap_ci({"ba": _ba}, y2, (s2 >= 0.5).astype(int), g2, n_resamples=300, seed=0)["ba"]
    assert calm.n_skipped == 0 and not calm.unreliable


def test_group_ids_are_told_apart_as_strings() -> None:
    """1 and "1" are one group, as resampling treats them, so this is refused as a single group."""
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="at least 2 groups"):
        cluster_bootstrap_ci({"ba": _ba}, y, y, np.array([1, "1", 1, "1"], dtype=object), n_resamples=10, seed=0)


def test_single_class_split_is_json_safe() -> None:
    """With one class in the scored split, undefined values are None, never NaN, and to_dict is strict JSON."""
    y = np.array(["neg"] * 12)
    s = np.linspace(0.1, 0.9, 12)
    g = np.repeat(["a", "b", "c", "d"], 3)
    report = score_classification(
        y,
        np.where(s >= 0.5, "pos", "neg"),
        labels=["neg", "pos"],
        protocol="unseen_subject",
        split_name="test",
        n_dropped=0,
        split_report=None,
        y_score=s,
        positive_label="pos",
        groups=g,
        n_resamples=50,
        seed=0,
    )
    assert report.roc_auc is None and report.average_precision is None
    assert report.ci is not None
    assert report.ci["roc_auc"].estimate is None and report.ci["roc_auc"].low is None
    assert report.ci["balanced_accuracy"].unreliable and report.ci["balanced_accuracy"].n_skipped == 50
    json.dumps(report.to_dict(), allow_nan=False)


def test_balanced_accuracy_has_one_public_home() -> None:
    """The headline function lives in metrics; classification does not re-export it."""
    from senselab.utils.evaluation import classification, metrics

    assert callable(metrics.balanced_accuracy) and callable(metrics.legacy_macro_recall)
    assert not hasattr(classification, "balanced_accuracy") and not hasattr(classification, "legacy_macro_recall")
