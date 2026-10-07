"""Tests for the metrics vocabulary and validator."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from senselab.utils.reproduce.metrics import (
    CODES,
    DRIVER_KEYS,
    MetricsError,
    metric_address_error,
    metric_value,
    validate_metrics,
)
from senselab.utils.reproduce.spec import MetricsSchema


def _schema(**over: Any) -> MetricsSchema:  # noqa: ANN401
    data: dict[str, Any] = {
        "kind": "classification",
        "headline": ["balanced_accuracy", "mcc"],
        "headline_labels": ["sys_a", "sys_b"],
        "counts": ["n_systems", "n_flagged"],
        "per_class": ["pos", "neg"],
    }
    data.update(over)
    return MetricsSchema.model_validate(data)


def _split(**over: Any) -> dict[str, Any]:  # noqa: ANN401
    split: dict[str, Any] = {
        "protocol": "seen_subject",
        "unit": "subject",
        "stratify": "class",
        "folds": [{"n_groups": 12, "prevalence": {"pos": 0.4, "neg": 0.6}}, {"n_groups": 3, "prevalence": 0.5}],
        "invariant_ok": True,
    }
    split.update(over)
    return split


def _obj() -> dict[str, Any]:
    return {
        "headline": [
            {"name": "balanced_accuracy", "label": "sys_a", "value": 0.8, "floor": 0.5, "lift": 0.3, "n": 3},
            {"name": "balanced_accuracy", "label": "sys_b", "value": 0.7, "ci_low": 0.6, "ci_high": 0.8, "floor": 0.5},
            {"name": "mcc", "value": 0.4, "floor": 0.0, "n_folds": 3, "numdf": 3, "dendf": 60.0, "unit": "fraction"},
        ],
        "per_class": {
            "pos": {"recall": 1.0, "precision": 0.476, "f1": 0.645, "prevalence": 0.4},
            "neg": {"recall": 0.2, "prevalence": 0.6},
        },
        "confusion": {"tp": 10, "fp": 11, "fn": 0, "tn": 6},
        "counts": {"n_systems": 3, "n_flagged": 40},
        "split_report": _split(),
        "threshold_tuning": "validation",
        "n": 120,
        "n_folds": 3,
    }


def _codes(obj: Any, schema: MetricsSchema | None = None, **kw: Any) -> list[str]:  # noqa: ANN401
    clean, codes = validate_metrics(obj, schema or _schema(), **kw)
    if codes:
        assert clean is None
    return codes


def test_valid_output_passes_and_is_suppressed() -> None:
    """Valid output passes; small n, counts and confusion cells are suppressed, df and n_folds are not."""
    obj = _obj()
    clean, codes = validate_metrics(obj, _schema())
    assert codes == []
    assert clean is not None
    assert clean["confusion"] == {"tp": 10, "fp": 11, "fn": "<5", "tn": 6}
    assert clean["headline"][0]["n"] == "<5"
    assert clean["counts"] == {"n_systems": "<5", "n_flagged": 40}
    assert clean["n"] == 120
    assert clean["per_class"]["pos"]["recall"] == 1.0
    assert clean["headline"][2]["numdf"] == 3
    assert clean["headline"][2]["n_folds"] == 3
    assert clean["n_folds"] == 3
    assert clean["split_report"]["folds"][1]["n_groups"] == 3
    assert obj["confusion"]["fn"] == 0, "the input is not modified"


def test_min_cell_is_configurable() -> None:
    """The suppression threshold and its token follow ``min_cell``."""
    clean, codes = validate_metrics(_obj(), _schema(), min_cell=11)
    assert codes == []
    assert clean is not None
    assert clean["confusion"] == {"tp": "<11", "fp": 11, "fn": "<11", "tn": "<11"}


def test_already_suppressed_token_is_accepted() -> None:
    """A cell already replaced by the token passes again."""
    obj = _obj()
    obj["confusion"]["fn"] = "<5"
    assert _codes(obj) == []


@pytest.mark.parametrize(
    ("path", "value", "code"),
    [
        (("extra",), 1, "E_METRICS_UNKNOWN_KEY"),
        (("per_class", "pos", "auc"), 0.5, "E_METRICS_UNKNOWN_KEY"),
        (("split_report", "seed"), 1, "E_METRICS_UNKNOWN_KEY"),
        (("headline", 0, "value"), "0.8", "E_METRICS_NOT_NUMBER"),
        (("headline", 0, "value"), True, "E_METRICS_NOT_NUMBER"),
        (("headline", 0, "value"), float("nan"), "E_METRICS_NOT_NUMBER"),
        (("headline", 0, "value"), None, "E_METRICS_NOT_NUMBER"),
        (("per_class", "pos", "recall"), "1.0", "E_METRICS_NOT_NUMBER"),
        (("confusion", "fn"), "zero", "E_METRICS_NOT_NUMBER"),
        (("confusion", "fn"), 1.5, "E_METRICS_NOT_INT"),
        (("counts", "n_systems"), -1, "E_METRICS_NEGATIVE"),
        (("per_class", "child_17"), {"prevalence": 0.1}, "E_METRICS_UNDECLARED_LABEL"),
        (("headline", 0, "label"), "sys_z", "E_METRICS_UNDECLARED_LABEL"),
        (("headline", 0, "name"), "top1", "E_METRICS_UNDECLARED_HEADLINE"),
        (("headline", 0, "name"), "anything", "E_METRICS_UNDECLARED_HEADLINE"),
        (("headline", 0, "name"), ["mcc"], "E_METRICS_UNDECLARED_HEADLINE"),
        (("headline", 0, "label"), ["sys_a"], "E_METRICS_UNDECLARED_LABEL"),
        (("counts", "n_rows"), 9, "E_METRICS_UNDECLARED_COUNT"),
        (("headline", 0, "unit"), "percent", "E_METRICS_BAD_TOKEN"),
        (("headline",), {"name": "mcc"}, "E_METRICS_BAD_SHAPE"),
        (("per_class",), [], "E_METRICS_BAD_SHAPE"),
        (("headline", 0, "lift"), 0.2, "E_METRICS_LIFT_MISMATCH"),
        (("threshold_tuning",), "test", "E_METRICS_BAD_TUNING_SPLIT"),
    ],
)
def test_rejections(path: tuple[Any, ...], value: Any, code: str) -> None:  # noqa: ANN401
    """Each single change outside the vocabulary or schema gives its code."""
    obj = _obj()
    target = obj
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert code in _codes(obj)


def test_top_level_must_be_an_object() -> None:
    """Top level must be an object."""
    assert _codes([1, 2]) == ["E_METRICS_NOT_OBJECT"]


@pytest.mark.parametrize("key", sorted(DRIVER_KEYS))
@pytest.mark.parametrize("where", ["top", "headline", "per_class"])
def test_adapter_may_not_write_driver_keys(key: str, where: str) -> None:
    """Adapter may not write driver keys."""
    obj = _obj()
    value: Any = True if key == "pass" else 0.1
    {"top": obj, "headline": obj["headline"][0], "per_class": obj["per_class"]["pos"]}[where][key] = value
    assert _codes(obj) == ["E_METRICS_DRIVER_KEY"]


def test_driver_may_write_driver_keys() -> None:
    """Driver may write driver keys."""
    obj = _obj()
    obj["headline"][0].update({"target": 0.79, "tolerance": 0.01, "delta": 0.01, "pass": True})
    obj["pass"] = True
    assert _codes(obj, writer="driver") == []
    obj["pass"] = 1
    assert _codes(obj, writer="driver") == ["E_METRICS_NOT_BOOL"]
    obj["pass"] = True
    obj["headline"][0]["target"] = "0.79"
    assert _codes(obj, writer="driver") == ["E_METRICS_NOT_NUMBER"]


def test_classification_without_prevalence_is_rejected() -> None:
    """Classification without prevalence is rejected."""
    obj = _obj()
    del obj["per_class"]["neg"]["prevalence"]
    assert _codes(obj) == ["E_METRICS_NO_PREVALENCE"]
    obj = _obj()
    del obj["per_class"]
    assert _codes(obj) == ["E_METRICS_NO_PREVALENCE"]
    obj = _obj()
    del obj["per_class"]["neg"]
    assert _codes(obj) == ["E_METRICS_NO_PREVALENCE"]


def test_classification_without_floor_is_rejected() -> None:
    """Classification without floor is rejected."""
    obj = _obj()
    del obj["headline"][1]["floor"]
    assert _codes(obj) == ["E_METRICS_NO_FLOOR"]


def test_classification_without_tuning_split_is_rejected() -> None:
    """Classification without tuning split is rejected."""
    obj = _obj()
    del obj["threshold_tuning"]
    assert _codes(obj) == ["E_METRICS_NO_TUNING_SPLIT"]


def test_detection_kind_also_needs_prevalence_and_floor() -> None:
    """Detection kind also needs prevalence and floor."""
    schema = _schema(kind="detection", headline=["avg_map", "map_at"], headline_labels=[], counts=[])
    obj: dict[str, Any] = {
        "headline": [{"name": "avg_map", "value": 0.0595}, {"name": "map_at", "value": 0.0948, "floor": 0.02}],
        "per_class": {"pos": {"prevalence": 0.3}, "neg": {"prevalence": 0.7}},
        "threshold_tuning": "none",
    }
    assert _codes(copy.deepcopy(obj), schema) == ["E_METRICS_NO_FLOOR"]
    obj["headline"][0]["floor"] = 0.02
    assert _codes(obj, schema) == []


def test_statistic_kind_needs_no_floor_or_prevalence() -> None:
    """Statistic kind needs no floor or prevalence."""
    schema = MetricsSchema.model_validate({"kind": "statistic", "headline": ["f_stat"]})
    obj = {"headline": [{"name": "f_stat", "value": 4.2, "numdf": 3, "dendf": 60}, {"name": "f_stat", "value": 1.1}]}
    clean, codes = validate_metrics(obj, schema)
    assert codes == []
    assert clean == obj


def test_declared_headline_and_counts_must_be_present() -> None:
    """Declared headline and counts must be present."""
    obj = _obj()
    obj["headline"] = [h for h in obj["headline"] if h["name"] != "mcc"]
    assert _codes(obj) == ["E_METRICS_MISSING_HEADLINE"]
    obj = _obj()
    del obj["counts"]["n_flagged"]
    assert _codes(obj) == ["E_METRICS_MISSING_COUNT"]


def test_count_headline_must_be_an_integer() -> None:
    """Count headline must be an integer."""
    schema = MetricsSchema.model_validate({"kind": "count", "headline": ["count"]})
    assert _codes({"headline": [{"name": "count", "value": 2.5}]}, schema) == ["E_METRICS_NOT_INT"]
    assert _codes({"headline": [{"name": "count", "value": 2}]}, schema) == []


def test_split_report_seen_subject_passes() -> None:
    """Both protocols are valid, whatever the invariant outcome."""
    obj = _obj()
    obj["split_report"] = _split(protocol="seen_subject", invariant_ok=True)
    assert _codes(obj) == []
    obj["split_report"] = _split(protocol="unseen_subject", invariant_ok=False)
    assert _codes(obj) == []


def test_split_report_minimal() -> None:
    """A split report needs only the protocol and the invariant result."""
    obj = _obj()
    obj["split_report"] = {"protocol": "unseen_subject", "invariant_ok": True}
    assert _codes(obj) == []


@pytest.mark.parametrize(
    ("over", "drop", "code"),
    [
        ({}, "protocol", "E_SPLIT_NO_PROTOCOL"),
        ({"protocol": "within_subject_leak"}, None, "E_SPLIT_BAD_PROTOCOL"),
        ({}, "invariant_ok", "E_SPLIT_BAD_INVARIANT"),
        ({"invariant_ok": 1}, None, "E_SPLIT_BAD_INVARIANT"),
        ({"invariant_ok": "true"}, None, "E_SPLIT_BAD_INVARIANT"),
        ({"unit": "sub_0123"}, None, "E_METRICS_BAD_TOKEN"),
        ({"stratify": "Class Label"}, None, "E_METRICS_BAD_TOKEN"),
        ({"folds": [{"n_groups": 2.5}]}, None, "E_METRICS_NOT_INT"),
        ({"folds": [{"prevalence": {"child_9": 0.5}}]}, None, "E_METRICS_UNDECLARED_LABEL"),
        ({"folds": [{"prevalence": "half"}]}, None, "E_METRICS_NOT_NUMBER"),
        ({"folds": {"n_groups": 2}}, None, "E_METRICS_BAD_SHAPE"),
    ],
)
def test_split_report_rejections(over: dict[str, Any], drop: str | None, code: str) -> None:
    """A missing or unknown protocol, a non-bool invariant, free text and undeclared labels are refused."""
    obj = _obj()
    split = _split(**over)
    if drop:
        del split[drop]
    obj["split_report"] = split
    assert code in _codes(obj)


def test_rejection_never_echoes_content() -> None:
    """Rejection never echoes content."""
    obj = _obj()
    obj["participant_secret_key"] = "secret-value"
    obj["per_class"]["secret_label"] = {"prevalence": 0.1}
    _, codes = validate_metrics(obj, _schema())
    assert all(c in CODES for c in codes)
    assert not any("secret" in c for c in codes)


def test_codes_are_sorted_and_unique() -> None:
    """Codes are sorted and unique."""
    obj = _obj()
    obj["a"] = 1
    obj["b"] = 2
    obj["headline"][0]["value"] = "x"
    codes = _codes(obj)
    assert codes == sorted(set(codes))
    assert codes == ["E_METRICS_NOT_NUMBER", "E_METRICS_UNKNOWN_KEY"]


@pytest.mark.parametrize(
    ("address", "error"),
    [
        ("mcc", None),
        ("balanced_accuracy@sys_a", None),
        ("counts.n_systems", None),
        ("per_class.pos.recall", None),
        ("confusion.tp", None),
        ("top1", "E_METRICS_UNDECLARED_HEADLINE"),
        ("balanced_accuracy@sys_z", "E_METRICS_UNDECLARED_LABEL"),
        ("counts.n_rows", "E_METRICS_UNDECLARED_COUNT"),
        ("per_class.child.recall", "E_METRICS_UNDECLARED_LABEL"),
        ("per_class.pos.auc", "E_METRICS_BAD_ADDRESS"),
        ("confusion.xx", "E_METRICS_BAD_ADDRESS"),
        ("a.b.c.d", "E_METRICS_BAD_ADDRESS"),
    ],
)
def test_metric_address(address: str, error: str | None) -> None:
    """Addresses are checked against the schema's declared names and labels."""
    assert metric_address_error(address, _schema()) == error


def test_metric_value_resolves_and_refuses_ambiguity() -> None:
    """Metric value resolves and refuses ambiguity."""
    clean, _ = validate_metrics(_obj(), _schema())
    assert clean is not None
    assert metric_value(clean, "mcc") == 0.4
    assert metric_value(clean, "balanced_accuracy@sys_b") == 0.7
    assert metric_value(clean, "per_class.pos.precision") == 0.476
    assert metric_value(clean, "counts.n_flagged") == 40
    with pytest.raises(MetricsError) as info:
        metric_value(clean, "balanced_accuracy")
    assert info.value.code == "E_METRICS_AMBIGUOUS"
    with pytest.raises(MetricsError) as info:
        metric_value(clean, "confusion.fn")
    assert info.value.code == "E_METRICS_SUPPRESSED"
    with pytest.raises(MetricsError) as info:
        metric_value(clean, "per_class.neg.precision")
    assert info.value.code == "E_METRICS_ABSENT"
