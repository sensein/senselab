"""Tests for DAG ordering and the cross-step expectation check."""

from __future__ import annotations

import itertools
from typing import Any

import pytest

from senselab.utils.reproduce.dag import DagError, check_expect_sources, toposort
from senselab.utils.reproduce.spec import StepSpec

COMMIT = "0123456789abcdef0123456789abcdef01234567"
SHA = "ab" * 32
CLASSIFICATION = {
    "kind": "classification",
    "headline": ["top1", "macro_recall", "mcc"],
    "headline_labels": ["sys_a"],
    "counts": ["n_clips"],
    "per_class": ["walk", "run"],
}


def _step(name: str, needs: list[str] | None = None, **over: Any) -> StepSpec:  # noqa: ANN401
    data: dict[str, Any] = {
        "name": name,
        "repo": "https://git.example/r.git",
        "commit": COMMIT,
        "kind": "python",
        "env": {
            "python": "3.12",
            "python_exact": "3.12.11",
            "torch": None,
            "body": {"lock": "b.lock", "lock_sha256": SHA, "source": "uv-export", "export_cmd": ["uv"]},
            "exclude_newer": "2026-01-01",
        },
        "argv": ["{venv}/bin/python", "run.py"],
        "needs": needs or [],
        "resources": {"profile": "cpu"},
        "adapter": "a.py",
        "hf_weights": [],
        "metrics_schema": CLASSIFICATION,
    }
    data.update(over)
    return StepSpec.model_validate(data)


def _names(steps: list[StepSpec]) -> list[str]:
    return [s.name for s in steps]


def test_toposort_puts_needs_first() -> None:
    """Every step follows all of its needs."""
    steps = [_step("d", ["b", "c"]), _step("b", ["a"]), _step("c", ["a"]), _step("a")]
    order = _names(toposort(steps))
    assert order == ["a", "b", "c", "d"]


def test_toposort_is_deterministic_under_any_input_order() -> None:
    """Ties break by step name, so every permutation of the input gives the same order."""
    steps = [_step("z"), _step("m", ["z"]), _step("a"), _step("k", ["a"]), _step("b", ["m", "k"])]
    orders = {tuple(_names(toposort(list(p)))) for p in itertools.permutations(steps)}
    assert orders == {("a", "k", "z", "m", "b")}


def test_toposort_refuses_a_cycle() -> None:
    """A cycle raises, naming only the steps on it."""
    steps = [_step("a", ["c"]), _step("b", ["a"]), _step("c", ["b"]), _step("free")]
    with pytest.raises(DagError) as info:
        toposort(steps)
    assert info.value.code == "E_DAG_CYCLE"
    assert info.value.steps == ("a", "b", "c")


def test_toposort_refuses_an_unknown_need() -> None:
    """A need naming no step in the set raises."""
    with pytest.raises(DagError) as info:
        toposort([_step("a"), _step("b", ["a", "ghost"])])
    assert info.value.code == "E_DAG_UNKNOWN_NEED"
    assert info.value.steps == ("b",)


def test_toposort_refuses_duplicate_names() -> None:
    """Two steps with one name raise."""
    with pytest.raises(DagError) as info:
        toposort([_step("a"), _step("a")])
    assert info.value.code == "E_DAG_DUPLICATE_STEP"


def _expecting(target: str, metric: str, from_metric: str, **expect: Any) -> StepSpec:  # noqa: ANN401
    return _step(
        "score",
        [target],
        expect=[{"metric": metric, "from_step": {"step": target, "metric": from_metric}, "kind": "exact", **expect}],
    )


ACCEPTED = {
    "same-headline": ("top1", "top1", {}),
    "labelled-headline": ("macro_recall", "top1@sys_a", {}),
    "rate-from-per-class": ("top1", "per_class.walk.recall", {}),
    "count-from-count": ("counts.n_clips", "counts.n_clips", {}),
    "mcc-from-mcc": ("mcc", "mcc", {"unit": "pp"}),
}


@pytest.mark.parametrize("case", list(ACCEPTED), ids=list(ACCEPTED))
def test_from_step_metric_accepted(case: str) -> None:
    """A from_step metric the upstream schema declares, of the same quantity, is accepted."""
    metric, from_metric, expect = ACCEPTED[case]
    steps = [_step("target"), _expecting("target", metric, from_metric, **expect)]
    check_expect_sources(steps)
    assert _names(toposort(steps)) == ["target", "score"]


REFUSED = {
    "undeclared-headline": ("top1", "auroc", {}, "E_DAG_FROM_STEP_METRIC"),
    "undeclared-label": ("top1", "top1@sys_b", {}, "E_DAG_FROM_STEP_METRIC"),
    "undeclared-count": ("counts.n_clips", "counts.n_rows", {}, "E_DAG_FROM_STEP_METRIC"),
    "undeclared-class": ("top1", "per_class.jump.recall", {}, "E_DAG_FROM_STEP_METRIC"),
    "bad-address": ("top1", "per_class.walk", {}, "E_DAG_FROM_STEP_METRIC"),
    "rate-from-count": ("top1", "counts.n_clips", {}, "E_DAG_FROM_STEP_UNIT"),
    "count-from-rate": ("counts.n_clips", "top1", {}, "E_DAG_FROM_STEP_UNIT"),
    "rate-from-mcc": ("top1", "mcc", {}, "E_DAG_FROM_STEP_UNIT"),
    "unit-on-a-count": ("counts.n_clips", "counts.n_clips", {"unit": "pp"}, "E_DAG_FROM_STEP_UNIT"),
}


@pytest.mark.parametrize("case", list(REFUSED), ids=list(REFUSED))
def test_from_step_metric_refused(case: str) -> None:
    """An undeclared upstream metric, or one of another quantity, is refused by the check and by toposort."""
    metric, from_metric, expect, code = REFUSED[case]
    steps = [_step("target"), _expecting("target", metric, from_metric, **expect)]
    for call in (check_expect_sources, toposort):
        with pytest.raises(DagError) as info:
            call(steps)
        assert info.value.code == code
        assert info.value.steps == ("score",)


def test_from_step_metric_against_a_statistic_step() -> None:
    """An F statistic target needs an F statistic expectation."""
    target = _step("target", metrics_schema={"kind": "statistic", "headline": ["f_stat"]})
    good = _step(
        "score",
        ["target"],
        metrics_schema={"kind": "statistic", "headline": ["f_stat"]},
        expect=[{"metric": "f_stat", "from_step": {"step": "target", "metric": "f_stat"}, "kind": "exact"}],
    )
    check_expect_sources([target, good])
    with pytest.raises(DagError) as info:
        check_expect_sources([target, _expecting("target", "top1", "f_stat")])
    assert info.value.code == "E_DAG_FROM_STEP_UNIT"
    with pytest.raises(DagError) as info:
        check_expect_sources([target, _expecting("target", "top1", "top1")])
    assert info.value.code == "E_DAG_FROM_STEP_METRIC"


def test_tolerance_from_steps_must_declare_the_metric() -> None:
    """Calibration steps named by tolerance_from must declare the expectation's metric."""
    other = {"kind": "statistic", "headline": ["f_stat"]}
    expect = {
        "metric": "top1",
        "from_step": {"step": "target", "metric": "top1"},
        "tolerance_from": {"steps": ["s1", "s2"], "rule": "max(a, k*sd)", "a": 0.5, "k": 2},
        "unit": "pp",
        "kind": "abs",
    }
    score = _step("score", ["target", "s1", "s2"], expect=[expect])
    check_expect_sources([_step("target"), _step("s1"), _step("s2"), score])
    with pytest.raises(DagError) as info:
        check_expect_sources([_step("target"), _step("s1"), _step("s2", metrics_schema=other), score])
    assert info.value.code == "E_DAG_TOLERANCE_METRIC"
    assert info.value.steps == ("score",)
