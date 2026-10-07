"""Tests for validation-only threshold tuning."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef

from senselab.utils.evaluation import thresholds
from senselab.utils.evaluation.classification import balanced_accuracy, score_classification
from senselab.utils.evaluation.splits import nested_grouped_splits
from senselab.utils.evaluation.thresholds import ThresholdChoice, load_threshold_grid, tune_threshold
from tests.utils.evaluation.conftest import Cohort


def _separable() -> tuple[np.ndarray, np.ndarray]:
    """Negatives all score 0.1 and positives 0.9: every threshold in (0.1, 0.9] is equally perfect."""
    return np.array(["neg"] * 10 + ["pos"] * 10), np.array([0.1] * 10 + [0.9] * 10)


def _ninety_ten(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """900 negatives scoring around 0.35 and 100 positives around 0.65."""
    rng = np.random.default_rng(seed)
    y = np.array(["neg"] * 900 + ["pos"] * 100)
    scores = np.clip(np.where(y == "pos", rng.normal(0.65, 0.15, 1000), rng.normal(0.35, 0.15, 1000)), 0, 1)
    return y, scores


def _predict(scores: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(scores >= threshold, "pos", "neg")


def test_default_grid_is_the_ported_convention() -> None:
    """The packaged grid is 0.05 to 0.95 in steps of 0.05, ties to 0.5, rounded to 4 places."""
    grid = load_threshold_grid()
    assert grid.values == tuple(round(0.05 * i, 4) for i in range(1, 20))
    assert (grid.tie_break, grid.round) == ("closest_to_0.5", 4)
    raw = yaml.safe_load(thresholds.GRID_PATH.read_text())
    assert set(raw) == {"start", "stop", "step", "tie_break", "round"}


def test_ties_go_to_the_threshold_nearest_one_half() -> None:
    """When many thresholds score the same, the one nearest 0.5 wins."""
    y, scores = _separable()
    choice = tune_threshold(y, scores, split_name="val", positive_label="pos")
    assert choice.threshold == 0.5 and choice.value == 1.0 and choice.objective == "balanced_accuracy"


def test_balanced_accuracy_objective_uses_the_headline_function() -> None:
    """The BA objective's value is classification.balanced_accuracy at the chosen threshold."""
    y, scores = _ninety_ten(0)
    choice = tune_threshold(y, scores, split_name="val", positive_label="pos")
    assert choice.value == balanced_accuracy(y, _predict(scores, choice.threshold))
    assert all(balanced_accuracy(y, _predict(scores, t)) <= choice.value for t in choice.grid)


def test_ba_tuning_beats_f1_tuning_on_balanced_accuracy() -> None:
    """On a 90/10 case the BA-tuned threshold gives higher BA on fresh data than the F1-tuned one."""
    y_val, s_val = _ninety_ten(1)
    y_test, s_test = _ninety_ten(2)
    by_ba = tune_threshold(y_val, s_val, split_name="val", positive_label="pos")
    by_f1 = tune_threshold(y_val, s_val, split_name="val", positive_label="pos", objective="f1")
    assert by_ba.threshold != by_f1.threshold
    ba_at = {c.objective: balanced_accuracy(y_test, _predict(s_test, c.threshold)) for c in (by_ba, by_f1)}
    assert ba_at["balanced_accuracy"] > ba_at["f1"]


def test_f1_tuned_report_keeps_the_ba_headline() -> None:
    """A report built with an F1-tuned threshold records the objective; BA is computed at that threshold."""
    y_val, s_val = _ninety_ten(1)
    y_test, s_test = _ninety_ten(2)
    choice = tune_threshold(y_val, s_val, split_name="val", positive_label="pos", objective="f1")
    report = score_classification(
        y_test,
        None,
        labels=["neg", "pos"],
        protocol="unseen_subject",
        split_name="test",
        n_dropped=0,
        split_report=None,
        y_score=s_test,
        positive_label="pos",
        threshold=choice,
    )
    assert report.threshold is not None and report.threshold["objective"] == "f1"
    assert report.threshold["split_name"] == "val" and report.threshold_tuning == "validation"
    assert report.headline == "balanced_accuracy"
    assert report.headline_value == balanced_accuracy(y_test, _predict(s_test, choice.threshold))


def test_mcc_objective_is_available_and_recorded() -> None:
    """MCC is an explicit objective; an unknown objective is refused."""
    y, scores = _ninety_ten(0)
    assert tune_threshold(y, scores, split_name="val", positive_label="pos", objective="mcc").objective == "mcc"
    with pytest.raises(ValueError, match="objective"):
        tune_threshold(y, scores, split_name="val", positive_label="pos", objective="accuracy")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "name",
    [
        "test",
        "Test_fold2",
        " outer-TEST ",
        "holdout",
        "fold3-heldout",
        "test_free_val",
        "test1",
        "testset",
        "fold0test",
        "outerTest",
        "innertest",
        "hold-out",
        "held_out",
        "HoldOut",
    ],
)
def test_tuning_on_a_test_split_is_refused(name: str) -> None:
    """A split name that marks held-out data (test, holdout, heldout, in joined or split forms) is refused."""
    y, scores = _separable()
    with pytest.raises(ValueError, match="test"):
        tune_threshold(y, scores, split_name=name, positive_label="pos")


def test_scoring_on_the_tuning_split_is_refused() -> None:
    """The tuned threshold cannot score the split it was tuned on."""
    y, scores = _separable()
    choice = tune_threshold(y, scores, split_name="val", positive_label="pos")
    with pytest.raises(ValueError, match="tuned on the split"):
        score_classification(
            y,
            None,
            labels=["neg", "pos"],
            protocol="unseen_subject",
            split_name="val",
            n_dropped=0,
            split_report=None,
            y_score=scores,
            positive_label="pos",
            threshold=choice,
        )


def test_tuning_needs_both_classes_and_finite_scores() -> None:
    """One class, NaN scores or mismatched lengths are refused."""
    with pytest.raises(ValueError, match="two"):
        tune_threshold(np.array(["neg"] * 4), np.array([0.1, 0.2, 0.3, 0.4]), split_name="val", positive_label="pos")
    with pytest.raises(ValueError, match="finite"):
        tune_threshold(np.array(["neg", "pos"]), np.array([0.1, np.nan]), split_name="val", positive_label="pos")
    with pytest.raises(ValueError, match="length"):
        tune_threshold(np.array(["neg", "pos"]), np.array([0.1]), split_name="val", positive_label="pos")


def test_nested_tuning_never_touches_outer_test_rows(imbalanced_cohort: Cohort) -> None:
    """Thresholds tuned on inner validation rows never use an outer-test row."""
    c = imbalanced_cohort
    rng = np.random.default_rng(0)
    scores = np.clip(0.3 + 0.3 * c.labels + rng.normal(0, 0.15, len(c.labels)), 0, 1)
    y = np.where(c.labels == 1, "pos", "neg")
    nested, _, _ = nested_grouped_splits(
        c.labels,
        c.subjects,
        protocol="unseen_subject",
        seed=0,
        outer={"scheme": "stratified_group_kfold", "n_splits": 5},
        inner={"scheme": "stratified_group_kfold", "n_splits": 3},
    )
    for i, (outer, inners) in enumerate(nested):
        tuning_rows = np.unique(np.concatenate([inner.partitions["test"] for inner in inners]))
        assert not set(tuning_rows.tolist()) & set(outer.partitions["test"].tolist())
        choice = tune_threshold(y[tuning_rows], scores[tuning_rows], split_name=f"inner_val_{i}", positive_label="pos")
        assert isinstance(choice, ThresholdChoice) and choice.split_name == f"inner_val_{i}"


def test_the_grid_comes_from_the_data_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Editing a temp copy of threshold_grid.yaml changes the chosen threshold."""
    y, scores = _separable()
    assert tune_threshold(y, scores, split_name="val", positive_label="pos").threshold == 0.5
    copy = tmp_path / "threshold_grid.yaml"
    shutil.copy(thresholds.GRID_PATH, copy)
    raw = yaml.safe_load(copy.read_text())
    raw.update({"start": 0.6, "stop": 0.9, "step": 0.1})
    copy.write_text(yaml.safe_dump(raw))
    monkeypatch.setattr(thresholds, "GRID_PATH", copy)
    choice = tune_threshold(y, scores, split_name="val", positive_label="pos")
    assert choice.threshold == 0.6 and choice.grid == (0.6, 0.7, 0.8, 0.9)


# ── Review fixes ──────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["latest", "contest", "pretest", "attestation", "val", "inner_val_3", "fold2_val"])
def test_split_names_that_merely_contain_test_are_accepted(name: str) -> None:
    """The guard matches whole tokens, so "latest" or "pretest" is not a test split."""
    y, scores = _separable()
    assert tune_threshold(y, scores, split_name=name, positive_label="pos").split_name == name


@pytest.mark.parametrize(
    ("y", "s", "expected"),
    [
        (["pos", "pos", "neg", "neg"], [0.32, 0.9, 0.1, 0.67], 0.3),
        (["pos", "neg", "pos", "neg"], [0.97, 0.03, 0.06, 0.94], 0.05),
    ],
    ids=["0.3-vs-0.7", "0.05-vs-0.95"],
)
def test_equidistant_ties_go_to_the_lower_threshold(y: list, s: list, expected: float) -> None:
    """Thresholds equally far from 0.5 tie exactly; the lower one wins, whatever the float error."""
    assert tune_threshold(np.array(y), np.array(s), split_name="val", positive_label="pos").threshold == expected


def test_a_score_equal_to_the_threshold_is_positive() -> None:
    """Prediction is positive at or above the threshold."""
    choice = tune_threshold(np.array(["neg", "pos"]), np.array([0.45, 0.5]), split_name="val", positive_label="pos")
    assert choice.threshold == 0.5


def _origin_rule(y: np.ndarray, s: np.ndarray) -> float:
    """The earlier lab code's rule, reimplemented: unrounded arange grid, strict improvement, round at the end."""
    best_t, best = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.05):
        value = balanced_accuracy_score(y == "pos", s >= t)
        if value > best or (value == best and abs(t - 0.5) < abs(best_t - 0.5)):
            best, best_t = value, float(t)
    return round(best_t, 4)


def test_choices_agree_with_the_origin_rule_off_the_grid() -> None:
    """On 200 small seeded cases (many ties) with no score on a grid value, the choice equals the origin's."""
    for seed in range(200):
        rng = np.random.default_rng(seed)
        y = np.array(["pos", "neg"] * 4)
        s = rng.random(8)
        assert tune_threshold(y, s, split_name="val", positive_label="pos").threshold == _origin_rule(y, s)


def test_a_score_on_a_grid_value_is_where_the_origin_differs() -> None:
    """The origin's unrounded grid holds 0.6500000000000001, so it treats a score of 0.65 as below 0.65."""
    y = np.array(["pos", "neg", "pos", "neg", "pos", "neg", "pos", "neg"])
    s = np.array([0.69, 0.64, 0.13, 0.11, 0.65, 0.85, 0.2, 0.22])
    assert tune_threshold(y, s, split_name="val", positive_label="pos").threshold == 0.65
    assert _origin_rule(y, s) == 0.5


def test_f1_and_mcc_values_match_scikit_learn() -> None:
    """The f1 objective is the positive-class F1; the mcc objective is matthews_corrcoef."""
    y, scores = _ninety_ten(3)
    by_f1 = tune_threshold(y, scores, split_name="val", positive_label="pos", objective="f1")
    pred = _predict(scores, by_f1.threshold)
    assert by_f1.value == pytest.approx(f1_score(y, pred, pos_label="pos"))
    by_mcc = tune_threshold(y, scores, split_name="val", positive_label="pos", objective="mcc")
    assert by_mcc.value == pytest.approx(matthews_corrcoef(y, _predict(scores, by_mcc.threshold)))


@pytest.mark.parametrize(
    "fields",
    [{"start": 0.0}, {"stop": 1.0}, {"step": 0.001, "round": 2}],
    ids=["start-0", "stop-1", "duplicates-after-rounding"],
)
def test_bad_grids_are_refused(tmp_path: Path, fields: dict) -> None:
    """Grids must lie inside (0, 1) and be strictly increasing after rounding."""
    raw = yaml.safe_load(thresholds.GRID_PATH.read_text())
    raw.update(fields)
    path = tmp_path / "grid.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="grid"):
        load_threshold_grid(path)


def test_the_grid_file_is_not_gitignored() -> None:
    """The packaged grid must not match .gitignore, or the wheel build leaves it out."""
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    repo = Path(thresholds.__file__).resolve().parents[4]
    if not (repo / ".gitignore").exists():
        pytest.skip("not running from a source checkout")
    done = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", str(thresholds.GRID_PATH.resolve())], cwd=repo, check=False
    )
    assert done.returncode == 1, "threshold_grid.yaml is gitignored"
