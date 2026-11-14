"""Tests for the grouped linear probe."""

import numpy as np
import pytest

from senselab.utils.evaluation.metrics import balanced_accuracy
from senselab.utils.evaluation.splits import Split
from senselab.utils.tasks.linear_probe import LinearProbeResult, linear_probe

GRID = [0.001, 0.01, 0.1, 1, 10, 100]


def _data(n_groups: int = 24, rows: int = 5, seed: int = 0, separation: float = 6.0, noise: float = 1.0) -> tuple:
    """Binary features in `n_groups` groups, separated by `separation` on the first axis; `noise` scales the others."""
    rng = np.random.default_rng(seed)
    y = np.tile(np.array([0, 1, 0, 1, 1])[:rows], n_groups)
    g = np.repeat(np.arange(n_groups), rows)
    x = rng.normal(0, noise, (len(y), 4))
    x[:, 0] += separation * (y - 0.5)
    return x, y, g


def _splits(g: np.ndarray, n_groups: int = 24) -> list:
    """Three outer folds: groups i % 3 == k are the test fold; of the rest, groups i % 4 == 0 are `val`."""
    out = []
    for k in range(3):
        test = np.flatnonzero(g % 3 == k)
        rest = np.flatnonzero(g % 3 != k)
        inner = rest[(g[rest] % 4) == 0]
        train = np.setdiff1d(rest, inner)
        out.append(Split(name=f"fold{k}", partitions={"train": train, "val": inner, "test": test}))
    return out


def _run(x: np.ndarray, y: np.ndarray, g: np.ndarray, splits: list, seed: int = 42) -> LinearProbeResult:
    return linear_probe(x, y, g, splits, grid=GRID, class_weight="balanced", objective="balanced_accuracy", seed=seed)


def test_separable_data_gives_balanced_accuracy_one() -> None:
    """Well-separated classes are all predicted right on every test row."""
    x, y, g = _data()
    result = _run(x, y, g, _splits(g))
    assert sorted(result.rows.tolist()) == list(range(len(y)))
    assert balanced_accuracy(y[result.rows], result.predictions) == 1.0
    assert result.scores.shape == (len(y), 2) and len(result.selected) == 3
    assert all(c in GRID for c in result.selected)


def test_selection_uses_only_inner_validation_rows() -> None:
    """Flipping a test row's label changes neither that split's selected value nor its predictions.

    (The row is a training row of the other splits, so only split 0 is compared.)
    """
    x, y, g = _data(separation=1.0, seed=3)
    splits = _splits(g)
    base = _run(x, y, g, splits)
    y2 = y.copy()
    y2[splits[0].partitions["test"][0]] = 1 - y2[splits[0].partitions["test"][0]]
    flipped = _run(x, y2, g, splits)
    assert flipped.selected[0] == base.selected[0]
    first = base.split_index == 0
    assert np.array_equal(flipped.predictions[first], base.predictions[first])


def test_selection_follows_the_inner_validation_rows() -> None:
    """Flipping split 0's inner-validation labels turns its balanced accuracy into 1 - BA, so a different value wins."""
    x, y, g = _data(separation=0.3, seed=3)
    splits = _splits(g)
    base = _run(x, y, g, splits)
    y2 = y.copy()
    inner = splits[0].partitions["val"]
    y2[inner] = 1 - y2[inner]
    assert _run(x, y2, g, splits).selected[0] != base.selected[0]


def test_leaking_split_raises() -> None:
    """A split whose training rows and test rows share a group is refused."""
    x, y, g = _data()
    s = _splits(g)
    leak = np.concatenate([s[0].partitions["train"], s[0].partitions["test"][:1]])
    s[0] = Split(name="fold0", partitions={**s[0].partitions, "train": np.sort(leak)})
    with pytest.raises(ValueError, match="more than one partition"):
        _run(x, y, g, s)


def test_leaking_inner_validation_raises() -> None:
    """An inner-validation row from a test group is refused too."""
    x, y, g = _data()
    s = _splits(g)
    bad = np.concatenate([s[1].partitions["val"], s[1].partitions["test"][:1]])
    s[1] = Split(name="fold1", partitions={**s[1].partitions, "val": np.sort(bad)})
    with pytest.raises(ValueError, match="more than one partition"):
        _run(x, y, g, s)


def test_two_runs_are_identical() -> None:
    """The same inputs give the same rows, predictions, scores and selections."""
    x, y, g = _data(separation=1.0, seed=5)
    a, b = _run(x, y, g, _splits(g)), _run(x, y, g, _splits(g))
    assert np.array_equal(a.rows, b.rows) and np.array_equal(a.predictions, b.predictions)
    assert np.array_equal(a.scores, b.scores) and a.selected == b.selected


def test_unknown_objective_and_missing_partition_raise() -> None:
    """Only named objectives are accepted, and every split needs train, inner_val and test."""
    x, y, g = _data()
    with pytest.raises(ValueError, match="objective"):
        linear_probe(x, y, g, _splits(g), grid=GRID, class_weight="balanced", objective="f1_yes", seed=0)
    s = _splits(g)
    s[0] = Split(name="fold0", partitions={"train": s[0].partitions["train"], "test": s[0].partitions["test"]})
    with pytest.raises(ValueError, match="val"):
        _run(x, y, g, s)


def test_inner_partition_name_is_a_parameter() -> None:
    """A split whose validation partition is named otherwise works when that name is given, and not by default."""
    x, y, g = _data()
    renamed = [
        Split(
            name=s.name,
            partitions={"train": s.partitions["train"], "tune": s.partitions["val"], "test": s.partitions["test"]},
        )
        for s in _splits(g)
    ]
    with pytest.raises(ValueError, match="val"):
        _run(x, y, g, renamed)
    result = linear_probe(
        x,
        y,
        g,
        renamed,
        grid=GRID,
        class_weight="balanced",
        objective="balanced_accuracy",
        seed=42,
        inner_partition="tune",
    )
    assert result.selected == _run(x, y, g, _splits(g)).selected


def test_train_and_inner_validation_sharing_a_group_raises() -> None:
    """A group split between train and the inner validation partition is caught, not only a leak into test."""
    x, y, g = _data()
    s = _splits(g)
    train, inner = s[0].partitions["train"], s[0].partitions["val"]
    s[0] = Split(
        name="fold0",
        partitions={**s[0].partitions, "train": train[1:], "val": np.sort(np.concatenate([inner, train[:1]]))},
    )
    with pytest.raises(ValueError, match="fold0: .*subjects appear in more than one partition"):
        _run(x, y, g, s)


def test_empty_grid_raises() -> None:
    """A grid with no value cannot select anything."""
    x, y, g = _data()
    with pytest.raises(ValueError, match="grid"):
        linear_probe(x, y, g, _splits(g), grid=[], class_weight="balanced", objective="balanced_accuracy", seed=0)


def test_mismatched_input_lengths_raise() -> None:
    """Features, labels and groups must each have one entry per row."""
    x, y, g = _data()
    with pytest.raises(ValueError, match="one entry per row"):
        _run(x, y[:-1], g, _splits(g))
    with pytest.raises(ValueError, match="one entry per row"):
        _run(x, y, g[:-1], _splits(g))


def test_training_partition_missing_a_class_raises() -> None:
    """A training partition holding one class only is refused rather than fitted."""
    x, y, g = _data()
    s = _splits(g)
    train = s[0].partitions["train"]
    s[0] = Split(name="fold0", partitions={**s[0].partitions, "train": train[y[train] == 0]})
    with pytest.raises(ValueError, match="lack a class"):
        _run(x, y, g, s)
