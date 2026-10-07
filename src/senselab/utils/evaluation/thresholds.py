"""Decision thresholds tuned on a validation split.

`tune_threshold` picks, from the grid in `data/threshold_grid.yaml`, the threshold that maximises an
objective on validation data. The default objective is balanced accuracy, computed with
`classification.balanced_accuracy` (the report's headline function); `f1` (positive class) and `mcc`
are explicit, non-default choices, and the returned `ThresholdChoice` records which was used. A
`ThresholdChoice` can be passed to `score_classification(threshold=...)`, which refuses it on the split
it was tuned on and still reports balanced accuracy as the headline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union, get_args

import numpy as np
import yaml
from sklearn.metrics import f1_score, matthews_corrcoef

from senselab.utils.evaluation.classification import balanced_accuracy

Objective = Literal["balanced_accuracy", "f1", "mcc"]
OBJECTIVES: Tuple[str, ...] = get_args(Objective)
TieBreak = Literal["closest_to_0.5"]
GRID_PATH = Path(__file__).parent / "data" / "threshold_grid.yaml"
GRID_KEYS = frozenset({"start", "stop", "step", "tie_break", "round"})


@dataclass(frozen=True)
class ThresholdGrid:
    """Candidate thresholds and how ties between them are broken.

    Attributes:
        values: The candidate thresholds, ascending, each rounded to `round` places.
        tie_break: `closest_to_0.5`: among equally good thresholds, the one nearest 0.5, then the lower.
        round: Decimal places of the candidates.
    """

    values: Tuple[float, ...]
    tie_break: TieBreak
    round: int


@dataclass(frozen=True)
class ThresholdChoice:
    """A threshold chosen on a validation split.

    Attributes:
        threshold: The chosen threshold; the positive label is predicted at or above it.
        objective: The objective it maximises.
        value: The objective's value at the threshold, on the tuning split.
        split_name: The split it was tuned on.
        grid: The candidate thresholds that were searched.
    """

    threshold: float
    objective: Objective
    value: float
    split_name: str
    grid: Tuple[float, ...]


def load_threshold_grid(path: Optional[Union[str, Path]] = None) -> ThresholdGrid:
    """Load a threshold grid from YAML.

    Args:
        path: The YAML file; defaults to the packaged `data/threshold_grid.yaml` (`GRID_PATH`).

    Returns:
        The grid.

    Raises:
        ValueError: If the file does not hold exactly the keys start, stop, step, tie_break and round,
            or they do not describe a non-empty ascending grid in (0, 1).
    """
    raw: Dict[str, Any] = yaml.safe_load(Path(path if path is not None else GRID_PATH).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != GRID_KEYS:
        raise ValueError(f"a threshold grid holds exactly the keys {sorted(GRID_KEYS)}")
    if raw["tie_break"] not in get_args(TieBreak):
        raise ValueError(f"tie_break must be one of {get_args(TieBreak)}")
    start, stop, step, places = float(raw["start"]), float(raw["stop"]), float(raw["step"]), int(raw["round"])
    if not (0 < start <= stop < 1 and step > 0):
        raise ValueError("a threshold grid needs 0 < start <= stop < 1 and step > 0")
    count = int(math.floor((stop - start) / step + 1e-9)) + 1
    values = tuple(round(start + i * step, places) for i in range(count))
    return ThresholdGrid(values=values, tie_break=raw["tie_break"], round=places)


def _objective_value(objective: str, truth: np.ndarray, pred: np.ndarray, positive: str) -> float:
    """The objective at one threshold's predictions."""
    if objective == "balanced_accuracy":
        return balanced_accuracy(truth, pred)
    if objective == "f1":
        return float(f1_score(truth == positive, pred == positive, zero_division=0))
    if len(set(truth.tolist()) | set(pred.tolist())) < 2:
        return 0.0
    return float(matthews_corrcoef(truth, pred))


def tune_threshold(
    y_val: Union[np.ndarray, Sequence[Any]],
    scores_val: Union[np.ndarray, Sequence[float]],
    *,
    split_name: str,
    positive_label: Union[str, int],
    objective: Objective = "balanced_accuracy",
    grid: Optional[ThresholdGrid] = None,
) -> ThresholdChoice:
    """Choose the grid threshold that maximises an objective on a validation split.

    Args:
        y_val: True label per validation row: the positive label and one other label (compared as strings).
        scores_val: The positive label's score per row; finite.
        split_name: The validation split's name; a name containing "test" (any case) is refused.
        positive_label: The positive label.
        objective: `balanced_accuracy` (default), or the explicit choices `f1` (positive class) and `mcc`.
        grid: Candidate thresholds; defaults to `load_threshold_grid()`.

    Returns:
        The chosen threshold with its objective, value, tuning split and grid.

    Raises:
        ValueError: If the split looks like a test split, the objective is unknown, `y_val` does not hold
            exactly two labels including the positive one, or the scores are not finite or of the wrong length.
    """
    if "test" in split_name.casefold():
        raise ValueError(f"thresholds are tuned on validation data only; split {split_name!r} looks like a test split")
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    truth = np.array([str(v) for v in np.asarray(y_val, dtype=object).tolist()], dtype=object)
    scores = np.asarray(scores_val, dtype=float)
    if scores.shape != truth.shape:
        raise ValueError("y_val and scores_val must have the same length")
    if not np.isfinite(scores).all():
        raise ValueError("scores_val must be finite (no NaN or infinity)")
    positive = str(positive_label)
    present = set(truth.tolist())
    if len(present) != 2 or positive not in present:
        raise ValueError("y_val must hold exactly two labels, the positive one among them")
    negative = (present - {positive}).pop()
    candidates = grid if grid is not None else load_threshold_grid()
    best: Optional[Tuple[float, float]] = None
    for t in candidates.values:
        pred = np.where(scores >= t, positive, negative).astype(object)
        value = _objective_value(objective, truth, pred, positive)
        if best is None or value > best[1] or (value == best[1] and abs(t - 0.5) < abs(best[0] - 0.5)):
            best = (t, value)
    assert best is not None
    return ThresholdChoice(
        threshold=round(best[0], candidates.round),
        objective=objective,
        value=best[1],
        split_name=split_name,
        grid=candidates.values,
    )
