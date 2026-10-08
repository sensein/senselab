"""Tests for event metrics: onset-tolerance F1 and segment mAP."""

import json

import numpy as np
import pytest

from senselab.utils.data_structures.events import BehaviorEvent, EventTrack
from senselab.utils.evaluation.event_metrics import event_f1, segment_map


def _ev(onset: float, offset: float, label: str = "reach", confidence: float | None = None) -> BehaviorEvent:
    kind = "none" if confidence is None else "prob"
    return BehaviorEvent(
        onset_s=onset, offset_s=offset, label=label, source="m", confidence=confidence, confidence_kind=kind
    )


# ── onset-tolerance F1 ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("pred", "ref", "expected"),
    [
        ([_ev(1.0, 2.0)], [_ev(1.0, 2.0)], (1, 0, 0)),
        ([_ev(1.2, 2.0)], [_ev(1.0, 2.0)], (1, 0, 0)),
        ([_ev(1.4, 2.0)], [_ev(1.0, 2.0)], (0, 1, 1)),
        ([_ev(0.9, 2.0), _ev(1.1, 2.0)], [_ev(1.0, 2.0)], (1, 1, 0)),
        ([], [_ev(1.0, 2.0), _ev(3.0, 4.0)], (0, 0, 2)),
    ],
    ids=["exact", "within-tolerance", "outside-tolerance", "two-predictions-one-reference", "empty-prediction"],
)
def test_event_f1_cases(pred: list, ref: list, expected: tuple) -> None:
    """Greedy one-to-one onset matching at a 0.25 s tolerance."""
    result = event_f1(pred, ref, tolerance_s=0.25)
    assert (result["tp"], result["fp"], result["fn"]) == expected
    tp, fp, fn = expected
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    assert result["precision"] == precision and result["recall"] == recall
    assert result["f1"] == (2 * precision * recall / (precision + recall) if precision + recall else 0.0)


def test_event_f1_prefers_the_nearest_unused_reference() -> None:
    """A prediction takes the closest unused reference within tolerance, leaving the other for the next."""
    ref = [_ev(1.0, 1.5), _ev(1.3, 2.0)]
    pred = [_ev(1.25, 2.0), _ev(1.05, 1.5)]
    assert event_f1(pred, ref, tolerance_s=0.3)["tp"] == 2


def test_label_none_ignores_labels() -> None:
    """Without a label, matching is on onsets alone, whatever the labels say (the ported behaviour)."""
    pred = [_ev(1.0, 2.0, "wave")]
    ref = [_ev(1.0, 2.0, "reach")]
    assert event_f1(pred, ref, tolerance_s=0.1)["tp"] == 1
    assert event_f1(pred, ref, tolerance_s=0.1, label="reach") == {
        "tp": 0,
        "fp": 0,
        "fn": 1,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }


def test_event_f1_accepts_tracks() -> None:
    """EventTracks can be passed directly."""
    track = EventTrack(schema_id="s", events=[_ev(1.0, 2.0), _ev(3.0, 4.0)])
    assert event_f1(track, track, tolerance_s=0.0)["f1"] == 1.0


def _origin_matching(pred: list, ref: list, tol: float) -> tuple[int, int, int]:
    """The earlier lab onset matching, rewritten here from its algorithm for the parity check."""
    ps, gs = sorted(pred), sorted(ref)
    used = [False] * len(gs)
    tp = fp = 0
    for p_on, _ in ps:
        best, best_diff = -1, float("inf")
        for j, (g_on, _) in enumerate(gs):
            if used[j]:
                continue
            diff = abs(p_on - g_on)
            if diff <= tol and diff < best_diff:
                best, best_diff = j, diff
            elif g_on > p_on + tol:
                break
        if best >= 0:
            tp += 1
            used[best] = True
        else:
            fp += 1
    return tp, fp, used.count(False)


def test_event_f1_matches_the_earlier_implementation() -> None:
    """On 300 random cases the counts equal the earlier implementation's."""
    rng = np.random.default_rng(0)
    for _ in range(300):

        def draw(k: int) -> list[tuple[float, float]]:
            on = np.round(rng.uniform(0, 20, k), 2)
            return sorted((float(o), float(o) + 0.5) for o in on)

        p, g = draw(int(rng.integers(0, 15))), draw(int(rng.integers(0, 15)))
        tol = float(rng.choice([0.1, 0.25, 0.5]))
        result = event_f1([_ev(a, b) for a, b in p], [_ev(a, b) for a, b in g], tolerance_s=tol)
        assert (result["tp"], result["fp"], result["fn"]) == _origin_matching(p, g, tol)


def test_event_f1_refuses_negative_tolerance() -> None:
    """The tolerance is a non-negative number of seconds."""
    with pytest.raises(ValueError, match="tolerance"):
        event_f1([], [], tolerance_s=-0.1)


# ── segment mAP ─────────────────────────────────────────────────────────


def test_perfect_predictions_give_one() -> None:
    """Every reference predicted exactly, with confidence, gives AP 1 at every threshold."""
    refs = {"v1": [_ev(0, 2, "a"), _ev(5, 7, "b")], "v2": [_ev(1, 3, "a")]}
    preds = {v: [_ev(e.onset_s, e.offset_s, e.label, 0.9) for e in evs] for v, evs in refs.items()}
    result = segment_map(preds, refs)
    assert all(v == 1.0 for v in result["map_at"].values()) and result["average_map"] == 1.0


def test_ranked_list_average_precision() -> None:
    """TP at 0.9, FP at 0.8, TP at 0.7 against two references: AP = 0.5 * 1 + 0.5 * 2/3."""
    refs = {"v": [_ev(0, 2, "a"), _ev(10, 12, "a")]}
    preds = {"v": [_ev(0, 2, "a", 0.9), _ev(20, 22, "a", 0.8), _ev(10, 12, "a", 0.7)]}
    result = segment_map(preds, refs, tiou_thresholds=(0.5,))
    assert abs(result["per_class_ap"]["0.5"]["a"] - (0.5 + 0.5 * 2 / 3)) <= 1e-12


def test_tiou_threshold_decides_a_match() -> None:
    """A prediction with tIoU 0.6 is a match at 0.5 and not at 0.7."""
    refs = {"v": [_ev(0, 10, "a")]}
    preds = {"v": [_ev(0, 6, "a", 0.9)]}
    result = segment_map(preds, refs, tiou_thresholds=(0.5, 0.7))
    assert result["map_at"] == {"0.5": 1.0, "0.7": 0.0}
    assert result["average_map"] == 0.5


def test_duplicates_are_false_positives_and_foreign_classes_are_excluded() -> None:
    """A second detection of a matched reference is a false positive; a class absent from the references is ignored."""
    refs = {"v": [_ev(0, 2, "a")]}
    preds = {"v": [_ev(0, 2, "a", 0.5), _ev(0, 2, "a", 0.9), _ev(5, 6, "z", 0.99)]}
    result = segment_map(preds, refs, tiou_thresholds=(0.5,))
    assert set(result["per_class_ap"]["0.5"]) == {"a"}
    assert result["map_at"]["0.5"] == 1.0
    lower_first = {"v": [_ev(0, 2, "a", 0.9), _ev(30, 32, "a", 0.95)]}
    assert segment_map(lower_first, refs, tiou_thresholds=(0.5,))["map_at"]["0.5"] == 0.5


def test_matching_stays_within_a_video() -> None:
    """A prediction never matches a reference from another video."""
    refs = {"v1": [_ev(0, 2, "a")], "v2": []}
    preds = {"v2": [_ev(0, 2, "a", 0.9)]}
    assert segment_map(preds, refs, tiou_thresholds=(0.5,))["map_at"]["0.5"] == 0.0


def test_predictions_need_a_confidence() -> None:
    """Ranking needs confidences; a prediction without one is refused."""
    with pytest.raises(ValueError, match="confidence"):
        segment_map({"v": [_ev(0, 1, "a")]}, {"v": [_ev(0, 1, "a")]})


def test_result_is_plain_json() -> None:
    """The result serialises to strict JSON with string threshold keys."""
    refs = {"v": [_ev(0, 2, "a")]}
    result = segment_map({"v": [_ev(0, 2, "a", 0.9)]}, refs)
    assert list(result["map_at"]) == ["0.3", "0.4", "0.5", "0.6", "0.7"]
    json.dumps(result, allow_nan=False)
