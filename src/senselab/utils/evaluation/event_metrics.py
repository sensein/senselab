"""Event detection metrics: onset-tolerance F1 and temporal-localisation mean average precision.

`event_f1` matches predicted to reference events by onset within a tolerance, greedily and one to
one, and reports counts, precision, recall and F1. `segment_map` scores ranked, confidence-carrying
segment predictions against references per class with the ActivityNet temporal-localisation
definition, at several temporal-IoU thresholds.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, TypedDict, Union

import numpy as np

from senselab.utils.data_structures.events import BehaviorEvent, EventTrack

Events = Union[EventTrack, Sequence[BehaviorEvent]]
DEFAULT_TIOU_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)


class EventF1(TypedDict):
    """Counts and shares from `event_f1`."""

    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


class SegmentMAP(TypedDict):
    """Mean average precision from `segment_map`, keyed by threshold strings such as "0.5"."""

    map_at: Dict[str, float]
    average_map: float
    per_class_ap: Dict[str, Dict[str, float]]


def _events(events: Events) -> List[BehaviorEvent]:
    """The events of a track or sequence as a list."""
    return list(events.events) if isinstance(events, EventTrack) else list(events)


def _ratio(numerator: int, denominator: int) -> float:
    """A share, 0.0 when the denominator is 0 (the ported convention)."""
    return numerator / denominator if denominator else 0.0


def event_f1(pred: Events, ref: Events, *, tolerance_s: float, label: Optional[str] = None) -> EventF1:
    """Onset-tolerance precision, recall and F1.

    Predictions are taken in onset order (then offset); each takes the unused reference whose onset is
    nearest its own, if that is within `tolerance_s` (ties go to the earlier reference), and counts as a
    true positive; otherwise it is a false positive. References left unused are false negatives.

    With `label=None` labels are ignored and every event takes part, as in the implementation this
    is ported from. With a label, only events with that label take part on either side.

    Args:
        pred: Predicted events (an `EventTrack` or a sequence of `BehaviorEvent`).
        ref: Reference events.
        tolerance_s: Largest onset difference that still matches, in seconds; at least 0.
        label: Restrict both sides to this label, or `None` to ignore labels.

    Returns:
        `{tp, fp, fn, precision, recall, f1}`; a share with a zero denominator is 0.0.

    Raises:
        ValueError: If the tolerance is negative or not finite.
    """
    if not (math.isfinite(tolerance_s) and tolerance_s >= 0):
        raise ValueError("tolerance_s must be a finite number >= 0")
    predicted = [e for e in _events(pred) if label is None or e.label == label]
    reference = [e for e in _events(ref) if label is None or e.label == label]
    pred_times = sorted((e.onset_s, e.offset_s) for e in predicted)
    ref_onsets = [onset for onset, _ in sorted((e.onset_s, e.offset_s) for e in reference)]
    used = [False] * len(ref_onsets)
    tp = fp = 0
    for onset, _ in pred_times:
        best, best_diff = -1, math.inf
        for j, ref_onset in enumerate(ref_onsets):
            if used[j]:
                continue
            diff = abs(onset - ref_onset)
            if diff <= tolerance_s and diff < best_diff:
                best, best_diff = j, diff
            elif ref_onset > onset + tolerance_s:
                break
        if best >= 0:
            used[best] = True
            tp += 1
        else:
            fp += 1
    fn = used.count(False)
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _tiou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Temporal intersection over union of two half-open intervals."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def _average_precision(hits: List[bool], n_refs: int) -> float:
    """All-point interpolated average precision of a ranked list of hits against `n_refs` references."""
    if not hits:
        return 0.0
    cum_tp = np.cumsum(hits)
    cum_fp = np.cumsum([not h for h in hits])
    precision = np.concatenate([[0.0], cum_tp / (cum_tp + cum_fp), [0.0]])
    recall = np.concatenate([[0.0], cum_tp / n_refs, [1.0]])
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    steps = np.flatnonzero(recall[1:] != recall[:-1]) + 1
    return float(np.sum((recall[steps] - recall[steps - 1]) * precision[steps]))


def _class_ap(
    preds: List[Tuple[str, float, Tuple[float, float]]],
    refs: Mapping[str, List[Tuple[float, float]]],
    n_refs: int,
    threshold: float,
) -> float:
    """AP of one class: predictions ranked by confidence, each matched to its best unmatched same-video reference."""
    order = sorted(range(len(preds)), key=lambda i: -preds[i][1])
    matched = {video: [False] * len(segments) for video, segments in refs.items()}
    hits: List[bool] = []
    for i in order:
        video, _, segment = preds[i]
        best, best_tiou = -1, -1.0
        for j, ref_segment in enumerate(refs.get(video, [])):
            if matched[video][j]:
                continue
            value = _tiou(segment, ref_segment)
            if value >= threshold and value > best_tiou:
                best, best_tiou = j, value
        if best >= 0:
            matched[video][best] = True
        hits.append(best >= 0)
    return _average_precision(hits, n_refs)


def segment_map(
    preds: Mapping[str, Events],
    refs: Mapping[str, Events],
    *,
    tiou_thresholds: Sequence[float] = DEFAULT_TIOU_THRESHOLDS,
) -> SegmentMAP:
    """Temporal-localisation mean average precision, ActivityNet style.

    For each class present in the references and each threshold: predictions of that class from all
    videos are ranked by confidence (ties keep input order); each is a true positive if it has an
    unmatched reference of the same class in the same video with temporal IoU at or above the threshold,
    taking the one with the highest IoU, and a false positive otherwise (so a duplicate detection is a
    false positive). AP is all-point interpolated over the ranked list, recall counting all of the
    class's references. The mean is over classes present in the references; predicted classes absent
    from them are ignored.

    Args:
        preds: Video id to predicted events; every prediction needs a confidence.
        refs: Video id to reference events.
        tiou_thresholds: Temporal-IoU thresholds, each in (0, 1].

    Returns:
        `{map_at: {threshold: mAP}, average_map, per_class_ap: {threshold: {class: AP}}}`, thresholds as
        strings (for example "0.5"), so mAP at each threshold and their average are reported separately.

    Raises:
        ValueError: If a prediction has no confidence, there are no reference events, or a threshold is
            outside (0, 1].
    """
    if not tiou_thresholds or any(not 0 < t <= 1 for t in tiou_thresholds):
        raise ValueError("tiou_thresholds must be non-empty, each in (0, 1]")
    ref_by_class: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
    for video, events in refs.items():
        for e in _events(events):
            ref_by_class.setdefault(e.label, {}).setdefault(video, []).append((e.onset_s, e.offset_s))
    if not ref_by_class:
        raise ValueError("segment_map needs at least one reference event")
    pred_by_class: Dict[str, List[Tuple[str, float, Tuple[float, float]]]] = {}
    for video, events in preds.items():
        for e in _events(events):
            if e.confidence is None:
                raise ValueError("every prediction needs a confidence to be ranked")
            pred_by_class.setdefault(e.label, []).append((video, e.confidence, (e.onset_s, e.offset_s)))
    classes = sorted(ref_by_class)
    map_at: Dict[str, float] = {}
    per_class: Dict[str, Dict[str, float]] = {}
    for threshold in tiou_thresholds:
        key = f"{threshold:g}"
        per_class[key] = {
            c: _class_ap(
                pred_by_class.get(c, []),
                ref_by_class[c],
                sum(len(v) for v in ref_by_class[c].values()),
                threshold,
            )
            for c in classes
        }
        map_at[key] = float(math.fsum(per_class[key].values()) / len(classes))
    return {
        "map_at": map_at,
        "average_map": float(math.fsum(map_at.values()) / len(map_at)),
        "per_class_ap": per_class,
    }
