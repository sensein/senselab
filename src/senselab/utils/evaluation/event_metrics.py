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
    """Counts and shares from `event_f1`; a share is `None` where it is undefined."""

    tp: int
    fp: int
    fn: int
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]


class SegmentMAP(TypedDict):
    """Mean average precision from `segment_map`, keyed by threshold strings such as "0.5"."""

    map_at: Dict[str, float]
    average_map: float
    per_class_ap: Dict[str, Dict[str, float]]


def _events(events: Events) -> List[BehaviorEvent]:
    """The events of a track or sequence as a list."""
    return list(events.events) if isinstance(events, EventTrack) else list(events)


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    """A share, `None` when the denominator is 0."""
    return numerator / denominator if denominator else None


def event_f1(pred: Events, ref: Events, *, tolerance_s: float, label: Optional[str] = None) -> EventF1:
    """Onset-tolerance precision, recall and F1.

    Predictions are taken in onset order (then offset); each takes the unused reference whose onset is
    nearest its own, if that is within `tolerance_s` (ties go to the earlier reference), and counts as a
    true positive; otherwise it is a false positive. References left unused are false negatives.

    With `label=None` labels are ignored and every event takes part, as in the implementation this
    is ported from. With a label, only events with that label take part on either side.

    Precision is `None` without predictions, recall `None` without references, and F1 `None` unless
    both are defined. The earlier implementation reported 0.0 in those cases and pooled clips; to
    reproduce its numbers, keep only the labels it kept on each side yourself, call this per clip, sum
    `tp`, `fp` and `fn` over clips, and compute the shares from the sums.

    Args:
        pred: Predicted events (an `EventTrack` or a sequence of `BehaviorEvent`).
        ref: Reference events.
        tolerance_s: Largest onset difference that still matches, in seconds; at least 0.
        label: Restrict both sides to this label, or `None` to ignore labels.

    Returns:
        `{tp, fp, fn, precision, recall, f1}`; a share with a zero denominator is `None`.

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
    f1: Optional[float] = None
    if precision is not None and recall is not None:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _tiou(pred: Tuple[float, float], ref: Tuple[float, float]) -> float:
    """Temporal IoU as the ActivityNet evaluator computes it: union = len(ref) + len(pred) - intersection."""
    inter = max(0.0, min(pred[1], ref[1]) - max(pred[0], ref[0]))
    union = (ref[1] - ref[0]) + (pred[1] - pred[0]) - inter
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
    """AP of one class, ranking and matching in the ActivityNet evaluator's order.

    Predictions are ranked by `argsort()[::-1]` of their confidences, so tied confidences come out in
    reversed input order. Each prediction walks its video's references by descending tIoU (again
    `argsort()[::-1]`, so on tied tIoU the later reference comes first): the first reference below the
    threshold ends the walk (false positive), a matched one is passed over, an unmatched one is a hit.
    """
    order = np.asarray([p[1] for p in preds], dtype=float).argsort()[::-1]
    matched = {video: [False] * len(segments) for video, segments in refs.items()}
    hits: List[bool] = []
    for i in order:
        video, _, segment = preds[int(i)]
        candidates = refs.get(video, [])
        hit = False
        if candidates:
            tious = np.array([_tiou(segment, ref_segment) for ref_segment in candidates])
            for j in tious.argsort()[::-1]:
                if tious[j] < threshold:
                    break
                if matched[video][j]:
                    continue
                matched[video][j] = True
                hit = True
                break
        hits.append(hit)
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
    class's references. The mean is over classes present in the references, in their order of first
    appearance (as the ActivityNet evaluator indexes them); predicted classes absent from them are
    ignored. Ties are ordered as that evaluator orders them (see `_class_ap`), for parity with published
    numbers.

    Args:
        preds: Video id to predicted events; every prediction needs a confidence.
        refs: Video id to reference events.
        tiou_thresholds: Temporal-IoU thresholds, each in (0, 1].

    Returns:
        `{map_at: {threshold: mAP}, average_map, per_class_ap: {threshold: {class: AP}}}`, thresholds as
        strings (for example "0.5"), so mAP at each threshold and their average are reported separately.

    Raises:
        ValueError: If a prediction has no confidence, there are no reference events, a threshold is
            outside (0, 1], or two thresholds format to the same key.
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
    keys = [f"{t:g}" for t in tiou_thresholds]
    if len(set(keys)) != len(keys):
        raise ValueError("tiou_thresholds must be distinct after formatting to keys like '0.5'")
    classes = list(ref_by_class)
    map_at: Dict[str, float] = {}
    per_class: Dict[str, Dict[str, float]] = {}
    for key, threshold in zip(keys, tiou_thresholds):
        per_class[key] = {
            c: _class_ap(
                pred_by_class.get(c, []),
                ref_by_class[c],
                sum(len(v) for v in ref_by_class[c].values()),
                threshold,
            )
            for c in classes
        }
        map_at[key] = float(np.mean([per_class[key][c] for c in classes]))
    return {
        "map_at": map_at,
        "average_map": float(np.mean(list(map_at.values()))),
        "per_class_ap": per_class,
    }
