"""Metrics for temporal localization: mAP, segment and event precision/recall, presence and repeat counts, floors.

Inputs are plain lists of dicts: predictions ``{video, label, start_s, end_s, score}`` and ground truth ``{video, label,
start_s, end_s}``; ``videos`` is the scored set of recordings (a recording with no ground truth counts: its
predictions are false positives). Nothing here reads a file.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

Seg = Mapping[str, Any]
TIOUS = (0.3, 0.4, 0.5, 0.6, 0.7)


def tiou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Temporal intersection over union of two ``(start, end)`` intervals."""
    inter = min(a[1], b[1]) - max(a[0], b[0])
    if inter <= 0:
        return 0.0
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def _div(a: float, b: float) -> float:
    """Ratio, 0 when the denominator is 0."""
    return a / b if b else 0.0


# --------------------------------------------------------------------------------------------------------------------
# mAP
# --------------------------------------------------------------------------------------------------------------------


def _ap(preds: List[Seg], gts: List[Seg], thr: float) -> float:
    """AP of one class at one tIoU threshold. ``preds``/``gts`` carry ``video``; ``gts`` non-empty."""
    by_video: Dict[Any, List[Tuple[float, float]]] = defaultdict(list)
    for g in gts:
        by_video[g["video"]].append((float(g["start_s"]), float(g["end_s"])))
    used = {v: [False] * len(x) for v, x in by_video.items()}
    order = sorted(range(len(preds)), key=lambda i: (-float(preds[i].get("score", 1.0)), i))
    tp: List[int] = []
    for i in order:
        p = preds[i]
        cand = by_video.get(p["video"], [])
        best, bj = 0.0, -1
        for j, iv in enumerate(cand):
            if used[p["video"]][j]:
                continue
            u = tiou((float(p["start_s"]), float(p["end_s"])), iv)
            if u > best:
                best, bj = u, j
        if bj >= 0 and best >= thr:
            used[p["video"]][bj] = True
            tp.append(1)
        else:
            tp.append(0)
    if not tp:
        return 0.0
    n_gt = len(gts)
    ctp, cfp = 0, 0
    rec: List[float] = []
    prec: List[float] = []
    for t in tp:
        ctp += t
        cfp += 1 - t
        rec.append(ctp / n_gt)
        prec.append(ctp / (ctp + cfp))
    # all-point interpolation: precision envelope, area under the PR curve
    mrec = [0.0] + rec + [1.0]
    mpre = [0.0] + prec + [0.0]
    for k in range(len(mpre) - 2, -1, -1):
        mpre[k] = max(mpre[k], mpre[k + 1])
    return sum((mrec[k + 1] - mrec[k]) * mpre[k + 1] for k in range(len(mrec) - 1))


def map_tal(
    preds: Sequence[Seg],
    gts: Sequence[Seg],
    thresholds: Sequence[float] = TIOUS,
    classes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Per-class AP and mAP at tIoU thresholds (the ActivityNet / OpenTAD convention).

    Predictions of a class are ranked by score over all videos and each is matched to the unmatched ground truth of the
    same video and class with the highest tIoU of at least the threshold; AP is the area under the all-point
    interpolated precision-recall curve. Classes with no ground truth are left out of the mean.

    Returns:
        ``{"map": {thr: value}, "avg_map": mean over thresholds, "ap": {class: {thr: value}}, "n_gt": {class: n}}``
        with values as fractions.
    """
    cls = sorted(set(classes) if classes is not None else {g["label"] for g in gts})
    ap: Dict[str, Dict[float, float]] = {}
    n_gt: Dict[str, int] = {}
    for c in cls:
        g = [x for x in gts if x["label"] == c]
        n_gt[c] = len(g)
        if not g:
            continue
        p = [x for x in preds if x["label"] == c]
        ap[c] = {t: _ap(p, g, t) for t in thresholds}
    m = {t: _div(sum(ap[c][t] for c in ap), len(ap)) for t in thresholds}
    return {"map": m, "avg_map": _div(sum(m.values()), len(m)), "ap": ap, "n_gt": n_gt}


def binary_map(preds: Sequence[Seg], gts: Sequence[Seg], thresholds: Sequence[float] = TIOUS) -> Dict[str, Any]:
    """Class-agnostic mAP (every label becomes ``event``): the thesis's binary ActionFormer protocol."""

    def one_class(xs: Sequence[Seg]) -> List[Dict[str, Any]]:
        return [{**x, "label": "event"} for x in xs]

    return map_tal(one_class(preds), one_class(gts), thresholds)


# --------------------------------------------------------------------------------------------------------------------
# Segment and event P/R/F1
# --------------------------------------------------------------------------------------------------------------------


def seg_prf(preds: Sequence[Seg], gts: Sequence[Seg], iou: float, by_label: bool = True) -> Dict[str, float]:
    """Micro precision, recall and F1 at one tIoU threshold.

    A prediction is correct when it overlaps an unmatched ground truth of the same video (and label when ``by_label``)
    at ``iou`` or more; predictions are visited by score and each takes its best-IoU ground truth.
    """
    gv: Dict[Any, List[Seg]] = defaultdict(list)
    for g in gts:
        gv[g["video"]].append(g)
    used = {v: [False] * len(x) for v, x in gv.items()}
    tp = 0
    for p in sorted(preds, key=lambda x: -float(x.get("score", 1.0))):
        best, bj = 0.0, -1
        for j, g in enumerate(gv.get(p["video"], [])):
            if used[p["video"]][j] or (by_label and g["label"] != p["label"]):
                continue
            u = tiou((float(p["start_s"]), float(p["end_s"])), (float(g["start_s"]), float(g["end_s"])))
            if u > best:
                best, bj = u, j
        if bj >= 0 and best >= iou:
            used[p["video"]][bj] = True
            tp += 1
    fp, fn = len(preds) - tp, len(gts) - tp
    pr, rc = _div(tp, tp + fp), _div(tp, tp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": pr, "recall": rc, "f1": _div(2 * pr * rc, pr + rc)}


def event_pairs(
    preds: Sequence[Seg], gts: Sequence[Seg], tol_s: float, by_label: bool = False
) -> List[Tuple[Seg, Seg]]:
    """One-to-one matches of point events: segment centres within ``tol_s`` seconds, closest pairs first per video."""
    pv: Dict[Any, List[Tuple[float, Seg]]] = defaultdict(list)
    gv: Dict[Any, List[Tuple[float, Seg]]] = defaultdict(list)
    for p in preds:
        pv[p["video"]].append(((float(p["start_s"]) + float(p["end_s"])) / 2, p))
    for g in gts:
        gv[g["video"]].append(((float(g["start_s"]) + float(g["end_s"])) / 2, g))
    out: List[Tuple[Seg, Seg]] = []
    for v, ps in pv.items():
        gl = gv.get(v, [])
        pairs = sorted(
            (abs(a - b), i, j)
            for i, (a, p) in enumerate(ps)
            for j, (b, g) in enumerate(gl)
            if abs(a - b) <= tol_s and (not by_label or p["label"] == g["label"])
        )
        ui, uj = set(), set()
        for _, i, j in pairs:
            if i in ui or j in uj:
                continue
            ui.add(i)
            uj.add(j)
            out.append((ps[i][1], gl[j][1]))
    return out


def event_prf(preds: Sequence[Seg], gts: Sequence[Seg], tol_s: float, by_label: bool = False) -> Dict[str, float]:
    """Precision, recall and F1 of point events.

    Segment centres within ``tol_s`` seconds match one-to-one, closest pairs first.
    """
    tp = len(event_pairs(preds, gts, tol_s, by_label))
    fp, fn = len(preds) - tp, len(gts) - tp
    pr, rc = _div(tp, tp + fp), _div(tp, tp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": pr, "recall": rc, "f1": _div(2 * pr * rc, pr + rc)}


# --------------------------------------------------------------------------------------------------------------------
# Presence and counts
# --------------------------------------------------------------------------------------------------------------------


def presence(preds: Sequence[Seg], gts: Sequence[Seg], videos: Sequence[Any], labels: Sequence[str]) -> Dict[str, Any]:
    """Per label, video-level presence (>= 1 segment) metrics and the repeat count error; plus ``any`` (any label)."""
    out: Dict[str, Any] = {}
    for lab in list(labels) + ["__any__"]:
        pc: Dict[Any, int] = defaultdict(int)
        gc: Dict[Any, int] = defaultdict(int)
        for p in preds:
            if lab == "__any__" or p["label"] == lab:
                pc[p["video"]] += 1
        for g in gts:
            if lab == "__any__" or g["label"] == lab:
                gc[g["video"]] += 1
        tp = fp = fn = tn = 0
        err: List[int] = []
        for v in videos:
            a, b = pc.get(v, 0) > 0, gc.get(v, 0) > 0
            tp += a and b
            fp += a and not b
            fn += (not a) and b
            tn += (not a) and (not b)
            if b:  # count error on videos that have the behaviour
                err.append(pc.get(v, 0) - gc[v])
        n = tp + fp + fn + tn
        tpr, tnr = _div(tp, tp + fn), _div(tn, tn + fp)
        den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        out[lab] = {
            "n_videos": n,
            "prevalence": _div(tp + fn, n),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": _div(tp, tp + fp),
            "recall": tpr,
            "f1": _div(2 * _div(tp, tp + fp) * tpr, _div(tp, tp + fp) + tpr),
            "ba": (tpr + tnr) / 2,
            "mcc": _div(tp * tn - fp * fn, den),
            "count_n": len(err),
            "count_mae": _div(sum(abs(e) for e in err), len(err)),
            "count_exact": _div(sum(1 for e in err if e == 0), len(err)),
            "count_off1": _div(sum(1 for e in err if abs(e) <= 1), len(err)),
            "count_bias": _div(sum(err), len(err)),
        }
    return out


def variant_confusion(
    preds: Sequence[Seg], gts: Sequence[Seg], labels: Sequence[str], iou: float = 0.3
) -> Dict[str, Dict[str, int]]:
    """Confusion counts ``conf[true][pred]`` over greedy label-agnostic matches at tIoU of ``iou`` or more.

    Unmatched ground truth goes to ``pred = none`` and unmatched predictions to ``true = none``.
    """
    names = list(labels) + ["none"]
    conf = {t: {p: 0 for p in names} for t in names}
    gv: Dict[Any, List[Seg]] = defaultdict(list)
    for g in gts:
        gv[g["video"]].append(g)
    used = {v: [False] * len(x) for v, x in gv.items()}
    for p in sorted(preds, key=lambda x: -float(x.get("score", 1.0))):
        best, bj = 0.0, -1
        for j, g in enumerate(gv.get(p["video"], [])):
            if used[p["video"]][j]:
                continue
            u = tiou((float(p["start_s"]), float(p["end_s"])), (float(g["start_s"]), float(g["end_s"])))
            if u > best:
                best, bj = u, j
        if bj >= 0 and best >= iou:
            used[p["video"]][bj] = True
            conf[gv[p["video"]][bj]["label"]][p["label"]] += 1
        else:
            conf["none"][p["label"]] += 1
    for v, flags in used.items():
        for j, u in enumerate(flags):
            if not u:
                conf[gv[v][j]["label"]]["none"] += 1
    return conf


# --------------------------------------------------------------------------------------------------------------------
# Floors
# --------------------------------------------------------------------------------------------------------------------


def shuffle_floor(
    preds: Sequence[Seg], durations: Mapping[Any, float], seed: int = 0, draws: int = 10
) -> List[List[Dict[str, Any]]]:
    """Draw ``draws`` lists of the predictions moved to a uniform start in their video (same label and length)."""
    rng = random.Random(seed)
    out: List[List[Dict[str, Any]]] = []
    for _ in range(draws):
        one = []
        for p in preds:
            dur = float(durations[p["video"]])
            ln = min(float(p["end_s"]) - float(p["start_s"]), dur)
            s = rng.uniform(0.0, max(dur - ln, 0.0))
            one.append({"video": p["video"], "label": p["label"], "start_s": s, "end_s": s + ln, "score": rng.random()})
        out.append(one)
    return out


def whole_video(videos: Sequence[Any], durations: Mapping[Any, float], labels: Sequence[str]) -> List[Dict[str, Any]]:
    """Every label over the whole of every video, score 1."""
    return [
        {"video": v, "label": lab, "start_s": 0.0, "end_s": float(durations[v]), "score": 1.0}
        for v in videos
        for lab in labels
    ]


def mean_over(draws: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    """Mean of each key over a list of metric dicts."""
    keys = draws[0].keys() if draws else []
    return {k: sum(d[k] for d in draws) / len(draws) for k in keys}
