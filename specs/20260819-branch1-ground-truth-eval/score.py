"""Score branch 1's propose -> confirm -> bound chain against the six verified events.

Five probes, in the order measurement.md reports them:

0. the denominators, and the smallest effect each probe could show;
1. the proposer, threshold-swept, recall against false positives;
2. the same on nine degraded copies;
3. the confirmer, threshold-swept, over the proposer's output;
4. the bounder, against the two verified cough spans;
5. the three stages composed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import io_util
import numpy as np
import scipy.stats as st
import soundfile as sf
from bounder import DELTAS_DB, bound_span, envelope
from ground_truth import DURATION, EVENTS, UNLABELLED, WAV, overlap, scorable_empty, scorable_empty_seconds

HERE = Path(__file__).parent
RAW = HERE / "raw"

DRAFT_VOCAB = ("Breathing", "Cough", "Gasp", "Sigh", "Throat clearing", "Sneeze", "Snoring", "Speech")
"""The AudioSet classes branch-1-airway.md names as the proposer's vocabulary, plus Speech."""

PROPOSERS = ("vocab", "nonsilence")
THRESHOLDS = [round(x, 3) for x in np.concatenate([np.arange(0.02, 0.30, 0.02), np.arange(0.30, 1.00, 0.05)])]
HEAR_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

EMPTY = scorable_empty()
EMPTY_SECONDS = scorable_empty_seconds()
EMPTY_MINUTES = EMPTY_SECONDS / 60.0


# --------------------------------------------------------------------------- proposer scoring


def proposer_scores(run: Dict[str, Any], kind: str) -> np.ndarray:
    """Per-window proposal score for one reading of "YAMNet proposes"."""
    labels = run["labels"]
    scores = np.asarray(run["scores"], dtype=np.float64)
    index = {name: i for i, name in enumerate(labels)}
    if kind == "vocab":
        cols = [index[name] for name in DRAFT_VOCAB]
        return scores[:, cols].max(axis=1)
    if kind == "nonsilence":
        return 1.0 - scores[:, index["Silence"]]
    raise ValueError(kind)


def windows(run: Dict[str, Any]) -> List[Tuple[float, float]]:
    """Window intervals in seconds."""
    return list(zip(run["starts"], run["ends"]))


def merge_runs(fired: Sequence[int], wins: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Merge maximal runs of consecutive fired windows into detection intervals."""
    out: List[Tuple[float, float]] = []
    for i in fired:
        if out and i - 1 in fired and out[-1][1] >= wins[i][0]:
            out[-1] = (out[-1][0], wins[i][1])
        else:
            out.append((wins[i][0], wins[i][1]))
    return out


def touches_unlabelled(interval: Tuple[float, float]) -> bool:
    """True when an interval reaches into the one genuinely unlabelled stretch."""
    return overlap(interval, UNLABELLED) > 0.0


def wholly_inside_empty(interval: Tuple[float, float]) -> bool:
    """True when an interval lies wholly inside one scorable-empty region."""
    return any(a <= interval[0] and interval[1] <= b for a, b in EMPTY)


def empty_window_indices(wins: Sequence[Tuple[float, float]]) -> List[int]:
    """Windows that lie wholly inside a scorable-empty region — the unambiguous FP denominator."""
    return [i for i, w in enumerate(wins) if wholly_inside_empty(w)]


def event_free_window_indices(wins: Sequence[Tuple[float, float]]) -> List[int]:
    """Windows with zero overlap on every event, excluding any that touch the unlabelled tail."""
    return [
        i
        for i, w in enumerate(wins)
        if not touches_unlabelled(w) and all(overlap(w, (e.onset, e.offset)) == 0.0 for e in EVENTS)
    ]


def covered_seconds(intervals: Sequence[Tuple[float, float]], regions: Sequence[Tuple[float, float]]) -> float:
    """Seconds of ``regions`` covered by the union of ``intervals``."""
    total = 0.0
    for a, b in regions:
        marks = np.zeros(int(round((b - a) * 1000)), dtype=bool)
        for x, y in intervals:
            lo = max(a, x)
            hi = min(b, y)
            if hi > lo:
                marks[int(round((lo - a) * 1000)) : int(round((hi - a) * 1000))] = True
        total += marks.sum() / 1000.0
    return total


def score_proposer(
    run: Dict[str, Any], kind: str, threshold: float, empty_idx: List[int], free_idx: List[int]
) -> Dict[str, Any]:
    """Recall over the six events and false positives over the verified-empty stretches."""
    wins = windows(run)
    values = proposer_scores(run, kind)
    fired = [i for i, v in enumerate(values) if v >= threshold]
    fired_set = set(fired)
    detections = merge_runs(fired, wins)

    recalled: List[int] = []
    onset_recalled: List[int] = []
    for event in EVENTS:
        extent = (event.onset, event.offset)
        if any(overlap(w, extent) > 0.0 for w in (wins[i] for i in fired)):
            recalled.append(event.index)
        if any(w[0] <= event.onset <= w[1] for w in (wins[i] for i in fired)):
            onset_recalled.append(event.index)

    fp_detections = [
        d
        for d in detections
        if not touches_unlabelled(d) and all(overlap(d, (e.onset, e.offset)) == 0.0 for e in EVENTS)
    ]
    tp_detections = [d for d in detections if any(overlap(d, (e.onset, e.offset)) > 0.0 for e in EVENTS)]

    fp_windows = sum(1 for i in empty_idx if i in fired_set)
    fp_free_windows = sum(1 for i in free_idx if i in fired_set)

    return {
        "threshold": threshold,
        "n_fired_windows": len(fired),
        "n_detections": len(detections),
        "recall": len(recalled) / len(EVENTS),
        "recalled": recalled,
        "onset_recall": len(onset_recalled) / len(EVENTS),
        "onset_recalled": onset_recalled,
        "fp_detections": len(fp_detections),
        "fp_per_min": len(fp_detections) / EMPTY_MINUTES,
        "tp_detections": len(tp_detections),
        "precision_detections": (len(tp_detections) / len(detections)) if detections else float("nan"),
        "fp_windows": fp_windows,
        "n_empty_windows": len(empty_idx),
        "fp_free_windows": fp_free_windows,
        "n_free_windows": len(free_idx),
        # One fired window = one alarm. Unlike fp_per_min this does not collapse when adjacent
        # detections merge, which on this file they always do.
        "fp_per_min_windows": fp_free_windows / EMPTY_MINUTES,
        "fa_duty_all": covered_seconds(detections, EMPTY) / EMPTY_SECONDS,
        "fa_duty_fp_only": covered_seconds(fp_detections, EMPTY) / EMPTY_SECONDS,
        "detections": detections,
    }


def clopper_pearson(k: int, n: int) -> Tuple[float, float]:
    """Exact 95% binomial interval, so a count of 6 is reported with its real width."""
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else st.beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else st.beta.ppf(0.975, k + 1, n - k)
    return (float(lo), float(hi))


# --------------------------------------------------------------------------- confirmer


def hear_at(run: Dict[str, Any], interval: Tuple[float, float]) -> Dict[str, Any]:
    """HeAR's verdict on a proposal, two ways.

    ``centred`` is the single 2 s window whose centre is nearest the proposal's centre — the
    confirmer as the design describes it. ``any_overlapping`` is the maximum over every window
    overlapping the proposal, which is the generous reading: an event's plateau spans several
    windows, so taking the best of them can only help the confirmer.
    """
    labels = run["labels"]
    scores = np.asarray(run["scores"], dtype=np.float64)
    starts = np.asarray(run["starts"], dtype=np.float64)
    ends = np.asarray(run["ends"], dtype=np.float64)
    centres = (starts + ends) / 2.0
    target = (interval[0] + interval[1]) / 2.0

    nearest = int(np.argmin(np.abs(centres - target)))
    overlapping = [i for i in range(len(starts)) if overlap((starts[i], ends[i]), interval) > 0.0] or [nearest]
    best = scores[overlapping].max(axis=0)

    def top(row: np.ndarray) -> Dict[str, Any]:
        j = int(np.argmax(row))
        return {
            "label": labels[j],
            "score": float(row[j]),
            "all": {labels[k]: float(row[k]) for k in range(len(labels))},
        }

    return {
        "centred_window": (float(starts[nearest]), float(ends[nearest])),
        "centred": top(scores[nearest]),
        "any_overlapping": top(best),
        "n_overlapping": len(overlapping),
    }


def score_confirmer(
    yam: Dict[str, Any], hear: Dict[str, Any], kind: str, threshold: float, mode: str
) -> List[Dict[str, Any]]:
    """Confirmation sweep over HeAR's threshold for a fixed proposer operating point."""
    empty_idx = empty_window_indices(windows(yam))
    free_idx = event_free_window_indices(windows(yam))
    base = score_proposer(yam, kind, threshold, empty_idx, free_idx)
    detections = base["detections"]
    verdicts = [hear_at(hear, d) for d in detections]

    rows: List[Dict[str, Any]] = []
    for theta in HEAR_THRESHOLDS:
        kept: List[Tuple[float, float]] = []
        kept_labels: List[str] = []
        for det, verdict in zip(detections, verdicts):
            top = verdict[mode]
            if top["score"] >= theta:
                kept.append(det)
                kept_labels.append(top["label"])
        tp = [d for d in kept if any(overlap(d, (e.onset, e.offset)) > 0.0 for e in EVENTS)]
        fp = [
            d for d in kept if not touches_unlabelled(d) and all(overlap(d, (e.onset, e.offset)) == 0.0 for e in EVENTS)
        ]
        recalled = [e.index for e in EVENTS if any(overlap(d, (e.onset, e.offset)) > 0.0 for d in kept)]
        rows.append(
            {
                "hear_threshold": theta,
                "kept": len(kept),
                "labels": kept_labels,
                "recall": len(recalled) / len(EVENTS),
                "recalled": recalled,
                "precision": (len(tp) / len(kept)) if kept else float("nan"),
                "fp_detections": len(fp),
                "fp_per_min": len(fp) / EMPTY_MINUTES,
            }
        )
    return rows


# --------------------------------------------------------------------------- report


def _wave() -> Tuple[np.ndarray, int]:
    wave, sr = sf.read(WAV, dtype="float64", always_2d=False)
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    return np.ascontiguousarray(wave), sr


def _fmt_int_list(values: Sequence[int]) -> str:
    return ",".join(str(v) for v in values) if values else "-"


def main() -> None:  # noqa: C901, PLR0915
    """Run every probe and print the report score.md is written from."""
    yamnet = io_util.load(RAW / "yamnet.json.gz")
    hear = io_util.load(RAW / "hear.json.gz")
    clean = yamnet["clean"]
    wins = windows(clean)
    empty_idx = empty_window_indices(wins)
    free_idx = event_free_window_indices(wins)
    results: Dict[str, Any] = {}

    print("=" * 78)
    print("PROBE 0 - denominators, and the smallest effect each probe can show")
    print("=" * 78)
    print(f"events                    : {len(EVENTS)}  (recall resolution 1/6 = {1 / 6:.3f})")
    print(f"scorable-empty            : {EMPTY_SECONDS:.3f} s = {EMPTY_MINUTES:.4f} min")
    print(f"one false-positive detection is worth {1 / EMPTY_MINUTES:.2f} FP/min")
    print(f"YAMNet windows            : {len(wins)} at {clean['win_length']}s / {clean['hop_length']}s hop")
    print(f"windows wholly inside empty: {len(empty_idx)}  -> {_fmt_int_list(empty_idx)}")
    print(f"windows with zero event overlap: {len(free_idx)} -> {_fmt_int_list(free_idx)}")
    print(
        f"HeAR windows              : {len(hear['clean']['starts'])} at 2.0s / {hear['clean']['hop_length']:.3f}s hop"
    )
    print()
    print("Effect visible if: the proposer's FP count over the empty stretches is >=1, since 1 is")
    print(
        "the resolution. A true FP rate below 1/{0} = {1:.2f} of empty windows cannot be".format(
            len(free_idx), 1.0 / len(free_idx)
        )
    )
    print("distinguished from zero on this file. fa_duty is reported alongside because it is")
    print("continuous in time and does not quantise at 1.")
    results["probe0"] = {
        "n_events": len(EVENTS),
        "empty_seconds": EMPTY_SECONDS,
        "fp_per_min_quantum": 1 / EMPTY_MINUTES,
        "n_yamnet_windows": len(wins),
        "empty_window_indices": empty_idx,
        "event_free_window_indices": free_idx,
        "empty_regions": EMPTY,
    }

    print()
    print("=" * 78)
    print("PROBE 1 - the proposer, swept")
    print("=" * 78)
    results["probe1"] = {}
    for kind in PROPOSERS:
        rows = [score_proposer(clean, kind, t, empty_idx, free_idx) for t in THRESHOLDS]
        results["probe1"][kind] = [{k: v for k, v in r.items() if k != "detections"} for r in rows]
        print()
        print(
            f"--- proposer = {kind} " + ("(max over the draft vocabulary)" if kind == "vocab" else "(1 - P(Silence))")
        )
        print("  tau   fired  det   recall  events      FPwin/N  FP/min_win  FPdet  FP/min_det  duty_all")
        for r in rows:
            print(
                f"  {r['threshold']:.2f}  {r['n_fired_windows']:5d} {r['n_detections']:4d}  "
                f"{r['recall']:.3f}  {_fmt_int_list(r['recalled']):11s} "
                f"{r['fp_free_windows']}/{r['n_free_windows']}      {r['fp_per_min_windows']:8.2f}  "
                f"{r['fp_detections']:4d}  {r['fp_per_min']:8.2f}    {r['fa_duty_all']:.3f}"
            )
        zero_fp = [r for r in rows if r["fp_detections"] == 0]
        best = max(zero_fp, key=lambda r: r["recall"]) if zero_fp else None
        print(
            f"  best recall at zero FP: {best['recall']:.3f} at tau={best['threshold']:.2f}"
            if best
            else "  no threshold reaches zero FP"
        )
        full = [r for r in rows if r["recall"] >= 1.0 - 1e-9]
        if full:
            cheapest = min(full, key=lambda r: (r["fp_detections"], -r["threshold"]))
            print(
                f"  cheapest full recall  : tau={cheapest['threshold']:.2f} at "
                f"{cheapest['fp_free_windows']}/{cheapest['n_free_windows']} FP windows "
                f"({cheapest['fp_per_min_windows']:.2f}/min), duty_all={cheapest['fa_duty_all']:.3f}"
            )
        else:
            print("  full recall (6/6) is not reached at any threshold")
        for r in rows:
            lo, hi = clopper_pearson(r["fp_free_windows"], r["n_free_windows"])
            r["fp_window_rate_ci95"] = (lo, hi)
        mid = next(r for r in rows if abs(r["threshold"] - 0.5) < 1e-9)
        lo, hi = clopper_pearson(mid["fp_free_windows"], mid["n_free_windows"])
        print(
            f"  at tau=0.50: FP window rate {mid['fp_free_windows']}/{mid['n_free_windows']} "
            f"= {mid['fp_free_windows'] / mid['n_free_windows']:.3f}, exact 95% CI [{lo:.3f}, {hi:.3f}] "
            f"-> FP/min in [{lo * 60 / (clean['hop_length']):.1f}, {hi * 60 / (clean['hop_length']):.1f}] "
            "if one fired window per hop were one alarm"
        )

    print()
    print("Per-window trace on the clean run. 'ev' names the events the window overlaps.")
    labels = clean["labels"]
    scores = np.asarray(clean["scores"], dtype=np.float64)
    index = {name: i for i, name in enumerate(labels)}
    vocab_v = proposer_scores(clean, "vocab")
    nonsil_v = proposer_scores(clean, "nonsilence")
    print("   i  window          ev     vocab  1-P(Sil)  top-1 class            score  argmax_in_vocab")
    for i, (a, b) in enumerate(wins):
        hit = ",".join(str(e.index) for e in EVENTS if overlap((a, b), (e.onset, e.offset)) > 0.0) or "-"
        j = int(np.argmax(scores[i]))
        best_vocab = max(DRAFT_VOCAB, key=lambda n: scores[i, index[n]])
        print(
            f"  {i:2d}  {a:5.2f}-{b:5.2f}  {hit:6s} {vocab_v[i]:.3f}  {nonsil_v[i]:.3f}     "
            f"{labels[j]:22s} {scores[i, j]:.3f}  {best_vocab}"
        )

    print()
    print("=" * 78)
    print("PROBE 2 - robustness: the same proposer on nine variants")
    print("=" * 78)
    ops = [0.10, 0.30, 0.50]
    results["probe2"] = {}
    for kind in PROPOSERS:
        print()
        print(f"--- proposer = {kind}")
        for tau in ops:
            print(f"  tau={tau:.2f}")
            print("    variant            recall  events      FPwin/N  FP/min_win  duty_all")
            per_variant = {}
            for name, run in yamnet.items():
                w = windows(run)
                r = score_proposer(run, kind, tau, empty_window_indices(w), event_free_window_indices(w))
                per_variant[name] = {k: v for k, v in r.items() if k != "detections"}
                print(
                    f"    {name:18s} {r['recall']:.3f}  {_fmt_int_list(r['recalled']):11s} "
                    f"{r['fp_free_windows']}/{r['n_free_windows']}      {r['fp_per_min_windows']:8.2f}    "
                    f"{r['fa_duty_all']:.3f}"
                )
            results["probe2"].setdefault(kind, {})[f"tau={tau}"] = per_variant

    print()
    print("=" * 78)
    print("PROBE 3 - the confirmer")
    print("=" * 78)
    print()
    print("Sanity check first: HeAR at 6.60-7.10 s, which the human verified as containing nothing")
    print("and where an earlier run read Breathe 0.49.")
    verdict = hear_at(hear["clean"], (6.60, 7.10))
    print(
        f"  centred window {verdict['centred_window'][0]:.2f}-{verdict['centred_window'][1]:.2f}: "
        f"top {verdict['centred']['label']} {verdict['centred']['score']:.3f}"
    )
    print(
        f"  Breathe centred={verdict['centred']['all']['Breathe']:.3f}  "
        f"best-overlapping={verdict['any_overlapping']['all']['Breathe']:.3f}"
    )
    results["probe3_sanity"] = verdict

    print()
    print("PROBE 3a - can the confirmer produce a negative at all on this file?")
    print("Effect visible if some 2 s window overlaps no verified event. If none does, HeAR has no")
    print("negative available and its corroboration carries no information about the proposal.")
    hstarts = np.asarray(hear["clean"]["starts"], dtype=float)
    hends = np.asarray(hear["clean"]["ends"], dtype=float)
    clean_windows = [
        (float(a), float(b))
        for a, b in zip(hstarts, hends)
        if all(overlap((float(a), float(b)), (e.onset, e.offset)) == 0.0 for e in EVENTS)
    ]
    gaps = [
        (EVENTS[i].offset, EVENTS[i + 1].onset, EVENTS[i + 1].onset - EVENTS[i].offset) for i in range(len(EVENTS) - 1)
    ]
    print(f"  HeAR windows overlapping no event: {len(clean_windows)} of {len(hstarts)}")
    print(f"  head gap (0 -> first onset)   : {EVENTS[0].onset:.3f} s")
    print(f"  tail gap (last offset -> end) : {DURATION - EVENTS[-1].offset:.3f} s")
    for a, b, g in gaps:
        print(f"  gap {a:6.3f} -> {b:6.3f} : {g:.3f} s   {'< 2 s' if g < 2.0 else '>= 2 s'}")
    widest = max([g for _, _, g in gaps] + [EVENTS[0].onset, DURATION - EVENTS[-1].offset])
    print(f"  widest event-free stretch: {widest:.3f} s")
    results["probe3a"] = {
        "n_hear_windows": len(hstarts),
        "n_event_free_hear_windows": len(clean_windows),
        "gaps": gaps,
        "head_gap": EVENTS[0].onset,
        "tail_gap": DURATION - EVENTS[-1].offset,
    }

    print()
    print("PROBE 3b - the confirmer's specificity, on proposals planted in verified-empty audio")
    print("Effect visible if HeAR scores these below the threshold that keeps the true events.")
    print("  region                 planted proposal      centred_top      score   best_overlapping_top  score")
    planted = []
    for a, b in EMPTY:
        mid = (a + b) / 2.0
        proposal = (max(a, mid - 0.24), min(b, mid + 0.24))
        v = hear_at(hear["clean"], proposal)
        planted.append({"region": (a, b), "proposal": proposal, "verdict": v})
        print(
            f"  {a:6.3f}-{b:6.3f}        {proposal[0]:6.3f}-{proposal[1]:6.3f}       "
            f"{v['centred']['label']:12s} {v['centred']['score']:.3f}   "
            f"{v['any_overlapping']['label']:12s} {v['any_overlapping']['score']:.3f}"
        )
    for theta in HEAR_THRESHOLDS:
        n_c = sum(1 for p in planted if p["verdict"]["centred"]["score"] >= theta)
        n_o = sum(1 for p in planted if p["verdict"]["any_overlapping"]["score"] >= theta)
        print(f"  theta={theta:.1f}: corroborates {n_c}/{len(planted)} centred, {n_o}/{len(planted)} best-overlapping")
    results["probe3b"] = planted

    print()
    print("Per-event HeAR response, window centred on the verified extent:")
    print("  ev  element                 centred_top          score   Cough  Breathe  Speech")
    for event in EVENTS:
        v = hear_at(hear["clean"], (event.onset, event.offset))
        a = v["centred"]["all"]
        print(
            f"  {event.index}   {event.element:22s} {v['centred']['label']:18s} "
            f"{v['centred']['score']:.3f}   {a['Cough']:.3f}  {a['Breathe']:.3f}   {a['Speech']:.3f}"
        )

    print()
    print("Confirmation over the proposer's detections. mode=centred is the design's confirmer;")
    print("mode=any_overlapping is the generous variant (best of every overlapping window).")
    results["probe3"] = {}
    for kind in PROPOSERS:
        for tau in [0.10, 0.30, 0.50]:
            for mode in ("centred", "any_overlapping"):
                rows = score_confirmer(clean, hear["clean"], kind, tau, mode)
                base = score_proposer(clean, kind, tau, empty_idx, free_idx)
                print()
                print(f"--- {kind}, tau={tau:.2f}, mode={mode}")
                print(
                    f"    YAMNet alone: recall {base['recall']:.3f} ({_fmt_int_list(base['recalled'])}), "
                    f"precision {base['precision_detections']:.3f}, "
                    f"{base['n_detections']} detections, {base['fp_detections']} FP"
                )
                print("    theta  kept  recall  events      precision  FPdet  FP/min  labels")
                for r in rows:
                    print(
                        f"    {r['hear_threshold']:.2f}  {r['kept']:4d}  {r['recall']:.3f}  "
                        f"{_fmt_int_list(r['recalled']):11s} {r['precision']:9.3f}  "
                        f"{r['fp_detections']:4d}  {r['fp_per_min']:6.2f}  {','.join(r['labels'])}"
                    )
                results["probe3"][f"{kind}|tau={tau}|{mode}"] = {
                    "base": {k: v for k, v in base.items() if k != "detections"},
                    "sweep": rows,
                }

    print()
    print("=" * 78)
    print("PROBE 4 - the bounder, against the two verified cough spans")
    print("=" * 78)
    wave, sr = _wave()
    env = envelope(wave, sr)
    print(
        f"envelope: {len(env[0])} frames, {1000.0 / sr * round(1e-3 * sr):.3f} ms hop, "
        f"grid quantisation {1000.0 * (env[0][1] - env[0][0]):.3f} ms"
    )
    print("Effect visible if: |onset error| <= 5 ms is reachable at some delta. The 1 ms grid is a")
    print("fifth of the claimed tolerance, so an error of tens of ms is the bounder's, not the grid's.")
    print()
    span_events = [e for e in EVENTS if e.span_verified]
    results["probe4"] = {}
    for seed_name in ("oracle", "chain"):
        print(f"--- seed = {seed_name}")
        print("  ev  element   delta   onset    off     onset_err_ms  offset_err_ms  floor_dB  peak_dB")
        for event in span_events:
            if seed_name == "oracle":
                seed: Optional[Tuple[float, float]] = (event.onset, event.offset)
            else:
                base = score_proposer(clean, "nonsilence", 0.10, empty_idx, free_idx)
                hits = [d for d in base["detections"] if overlap(d, (event.onset, event.offset)) > 0.0]
                seed = hits[0] if hits else None
            if seed is None:
                print(f"  {event.index}   {event.element:8s} no proposal overlaps this event")
                continue
            for delta in DELTAS_DB:
                b = bound_span(wave, sr, seed, delta, precomputed=env)
                on_err = 1000.0 * (b["onset"] - event.onset)
                off_err = 1000.0 * (b["offset"] - event.offset)
                print(
                    f"  {event.index}   {event.element:8s} {delta:5.1f}  {b['onset']:6.3f}  {b['offset']:6.3f}  "
                    f"{on_err:+12.1f}  {off_err:+13.1f}  {b['floor_db']:8.1f}  {b['peak_db']:7.1f}"
                )
                results["probe4"].setdefault(seed_name, {}).setdefault(str(event.index), []).append(
                    {"delta_db": delta, "seed": seed, "onset_err_ms": on_err, "offset_err_ms": off_err, **b}
                )
        print()

    print("=" * 78)
    print("PROBE 5 - the chain, end to end")
    print("=" * 78)
    results["probe5"] = {}
    for kind, tau, theta, mode in [
        ("nonsilence", 0.10, 0.5, "centred"),
        ("nonsilence", 0.30, 0.5, "centred"),
        ("vocab", 0.30, 0.5, "centred"),
        ("vocab", 0.50, 0.5, "centred"),
    ]:
        base = score_proposer(clean, kind, tau, empty_idx, free_idx)
        detections = base["detections"]
        print()
        print(f"--- proposer={kind} tau={tau:.2f} | confirmer theta={theta} mode={mode} | bounder delta=12 dB")
        print("  ev  element                 proposed  confirmed  label        onset_err_ms  offset_err_ms")
        rows = []
        for event in EVENTS:
            extent = (event.onset, event.offset)
            hits = [d for d in detections if overlap(d, extent) > 0.0]
            row: Dict[str, Any] = {"event": event.index, "element": event.element, "proposed": bool(hits)}
            if not hits:
                print(f"  {event.index}   {event.element:22s} no        -          -            -             -")
                rows.append(row)
                continue
            det = max(hits, key=lambda d: overlap(d, extent))
            verdict = hear_at(hear["clean"], det)[mode]
            confirmed = verdict["score"] >= theta
            row.update(
                {
                    "detection": det,
                    "hear_label": verdict["label"],
                    "hear_score": verdict["score"],
                    "confirmed": confirmed,
                }
            )
            if confirmed:
                b = bound_span(wave, sr, det, 12.0, precomputed=env)
                on_err = 1000.0 * (b["onset"] - event.onset)
                off_err = 1000.0 * (b["offset"] - event.offset)
                row.update({"onset_err_ms": on_err, "offset_err_ms": off_err, "bound": b})
                off_txt = f"{off_err:+13.1f}" if event.span_verified else " " * 8 + "n/a"
                print(
                    f"  {event.index}   {event.element:22s} yes       yes        "
                    f"{verdict['label']:12s} {on_err:+12.1f}  {off_txt}"
                )
            else:
                print(
                    f"  {event.index}   {event.element:22s} yes       no         "
                    f"{verdict['label']:12s} ({verdict['score']:.3f})  -"
                )
            rows.append(row)
        n_prop = sum(1 for r in rows if r["proposed"])
        n_conf = sum(1 for r in rows if r.get("confirmed"))
        n_label = sum(
            1
            for r in rows
            if r.get("confirmed") and r["hear_label"] in dict((e.index, e.hear_labels) for e in EVENTS)[r["event"]]
        )
        print(f"  recovered: proposed {n_prop}/6, confirmed {n_conf}/6, correct label {n_label}/6")
        event_free = [
            d
            for d in detections
            if not touches_unlabelled(d) and all(overlap(d, (e.onset, e.offset)) == 0.0 for e in EVENTS)
        ]
        survivors = sum(1 for d in event_free if hear_at(hear["clean"], d)[mode]["score"] >= theta)
        print(f"  event-free detections: {len(event_free)}, of which the confirmer keeps {survivors}")
        results["probe5"][f"{kind}|tau={tau}|theta={theta}|{mode}"] = rows

    (HERE / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print()
    print(f"wrote {HERE / 'results.json'}")


if __name__ == "__main__":
    main()
