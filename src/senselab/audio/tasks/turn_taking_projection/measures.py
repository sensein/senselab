"""Dyadic measures read from VAP frames against the caller's turns, IPUs and backchannels.

Every measure here is a reading of the model's projection at moments the caller's annotation
defines; nothing is re-segmented from the frames themselves. Three event families:

- **Shifts** (consecutive turns by different speakers): the next speaker's ``p_now`` over the
  window before the floor changed, i.e. how far ahead the model saw the turn end coming.
- **Pauses** (mutual silences between IPUs): the other speaker's ``p_now`` shortly into the pause,
  called as shift or hold and scored against what happened.
- **Backchannel opportunities** (peaks of a listener's ``p_bc``): matched against the backchannels
  that listener produced.

Window lengths and thresholds come from ``data/parameters.json``, where each carries its
derivation; the continuous score behind every call is reported, so a caller can re-threshold.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from senselab.audio.tasks.turn_taking_projection.result import VAPResult
from senselab.audio.tasks.turn_taking_projection.separation import parameters
from senselab.utils.data_structures import ScriptLine

Interval = Tuple[str, float, float]


def _intervals(lines: Optional[Sequence[ScriptLine]], speakers: Tuple[str, str]) -> List[Interval]:
    out: List[Interval] = []
    for line in lines or ():
        if line.speaker in speakers and line.start is not None and line.end is not None and line.end > line.start:
            out.append((line.speaker, float(line.start), float(line.end)))
    return sorted(out, key=lambda x: (x[1], x[2]))


def _window_mean(times: np.ndarray, values: np.ndarray, lo: float, hi: float) -> Optional[float]:
    """Mean of frames with ``lo < time <= hi``; the single frame at ``hi`` when the window holds none."""
    mask = (times > lo) & (times <= hi)
    if mask.any():
        return float(values[mask].mean())
    return _at(times, values, hi)


def _at(times: np.ndarray, values: np.ndarray, t: float) -> Optional[float]:
    """The latest frame at or before ``t`` (the model is causal), or ``None`` before the first frame."""
    index = int(np.searchsorted(times, t, side="right")) - 1
    return float(values[index]) if index >= 0 else None


def derive_backchannels(
    ipus: Sequence[ScriptLine], turns: Sequence[ScriptLine], speakers: Tuple[str, str], max_seconds: float
) -> List[Interval]:
    """Short listener IPUs whose midpoint lies inside the other speaker's turn.

    Args:
        ipus: IPUs.
        turns: Turns.
        speakers: The dyad.
        max_seconds: Longest IPU taken as a backchannel.

    Returns:
        ``(speaker, start, end)`` per backchannel.
    """
    turn_spans = _intervals(turns, speakers)
    out: List[Interval] = []
    for speaker, start, end in _intervals(ipus, speakers):
        if end - start > max_seconds:
            continue
        mid = 0.5 * (start + end)
        if any(s != speaker and ts <= mid <= te for s, ts, te in turn_spans):
            out.append((speaker, start, end))
    return out


def shift_events(vap: VAPResult, turns: Sequence[ScriptLine]) -> List[Dict[str, Any]]:
    """One row per speaker change between consecutive turns.

    The floor changes at ``min(previous end, next start)``: the previous speaker's end after a gap,
    the next speaker's onset in an overlap. ``p_now_next_mean`` is the next speaker's mean ``p_now``
    over ``projection_window_seconds`` before that moment; ``lead_s`` is how long ``p_now`` of the
    next speaker had been continuously above ``shift_probability_threshold`` at that moment.

    Args:
        vap: The frames.
        turns: Turns with speaker labels.

    Returns:
        Rows with ``from``, ``to``, ``prev_end``, ``next_start``, ``gap_s``, ``p_now_next_mean``,
        ``p_now_next_at_change``, ``p_future_next_mean``, ``projected``, ``lead_s``.
    """
    params = parameters()
    window = float(params["projection_window_seconds"])
    threshold = float(params["shift_probability_threshold"])
    times = vap.times
    spans = _intervals(turns, vap.speakers)
    rows: List[Dict[str, Any]] = []
    for (prev_spk, prev_start, prev_end), (next_spk, next_start, _) in zip(spans, spans[1:]):
        if prev_spk == next_spk:
            continue
        change = min(prev_end, next_start)
        p_now = vap.series("p_now", next_spk)
        p_future = vap.series("p_future", next_spk)
        mean_now = _window_mean(times, p_now, change - window, change)
        at_change = _at(times, p_now, change)
        lead = 0.0
        index = int(np.searchsorted(times, change, side="right")) - 1
        while index >= 0 and times[index] > prev_start and p_now[index] > threshold:
            lead = change - float(times[index]) + 1.0 / vap.frame_rate
            index -= 1
        rows.append(
            {
                "from": prev_spk,
                "to": next_spk,
                "prev_end": prev_end,
                "next_start": next_start,
                "gap_s": next_start - prev_end,
                "p_now_next_mean": mean_now,
                "p_now_next_at_change": at_change,
                "p_future_next_mean": _window_mean(times, p_future, change - window, change),
                "projected": None if mean_now is None else bool(mean_now > threshold),
                "lead_s": lead,
            }
        )
    return rows


def pause_events(vap: VAPResult, ipus: Sequence[ScriptLine], backchannels: Sequence[Interval]) -> List[Dict[str, Any]]:
    """One row per mutual silence of at least ``min_pause_seconds`` between non-backchannel IPUs.

    The last speaker is the one whose IPU ended last before the silence; the outcome is ``shift``
    when the next IPU is the other speaker's. The score is the other speaker's mean ``p_now`` from
    ``pause_read_offset_seconds`` to ``pause_read_offset_seconds + pause_read_seconds`` into the
    pause, clipped to the pause. A pause with a backchannel anywhere in it is not a mutual silence:
    its row carries ``backchannel_in_pause=True`` and is left out of the summaries, because the
    listener's own voice in the read window moves ``p_now`` towards the listener whatever the
    floor does next.

    Args:
        vap: The frames.
        ipus: IPUs with speaker labels.
        backchannels: Backchannels to leave out of the IPU stream.

    Returns:
        Rows with ``last``, ``next``, ``start``, ``end``, ``outcome``, ``p_shift``, ``predicted``,
        ``correct``, ``backchannel_in_pause``.
    """
    params = parameters()
    offset = float(params["pause_read_offset_seconds"])
    read = float(params["pause_read_seconds"])
    min_pause = float(params["min_pause_seconds"])
    threshold = float(params["shift_probability_threshold"])
    bc = {(s, round(a, 4), round(b, 4)) for s, a, b in backchannels}
    stream = [x for x in _intervals(ipus, vap.speakers) if (x[0], round(x[1], 4), round(x[2], 4)) not in bc]
    times = vap.times
    rows: List[Dict[str, Any]] = []
    latest_end = -np.inf
    last_speaker: Optional[str] = None
    for speaker, start, end in stream:
        if last_speaker is not None and start - latest_end >= min_pause:
            other = vap.speakers[1] if last_speaker == vap.speakers[0] else vap.speakers[0]
            lo = latest_end + offset
            hi = min(start, lo + read)
            score = _window_mean(times, vap.series("p_now", other), lo, hi)
            outcome = "shift" if speaker != last_speaker else "hold"
            predicted = None if score is None else ("shift" if score > threshold else "hold")
            rows.append(
                {
                    "last": last_speaker,
                    "next": speaker,
                    "start": latest_end,
                    "end": start,
                    "outcome": outcome,
                    "p_shift": score,
                    "predicted": predicted,
                    "correct": None if predicted is None else predicted == outcome,
                    "backchannel_in_pause": any(a < start and b > latest_end for _, a, b in backchannels),
                }
            )
        if end >= latest_end:
            latest_end, last_speaker = end, speaker
    return rows


def find_peaks(times: np.ndarray, values: np.ndarray, threshold: float, min_separation: float) -> List[int]:
    """Indices of local maxima above ``threshold``, greedily kept tallest-first ``min_separation`` apart."""
    if values.size < 1:
        return []
    left = np.concatenate([[-np.inf], values[:-1]])
    right = np.concatenate([values[1:], [-np.inf]])
    candidates = np.flatnonzero((values > threshold) & (values >= left) & (values > right))
    kept: List[int] = []
    for index in sorted(candidates, key=lambda i: -values[i]):
        if all(abs(times[index] - times[k]) >= min_separation for k in kept):
            kept.append(int(index))
    return sorted(kept)


def backchannel_events(
    vap: VAPResult, backchannels: Sequence[Interval]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Each listener's ``p_bc`` peaks, matched one-to-one to the backchannels they produced.

    A backchannel answers an opportunity when its onset lies within ``bc_response_window_seconds``
    of the peak; matching is greedy in time order.

    Args:
        vap: The frames, with ``p_bc_*`` columns.
        backchannels: ``(speaker, start, end)``.

    Returns:
        ``(opportunities, produced)``: one row per peak (``listener``, ``time``, ``p_bc``,
        ``answered_by``) and one per backchannel (``speaker``, ``start``, ``end``, ``p_bc_at_onset``,
        ``answers``).
    """
    params = parameters()
    threshold = float(params["bc_peak_threshold"])
    separation = float(params["bc_peak_min_separation_seconds"])
    early, late = (float(v) for v in params["bc_response_window_seconds"])
    times = vap.times
    opportunities: List[Dict[str, Any]] = []
    produced: List[Dict[str, Any]] = []
    for listener in vap.speakers:
        p_bc = vap.series("p_bc", listener)
        peak_times = [float(times[i]) for i in find_peaks(times, p_bc, threshold, separation)]
        onsets = [(a, b) for s, a, b in backchannels if s == listener]
        answered_by: List[Optional[int]] = [None] * len(peak_times)
        answers: List[Optional[int]] = [None] * len(onsets)
        for k, (onset, _) in enumerate(onsets):
            for j, peak_time in enumerate(peak_times):
                if answered_by[j] is None and early <= onset - peak_time <= late:
                    answered_by[j], answers[k] = k, j
                    break
        peaks = [
            {"listener": listener, "time": t, "p_bc": _at(times, p_bc, t), "answered_by": answered_by[j]}
            for j, t in enumerate(peak_times)
        ]
        own = [
            {"speaker": listener, "start": a, "end": b, "p_bc_at_onset": _at(times, p_bc, a), "answers": answers[k]}
            for k, (a, b) in enumerate(onsets)
        ]
        opportunities += peaks
        produced += own
    return opportunities, produced


def _mean(values: List[Optional[float]]) -> Optional[float]:
    kept = [v for v in values if v is not None]
    return float(np.mean(kept)) if kept else None


def _auc(positive: List[float], negative: List[float]) -> Optional[float]:
    """Probability a random positive scores above a random negative (ties count half)."""
    if not positive or not negative:
        return None
    pos = np.asarray(positive)[:, None]
    neg = np.asarray(negative)[None, :]
    return float(((pos > neg).sum() + 0.5 * (pos == neg).sum()) / (pos.size * neg.size))


def _pause_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [r for r in rows if r["p_shift"] is not None and not r["backchannel_in_pause"]]
    shifts = [r["p_shift"] for r in scored if r["outcome"] == "shift"]
    holds = [r["p_shift"] for r in scored if r["outcome"] == "hold"]
    recall = [
        float(np.mean([r["correct"] for r in scored if r["outcome"] == k]))
        for k in ("shift", "hold")
        if any(r["outcome"] == k for r in scored)
    ]
    return {
        "n_pauses": len(rows),
        "n_with_backchannel": sum(1 for r in rows if r["backchannel_in_pause"]),
        "n_shift": len(shifts),
        "n_hold": len(holds),
        "accuracy": float(np.mean([r["correct"] for r in scored])) if scored else None,
        "balanced_accuracy": float(np.mean(recall)) if len(recall) == 2 else None,
        "mean_p_shift_at_shifts": _mean(list(shifts)),
        "mean_p_shift_at_holds": _mean(list(holds)),
        "auc": _auc(shifts, holds),
    }


def summarize_events(
    vap: VAPResult,
    turns: Sequence[ScriptLine],
    ipus: Sequence[ScriptLine],
    backchannels: Optional[Sequence[ScriptLine]] = None,
) -> Dict[str, Any]:
    """All three event families, per speaker and overall.

    Args:
        vap: The frames.
        turns: Turns with speaker labels; consecutive turns by different speakers are shifts.
        ipus: IPUs with speaker labels, backchannels included or not.
        backchannels: The backchannels produced. ``None`` derives them from ``ipus`` (a listener
            IPU no longer than ``backchannel_max_seconds`` inside the other speaker's turn).

    Returns:
        ``{"speakers": {label: {...}}, "overall": {...}, "events": {...}, "parameters": {...}}``.
    """
    params = parameters()
    speakers = vap.speakers
    if backchannels is None:
        bcs = derive_backchannels(ipus, turns, speakers, float(params["backchannel_max_seconds"]))
        bc_source = "derived"
    else:
        bcs = _intervals(backchannels, speakers)
        bc_source = "given"
    shifts = shift_events(vap, turns)
    pauses = pause_events(vap, ipus, bcs)
    if vap.has_backchannel:
        opportunities, produced = backchannel_events(vap, bcs)
    else:
        opportunities, produced = [], []

    bc_keys = {(s, round(a, 4), round(b, 4)) for s, a, b in bcs}
    floor_ipus = [x for x in _intervals(ipus, speakers) if (x[0], round(x[1], 4), round(x[2], 4)) not in bc_keys]
    speech = {s: sum(b - a for spk, a, b in floor_ipus if spk == s) for s in speakers}
    per_speaker: Dict[str, Any] = {}
    for speaker in speakers:
        other = speakers[1] if speaker == speakers[0] else speakers[0]
        ends = [r for r in shifts if r["from"] == speaker]
        takes = [r for r in shifts if r["to"] == speaker]
        listener_opportunities = [o for o in opportunities if o["listener"] == speaker]
        own_bc = [b for b in produced if b["speaker"] == speaker]
        other_minutes = speech[other] / 60.0
        per_speaker[speaker] = {
            "speech_s": speech[speaker],
            "n_turns": sum(1 for s, _, _ in _intervals(turns, speakers) if s == speaker),
            "n_ipus": sum(1 for s, _, _ in floor_ipus if s == speaker),
            "turn_ends": {
                "n": len(ends),
                "mean_p_now_next": _mean([r["p_now_next_mean"] for r in ends]),
                "projected_rate": _mean([float(r["projected"]) for r in ends if r["projected"] is not None]),
                "mean_lead_s": _mean([r["lead_s"] for r in ends]),
            },
            "turn_takes": {
                "n": len(takes),
                "mean_gap_s": _mean([r["gap_s"] for r in takes]),
                "median_gap_s": float(np.median([r["gap_s"] for r in takes])) if takes else None,
                "mean_p_now_self_before": _mean([r["p_now_next_mean"] for r in takes]),
            },
            "pauses_after_own_ipus": _pause_summary([r for r in pauses if r["last"] == speaker]),
            "backchannel": {
                "n_produced": len(own_bc),
                "n_opportunities": len(listener_opportunities),
                "n_answered": sum(1 for o in listener_opportunities if o["answered_by"] is not None),
                "responsiveness": (
                    sum(1 for o in listener_opportunities if o["answered_by"] is not None) / len(listener_opportunities)
                    if listener_opportunities
                    else None
                ),
                "timed_fraction": (
                    sum(1 for b in own_bc if b["answers"] is not None) / len(own_bc) if own_bc else None
                ),
                "produced_per_min_of_other_speech": len(own_bc) / other_minutes if other_minutes > 0 else None,
                "opportunities_per_min_of_other_speech": len(listener_opportunities) / other_minutes
                if other_minutes > 0
                else None,
                "mean_p_bc_at_onset": _mean([b["p_bc_at_onset"] for b in own_bc]),
            }
            if vap.has_backchannel
            else None,
        }
    overall = {
        "n_shifts": len(shifts),
        "mean_p_now_next": _mean([r["p_now_next_mean"] for r in shifts]),
        "projected_rate": _mean([float(r["projected"]) for r in shifts if r["projected"] is not None]),
        "pauses": _pause_summary(pauses),
        "backchannel_source": bc_source,
    }
    return {
        "speakers": per_speaker,
        "overall": overall,
        "events": {
            "shifts": shifts,
            "pauses": pauses,
            "backchannel_opportunities": opportunities,
            "backchannels": produced,
        },
        "parameters": dict(params),
    }
