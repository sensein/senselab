"""Per-speaker summaries over the analysis tables."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from senselab.text.tasks.conversation_analysis.data_structures import (
    IPU,
    QUESTION_ACTS,
    DialogueAct,
    DialogueActSegment,
    Disfluency,
    FloorTransfer,
    Overlap,
    Turn,
    TurnMarkers,
    Word,
)
from senselab.text.tasks.conversation_analysis.markers import POLITENESS_STRATEGIES
from senselab.utils.data_structures import ScriptLine


def _mean(values: Sequence[float]) -> Optional[float]:
    return float(np.mean(values)) if len(values) else None


def _median(values: Sequence[float]) -> Optional[float]:
    return float(np.median(values)) if len(values) else None


def _sd(values: Sequence[float]) -> Optional[float]:
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def _rate(count: float, denominator: Optional[float], scale: float = 1.0) -> Optional[float]:
    if not denominator:
        return None
    return float(count) * scale / denominator


def _ratio(numerator: float, denominator: float) -> Optional[float]:
    return float(numerator) / denominator if denominator else None


def speaker_summaries(
    speakers: Sequence[str],
    duration: float,
    words: Sequence[Word],
    ipus: Sequence[IPU],
    turns: Sequence[Turn],
    transfers: Sequence[FloorTransfer],
    overlaps: Sequence[Overlap],
    disfluencies: Sequence[Disfluency],
    segments: Sequence[DialogueActSegment],
    markers: Sequence[TurnMarkers],
    diarization: Optional[Sequence[ScriptLine]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Per-speaker turn-taking, disfluency, dialogue-act and marker summary.

    Rates "per min" divide by the conversation's duration, except disfluency rates per minute, which
    divide by the speaker's own speaking time within turns. "Per 100 words" divides by the speaker's
    words within turns. Undefined values are ``None``.

    Args:
        speakers: Speaker labels.
        duration: Conversation span in seconds.
        words: Words.
        ipus: IPUs.
        turns: Turns.
        transfers: Floor transfers.
        overlaps: Overlaps.
        disfluencies: Disfluencies.
        segments: Labelled dialogue-act segments.
        markers: Turn markers.
        diarization: Diarization segments, for ``diarized_speaking_time_s``.

    Returns:
        ``{speaker: {metric: value}}``.
    """
    minutes = duration / 60.0 if duration > 0 else None
    total_speaking = sum(i.duration for i in ipus)
    ipu_by_id = {i.id: i for i in ipus}
    rule_acts = [a.value for a in DialogueAct]
    all_acts = rule_acts + sorted({g.act for g in segments if g.act is not None} - set(rule_acts))
    out: Dict[str, Dict[str, Any]] = {}
    for spk in speakers:
        s: Dict[str, Any] = {}
        my_ipus = [i for i in ipus if i.speaker == spk]
        my_turns = [t for t in turns if t.speaker == spk]
        turn_words = sum(t.n_words for t in my_turns)
        turn_time = sum(ipu_by_id[i].duration for t in my_turns for i in t.ipu_ids)
        turn_minutes = turn_time / 60.0 if turn_time > 0 else None
        speaking = sum(i.duration for i in my_ipus)

        s["n_words"] = sum(1 for w in words if w.speaker == spk)
        s["n_turn_words"] = turn_words
        s["speaking_time_s"] = speaking
        s["speaking_share"] = _ratio(speaking, total_speaking)
        if diarization is not None:
            s["diarized_speaking_time_s"] = float(
                sum(
                    (seg.end or 0.0) - (seg.start or 0.0)
                    for seg in diarization
                    if seg.speaker == spk and seg.start is not None and seg.end is not None
                )
            )

        durations = [t.duration for t in my_turns]
        nwords = [float(t.n_words) for t in my_turns]
        s["n_turns"] = len(my_turns)
        s["n_floor_turns"] = sum(1 for t in my_turns if not t.embedded)
        s["turns_per_min"] = _rate(len(my_turns), minutes)
        s["turn_duration_mean_s"] = _mean(durations)
        s["turn_duration_median_s"] = _median(durations)
        s["turn_duration_sd_s"] = _sd(durations)
        s["turn_words_mean"] = _mean(nwords)
        s["turn_words_median"] = _median(nwords)
        s["turn_words_sd"] = _sd(nwords)
        s["longest_monologue_s"] = max(durations) if durations else None
        s["longest_monologue_words"] = int(max(nwords)) if nwords else None

        bcs = [i for i in my_ipus if i.is_backchannel]
        s["n_backchannels"] = len(bcs)
        s["n_backchannels_overlapping"] = sum(1 for i in bcs if i.backchannel_type == "overlapping")
        s["n_backchannels_in_pause"] = sum(1 for i in bcs if i.backchannel_type == "in_pause")
        s["backchannels_per_min"] = _rate(len(bcs), minutes)

        competitive = [o for o in overlaps if o.overlap_type == "competitive"]
        made = [o for o in competitive if o.overlapper_speaker == spk]
        received = [o for o in competitive if o.holder_speaker == spk]
        s["interruptions_made"] = len(made)
        s["interruptions_made_successful"] = sum(1 for o in made if o.interruption_outcome == "successful")
        s["interruptions_made_unsuccessful"] = sum(1 for o in made if o.interruption_outcome == "unsuccessful")
        s["interruptions_received"] = len(received)
        s["interruptions_received_successful"] = sum(1 for o in received if o.interruption_outcome == "successful")
        s["interruptions_received_unsuccessful"] = sum(1 for o in received if o.interruption_outcome == "unsuccessful")
        wins = s["interruptions_made_successful"] + s["interruptions_received_unsuccessful"]
        s["floor_holding_index"] = _ratio(wins, len(made) + len(received))
        s["n_overlaps"] = sum(1 for o in overlaps if spk in (o.holder_speaker, o.overlapper_speaker))

        taken = [f for f in transfers if f.to_speaker == spk]
        ftos = [f.fto for f in taken]
        s["n_floor_takings"] = len(taken)
        s["fto_mean_s"] = _mean(ftos)
        s["fto_median_s"] = _median(ftos)
        s["prop_gap"] = _ratio(sum(1 for f in taken if f.transfer_type == "gap"), len(taken))
        s["prop_overlap"] = _ratio(sum(1 for f in taken if f.transfer_type == "overlap"), len(taken))
        s["prop_no_gap_no_overlap"] = _ratio(
            sum(1 for f in taken if f.transfer_type == "no_gap_no_overlap"), len(taken)
        )
        latencies = [f.fto for f in taken if f.after_question]
        s["n_responses_after_question"] = len(latencies)
        s["response_latency_after_question_mean_s"] = _mean(latencies)
        s["response_latency_after_question_median_s"] = _median(latencies)

        s.update(_disfluency_summary(spk, disfluencies, turn_words, turn_minutes))

        my_segments = [g for g in segments if g.speaker == spk]
        questions = sum(1 for g in my_segments if g.act in QUESTION_ACTS)
        s["questions_asked"] = questions
        s["questions_per_min"] = _rate(questions, minutes)
        acts: Dict[str, int] = {}
        for g in my_segments:
            if g.act is not None:
                acts[g.act] = acts.get(g.act, 0) + 1
        for act in all_acts:
            s[f"act_{act}"] = acts.get(act, 0)

        my_markers = [m for m in markers if m.speaker == spk]
        s["discourse_markers_per_100_words"] = _rate(sum(m.discourse_markers for m in my_markers), turn_words, 100)
        s["hedges_per_100_words"] = _rate(sum(m.hedges for m in my_markers), turn_words, 100)
        for pron in ("i", "we", "you"):
            total = sum(getattr(m, f"pronoun_{pron}") for m in my_markers)
            s[f"pronoun_{pron}_per_100_words"] = _rate(total, turn_words, 100)
        for strategy in POLITENESS_STRATEGIES:
            s[f"politeness_{strategy}"] = sum(m.politeness.get(strategy, 0) for m in my_markers)
        aligned = [m.lexical_alignment for m in my_markers if m.lexical_alignment is not None]
        s["lexical_alignment_mean"] = _mean(aligned)
        out[spk] = s
    return out


def _disfluency_summary(
    spk: str, disfluencies: Sequence[Disfluency], turn_words: int, turn_minutes: Optional[float]
) -> Dict[str, Any]:
    s: Dict[str, Any] = {}
    mine = [d for d in disfluencies if d.speaker == spk]
    names = {
        "filled_pause": "filled_pauses",
        "repetition": "repetitions",
        "cutoff": "cutoffs",
        "prolongation": "prolongations",
    }
    for kind, name in names.items():
        n = sum(1 for d in mine if d.disfluency_type == kind)
        s[f"n_{name}"] = n
        s[f"{name}_per_100_words"] = _rate(n, turn_words, 100)
        s[f"{name}_per_min"] = _rate(n, turn_minutes)
    s["n_disfluencies"] = len(mine)
    s["disfluencies_per_100_words"] = _rate(len(mine), turn_words, 100)
    s["disfluencies_per_min"] = _rate(len(mine), turn_minutes)

    fillers: List[Disfluency] = [d for d in mine if d.disfluency_type == "filled_pause"]
    n_uh = sum(1 for d in fillers if d.filler_family == "uh")
    n_um = sum(1 for d in fillers if d.filler_family == "um")
    s["n_uh"] = n_uh
    s["n_um"] = n_um
    s["uh_um_ratio"] = _ratio(n_uh, n_um)
    s["filler_duration_mean_s"] = _mean([d.duration for d in fillers])
    s["filler_duration_median_s"] = _median([d.duration for d in fillers])
    s["filler_pause_before_mean_s"] = _mean([d.pause_before for d in fillers if d.pause_before is not None])
    s["filler_pause_after_mean_s"] = _mean([d.pause_after for d in fillers if d.pause_after is not None])
    for position in ("turn_initial", "turn_medial", "turn_final", "turn_isolated"):
        s[f"fillers_{position}"] = sum(1 for d in fillers if d.position == position)
    s["fillers_pre_pause"] = sum(1 for d in fillers if d.pre_pause)
    s["fillers_post_pause"] = sum(1 for d in fillers if d.post_pause)
    s["p_filler_to_pause"] = _ratio(sum(1 for d in fillers if d.next_event == "pause"), len(fillers))
    s["p_filler_to_speaker_change"] = _ratio(sum(1 for d in fillers if d.next_event == "speaker_change"), len(fillers))
    return s
