"""Filled pauses, repetitions, cut-offs and prolongations."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from senselab.text.tasks.conversation_analysis.config import ConversationAnalysisConfig, Lexicons
from senselab.text.tasks.conversation_analysis.data_structures import Disfluency, Turn, Word
from senselab.text.tasks.conversation_analysis.text_utils import bare_token


def filler_family(token: str, lexicons: Lexicons) -> Optional[Literal["uh", "um", "other"]]:
    """The filled-pause family of a normalised token, case- and bracket-insensitive.

    Args:
        token: Normalised token (``"[uh]"``, ``"um"``, ``"uh-"`` ...).
        lexicons: Lexicons.

    Returns:
        ``"uh"``, ``"um"`` or ``"other"``; ``None`` when the token is not a filled pause.
    """
    bare = bare_token(token).rstrip("-")
    if bare in lexicons.uh_fillers:
        return "uh"
    if bare in lexicons.um_fillers:
        return "um"
    if bare in lexicons.other_fillers:
        return "other"
    return None


def is_cutoff(token: str) -> bool:
    """Whether a normalised token is a cut-off (ends in a hyphen or dash after at least one letter).

    Args:
        token: Normalised token.

    Returns:
        True for ``"wh-"``, ``"basi—"``; False for ``"mm-hm"`` or ``"-"``.
    """
    bare = bare_token(token)
    stem = bare.rstrip("-—–")
    return stem != bare and any(c.isalpha() for c in stem)


def _letters(token: str) -> int:
    return sum(1 for c in token if c.isalpha())


def detect_disfluencies(
    turns: Sequence[Turn],
    words: Sequence[Word],
    config: ConversationAnalysisConfig,
) -> List[Disfluency]:
    """Every disfluency in the turns' words. Backchannel IPUs are not turns and are not scanned.

    Args:
        turns: Turns.
        words: Words.
        config: Analysis configuration.

    Returns:
        Events sorted by onset, then type.
    """
    lex = config.lexicons
    events: List[Disfluency] = []
    ordered_turns = sorted(turns, key=lambda t: (t.start, t.id))

    for turn in ordered_turns:
        tws = [words[w] for w in turn.word_ids]
        for k, word in enumerate(tws):
            family = filler_family(word.token, lex)
            if family is not None:
                later_other = any(t.speaker != turn.speaker and t.end > word.end for t in turns)
                events.append(_filler(word, k, tws, turn, family, later_other, config))
            elif is_cutoff(word.token):
                events.append(_event(turn, "cutoff", [word]))

        content: List[Tuple[Word, str]] = [
            (w, bare_token(w.token)) for w in tws if filler_family(w.token, lex) is None and bare_token(w.token)
        ]
        i = 0
        while i < len(content):
            matched = 0
            for size in range(min(config.repetition_max_ngram, (len(content) - i) // 2), 0, -1):
                first = [t for _, t in content[i : i + size]]
                second = [t for _, t in content[i + size : i + 2 * size]]
                if first == second and not all(t in lex.repetition_exempt for t in first):
                    events.append(_event(turn, "repetition", [w for w, _ in content[i : i + 2 * size]]))
                    matched = size
                    break
            i += matched or 1

    events.extend(_prolongations(ordered_turns, words, config))
    events.sort(key=lambda e: (e.start, e.disfluency_type, e.word_ids))
    return [e.model_copy(update={"id": i}) for i, e in enumerate(events)]


def _event(turn: Turn, kind: Literal["repetition", "cutoff", "prolongation"], ws: List[Word]) -> Disfluency:
    return Disfluency(
        id=0,
        speaker=turn.speaker,
        disfluency_type=kind,
        word_ids=[w.id for w in ws],
        text=" ".join(w.text for w in ws),
        start=min(w.start for w in ws),
        end=max(w.end for w in ws),
        turn_id=turn.id,
    )


def _filler(
    word: Word,
    k: int,
    tws: Sequence[Word],
    turn: Turn,
    family: Literal["uh", "um", "other"],
    later_other: bool,
    config: ConversationAnalysisConfig,
) -> Disfluency:
    n = len(tws)
    position: Literal["turn_initial", "turn_medial", "turn_final", "turn_isolated"]
    if n == 1:
        position = "turn_isolated"
    elif k == 0:
        position = "turn_initial"
    elif k == n - 1:
        position = "turn_final"
    else:
        position = "turn_medial"
    pause_before = word.start - tws[k - 1].end if k > 0 else None
    pause_after = tws[k + 1].start - word.end if k < n - 1 else None
    next_event: Literal["pause", "word", "speaker_change", "end"]
    if pause_after is None:
        next_event = "speaker_change" if later_other else "end"
    elif pause_after >= config.silent_pause_min_s:
        next_event = "pause"
    else:
        next_event = "word"
    return Disfluency(
        id=0,
        speaker=turn.speaker,
        disfluency_type="filled_pause",
        word_ids=[word.id],
        text=word.text,
        start=word.start,
        end=word.end,
        turn_id=turn.id,
        filler_family=family,
        position=position,
        pause_before=pause_before,
        pause_after=pause_after,
        pre_pause=None if pause_after is None else pause_after >= config.silent_pause_min_s,
        post_pause=None if pause_before is None else pause_before >= config.silent_pause_min_s,
        next_event=next_event,
    )


def _prolongations(
    turns: Sequence[Turn], words: Sequence[Word], config: ConversationAnalysisConfig
) -> List[Disfluency]:
    lex = config.lexicons
    per_speaker: Dict[str, List[Tuple[Word, Turn, float]]] = {}
    for turn in turns:
        for wid in turn.word_ids:
            w = words[wid]
            letters = _letters(bare_token(w.token))
            if w.duration <= 0 or letters == 0 or filler_family(w.token, lex) is not None:
                continue
            per_speaker.setdefault(turn.speaker, []).append((w, turn, w.duration / letters))
    out: List[Disfluency] = []
    for items in per_speaker.values():
        if len(items) < config.prolongation_min_words:
            continue
        values = np.array([v for _, _, v in items])
        sd = float(values.std())
        if sd == 0.0:
            continue
        mean = float(values.mean())
        for w, turn, v in items:
            z = (v - mean) / sd
            if z > config.prolongation_z:
                out.append(_event(turn, "prolongation", [w]).model_copy(update={"zscore": z}))
    return out
