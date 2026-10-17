"""Words to inter-pausal units, backchannels and turns."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from senselab.text.tasks.conversation_analysis.config import ConversationAnalysisConfig
from senselab.text.tasks.conversation_analysis.data_structures import IPU, Turn, Word
from senselab.text.tasks.conversation_analysis.text_utils import (
    bare_token,
    composed_of,
    normalize_token,
    terminal_punctuation,
)
from senselab.utils.data_structures import ScriptLine

WordInput = Union[ScriptLine, Mapping[str, Any]]


def _flatten(words: Sequence[WordInput]) -> List[Dict[str, Any]]:
    flat: List[Dict[str, Any]] = []
    for item in words:
        if isinstance(item, ScriptLine):
            if item.chunks:
                for chunk in _flatten(list(item.chunks)):
                    if chunk.get("speaker") is None:
                        chunk["speaker"] = item.speaker
                    flat.append(chunk)
                continue
            flat.append(
                {
                    "speaker": item.speaker,
                    "start": item.start,
                    "end": item.end,
                    "text": item.text,
                    "confidence": item.score,
                }
            )
        else:
            record = dict(item)
            if "confidence" not in record and "score" in record:
                record["confidence"] = record["score"]
            flat.append(record)
    return flat


def _speaker_from_diarization(start: float, end: float, diarization: Sequence[ScriptLine]) -> Optional[str]:
    best: Tuple[float, float, Optional[str]] = (-1.0, float("inf"), None)
    mid = (start + end) / 2.0
    for seg in diarization:
        if seg.start is None or seg.end is None or seg.speaker is None:
            continue
        overlap = max(0.0, min(end, seg.end) - max(start, seg.start))
        distance = 0.0 if seg.start <= mid <= seg.end else min(abs(mid - seg.start), abs(mid - seg.end))
        if overlap > best[0] or (overlap == best[0] and distance < best[1]):
            best = (overlap, distance, seg.speaker)
    return best[2]


def normalize_words(
    words: Sequence[WordInput],
    diarization: Optional[Sequence[ScriptLine]] = None,
) -> List[Word]:
    """Validate, attribute and time-sort the input words.

    A ``ScriptLine`` with ``chunks`` contributes its chunks (each inheriting the line's speaker when it
    has none). Words with empty text are kept only if they contain non-whitespace characters; no word
    is dropped or rewritten for being a filler.

    Args:
        words: ``ScriptLine`` objects or mappings with ``speaker``, ``start``, ``end``, ``text`` and
            optionally ``confidence`` (or ``score``).
        diarization: Speaker segments used to attribute words that have no ``speaker``: the segment
            overlapping the word most, ties broken by distance to the word midpoint.

    Returns:
        Words sorted by ``(start, end)``, ids assigned in that order.

    Raises:
        ValueError: A word lacks a time, has ``end < start``, or has no speaker and none can be assigned.
    """
    records = _flatten(words)
    out: List[Tuple[float, float, int, Dict[str, Any]]] = []
    for position, record in enumerate(records):
        text = record.get("text")
        if text is None or not str(text).strip():
            continue
        start, end = record.get("start"), record.get("end")
        if start is None or end is None:
            raise ValueError(f"word {position} ({text!r}) has no start/end time")
        start, end = float(start), float(end)
        if end < start:
            raise ValueError(f"word {position} ({text!r}) ends before it starts: {start} > {end}")
        speaker = record.get("speaker")
        if speaker is None and diarization:
            speaker = _speaker_from_diarization(start, end, diarization)
        if speaker is None:
            raise ValueError(f"word {position} ({text!r}) has no speaker and no diarization to assign one")
        out.append((start, end, position, {**record, "speaker": str(speaker), "text": str(text)}))
    out.sort(key=lambda item: (item[0], item[1], item[2]))
    return [
        Word(
            id=i,
            speaker=record["speaker"],
            start=start,
            end=end,
            text=record["text"],
            token=normalize_token(record["text"]),
            confidence=record.get("confidence"),
        )
        for i, (start, end, _, record) in enumerate(out)
    ]


def build_ipus(words: Sequence[Word], config: ConversationAnalysisConfig) -> List[IPU]:
    """Merge each speaker's consecutive words separated by less than ``ipu_gap_s`` of silence.

    Args:
        words: Normalised words.
        config: Analysis configuration.

    Returns:
        IPUs sorted by ``(start, speaker)`` with ids in that order; backchannel fields unset.
    """
    by_speaker: Dict[str, List[Word]] = {}
    for word in words:
        by_speaker.setdefault(word.speaker, []).append(word)
    groups: List[List[Word]] = []
    for speaker_words in by_speaker.values():
        current: List[Word] = []
        current_end = float("-inf")
        for word in speaker_words:
            if current and word.start - current_end >= config.ipu_gap_s:
                groups.append(current)
                current = []
            current.append(word)
            current_end = max(current_end, word.end) if len(current) > 1 else word.end
        if current:
            groups.append(current)
    groups.sort(key=lambda g: (g[0].start, g[0].speaker))
    return [
        IPU(
            id=i,
            speaker=g[0].speaker,
            start=g[0].start,
            end=max(w.end for w in g),
            word_ids=[w.id for w in g],
            text=" ".join(w.text for w in g),
            n_words=len(g),
        )
        for i, g in enumerate(groups)
    ]


def _covered(start: float, end: float, intervals: Sequence[Tuple[float, float]]) -> bool:
    """Whether ``[start, end]`` lies inside the union of ``intervals`` (sorted by start)."""
    reach = start
    for a, b in intervals:
        if a > reach:
            break
        reach = max(reach, b)
        if reach >= end:
            return True
    return False


def _answers_question(ipu: IPU, ipus: Sequence[IPU], words: Sequence[Word]) -> bool:
    """Whether the latest other-speaker IPU ending by this IPU's onset ends in a question mark."""
    before = [o for o in ipus if o.speaker != ipu.speaker and o.end <= ipu.start]
    if not before:
        return False
    previous = max(before, key=lambda o: (o.end, o.id))
    return terminal_punctuation(words[previous.word_ids[-1]].text) == "?"


def mark_backchannels(ipus: List[IPU], words: Sequence[Word], config: ConversationAnalysisConfig) -> List[IPU]:
    """Flag backchannel IPUs.

    An IPU is a backchannel when it is short (``<= backchannel_max_duration_s`` and
    ``<= backchannel_max_words``), is lexically backchannel-like or lies wholly inside other speakers'
    speech, and does not take the floor: another speaker is still speaking at its end or starts within
    ``backchannel_floor_window_s`` of it.

    Args:
        ipus: IPUs from :func:`build_ipus`.
        words: The words they index.
        config: Analysis configuration.

    Returns:
        Copies of the IPUs with ``is_backchannel`` and ``backchannel_type`` set.
    """
    lex = config.lexicons
    out: List[IPU] = []
    for ipu in ipus:
        others = sorted((o.start, o.end) for o in ipus if o.speaker != ipu.speaker)
        short = ipu.duration <= config.backchannel_max_duration_s and ipu.n_words <= config.backchannel_max_words
        is_bc = False
        if short and others:
            tokens = [bare_token(words[i].token) for i in ipu.word_ids]
            tokens = [t for t in tokens if t]
            lexical = composed_of(tokens, lex.backchannel)
            overlapped = _covered(ipu.start, ipu.end, others)
            continues = any(b > ipu.end and a <= ipu.end + config.backchannel_floor_window_s for a, b in others)
            is_bc = (lexical or overlapped) and continues and not _answers_question(ipu, ipus, words)
        bc_type = None
        if is_bc:
            touches = any(a < ipu.end and b > ipu.start for a, b in others)
            bc_type = "overlapping" if touches else "in_pause"
        out.append(ipu.model_copy(update={"is_backchannel": is_bc, "backchannel_type": bc_type}))
    return out


def build_turns(ipus: Sequence[IPU], words: Sequence[Word]) -> List[Turn]:
    """Group non-backchannel IPUs into turns.

    IPUs are visited in start order. One floor-holding turn is open at a time: an IPU by its speaker
    extends it; an IPU by another speaker that lies wholly inside it becomes an ``embedded`` turn and
    leaves the floor where it is; any other IPU by another speaker opens a new floor-holding turn.

    Args:
        ipus: IPUs with backchannels marked.
        words: The words they index.

    Returns:
        Turns in start order, ids in that order.
    """
    drafts: List[Dict[str, Any]] = []
    floor: Optional[Dict[str, Any]] = None
    for ipu in sorted((i for i in ipus if not i.is_backchannel), key=lambda i: (i.start, i.id)):
        if floor is not None and ipu.speaker == floor["speaker"]:
            floor["ipus"].append(ipu)
            floor["end"] = max(floor["end"], ipu.end)
        elif floor is not None and ipu.start >= floor["start"] and ipu.end <= floor["end"]:
            drafts.append({"speaker": ipu.speaker, "start": ipu.start, "end": ipu.end, "ipus": [ipu], "embedded": True})
        else:
            floor = {"speaker": ipu.speaker, "start": ipu.start, "end": ipu.end, "ipus": [ipu], "embedded": False}
            drafts.append(floor)
    drafts.sort(key=lambda d: (d["start"], d["embedded"]))
    turns: List[Turn] = []
    for i, d in enumerate(drafts):
        word_ids = sorted((w for ipu in d["ipus"] for w in ipu.word_ids), key=lambda w: (words[w].start, w))
        texts = [words[w].text for w in word_ids]
        turns.append(
            Turn(
                id=i,
                speaker=d["speaker"],
                start=d["start"],
                end=d["end"],
                ipu_ids=[ipu.id for ipu in d["ipus"]],
                word_ids=word_ids,
                words=texts,
                text=" ".join(texts),
                n_words=len(word_ids),
                embedded=d["embedded"],
            )
        )
    return turns
