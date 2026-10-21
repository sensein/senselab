"""Floor transfers and overlap episodes."""

from __future__ import annotations

from typing import List, Literal, Optional, Sequence

from senselab.text.tasks.conversation_analysis.config import ConversationAnalysisConfig
from senselab.text.tasks.conversation_analysis.data_structures import IPU, FloorTransfer, Overlap, Turn


def floor_transfers(turns: Sequence[Turn], config: ConversationAnalysisConfig) -> List[FloorTransfer]:
    """Floor-transfer offsets between consecutive floor-holding (non-embedded) turns.

    Args:
        turns: Turns from :func:`build_turns`.
        config: Analysis configuration.

    Returns:
        One transfer per change of floor, ``fto = next.start - previous.end``; ``|fto| <
        no_gap_no_overlap_s`` is ``"no_gap_no_overlap"``, otherwise the sign decides gap/overlap.
    """
    floor = [t for t in turns if not t.embedded]
    out: List[FloorTransfer] = []
    for prev, nxt in zip(floor, floor[1:]):
        if prev.speaker == nxt.speaker:
            continue
        fto = nxt.start - prev.end
        kind: Literal["gap", "overlap", "no_gap_no_overlap"]
        if abs(fto) < config.no_gap_no_overlap_s:
            kind = "no_gap_no_overlap"
        elif fto > 0:
            kind = "gap"
        else:
            kind = "overlap"
        out.append(
            FloorTransfer(
                id=len(out),
                from_turn_id=prev.id,
                to_turn_id=nxt.id,
                from_speaker=prev.speaker,
                to_speaker=nxt.speaker,
                fto=fto,
                transfer_type=kind,
            )
        )
    return out


def overlaps(ipus: Sequence[IPU], config: ConversationAnalysisConfig) -> List[Overlap]:
    """Overlap episodes between IPUs of different speakers, classified.

    The holder is the speaker whose IPU started first (ties: lower IPU id). An overlap is
    ``"backchannel"`` when either IPU is a backchannel; ``"non_competitive"`` when it starts within
    ``terminal_overlap_window_s`` of the end of the holder's IPU; otherwise ``"competitive"``. A
    competitive overlap is ``"successful"`` when the holder's IPU ends within
    ``interruption_yield_window_s`` of the overlap onset and the overlapper's IPU ends after it.

    Args:
        ipus: IPUs with backchannels marked.
        config: Analysis configuration.

    Returns:
        Overlaps sorted by onset.
    """
    ordered = sorted(ipus, key=lambda i: (i.start, i.id))
    episodes: List[Overlap] = []
    for a_index, holder in enumerate(ordered):
        for over in ordered[a_index + 1 :]:
            if over.start >= holder.end:
                break
            if over.speaker == holder.speaker:
                continue
            start, end = over.start, min(holder.end, over.end)
            if end <= start:
                continue
            outcome: Optional[Literal["successful", "unsuccessful"]] = None
            kind: Literal["backchannel", "non_competitive", "competitive"]
            if holder.is_backchannel or over.is_backchannel:
                kind = "backchannel"
            elif holder.end - start <= config.terminal_overlap_window_s:
                kind = "non_competitive"
            else:
                kind = "competitive"
                yielded = holder.end <= start + config.interruption_yield_window_s and over.end > holder.end
                outcome = "successful" if yielded else "unsuccessful"
            episodes.append(
                Overlap(
                    id=0,
                    start=start,
                    end=end,
                    holder_speaker=holder.speaker,
                    overlapper_speaker=over.speaker,
                    holder_ipu_id=holder.id,
                    overlapper_ipu_id=over.id,
                    overlap_type=kind,
                    interruption_outcome=outcome,
                )
            )
    episodes.sort(key=lambda o: (o.start, o.holder_ipu_id, o.overlapper_ipu_id))
    return [o.model_copy(update={"id": i}) for i, o in enumerate(episodes)]
