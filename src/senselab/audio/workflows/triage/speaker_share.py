"""How the diarizer's speakers divide one task extent: every speaker's seconds, and the time a second voice holds alone.

A reading, not a judgement. SPEECH writes it beside each task extent and VERDICT's speaker gate reads
``exclusive_share``. The derivation of the exclusive reading is
``specs/20261012-r37-words-speaker/design.md`` ("C1. The speaker gate on exclusive time").
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

Interval = tuple[float, float]


def merge_intervals(extents: Sequence[Interval]) -> list[Interval]:
    """Coalesce touching or overlapping intervals.

    Args:
        extents: The intervals, in any order.

    Returns:
        The disjoint intervals, earliest first.
    """
    out: list[Interval] = []
    for start, end in sorted((float(a), float(b)) for a, b in extents):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def subtract_intervals(spans: Sequence[Interval], cut: Sequence[Interval]) -> list[Interval]:
    """The parts of ``spans`` no interval of ``cut`` covers.

    Args:
        spans: The intervals to keep from.
        cut: The intervals removed.

    Returns:
        The remaining pieces, earliest first.
    """
    removed = merge_intervals(cut)
    kept: list[Interval] = []
    for start, end in merge_intervals(spans):
        cursor = start
        for low, high in removed:
            if high <= cursor or low >= end:
                continue
            if low > cursor:
                kept.append((cursor, low))
            cursor = max(cursor, high)
        if end > cursor:
            kept.append((cursor, end))
    return kept


def _clip(extent: Interval, spans: Sequence[Interval]) -> list[Interval]:
    return [(max(a, extent[0]), min(b, extent[1])) for a, b in spans if min(b, extent[1]) > max(a, extent[0])]


def speakers_within(
    extent: Interval, segments: Sequence[tuple[str, str, Interval]], *, exclusive_min_s: float
) -> dict[str, Any]:
    """How the diarizer's speakers divide one task extent between them.

    Args:
        extent: The task extent.
        segments: ``(entity_id, speaker label, extent)`` per diarized segment, on the same clock.
        exclusive_min_s: The exclusive second-voice time under which a second voice holds no turn of its own;
            under it ``exclusive_share`` reads 1.0.

    Returns:
        ``speaker_labels``, ``speaker_seconds``, ``attributed_s``, ``dominant_s``, ``secondary_s`` and
        ``extent_s``; ``share``, the largest contributor's seconds over every attributed second, or None where
        no segment intersects the extent; ``secondary_exclusive_s`` and ``secondary_exclusive_spans``, the
        time a speaker other than the largest holds with the largest silent; ``exclusive_share``, the largest
        contributor's seconds over those plus the exclusive time where that reaches ``exclusive_min_s``;
        ``exclusive_min_s``; and ``sources``, the segment ids the reading was taken off.
    """
    held: dict[str, float] = {}
    sources: list[str] = []
    for segment_id, label, (start, end) in segments:
        overlap = min(extent[1], end) - max(extent[0], start)
        if overlap <= 0.0:
            continue
        held[label] = held.get(label, 0.0) + overlap
        sources.append(segment_id)
    labels = sorted(held)
    seconds = [held[label] for label in labels]
    attributed = sum(seconds)
    dominant = max(seconds) if seconds else 0.0
    dominant_label = labels[seconds.index(dominant)] if seconds else None
    exclusive: list[Interval] = []
    if dominant_label is not None:
        mine = _clip(extent, [span for _, label, span in segments if label == dominant_label])
        theirs = _clip(extent, [span for _, label, span in segments if label != dominant_label])
        exclusive = subtract_intervals(theirs, mine)
    exclusive_s = sum(b - a for a, b in exclusive)
    counted = exclusive_s if exclusive_s >= exclusive_min_s else 0.0
    return {
        "speaker_labels": labels,
        "speaker_seconds": [round(value, 6) for value in seconds],
        "attributed_s": round(attributed, 6),
        "dominant_s": round(dominant, 6),
        "secondary_s": round(attributed - dominant, 6),
        "extent_s": round(extent[1] - extent[0], 6),
        "secondary_exclusive_s": round(exclusive_s, 6),
        "secondary_exclusive_spans": [[round(a, 3), round(b, 3)] for a, b in exclusive],
        "exclusive_min_s": float(exclusive_min_s),
        "share": (dominant / attributed) if attributed > 0.0 else None,
        "exclusive_share": (dominant / (dominant + counted)) if dominant + counted > 0.0 else None,
        "sources": sources,
    }
