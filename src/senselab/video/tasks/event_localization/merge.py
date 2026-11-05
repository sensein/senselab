"""Merge the segment lists of overlapping windows into one list, and rasterize segments to frame labels."""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from senselab.video.tasks.event_localization.windows import Segment, Window

_Member = Tuple[str, float, float, int, Optional[Mapping[str, str]]]


def _cover(windows: Sequence[Window], t: float) -> int:
    return sum(1 for a, b in windows if a - 1e-9 <= t <= b + 1e-9)


def merge_windows(
    per_window: Sequence[Tuple[Window, Sequence[Segment]]], gap_s: float = 1.0, min_len_s: float = 0.0
) -> List[Segment]:
    """Merge per-window segments (absolute times) into one segment list.

    Per label, the segments of all windows are sorted by start and chained into one segment when they overlap or are at
    most ``gap_s`` apart (the union of the members). ``score`` is the number of distinct windows that contributed a
    member divided by the number of windows that cover the merged segment's midpoint, in ``(0, 1]`` (1 for a one-window
    recording), so a segment seen in both overlapping windows outranks one seen in a single window. ``n_windows`` and
    ``n_members`` are kept. Values of the optional ``extra`` mapping (for example an attribute the model reported) are
    carried as the most frequent value among the members.

    Args:
        per_window: ``((t0, t1), segments)`` per window, segments with absolute ``start_s``/``end_s`` and a ``label``.
        gap_s: Largest gap between two segments of the same label that is bridged.
        min_len_s: Merged segments shorter than this are dropped; 0 keeps point events.

    Returns:
        Segments with ``label``, ``start_s``, ``end_s``, ``score``, ``n_windows``, ``n_members`` (and ``extra``),
        sorted by start, end and label.
    """
    wins = [w for w, _ in per_window]
    items: List[_Member] = []
    for wi, (_, segs) in enumerate(per_window):
        for s in segs:
            items.append((s["label"], float(s["start_s"]), float(s["end_s"]), wi, s.get("extra")))
    out: List[Segment] = []
    for label in sorted({i[0] for i in items}):
        mine = sorted((i for i in items if i[0] == label), key=lambda x: (x[1], x[2]))
        cur: List[_Member] = []
        cur_end = -1e18
        groups: List[List[_Member]] = []
        for it in mine:
            if cur and it[1] <= cur_end + gap_s:
                cur.append(it)
                cur_end = max(cur_end, it[2])
            else:
                if cur:
                    groups.append(cur)
                cur, cur_end = [it], it[2]
        if cur:
            groups.append(cur)
        for g in groups:
            s0, e0 = min(x[1] for x in g), max(x[2] for x in g)
            if e0 - s0 < min_len_s:
                continue
            nwin = len({x[3] for x in g})
            cov = max(_cover(wins, (s0 + e0) / 2.0), nwin, 1)
            seg: Segment = {
                "label": label,
                "start_s": round(s0, 3),
                "end_s": round(e0, 3),
                "score": round(nwin / cov, 4),
                "n_windows": nwin,
                "n_members": len(g),
            }
            votes: Dict[str, Dict[str, int]] = {}
            for x in g:
                for k, v in (x[4] or {}).items():
                    votes.setdefault(k, {}).setdefault(v, 0)
                    votes[k][v] += 1
            if votes:
                seg["extra"] = {k: max(sorted(d), key=lambda v: d[v]) for k, d in votes.items()}
            out.append(seg)
    out.sort(key=lambda s: (s["start_s"], s["end_s"], s["label"]))
    return out


def rasterize(segments: Sequence[Segment], n_frames: int, fps: float, background: str = "background") -> List[str]:
    """Turn segments into one label per frame.

    Frame ``i`` covers ``[i / fps, (i + 1) / fps)`` and takes the label of the segment that contains its centre. On an
    overlap the segment with the higher ``score`` wins (ties: the shorter one, then the earlier one).

    Args:
        segments: Segments with ``label``, ``start_s``, ``end_s`` and an optional ``score`` (default 1).
        n_frames: Number of frames.
        fps: Frame rate.
        background: Label of a frame in no segment.

    Returns:
        ``n_frames`` labels.
    """
    labels = [background] * n_frames
    best: List[Optional[Tuple[float, float, float]]] = [None] * n_frames
    for s in segments:
        a, b = float(s["start_s"]), float(s["end_s"])
        i0 = max(0, int(math.ceil(a * fps - 0.5)))
        i1 = min(n_frames, int(math.ceil(b * fps - 0.5)))
        key = (float(s.get("score", 1.0)), -(b - a), -a)
        for i in range(i0, i1):
            cur = best[i]
            if cur is None or key > cur:
                best[i] = key
                labels[i] = s["label"]
    return labels
