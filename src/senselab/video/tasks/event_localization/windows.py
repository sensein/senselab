"""Overlapping time windows over a long recording, and the conversion of window-relative times to absolute times."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

Window = Tuple[float, float]
Segment = Dict[str, Any]


def plan_windows(
    duration_s: float, win_s: float = 30.0, stride_s: float = 20.0, min_tail_s: float = 5.0
) -> List[Window]:
    """Plan overlapping windows ``[t0, t1]`` that cover ``[0, duration_s]``.

    A recording no longer than ``win_s + min_tail_s`` is one window. Otherwise windows start every ``stride_s`` and the
    last one is pulled back so that it ends at ``duration_s`` and is ``win_s`` long, so a short remainder never
    becomes a tiny window. Every instant lies in at least one window.

    Args:
        duration_s: Length of the recording in seconds.
        win_s: Window length in seconds.
        stride_s: Distance between window starts in seconds; at most ``win_s``.
        min_tail_s: Slack beyond ``win_s`` that still counts as a single window.

    Returns:
        ``(start, end)`` pairs in seconds on the recording's own time axis; empty for a non-positive duration.

    Raises:
        ValueError: If ``win_s`` or ``stride_s`` is not positive or ``stride_s`` exceeds ``win_s``.
    """
    if duration_s <= 0:
        return []
    if win_s <= 0 or stride_s <= 0 or stride_s > win_s:
        raise ValueError("need 0 < stride_s <= win_s")
    if duration_s <= win_s + min_tail_s:
        return [(0.0, float(duration_s))]
    out: List[Window] = []
    t0 = 0.0
    while True:
        t1 = min(t0 + win_s, duration_s)
        if t1 >= duration_s - 1e-9:
            t0 = max(0.0, duration_s - win_s)
            if not out or out[-1][0] < t0 - 1e-9:
                out.append((t0, float(duration_s)))
            else:
                out[-1] = (out[-1][0], float(duration_s))
            break
        out.append((t0, t1))
        t0 += stride_s
    return out


def split_window(window: Window, parts: int = 2) -> List[Window]:
    """Cut a window into ``parts`` equal consecutive windows (for a window the model's input limit refuses)."""
    t0, t1 = window
    step = (t1 - t0) / parts
    return [(t0 + i * step, t0 + (i + 1) * step if i < parts - 1 else t1) for i in range(parts)]


def shift_segments(segments: Iterable[Segment], offset_s: float) -> List[Segment]:
    """Add ``offset_s`` to the start and end of every segment (window-relative to recording-absolute time)."""
    out = []
    for s in segments:
        t = dict(s)
        t["start_s"] = round(float(s["start_s"]) + offset_s, 3)
        t["end_s"] = round(float(s["end_s"]) + offset_s, 3)
        out.append(t)
    return out
