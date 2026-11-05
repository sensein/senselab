"""Localize labelled events in a long recording by asking a model about overlapping windows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence

from senselab.video.tasks.event_localization.merge import merge_windows
from senselab.video.tasks.event_localization.parsing import COUNT_KEYS, parse_reply
from senselab.video.tasks.event_localization.windows import Segment, Window, plan_windows, shift_segments, split_window


class WindowTooLarge(Exception):
    """Raised by an ``ask`` callable when the model refuses a window for its size (context, encoder cache)."""


@dataclass
class WindowReply:
    """One answered window: its absolute extent, the raw reply, the validated absolute segments and the parse counts."""

    window: Window
    reply: str
    segments: List[Segment]
    counts: Dict[str, int]


@dataclass
class LocalizationResult:
    """Merged segments of a recording and what happened per window."""

    segments: List[Segment]
    windows: List[WindowReply] = field(default_factory=list)
    n_split: int = 0
    n_unfit: int = 0

    def counts(self) -> Dict[str, int]:
        """Parse counts summed over the windows."""
        return {k: sum(w.counts.get(k, 0) for w in self.windows) for k in COUNT_KEYS}


def localize_events(
    duration_s: float,
    ask: Callable[[Window], str],
    labels: Sequence[str],
    win_s: float = 30.0,
    stride_s: float = 20.0,
    gap_s: float = 1.0,
    point_labels: Sequence[str] = (),
    min_win_s: float = 4.0,
    strict: bool = False,
    **merge_kwargs: Any,  # noqa: ANN401
) -> LocalizationResult:
    """Ask a model about overlapping windows of a recording and merge the answers into one segment list.

    Args:
        duration_s: Length of the recording in seconds.
        ask: Called with an absolute ``(start, end)`` window; returns the model's reply for that stretch. The reply's
            times are relative to the window start. It raises `WindowTooLarge` for a window the model refuses by size.
        labels: The closed label set the reply must use.
        win_s: Window length in seconds.
        stride_s: Distance between window starts in seconds.
        gap_s: Gap between same-label segments that `merge_windows` bridges.
        point_labels: Labels that are point events (zero-length segments are kept).
        min_win_s: A refused window is halved until its halves would be shorter than twice this.
        strict: Pass to `parse_reply`.
        **merge_kwargs: Further `merge_windows` arguments (``min_len_s``).

    Returns:
        The merged absolute segments with the per-window replies, parse counts and the number of split / unfit windows.
    """
    result = LocalizationResult(segments=[])

    def one(w: Window) -> None:
        try:
            text = ask(w)
        except WindowTooLarge:
            if w[1] - w[0] > 2 * min_win_s:
                result.n_split += 1
                for h in split_window(w, 2):
                    one(h)
            else:
                result.n_unfit += 1
            return
        segs, counts = parse_reply(text, labels, w[1] - w[0], point_labels=point_labels, strict=strict)
        result.windows.append(WindowReply((w[0], w[1]), text, shift_segments(segs, w[0]), counts))

    for w in plan_windows(duration_s, win_s, stride_s):
        one(w)
    result.segments = merge_windows([(r.window, r.segments) for r in result.windows], gap_s=gap_s, **merge_kwargs)
    return result
