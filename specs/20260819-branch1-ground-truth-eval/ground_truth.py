"""The verified labels for the project's only human-verified recording, and the scoring regions.

Human-verified 2026-08-18. Two events carry verified spans (onset and offset); the rest carry
verified onsets with an approximate duration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

WAV = "/Users/satra/Downloads/streaming-audio-2026-07-30T04-21-56-487Z.wav"
DURATION = 14.026666666666666


@dataclass(frozen=True)
class Event:
    """One verified event."""

    index: int
    element: str
    onset: float
    offset: float
    span_verified: bool
    yamnet_labels: Tuple[str, ...]
    hear_labels: Tuple[str, ...]


EVENTS: List[Event] = [
    Event(
        1,
        "mouth non-speech sound",
        0.893,
        0.893 + 0.202,
        False,
        ("__mouth__",),
        ("Throat Clear", "Cough"),
    ),
    Event(2, "exhalation (breath)", 2.275, 2.275 + 1.221, False, ("Breathing",), ("Breathe",)),
    Event(3, "exhalation (breath)", 5.308, 5.308 + 0.983, False, ("Breathing",), ("Breathe",)),
    Event(4, "cough", 7.926, 8.494, True, ("Cough",), ("Cough",)),
    Event(5, "cough", 9.610, 10.250, True, ("Cough",), ("Cough",)),
    Event(6, "speech", 11.62, 13.20, True, ("Speech",), ("Speech",)),
]

EMPTY_AS_GIVEN: List[Tuple[float, float]] = [
    (0.0, 0.78),
    (1.0, 2.3),
    (3.5, 5.3),
    (6.3, 7.9),
    (8.5, 9.6),
    (10.25, 11.65),
    (13.2, 14.03),
]

UNLABELLED: Tuple[float, float] = (13.79, 14.04)
"""Excluded from scoring in both directions: Brouhaha's VAD rises here, community-1 stays at
zero, and no human verdict exists."""


def _subtract(regions: List[Tuple[float, float]], cuts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    out = list(regions)
    for cut_a, cut_b in cuts:
        nxt: List[Tuple[float, float]] = []
        for a, b in out:
            if cut_b <= a or cut_a >= b:
                nxt.append((a, b))
                continue
            if a < cut_a:
                nxt.append((a, cut_a))
            if cut_b < b:
                nxt.append((cut_b, b))
        out = nxt
    return [(a, b) for a, b in out if b - a > 1e-9]


def scorable_empty() -> List[Tuple[float, float]]:
    """The verified-empty stretches, minus every event extent and minus the unlabelled tail.

    The as-given empty stretches were written against verified onsets; three of them overlap an
    event extent by 5-95 ms because the extents come from approximate durations. Subtracting the
    extents keeps a detection that lands on a real event out of the false-positive count.
    """
    cuts = [(e.onset, e.offset) for e in EVENTS] + [UNLABELLED]
    return _subtract(EMPTY_AS_GIVEN, cuts)


def scorable_empty_seconds() -> float:
    """Total scorable-empty duration in seconds."""
    return sum(b - a for a, b in scorable_empty())


def overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Overlap of two intervals in seconds (0 if disjoint)."""
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def inside_any(window: Tuple[float, float], regions: List[Tuple[float, float]]) -> bool:
    """True when the window lies wholly inside one of the regions."""
    return any(a <= window[0] and window[1] <= b for a, b in regions)


if __name__ == "__main__":
    regions = scorable_empty()
    print("scorable-empty regions:")
    for a, b in regions:
        print(f"  {a:6.3f} - {b:6.3f}   ({b - a:.3f} s)")
    total = scorable_empty_seconds()
    print(f"total {total:.3f} s = {total / 60.0:.4f} min")
    print(f"one false positive is worth {60.0 / total:.2f} FP/min")
