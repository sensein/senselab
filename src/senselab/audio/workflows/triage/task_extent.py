"""Where a task's extent begins and ends, and whether it holds the whole task.

Span-level pieces the branches share: bounded growth of an event cluster, the envelope-modulation event
detector, the stops growth halts at, and the completeness reading (``extent_complete``). Every parameter
and switch is in ``data/task_extent.yaml``; the design is ``specs/20261012-r37-extents/design.md``.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

TASK_EXTENT_PATH = Path(__file__).parent / "data" / "task_extent.yaml"

Span = tuple[float, float]

COMPLETE = "complete"
TOO_NARROW = "too_narrow"
TOO_WIDE = "too_wide"
BOTH = "both"
NO_TASK = "no_task"
STATUSES = (COMPLETE, TOO_NARROW, TOO_WIDE, BOTH, NO_TASK)

EDGE = "edge"
GAP = "gap"
STOP = "stop"

STIMULUS_EXTENT_CHECK = "stimulus_extent_check"
"""SPEECH's measurement of whether a read stimulus's extent holds every word read as the stimulus."""


@functools.cache
def task_extent_parameters() -> dict[str, Any]:
    """The parameters of ``data/task_extent.yaml``.

    Returns:
        The parsed mapping.
    """
    return dict(yaml.safe_load(TASK_EXTENT_PATH.read_text()) or {})


def enabled(section: str, p: Mapping[str, Any] | None = None) -> bool:
    """Whether a section of ``data/task_extent.yaml`` is switched on.

    Args:
        section: The section's key.
        p: The parameters; the packaged file when None.

    Returns:
        Its ``enabled`` value.
    """
    return bool(((p or task_extent_parameters()).get(section) or {}).get("enabled"))


def overlaps(span: Span, spans: Sequence[Span]) -> bool:
    """Whether a span overlaps any of ``spans``.

    Args:
        span: The span.
        spans: The others.

    Returns:
        True where one of them overlaps it by a positive length.
    """
    return any(a < span[1] and span[0] < b for a, b in spans)


def median_interval_s(core: Sequence[Span], *, floor_s: float, single_s: float) -> float:
    """The cluster's median interval from one event's start to the next one's.

    Args:
        core: The cluster's events, in time order.
        floor_s: The least the median is read as.
        single_s: The interval a one-event cluster is given.

    Returns:
        The median onset interval, at least ``floor_s``; ``single_s`` with fewer than two events.
    """
    if len(core) < 2:
        return float(single_s)
    gaps = [max(0.0, b[0] - a[0]) for a, b in zip(core, core[1:])]
    return max(float(floor_s), float(np.median(gaps)))


@dataclass(frozen=True)
class Growth:
    """How far bounded growth moved a cluster's edges.

    Attributes:
        added: The events taken in, in time order.
        before_s: How far the start moved back.
        after_s: How far the end moved on.
        limit_s: The largest gap growth crossed: ``k`` times the cluster's median onset interval.
        stopped: Why growth stopped before the start and after the end: ``edge`` (no event left), ``gap``
            (the next event lay beyond the limit) or ``stop`` (speech or another speaker lay in the way).
        crossed_n: The gaps growth crossed that held speech or another speaker (``growth.cross_speech``).
    """

    added: tuple[Span, ...] = ()
    before_s: float = 0.0
    after_s: float = 0.0
    limit_s: float = 0.0
    stopped: tuple[str, str] = (EDGE, EDGE)
    crossed_n: int = 0

    @property
    def seconds(self) -> float:
        """The growth on both sides."""
        return self.before_s + self.after_s

    def record(self) -> dict[str, Any]:
        """The growth, for the store."""
        return {
            "added_n": len(self.added),
            "before_s": round(self.before_s, 3),
            "after_s": round(self.after_s, 3),
            "seconds": round(self.seconds, 3),
            "limit_s": round(self.limit_s, 3),
            "stopped": list(self.stopped),
            "crossed_n": self.crossed_n,
        }


def grow(
    core: Sequence[Span],
    candidates: Sequence[Span],
    *,
    limit_s: float,
    stops: Sequence[Span],
    cross_stops: bool = False,
) -> Growth:
    """Grow a cluster outward one candidate event at a time.

    Args:
        core: The cluster's events, in time order.
        candidates: The events growth may take in; those inside the cluster's hull are ignored.
        limit_s: The largest gap from the current edge to the next event that growth crosses.
        stops: Speech and other-speaker spans; growth halts at the first one between the edge and the next
            event, or overlapping that event.
        cross_stops: Whether growth crosses a stop between the edge and the next event (breathing continuing
            across speech); an event overlapping a stop still halts it, and so does the gap.

    Returns:
        The growth; nothing added where ``core`` is empty.
    """
    if not core:
        return Growth(limit_s=limit_s)
    lo, hi = core[0][0], max(end for _, end in core)
    before = sorted((c for c in candidates if c[0] < lo and c[1] <= hi), key=lambda c: -c[1])
    after = sorted((c for c in candidates if c[1] > hi and c[0] >= lo), key=lambda c: c[0])
    added: list[Span] = []
    crossed = 0
    start, why_before = lo, EDGE
    for event in before:
        if start - event[1] > limit_s:
            why_before = GAP
            break
        if overlaps(event if cross_stops else (event[0], start), stops):
            why_before = STOP
            break
        crossed += int(overlaps((event[1], start), stops))
        added.append(event)
        start = min(start, event[0])
    end, why_after = hi, EDGE
    for event in after:
        if event[0] - end > limit_s:
            why_after = GAP
            break
        if overlaps(event if cross_stops else (end, event[1]), stops):
            why_after = STOP
            break
        crossed += int(overlaps((end, event[0]), stops))
        added.append(event)
        end = max(end, event[1])
    return Growth(tuple(sorted(added)), lo - start, end - hi, limit_s, (why_before, why_after), crossed)


def reachable_outside(extent: Span | None, outside: Sequence[Span], stops: Sequence[Span]) -> list[Span]:
    """The events outside an extent with no stop between them and it.

    Args:
        extent: The extent, or None.
        outside: Events outside it.
        stops: Speech and other-speaker spans.

    Returns:
        The events no stop separates from the extent; empty without an extent.
    """
    if extent is None:
        return []
    out = []
    for event in outside:
        if event[1] <= extent[0]:
            between = (event[0], extent[0])
        elif event[0] >= extent[1]:
            between = (extent[1], event[1])
        else:
            continue
        if not overlaps(between, stops):
            out.append(event)
    return out


@dataclass(frozen=True)
class ExtentCheck:
    """Whether the extent holds the whole task, and nothing beyond it.

    Attributes:
        status: ``complete``, ``too_narrow``, ``too_wide``, ``both`` or ``no_task``.
        outside_n: Task events (or task tokens) outside the extent that it could have reached.
        checks: The signals that fired, ``narrow:<name>`` or ``wide:<name>``; ``no_extent`` for ``no_task``.
        values: The readings behind each signal.
    """

    status: str
    outside_n: int = 0
    checks: tuple[str, ...] = ()
    values: dict[str, Any] = field(default_factory=dict)

    def record(self) -> dict[str, Any]:
        """The check, for the store."""
        return {"status": self.status, "outside_n": self.outside_n, "checks": list(self.checks), **self.values}


def extent_check(
    *,
    has_extent: bool,
    narrow: Mapping[str, Any],
    wide: Mapping[str, Any],
    outside_n: int = 0,
) -> ExtentCheck:
    """The completeness status from the signals a branch read.

    Args:
        has_extent: Whether the branch placed an extent.
        narrow: Each too-narrow signal's reading; a truthy reading fires.
        wide: Each too-wide signal's reading; a truthy reading fires.
        outside_n: The task events outside the extent that it could have reached.

    Returns:
        ``no_task`` without an extent; otherwise ``both``, ``too_narrow``, ``too_wide`` or ``complete`` by the
        signals that fired.
    """
    if not has_extent:
        return ExtentCheck(NO_TASK, 0, ("no_extent",))
    fired_narrow = [f"narrow:{name}" for name, value in narrow.items() if value]
    fired_wide = [f"wide:{name}" for name, value in wide.items() if value]
    status = (
        BOTH if fired_narrow and fired_wide else TOO_NARROW if fired_narrow else TOO_WIDE if fired_wide else COMPLETE
    )
    values = {name: value for name, value in {**narrow, **wide}.items() if value}
    return ExtentCheck(status, outside_n, tuple(fired_narrow + fired_wide), values)


def modulation_events(
    times_s: np.ndarray, over_db: np.ndarray, hop_s: float, p: Mapping[str, Any] | None = None
) -> list[Span]:
    """Events from the envelope's modulation in the breathing band: one per prominent positive lobe.

    Args:
        times_s: Each frame's time.
        over_db: Each frame's mean band level over the stationary floor, dB, clipped at zero.
        hop_s: The frames' hop.
        p: The ``modulation`` section of ``data/task_extent.yaml``; the packaged one when None.

    Returns:
        Per breathing-band lobe whose band-passed envelope stands ``prominence_db`` proud, the run of frames
        around its peak standing ``margin_db`` over the floor, kept where it lasts ``min_s`` to ``max_s``; in
        time order.
    """
    from scipy.signal import find_peaks  # noqa: PLC0415

    q = dict(p or task_extent_parameters()["modulation"])
    if len(over_db) < 4 or hop_s <= 0:
        return []
    step = max(1, int(round(1.0 / (float(q["rate_hz"]) * hop_s))))
    usable = len(over_db) // step * step
    if usable < 2 * step:
        return []
    series = over_db[:usable].reshape(-1, step).mean(axis=1)
    rate = 1.0 / (step * hop_s)
    spectrum = np.fft.rfft(series - series.mean())
    freqs = np.fft.rfftfreq(len(series), d=1.0 / rate)
    low, high = float(q["band_hz"][0]), float(q["band_hz"][1])
    spectrum[(freqs < low) | (freqs > high)] = 0.0
    band = np.fft.irfft(spectrum, n=len(series))
    peaks, _ = find_peaks(band, prominence=float(q["prominence_db"]))
    out: list[Span] = []
    margin = float(q["margin_db"])
    for peak in peaks:
        left = peak
        while left > 0 and band[left - 1] > 0:
            left -= 1
        right = peak
        while right < len(band) - 1 and band[right + 1] > 0:
            right += 1
        first, last = left * step, min(len(over_db), (right + 1) * step)
        frames = np.flatnonzero(over_db[first:last] >= margin)
        if not frames.size:
            continue
        centre = min(max(peak * step + step // 2, first), last - 1) - first
        runs = np.split(frames, np.flatnonzero(np.diff(frames) > 1) + 1)
        run = min(runs, key=lambda r: 0 if r[0] <= centre <= r[-1] else min(abs(r[0] - centre), abs(r[-1] - centre)))
        start, end = float(times_s[first + run[0]]), float(times_s[first + run[-1]] + hop_s)
        if float(q["min_s"]) <= end - start <= float(q["max_s"]):
            out.append((round(start, 3), round(end, 3)))
    return out


@dataclass(frozen=True)
class Trim:
    """What trimming a syllable train to its own extent removed.

    Attributes:
        kept: The indices of the events kept, in time order.
        speech_n: Events dropped for lying on, or beyond, a run of agreed non-task words.
        edge_n: Edge events dropped for lasting far longer than the train's own events.
    """

    kept: tuple[int, ...]
    speech_n: int = 0
    edge_n: int = 0

    def record(self) -> dict[str, Any]:
        """The trim, for the store."""
        return {"speech_n": self.speech_n, "edge_n": self.edge_n}


def trim_train(events: Sequence[Span], runs: Sequence[Span], *, ratio: float, kept_share_min: float = 0.0) -> Trim:
    """A syllable train trimmed to itself: split at speech runs, edge events of another length dropped.

    Args:
        events: The train's events, in time order.
        runs: Runs of agreed non-task words (:func:`agreed_speech_runs`).
        ratio: An edge event lasting more than this times the kept events' median duration is dropped.
        kept_share_min: The share of the events the largest piece must hold for the speech split to stand; under
            it the runs are read as the task misread and the train is not split.

    Returns:
        The events of the largest piece the runs leave (earliest on a tie), each overlapping no run, with its
        over-long edge events dropped one at a time; at least one event is kept where any was given.
    """
    if not events:
        return Trim(())
    pieces: list[list[int]] = [[]]
    for index, event in enumerate(events):
        if overlaps(event, runs):
            pieces.append([])
            continue
        if pieces[-1] and overlaps((events[pieces[-1][-1]][1], event[0]), runs):
            pieces.append([])
        pieces[-1].append(index)
    best = max((piece for piece in pieces if piece), key=len, default=[])
    if len(best) < kept_share_min * len(events):
        best = list(range(len(events)))
    if not best:
        return Trim((), speech_n=len(events))
    kept = list(best)
    edge = 0
    while len(kept) > 2:
        durations = [events[i][1] - events[i][0] for i in kept]
        median = float(np.median(durations))
        if durations[0] > ratio * median:
            kept.pop(0)
        elif durations[-1] > ratio * median:
            kept.pop()
        else:
            break
        edge += 1
    return Trim(tuple(kept), speech_n=len(events) - len(best), edge_n=edge)


def margined_extent(
    original: Span,
    kept: Sequence[Span],
    *,
    head: bool,
    tail: bool,
    intervals: float,
    runs: Sequence[Span],
) -> Span:
    """A trimmed train's extent, with a margin of its own onset interval kept beyond each trimmed edge.

    Args:
        original: The extent before trimming.
        kept: The events kept, in time order.
        head: Whether events were trimmed before the first kept one.
        tail: Whether events were trimmed after the last kept one.
        intervals: How many of the kept train's median onset intervals the margin is.
        runs: Speech runs the margin never enters.

    Returns:
        The kept events' hull, moved out by the margin on each trimmed edge, never past the original extent nor
        into a speech run; the hull alone where fewer than two events give no interval.
    """
    start, end = kept[0][0], max(b for _, b in kept)
    if len(kept) < 2:
        return start, end
    margin = intervals * float(np.median([b[0] - a[0] for a, b in zip(kept, kept[1:])]))
    if head:
        floor = max([original[0], *(b for a, b in runs if b <= start)])
        start = max(floor, start - margin)
    if tail:
        ceiling = min([original[1], *(a for a, b in runs if a >= end)])
        end = min(ceiling, end + margin)
    return start, end


def word_span(word: Any) -> Span:  # noqa: ANN401 -- a consensus word entity
    """Where a consensus word stands: its consensus extent, else the hull of its placements.

    Args:
        word: A consensus ``word`` entity.

    Returns:
        ``(start, end)``.
    """
    from senselab.audio.workflows.triage.nodes.common import word_hull  # noqa: PLC0415

    if word.extent is not None and float(word.extent[1]) > float(word.extent[0]):
        return float(word.extent[0]), float(word.extent[1])
    return word_hull(word)


def agreed_speech_runs(words: Sequence[Any], family: str | None, run_min: int) -> list[Span]:
    """Runs of agreed dictionary words that are not the task's own syllables, as spans.

    Args:
        words: The consensus words, in stream order.
        family: The declared family, whose task syllables (``cohort_stage.yaml`` ``non_task.task_syllables``) are
            never speech.
        run_min: The fewest adjacent words that make a run.

    Returns:
        Each run's span, from its first word's start to its last word's end (:func:`word_span`), in time order.
    """
    from senselab.audio.workflows.triage.cohort_stage import spells_task_syllables  # noqa: PLC0415
    from senselab.audio.workflows.triage.task_content import agreed_runs  # noqa: PLC0415

    syllables = {w.id for w in words if family and spells_task_syllables(str(w.attributes.get("text") or ""), family)}
    spoken = agreed_runs(words, syllables, run_min)
    runs: list[list[Span]] = [[]]
    for word in words:
        if word.id in spoken:
            runs[-1].append(word_span(word))
        elif runs[-1]:
            runs.append([])
    return sorted((min(a for a, _ in run), max(b for _, b in run)) for run in runs if run)


def diarized_turns(store: Any, run_dir: Path, min_s: float, stream: str = "enhanced") -> list[Span]:  # noqa: ANN401
    """The diarizer's turns at least ``min_s`` long, of any speaker.

    Args:
        store: The provenance store.
        run_dir: The run directory the sidecar is relative to.
        min_s: The shortest turn kept.
        stream: The diarized stream.

    Returns:
        The turns, in time order; empty where no diarization was stored.
    """
    from senselab.audio.workflows.triage.nodes.branches import derivative_arrays  # noqa: PLC0415

    arrays = derivative_arrays(store, run_dir, f"{stream}_diarization")
    if arrays is None or "starts" not in arrays:
        return []
    return sorted(
        (float(a), float(b)) for a, b in zip(arrays["starts"], arrays["ends"]) if float(b) - float(a) >= min_s
    )
