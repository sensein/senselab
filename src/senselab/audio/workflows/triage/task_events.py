"""The task layer's shared reading: task events over the generic background, their evidence and extent.

A branch finds candidate task events with its own type test; this module reads every candidate
against the recording-level background (:mod:`~senselab.audio.workflows.triage.background_model`):
its SNR over the stationary floor in its own window with impulse frames masked, whether an impulse
overlaps or abuts it, the rhythm of the recording's activity, the dominant cluster and the extent
with its preparatory inhale, and the present / review / absent decision on that evidence. Every
parameter is in ``data/task_events.yaml``; the design is
``specs/20261007-task-events-in-background/design.md``.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from senselab.audio.workflows.triage.background_model import (
    PLAIN_STREAM,
    RESIDUAL_STREAM,
    BandFrames,
    Floor,
    Impulse,
    Region,
    _named_stream,
    background_model_parameters,
    band_frames,
    floor_of,
    impulses_of,
    regions_of,
)
from senselab.utils.prov_store import ProvStore

TASK_EVENTS_PATH = Path(__file__).parent / "data" / "task_events.yaml"

PRESENT = "present"
REVIEW = "review"
ABSENT = "absent"
DECISIONS = (PRESENT, REVIEW, ABSENT)

Span = tuple[float, float]


@functools.cache
def task_events_parameters() -> dict[str, Any]:
    """The parameters of ``data/task_events.yaml``.

    Returns:
        The parsed mapping.
    """
    return dict(yaml.safe_load(TASK_EVENTS_PATH.read_text()) or {})


@dataclass(frozen=True)
class GenericView:
    """What a branch reads of the recording-level background.

    Attributes:
        frames: The plain stream's band frames.
        floor: The stationary floor per band.
        impulses: The impulses.
        regions: The regions of activity.
        broadband_floor_db: The floor summed over the bands, dB.
        hop_s: The frames' hop.
    """

    frames: BandFrames
    floor: Floor
    impulses: tuple[Impulse, ...]
    regions: tuple[Region, ...]
    broadband_floor_db: float
    hop_s: float

    @property
    def duration_s(self) -> float:
        """The frames' span."""
        return float(len(self.frames.times_s) * self.hop_s)


def generic_view(plain: tuple[np.ndarray, int], residual: tuple[np.ndarray, int] | None) -> GenericView:
    """The recording-level background of one recording, read as QUALITY reads it.

    Args:
        plain: The plain stream.
        residual: The residual stream, or None.

    Returns:
        The view.
    """
    p = background_model_parameters()
    frames = band_frames(plain, p)
    floor = floor_of(frames, band_frames(residual, p) if residual is not None else None, p)
    impulses = tuple(impulses_of(plain, p))
    regions = tuple(regions_of(frames, floor.band_db, impulses, p))
    broadband = float(10.0 * np.log10(np.sum(10.0 ** (floor.band_db / 10.0)) + 1e-12))
    return GenericView(frames, floor, impulses, regions, broadband, float(p["hop_s"]))


def generic_view_of(store: ProvStore, run_dir: Path) -> GenericView | None:
    """The recording-level background from the store's streams.

    Args:
        store: The provenance store, read for the ``plain`` and ``residual`` streams.
        run_dir: The run directory their paths are relative to.

    Returns:
        The view, or None where the plain stream is absent.
    """
    plain = _named_stream(store, run_dir, PLAIN_STREAM)
    if plain is None:
        return None
    return generic_view(plain, _named_stream(store, run_dir, RESIDUAL_STREAM))


def _impulse_mask(view: GenericView) -> np.ndarray:
    mask = np.zeros(len(view.frames.times_s), dtype=bool)
    for imp in view.impulses:
        mask[(view.frames.times_s >= imp.start_s - view.hop_s) & (view.frames.times_s <= imp.end_s + view.hop_s)] = True
    return mask


def event_snr_db(view: GenericView, start_s: float, end_s: float) -> float:
    """An event's level over the broadband floor in its own window, impulse frames masked.

    Args:
        view: The background.
        start_s: The event's start.
        end_s: Its end.

    Returns:
        The highest broadband level over the floor among the window's frames no impulse holds; the
        floor's own level (0 dB) where every frame is masked or the window holds none.
    """
    times = view.frames.times_s
    inside = (times >= start_s) & (times <= end_s) & ~_impulse_mask(view)
    if not inside.any():
        return 0.0
    return float(view.frames.level_db[inside].max() - view.broadband_floor_db)


def local_snr_db(view: GenericView, span: Span, others: Sequence[Span], *, window_s: float, percentile: float) -> float:
    """An event's level over the background around it: the window either side, outside every event.

    Args:
        view: The background.
        span: The event.
        others: Every candidate event, whose frames are not background.
        window_s: How far either side the background is read.
        percentile: The percentile of the background frames' broadband level taken as the background.

    Returns:
        The event's highest unmasked broadband level over the larger of the local background and the
        stationary floor; the floor-relative level where the window holds too few background frames.
    """
    times = view.frames.times_s
    masked = _impulse_mask(view)
    start, end = span
    inside = (times >= start) & (times <= end) & ~masked
    if not inside.any():
        return 0.0
    taken = np.zeros(len(times), dtype=bool)
    for a, b in others:
        taken |= (times >= a) & (times <= b)
    around = (((times >= start - window_s) & (times < start)) | ((times > end) & (times <= end + window_s))) & ~(
        masked | taken
    )
    background = view.broadband_floor_db
    if around.sum() >= 5:
        background = max(background, float(np.percentile(view.frames.level_db[around], percentile)))
    return float(view.frames.level_db[inside].max() - background)


def entangled(view: GenericView, span: Span, abut_s: float) -> bool:
    """Whether an impulse overlaps or abuts an event.

    Args:
        view: The background.
        span: The event.
        abut_s: How close an impulse may end before or start after the event and still abut it.

    Returns:
        True where an impulse lies within ``abut_s`` of the event.
    """
    start, end = span
    return any(imp.end_s >= start - abut_s and imp.start_s <= end + abut_s for imp in view.impulses)


def impulse_explained(view: GenericView, span: Span, margin_db: float) -> bool:
    """Whether an event is only an impulse: masked of impulse frames, nothing stands over the floor.

    Args:
        view: The background.
        span: The event.
        margin_db: The level over the floor the rest of the event must reach.

    Returns:
        True where an impulse lies in the event and its other frames stay under ``margin_db``.
    """
    start, end = span
    if not any(imp.end_s >= start and imp.start_s <= end for imp in view.impulses):
        return False
    return event_snr_db(view, start, end) < margin_db


@dataclass(frozen=True)
class Rhythm:
    """The dominant modulation rate of the recording's band envelopes in the breathing band.

    Attributes:
        hz: The peak's frequency.
        prominence_db: The peak over the spectrum's median in the read range.
    """

    hz: float
    prominence_db: float

    @property
    def period_s(self) -> float:
        """One cycle."""
        return 1.0 / self.hz

    def record(self) -> dict[str, float]:
        """The rhythm, for the store."""
        return {"hz": round(self.hz, 3), "prominence_db": round(self.prominence_db, 2)}


def rhythm_of(view: GenericView, band_hz: Sequence[float], prominence_min_db: float) -> Rhythm | None:
    """The breathing-band modulation peak of the mean band envelope, where one stands.

    Args:
        view: The background.
        band_hz: The breathing band, ``(low, high)``.
        prominence_min_db: The prominence a peak needs to stand.

    Returns:
        The rhythm, or None where the recording is shorter than two cycles of the band's low edge or
        no peak stands.
    """
    from scipy.signal import welch  # noqa: PLC0415

    low, high = float(band_hz[0]), float(band_hz[1])
    fs = 1.0 / view.hop_s
    over = view.frames.band_db - view.floor.band_db[None, :]
    envelope = np.clip(over, 0.0, None).mean(axis=1)
    if len(envelope) * view.hop_s < 2.0 / low:
        return None
    step = max(1, int(round(fs / 10.0)))
    series = envelope[: len(envelope) // step * step].reshape(-1, step).mean(axis=1)
    series = series - series.mean()
    if not np.any(series):
        return None
    freqs, psd = welch(series, fs=fs / step, nperseg=len(series), detrend="linear")
    db = 10.0 * np.log10(psd + 1e-20)
    read = (freqs >= low / 2.0) & (freqs <= 4.0)
    band = (freqs >= low) & (freqs <= high)
    if not band.any() or not read.any():
        return None
    peak = int(np.flatnonzero(band)[int(np.argmax(db[band]))])
    prominence = float(db[peak] - np.median(db[read]))
    return Rhythm(float(freqs[peak]), prominence) if prominence >= prominence_min_db else None


@dataclass(frozen=True)
class TaskEvent:
    """One task event the branch's type test kept, with its evidence.

    Attributes:
        start_s: Its start.
        end_s: Its end.
        snr_db: Its level over the stationary floor (:func:`event_snr_db`).
        local_snr_db: Its level over the background around it (:func:`local_snr_db`); the floor-relative
            level where none was read.
        entangled: Whether an impulse overlaps or abuts it.
        recovered: Whether the rhythm prior recovered it from a region the branch's finder missed.
        stream: The stream it was found on.
    """

    start_s: float
    end_s: float
    snr_db: float
    local_snr_db: float | None = None
    entangled: bool = False
    recovered: bool = False
    stream: str = "plain"

    @property
    def clear_db(self) -> float:
        """The level the decision's clear cut reads: the local SNR where read, else the floor SNR."""
        return self.snr_db if self.local_snr_db is None else self.local_snr_db

    def record(self) -> dict[str, Any]:
        """The event, for the store."""
        return {
            "start_s": round(self.start_s, 3),
            "end_s": round(self.end_s, 3),
            "snr_db": round(self.snr_db, 2),
            "local_snr_db": None if self.local_snr_db is None else round(self.local_snr_db, 2),
            "entangled": self.entangled,
            "recovered": self.recovered,
            "stream": self.stream,
        }


def task_event(
    view: GenericView, span: Span, others: Sequence[Span], p: dict[str, Any], *, recovered: bool = False
) -> TaskEvent:
    """One candidate read against the background.

    Args:
        view: The background.
        span: The candidate.
        others: Every candidate, whose frames the local background excludes.
        p: The parameters of ``data/task_events.yaml``.
        recovered: Whether the rhythm prior recovered it.

    Returns:
        The event.
    """
    start, end = span
    local = local_snr_db(
        view, span, others, window_s=float(p["local"]["window_s"]), percentile=float(p["local"]["percentile"])
    )
    return TaskEvent(
        float(start),
        float(end),
        event_snr_db(view, start, end),
        local,
        entangled(view, span, float(p["entangle_abut_s"])),
        recovered,
    )


def dominant_cluster(events: Sequence[TaskEvent], gap_s: float) -> list[TaskEvent]:
    """The events split at gaps over ``gap_s``; the cluster with most events, earliest on a tie.

    Args:
        events: The events, in time order.
        gap_s: The longest gap inside a cluster.

    Returns:
        The cluster; empty with no event.
    """
    clusters: list[list[TaskEvent]] = []
    for event in sorted(events, key=lambda e: e.start_s):
        if clusters and event.start_s - clusters[-1][-1].end_s <= gap_s:
            clusters[-1].append(event)
        else:
            clusters.append([event])
    return max(clusters, key=lambda c: (len(c), -c[0].start_s)) if clusters else []


def inhale_start(view: GenericView, start_s: float, margin_db: float, back_max_s: float) -> float:
    """Where the level rising into the first event leaves the floor, at most ``back_max_s`` before it.

    Args:
        view: The background.
        start_s: The first event's start.
        margin_db: The broadband level over the floor that counts as risen.
        back_max_s: The furthest the start is moved back.

    Returns:
        The start, moved back across contiguous frames standing ``margin_db`` over the floor.
    """
    times = view.frames.times_s
    level = view.frames.level_db - view.broadband_floor_db
    index = int(np.searchsorted(times, start_s))
    first = index
    while first > 0 and level[first - 1] >= margin_db and start_s - times[first - 1] <= back_max_s:
        first -= 1
    return float(max(0.0, times[first] - view.hop_s / 2)) if first < index else float(start_s)


@dataclass(frozen=True)
class TaskEvidence:
    """The evidence a branch's events give, and the decision it supports.

    Attributes:
        events: The dominant cluster's events, in time order.
        found: Every event the type test kept, in the cluster or not.
        rhythm: The breathing-band rhythm, or None.
        extent: The cluster's extent with its preparatory inhale, or None with no event.
        decision: ``present``, ``review`` or ``absent`` (:func:`decide`).
        why: The condition that decided it.
    """

    events: tuple[TaskEvent, ...]
    found: tuple[TaskEvent, ...]
    rhythm: Rhythm | None
    extent: Span | None
    decision: str
    why: str

    @property
    def events_found_n(self) -> int:
        """Every event the type test kept, in the cluster or not."""
        return len(self.found)

    @property
    def best_snr_db(self) -> float | None:
        """The strongest event's SNR."""
        return max((e.snr_db for e in self.events), default=None)

    @property
    def median_snr_db(self) -> float | None:
        """The events' median SNR."""
        return float(np.median([e.snr_db for e in self.events])) if self.events else None

    def record(self) -> dict[str, Any]:
        """The evidence, for the store."""
        best, median = self.best_snr_db, self.median_snr_db
        return {
            "decision": self.decision,
            "why": self.why,
            "events_n": len(self.events),
            "events_found_n": self.events_found_n,
            "recovered_n": sum(e.recovered for e in self.events),
            "best_snr_db": None if best is None else round(best, 2),
            "median_snr_db": None if median is None else round(median, 2),
            "rhythm": None if self.rhythm is None else self.rhythm.record(),
            "extent": None if self.extent is None else [round(self.extent[0], 3), round(self.extent[1], 3)],
            "events": [e.record() for e in self.events],
        }


def decide(events: Sequence[TaskEvent], p: dict[str, Any]) -> tuple[str, str]:
    """The decision on the events the type test kept (C5).

    Args:
        events: Every event the type test kept; the cluster bounds the extent, not the decision.
        p: The ``decision`` section of ``data/task_events.yaml``.

    Returns:
        ``(decision, why)``: ``absent`` where no found event stands ``snr_low_db`` over the floor;
        ``review`` where none of those stands ``snr_high_db`` over its local background (``weak``), or
        an impulse overlaps or abuts every one that does (``entangled``); ``present`` otherwise.
        Events the rhythm prior recovered count phases and extent, never the decision.
    """
    standing = [e for e in events if not e.recovered and e.snr_db >= p["snr_low_db"]]
    if not standing:
        return ABSENT, "no event over the floor"
    clear = [e for e in standing if e.clear_db >= p["snr_high_db"]]
    if not clear:
        return REVIEW, "weak"
    if all(e.entangled for e in clear):
        return REVIEW, "entangled"
    return PRESENT, "clear"


def evidence_of(
    view: GenericView,
    candidates: Sequence[Span],
    *,
    gap_s: float,
    rhythm: Rhythm | None = None,
    recovered: Sequence[Span] = (),
    inhale: bool = True,
    p: dict[str, Any] | None = None,
) -> TaskEvidence:
    """Read a branch's candidates against the background: events, cluster, extent and decision.

    Args:
        view: The background.
        candidates: The spans the branch's type test kept.
        gap_s: The longest gap inside the dominant cluster.
        rhythm: The rhythm, where one stands, recorded beside the evidence.
        recovered: Further spans the rhythm prior recovered.
        inhale: Whether the extent runs back across a preparatory inhale.
        p: The parameters; ``data/task_events.yaml`` when None.

    Returns:
        The evidence.
    """
    p = p or task_events_parameters()
    margin = float(p["impulse_margin_db"])
    kept = [s for s in candidates if not impulse_explained(view, s, margin)]
    extra = [s for s in recovered if not impulse_explained(view, s, margin)]
    others = [*kept, *extra]
    events = [task_event(view, s, others, p) for s in kept]
    events += [task_event(view, s, others, p, recovered=True) for s in extra]
    cluster = dominant_cluster([e for e in events if e.snr_db >= p["decision"]["snr_low_db"]], gap_s) or (
        dominant_cluster(events, gap_s)
    )
    decision, why = decide(events, p["decision"])
    extent: Span | None = None
    if cluster:
        start = cluster[0].start_s
        if inhale:
            start = inhale_start(view, start, float(p["inhale"]["margin_db"]), float(p["inhale"]["back_max_s"]))
        extent = (start, max(e.end_s for e in cluster))
    return TaskEvidence(tuple(cluster), tuple(events), rhythm, extent, decision, why)
