"""The diadochokinesis task layer: syllable and cycle events over the background, their identity, the decision.

Syllable nuclei are read off BACKGROUND's activity regions, split at troughs of the level over the
floor and of voicing. A single-syllable family's events are its syllables; a sequence family's are
cycles of the target word, grouped from the syllables with the template's syllable count and the
rhythm prior. Each event is read against the background (:func:`~senselab.audio.workflows.triage.
task_events.evidence_of`) and scored against the declared template on the phonetic posteriorgram by a
local alignment that allows skips, deletions and substitutions. The decision is the shared one
(present / review / absent) on clear events and identity; counts, rates and regularity are
annotations. Every parameter is in ``data/task_events.yaml`` (``ddk``); the design is
``specs/20261007-task-events-in-background/design.md`` ("DDK task layer").
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import numpy as np

from senselab.audio.workflows.triage.task_events import (
    ABSENT,
    PRESENT,
    REVIEW,
    GenericView,
    Rhythm,
    TaskEvent,
    TaskEvidence,
    dominant_cluster,
    evidence_of,
    rhythm_of,
    task_events_parameters,
)

DDK_READING = "ddk_task_reading"
"""The measurement SPEECH writes for a declared syllable-repetition task: its events and decision."""

SYLLABLE = "syllable"
CYCLE = "cycle"

Span = tuple[float, float]


def ddk_parameters() -> dict[str, Any]:
    """The ``ddk`` section of ``data/task_events.yaml``.

    Returns:
        The mapping.
    """
    return dict(task_events_parameters()["ddk"])


@dataclass(frozen=True)
class Voicing:
    """Which frames of the pitch track are voiced.

    Attributes:
        times_s: Each track frame's time.
        voiced: Whether it is voiced.
    """

    times_s: np.ndarray
    voiced: np.ndarray

    def at(self, times_s: np.ndarray) -> np.ndarray:
        """Whether the nearest track frame to each time is voiced.

        Args:
            times_s: The times.

        Returns:
            One boolean per time; all False where the track is empty.
        """
        if not len(self.times_s):
            return np.zeros(len(times_s), dtype=bool)
        index = np.clip(np.searchsorted(self.times_s, times_s), 0, len(self.times_s) - 1)
        return np.asarray(self.voiced[index], dtype=bool)


@dataclass(frozen=True)
class PositionMasses:
    """The posterior mass each template position's class holds per posteriorgram frame.

    Attributes:
        template: The declared phoneme sequence.
        classes: Each position's class.
        masses: ``(frame, position)``, the summed mass of each position's class.
        filler: ``(frame,)``, the mass the alignment's fillers explain: silence.
        seconds_per_frame: The frame period.
        floor: The smallest mass told from zero, so an empty class stays finite in log space.
    """

    template: tuple[str, ...]
    classes: tuple[str, ...]
    masses: np.ndarray
    filler: np.ndarray
    seconds_per_frame: float
    floor: float = 1e-6


@dataclass(frozen=True)
class Identity:
    """One event scored against the template.

    Attributes:
        score: The mean over positions of each position's peak class mass, a deleted position counting 0.
        realised: Per position, the mean class mass over the frames charged to it; None where deleted.
        peaks: Per position, the highest class mass over the frames charged to it; None where deleted.
    """

    score: float
    realised: tuple[float | None, ...]
    peaks: tuple[float | None, ...] = ()

    def record(self) -> dict[str, Any]:
        """The identity, for the store."""
        return {
            "score": round(self.score, 3),
            "realised": [None if m is None else round(m, 3) for m in self.realised],
            "peaks": [None if m is None else round(m, 3) for m in self.peaks],
        }


def nucleus_score(view: GenericView, voicing: Voicing | None, q: dict[str, Any]) -> np.ndarray:
    """The per-frame series syllable nuclei are peaks of: the level over the floor, lowered where unvoiced.

    Args:
        view: The background.
        voicing: The pitch track's voicing, or None where it is not stored.
        q: The ``ddk`` parameters.

    Returns:
        The broadband level over the floor, smoothed over ``smooth_s``, less ``unvoiced_db`` on
        unvoiced frames.
    """
    level = view.frames.level_db - view.broadband_floor_db
    if voicing is not None:
        level = level - float(q["unvoiced_db"]) * ~voicing.at(view.frames.times_s)
    width = max(1, int(round(float(q["smooth_s"]) / view.hop_s)))
    if width > 1:
        level = np.convolve(level, np.ones(width) / width, mode="same")
    return np.asarray(level, dtype=float)


def syllable_runs(view: GenericView, voicing: Voicing | None, q: dict[str, Any]) -> list[list[Span]]:
    """The syllables of each activity region, split at the troughs between nuclei.

    Args:
        view: The background; its regions are where syllables are looked for.
        voicing: The pitch track's voicing, or None.
        q: The ``ddk`` parameters.

    Returns:
        One run per region holding a syllable, each a list of ``(start, end)`` in time order. A nucleus
        is a peak of :func:`nucleus_score` standing ``nucleus_prominence_db`` over the troughs either
        side, at least ``nucleus_spacing_s`` from the next; a syllable runs from the trough before its
        nucleus to the trough after, the region's edges closing the first and last. A syllable longer
        than ``syllable_max_s`` or shorter than ``syllable_min_s`` is not one.
    """
    from scipy.signal import find_peaks  # noqa: PLC0415

    score = nucleus_score(view, voicing, q)
    times = view.frames.times_s
    hop = view.hop_s
    runs: list[list[Span]] = []
    for region in view.regions:
        inside = np.flatnonzero((times >= region.start_s) & (times <= region.end_s))
        if not len(inside):
            continue
        series = score[inside]
        peaks, _ = find_peaks(
            series,
            prominence=float(q["nucleus_prominence_db"]),
            distance=max(1, int(round(float(q["nucleus_spacing_s"]) / hop))),
        )
        if not len(peaks):
            peaks = np.asarray([int(np.argmax(series))])
        cuts = [int(a + np.argmin(series[a : b + 1])) for a, b in zip(peaks[:-1], peaks[1:])]
        edges = [float(region.start_s), *(float(times[inside[c]]) for c in cuts), float(region.end_s)]
        run = [
            (a, b)
            for a, b in zip(edges[:-1], edges[1:])
            if float(q["syllable_min_s"]) <= b - a <= float(q["syllable_max_s"])
        ]
        if run:
            runs.append(run)
    return runs


def onset_intervals(runs: Sequence[Sequence[Span]]) -> list[float]:
    """The onset-to-onset intervals between consecutive spans inside each run.

    Args:
        runs: Spans grouped into runs.

    Returns:
        The intervals, run by run.
    """
    return [b[0] - a[0] for run in runs for a, b in zip(run[:-1], run[1:])]


def cycle_period_s(
    runs: Sequence[Sequence[Span]], syllables_per_cycle: int, rhythm: Rhythm | None, q: dict[str, Any]
) -> float | None:
    """The cycle period the grouping reads: the rhythm's where it agrees with the syllables, else theirs.

    Args:
        runs: The syllable runs.
        syllables_per_cycle: The template's syllable count.
        rhythm: The cycle-band modulation peak, or None.
        q: The ``ddk`` parameters.

    Returns:
        The rhythm's period where it lies within ``rhythm_agreement`` (a factor) of the syllable count
        times the median syllable interval; that product where it does not or no rhythm stands; the
        rhythm's period with no interval to read; None with neither.
    """
    intervals = onset_intervals(runs)
    from_syllables = syllables_per_cycle * float(np.median(intervals)) if intervals else None
    if rhythm is None:
        return from_syllables
    if from_syllables is None:
        return rhythm.period_s
    factor = float(q["rhythm_agreement"])
    return rhythm.period_s if from_syllables / factor <= rhythm.period_s <= from_syllables * factor else from_syllables


def cycles_of(runs: Sequence[Sequence[Span]], period_s: float | None, syllables_per_cycle: int) -> list[list[Span]]:
    """The syllables grouped into cycles of the target word.

    A run is split into ``max(1, round(duration / period))`` cycles, each boundary at the syllable
    boundary nearest its share of the run, so a pause always closes a cycle and a leading or
    trailing partial cycle stands as one.

    Args:
        runs: The syllable runs.
        period_s: The cycle period (:func:`cycle_period_s`); None splits each run into cycles of
            ``syllables_per_cycle`` syllables.
        syllables_per_cycle: The template's syllable count.

    Returns:
        The cycles, each a list of its syllables, in time order.
    """
    cycles: list[list[Span]] = []
    for run in runs:
        run = list(run)
        duration = run[-1][1] - run[0][0]
        if period_s is None or period_s <= 0.0:
            count = max(1, int(round(len(run) / max(1, syllables_per_cycle))))
        else:
            count = max(1, int(round(duration / period_s)))
        count = min(count, len(run))
        starts = [s for s, _ in run]
        cuts: list[int] = []
        for j in range(1, count):
            target = run[0][0] + j * duration / count
            index = int(np.argmin([abs(s - target) for s in starts[1:]])) + 1
            if index not in cuts and (not cuts or index > cuts[-1]):
                cuts.append(index)
        bounds = [0, *cuts, len(run)]
        cycles.extend(run[a:b] for a, b in zip(bounds[:-1], bounds[1:]) if b > a)
    return cycles


def identity_of(positions: PositionMasses, span: Span, q: dict[str, Any]) -> Identity:
    """One event scored against the template by a local alignment allowing skips, deletions and substitutions.

    The states are a filler before the template, its positions in order, and a filler after; the
    fillers explain silence only, so a phoneme outside the template inside the event is charged to a
    position, a substitution that keeps its low mass. A path may enter at any position and leave from
    any, stay at a position over several frames, and jump forward over positions; each position
    jumped over is a deletion costing ``skip_log``.

    Args:
        positions: The per-position class masses.
        span: The event, widened by ``identity_pad_s`` either side.
        q: The ``ddk`` parameters.

    Returns:
        The identity: per position the mean and the peak class mass over the frames charged to it, and
        the mean of the peaks as the score; a deleted position's masses are None and count 0.
    """
    n = len(positions.template)
    spf = positions.seconds_per_frame
    pad = float(q["identity_pad_s"])
    first = max(0, int(np.floor((span[0] - pad) / spf)))
    last = min(positions.masses.shape[0], int(np.ceil((span[1] + pad) / spf)))
    if n == 0 or last <= first:
        return Identity(0.0, tuple(None for _ in range(n)), tuple(None for _ in range(n)))
    masses = positions.masses[first:last]
    filler = np.log(np.maximum(positions.filler[first:last], positions.floor))[:, None]
    floor = positions.floor
    emissions = np.concatenate([filler, np.log(np.maximum(masses, floor)), filler], axis=1)
    states = n + 2
    skip = float(q["skip_log"])
    transition = np.full((states, states), -np.inf)
    transition[0, 0] = 0.0
    transition[n + 1, n + 1] = 0.0
    for j in range(n):
        transition[0, 1 + j] = skip * j
        transition[1 + j, n + 1] = skip * (n - 1 - j)
        for i in range(j, n):
            transition[1 + j, 1 + i] = 0.0 if i == j else skip * (i - j - 1)
    start = np.array([0.0, *(skip * j for j in range(n)), skip * n])
    score = start + emissions[0]
    frames = emissions.shape[0]
    back = np.zeros((frames, states), dtype=int)
    for t in range(1, frames):
        candidates = score[:, None] + transition
        back[t] = np.argmax(candidates, axis=0)
        score = candidates[back[t], np.arange(states)] + emissions[t]
    end = score + np.array([skip * n, *(skip * (n - 1 - j) for j in range(n)), 0.0])
    path = np.zeros(frames, dtype=int)
    path[-1] = int(np.argmax(end))
    for t in range(frames - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    realised: list[float | None] = []
    peaks: list[float | None] = []
    for j in range(n):
        charged = path == 1 + j
        realised.append(float(masses[charged, j].mean()) if charged.any() else None)
        peaks.append(float(masses[charged, j].max()) if charged.any() else None)
    return Identity(float(np.mean([m or 0.0 for m in peaks])), tuple(realised), tuple(peaks))


def dispersion(intervals: Sequence[float]) -> float | None:
    """The coefficient of variation of an interval sequence.

    Args:
        intervals: The periods.

    Returns:
        The sample standard deviation over the mean, or None on fewer than two intervals or a
        non-positive mean.
    """
    values = np.asarray(intervals, dtype=float)
    if values.size < 2:
        return None
    mean = float(values.mean())
    if mean <= 0.0:
        return None
    return float(np.std(values, ddof=1) / mean)


def trend(intervals: Sequence[float]) -> float | None:
    """How the interval changes across the train: seconds per step.

    Args:
        intervals: The periods.

    Returns:
        The least-squares slope, or None on fewer than two intervals.
    """
    values = np.asarray(intervals, dtype=float)
    if values.size < 2:
        return None
    return float(np.polyfit(np.arange(values.size, dtype=float), values, 1)[0])


def _round(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(float(value), digits)


@dataclass(frozen=True)
class DdkReading:
    """What the task layer read of one declared syllable-repetition task.

    Attributes:
        unit: ``syllable`` for a single-syllable family, ``cycle`` for a sequence.
        evidence: The events read against the background, with the DDK decision in place of the
            shared one.
        identities: One identity per found event, in the evidence's ``found`` order.
        syllables: Every syllable, in time order.
        identity: The median identity over the events standing over the floor, or None with none.
        annotations: Counts, rates and regularity; they decide nothing.
        absent: The stored inputs the reading could not be taken without.
    """

    unit: str
    evidence: TaskEvidence | None
    identities: tuple[Identity, ...] = ()
    syllables: tuple[Span, ...] = ()
    identity: float | None = None
    annotations: dict[str, Any] = field(default_factory=dict)
    absent: tuple[str, ...] = ()

    @property
    def decision(self) -> str | None:
        """The decision, or None where the reading was not taken."""
        return None if self.evidence is None else self.evidence.decision

    @property
    def extent(self) -> Span | None:
        """The first task event to the last."""
        return None if self.evidence is None else self.evidence.extent

    def record(self) -> dict[str, Any]:
        """The reading, for the store: the decision, its inputs, and the evidence it was read off."""
        if self.evidence is None:
            return {"decision": None, "unit": self.unit, "absent": list(self.absent)}
        evidence = self.evidence.record()
        by_span = {
            (round(e.start_s, 3), round(e.end_s, 3)): i.record() for e, i in zip(self.evidence.found, self.identities)
        }
        for event in evidence["events"]:
            event["identity"] = by_span.get((event["start_s"], event["end_s"]))
        return {
            "decision": self.evidence.decision,
            "why": self.evidence.why,
            "unit": self.unit,
            "events_n": len(self.evidence.events),
            "events_found_n": self.evidence.events_found_n,
            "syllables_n": len(self.syllables),
            "identity": _round(self.identity),
            "inputs": dict(self.evidence.inputs),
            "annotations": dict(self.annotations),
            "absent": list(self.absent),
            "reading": {
                "evidence": evidence,
                "syllables": [[round(a, 3), round(b, 3)] for a, b in self.syllables],
            },
        }


def decide_ddk(
    events: Sequence[TaskEvent], identities: Sequence[Identity], p: dict[str, Any], q: dict[str, Any], unit: str
) -> tuple[str, str, dict[str, Any], float | None]:
    """The DDK decision on the events and their identities.

    Args:
        events: Every event the type test kept.
        identities: One per event, in the same order.
        p: The shared ``decision`` section of ``data/task_events.yaml``.
        q: The ``ddk`` section.
        unit: ``syllable`` or ``cycle``, which selects ``events_min``.

    Returns:
        ``(decision, why, inputs, identity)``. ``absent`` where no event stands ``snr_low_db`` over the
        floor, or the events' median identity is under ``identity_absent_max`` (another activity);
        ``present`` where at least ``events_min`` events stand ``snr_high_db`` over their local
        background with no impulse touching them and the identity reaches ``identity_min``;
        ``review`` otherwise: the identity is ambiguous, the clear events are entangled, or too few
        are clear (``weak``).
    """
    low, high = float(p["snr_low_db"]), float(p["snr_high_db"])
    standing = [(e, i) for e, i in zip(events, identities) if not e.recovered and e.snr_db >= low]
    identity = float(np.median([i.score for _, i in standing])) if standing else None
    clear = [e for e, _ in standing if e.clear_db >= high]
    free = [e for e in clear if not e.entangled]
    needed = int(q["events_min"][unit])
    inputs = {
        "floor_db": round(max(e.snr_db for e in events), 2) if events else None,
        "standing_n": len(standing),
        "clear_n": len(clear),
        "clear_free_n": len(free),
        "events_min": needed,
        "identity": _round(identity),
        "identity_min": float(q["identity_min"]),
        "identity_absent_max": float(q["identity_absent_max"]),
        "snr_low_db": low,
        "snr_high_db": high,
    }
    if not standing:
        return ABSENT, "no syllable train over the floor", inputs, identity
    if identity is not None and identity < float(q["identity_absent_max"]):
        return ABSENT, "not the target", inputs, identity
    if identity is not None and identity < float(q["identity_min"]):
        return REVIEW, "identity", inputs, identity
    if len(free) >= needed:
        return PRESENT, "clear", inputs, identity
    if len(clear) >= needed:
        return REVIEW, "entangled", inputs, identity
    return REVIEW, "weak", inputs, identity


def ddk_reading_of(
    view: GenericView | None,
    positions: PositionMasses | None,
    *,
    sequence: bool,
    syllables_per_cycle: int,
    voicing: Voicing | None = None,
    required_count: int | None = None,
    recording_s: float | None = None,
    p: dict[str, Any] | None = None,
) -> DdkReading:
    """Read one declared syllable-repetition task off the background and the posteriorgram.

    Args:
        view: The background, or None where BACKGROUND wrote none.
        positions: The template's class masses, or None where the posteriorgram is absent.
        sequence: Whether the family repeats a sequence (its events are cycles) or one syllable.
        syllables_per_cycle: The template's syllable count.
        voicing: The pitch track's voicing, or None.
        required_count: The count the instruction spoke, or None.
        recording_s: The recording's duration, for the train fraction.
        p: The parameters; ``data/task_events.yaml`` when None.

    Returns:
        The reading; with ``absent`` naming ``background_model`` or ``ppg_posteriorgram`` where either
        input is missing. Its task events are the dominant cluster (``gap_s``) of the events standing
        over the floor whose identity reaches ``identity_absent_max``; the extent is the first of them
        to the last, with nothing added.
    """
    unit = CYCLE if sequence else SYLLABLE
    absent = tuple(
        name for name, held in (("background_model", view), ("ppg_posteriorgram", positions)) if held is None
    )
    if view is None or positions is None:
        return DdkReading(unit, None, absent=absent)
    p = p or task_events_parameters()
    q = dict(p["ddk"])
    runs = syllable_runs(view, voicing, q)
    syllables = tuple(s for run in runs for s in run)
    rhythm = rhythm_of(view, q["cycle_band_hz"], float(q["rhythm_prominence_min_db"])) if sequence else None
    period = cycle_period_s(runs, syllables_per_cycle, rhythm, q) if sequence else None
    groups = cycles_of(runs, period, syllables_per_cycle) if sequence else [[s] for s in syllables]
    candidates = [(g[0][0], g[-1][1]) for g in groups]
    shared = evidence_of(view, candidates, gap_s=float(q["gap_s"]), rhythm=rhythm, inhale=False, p=p)
    shared = _entangled_from_outside(view, shared, runs, float(p["entangle_abut_s"]))
    identities = tuple(identity_of(positions, (e.start_s, e.end_s), q) for e in shared.found)
    decision, why, inputs, identity = decide_ddk(shared.found, identities, p["decision"], q, unit)
    low, other = float(p["decision"]["snr_low_db"]), float(q["identity_absent_max"])
    targets = [e for e, i in zip(shared.found, identities) if not e.recovered and e.snr_db >= low and i.score >= other]
    cluster = dominant_cluster(targets, float(q["gap_s"]))
    extent = (cluster[0].start_s, max(e.end_s for e in cluster)) if cluster else None
    evidence = replace(shared, events=tuple(cluster), extent=extent, decision=decision, why=why, inputs=inputs)
    annotations = _annotations(
        evidence,
        identities,
        runs,
        groups,
        positions,
        required_count=required_count,
        recording_s=recording_s,
        sequence=sequence,
        period_s=period,
    )
    return DdkReading(unit, evidence, identities, syllables, identity, annotations)


def _entangled_from_outside(
    view: GenericView, evidence: TaskEvidence, runs: Sequence[Sequence[Span]], abut_s: float
) -> TaskEvidence:
    """The evidence with each event entangled only by an impulse outside every syllable run.

    A stop's release is impulsive and the syllables tile their run, so an impulse inside a run is
    the task's own; one outside every run that overlaps or abuts an event entangles it.

    Args:
        view: The background.
        evidence: The shared evidence.
        runs: The syllable runs.
        abut_s: How close an impulse may lie and still abut an event.

    Returns:
        The evidence with ``entangled`` re-read on its found events and its cluster.
    """
    hulls = [(run[0][0], run[-1][1]) for run in runs if run]
    outside = [imp for imp in view.impulses if not any(a <= imp.start_s and imp.end_s <= b for a, b in hulls)]

    def reread(event: TaskEvent) -> TaskEvent:
        touched = any(imp.end_s >= event.start_s - abut_s and imp.start_s <= event.end_s + abut_s for imp in outside)
        return replace(event, entangled=touched)

    return replace(
        evidence,
        events=tuple(reread(e) for e in evidence.events),
        found=tuple(reread(e) for e in evidence.found),
    )


def _annotations(
    evidence: TaskEvidence,
    identities: Sequence[Identity],
    runs: Sequence[Sequence[Span]],
    groups: Sequence[Sequence[Span]],
    positions: PositionMasses,
    *,
    required_count: int | None,
    recording_s: float | None,
    sequence: bool,
    period_s: float | None,
) -> dict[str, Any]:
    """The readings beside the decision: count against the instruction, rates, regularity, mass."""
    cluster = list(evidence.events)
    onsets = [e.start_s for e in cluster]
    periods = [b - a for a, b in zip(onsets[:-1], onsets[1:])]
    intervals = onset_intervals(runs)
    syllable_rate = 1.0 / float(np.median(intervals)) if intervals else None
    cycle_rate = 1.0 / float(np.median(periods)) if sequence and periods else None
    extent = evidence.extent
    by_event = {(e.start_s, e.end_s): i for e, i in zip(evidence.found, identities)}
    charged = [by_event[(e.start_s, e.end_s)] for e in cluster if (e.start_s, e.end_s) in by_event]
    per_position = [
        _round(float(np.mean([i.realised[j] or 0.0 for i in charged]))) if charged else None
        for j in range(len(positions.template))
    ]
    count = len(cluster)
    return {
        "events_n": count,
        "required_count": required_count,
        "count_fraction": None if not required_count else round(count / required_count, 3),
        "syllable_rate_hz": _round(syllable_rate),
        "cycle_rate_hz": _round(cycle_rate),
        "cycle_period_prior_s": _round(period_s),
        "syllables_per_event": _round(float(np.mean([len(g) for g in groups]))) if groups and sequence else None,
        "period_cv": _round(dispersion(periods)),
        "period_trend_s_per_step": _round(trend(periods), 4),
        "train_fraction": None
        if extent is None or not recording_s
        else round((extent[1] - extent[0]) / recording_s, 3),
        "positions": list(positions.template),
        "classes": list(positions.classes),
        "realised_mass": per_position,
    }
