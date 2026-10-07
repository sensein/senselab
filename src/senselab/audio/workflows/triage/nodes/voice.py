"""VOICE — sustained phonation, proposed from the evidence the branch already routes on.

Two entry points, selected by :func:`~senselab.audio.workflows.triage.nodes.branches.dispatch` from
the declared task family. :func:`align_voice` evaluates a voice-eliciting task against what its
instruction asked for; :func:`detect_voice` marks sustained phonation on a task of any other kind
and evaluates nothing.

Both write by ``propose`` only. The subject is PREPROCESS's ``amplitude`` spans qualified by
``phonation_tracks`` and ``continuity_trace``, over which VOICE mints its own ``family: "voice"``
spans. It waits for no ``phonation`` span and edits none. Every span a qualifier discards is
reported as a ``carrier_rejected`` measurement naming the gate. The qualifier's bounds are the task
group's own, read from ``verdict.gates``; the readings it took off the carrier it selected are
reported as measurements, and VERDICT applies the same bounds to them to reach a conformance. This
branch reports no conformance at all.

The design is ``specs/20260817-triage-workflow-dag/branch-voice.md``; the grounds are
``voice-flag-grounds.md`` and ``branch-voice-implementation.md`` beside it, and the gates are
``specs/20260921-gates-in-verdict/``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.branches import (
    DETECT_GROUP,
    UNDETERMINED,
    VOICE_EXPECTATIONS,
    BranchParams,
    ContinuityTrack,
    Expectation,
    Finding,
    Pattern,
    PhonationTracks,
    Proposal,
    Result,
    TrackSlice,
    amplitude_spans,
    branch_params,
    contest,
    count,
    deviation,
    deviation_names,
    dispatch,
    duration,
    lexical,
    longest_free_interval,
    longest_monotone_run,
    measured,
    mode_of,
    monotone_reversal,
    ordered_run,
    overlaps,
    propose_spans,
    read_continuity_track,
    read_phonation_tracks,
    semitones,
    smoothed_pitch,
    stream_entity,
    touches_edge,
    trace_slice,
    track_slice,
    typical_windowed_spread,
    unviable,
    voice_span,
    voiced_extent,
    word_extent,
    word_text,
    write_findings,
)
from senselab.audio.workflows.triage.nodes.common import (
    BranchResult,
    consensus_words,
    find_measurement,
    live_entities,
    software_agent,
    write_measurement,
    write_report,
)
from senselab.audio.workflows.triage.vocabulary import TASK
from senselab.audio.workflows.triage.voice_phonation import PhonationReading, phonation_reading_of
from senselab.utils.prov_store import Entity, ProvStore

NODE = "VOICE"
KIND = "voice"

PHONATION_TRACKS = "phonation_tracks"
"""PREPROCESS's per-frame F0 and voicing strength. Without it no boundary can be placed."""

CONTINUITY_TRACE = "continuity_trace"
"""PREPROCESS's spectral-stationarity trace, which is the qualifier separating a held vowel."""

COUNT_IN = "count_in"
"""The role of the lexical count-in span, excluded from the vowel's measurement window."""

TASK_EXTENT = "task_extent"
"""The role of the span the task's own production runs over."""

PHONATION_READING = "voice_phonation_reading"
"""The measurement holding VOICE's phonation reading of a declared voice task, or the inputs it lacked."""

PHONATION_ROLE = "phonation"
"""The role :func:`detect_voice` mints under: sustained phonation, on nobody's declared task."""

TRACKS_ABSENT = "phonation_tracks is absent; no phonation boundary can be placed"
"""The ``unviable`` reason a mode writes when ``phonation_tracks`` never reached the store."""

_TRACKS_SIDECAR = "derivatives/phonation_tracks.npz"
"""Where PREPROCESS writes the tracks when the measurement carries no ``path`` of its own."""


@dataclass(frozen=True)
class Evidence:
    """Everything a VOICE mode reads, loaded once per node call.

    Attributes:
        spans: Every live span entity.
        words: The consensus words, in index order.
        tracks: F0 and its strength, or None when PREPROCESS wrote none.
        continuity: The spectral-stationarity trace, or None when PREPROCESS wrote none.
        tracks_id: The ``phonation_tracks`` measurement's id, or None when it is absent.
        continuity_id: The ``continuity_trace`` measurement's id, or None when it is absent.
        file_extent: The recording's own extent, or None when no stream carries one.
        file_id: The recording stream's id, or None when no stream carries an extent.
    """

    spans: list[Entity]
    words: list[Entity]
    tracks: PhonationTracks | None
    continuity: ContinuityTrack | None
    tracks_id: str | None
    continuity_id: str | None
    file_extent: tuple[float, float] | None
    file_id: str | None

    def derivations(self, *ids: str | None) -> tuple[str, ...]:
        """The evidence ids that exist, in the order given.

        Args:
            *ids: Candidate entity ids, any of which may be None.

        Returns:
            The ones that are not None.
        """
        return tuple(entity_id for entity_id in ids if entity_id is not None)


def read_tracks(store: ProvStore, run_dir: Path) -> PhonationTracks | None:
    """The phonation tracks, by the measurement's ``path`` when it carries one.

    Args:
        store: The provenance store.
        run_dir: The run directory sidecar paths are relative to.

    Returns:
        The tracks, or None when neither carrier yields them.
    """
    tracks = read_phonation_tracks(store, run_dir)
    if tracks is not None:
        return tracks
    if find_measurement(store, PHONATION_TRACKS) is None:
        return None
    sidecar = run_dir / _TRACKS_SIDECAR
    if not sidecar.is_file():
        return None
    with np.load(sidecar) as loaded:
        arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
    if not {"times_s", "f0_hz", "strength"} <= set(arrays):
        return None
    return PhonationTracks(
        times_s=np.asarray(arrays["times_s"], dtype=float),
        f0_hz=np.asarray(arrays["f0_hz"], dtype=float),
        strength=np.asarray(arrays["strength"], dtype=float),
    )


def read_evidence(store: ProvStore, run_dir: Path) -> Evidence:
    """Load every derivative and store read the two modes share.

    Args:
        store: The provenance store.
        run_dir: The run directory sidecar paths are relative to.

    Returns:
        The record. An absent derivative is None rather than an error.
    """
    recording = stream_entity(store)
    tracks_measurement = find_measurement(store, PHONATION_TRACKS)
    continuity_measurement = find_measurement(store, CONTINUITY_TRACE)
    return Evidence(
        spans=live_entities(store, "span"),
        words=consensus_words(store),
        tracks=read_tracks(store, run_dir),
        continuity=read_continuity_track(store, run_dir),
        tracks_id=None if tracks_measurement is None else tracks_measurement.id,
        continuity_id=None if continuity_measurement is None else continuity_measurement.id,
        file_extent=None if recording is None else recording.extent,
        file_id=None if recording is None else recording.id,
    )


@dataclass(frozen=True)
class Carrier:
    """One amplitude span that qualified as a phonation carrier, and why.

    Attributes:
        span: The amplitude span the production was found inside.
        track: The phonation tracks over it, with voicing already decided.
        voiced_fraction: How much of the carrier the tracker found F0 for.
        f0_spread_semitones: The representative local F0 spread over it.
        stationarity: The median spectral continuity over it.
        failed: The quality gates this carrier did not meet, evaluated but not applied. A carrier
            carrying any of these is still a carrier.
    """

    span: Entity
    track: TrackSlice
    voiced_fraction: float
    f0_spread_semitones: float
    stationarity: float
    failed: tuple["Rejection", ...] = ()

    def qualifiers(self) -> dict[str, Any]:
        """The three properties every proposal and measurement carries.

        Returns:
            The qualifiers, rounded for the store.
        """
        spread = self.f0_spread_semitones
        return {
            "voiced_fraction": round(self.voiced_fraction, 4),
            "f0_spread_semitones": round(spread, 3) if np.isfinite(spread) else None,
            "stationarity": round(self.stationarity, 4),
            "support_frames": int(self.track.voiced.sum()),
        }


@dataclass(frozen=True)
class Rejection:
    """One amplitude span this branch examined and discarded, and the gate that discarded it.

    Written to the store as a measurement, not a deviation.

    Attributes:
        span: The amplitude span that was examined.
        gate: The operating point that discarded it, by its ``branch.*`` key or role name.
        value: What was read for that gate, or None where the gate reads no scalar.
        bound: The value it was read against, or None for the same reason.
        qualifiers: Whatever else had been measured over the span by the time it was discarded.
    """

    span: Entity
    gate: str
    value: float | None
    bound: float | None
    qualifiers: dict[str, Any]

    def finding(self) -> Finding:
        """This rejection, as the measurement the store records.

        Returns:
            One ``carrier_rejected`` measurement over the discarded span's own extent.
        """
        start, end = self.span.extent or (0.0, 0.0)
        return measured(
            "carrier_rejected",
            start,
            end,
            self.gate,
            self.span.id,
            value_read=self.value,
            bound=self.bound,
            carrier_s=round(end - start, 3),
            **self.qualifiers,
        )

    def record(self) -> dict[str, Any]:
        """This rejection, as the report's own summary of it.

        Returns:
            The gate, the value read and the carrier's length.
        """
        start, end = self.span.extent or (0.0, 0.0)
        return {"gate": self.gate, "value_read": self.value, "carrier_s": round(end - start, 3)}


@dataclass(frozen=True)
class Qualification:
    """What the qualifier made of every amplitude span it was given.

    Attributes:
        carriers: The spans a production exists inside, in the order the spans were given. A
            carrier that missed a quality gate is here, carrying that gate in its ``failed``.
        rejected: The spans no production exists inside, each naming the criterion that decided so.
    """

    carriers: list[Carrier]
    rejected: list[Rejection]

    @property
    def steady(self) -> list[Carrier]:
        """The carriers that also met every quality gate.

        Returns:
            The carriers with an empty ``failed``, in the order the spans were given.
        """
        return [carrier for carrier in self.carriers if not carrier.failed]

    def findings(self) -> list[Finding]:
        """One measurement per span no production exists inside.

        Returns:
            The findings, in the order the spans were given.
        """
        return [rejection.finding() for rejection in self.rejected]

    def quality_findings(self) -> list[Finding]:
        """One measurement per carrier a quality gate discarded, for the arm that applies them.

        Returns:
            The findings, in the order the spans were given.
        """
        return [rejection.finding() for carrier in self.carriers for rejection in carrier.failed]


CARRIER_DURATION = "carrier_duration_s"
"""The reading ``production_min_s`` is read against: how long the qualifying carrier ran."""

CARRIER_VOICED_FRACTION = "carrier_voiced_fraction"
"""The reading ``voiced_fraction_min`` is read against."""

CARRIER_SPREAD = "carrier_f0_spread_semitones"
"""The reading ``f0_spread_max_semitones`` is read against."""

CARRIER_CONTINUITY = "carrier_continuity"
"""The reading ``continuity_min`` is read against."""

NO_VOICING = "no_voicing"
"""The existence criterion no bound configures: the tracker voiced not one frame of the carrier."""

GLIDE_EXTENT = "glide_extent_semitones"
"""The reading ``glide_extent_min_semitones`` is read against: how far the dominant run's pitch travelled."""


def qualifying_phonation(evidence: Evidence, expectation: Expectation, params: BranchParams) -> Qualification:
    """V1's stationarity qualifier, over PREPROCESS's ``measure == "amplitude"`` spans.

    Two kinds of criterion are separated here. An **existence** criterion decides that no
    production is present to measure, and discards the span. A **quality** criterion reads how good
    the production was; it is evaluated onto the carrier's ``failed`` and applied by the caller, so
    the arm that knows a task was declared can report the production instead of an absence.

    Every span it discards is returned beside the ones it keeps, named by the criterion that
    discarded it.

    Args:
        evidence: The derivatives and store reads.
        expectation: The row, whose ``lexical_separator`` excludes a count-in from the carriers: the words
            its ``tokens`` match where it prescribes them, else every lexical word.
        params: The operating points.

    Returns:
        The carriers a production exists inside and the spans none does.
    """
    if evidence.tracks is None:
        return Qualification([], [])
    minimum_s = params.gate("production_min_s")
    strength_min = params.point("voiced_strength_min")
    if minimum_s is None or strength_min is None:
        return Qualification([], [])
    fraction_min = params.gate("voiced_fraction_min")
    spread_window_s = params.point("f0_spread_window_s")
    spread_max = params.gate("f0_spread_max_semitones")
    continuity_min = params.gate("continuity_min")
    words = lexical(evidence.words)
    separators = words
    if expectation.lexical_separator and expectation.tokens:
        matched, _ = ordered_run(expectation.tokens, words, params.p_normalise)
        separators = [word for _, word in matched]
    out: list[Carrier] = []
    rejected: list[Rejection] = []
    for span in amplitude_spans(evidence.spans):
        if span.extent is None or duration(span.extent) < minimum_s:
            rejected.append(Rejection(span, "production_min_s", round(duration(span.extent), 3), minimum_s, {}))
            continue
        carrier_extent: tuple[float, float] = span.extent
        if expectation.lexical_separator and any(overlaps(word_extent(word), carrier_extent) for word in separators):
            free = longest_free_interval(carrier_extent, [word_extent(word) for word in separators])
            if free is None or duration(free) < minimum_s:
                rejected.append(Rejection(span, "lexical_separator", None, None, {}))
                continue
            carrier_extent = free
            span = replace(span, extent=free)
        track = track_slice(evidence.tracks, carrier_extent, strength_min)
        if track.strength.size == 0:
            rejected.append(Rejection(span, "no_track_over_carrier", None, None, {}))
            continue
        voiced_fraction = float(track.voiced.mean())
        if voiced_fraction <= 0.0:
            rejected.append(Rejection(span, NO_VOICING, 0.0, None, {}))
            continue
        pitch = semitones(np.where(track.voiced, track.f0_hz, np.nan))
        spread = (
            float("nan") if spread_window_s is None else typical_windowed_spread(pitch, track.hop_s, spread_window_s)
        )
        trace = (
            np.empty(0, dtype=float)
            if evidence.continuity is None
            else trace_slice(evidence.continuity, carrier_extent)
        )
        stationarity = float(np.median(trace)) if trace.size else 0.0
        carrier = Carrier(span, track, voiced_fraction, spread, stationarity)
        # An unmeasured bound, or an unmeasurable reading, leaves its gate unread.
        missed: Rejection | None = None
        if fraction_min is not None and voiced_fraction < fraction_min:
            missed = Rejection(
                span, "voiced_fraction_min", round(voiced_fraction, 4), fraction_min, carrier.qualifiers()
            )
        elif spread_max is not None and spread_window_s is not None and np.isfinite(spread) and spread > spread_max:
            missed = Rejection(span, "f0_spread_max_semitones", round(spread, 3), spread_max, carrier.qualifiers())
        elif continuity_min is not None and stationarity < continuity_min:
            missed = Rejection(span, "continuity_min", round(stationarity, 4), continuity_min, carrier.qualifiers())
        out.append(carrier if missed is None else replace(carrier, failed=(missed,)))
    return Qualification(out, rejected)


def phonation_words(store: ProvStore) -> list[tuple[float, float, str]]:
    """The lexical consensus words, as the phonation measure reads them.

    Args:
        store: The provenance store.

    Returns:
        ``(start, end, text)`` per lexical word that carries an extent.
    """
    return [
        (float(word.extent[0]), float(word.extent[1]), word_text(word))
        for word in lexical(consensus_words(store))
        if word.extent is not None
    ]


def read_phonation(store: ProvStore, run_dir: Path, task_family: str) -> PhonationReading | tuple[str, ...]:
    """The phonation measure over a declared voice task's streams.

    Args:
        store: The provenance store.
        run_dir: The run directory the streams' paths are relative to.
        task_family: The declared family, a key of :data:`VOICE_EXPECTATIONS`.

    Returns:
        The reading, or the inputs that were absent.
    """
    expectation = VOICE_EXPECTATIONS[task_family]
    direction = expectation.declared_direction if expectation.pattern is Pattern.GLIDE else None
    return phonation_reading_of(store, run_dir, words=phonation_words(store), glide_direction=direction)


def phonation_attributes(read: PhonationReading | tuple[str, ...]) -> dict[str, Any]:
    """The ``voice_phonation_reading`` measurement's attributes.

    Args:
        read: What :func:`read_phonation` returned.

    Returns:
        ``absent`` naming the missing inputs, or the reading's record beside an empty ``absent``.
    """
    if not isinstance(read, PhonationReading):
        return {"absent": list(read)}
    return {"absent": [], **read.record()}


def align_voice(
    task_family: str,
    store: ProvStore,
    hint: AudioHints | None,
    params: BranchParams,
    *,
    run_dir: Path,
    reading: PhonationReading | tuple[str, ...] | None = None,
) -> Result:
    """Evaluate a voice-eliciting task on its phonation attempt.

    The task extent is the attempt :func:`~senselab.audio.workflows.triage.voice_phonation.measure_phonation`
    places; the readings VERDICT's gates read are taken over it.

    Args:
        task_family: The declared family, which is a key of :data:`VOICE_EXPECTATIONS`.
        store: The provenance store.
        hint: What the recording was declared to contain.
        params: The operating points.
        run_dir: The run directory sidecar paths are relative to.
        reading: The phonation reading, where the caller has already taken it; None takes it here.

    Returns:
        The count-in and task-extent spans, and the readings.

    Raises:
        KeyError: If ``task_family`` is not a key of :data:`VOICE_EXPECTATIONS`.
    """
    expectation = VOICE_EXPECTATIONS[task_family]
    params.bind(expectation.pattern, task_family)
    evidence = read_evidence(store, run_dir)
    read = read_phonation(store, run_dir, task_family) if reading is None else reading
    components: list[Proposal] = []
    findings: list[Finding] = []
    if expectation.pattern is Pattern.SUSTAINED:
        _, components, findings = _count_in(expectation, evidence, params)
    if not isinstance(read, PhonationReading):
        name = "sweep_extent" if expectation.pattern is Pattern.GLIDE else "phonation_extent"
        findings.append(unviable(name, f"{', '.join(read)} is absent; no phonation can be read"))
        return Result(components, findings)
    if read.extent is None:
        findings.append(count("attempt_count", 0, None))
        return Result(components, findings)
    extent = (float(read.extent["start_s"]), float(read.extent["end_s"]))
    record = read.record()
    plain = next(
        (
            s.id
            for s in store.entities("stream")
            if s.attributes.get("name") == "plain" and not store.is_invalidated(s.id)
        ),
        None,
    )
    read_off = evidence.derivations(plain, evidence.file_id)
    components.append(
        voice_span(
            TASK_EXTENT,
            extent,
            *read_off,
            production="glide" if expectation.pattern is Pattern.GLIDE else "sustained",
            extent_from="phonation",
            onset_s=read.extent["onset_s"],
            holds_n=record["holds_n"],
            longest_hold_s=record["longest_hold_s"],
            voiced_fraction=record["voiced_fraction"],
        )
    )
    start, end = extent
    phonation_s = end - float(read.extent["onset_s"])
    findings.extend(
        [
            count("attempt_count", 1, None),
            measured(CARRIER_DURATION, start, end, round(end - start, 3), *read_off),
            measured(CARRIER_VOICED_FRACTION, start, end, record["voiced_fraction"], *read_off),
            measured(CARRIER_SPREAD, start, end, record["f0_spread_semitones"], *read_off),
            measured("phonation_onset_to_offset_s", start, end, round(phonation_s, 3), *read_off),
            measured("voiced_duration_s", start, end, record["voiced_s"], *read_off),
            measured("longest_hold_s", start, end, record["longest_hold_s"], *read_off, holds=record["holds"]),
        ]
    )
    if expectation.pattern is Pattern.GLIDE:
        findings.append(
            measured(
                GLIDE_EXTENT,
                start,
                end,
                record["glide"].get("travel_declared_semitones"),
                *read_off,
                direction=expectation.declared_direction,
                shape=record["shape"],
            )
        )
    return Result(components, findings)


def _count_in(
    expectation: Expectation, evidence: Evidence, params: BranchParams
) -> tuple[bool, list[Proposal], list[Finding]]:
    """The lexical count-in the instruction prescribes, as its own span.

    The span is marked ``excluded_from_measurement`` so the vowel's window does not cover it.

    Args:
        expectation: The row, whose ``tokens`` are the prescribed count-in.
        evidence: The derivatives and store reads.
        params: The operating points.

    Returns:
        Whether the count-in was realised, the span it proposes, and one ``omission`` per token
        nothing realised. The bool is the count-in's own reading and no part of the task's
        conformance.
    """
    if expectation.tokens is None:
        return True, [], []
    matched, omissions = ordered_run(expectation.tokens, lexical(evidence.words), params.p_normalise)
    findings = [deviation("omission", None, None, expected=token) for token in omissions]
    if not matched:
        return False, [], findings
    extent = (word_extent(matched[0][1])[0], word_extent(matched[-1][1])[1])
    proposals = [
        voice_span(
            COUNT_IN,
            extent,
            *(word.id for _, word in matched),
            tokens=[token for token, _ in matched],
            excluded_from_measurement=True,
        )
    ]
    return True, proposals, findings


def _interruptions(carrier: Carrier, *derived_from: str) -> list[Finding]:
    """The unvoiced intervals inside one attempt: their number, locations and total duration.

    Args:
        carrier: The qualifying carrier.
        *derived_from: The entity ids the attempt was read off.

    Returns:
        One ``interruptions`` measurement over the attempt.
    """
    times, voiced, hop = carrier.track.times_s, carrier.track.voiced, carrier.track.hop_s
    marks = np.flatnonzero(voiced)
    intervals: list[tuple[float, float]] = []
    start: float | None = None
    for index in range(int(marks[0]), int(marks[-1]) + 1) if marks.size else ():
        if not voiced[index] and start is None:
            start = float(times[index])
        elif voiced[index] and start is not None:
            intervals.append((start, float(times[index])))
            start = None
    extent = voiced_extent(carrier.span.extent or (0.0, 0.0), carrier.track)
    return [
        measured(
            "interruptions",
            extent[0],
            extent[1],
            len(intervals),
            *derived_from,
            total_s=round(sum(end - begin for begin, end in intervals), 3),
            locations=[[round(begin, 3), round(end, 3)] for begin, end in intervals],
            hop_s=round(hop, 4),
        )
    ]


def detect_voice(store: ProvStore, params: BranchParams, *, run_dir: Path) -> Result:
    """Mark sustained phonation wherever it occurs, and evaluate no task.

    Runs :func:`qualifying_phonation` against a neutral expectation, so no instruction is read.
    Nothing declared this recording to hold a held vowel, so the quality gates are what separate
    one from connected speech and this arm applies them.

    Args:
        store: The provenance store.
        params: The operating points.
        run_dir: The run directory sidecar paths are relative to.

    Returns:
        One span per sustained region, and the findings. It evaluates no task, so VERDICT applies
        no conformance gate to what it reports.
    """
    params.bind(DETECT_GROUP[NODE])
    evidence = read_evidence(store, run_dir)
    if evidence.tracks is None:
        return Result([], [unviable("phonation_extent", TRACKS_ABSENT)])

    neutral = Expectation(pattern=Pattern.SUSTAINED)
    qualification = qualifying_phonation(evidence, neutral, params)
    carriers = qualification.steady
    qualifying_ids = {carrier.span.id for carrier in carriers}

    components: list[Proposal] = []
    findings: list[Finding] = [*qualification.findings(), *qualification.quality_findings()]
    for carrier in carriers:
        assert carrier.span.extent is not None  # noqa: S101 — qualifying_phonation admits no other case
        extent = voiced_extent(carrier.span.extent, carrier.track)
        components.append(
            voice_span(
                PHONATION_ROLE,
                extent,
                carrier.span.id,
                *evidence.derivations(evidence.tracks_id, evidence.continuity_id),
                production="sustained",
                carrier_extent=list(carrier.span.extent),
                evaluates_no_task=True,
                **carrier.qualifiers(),
            )
        )
        read_off = (carrier.span.id, *evidence.derivations(evidence.tracks_id))
        findings.append(
            measured(
                "phonation_onset_to_offset_s",
                extent[0],
                extent[1],
                round(duration(extent), 3),
                *read_off,
                **carrier.qualifiers(),
            )
        )
        findings.extend(_interruptions(carrier, *read_off))
    for span in evidence.spans:
        if span.attributes.get("label") == PHONATION_ROLE and span.id not in qualifying_ids:
            findings.append(
                contest(
                    span.id,
                    span.extent or (0.0, 0.0),
                    PHONATION_ROLE,
                    "fails_the_stationarity_qualifier",
                )
            )
    findings.append(count("phonation_spans", len(components), None, *(carrier.span.id for carrier in carriers)))
    return Result(components, findings)


def voice(
    store: ProvStore,
    source: str,
    config: TriageConfig,
    hint: AudioHints | None = None,
    *,
    run_dir: Path,
) -> BranchResult:
    """Propose the sustained phonation in this recording, and evaluate the task when it declares one.

    Args:
        store: The provenance store, holding PREPROCESS's amplitude spans, its ``phonation_tracks``
            and its ``continuity_trace``.
        source: The store-held stream the findings are taken over, ``"plain"``.
        config: The triage configuration.
        hint: What the recording was declared to contain; it selects the mode and carries the task
            index.
        run_dir: The run directory sidecar paths are relative to.

    Returns:
        The branch report, the view over the spans, findings and measurements written, and the
        ``branch_report`` entity's id.

    Raises:
        ValueError: If a ``branch.*`` key a reached body needs is unmeasured.
    """
    params = branch_params(config)
    readings: dict[str, PhonationReading | tuple[str, ...]] = {}

    def align(task_family: str, store: ProvStore, hint: AudioHints | None, params: BranchParams) -> Result:
        """Bind the run directory the mode contract does not carry. Names match ``AlignMode``."""
        readings[task_family] = read_phonation(store, run_dir, task_family)
        return align_voice(task_family, store, hint, params, run_dir=run_dir, reading=readings[task_family])

    def detect(store: ProvStore, params: BranchParams) -> Result:
        """Bind the run directory the mode contract does not carry. Names match ``DetectMode``."""
        return detect_voice(store, params, run_dir=run_dir)

    mode, declared = mode_of(NODE, store, hint)
    result = dispatch(NODE, store, params, hint, align=align, detect=detect)
    tracks_absent = find_measurement(store, PHONATION_TRACKS) is None or read_tracks(store, run_dir) is None

    software = software_agent(store)
    activity = store.activity(
        node=NODE,
        step="phonation",
        parameters={
            "signal": source,
            "mode": mode,
            "declared_task_family": declared,
            "phonation_tracks": not tracks_absent,
            "continuity_trace": find_measurement(store, CONTINUITY_TRACE) is not None,
        },
    )
    store.was_associated_with(activity, software)
    for name in (PHONATION_TRACKS, CONTINUITY_TRACE):
        measurement = find_measurement(store, name)
        if measurement is not None:
            store.used(activity, measurement.id)
    for span in amplitude_spans(live_entities(store, "span")):
        store.used(activity, span.id)

    findings = [*result.deviations, *params.record()]
    span_ids = propose_spans(store, activity, software, result.components)
    finding_ids = write_findings(store, activity, software, findings, signal=source)
    for read in readings.values():
        finding_ids.append(
            write_measurement(
                store, activity, software, name=PHONATION_READING, signal="plain", attributes=phonation_attributes(read)
            )
        )

    extents = [(proposal.start, proposal.end) for proposal in result.components]
    phonation_s = sum(end - start for start, end in extents)
    longest_span_s = max((end - start for start, end in extents), default=0.0)
    rejected = [
        {
            "gate": str(finding.evidence["value"]),
            "value_read": finding.evidence.get("value_read"),
            "carrier_s": finding.evidence.get("carrier_s"),
        }
        for finding in findings
        if finding.kind == "measure" and finding.name == "carrier_rejected"
    ]
    notes: list[str] = []
    if tracks_absent:
        notes.append(TRACKS_ABSENT)
    if params.missing:
        notes.append(f"branch.* unmeasured: {', '.join(params.missing)}")
    report_id, report = write_report(
        store,
        activity,
        software,
        node=NODE,
        kind=KIND,
        conformance=UNDETERMINED,
        conformance_of=TASK,
        deviations=deviation_names(findings),
        unmeasured=tuple(params.missing),
        in_family=mode == "align",
        detail={
            "signal": source,
            "mode": mode,
            "declared_task_family": declared,
            "spans_n": len(result.components),
            "phonation_s": round(phonation_s, 3),
            "longest_span_s": round(longest_span_s, 3),
            "roles": sorted({proposal.role for proposal in result.components}),
            "phonation_tracks": not tracks_absent,
            "carriers_rejected": rejected,
            "carriers_rejected_n": len(rejected),
            "notes": notes,
        },
    )
    return BranchResult(report=report, view=(*span_ids, *finding_ids, report_id), report_entity_id=report_id)
