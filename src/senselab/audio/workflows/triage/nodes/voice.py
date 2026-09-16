"""VOICE — sustained phonation, proposed from the evidence the branch already routes on.

Two entry points, selected by :func:`~senselab.audio.workflows.triage.nodes.branches.dispatch` from
the declared task family. :func:`align_voice` evaluates a voice-eliciting task against what its
instruction asked for; :func:`detect_voice` marks sustained phonation on a task of any other kind
and evaluates nothing.

Both write by ``propose`` only. The subject is PREPROCESS's ``amplitude`` spans qualified by
``phonation_tracks`` and ``continuity_trace`` — the same evidence the route was decided on — and
VOICE mints its own ``family: "voice"`` spans over what qualifies. It waits for no ``phonation``
span and edits none.

The design is ``specs/20260817-triage-workflow-dag/branch-voice.md``; what porting it decided is
``specs/20260817-triage-workflow-dag/branch-voice-implementation.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.branches import (
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
    longest_monotone_run,
    max_windowed_spread,
    measured,
    mode_of,
    ordered_run,
    overlaps,
    propose_spans,
    read_continuity_track,
    read_phonation_tracks,
    semitones,
    stream_extent,
    touches_edge,
    trace_slice,
    track_slice,
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
    write_report,
)
from senselab.audio.workflows.triage.vocabulary import TASK
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

PHONATION_ROLE = "phonation"
"""The role :func:`detect_voice` mints under: sustained phonation, on nobody's declared task."""

TRACKS_ABSENT = "phonation_tracks is absent; no phonation boundary can be placed"
"""Why a mode could not look. An absent instrument is :data:`UNDETERMINED`, never a verdict."""

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
    """

    spans: list[Entity]
    words: list[Entity]
    tracks: PhonationTracks | None
    continuity: ContinuityTrack | None
    tracks_id: str | None
    continuity_id: str | None
    file_extent: tuple[float, float] | None

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
    tracks_measurement = find_measurement(store, PHONATION_TRACKS)
    continuity_measurement = find_measurement(store, CONTINUITY_TRACE)
    return Evidence(
        spans=live_entities(store, "span"),
        words=consensus_words(store),
        tracks=read_tracks(store, run_dir),
        continuity=read_continuity_track(store, run_dir),
        tracks_id=None if tracks_measurement is None else tracks_measurement.id,
        continuity_id=None if continuity_measurement is None else continuity_measurement.id,
        file_extent=stream_extent(store),
    )


@dataclass(frozen=True)
class Carrier:
    """One amplitude span that qualified as a phonation carrier, and why.

    Attributes:
        span: The amplitude span the production was found inside.
        track: The phonation tracks over it, with voicing already decided.
        voiced_fraction: How much of the carrier the tracker found F0 for.
        f0_spread_semitones: The worst local F0 spread over it.
        stationarity: The median spectral continuity over it.
    """

    span: Entity
    track: TrackSlice
    voiced_fraction: float
    f0_spread_semitones: float
    stationarity: float

    def qualifiers(self) -> dict[str, Any]:
        """The three properties every proposal and measurement carries.

        Returns:
            The qualifiers, rounded for the store.
        """
        return {
            "voiced_fraction": round(self.voiced_fraction, 4),
            "f0_spread_semitones": round(self.f0_spread_semitones, 3),
            "stationarity": round(self.stationarity, 4),
            "support_frames": int(self.track.voiced.sum()),
        }


def qualifying_phonation(evidence: Evidence, expectation: Expectation, params: BranchParams) -> list[Carrier]:
    """V1's stationarity qualifier, over PREPROCESS's amplitude spans. One body, three callers.

    The subject is ``measure == "amplitude"``, not ``family == "phonation"``: VOICE reads the
    amplitude spans as evidence and mints its own span over what qualifies.

    Args:
        evidence: The derivatives and store reads.
        expectation: The row, whose ``lexical_separator`` excludes a count-in from the carriers.
        params: The operating points.

    Returns:
        The qualifying carriers, in the order the spans were given.
    """
    if evidence.tracks is None:
        return []
    minimum_s = params.point("production_min_s")
    strength_min = params.point("voiced_strength_min")
    if minimum_s is None or strength_min is None:
        return []
    fraction_min = params.point("voiced_fraction_min")
    spread_window_s = params.point("f0_spread_window_s")
    spread_max = params.point("f0_spread_max_semitones")
    continuity_min = params.point("continuity_min")
    words = lexical(evidence.words)
    out: list[Carrier] = []
    for span in amplitude_spans(evidence.spans):
        if span.extent is None or duration(span.extent) < minimum_s:
            continue
        if expectation.lexical_separator and any(overlaps(word_extent(word), span.extent) for word in words):
            continue
        track = track_slice(evidence.tracks, span.extent, strength_min)
        if track.strength.size == 0:
            continue
        voiced_fraction = float(track.voiced.mean())
        pitch = semitones(np.where(track.voiced, track.f0_hz, np.nan))
        spread = float("nan") if spread_window_s is None else max_windowed_spread(pitch, track.hop_s, spread_window_s)
        trace = (
            np.empty(0, dtype=float) if evidence.continuity is None else trace_slice(evidence.continuity, span.extent)
        )
        stationarity = float(np.median(trace)) if trace.size else 0.0
        # A qualifier whose own boundary is unmeasured is not applied: it could neither admit nor
        # reject this carrier, and rejecting on it would be this branch deciding for want of a
        # number. The ask is recorded in `params.missing`.
        if (
            (fraction_min is None or voiced_fraction >= fraction_min)
            and (spread_max is None or spread_window_s is None or spread <= spread_max)
            and (continuity_min is None or stationarity >= continuity_min)
        ):
            out.append(Carrier(span, track, voiced_fraction, spread, stationarity))
    return out


def align_voice(
    task_family: str,
    store: ProvStore,
    hint: AudioHints | None,
    params: BranchParams,
    *,
    run_dir: Path,
) -> Result:
    """Evaluate a voice-eliciting task against what its instruction asked for.

    Args:
        task_family: The declared family, which is a key of :data:`VOICE_EXPECTATIONS`.
        store: The provenance store.
        hint: What the recording was declared to contain.
        params: The operating points.
        run_dir: The run directory sidecar paths are relative to.

    Returns:
        Whether the expected patterns were found, the spans proposed, and the findings.

    Raises:
        KeyError: If ``task_family`` is not a VOICE family, which is the caller's error: the four
            ``VOICE_EXPECTATIONS_PENDING_DECLARATION`` rows are out of family and take
            :func:`detect_voice`.
        NotImplementedError: If a row carries a pattern no reachable matcher serves.
    """
    expectation = VOICE_EXPECTATIONS[task_family]
    evidence = read_evidence(store, run_dir)
    if expectation.pattern is Pattern.SUSTAINED:
        return _voice_sustained(expectation, evidence, hint, params)
    if expectation.pattern is Pattern.GLIDE:
        return _voice_glide(expectation, evidence, params)
    raise NotImplementedError(
        f"{task_family} carries pattern {expectation.pattern.value!r}; align_voice serves "
        f"{Pattern.SUSTAINED.value!r} and {Pattern.GLIDE.value!r}, which is every pattern in "
        "VOICE_EXPECTATIONS"
    )


def _count_in(
    expectation: Expectation, evidence: Evidence, params: BranchParams
) -> tuple[bool, list[Proposal], list[Finding]]:
    """The lexical count-in the instruction prescribes, as its own span.

    It gets one precisely because it must be excluded from the vowel's measurement window.

    Args:
        expectation: The row, whose ``tokens`` are the prescribed count-in.
        evidence: The derivatives and store reads.
        params: The operating points.

    Returns:
        Whether the count-in was realised, the span it proposes, and one ``omission`` per token
        nothing realised.
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


def _voice_sustained(
    expectation: Expectation, evidence: Evidence, hint: AudioHints | None, params: BranchParams
) -> Result:
    """The held vowel: a ``count_in`` where the instruction prescribes one, and a ``task_extent``.

    Args:
        expectation: The row.
        evidence: The derivatives and store reads.
        hint: What the recording was declared to contain.
        params: The operating points.

    Returns:
        The result. :data:`UNDETERMINED` when the tracks are absent, so no boundary can be placed.
    """
    count_in_found, components, findings = _count_in(expectation, evidence, params)
    if evidence.tracks is None:
        findings.append(unviable("phonation_extent", TRACKS_ABSENT))
        return Result(UNDETERMINED, components, findings)

    carriers = sorted(
        qualifying_phonation(evidence, expectation, params),
        key=lambda carrier: duration(carrier.span.extent),
        reverse=True,
    )
    findings.append(count("attempt_count", len(carriers), expectation.expected_event_count))
    if not carriers:
        return Result(False, components, findings)

    carrier = carriers[0]
    assert carrier.span.extent is not None  # noqa: S101 — qualifying_phonation admits no other case
    extent = voiced_extent(carrier.span.extent, carrier.track)
    components.append(
        voice_span(
            TASK_EXTENT,
            extent,
            carrier.span.id,
            *evidence.derivations(evidence.tracks_id, evidence.continuity_id),
            production="sustained",
            carrier_extent=list(carrier.span.extent),
            **carrier.qualifiers(),
        )
    )
    findings.append(
        measured(
            "phonation_onset_to_offset_s", extent[0], extent[1], round(duration(extent), 3), **carrier.qualifiers()
        )
    )
    findings.append(
        measured(
            "voiced_duration_s",
            extent[0],
            extent[1],
            round(float(carrier.track.voiced.sum()) * carrier.track.hop_s, 3),
            **carrier.qualifiers(),
        )
    )
    findings.extend(_interruptions(carrier))
    if evidence.file_extent is not None and touches_edge(extent, evidence.file_extent):
        findings.append(deviation("truncation", extent[0], extent[1], reading="right_censored"))
    for extra in carriers[1:]:
        assert extra.span.extent is not None  # noqa: S101 — as above
        start, end = voiced_extent(extra.span.extent, extra.track)
        findings.append(deviation("repeat_attempt", start, end, **extra.qualifiers()))

    if expectation.expect_inhale:
        findings.append(count("inhale_expected_in_file", True, None))
    if expectation.forbid_lexical:
        for word in lexical(evidence.words):
            start, end = word_extent(word)
            findings.append(deviation("lexical_content", start, end, text=word_text(word)))
    if hint is not None:
        token = (hint.metadata or {}).get("task_token")
        if token is not None:
            findings.append(count("task_index", str(token).rsplit("-", 1)[-1], None))
    if expectation.declared_duration_s is not None:
        findings.append(
            count(
                "declared_duration_s",
                round(duration(evidence.file_extent), 2),
                expectation.declared_duration_s,
            )
        )
    return Result(count_in_found, components, findings)


def _interruptions(carrier: Carrier) -> list[Finding]:
    """The unvoiced intervals inside one attempt: their number, locations and total duration.

    Args:
        carrier: The qualifying carrier.

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
            total_s=round(sum(end - begin for begin, end in intervals), 3),
            locations=[[round(begin, 3), round(end, 3)] for begin, end in intervals],
            hop_s=round(hop, 4),
        )
    ]


def _voice_glide(expectation: Expectation, evidence: Evidence, params: BranchParams) -> Result:
    """The pitch sweep: one ``task_extent`` over the dominant monotone segment.

    Args:
        expectation: The row, whose ``declared_direction`` the measured one is read against.
        evidence: The derivatives and store reads.
        params: The operating points.

    Returns:
        The result. :data:`UNDETERMINED` when the tracks are absent, so no sweep can be placed.
    """
    if evidence.tracks is None:
        return Result(UNDETERMINED, [], [unviable("sweep_extent", TRACKS_ABSENT)])

    minimum_s = params.point("production_min_s")
    strength_min = params.point("voiced_strength_min")
    tolerance = params.point("monotone_tolerance_semitones")
    if minimum_s is None or strength_min is None or tolerance is None:
        return Result(UNDETERMINED, [], params.record())
    fraction_min = params.point("voiced_fraction_min")
    dominant_min = params.point("dominant_segment_min_fraction")

    best: tuple[Entity, TrackSlice, int, float, tuple[float, float]] | None = None
    for span in amplitude_spans(evidence.spans):
        if span.extent is None or duration(span.extent) < minimum_s:
            continue
        track = track_slice(evidence.tracks, span.extent, strength_min)
        if track.strength.size == 0:
            continue
        if fraction_min is not None and float(track.voiced.mean()) < fraction_min:
            continue
        pitch = semitones(np.where(track.voiced, track.f0_hz, np.nan))
        run = longest_monotone_run(pitch, tolerance)
        if run is None:
            continue
        first, last, sign = run
        sweep = (float(track.times_s[first]), float(track.times_s[last]) + track.hop_s)
        if dominant_min is not None and duration(sweep) / max(duration(span.extent), 1e-9) < dominant_min:
            continue
        if best is None or duration(sweep) > duration(best[4]):
            best = (span, track, sign, abs(float(pitch[last] - pitch[first])), sweep)

    if best is None:
        return Result(False, [], [count("sweep_found", False, True)])

    span, track, sign, extent_semitones, sweep = best
    direction = "up" if sign > 0 else "down"
    components = [
        voice_span(
            TASK_EXTENT,
            sweep,
            span.id,
            *evidence.derivations(evidence.tracks_id),
            production="glide",
            direction=direction,
            support_frames=int(track.voiced.sum()),
        )
    ]
    findings: list[Finding] = [
        measured(
            "glide_extent_semitones",
            sweep[0],
            sweep[1],
            round(extent_semitones, 2),
            direction=direction,
            support_frames=int(track.voiced.sum()),
        )
    ]
    if direction != expectation.declared_direction:
        findings.append(
            deviation(
                "sweep_direction_mismatch",
                sweep[0],
                sweep[1],
                declared=expectation.declared_direction,
                measured=direction,
                extent_semitones=round(extent_semitones, 2),
            )
        )
    if evidence.file_extent is not None and touches_edge(sweep, evidence.file_extent):
        findings.append(deviation("truncation", sweep[0], sweep[1], reading="right_censored"))
    return Result(True, components, findings)


def detect_voice(store: ProvStore, params: BranchParams, *, run_dir: Path) -> Result:
    """Mark sustained phonation wherever it occurs, and evaluate no task.

    Shares :func:`qualifying_phonation` with the in-family mode — the same qualifier, the same
    amplitude spans, the same tracks. What differs is that no expectation is consulted and no task
    is evaluated: ``done`` is :data:`UNDETERMINED`, always.

    Args:
        store: The provenance store.
        params: The operating points.
        run_dir: The run directory sidecar paths are relative to.

    Returns:
        A result whose ``done`` is :data:`UNDETERMINED`, one span per sustained region, and the
        findings.
    """
    evidence = read_evidence(store, run_dir)
    if evidence.tracks is None:
        return Result(UNDETERMINED, [], [unviable("phonation_extent", TRACKS_ABSENT)])

    neutral = Expectation(pattern=Pattern.SUSTAINED)
    carriers = qualifying_phonation(evidence, neutral, params)
    qualifying_ids = {carrier.span.id for carrier in carriers}

    components: list[Proposal] = []
    findings: list[Finding] = []
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
        findings.append(
            measured(
                "phonation_onset_to_offset_s",
                extent[0],
                extent[1],
                round(duration(extent), 3),
                **carrier.qualifiers(),
            )
        )
        findings.extend(_interruptions(carrier))
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
    findings.append(count("phonation_spans", len(components), None))
    return Result(UNDETERMINED, components, findings)


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
            index. A declaration this branch's measurements contradict is named by VERDICT's fold.
        run_dir: The run directory sidecar paths are relative to.

    Returns:
        The verdict, the view over the spans, findings and measurements written, and the verdict id.

    Raises:
        ValueError: If a ``branch.*`` key a reached body needs is unmeasured. The message names the
            key, on the recording it was asked about.
    """
    params = branch_params(config)

    def align(task_family: str, store: ProvStore, hint: AudioHints | None, params: BranchParams) -> Result:
        """Bind the run directory the mode contract does not carry. Names match ``AlignMode``."""
        return align_voice(task_family, store, hint, params, run_dir=run_dir)

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

    extents = [(proposal.start, proposal.end) for proposal in result.components]
    phonation_s = sum(end - start for start, end in extents)
    longest_span_s = max((end - start for start, end in extents), default=0.0)
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
        conformance=result.done,
        conformance_of=TASK,
        deviations=deviation_names(findings),
        unmeasured=tuple(params.missing),
        detail={
            "signal": source,
            "mode": mode,
            "declared_task_family": declared,
            "spans_n": len(result.components),
            "phonation_s": round(phonation_s, 3),
            "longest_span_s": round(longest_span_s, 3),
            "roles": sorted({proposal.role for proposal in result.components}),
            "phonation_tracks": not tracks_absent,
            "notes": notes,
        },
    )
    return BranchResult(report=report, view=(*span_ids, *finding_ids, report_id), report_entity_id=report_id)
