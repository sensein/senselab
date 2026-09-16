"""DDK — find the repeated syllable train, measure its rate and regularity, evaluate a declared task.

Two modes, as every branch has: :func:`align_ddk` evaluates one of the ten ``SYLLABLE_REPETITION``
families against what its instruction asked for, and :func:`detect_ddk` finds rapid repetition
wherever it occurs on a task of any other kind and evaluates nothing.

The design is ``specs/20260817-triage-workflow-dag/branch-ddk.md`` (D1–D6); the ported bodies are
``expected-patterns.md``; what porting decided, and the two places this module departs from either,
is ``specs/20260817-triage-workflow-dag/branch-ddk-implementation.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.branches import (
    DDK_EXPECTATIONS,
    UNDETERMINED,
    BranchParams,
    Done,
    EnvelopeTrack,
    Expectation,
    Finding,
    Pattern,
    Proposal,
    Result,
    SpectrogramBlock,
    acquisition_covariates,
    amplitude_spans,
    band_power,
    branch_params,
    count,
    ddk_span,
    declared_duration_count,
    deviation,
    deviation_names,
    dispatch,
    duration,
    events_in_span,
    hull,
    lexical,
    measured,
    mode_of,
    propose_spans,
    read_envelope_track,
    read_spectrogram_block,
    stream_extent,
    touches_edge,
    train_rate_hz,
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

NODE = "DDK"
"""The branch's name, in the vocabulary's own spelling."""

KIND = "ddk"
"""The subject this branch is the authority on; also its span family."""

RATE = "ddk_syllable_rate_from_envelope_modulation_hz"
"""The one rate measurement, named for the instrument that took it."""

ENVELOPE = "energy_envelope"
WIDEBAND = "spectrogram_wideband"
TRANSCRIPT = "consensus_transcript"

UNRESOLVED = "unresolved"
"""The place a burst spectrum did not separate, which is not a substitution."""

SYLLABLES_PER_S = "syllables_per_s"
CYCLES_OR_SYLLABLES_PER_S = "cycles_or_syllables_per_s"
"""A sequential train modulates at the cycle rate as well as the syllable rate; the peak is one of
the two and the harmonic-equality tolerance that would separate them is unmeasured, so the unit is
carried as ambiguous rather than resolved by assumption."""

TRAIN_ROLES = ("task_extent", "repetition")
"""The roles that are a train. ``lexical_repetition`` is a repeated word, not a syllable train."""

NO_TRAIN = "no syllable train was found"
NO_INSTRUMENT = "the energy envelope is absent; this branch's only rate instrument could not be read"


# --------------------------------------------------------------------- what the two modes read


@dataclass(frozen=True)
class DdkReads:
    """The stored derivatives DDK measures over, read once by the node.

    The two mode signatures carry ``(store, params)`` and no run directory, and every sidecar path
    in the store is relative to one, so the loaders cannot run inside a mode. They run in
    :func:`ddk`, which has the run directory, and their results arrive here.

    Attributes:
        envelope: The energy envelope and its global floor, or None when the derivative is absent.
        envelope_id: That measurement's entity id, for the derivation.
        wideband: The wideband spectrogram, or None when it or the working rate is absent.
        wideband_id: That measurement's entity id.
        transcript_id: The consensus transcript's entity id, for a lexical proposal's derivation.
    """

    envelope: EnvelopeTrack | None = None
    envelope_id: str | None = None
    wideband: SpectrogramBlock | None = None
    wideband_id: str | None = None
    transcript_id: str | None = None


def working_rate(store: ProvStore, source: str) -> float | None:
    """The rate the spectrogram's hop and bin centres are in, which the derivative does not record.

    Args:
        store: The provenance store.
        source: The stream the derivatives were taken over.

    Returns:
        The rate in Hz from that stream entity, or from the envelope measurement, which PREPROCESS
        writes at the same resampled rate; None when neither carrier is in the store.
    """
    streams = [
        entity
        for entity in live_entities(store, "stream")
        if entity.attributes.get("name") == source and entity.attributes.get("sampling_rate")
    ]
    if streams:
        return float(streams[-1].attributes["sampling_rate"])
    envelope = find_measurement(store, ENVELOPE)
    rate = None if envelope is None else envelope.attributes.get("sampling_rate")
    return None if rate is None else float(rate)


def read_ddk(store: ProvStore, run_dir: Path, source: str) -> DdkReads:
    """Load every derivative the two modes measure over, each independently absent.

    Args:
        store: The provenance store.
        run_dir: The run directory sidecar paths are relative to.
        source: The stream the derivatives were taken over.

    Returns:
        The reads. A derivative that never reached the store, or whose sidecar is gone, is None
        rather than an error: an absent instrument is an absence, never a negative reading.
    """
    envelope = find_measurement(store, ENVELOPE)
    wideband = find_measurement(store, WIDEBAND)
    transcript = find_measurement(store, TRANSCRIPT)
    rate = working_rate(store, source)
    block = None if rate is None else read_spectrogram_block(store, run_dir, WIDEBAND, rate)
    return DdkReads(
        envelope=read_envelope_track(store, run_dir),
        envelope_id=None if envelope is None else envelope.id,
        wideband=block,
        wideband_id=None if block is None or wideband is None else wideband.id,
        transcript_id=None if transcript is None else transcript.id,
    )


def _absent(name: str) -> Finding:
    """One derivative's absence, recorded as a measurement that has no value.

    Args:
        name: The measurement that could not be taken.

    Returns:
        The finding, carrying which derivative was missing.
    """
    return measured(RATE, None, None, None, unavailable=name)


def _evidence(*ids: str | None) -> tuple[str, ...]:
    """The entity ids that are actually in the store, in the order given.

    Args:
        *ids: Candidate ids, any of which may be None when its derivative is absent.

    Returns:
        The ones that are not None.
    """
    return tuple(entity_id for entity_id in ids if entity_id is not None)


# --------------------------------------------------------------------- the instruments


def ddk_carrier(store: ProvStore, params: BranchParams, envelope: EnvelopeTrack) -> tuple[Entity | None, float | None]:
    """D1. The longest amplitude span that holds a readable repetition rate, and that rate.

    Envelope-first: the carrier is proposed from amplitude and only then qualified by modulation, so
    a train whose repetition is irregular is still a train reported with weak structure rather than
    an absence of data.

    Args:
        store: The provenance store.
        params: The operating points.
        envelope: The energy envelope.

    Returns:
        ``(span, rate_hz)``, or ``(None, None)`` when no carrier clears the train minimum with a
        modulation peak that stands over its own band.
    """
    minimum_s = params.point("train_min_s")
    best: Entity | None = None
    best_rate: float | None = None
    for span in amplitude_spans(live_entities(store, "span")):
        if span.extent is None or minimum_s is None or duration(span.extent) < minimum_s:
            continue
        rate = train_rate_hz(envelope, span.extent, params)
        if rate is None:
            continue
        if best is None or duration(span.extent) > duration(best.extent):
            best, best_rate = span, rate
    return best, best_rate


def ddk_places(
    onsets: Sequence[tuple[float, float]], params: BranchParams, wideband: SpectrogramBlock | None
) -> list[str]:
    """D6. Each syllable's place of articulation, from the burst spectrum at its onset.

    Args:
        onsets: The syllable extents, in order.
        params: The operating points.
        wideband: The wideband spectrogram, or None when it is absent.

    Returns:
        One place per onset, or :data:`UNRESOLVED` where the leading band did not beat the next by
        the declared margin — and for every onset when the spectrogram is absent, which reads the
        keys not at all, or when one of the three keys is unmeasured.
    """
    if wideband is None:
        return [UNRESOLVED] * len(onsets)
    places: list[str] = []
    burst_ms = params.point("burst_window_ms")
    bands = params.point("place_centroid_bands_hz")
    margin = params.point("place_margin_db")
    if burst_ms is None or bands is None or margin is None:
        # The same reading an absent spectrogram gets: the place could not be resolved. A branch
        # does not refuse over an unmeasured point, and the ask is recorded in `params.missing`.
        return [UNRESOLVED] * len(onsets)
    window_s = burst_ms / 1000.0
    for start, _ in onsets:
        burst = (start, start + window_s)
        energies = {place: band_power(wideband, burst, lo, hi) for place, (lo, hi) in bands.items()}
        finite = {place: value for place, value in energies.items() if np.isfinite(value) and value > 0.0}
        if len(finite) < 2:
            places.append(UNRESOLVED)
            continue
        ranked = sorted(finite.items(), key=lambda item: item[1], reverse=True)
        margin_db = 10.0 * float(np.log10(ranked[0][1] / ranked[1][1]))
        places.append(ranked[0][0] if margin_db >= margin else UNRESOLVED)
    return places


def intervals_of(onsets: Sequence[tuple[float, float]]) -> list[float]:
    """The inter-onset intervals of a train, in seconds.

    Args:
        onsets: The syllable extents, in order.

    Returns:
        One interval per consecutive pair.
    """
    starts = [start for start, _ in onsets]
    return [float(second - first) for first, second in zip(starts, starts[1:])]


def dispersion(intervals: Sequence[float]) -> float | None:
    """D3. The coefficient of variation of an interval sequence.

    Args:
        intervals: The inter-onset intervals.

    Returns:
        The sample standard deviation over the mean, or None when the sequence is shorter than the
        two values the sample deviation is defined over, or its mean is not positive.
    """
    values = np.asarray(intervals, dtype=float)
    if values.size < 2:
        return None
    mean = float(values.mean())
    if mean <= 0.0:
        return None
    return float(np.std(values, ddof=1) / mean)


def trend(intervals: Sequence[float]) -> float | None:
    """D3. How the interval changes across the train: seconds per syllable step.

    Negative is festination and positive is slowing; no threshold names either, because naming one
    would be an operating point nobody has measured.

    Args:
        intervals: The inter-onset intervals.

    Returns:
        The least-squares slope, or None when the sequence is shorter than the two points a slope is
        defined over.
    """
    values = np.asarray(intervals, dtype=float)
    if values.size < 2:
        return None
    return float(np.polyfit(np.arange(values.size, dtype=float), values, 1)[0])


def dispersion_by_position(intervals: Sequence[float], cycle: int) -> dict[str, float | None]:
    """D3. One dispersion per position in the cycle, for a sequential train.

    In ``/pa-ta-ka/`` the envelope onset lands differently relative to the release for each place,
    so the within-cycle intervals are unequal by measurement convention and a pooled dispersion has
    a floor set by syllable identity rather than by motor control.

    Args:
        intervals: The inter-onset intervals.
        cycle: How many syllables one cycle holds.

    Returns:
        ``{position: dispersion}``, empty when the cycle is not positive.
    """
    if cycle <= 0:
        return {}
    return {str(position): dispersion(intervals[position::cycle]) for position in range(cycle)}


# --------------------------------------------------------------------- the two modes


def align_ddk(
    task_family: str,
    store: ProvStore,
    hint: AudioHints | None,
    params: BranchParams,
    *,
    reads: DdkReads = DdkReads(),
) -> Result:
    """Evaluate one declared ``SYLLABLE_REPETITION`` task against what its instruction asked for.

    Spans proposed: **one**, the train, as ``task_extent``. An individual syllable is not a span —
    rate, interval dispersion and sequence collapse are statistics over the onset series, and one
    span per syllable would add roughly thirty spans per recording carrying no measurement of their
    own. The onsets travel as a ``counts`` entry, and a syllable that is not the one the sequence
    expected travels as a ``syllable_sequence_mismatch`` deviation with its own extent.

    Args:
        task_family: The declared family, which is a key of :data:`DDK_EXPECTATIONS`.
        store: The provenance store.
        hint: Accepted for the shared mode shape; not read. The declaration selects the mode and
            never supplies the answer, and this branch's own evidence is the recording.
        params: The operating points.
        reads: The derivatives, loaded by :func:`ddk`.

    Returns:
        Whether the expected patterns were found, the one train span, and the findings.

    Raises:
        KeyError: If ``task_family`` is not a DDK family, which is the caller owing
            :func:`detect_ddk` instead.
    """
    expectation = DDK_EXPECTATIONS[task_family]
    if expectation.pattern is Pattern.ORDERED_TOKENS:
        return _repeated_word(expectation, store, params, reads)
    declared = declared_duration_count(store, expectation.declared_duration_s)
    if reads.envelope is None:
        return Result(UNDETERMINED, [], [_absent(ENVELOPE), *declared])

    train, rate_hz = ddk_carrier(store, params, reads.envelope)
    if train is None or train.extent is None:
        return Result(False, [], [count("expected_event_count", 0, expectation.expected_event_count), *declared])

    onsets = events_in_span(reads.envelope, train, params)
    extent = hull(onsets) or train.extent
    starts = [round(start, 3) for start, _ in onsets]
    intervals = intervals_of(onsets)
    span_s = duration(extent)
    recording_s = duration(stream_extent(store))
    sequence = expectation.sequence if expectation.pattern is Pattern.SYLLABLE_SEQUENCE else None
    unit = CYCLES_OR_SYLLABLES_PER_S if sequence else SYLLABLES_PER_S

    findings: list[Finding] = [
        count("expected_event_count", len(onsets), expectation.expected_event_count),
        measured(
            RATE,
            extent[0],
            extent[1],
            rate_hz,
            unit=unit,
            onset_rate_hz=None if span_s <= 0.0 else round(len(onsets) / span_s, 3),
            support_syllables=len(onsets),
            **acquisition_covariates(store, extent),
        ),
        count("inter_onset_interval_s", [round(value, 3) for value in intervals], None),
        count("syllable_onset_s", starts, expectation.expected_event_count),
        measured(
            "interval_dispersion",
            extent[0],
            extent[1],
            dispersion(intervals),
            support_intervals=len(intervals),
            trend_s_per_step=trend(intervals),
            by_position={} if sequence is None else dispersion_by_position(intervals, len(sequence)),
        ),
        measured(
            "train_fraction_of_recording",
            extent[0],
            extent[1],
            None if recording_s <= 0.0 else round(span_s / recording_s, 3),
            train_s=round(span_s, 3),
            recording_s=round(recording_s, 3),
        ),
    ]

    attributes: dict[str, Any] = {"syllables_n": len(onsets), "production": "syllable_train"}
    done: Done = rate_hz is not None and len(onsets) > 0
    if sequence is not None:
        places = ddk_places(onsets, params, reads.wideband)
        if reads.wideband is None:
            findings.append(measured("syllable_place", extent[0], extent[1], None, unavailable=WIDEBAND))
        for index, (onset, place) in enumerate(zip(onsets, places)):
            target = sequence[index % len(sequence)]
            if place not in (target, UNRESOLVED):
                findings.append(
                    deviation("syllable_sequence_mismatch", onset[0], onset[1], expected=target, measured=place)
                )
        resolved = [place for place in places if place != UNRESOLVED]
        cycles = sum(
            1
            for index in range(len(resolved) - len(sequence) + 1)
            if tuple(resolved[index : index + len(sequence)]) == tuple(sequence)
        )
        if resolved:
            dominant = max(set(resolved), key=resolved.count)
            findings.append(
                measured(
                    "sequence_collapse_fraction",
                    extent[0],
                    extent[1],
                    round(resolved.count(dominant) / len(resolved), 3),
                    dominant_place=dominant,
                    support_syllables=len(resolved),
                )
            )
        findings.append(count("realised_cycles", cycles, None))
        attributes.update(production="syllable_sequence", realised_cycles=cycles, resolved_n=len(resolved))
        done = bool(onsets) and cycles >= 1

    components = [
        ddk_span(
            "task_extent",
            extent,
            *_evidence(train.id, reads.envelope_id, reads.wideband_id),
            **attributes,
        )
    ]
    recording_extent = stream_extent(store)
    if recording_extent is not None and touches_edge(extent, recording_extent):
        findings.append(deviation("truncation", extent[0], extent[1]))
    findings.extend(declared)
    return Result(done, components, findings)


def _repeated_word(expectation: Expectation, store: ProvStore, params: BranchParams, reads: DdkReads) -> Result:
    """The two ``buttercup`` families: the only DDK tasks whose instruction names a real word.

    Spans proposed: **one**, the train, over the hull of the realised tokens. The consensus words
    serve this family directly, so it is the cheapest in the branch and the one family where
    ``ddk.lexical_repetition`` fires for the right reason rather than on function-word repetition.

    Args:
        expectation: The row, whose first token is the word the instruction names.
        store: The provenance store.
        params: The operating points.
        reads: The derivatives, loaded by :func:`ddk`.

    Returns:
        Whether the word was realised at all, the one train span, and the findings.

    Raises:
        ValueError: If the row names no token, which no DDK ``ORDERED_TOKENS`` row does.
    """
    if not expectation.tokens:
        raise ValueError("a DDK ordered-tokens row names the word its instruction asks for")
    target = params.p_normalise(expectation.tokens[0])
    hits = [word for word in lexical(consensus_words(store)) if params.p_normalise(word_text(word)) == target]
    findings: list[Finding] = [count("expected_event_count", len(hits), expectation.expected_event_count)]
    train = hull([word_extent(word) for word in hits])
    components: list[Proposal] = []
    if train is not None and train[1] > train[0]:
        components.append(
            ddk_span(
                "task_extent",
                train,
                *_evidence(reads.transcript_id, reads.envelope_id, *(word.id for word in hits)),
                production="lexical_repetition",
                token=target,
                repeats_n=len(hits),
            )
        )
        if reads.envelope is None:
            findings.append(_absent(ENVELOPE))
        else:
            findings.append(
                measured(
                    RATE,
                    train[0],
                    train[1],
                    train_rate_hz(reads.envelope, train, params),
                    unit=SYLLABLES_PER_S,
                    support_words=len(hits),
                    **acquisition_covariates(store, train),
                )
            )
    findings.extend(declared_duration_count(store, expectation.declared_duration_s))
    return Result(len(hits) > 0, components, findings)


def detect_ddk(store: ProvStore, params: BranchParams, *, reads: DdkReads = DdkReads()) -> Result:
    """Find rapid repetition wherever it occurs, on a task of any other kind, and evaluate nothing.

    This is most of the branch's corpus: DDK routes far more connected speech than DDK material, so
    the mode has to report what it found without asserting that a Harvard sentence failed to be a
    DDK task. Repetition occurs in ordinary speech — a stutter, a false start, a repeated word — and
    the two kinds are proposed under different roles: ``repetition`` is an acoustic modulation over
    an amplitude carrier, ``lexical_repetition`` is one token the recognisers placed many times.
    Neither is a train the branch evaluated.

    Args:
        store: The provenance store.
        params: The operating points.
        reads: The derivatives, loaded by :func:`ddk`.

    Returns:
        A result whose ``done`` is :data:`UNDETERMINED`, one span per repetition found, and the
        findings.
    """
    components: list[Proposal] = []
    findings: list[Finding] = []
    minimum_s = params.point("train_min_s")
    minimum_occurrences = params.point("repeat_min_occurrences")
    if reads.envelope is None:
        findings.append(_absent(ENVELOPE))
    else:
        for span in amplitude_spans(live_entities(store, "span")):
            if span.extent is None or minimum_s is None or duration(span.extent) < minimum_s:
                continue
            rate_hz = train_rate_hz(reads.envelope, span.extent, params)
            if rate_hz is None:
                continue
            components.append(
                ddk_span(
                    "repetition",
                    span.extent,
                    *_evidence(span.id, reads.envelope_id),
                    production="acoustic_repetition",
                    rate_hz=rate_hz,
                    evaluates_no_task=True,
                )
            )
            findings.append(
                measured(
                    RATE,
                    span.extent[0],
                    span.extent[1],
                    rate_hz,
                    unit=CYCLES_OR_SYLLABLES_PER_S,
                    reading="acoustic_repetition_not_a_declared_ddk_task",
                    **acquisition_covariates(store, span.extent),
                )
            )

    occurrences: dict[str, list[Entity]] = {}
    for word in lexical(consensus_words(store)):
        occurrences.setdefault(params.p_normalise(word_text(word)), []).append(word)
    for token, words in occurrences.items():
        if minimum_occurrences is None or len(words) < minimum_occurrences:
            continue
        extent = hull([word_extent(word) for word in words])
        if extent is None or not extent[1] > extent[0]:
            continue
        components.append(
            ddk_span(
                "lexical_repetition",
                extent,
                *_evidence(reads.transcript_id, *(word.id for word in words)),
                production="lexical_repetition",
                token=token,
                repeats_n=len(words),
                evaluates_no_task=True,
            )
        )
        findings.append(measured("transcript_repeat", extent[0], extent[1], len(words), token=token))
    return Result(UNDETERMINED, components, findings)


# --------------------------------------------------------------------- the node


def _value(findings: Sequence[Finding], name: str) -> Any:  # noqa: ANN401
    """The value of the first measurement of one name.

    Args:
        findings: The branch's findings.
        name: The measurement's name.

    Returns:
        Its value, or None when no measurement of that name was taken.
    """
    for finding in findings:
        if finding.kind == "measure" and finding.name == name:
            return finding.evidence.get("value")
    return None


def _covariate(findings: Sequence[Finding], name: str, key: str) -> Any:  # noqa: ANN401
    """One covariate of the first measurement of one name.

    Args:
        findings: The branch's findings.
        name: The measurement's name.
        key: The covariate's key.

    Returns:
        The covariate, or None.
    """
    for finding in findings:
        if finding.kind == "measure" and finding.name == name:
            return finding.evidence.get(key)
    return None


def _detail(result: Result, mode: str, task_family: str | None, notes: Sequence[str]) -> dict[str, Any]:
    """The report's own observation fields, read back off what the mode returned.

    Args:
        result: What the mode returned.
        mode: ``"align"`` or ``"detect"``.
        task_family: The declared family, or None when nothing carried one.
        notes: What this branch could not measure, in controlled vocabulary. Nothing folds it.

    Returns:
        The detail mapping, carrying rates and regularity as measurements and no normative reading
        of either.
    """
    trains = [component for component in result.components if component.role in TRAIN_ROLES]
    return {
        "mode": mode,
        "task_family": task_family,
        "trains_n": len(trains),
        "train_s": round(sum(component.end - component.start for component in trains), 3),
        "train_fraction": _value(result.deviations, "train_fraction_of_recording"),
        "modulation_peak_hz": _value(result.deviations, RATE),
        "modulation_unit": _covariate(result.deviations, RATE, "unit"),
        "interval_dispersion": _value(result.deviations, "interval_dispersion"),
        "interval_trend_s_per_step": _covariate(result.deviations, "interval_dispersion", "trend_s_per_step"),
        "lexical_repetitions_n": sum(1 for component in result.components if component.role == "lexical_repetition"),
        "spans_n": len(result.components),
        "notes": list(notes),
    }


def ddk(
    store: ProvStore,
    source: str,
    config: TriageConfig,
    hint: AudioHints | None = None,
    *,
    run_dir: Path,
) -> BranchResult:
    """Propose the repetition train, measure its rate and regularity, and conclude.

    The declared task family selects the mode and never supplies the answer: a DDK family takes
    :func:`align_ddk`, anything else — undeclared, unreadable, or another branch's kind — takes
    :func:`detect_ddk`, which evaluates no task.

    Args:
        store: The provenance store, holding PREPROCESS's spans, words and derivatives.
        source: The store-held stream the derivatives were taken over, ``"plain"``.
        config: The triage configuration.
        hint: What the recording was declared to contain.
        run_dir: The run directory the derivative sidecars are relative to.

    Returns:
        The verdict, the view over the spans and findings written, and the verdict's entity id.

    Raises:
        ValueError: If an operating point this recording's mode needs is null in the configuration,
            naming the key — a branch does not default a boundary nobody measured.
    """
    software = software_agent(store)
    reads = read_ddk(store, run_dir, source)
    params = branch_params(config)
    mode, task_family = mode_of(NODE, store, hint)

    activity = store.activity(
        node=NODE, step="branch", parameters={"mode": mode, "task_family": task_family, "signal": source}
    )
    store.was_associated_with(activity, software)
    for used in _evidence(reads.envelope_id, reads.wideband_id, reads.transcript_id):
        store.used(activity, used)

    def _align(task_family: str, store: ProvStore, hint: AudioHints | None, params: BranchParams) -> Result:
        """Bind the loaded derivatives to the in-family mode."""
        return align_ddk(task_family, store, hint, params, reads=reads)

    def _detect(store: ProvStore, params: BranchParams) -> Result:
        """Bind the loaded derivatives to the out-of-family mode."""
        return detect_ddk(store, params, reads=reads)

    result = dispatch(NODE, store, params, hint, align=_align, detect=_detect)
    findings = [*result.deviations, *params.record()]
    span_ids = propose_spans(store, activity, software, result.components)
    finding_ids = write_findings(store, activity, software, findings, signal=source)

    notes: list[str] = []
    if reads.envelope is None:
        notes.append(NO_INSTRUMENT)
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
        detail=_detail(result, mode, task_family, notes),
    )
    return BranchResult(report=report, view=(*span_ids, *finding_ids, report_id), report_entity_id=report_id)
