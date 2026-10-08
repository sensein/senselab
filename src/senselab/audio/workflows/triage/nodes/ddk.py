"""The syllable-train task: its events, their identity, their rate and regularity, for the ten tasks that ask for one.

An instrument, not a branch: DDK was dissolved into SPEECH, so ``vocabulary.BRANCHES`` is
``("AIRWAY", "SPEECH", "VOICE")`` and SPEECH's ``align_speech`` serves the ten
``SYLLABLE_REPETITION`` families through :func:`align_ddk`. The task events are the task layer's
(:mod:`~senselab.audio.workflows.triage.ddk_task`): syllables and cycles over BACKGROUND's activity
regions, each scored against the declared phoneme template on the phonetic posteriorgram. The energy
envelope's modulation peak is read beside them. Every measurement below is a SPEECH measurement.

See ``specs/20261007-task-events-in-background/design.md`` ("DDK task layer").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from senselab.audio.workflows.triage.background_model import BACKGROUND_MODEL
from senselab.audio.workflows.triage.ddk_task import (
    CYCLE,
    DDK_READING,
    DdkReading,
    PositionMasses,
    Voicing,
    ddk_parameters,
    ddk_reading_of,
)
from senselab.audio.workflows.triage.nodes.branches import (
    BranchParams,
    EnvelopeTrack,
    Expectation,
    Finding,
    Pattern,
    Proposal,
    Result,
    acquisition_covariates,
    amplitude_spans,
    count_against_instruction,
    declared_duration_count,
    derivative_arrays,
    deviation,
    duration,
    measured,
    overlaps,
    read_envelope_track,
    speech_span,
    stream_extent,
    stream_ids,
    touches_edge,
    train_rate_hz,
)
from senselab.audio.workflows.triage.nodes.common import (
    find_measurement,
    lexical_words,
    live_entities,
    word_hull,
)
from senselab.audio.workflows.triage.task_events import GenericView, generic_view_of
from senselab.utils.prov_store import Entity, ProvStore

RATE = "ddk_syllable_rate_from_envelope_modulation_hz"
"""The envelope modulation channel's rate, named for the instrument that took it."""

ENVELOPE = "energy_envelope"
PPG = "ppg_posteriorgram"
PHONATION_TRACKS = "phonation_tracks"

EVENTS_FOUND = "ddk_events_found"
"""How many task events (cycles or syllables) the task layer read; an annotation, never a gate."""

IDENTITY = "ddk_identity"
"""The task events' median identity against the declared template."""

SYLLABLE_RATE = "ddk_syllable_rate_hz"
"""Syllables per second over the syllables' onset intervals inside each run."""

CYCLE_RATE = "ddk_cycle_rate_hz"
"""Cycles per second over the cycle events' onset intervals; sequence families only."""

PERIOD_CV = "ddk_period_cv"
"""The coefficient of variation of the task events' onset intervals."""

POSITION_MASS = "ddk_position_realised_mass"
"""How much of the posterior the expected class held where the alignment charged each position, one
value per position, averaged over the task events."""

INSTRUMENT_READING = "ddk_cv_instrument_reading"
"""The posteriorgram's reading over the task extent: the per-position realised mass of each task event,
written beside the consensus transcript and never in place of it."""

TRANSCRIPT_CLAIM = "lexical_transcript"
"""What a contradicted consensus word claims, and what the contest is against."""

CONTRADICTED = "instrument_contradicted"
"""Why that claim is contested: the instrument covers the word's extent."""

CV_AUTHORITY = "cv_instrument"
"""Which instrument is authoritative over the contested extent, on a declared syllable task."""

SYLLABLES_PER_S = "syllables_per_s"
CYCLES_PER_S = "cycles_per_s"
CYCLES_OR_SYLLABLES_PER_S = "cycles_or_syllables_per_s"
"""The unit of an envelope peak that may be the cycle rate or the syllable rate."""

TASK_EXTENT = "task_extent"
"""The role that says where the declared task was performed. Exactly one may survive a recording."""

TASK_FROM_EVENTS = "syllable_task_from_events"
"""The extent is the task events' span, first task event's start to last one's end."""

NO_ENVELOPE = "the energy envelope is absent; the syllable train's modulation rate could not be read"
NO_PPG = "the phonetic posteriorgram is absent; the task events' identity could not be read"
NO_BACKGROUND = "the background view is absent; the task events could not be read"


# ---------------------------------------------------------------- what the instruments read


OTHER_CLASS = "other"
"""The part of the partition holding every phoneme no template position names, silence included."""

SILENCE = "<silent>"
"""The posteriorgram's silence label: what the identity alignment's fillers explain."""


@dataclass(frozen=True)
class Posteriorgram:
    """The stored phonetic posteriorgram.

    Attributes:
        frames: The posteriorgram, ``(frame, phoneme)``.
        phonemes: The phoneme axis's labels, in the array's own order.
        seconds_per_frame: The frame period, in seconds.
        dtype: The dtype the sidecar recorded the array under.
    """

    frames: np.ndarray
    phonemes: tuple[str, ...]
    seconds_per_frame: float
    dtype: str = "float16"

    @property
    def emission_floor(self) -> float:
        """The smallest value the recorded data distinguishes from zero."""
        try:
            return float(np.finfo(np.dtype(self.dtype)).smallest_subnormal)
        except TypeError:
            return float(np.finfo(np.float16).smallest_subnormal)


@dataclass(frozen=True)
class DdkReads:
    """The stored derivatives the syllable task reads, loaded by SPEECH.

    The bodies carry no run directory, so the loaders run in
    :func:`~senselab.audio.workflows.triage.nodes.speech.speech` and their results arrive here.

    Attributes:
        envelope: The energy envelope and its global floor, or None when the derivative is absent.
        envelope_id: That measurement's entity id.
        ppg: The phonetic posteriorgram, or None when the derivative is absent.
        ppg_id: That measurement's entity id.
        view: BACKGROUND's view of the recording, or None where it wrote none.
        view_id: BACKGROUND's measurement id.
        voicing: The pitch track's voicing, or None where the tracks are absent.
    """

    envelope: EnvelopeTrack | None = None
    envelope_id: str | None = None
    ppg: Posteriorgram | None = None
    ppg_id: str | None = None
    view: GenericView | None = None
    view_id: str | None = None
    voicing: Voicing | None = None


def read_voicing(store: ProvStore, run_dir: Path) -> Voicing | None:
    """The pitch track's voiced frames.

    Args:
        store: The provenance store.
        run_dir: The run directory sidecar paths are relative to.

    Returns:
        A frame is voiced where its f0 is positive and its strength reaches ``ddk.voicing_strength_min``;
        None where the tracks are absent.
    """
    arrays = derivative_arrays(store, run_dir, PHONATION_TRACKS)
    if arrays is None or "times_s" not in arrays or "f0_hz" not in arrays or "strength" not in arrays:
        return None
    floor = float(ddk_parameters()["voicing_strength_min"])
    voiced = (np.nan_to_num(arrays["f0_hz"]) > 0) & (np.nan_to_num(arrays["strength"]) >= floor)
    return Voicing(np.asarray(arrays["times_s"], dtype=float), np.asarray(voiced, dtype=bool))


def read_ddk(store: ProvStore, run_dir: Path, source: str) -> DdkReads:
    """Load every derivative the syllable task reads, each independently absent.

    Args:
        store: The provenance store.
        run_dir: The run directory sidecar paths are relative to.
        source: The stream the derivatives were taken over.

    Returns:
        The reads; a derivative absent from the store, or whose sidecar is gone, is None.
    """
    del source
    envelope = find_measurement(store, ENVELOPE)
    posteriorgram = find_measurement(store, PPG)
    background = find_measurement(store, BACKGROUND_MODEL)
    frames = read_posteriorgram(store, run_dir)
    view = generic_view_of(store, run_dir)
    return DdkReads(
        envelope=read_envelope_track(store, run_dir),
        envelope_id=None if envelope is None else envelope.id,
        ppg=frames,
        ppg_id=None if frames is None or posteriorgram is None else posteriorgram.id,
        view=view,
        view_id=None if view is None or background is None else background.id,
        voicing=read_voicing(store, run_dir),
    )


def _absent(name: str, measurement: str = RATE) -> Finding:
    """One derivative's absence, recorded as a measurement that has no value.

    Args:
        name: The derivative that was missing.
        measurement: The measurement that could not be taken without it.

    Returns:
        The finding, carrying which derivative was missing.
    """
    return measured(measurement, None, None, None, unavailable=name)


def _evidence(*ids: str | None) -> tuple[str, ...]:
    """The entity ids that are actually in the store, in the order given.

    Args:
        *ids: Candidate ids, any of which may be None when its derivative is absent.

    Returns:
        The ones that are not None.
    """
    return tuple(entity_id for entity_id in ids if entity_id is not None)


# --------------------------------------------------------------------- the envelope instrument


def ddk_carrier(store: ProvStore, params: BranchParams, envelope: EnvelopeTrack) -> tuple[Entity | None, float | None]:
    """D1. The longest amplitude span that holds a readable repetition rate, and that rate.

    Args:
        store: The provenance store.
        params: The operating points.
        envelope: The energy envelope.

    Returns:
        ``(span, rate_hz)``, or ``(None, None)`` when no carrier clears the train minimum with a
        modulation peak that stands over its own band.
    """
    minimum_s = params.gate("train_min_s")
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


# --------------------------------------------------------------------- the posteriorgram's classes


def read_posteriorgram(store: ProvStore, run_dir: Path) -> Posteriorgram | None:
    """The stored posteriorgram, its phoneme axis, its frame period and its recorded dtype.

    Args:
        store: The provenance store.
        run_dir: The run directory sidecar paths are relative to.

    Returns:
        The posteriorgram, or None when the derivative, its sidecar, its phoneme axis or a positive
        frame period is absent.
    """
    arrays = derivative_arrays(store, run_dir, PPG)
    measurement = find_measurement(store, PPG)
    if arrays is None or measurement is None or "posteriorgram" not in arrays or "phonemes" not in arrays:
        return None
    frames = np.asarray(arrays["posteriorgram"], dtype=float)
    labels = tuple(str(label) for label in np.asarray(arrays["phonemes"]).reshape(-1))
    period = _frame_period(arrays.get("seconds_per_frame"), measurement.attributes.get("seconds_per_frame"))
    if period is None or frames.ndim != 2 or frames.shape[0] == 0 or frames.shape[1] != len(labels):
        return None
    return Posteriorgram(
        frames=frames,
        phonemes=labels,
        seconds_per_frame=period,
        dtype=str(measurement.attributes.get("dtype") or "float16"),
    )


def _frame_period(stored: Any, declared: Any) -> float | None:  # noqa: ANN401 — two carriers of one scalar
    """The frame period from whichever carrier holds a positive one.

    Args:
        stored: The sidecar's own ``seconds_per_frame`` array, if it carries one.
        declared: The measurement entity's ``seconds_per_frame`` attribute.

    Returns:
        The period in seconds, or None when neither carrier holds a finite positive value.
    """
    for candidate in (stored, declared):
        if candidate is None:
            continue
        values = np.asarray(candidate, dtype=float).reshape(-1)
        if values.size and np.isfinite(values[0]) and values[0] > 0.0:
            return float(values[0])
    return None


def phoneme_classes(*mappings: dict[str, tuple[str, ...]]) -> dict[str, str]:
    """The class vocabularies as one phoneme-to-class lookup.

    Args:
        *mappings: ``branch.phoneme_place_classes`` and ``branch.phoneme_vowel_classes``.

    Returns:
        Each phoneme mapped to its class; a phoneme two classes name takes the first declared.
    """
    lookup: dict[str, str] = {}
    for mapping in mappings:
        for name, phonemes in mapping.items():
            for phoneme in phonemes:
                lookup.setdefault(phoneme, name)
    return lookup


def class_masses(ppg: Posteriorgram, classes: Sequence[str], lookup: dict[str, str]) -> np.ndarray:
    """The posterior mass each named class holds per frame, and what is left over.

    Args:
        ppg: The posteriorgram.
        classes: The distinct class names the template's positions belong to, in a fixed order.
        lookup: Each phoneme's class, from :func:`phoneme_classes`.

    Returns:
        ``(frame, len(classes) + 1)``, the last column being the mass outside every named class.
    """
    named = list(classes)
    indicator = np.zeros((len(ppg.phonemes), len(named) + 1), dtype=float)
    index = {name: position for position, name in enumerate(named)}
    for column, phoneme in enumerate(ppg.phonemes):
        indicator[column, index.get(lookup.get(phoneme, OTHER_CLASS), len(named))] = 1.0
    return np.asarray(ppg.frames @ indicator, dtype=float)


VOCABULARY_KEYS = ("phoneme_place_classes", "phoneme_vowel_classes")
"""The class vocabularies the identity is read through; an unmeasured one leaves it unread."""


def position_masses(ppg: Posteriorgram, params: BranchParams, template: Sequence[str]) -> PositionMasses | None:
    """The mass each template position's class holds per frame.

    Args:
        ppg: The posteriorgram.
        params: The operating points, for the class vocabularies.
        template: The declared phoneme sequence.

    Returns:
        The masses; None where a class vocabulary is unmeasured (``params.missing`` names it).
    """
    places = params.point("phoneme_place_classes")
    vowels = params.point("phoneme_vowel_classes")
    if places is None or vowels is None:
        return None
    lookup = phoneme_classes(places, vowels)
    classes = tuple(lookup.get(phoneme, OTHER_CLASS) for phoneme in template)
    named = tuple(dict.fromkeys(name for name in classes if name != OTHER_CLASS))
    masses = class_masses(ppg, named, lookup)
    column = {name: index for index, name in enumerate(named)}
    per_position = np.stack([masses[:, column.get(name, masses.shape[1] - 1)] for name in classes], axis=1)
    return PositionMasses(
        template=tuple(template),
        classes=classes,
        masses=per_position,
        filler=ppg.frames[:, ppg.phonemes.index(SILENCE)] if SILENCE in ppg.phonemes else masses[:, -1],
        seconds_per_frame=ppg.seconds_per_frame,
        floor=ppg.emission_floor,
    )


def syllables_per_cycle(params: BranchParams, template: Sequence[str]) -> int:
    """How many syllables one repetition of the template holds: its vowel positions.

    Args:
        params: The operating points, for the vowel classes.
        template: The declared phoneme sequence.

    Returns:
        The count, at least one.
    """
    vowels = params.point("phoneme_vowel_classes") or {}
    nuclei = {phoneme for members in vowels.values() for phoneme in members}
    return max(1, sum(1 for phoneme in template if phoneme in nuclei))


# --------------------------------------------------------------------- what the reading has to say


def contradicted_words(store: ProvStore, extent: tuple[float, float]) -> list[Entity]:
    """The consensus words the instrument's covered extent contradicts, in index order.

    Args:
        store: The provenance store.
        extent: The extent the instrument covers.

    Returns:
        Every live lexical consensus word whose hull shares any interval with it; bracketed tokens
        are left alone.
    """
    return [word for word in lexical_words(store) if overlaps(word_hull(word), extent)]


def instrument_authority(store: ProvStore, reading: DdkReading, evidence: Sequence[str]) -> list[Finding]:
    """The task events' identity recorded as authoritative, and each word it contradicts.

    Additive only: no word is invalidated, no transcript is rewritten and no text is copied into a
    finding.

    Args:
        store: The provenance store, for the consensus words.
        reading: The task layer's reading.
        evidence: The entity ids it was read off.

    Returns:
        One :data:`INSTRUMENT_READING` measurement over the task extent and one ``contest`` per
        contradicted word, or nothing at all where no task event stands.
    """
    extent = reading.extent
    if extent is None or reading.evidence is None:
        return []
    words = contradicted_words(store, extent)
    by_event = {(e.start_s, e.end_s): i for e, i in zip(reading.evidence.found, reading.identities)}
    rows = [
        [None if m is None else round(m, 3) for m in by_event[(e.start_s, e.end_s)].realised]
        for e in reading.evidence.events
        if (e.start_s, e.end_s) in by_event
    ]
    findings: list[Finding] = [
        measured(
            INSTRUMENT_READING,
            extent[0],
            extent[1],
            rows,
            *evidence,
            *(word.id for word in words),
            authority=CV_AUTHORITY,
            supersedes=TRANSCRIPT_CLAIM,
            positions=list(reading.annotations.get("positions") or ()),
            classes=list(reading.annotations.get("classes") or ()),
            onsets_s=[round(e.start_s, 3) for e in reading.evidence.events],
            events_n=len(reading.evidence.events),
            contradicted_words_n=len(words),
            contradicted_word_ids=[word.id for word in words],
        )
    ]
    findings.extend(
        Finding(
            "contest",
            TRANSCRIPT_CLAIM,
            *word_hull(word),
            {"reason": CONTRADICTED, "authority": CV_AUTHORITY, "index": int(word.attributes["index"])},
            (word.id, *evidence),
        )
        for word in words
    )
    return findings


def reading_findings(
    store: ProvStore, reading: DdkReading, evidence: Sequence[str], expectation: Expectation
) -> list[Finding]:
    """The task layer's reading as SPEECH measurements: the reading itself and its annotations.

    Args:
        store: The provenance store, for the consensus words the identity contests.
        reading: The task layer's reading.
        evidence: The entity ids it was read off.
        expectation: The family's row, for the instructed count.

    Returns:
        The :data:`DDK_READING` measurement, and where the reading was taken the events found, the
        identity, the rates, the period's dispersion, the per-position mass, the count against the
        instruction and the instrument's reading over the extent. None of them decides but the first.
    """
    record = reading.record()
    findings: list[Finding] = [measured(DDK_READING, None, None, record.get("decision"), *evidence, **record)]
    if reading.evidence is None:
        return findings
    notes = reading.annotations
    unit = CYCLES_PER_S if reading.unit == CYCLE else SYLLABLES_PER_S
    required = expectation.required_count
    findings.extend(
        [
            measured(
                EVENTS_FOUND,
                None,
                None,
                len(reading.evidence.events),
                *evidence,
                unit=reading.unit,
                required=None if required is None else required.value,
                syllables_n=len(reading.syllables),
            ),
            measured(IDENTITY, None, None, record.get("identity"), *evidence),
            measured(SYLLABLE_RATE, None, None, notes.get("syllable_rate_hz"), *evidence, unit=SYLLABLES_PER_S),
            measured(
                PERIOD_CV,
                None,
                None,
                notes.get("period_cv"),
                *evidence,
                unit=unit,
                trend_s_per_step=notes.get("period_trend_s_per_step"),
            ),
            measured(
                POSITION_MASS,
                None,
                None,
                list(notes.get("realised_mass") or ()),
                *evidence,
                positions=list(notes.get("positions") or ()),
                classes=list(notes.get("classes") or ()),
            ),
        ]
    )
    if reading.unit == CYCLE:
        findings.append(measured(CYCLE_RATE, None, None, notes.get("cycle_rate_hz"), *evidence, unit=CYCLES_PER_S))
    if required is not None:
        findings.append(count_against_instruction(required, len(reading.evidence.events), *evidence))
    findings.extend(instrument_authority(store, reading, evidence))
    return findings


def task_extent_span(reading: DdkReading, ids: Sequence[str]) -> Proposal | None:
    """The one ``task_extent`` this family leaves behind: the first task event to the last.

    Args:
        reading: The task layer's reading.
        ids: The entity ids it was read off.

    Returns:
        The span, or None where no task event stands.
    """
    extent = reading.extent
    if extent is None or reading.evidence is None or not ids:
        return None
    return speech_span(
        TASK_EXTENT,
        extent,
        *ids,
        production=TASK_FROM_EVENTS,
        events_n=len(reading.evidence.events),
        unit=reading.unit,
        syllables_n=len(reading.syllables),
    )


# ------------------------------------------------------------------ the declared-task body


def ddk_reading(
    expectation: Expectation, store: ProvStore, params: BranchParams, reads: DdkReads
) -> tuple[DdkReading, tuple[str, ...]]:
    """The task layer's reading of one declared syllable task, and the ids it was read off.

    Args:
        expectation: The family's row.
        store: The provenance store, for the recording's duration.
        params: The operating points, for the class vocabularies.
        reads: The derivatives.

    Returns:
        The reading and its evidence ids. A class vocabulary that is unmeasured leaves the reading
        untaken with that key named absent.
    """
    template = tuple(expectation.sequence or ())
    sequence = expectation.pattern is Pattern.SYLLABLE_SEQUENCE
    unit = CYCLE if sequence else "syllable"
    ids = _evidence(reads.view_id, reads.ppg_id)
    positions = None if reads.ppg is None else position_masses(reads.ppg, params, template)
    if reads.ppg is not None and positions is None:
        missing = tuple(f"branch.{key}" for key in VOCABULARY_KEYS if params.point(key) is None)
        return DdkReading(unit, None, absent=missing), ids
    required = expectation.required_count
    reading = ddk_reading_of(
        reads.view,
        positions,
        sequence=sequence,
        syllables_per_cycle=syllables_per_cycle(params, template),
        voicing=reads.voicing,
        required_count=None if required is None else required.value,
        recording_s=duration(stream_extent(store)),
    )
    return reading, ids


def align_ddk(
    expectation: Expectation,
    store: ProvStore,
    params: BranchParams,
    *,
    reads: DdkReads = DdkReads(),
) -> Result:
    """Evaluate one declared ``SYLLABLE_REPETITION`` task.

    Spans proposed: exactly one ``task_extent``, the task events' span (:func:`task_extent_span`), or
    none where no task event stands.

    Args:
        expectation: The row SPEECH holds for this family, whose pattern is ``SYLLABLE_TRAIN`` or
            ``SYLLABLE_SEQUENCE`` and whose ``sequence`` is its phoneme template.
        store: The provenance store.
        params: The operating points.
        reads: The derivatives, loaded by :func:`speech`.

    Returns:
        The task extent and the findings, the task layer's reading among them.
    """
    reading, ids = ddk_reading(expectation, store, params, reads)
    findings: list[Finding] = []
    if reads.ppg is None:
        findings.append(_absent(PPG, IDENTITY))
    sequence = expectation.pattern is Pattern.SYLLABLE_SEQUENCE
    notes = reading.annotations
    if reads.envelope is None:
        findings.append(_absent(ENVELOPE))
    else:
        train, rate_hz = ddk_carrier(store, params, reads.envelope)
        if train is not None and train.extent is not None:
            carrier_ids = _evidence(train.id, reads.envelope_id)
            findings.append(
                measured(
                    RATE,
                    train.extent[0],
                    train.extent[1],
                    rate_hz,
                    *carrier_ids,
                    unit=CYCLES_OR_SYLLABLES_PER_S if sequence else SYLLABLES_PER_S,
                    event_syllable_rate_hz=notes.get("syllable_rate_hz"),
                    event_cycle_rate_hz=notes.get("cycle_rate_hz"),
                    **acquisition_covariates(store, train.extent),
                )
            )

    span = task_extent_span(reading, ids)
    components = [] if span is None else [span]
    if span is not None:
        extent = (span.start, span.end)
        span_s = duration(extent)
        recording_s = duration(stream_extent(store))
        findings.append(
            measured(
                "train_fraction_of_recording",
                extent[0],
                extent[1],
                None if recording_s <= 0.0 else round(span_s / recording_s, 3),
                *span.derived_from,
                *stream_ids(store),
                train_s=round(span_s, 3),
                recording_s=round(recording_s, 3),
                production=span.attributes.get("production"),
            )
        )
        recording_extent = stream_extent(store)
        if recording_extent is not None and touches_edge(extent, recording_extent):
            findings.append(deviation("truncation", extent[0], extent[1], *span.derived_from, *stream_ids(store)))
    declared = declared_duration_count(store, expectation.declared_duration_s)
    return Result(components, [*findings, *reading_findings(store, reading, ids, expectation), *declared])


# --------------------------------------------------------------------- what SPEECH reports


def _value(findings: Sequence[Finding], name: str) -> Any:  # noqa: ANN401
    """The value of the first measurement of one name.

    Args:
        findings: The findings :func:`align_ddk` returned.
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
        findings: The findings :func:`align_ddk` returned.
        name: The measurement's name.
        key: The covariate's key.

    Returns:
        The covariate, or None.
    """
    for finding in findings:
        if finding.kind == "measure" and finding.name == name:
            return finding.evidence.get(key)
    return None


def syllable_detail(result: Result) -> dict[str, Any]:
    """SPEECH's report fields for a syllable-repetition task, read back off what the body returned.

    Args:
        result: What :func:`align_ddk` returned.

    Returns:
        The detail mapping, under the keys ``common.py``'s ``BRANCH_MEASURES["SPEECH"]`` names.
    """
    trains = [component for component in result.components if component.role == TASK_EXTENT]
    findings = result.deviations
    return {
        "trains_n": len(trains),
        "train_s": round(sum(component.end - component.start for component in trains), 3),
        "train_fraction": _value(findings, "train_fraction_of_recording"),
        "modulation_peak_hz": _value(findings, RATE),
        "modulation_unit": _covariate(findings, RATE, "unit"),
        "ddk_decision": _value(findings, DDK_READING),
        "ddk_why": _covariate(findings, DDK_READING, "why"),
        "ddk_unit": _covariate(findings, DDK_READING, "unit"),
        "ddk_events_n": _value(findings, EVENTS_FOUND),
        "ddk_required_count": _covariate(findings, EVENTS_FOUND, "required"),
        "ddk_syllables_n": _covariate(findings, EVENTS_FOUND, "syllables_n"),
        "ddk_identity": _value(findings, IDENTITY),
        "ddk_syllable_rate_hz": _value(findings, SYLLABLE_RATE),
        "ddk_cycle_rate_hz": _value(findings, CYCLE_RATE),
        "ddk_period_cv": _value(findings, PERIOD_CV),
        "ddk_period_trend_s_per_step": _covariate(findings, PERIOD_CV, "trend_s_per_step"),
        "ddk_positions": _covariate(findings, POSITION_MASS, "positions"),
        "ddk_realised_mass": _value(findings, POSITION_MASS),
        "ddk_contradicted_words_n": _covariate(findings, INSTRUMENT_READING, "contradicted_words_n"),
    }
