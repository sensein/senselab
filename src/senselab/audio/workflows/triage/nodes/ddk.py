"""The syllable-train instrument: its rate, its regularity, and the ten tasks that ask for one.

Not a branch. :func:`align_speech` serves the ten ``SYLLABLE_REPETITION`` families through
:func:`align_ddk`, which evaluates one of them against what its instruction asked for, and every
measurement below is a SPEECH measurement taken by this module's instruments.

The design is ``specs/20260817-triage-workflow-dag/branch-ddk.md`` (D1–D6); the ported bodies are
``expected-patterns.md``; what porting decided, and the two places this module departs from either,
is ``specs/20260817-triage-workflow-dag/branch-ddk-implementation.md``; why the branch became this
module is ``ddk-dissolved-into-speech.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from senselab.audio.workflows.triage.nodes.branches import (
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
    Syllable,
    acquisition_covariates,
    amplitude_spans,
    band_power,
    branch_params,
    count,
    declared_duration_count,
    derivative_arrays,
    deviation,
    duration,
    events_in_span,
    hull,
    measured,
    mode_of,
    propose_spans,
    read_envelope_track,
    read_spectrogram_block,
    speech_span,
    stream_extent,
    stream_ids,
    touches_edge,
    train_rate_hz,
)
from senselab.audio.workflows.triage.nodes.common import (
    find_measurement,
    live_entities,
)
from senselab.utils.prov_store import Entity, ProvStore

RATE = "ddk_syllable_rate_from_envelope_modulation_hz"
"""The one rate measurement, named for the instrument that took it."""

ENVELOPE = "energy_envelope"
WIDEBAND = "spectrogram_wideband"
PPG = "ppg_posteriorgram"

PPG_RATE = "ddk_syllable_rate_from_ppg_cv_onsets_hz"
"""The second rate measurement, named for the instrument that took it. Not a substitute for
:data:`RATE`: the two read different signals and are reported side by side."""

PPG_UNITS = "ddk_cv_unit_count"
PPG_DISPERSION = "ddk_ppg_interval_dispersion"
PPG_EXPECTED_PLACE = "ddk_expected_place_fraction"
PPG_EXPECTED_NUCLEUS = "ddk_expected_nucleus_fraction"
PPG_PLACE_AGREEMENT = "ddk_place_agreement_ppg_vs_burst"

UNRESOLVED = "unresolved"
"""The place a burst spectrum did not separate, or the nucleus class no declared class holds, which
is not a substitution."""

SYLLABLES_PER_S = "syllables_per_s"
CYCLES_OR_SYLLABLES_PER_S = "cycles_or_syllables_per_s"
"""A sequential train modulates at the cycle rate as well as the syllable rate; the peak is one of
the two and the harmonic-equality tolerance that would separate them is unmeasured, so the unit is
carried as ambiguous rather than resolved by assumption."""

TRAIN_ROLES = ("task_extent",)
"""The roles that are a train."""

NO_TRAIN = "no syllable train was found"
NO_ENVELOPE = "the energy envelope is absent; the syllable train's only rate instrument could not be read"
NO_PPG = "the phonetic posteriorgram is absent; the CV instrument could not be read"
PPG_PLACE_NOT_AUTHORITY = (
    "the posteriorgram's place per onset is a reported reading, not the place decision; nothing has "
    "measured its agreement with the burst spectrum on rapid nonsense syllables"
)


# --------------------------------------------------------------------- what the two modes read


@dataclass(frozen=True)
class DdkReads:
    """The stored derivatives the syllable body measures over, read once by SPEECH.

    The expectation bodies carry ``(store, params)`` and no run directory, and every sidecar path
    in the store is relative to one, so the loaders cannot run inside a body. They run in
    :func:`~senselab.audio.workflows.triage.nodes.speech.speech`, which has the run directory, and
    their results arrive here.

    Attributes:
        envelope: The energy envelope and its global floor, or None when the derivative is absent.
        envelope_id: That measurement's entity id, for the derivation.
        wideband: The wideband spectrogram, or None when it or the working rate is absent.
        wideband_id: That measurement's entity id.
        ppg: The phonetic posteriorgram, or None when the derivative is absent.
        ppg_id: That measurement's entity id, for the derivation of everything the CV walk reads.
    """

    envelope: EnvelopeTrack | None = None
    envelope_id: str | None = None
    wideband: SpectrogramBlock | None = None
    wideband_id: str | None = None
    ppg: Posteriorgram | None = None
    ppg_id: str | None = None


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
    posteriorgram = find_measurement(store, PPG)
    rate = working_rate(store, source)
    block = None if rate is None else read_spectrogram_block(store, run_dir, WIDEBAND, rate)
    frames = read_posteriorgram(store, run_dir)
    return DdkReads(
        envelope=read_envelope_track(store, run_dir),
        envelope_id=None if envelope is None else envelope.id,
        wideband=block,
        wideband_id=None if block is None or wideband is None else wideband.id,
        ppg=frames,
        ppg_id=None if frames is None or posteriorgram is None else posteriorgram.id,
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


# --------------------------------------------------------------------- the posteriorgram instrument


CONSONANT = "C"
VOWEL = "V"
OTHER = "O"
"""The three classes a frame's argmax phoneme falls into for the CV walk."""


@dataclass(frozen=True)
class Posteriorgram:
    """The stored phonetic posteriorgram, as the CV walk reads it.

    Attributes:
        frames: The posteriorgram, ``(frame, phoneme)``.
        phonemes: The phoneme axis's labels, in the array's own order.
        seconds_per_frame: The frame period, in seconds.
    """

    frames: np.ndarray
    phonemes: tuple[str, ...]
    seconds_per_frame: float


@dataclass(frozen=True)
class PhonemeRun:
    """One maximal stretch of frames sharing an argmax phoneme.

    Attributes:
        label: The phoneme.
        start_s: Where the run starts, in seconds.
        end_s: Where it ends, in seconds.
    """

    label: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class CvUnit:
    """One consonant run followed by a vowel run: the syllable the walk counts.

    Attributes:
        consonant: The stop the unit opens on.
        vowel: The nucleus it resolves to.
        start_s: The consonant run's start, which is the unit's onset.
        end_s: The vowel run's end.
    """

    consonant: str
    vowel: str
    start_s: float
    end_s: float

    @property
    def extent(self) -> tuple[float, float]:
        """The unit's extent, in the ``(start, end)`` form every other instrument here carries."""
        return (self.start_s, self.end_s)


@dataclass(frozen=True)
class SyllableTrain:
    """A contiguous stretch of CV onsets whose intervals stayed regular.

    Attributes:
        units: The units it spans, in order.
        repetitions: How many onsets it holds.
        period_s: The median inter-onset interval.
        rate_hz: One over that period.
        jitter: The population standard deviation of the intervals over their median.
    """

    units: tuple[CvUnit, ...]
    repetitions: int
    period_s: float
    rate_hz: float
    jitter: float

    @property
    def extent(self) -> tuple[float, float]:
        """The train's extent: its first onset to its last unit's end."""
        return (self.units[0].start_s, self.units[-1].end_s)


@dataclass(frozen=True)
class PpgReading:
    """What the CV walk read off one posteriorgram.

    Attributes:
        units: Every CV unit the walk found, in order.
        train: The train with the most repetitions, or None when none reached the minimum.
        readable: Whether the walk could run at all; False when a point it needs is unmeasured.
    """

    units: tuple[CvUnit, ...]
    train: SyllableTrain | None
    readable: bool


def read_posteriorgram(store: ProvStore, run_dir: Path) -> Posteriorgram | None:
    """The stored posteriorgram, its phoneme axis and its frame period.

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
    return Posteriorgram(frames=frames, phonemes=labels, seconds_per_frame=period)


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


def phoneme_runs(ppg: Posteriorgram) -> list[PhonemeRun]:
    """The argmax raster collapsed into maximal runs of one phoneme.

    Args:
        ppg: The posteriorgram.

    Returns:
        One run per stretch of consecutive frames sharing an argmax phoneme, in time order. Adjacent
        runs carry different labels by construction, which is why the CV walk needs no lookahead.
    """
    indices = np.asarray(np.argmax(ppg.frames, axis=1), dtype=int)
    if indices.size == 0:
        return []
    edges = [0, *(int(index) for index in np.flatnonzero(indices[1:] != indices[:-1]) + 1), int(indices.size)]
    period = ppg.seconds_per_frame
    return [
        PhonemeRun(label=ppg.phonemes[int(indices[first])], start_s=first * period, end_s=last * period)
        for first, last in zip(edges, edges[1:])
    ]


def stop_phonemes(stop_places: dict[str, tuple[str, ...]]) -> dict[str, str]:
    """The stop set as one phoneme-to-place lookup.

    Args:
        stop_places: ``branch.ddk_stop_places``: each place and the phonemes that are it.

    Returns:
        Each stop phoneme mapped to its place. The stop set the walk reads is this mapping's keys,
        so the set and the place vocabulary are one declaration and cannot drift apart.
    """
    return {phoneme: place for place, phonemes in stop_places.items() for phoneme in phonemes}


def nucleus_classes(classes: dict[str, tuple[str, ...]]) -> dict[str, str]:
    """The nucleus vocabulary as one phoneme-to-class lookup.

    Args:
        classes: ``branch.ddk_nucleus_classes``: each class and the phonemes that are it.

    Returns:
        Each nucleus phoneme mapped to its class. A phoneme two classes both name resolves to the
        first in declaration order.
    """
    lookup: dict[str, str] = {}
    for name, phonemes in classes.items():
        for phoneme in phonemes:
            lookup.setdefault(phoneme, name)
    return lookup


def admitted_nuclei(classes: dict[str, tuple[str, ...]], template: Sequence[Syllable] | None) -> tuple[str, ...]:
    """The nuclei the CV walk admits: the union of the classes the declared template names.

    Args:
        classes: ``branch.ddk_nucleus_classes``.
        template: The declared syllable template, or None when no row declares one.

    Returns:
        Every phoneme of every class the template names, in the vocabulary's declaration order.
        Every class when the template is None, which is the widest the vocabulary allows.
    """
    named = {position.nucleus for position in template} if template else set(classes)
    return tuple(phoneme for name, phonemes in classes.items() if name in named for phoneme in phonemes)


def phoneme_class(label: str, places: dict[str, str], nuclei: Sequence[str]) -> str:
    """Which of the three classes one phoneme falls into.

    Args:
        label: The phoneme.
        places: The stop-to-place lookup :func:`stop_phonemes` built.
        nuclei: The nuclei the declared template admits, from :func:`admitted_nuclei`.

    Returns:
        :data:`CONSONANT`, :data:`VOWEL` or :data:`OTHER`.
    """
    if label in places:
        return CONSONANT
    return VOWEL if label in nuclei else OTHER


def cv_units(runs: Sequence[PhonemeRun], places: dict[str, str], nuclei: Sequence[str]) -> list[CvUnit]:
    """The consonant-vowel syllables in a run sequence.

    Each consonant run is scanned forward: the first vowel run closes a unit whose onset is the
    consonant run's own start, and the first further consonant run ends the scan with nothing
    emitted. Runs alternate by construction, so the scan carries no window and no lookahead point.

    Args:
        runs: The phoneme runs, in time order.
        places: The stop-to-place lookup.
        nuclei: The nuclei the declared template admits, from :func:`admitted_nuclei`.

    Returns:
        The units, in onset order. Empty on material that holds no stops. A consonant with no
        following nucleus — a coda, such as the ``p`` of ``buttercup`` — closes no unit and is
        therefore counted as none.
    """
    classes = [phoneme_class(run.label, places, nuclei) for run in runs]
    units: list[CvUnit] = []
    for index, run in enumerate(runs):
        if classes[index] != CONSONANT:
            continue
        for ahead in range(index + 1, len(runs)):
            if classes[ahead] == VOWEL:
                units.append(
                    CvUnit(consonant=run.label, vowel=runs[ahead].label, start_s=run.start_s, end_s=runs[ahead].end_s)
                )
                break
            if classes[ahead] == CONSONANT:
                break
    return units


def syllable_trains(units: Sequence[CvUnit], tolerance: float, min_repetitions: int) -> list[SyllableTrain]:
    """Every maximal contiguous stretch of CV onsets whose intervals stayed regular.

    Args:
        units: The CV units, in onset order.
        tolerance: The factor an interval may differ from its stretch's running median by.
        min_repetitions: The onsets a stretch needs before it is reported as a train.

    Returns:
        The trains, in time order. The stretches do not overlap: the interval that ended one is
        where the next is tried from.
    """
    intervals = np.asarray(intervals_of([unit.extent for unit in units]), dtype=float)
    trains: list[SyllableTrain] = []
    first = 0
    while first < intervals.size:
        reach, last = first, first + 1
        while last <= intervals.size:
            segment = intervals[first:last]
            median = float(np.median(segment))
            if median <= 0.0 or float(segment.max()) > median * tolerance or float(segment.min()) < median / tolerance:
                break
            reach, last = last, last + 1
        if reach == first:
            first += 1
            continue
        segment = intervals[first:reach]
        median = float(np.median(segment))
        if reach - first + 1 >= min_repetitions:
            trains.append(
                SyllableTrain(
                    units=tuple(units[first : reach + 1]),
                    repetitions=reach - first + 1,
                    period_s=median,
                    rate_hz=1.0 / median,
                    jitter=float(np.std(segment) / median),
                )
            )
        first = reach
    return trains


def ppg_reading(
    ppg: Posteriorgram | None, params: BranchParams, template: Sequence[Syllable] | None = None
) -> PpgReading | None:
    """Run the CV walk and the train finder over one posteriorgram.

    Extraction is permissive and conformance is positional: the walk admits any nucleus in the union
    of the classes ``template`` names, and :func:`ppg_evidence` then checks each unit's place and
    nucleus class against the position the template gave it.

    Args:
        ppg: The posteriorgram, or None when the derivative is absent.
        params: The operating points.
        template: The declared syllable template, or None when no row declares one.

    Returns:
        The reading, or None when the derivative is absent, which is an absent instrument and not a
        negative reading. A reading whose ``readable`` is False is the same absence for a different
        reason: a segmentation point nobody has measured, which ``params.missing`` names.
    """
    if ppg is None:
        return None
    stop_places = params.point("ddk_stop_places")
    classes = params.point("ddk_nucleus_classes")
    tolerance = params.point("ddk_interval_tolerance")
    minimum = params.point("ddk_min_repetitions")
    if stop_places is None or classes is None or tolerance is None or minimum is None:
        return PpgReading(units=(), train=None, readable=False)
    units = cv_units(phoneme_runs(ppg), stop_phonemes(stop_places), admitted_nuclei(classes, template))
    trains = syllable_trains(units, tolerance, minimum)
    longest = max(trains, key=lambda train: train.repetitions, default=None)
    return PpgReading(units=tuple(units), train=longest, readable=True)


def unit_places(units: Sequence[CvUnit], stop_places: dict[str, tuple[str, ...]]) -> list[str]:
    """The place of articulation the posteriorgram read for each unit's consonant.

    Args:
        units: The CV units.
        stop_places: ``branch.ddk_stop_places``.

    Returns:
        One place per unit, :data:`UNRESOLVED` for a consonant the mapping does not name.
    """
    lookup = stop_phonemes(stop_places)
    return [lookup.get(unit.consonant, UNRESOLVED) for unit in units]


def unit_nuclei(units: Sequence[CvUnit], classes: dict[str, tuple[str, ...]]) -> list[str]:
    """The nucleus class the posteriorgram read for each unit's vowel.

    Args:
        units: The CV units.
        classes: ``branch.ddk_nucleus_classes``.

    Returns:
        One class per unit, :data:`UNRESOLVED` for a nucleus the vocabulary does not name.
    """
    lookup = nucleus_classes(classes)
    return [lookup.get(unit.vowel, UNRESOLVED) for unit in units]


def expected_fraction(realised: Sequence[str], expected: Sequence[str]) -> tuple[float | None, dict[str, Any]]:
    """How far one realised series matched the cycle its instruction asked for, position by position.

    Args:
        realised: One reading per unit, in order.
        expected: What the instruction asks for at each cycle position; length one for a
            single-syllable train.

    Returns:
        The fraction of resolved units whose reading is the one its cycle position expected, and the
        same fraction per position. Both are empty when nothing was resolved.
    """
    resolved = [(index, value) for index, value in enumerate(realised) if value != UNRESOLVED]
    if not resolved or not expected:
        return None, {}
    matched = sum(1 for index, value in resolved if value == expected[index % len(expected)])
    by_position: dict[str, Any] = {}
    for position in range(len(expected)):
        at = [value for index, value in resolved if index % len(expected) == position]
        hits = sum(1 for value in at if value == expected[position])
        by_position[str(position)] = None if not at else round(hits / len(at), 3)
    return round(matched / len(resolved), 3), by_position


def place_agreement(ppg_places: Sequence[str], burst_places: Sequence[str]) -> tuple[float | None, int]:
    """How often the two place instruments read the same place for one onset.

    Args:
        ppg_places: The posteriorgram's place per unit.
        burst_places: :func:`ddk_places`' place per unit, over the same extents.

    Returns:
        The fraction agreeing over the units both instruments resolved, and how many those were.
        ``(None, 0)`` when they resolved none in common.
    """
    both = [
        (ppg_place, burst_place)
        for ppg_place, burst_place in zip(ppg_places, burst_places)
        if ppg_place != UNRESOLVED and burst_place != UNRESOLVED
    ]
    if not both:
        return None, 0
    return round(sum(1 for ppg, burst in both if ppg == burst) / len(both), 3), len(both)


def ppg_evidence(
    store: ProvStore,
    reads: DdkReads,
    params: BranchParams,
    reading: PpgReading | None,
    template: Sequence[Syllable] | None,
) -> tuple[list[Proposal], list[Finding]]:
    """Everything the CV instrument has to say about one recording, in both modes.

    Args:
        store: The provenance store, for the acquisition covariates the rate is read against.
        reads: The derivatives, for the posteriorgram's entity id and the burst instrument.
        params: The operating points.
        reading: What the CV walk read, or None when the posteriorgram is absent.
        template: The syllable template the declared instruction cycles through, or None out of
            family.

    Returns:
        The one train span this instrument proposes and the findings it takes. An absent
        posteriorgram yields no span and one measurement that has no value; a posteriorgram the walk
        could not run over yields neither, because ``params.missing`` is where that is already said.
    """
    if reading is None:
        return [], [_absent(PPG, PPG_RATE)]
    if not reading.readable:
        return [], []
    evidence = _evidence(reads.ppg_id)
    findings: list[Finding] = [count(PPG_UNITS, len(reading.units), None, *evidence)]
    train = reading.train
    if train is None:
        return [], [*findings, measured(PPG_RATE, None, None, None, *evidence, unit=SYLLABLES_PER_S, reason=NO_TRAIN)]

    extent = train.extent
    extents = [unit.extent for unit in train.units]
    intervals = intervals_of(extents)
    cycle = len(template) if template else 1
    findings.extend(
        [
            measured(
                PPG_RATE,
                extent[0],
                extent[1],
                round(train.rate_hz, 3),
                *evidence,
                unit=SYLLABLES_PER_S,
                period_s=round(train.period_s, 4),
                jitter_over_median=round(train.jitter, 3),
                repetitions=train.repetitions,
                cv_units_n=len(reading.units),
                **acquisition_covariates(store, extent),
            ),
            measured(
                PPG_DISPERSION,
                extent[0],
                extent[1],
                dispersion(intervals),
                *evidence,
                support_intervals=len(intervals),
                trend_s_per_step=trend(intervals),
                by_position=dispersion_by_position(intervals, cycle),
            ),
            count("ppg_cv_onset_s", [round(start, 3) for start, _ in extents], None, *evidence),
            count("ppg_inter_onset_interval_s", [round(value, 3) for value in intervals], None, *evidence),
        ]
    )

    stop_places = params.point("ddk_stop_places") or {}
    places = unit_places(train.units, stop_places)
    nuclei = unit_nuclei(train.units, params.point("ddk_nucleus_classes") or {})
    if template:
        expected_places = [position.place for position in template]
        fraction, by_position = expected_fraction(places, expected_places)
        findings.append(
            measured(
                PPG_EXPECTED_PLACE,
                extent[0],
                extent[1],
                fraction,
                *evidence,
                expected_sequence=expected_places,
                by_position=by_position,
                realised_places=places,
                reading=PPG_PLACE_NOT_AUTHORITY,
            )
        )
        expected_nuclei = [position.nucleus for position in template]
        fraction, by_position = expected_fraction(nuclei, expected_nuclei)
        findings.append(
            measured(
                PPG_EXPECTED_NUCLEUS,
                extent[0],
                extent[1],
                fraction,
                *evidence,
                expected_sequence=expected_nuclei,
                by_position=by_position,
                realised_nuclei=nuclei,
            )
        )
    if reads.wideband is not None:
        agreement, support = place_agreement(places, ddk_places(extents, params, reads.wideband))
        findings.append(
            measured(
                PPG_PLACE_AGREEMENT,
                extent[0],
                extent[1],
                agreement,
                *_evidence(reads.ppg_id, reads.wideband_id),
                support_onsets=support,
                reading=PPG_PLACE_NOT_AUTHORITY,
            )
        )

    span = speech_span(
        "ppg_train",
        extent,
        *evidence,
        production="syllable_train_from_ppg",
        repetitions=train.repetitions,
        rate_hz=round(train.rate_hz, 3),
        period_s=round(train.period_s, 4),
        jitter_over_median=round(train.jitter, 3),
        cv_units_n=len(reading.units),
    )
    return [span], findings


def _with_ppg(done: Done, reading: PpgReading | None) -> Done:
    """Fold the CV instrument into a conformance the other instrument reached.

    Args:
        done: What the envelope instrument concluded.
        reading: What the CV walk read, or None when the posteriorgram is absent or unreadable.

    Returns:
        The conformance. An absent or unreadable instrument changes nothing. A train found where the
        instruction asked for one is conformance whichever instrument found it, so either suffices.
        A readable instrument that found no train, where the other found none either, is a task
        non-conformance rather than an unanswered question.
    """
    if reading is None or not reading.readable:
        return done
    return True if reading.train is not None or done is True else False


# --------------------------------------------------------------------- the one mode


def align_ddk(
    expectation: Expectation,
    store: ProvStore,
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
        expectation: The row SPEECH holds for this family, whose pattern is ``SYLLABLE_TRAIN`` or
            ``SYLLABLE_SEQUENCE`` and whose ``sequence`` is its syllable template.
        store: The provenance store.
        params: The operating points.
        reads: The derivatives, loaded by :func:`speech`.

    Returns:
        Whether the expected patterns were found, the one train span, and the findings.
    """
    template = expectation.sequence
    reading = ppg_reading(reads.ppg, params, template)
    cv_spans, cv_findings = ppg_evidence(store, reads, params, reading, template)
    declared = declared_duration_count(store, expectation.declared_duration_s)
    if reads.envelope is None:
        return Result(_with_ppg(UNDETERMINED, reading), cv_spans, [_absent(ENVELOPE), *cv_findings, *declared])

    train, rate_hz = ddk_carrier(store, params, reads.envelope)
    if train is None or train.extent is None:
        # No carrier has two causes and they are not the same report: no span held a readable train,
        # or the length guard's own boundary is unmeasured and no span could clear it. Only the
        # first is a reading of the recording, so only the first answers the conformance question.
        unmeasured_gate = params.point("train_min_s") is None
        return Result(
            _with_ppg(UNDETERMINED if unmeasured_gate else False, reading),
            cv_spans,
            [count("expected_event_count", 0, expectation.expected_event_count), *cv_findings, *declared],
        )

    onsets = events_in_span(reads.envelope, train, params)
    extent = hull(onsets) or train.extent
    starts = [round(start, 3) for start, _ in onsets]
    intervals = intervals_of(onsets)
    span_s = duration(extent)
    recording_s = duration(stream_extent(store))
    cycle = template if template and expectation.pattern is Pattern.SYLLABLE_SEQUENCE else None
    places_expected = None if cycle is None else [position.place for position in cycle]
    unit = CYCLES_OR_SYLLABLES_PER_S if cycle else SYLLABLES_PER_S

    carrier = _evidence(train.id, reads.envelope_id, reads.wideband_id)
    findings: list[Finding] = [
        count("expected_event_count", len(onsets), expectation.expected_event_count, *carrier),
        measured(
            RATE,
            extent[0],
            extent[1],
            rate_hz,
            *carrier,
            unit=unit,
            onset_rate_hz=None if span_s <= 0.0 else round(len(onsets) / span_s, 3),
            support_syllables=len(onsets),
            **acquisition_covariates(store, extent),
        ),
        count("inter_onset_interval_s", [round(value, 3) for value in intervals], None, *carrier),
        count("syllable_onset_s", starts, expectation.expected_event_count, *carrier),
        measured(
            "interval_dispersion",
            extent[0],
            extent[1],
            dispersion(intervals),
            *carrier,
            support_intervals=len(intervals),
            trend_s_per_step=trend(intervals),
            by_position={} if cycle is None else dispersion_by_position(intervals, len(cycle)),
        ),
        measured(
            "train_fraction_of_recording",
            extent[0],
            extent[1],
            None if recording_s <= 0.0 else round(span_s / recording_s, 3),
            *carrier,
            *stream_ids(store),
            train_s=round(span_s, 3),
            recording_s=round(recording_s, 3),
        ),
    ]

    attributes: dict[str, Any] = {"syllables_n": len(onsets), "production": "syllable_train"}
    # `ddk_carrier` only returns a span whose rate was readable, so a `rate_hz is not None` clause
    # here would be vacuous; the syllable count is the whole condition.
    done: Done = len(onsets) > 0
    if places_expected is not None:
        places = ddk_places(onsets, params, reads.wideband)
        if reads.wideband is None:
            findings.append(measured("syllable_place", extent[0], extent[1], None, *carrier, unavailable=WIDEBAND))
        for index, (onset, place) in enumerate(zip(onsets, places)):
            target = places_expected[index % len(places_expected)]
            if place not in (target, UNRESOLVED):
                findings.append(
                    deviation(
                        "syllable_sequence_mismatch", onset[0], onset[1], *carrier, expected=target, measured=place
                    )
                )
        resolved = [place for place in places if place != UNRESOLVED]
        cycles = sum(
            1
            for index in range(len(resolved) - len(places_expected) + 1)
            if resolved[index : index + len(places_expected)] == places_expected
        )
        if resolved:
            dominant = max(set(resolved), key=resolved.count)
            findings.append(
                measured(
                    "sequence_collapse_fraction",
                    extent[0],
                    extent[1],
                    round(resolved.count(dominant) / len(resolved), 3),
                    *carrier,
                    dominant_place=dominant,
                    support_syllables=len(resolved),
                )
            )
        findings.append(count("realised_cycles", cycles, None, *carrier))
        attributes.update(production="syllable_sequence", realised_cycles=cycles, resolved_n=len(resolved))
        done = bool(onsets) and cycles >= 1

    components = [speech_span("task_extent", extent, *carrier, **attributes), *cv_spans]
    recording_extent = stream_extent(store)
    if recording_extent is not None and touches_edge(extent, recording_extent):
        findings.append(deviation("truncation", extent[0], extent[1], *carrier, *stream_ids(store)))
    findings.extend(cv_findings)
    findings.extend(declared)
    return Result(_with_ppg(done, reading), components, findings)


# --------------------------------------------------------------------- what SPEECH reports


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


def syllable_detail(result: Result) -> dict[str, Any]:
    """SPEECH's report fields for a syllable-repetition task, read back off what the body returned.

    Args:
        result: What :func:`align_ddk` returned.

    Returns:
        The detail mapping, carrying rates and regularity as measurements and no normative reading
        of either. ``report.py``'s ``BRANCH_MEASURES["SPEECH"]`` names these keys.
    """
    trains = [component for component in result.components if component.role in TRAIN_ROLES]
    return {
        "trains_n": len(trains),
        "train_s": round(sum(component.end - component.start for component in trains), 3),
        "train_fraction": _value(result.deviations, "train_fraction_of_recording"),
        "modulation_peak_hz": _value(result.deviations, RATE),
        "modulation_unit": _covariate(result.deviations, RATE, "unit"),
        "interval_dispersion": _value(result.deviations, "interval_dispersion"),
        "interval_trend_s_per_step": _covariate(result.deviations, "interval_dispersion", "trend_s_per_step"),
        "ppg_rate_hz": _value(result.deviations, PPG_RATE),
        "ppg_repetitions": _covariate(result.deviations, PPG_RATE, "repetitions"),
        "ppg_period_s": _covariate(result.deviations, PPG_RATE, "period_s"),
        "ppg_jitter_over_median": _covariate(result.deviations, PPG_RATE, "jitter_over_median"),
        "ppg_cv_units_n": _covariate(result.deviations, PPG_RATE, "cv_units_n"),
        "ppg_interval_trend_s_per_step": _covariate(result.deviations, PPG_DISPERSION, "trend_s_per_step"),
        "ppg_expected_place_fraction": _value(result.deviations, PPG_EXPECTED_PLACE),
        "ppg_place_agreement": _value(result.deviations, PPG_PLACE_AGREEMENT),
        "ppg_expected_nucleus_fraction": _value(result.deviations, PPG_EXPECTED_NUCLEUS),
        "ppg_trains_n": sum(1 for component in result.components if component.role == "ppg_train"),
    }
