"""The syllable-train instrument: its rate, its regularity, and the ten tasks that ask for one.

Not a branch. :func:`align_speech` serves the ten ``SYLLABLE_REPETITION`` families through
:func:`align_ddk`, which evaluates one of them against what its instruction asked for, and every
measurement below is a SPEECH measurement taken by this module's instruments.

The design is ``specs/20260817-triage-workflow-dag/branch-ddk.md`` (D1-D6); the ported bodies are
``expected-patterns.md``; what porting decided is ``branch-ddk-implementation.md``; why the branch
became this module is ``ddk-dissolved-into-speech.md``. The posteriorgram instrument is a cyclic
decode of the token's phoneme sequence and its design, its parameters and the owner's decisions on
it are ``ddk-template-decode.md``.
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
    acquisition_covariates,
    amplitude_spans,
    branch_params,
    count,
    declared_duration_count,
    derivative_arrays,
    deviation,
    duration,
    measured,
    mode_of,
    overlaps,
    propose_spans,
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
from senselab.utils.prov_store import Entity, ProvStore

RATE = "ddk_syllable_rate_from_envelope_modulation_hz"
"""The envelope modulation channel's rate, named for the instrument that took it.

It reads periodicity without segmenting anything, which is why it survived the peak walk it used to
share a carrier with."""

ENVELOPE = "energy_envelope"
PPG = "ppg_posteriorgram"

PPG_RATE = "ddk_syllable_rate_from_ppg_decode_hz"
"""The second rate measurement, named for the instrument that took it. Not a substitute for
:data:`RATE`: the two read different signals -- phonetic identity and amplitude -- and are reported
side by side."""

PPG_REPETITIONS = "ddk_repetition_count_from_ppg_decode"
"""How many repetitions of the declared template the decode completed.

Zero is a reading and not an absence: the value is 0 and the per-position mass sits beside it."""

PPG_MASS = "ddk_position_realised_mass"
"""How much of the posterior the expected class held where the decode placed each position.

The primary sequence-realisation measurement. Defined at every position the decode reached, it
degrades continuously, and it is honestly ambiguous between produced differently and read poorly."""

PPG_DISPERSION = "ddk_ppg_period_dispersion"

INSTRUMENT_READING = "ddk_cv_instrument_reading"
"""The posteriorgram instrument's own reading of what was produced over the extent it covers.

Its value is the per-position realised mass of each completed repetition. Written beside the
consensus transcript, never in place of it; ``ddk-instrument-over-asr.md`` holds why the recogniser
text is neither replaced nor withheld."""

TRANSCRIPT_CLAIM = "lexical_transcript"
"""What a contradicted consensus word claims, and what the contest is against."""

CONTRADICTED = "instrument_contradicted"
"""Why that claim is contested: the instrument covers the word's extent and read repetitions."""

CV_AUTHORITY = "cv_instrument"
"""Which instrument is authoritative over the contested extent, on a declared syllable task."""

SYLLABLES_PER_S = "syllables_per_s"
CYCLES_PER_S = "cycles_per_s"
CYCLES_OR_SYLLABLES_PER_S = "cycles_or_syllables_per_s"
"""A sequential train modulates at the cycle rate as well as the syllable rate; the envelope peak is
one of the two and the harmonic-equality tolerance that would separate them is unmeasured, so the
unit is carried as ambiguous. The decode's own two rates travel beside it, so which one the peak
matched is checkable per recording."""

TASK_EXTENT = "task_extent"
"""The role that says where the declared task was performed. Exactly one may survive a recording."""

TASK_FROM_DECODE = "syllable_task_from_decode"
"""The extent is the decoded repetition span: first repetition's start to last repetition's end.

The only production a ``task_extent`` on this family carries. The envelope mints none."""

NO_REPETITIONS = "the decode completed no repetition of the declared template"
NO_ENVELOPE = "the energy envelope is absent; the syllable train's only rate instrument could not be read"
NO_PPG = "the phonetic posteriorgram is absent; the CV instrument could not be read"


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
        ppg: The phonetic posteriorgram, or None when the derivative is absent.
        ppg_id: That measurement's entity id, for the derivation of everything the decode reads.
    """

    envelope: EnvelopeTrack | None = None
    envelope_id: str | None = None
    ppg: Posteriorgram | None = None
    ppg_id: str | None = None


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
    del source
    envelope = find_measurement(store, ENVELOPE)
    posteriorgram = find_measurement(store, PPG)
    frames = read_posteriorgram(store, run_dir)
    return DdkReads(
        envelope=read_envelope_track(store, run_dir),
        envelope_id=None if envelope is None else envelope.id,
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


# --------------------------------------------------------------------- the envelope instrument


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


def dispersion(intervals: Sequence[float]) -> float | None:
    """D3. The coefficient of variation of an interval sequence.

    Args:
        intervals: The inter-repetition periods.

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
    """D3. How the interval changes across the train: seconds per repetition step.

    Args:
        intervals: The inter-repetition periods.

    Returns:
        The least-squares slope, or None when the sequence is shorter than the two points a slope is
        defined over.
    """
    values = np.asarray(intervals, dtype=float)
    if values.size < 2:
        return None
    return float(np.polyfit(np.arange(values.size, dtype=float), values, 1)[0])


# --------------------------------------------------------------------- the posteriorgram decode


OTHER_CLASS = "other"
"""The part of the partition holding every phoneme no template position names, silence included."""


@dataclass(frozen=True)
class Posteriorgram:
    """The stored phonetic posteriorgram, as the template decode reads it.

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
class Visit:
    """One unbroken stay at one template position.

    Attributes:
        position: Which template position, by index.
        first: The first frame charged to it.
        last: The last frame charged to it.
    """

    position: int
    first: int
    last: int


@dataclass(frozen=True)
class Decode:
    """What the cyclic decode read off one posteriorgram against one template.

    Attributes:
        template: The template's phonemes, in order.
        classes: The equivalence class each position scores the summed mass of, in the same order.
        repetitions: One ``(first frame, last frame)`` per completed repetition, in time order.
        realised_mass: The mean class mass over the frames charged to each position, one per
            position, or None for a position no completed repetition reached.
        occupancy: How many frames each position held across every completed repetition.
        filler_frames: How many frames the decode charged to a filler state.
        frames: How many frames were decoded.
        seconds_per_frame: The frame period, for turning any of the above into seconds.
        score_per_frame: The path's total log-likelihood over the frame count.
        per_repetition_mass: The per-position realised mass of each completed repetition, in time
            order, one inner tuple per repetition.
        vowel_classes: Which of the class names are vowel classes, so a syllable can be counted
            without the decode reaching back into the configuration it was built from.
        readable: Whether the decode could run at all; False when a class vocabulary is unmeasured.
    """

    template: tuple[str, ...] = ()
    classes: tuple[str, ...] = ()
    repetitions: tuple[tuple[int, int], ...] = ()
    realised_mass: tuple[float | None, ...] = ()
    occupancy: tuple[int, ...] = ()
    filler_frames: int = 0
    frames: int = 0
    seconds_per_frame: float = 0.0
    score_per_frame: float | None = None
    per_repetition_mass: tuple[tuple[float | None, ...], ...] = ()
    vowel_classes: tuple[str, ...] = ()
    readable: bool = True

    @property
    def count(self) -> int:
        """How many repetitions completed."""
        return len(self.repetitions)

    @property
    def extent(self) -> tuple[float, float] | None:
        """The first completed repetition's start to the last one's end, in seconds.

        A boundary and not a mask: every filler frame between the first and last repetition lies
        inside it. None when no repetition completed.
        """
        if not self.repetitions:
            return None
        start = self.repetitions[0][0] * self.seconds_per_frame
        end = (self.repetitions[-1][1] + 1) * self.seconds_per_frame
        return (start, end) if end > start else None

    @property
    def starts_s(self) -> list[float]:
        """Where each completed repetition started, in seconds."""
        return [round(first * self.seconds_per_frame, 3) for first, _ in self.repetitions]

    @property
    def periods_s(self) -> list[float]:
        """The elapsed time between the starts of consecutive repetitions, in seconds."""
        starts = [first * self.seconds_per_frame for first, _ in self.repetitions]
        return [round(second - first, 4) for first, second in zip(starts, starts[1:])]

    @property
    def period_s(self) -> float | None:
        """The median period, or None when fewer than two repetitions completed."""
        periods = self.periods_s
        if not periods:
            return None
        median = float(np.median(periods))
        return round(median, 4) if median > 0.0 else None

    @property
    def filler_fraction(self) -> float | None:
        """The share of decoded frames no template position held, or None over no frames."""
        if self.frames <= 0:
            return None
        return round(self.filler_frames / self.frames, 3)

    @property
    def vowel_positions(self) -> int:
        """How many of the template's positions are a vowel, which is its syllable count."""
        return sum(1 for name in self.classes if name in set(self.vowel_classes))

    @property
    def syllables(self) -> int:
        """How many syllables the completed repetitions hold, the unit a declared count is in."""
        return self.count * self.vowel_positions

    @property
    def rate_hz(self) -> float | None:
        """Syllables per second over the median period, or None when no period is defined."""
        period = self.period_s
        if period is None or not self.vowel_positions:
            return None
        return round(self.vowel_positions / period, 3)

    @property
    def cycle_rate_hz(self) -> float | None:
        """Repetitions per second over the median period, or None when no period is defined."""
        period = self.period_s
        return None if period is None else round(1.0 / period, 3)

    @property
    def occupancy_s(self) -> list[float | None]:
        """How long each position was held in total, in seconds."""
        return [round(frames * self.seconds_per_frame, 3) for frames in self.occupancy]


NO_DECODE = Decode()
"""The decode of a recording nothing was decoded over: no template declared."""

UNREADABLE = Decode(readable=False)
"""The decode that could not run: a class vocabulary is unmeasured and ``params.missing`` names it."""


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
        Each phoneme mapped to its class. A phoneme two classes both name resolves to the first in
        declaration order, mappings in the order given.
    """
    lookup: dict[str, str] = {}
    for mapping in mappings:
        for name, phonemes in mapping.items():
            for phoneme in phonemes:
                lookup.setdefault(phoneme, name)
    return lookup


def min_phone_frames(seconds_per_frame: float, burst_window_ms: float) -> int:
    """``D``: how many frames one template position's state chain is.

    Args:
        seconds_per_frame: The posteriorgram's own frame period.
        burst_window_ms: ``branch.burst_window_ms``.

    Returns:
        How many frames the burst window spans, at least one. Derived at read time from the stored
        frame period, so it is not a configuration point of its own.
    """
    if seconds_per_frame <= 0.0:
        return 1
    return max(1, int(np.ceil(burst_window_ms / (1000.0 * seconds_per_frame))))


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


def arcs(positions: int, chain: int) -> tuple[np.ndarray, np.ndarray]:
    """The decode's topology as a padded predecessor table.

    Args:
        positions: How many template positions, ``N``.
        chain: How many sub-states one position is, ``D``.

    Returns:
        ``(predecessors, valid)``, both ``(states, width)``. Row ``s`` holds the states an arc runs
        from into ``s``, padded, with ``valid`` False where the entry is padding.
    """
    states = positions * (chain + 1)

    def sub(position: int, step: int) -> int:
        return position * chain + step

    def filler(position: int) -> int:
        return positions * chain + position

    preds: list[list[int]] = [[] for _ in range(states)]
    for position in range(positions):
        previous = (position - 1) % positions
        preds[sub(position, 0)].extend([sub(previous, chain - 1), filler(previous)])
        for step in range(1, chain):
            preds[sub(position, step)].append(sub(position, step - 1))
        preds[sub(position, chain - 1)].append(sub(position, chain - 1))
        preds[filler(position)].extend([sub(position, chain - 1), filler(position)])
    width = max(len(entry) for entry in preds)
    table = np.zeros((states, width), dtype=int)
    valid = np.zeros((states, width), dtype=bool)
    for state, entry in enumerate(preds):
        table[state, : len(entry)] = entry
        valid[state, : len(entry)] = True
    return table, valid


def viterbi(emissions: np.ndarray, positions: int, chain: int) -> np.ndarray:
    """The best legal path through the cyclic topology, as one state per frame.

    Args:
        emissions: ``(frame, state)`` log-likelihoods.
        positions: How many template positions.
        chain: How many sub-states one position is.

    Returns:
        The state index per frame. The path may start and end in any state, so a recording that
        begins mid-performance is decodable and a trailing partial repetition is simply not one.
    """
    table, valid = arcs(positions, chain)
    total, states = emissions.shape
    rows = np.arange(states)
    score = emissions[0].copy()
    back = np.zeros((total, states), dtype=int)
    for frame in range(1, total):
        candidates = np.where(valid, score[table], -np.inf)
        best = candidates.argmax(axis=1)
        back[frame] = table[rows, best]
        score = candidates[rows, best] + emissions[frame]
    path = np.zeros(total, dtype=int)
    path[-1] = int(score.argmax())
    for frame in range(total - 1, 0, -1):
        path[frame - 1] = back[frame, path[frame]]
    return path


def visits(path: np.ndarray, positions: int, chain: int) -> list[Visit]:
    """The path's unbroken stays at template positions, in time order.

    Args:
        path: The state per frame.
        positions: How many template positions.
        chain: How many sub-states one position is.

    Returns:
        One :class:`Visit` per stay. A frame in a filler state closes the stay it follows and opens
        none of its own.
    """
    found: list[Visit] = []
    for frame, state in enumerate(int(value) for value in path):
        if state >= positions * chain:
            continue
        position = state // chain
        if found and found[-1].position == position and found[-1].last == frame - 1:
            found[-1] = Visit(position, found[-1].first, frame)
        else:
            found.append(Visit(position, frame, frame))
    return found


def repetitions_of(found: Sequence[Visit], positions: int) -> list[tuple[Visit, ...]]:
    """The completed repetitions among a visit sequence.

    Args:
        found: The visits, in time order.
        positions: How many template positions one repetition holds.

    Returns:
        One tuple of visits per repetition that ran position 0 through position ``N-1`` in order. A
        leading partial repetition and a trailing one are each simply not completed.
    """
    complete: list[tuple[Visit, ...]] = []
    open_run: list[Visit] = []
    for visit in found:
        if visit.position == 0:
            open_run = [visit]
        elif open_run and visit.position == open_run[-1].position + 1:
            open_run.append(visit)
        else:
            open_run = []
        if len(open_run) == positions:
            complete.append(tuple(open_run))
            open_run = []
    return complete


def decode_template(ppg: Posteriorgram | None, params: BranchParams, template: Sequence[str] | None) -> Decode | None:
    """Decode one posteriorgram against one template's phoneme sequence.

    Args:
        ppg: The posteriorgram, or None when the derivative is absent.
        params: The operating points, for the class vocabularies and the burst window.
        template: The declared phoneme sequence, or None when no row declares one.

    Returns:
        The decode, or None when the derivative is absent, which is an absent instrument and not a
        negative reading. :data:`UNREADABLE` when a class vocabulary is unmeasured, which
        ``params.missing`` already names. :data:`NO_DECODE` when no template was declared.
    """
    if ppg is None:
        return None
    places = params.point("phoneme_place_classes")
    vowels = params.point("phoneme_vowel_classes")
    window_ms = params.point("burst_window_ms")
    if places is None or vowels is None or window_ms is None:
        return UNREADABLE
    if not template:
        return NO_DECODE
    lookup = phoneme_classes(places, vowels)
    classes = tuple(lookup.get(phoneme, OTHER_CLASS) for phoneme in template)
    named = tuple(dict.fromkeys(name for name in classes if name != OTHER_CLASS))
    masses = class_masses(ppg, named, lookup)
    column = {name: index for index, name in enumerate(named)}
    per_position = np.stack([masses[:, column.get(name, masses.shape[1] - 1)] for name in classes], axis=1)
    chain = min_phone_frames(ppg.seconds_per_frame, float(window_ms))
    floor = ppg.emission_floor
    tiers = np.log(np.maximum(per_position, floor))
    filler = np.log(np.maximum(masses[:, -1], floor))
    emissions = np.concatenate(
        [np.repeat(tiers, chain, axis=1), np.repeat(filler[:, None], len(template), axis=1)], axis=1
    )
    path = viterbi(emissions, len(template), chain)
    complete = repetitions_of(visits(path, len(template), chain), len(template))
    charged: list[list[float]] = [[] for _ in template]
    per_repetition: list[tuple[float | None, ...]] = []
    occupancy = [0] * len(template)
    for run in complete:
        row: list[float | None] = []
        for visit in run:
            values = per_position[visit.first : visit.last + 1, visit.position]
            charged[visit.position].extend(float(value) for value in values)
            occupancy[visit.position] += visit.last - visit.first + 1
            row.append(round(float(values.mean()), 3) if values.size else None)
        per_repetition.append(tuple(row))
    total = int(emissions.shape[0])
    return Decode(
        template=tuple(template),
        classes=classes,
        repetitions=tuple((run[0].first, run[-1].last) for run in complete),
        realised_mass=tuple(round(float(np.mean(values)), 3) if values else None for values in charged),
        occupancy=tuple(occupancy),
        filler_frames=int(np.count_nonzero(path >= len(template) * chain)),
        frames=total,
        seconds_per_frame=ppg.seconds_per_frame,
        score_per_frame=round(float(emissions[np.arange(total), path].sum() / total), 4) if total else None,
        per_repetition_mass=tuple(per_repetition),
        vowel_classes=tuple(vowels),
        readable=True,
    )


# --------------------------------------------------------------------- what the decode has to say


def contradicted_words(store: ProvStore, extent: tuple[float, float]) -> list[Entity]:
    """The consensus words the instrument's covered extent contradicts, in index order.

    Args:
        store: The provenance store.
        extent: The extent the instrument covers.

    Returns:
        Every live lexical consensus word whose hull shares any interval with it. The hull rather
        than the fitted extent, because it is the read that misses no word; bracketed tokens are
        not a claim that a word was said and are left alone.
    """
    return [word for word in lexical_words(store) if overlaps(word_hull(word), extent)]


def instrument_authority(store: ProvStore, decode: Decode, evidence: Sequence[str]) -> list[Finding]:
    """The decode's reading recorded as authoritative, and each word it contradicts.

    Additive only: no word is invalidated, no transcript is rewritten, and no text is copied into
    a finding. ``specs/20260817-triage-workflow-dag/ddk-instrument-over-asr.md`` holds why, and
    which consumers this reaches.

    Args:
        store: The provenance store, for the consensus words.
        decode: What the decode read.
        evidence: The entity ids it was read off.

    Returns:
        One :data:`INSTRUMENT_READING` measurement over the decoded extent and one ``contest`` per
        contradicted word, or nothing at all when no repetition completed.
    """
    extent = decode.extent
    if extent is None:
        return []
    words = contradicted_words(store, extent)
    findings: list[Finding] = [
        measured(
            INSTRUMENT_READING,
            extent[0],
            extent[1],
            [list(row) for row in decode.per_repetition_mass],
            *evidence,
            *(word.id for word in words),
            authority=CV_AUTHORITY,
            supersedes=TRANSCRIPT_CLAIM,
            positions=list(decode.template),
            classes=list(decode.classes),
            onsets_s=decode.starts_s,
            repetitions=decode.count,
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


def decode_evidence(
    store: ProvStore,
    reads: DdkReads,
    params: BranchParams,
    decode: Decode | None,
    declared_event_count: int | None = None,
) -> list[Finding]:
    """Everything the posteriorgram instrument has to say about one recording.

    The decoded repetition count and the declared count are written beside each other and nothing
    folds them: no conformance term reads the pair, no score and no gate. ``ddk-template-decode.md``
    holds the owner's ruling that counts are heuristics and not targets.

    Args:
        store: The provenance store, for the acquisition covariates the rate is read against.
        reads: The derivatives, for the posteriorgram's entity id.
        params: The operating points, for the burst window the chain length is derived from.
        decode: What the decode read, or None when the posteriorgram is absent.
        declared_event_count: The instruction's own syllable count, or None when it declares none.

    Returns:
        The findings. An absent posteriorgram yields one measurement that has no value; a decode
        that could not run yields none at all, because ``params.missing`` is where that is already
        said.
    """
    if decode is None:
        return [_absent(PPG, PPG_RATE)]
    if not decode.readable or not decode.template:
        return []
    evidence = _evidence(reads.ppg_id)
    extent = decode.extent
    window_ms = params.point("burst_window_ms")
    floor = None if reads.ppg is None else reads.ppg.emission_floor
    chain = None if reads.ppg is None or window_ms is None else min_phone_frames(reads.ppg.seconds_per_frame, window_ms)
    start, end = (None, None) if extent is None else extent
    per_repetition = (
        None
        if not decode.vowel_positions or declared_event_count is None
        else (declared_event_count // decode.vowel_positions)
    )
    findings: list[Finding] = [
        measured(
            PPG_REPETITIONS,
            start,
            end,
            decode.count,
            *evidence,
            declared_event_count=declared_event_count,
            declared_repetitions=per_repetition,
            repetition_start_s=decode.starts_s,
            filler_fraction=decode.filler_fraction,
            score_per_frame=decode.score_per_frame,
            frames=decode.frames,
            min_phone_frames=chain,
        ),
        measured(
            PPG_MASS,
            start,
            end,
            list(decode.realised_mass),
            *evidence,
            positions=list(decode.template),
            classes=list(decode.classes),
            occupancy_frames=list(decode.occupancy),
            occupancy_s=decode.occupancy_s,
            repetitions=decode.count,
            emission_floor=floor,
        ),
        count("expected_event_count", decode.syllables, declared_event_count, *evidence),
    ]
    if extent is None:
        findings.append(measured(PPG_RATE, None, None, None, *evidence, unit=SYLLABLES_PER_S, reason=NO_REPETITIONS))
        return findings
    periods = decode.periods_s
    findings.extend(
        [
            measured(
                PPG_RATE,
                extent[0],
                extent[1],
                decode.rate_hz,
                *evidence,
                unit=SYLLABLES_PER_S,
                period_s=decode.period_s,
                cycle_rate_hz=decode.cycle_rate_hz,
                repetitions=decode.count,
                **acquisition_covariates(store, extent),
            ),
            measured(
                PPG_DISPERSION,
                extent[0],
                extent[1],
                dispersion(periods),
                *evidence,
                support_intervals=len(periods),
                trend_s_per_step=trend(periods),
            ),
            count("ppg_repetition_start_s", decode.starts_s, None, *evidence),
            count("ppg_repetition_period_s", periods, None, *evidence),
        ]
    )
    findings.extend(instrument_authority(store, decode, evidence))
    return findings


def task_extent_span(decode: Decode | None, decode_ids: Sequence[str]) -> Proposal | None:
    """The one ``task_extent`` this family leaves behind: the decoded repetition span.

    Args:
        decode: What the decode read, or None when the posteriorgram is absent.
        decode_ids: The entity ids it was read off.

    Returns:
        The span, or None when the decode read no repetition of the declared template.
    """
    extent = None if decode is None else decode.extent
    if decode is None or extent is None:
        return None
    return speech_span(
        TASK_EXTENT,
        extent,
        *decode_ids,
        production=TASK_FROM_DECODE,
        repetitions=decode.count,
        syllables_n=decode.syllables,
        filler_fraction=decode.filler_fraction,
    )


def _with_decode(done: Done, decode: Decode | None) -> Done:
    """Fold the posteriorgram instrument into a conformance the other instrument reached.

    Args:
        done: What the envelope instrument concluded.
        decode: What the decode read, or None when the posteriorgram is absent or unreadable.

    Returns:
        The conformance. An absent or unreadable instrument changes nothing. A repetition found
        where the instruction asked for one is conformance whichever instrument found it, so either
        suffices. A readable instrument that completed no repetition, where the other found no
        carrier either, is a task non-conformance rather than an unanswered question. A collapsed
        sequence, a weak position and fewer repetitions than declared are none of them ``false``.
    """
    if decode is None or not decode.readable:
        return done
    return True if decode.count >= 1 or done is True else False


# --------------------------------------------------------------------- the one mode


def align_ddk(
    expectation: Expectation,
    store: ProvStore,
    params: BranchParams,
    *,
    reads: DdkReads = DdkReads(),
) -> Result:
    """Evaluate one declared ``SYLLABLE_REPETITION`` task against what its instruction asked for.

    Spans proposed: exactly one ``task_extent``, or none when the decode read no repetition.
    :func:`task_extent_span` mints it and the posteriorgram is the only instrument that can;
    ``specs/20260817-triage-workflow-dag/ddk-envelope-mints-no-extent.md`` holds why, and
    ``ddk-task-extent-precedence.md`` what it superseded. An individual repetition is not a span:
    the rate, the period dispersion and the per-position mass are statistics over the decoded
    repetition series, and one span per repetition would add roughly ten spans per recording
    carrying no measurement of their own. The repetition starts and periods travel as ``counts``
    entries.

    Args:
        expectation: The row SPEECH holds for this family, whose pattern is ``SYLLABLE_TRAIN`` or
            ``SYLLABLE_SEQUENCE`` and whose ``sequence`` is its phoneme template.
        store: The provenance store.
        params: The operating points.
        reads: The derivatives, loaded by :func:`speech`.

    Returns:
        Whether the expected patterns were found, the one task extent, and the findings.
    """
    decode = decode_template(reads.ppg, params, expectation.sequence)
    ppg_findings = decode_evidence(store, reads, params, decode, expectation.expected_event_count)
    declared = declared_duration_count(store, expectation.declared_duration_s)
    ppg_ids = _evidence(reads.ppg_id)
    sequence = expectation.pattern is Pattern.SYLLABLE_SEQUENCE
    findings: list[Finding] = []
    carrier_extent: tuple[float, float] | None = None
    carrier_ids: tuple[str, ...] = ()
    done: Done = UNDETERMINED

    if reads.envelope is None:
        findings.append(_absent(ENVELOPE))
    else:
        train, rate_hz = ddk_carrier(store, params, reads.envelope)
        unmeasured_gate = params.point("train_min_s") is None
        if train is None or train.extent is None:
            # No carrier has two causes and they are not the same report: no span held a readable
            # train, or the length guard's own boundary is unmeasured and no span could clear it.
            # Only the first is a reading of the recording, so only the first answers conformance.
            done = UNDETERMINED if unmeasured_gate else False
        else:
            carrier_extent, carrier_ids = train.extent, _evidence(train.id, reads.envelope_id)
            done = True
            findings.append(
                measured(
                    RATE,
                    carrier_extent[0],
                    carrier_extent[1],
                    rate_hz,
                    *carrier_ids,
                    unit=CYCLES_OR_SYLLABLES_PER_S if sequence else SYLLABLES_PER_S,
                    decoded_syllable_rate_hz=None if decode is None else decode.rate_hz,
                    decoded_cycle_rate_hz=None if decode is None else decode.cycle_rate_hz,
                    **acquisition_covariates(store, carrier_extent),
                )
            )

    span = task_extent_span(decode, ppg_ids)
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

    return Result(_with_decode(done, decode), components, [*findings, *ppg_findings, *declared])


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
        The detail mapping, carrying rates, regularity and per-position realisation as measurements
        and no normative reading of any of them. ``common.py``'s ``BRANCH_MEASURES["SPEECH"]`` names
        these keys, and nothing here folds the decoded count against the declared one.
    """
    trains = [component for component in result.components if component.role == TASK_EXTENT]
    return {
        "trains_n": len(trains),
        "train_s": round(sum(component.end - component.start for component in trains), 3),
        "train_fraction": _value(result.deviations, "train_fraction_of_recording"),
        "modulation_peak_hz": _value(result.deviations, RATE),
        "modulation_unit": _covariate(result.deviations, RATE, "unit"),
        "ppg_syllable_rate_hz": _value(result.deviations, PPG_RATE),
        "ppg_cycle_rate_hz": _covariate(result.deviations, PPG_RATE, "cycle_rate_hz"),
        "ppg_repetitions": _value(result.deviations, PPG_REPETITIONS),
        "ppg_declared_event_count": _covariate(result.deviations, PPG_REPETITIONS, "declared_event_count"),
        "ppg_period_s": _covariate(result.deviations, PPG_RATE, "period_s"),
        "ppg_period_cv": _value(result.deviations, PPG_DISPERSION),
        "ppg_period_trend_s_per_step": _covariate(result.deviations, PPG_DISPERSION, "trend_s_per_step"),
        "ppg_positions": _covariate(result.deviations, PPG_MASS, "positions"),
        "ppg_realised_mass": _value(result.deviations, PPG_MASS),
        "ppg_occupancy_s": _covariate(result.deviations, PPG_MASS, "occupancy_s"),
        "ppg_filler_fraction": _covariate(result.deviations, PPG_REPETITIONS, "filler_fraction"),
        "ppg_score_per_frame": _covariate(result.deviations, PPG_REPETITIONS, "score_per_frame"),
        "ppg_contradicted_words_n": _covariate(result.deviations, INSTRUMENT_READING, "contradicted_words_n"),
    }
