"""The breathing pattern of an airway recording, read off PREPROCESS's stored derivatives.

The narrowband spectrogram (computed over the pre-emphasised stream) gives a breath-band envelope
and a spectral flatness per frame; ``phonation_tracks`` gives voicing. Frames well above the
recording's own floor and noise-like are active; active runs merge into events; events of a breath's
length that are mostly unvoiced are breaths. Their number and spacing name the pattern:
``no_breathing``, ``single_breath``, ``alternating_breaths`` or ``irregular_events``.

Every parameter is in ``data/breath_pattern.yaml``. See
``specs/20261005-breathing-pattern/design.md``.
"""

from __future__ import annotations

import functools
import json
import unicodedata
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml
from scipy.signal import find_peaks

from senselab.audio.tasks.classification.label_scores import label_scores
from senselab.audio.workflows.triage.nodes.common import find_measurement, lexical_words, live_entities, word_hull
from senselab.audio.workflows.triage.residue import is_non_lexical
from senselab.audio.workflows.triage.vocabulary import SUPERSEDES
from senselab.utils.prov_store import ProvStore

BREATH_PATTERN_PATH = Path(__file__).parent / "data" / "breath_pattern.yaml"

SPECTROGRAM = "spectrogram_narrowband"
PHONATION_TRACKS = "phonation_tracks"
YAMNET_SCORES = "yamnet_scores"
HEAR_SCORES = "hear_scores"
TASK_EXTENT_ROLE = "task_extent"
LATIN_LANGUAGES = ("en", "es", "fr", "pt", "de", "it")
"""Declared languages written in Latin script, whose speech words must carry a Latin letter."""

NO_BREATHING = "no_breathing"
SINGLE_BREATH = "single_breath"
ALTERNATING_BREATHS = "alternating_breaths"
IRREGULAR_EVENTS = "irregular_events"
PATTERNS = (NO_BREATHING, SINGLE_BREATH, ALTERNATING_BREATHS, IRREGULAR_EVENTS)
"""Every pattern :func:`measure_breath_pattern` names."""


@dataclass(frozen=True)
class BreathPatternParameters:
    """The measure's parameters, as ``data/breath_pattern.yaml`` holds them.

    Attributes:
        breath_band_hz: Where breath turbulence sits; the envelope sums power over it.
        flatness_band_hz: The band spectral flatness is measured over.
        smooth_s: The moving-median window on the band envelope.
        floor_percentile: The percentile of the envelope taken as the recording's own floor.
        rise_db: How far above that floor a frame must be to be active.
        flatness_min: The flatness a frame must reach to be noise-like.
        voicing_strength_min: The pitch strength at or above which a frame with an F0 is voiced.
        merge_gap_s: Active runs closer than this merge into one event.
        event_s: The duration range of one breath event.
        event_trim_db: An event's span is the frames within this of its own envelope peak.
        voiced_fraction_max: The share of an event's frames that may be voiced.
        period_s: The breathing-cycle range the envelope's autocorrelation is searched over.
        autocorr_min: The autocorrelation peak in that range that counts as a rhythm.
        interval_s: The onset-to-onset interval range that fits breathing.
        interval_share_min: The share of intervals in range that counts as a rhythm.
    """

    breath_band_hz: tuple[float, float]
    flatness_band_hz: tuple[float, float]
    smooth_s: float
    floor_percentile: float
    rise_db: float
    flatness_min: float
    voicing_strength_min: float
    merge_gap_s: float
    event_s: tuple[float, float]
    event_trim_db: float
    voiced_fraction_max: float
    period_s: tuple[float, float]
    autocorr_min: float
    interval_s: tuple[float, float]
    interval_share_min: float


@dataclass(frozen=True)
class ModulationParameters:
    """The modulation reading's parameters, ``data/breath_pattern.yaml``'s ``modulation`` section.

    Attributes:
        band_edges_hz: Edges of the subbands whose envelopes are read.
        fs_mod: The rate the envelopes are resampled to, in Hz.
        breath_band_hz: The breathing-rate modulation band.
        syllabic_band_hz: The syllabic-rate modulation band speech occupies.
        reference_band_hz: The modulation band each subband spectrum is normalised over.
        floor_percentile: The percentile of the broadband envelope taken as the floor.
        active_rise_db: How far above that floor the active span starts and ends.
        min_duration_s: The shortest recording the reading is taken over.
    """

    band_edges_hz: tuple[float, ...]
    fs_mod: float
    breath_band_hz: tuple[float, float]
    syllabic_band_hz: tuple[float, float]
    reference_band_hz: tuple[float, float]
    floor_percentile: float
    active_rise_db: float
    min_duration_s: float


@functools.cache
def modulation_parameters() -> ModulationParameters:
    """The ``modulation`` parameters of ``data/breath_pattern.yaml``.

    Returns:
        The parameters.
    """
    held = (yaml.safe_load(BREATH_PATTERN_PATH.read_text()) or {})["modulation"]

    def pair(key: str) -> tuple[float, float]:
        low, high = held[key]
        return float(low), float(high)

    return ModulationParameters(
        band_edges_hz=tuple(float(edge) for edge in held["band_edges_hz"]),
        fs_mod=float(held["fs_mod"]),
        breath_band_hz=pair("breath_band_hz"),
        syllabic_band_hz=pair("syllabic_band_hz"),
        reference_band_hz=pair("reference_band_hz"),
        floor_percentile=float(held["floor_percentile"]),
        active_rise_db=float(held["active_rise_db"]),
        min_duration_s=float(held["min_duration_s"]),
    )


@functools.cache
def breath_pattern_parameters() -> BreathPatternParameters:
    """The parameters of ``data/breath_pattern.yaml``.

    Returns:
        The parameters.
    """
    held = (yaml.safe_load(BREATH_PATTERN_PATH.read_text()) or {})["measure"]

    def pair(key: str) -> tuple[float, float]:
        low, high = held[key]
        return float(low), float(high)

    return BreathPatternParameters(
        breath_band_hz=pair("breath_band_hz"),
        flatness_band_hz=pair("flatness_band_hz"),
        smooth_s=float(held["smooth_s"]),
        floor_percentile=float(held["floor_percentile"]),
        rise_db=float(held["rise_db"]),
        flatness_min=float(held["flatness_min"]),
        voicing_strength_min=float(held["voicing_strength_min"]),
        merge_gap_s=float(held["merge_gap_s"]),
        event_s=pair("event_s"),
        event_trim_db=float(held["event_trim_db"]),
        voiced_fraction_max=float(held["voiced_fraction_max"]),
        period_s=pair("period_s"),
        autocorr_min=float(held["autocorr_min"]),
        interval_s=pair("interval_s"),
        interval_share_min=float(held["interval_share_min"]),
    )


@dataclass(frozen=True)
class ModulationReading:
    """What :func:`measure_modulation` read off the subband envelopes' modulation spectra.

    Attributes:
        breath_vs_syllabic_db: Breathing-band over syllabic-band modulation energy, in dB.
        active_span_s: The span the broadband envelope stays above its floor by the active rise.
    """

    breath_vs_syllabic_db: float
    active_span_s: float

    def record(self) -> dict[str, Any]:
        """The reading, as JSON-ready values.

        Returns:
            The fields, keyed by name.
        """
        return {
            "breath_vs_syllabic_db": self.breath_vs_syllabic_db,
            "active_span_s": self.active_span_s,
        }


EXTENT_BREATH_TRAIN = "breath_train"
EXTENT_MEASURE_EVENTS = "measure_events"
EXTENT_AIRWAY_EVENTS = "airway_events"
EXTENT_SOURCES = (EXTENT_BREATH_TRAIN, EXTENT_MEASURE_EVENTS, EXTENT_AIRWAY_EVENTS)


@dataclass(frozen=True)
class TrainParameters:
    """The ``train`` section of ``data/breath_pattern.yaml``.

    Attributes:
        band_edges_hz: Edges of the subbands whose envelopes are read.
        fs_mod: The rate the envelopes are resampled to, in Hz.
        smooth_s: The moving-median window on each subband level.
        noise_percentile: The per-bin percentile taken as stationary noise.
        noise_subtract: The share of that noise subtracted from each bin.
        tonal_db: How far a bin's noise sits over its neighbours' for the bin to be a tonal line.
        tonal_halfwidth_bins: The neighbours on each side that comparison reads.
        envelope_smooth_s: The moving-mean window on the combined robust-z envelope.
        prominence_min: The least prominence a burst needs, in robust-z units.
        prominence_range_frac: Or this share of the envelope's 5th-95th percentile range, if larger.
        weak_prominence_frac: The share of that prominence a weak burst needs.
        template_corr_min: The correlation with the strong bursts' spectrum a weak burst needs.
        distance_quick_s: The least spacing between bursts for a quick-breath family.
        distance_s: The least spacing for every other family.
        quick_families: The quick-breath families.
        burst_s: The duration range of one burst.
        flatness_min: The mean spectral flatness a burst needs.
        voiced_max: The share of a burst's frames that may be voiced.
        coherence_min: The share of subbands that must rise with the burst.
        coherence_window_s: The window either side of the peak a subband's rise is read in.
        rise_z: The rise, in robust-z units, that counts a subband as rising.
        floor_percentile: The percentile of the raw broadband level taken as the floor.
        rise_floor_db: How far over that floor a burst's peak must be.
        gap_min_s: Bursts further apart than this, or ``gap_cycles`` half-cycles, start a new run.
        gap_cycles: The half-cycles a gap may span.
        speech_words_min: The lexical words a speech run holds; a speech run splits the train.
        speech_gap_s: The longest gap between words of one speech run.
        pad_s: Padding on each side of the train's hull.
        continue_db: The median level over the floor the extent runs on through to a speech onset.
    """

    band_edges_hz: tuple[float, ...]
    fs_mod: float
    smooth_s: float
    noise_percentile: float
    noise_subtract: float
    tonal_db: float
    tonal_halfwidth_bins: int
    envelope_smooth_s: float
    prominence_min: float
    prominence_range_frac: float
    weak_prominence_frac: float
    template_corr_min: float
    distance_quick_s: float
    distance_s: float
    quick_families: tuple[str, ...]
    burst_s: tuple[float, float]
    flatness_min: float
    voiced_max: float
    coherence_min: float
    coherence_window_s: float
    rise_z: float
    floor_percentile: float
    rise_floor_db: float
    gap_min_s: float
    gap_cycles: float
    speech_words_min: int
    speech_gap_s: float
    pad_s: float
    continue_db: float


@functools.cache
def train_parameters() -> TrainParameters:
    """The ``train`` parameters of ``data/breath_pattern.yaml``.

    Returns:
        The parameters.
    """
    held = (yaml.safe_load(BREATH_PATTERN_PATH.read_text()) or {})["train"]
    ints = {"tonal_halfwidth_bins", "speech_words_min"}
    values: dict[str, Any] = {}
    for name, value in held.items():
        if name == "band_edges_hz":
            values[name] = tuple(float(edge) for edge in value)
        elif name == "quick_families":
            values[name] = tuple(str(family) for family in value)
        elif name == "burst_s":
            values[name] = (float(value[0]), float(value[1]))
        else:
            values[name] = int(value) if name in ints else float(value)
    return TrainParameters(**values)


@dataclass(frozen=True)
class ReviewParameters:
    """The ``review`` section of ``data/breath_pattern.yaml``.

    Attributes:
        min_phases: A train of fewer phases is in the review band.
        cycle_cv_min: A cycle coefficient of variation at or above which, with a median rise under
            ``irregular_rise_db_max``, the train is in the band.
        irregular_rise_db_max: That rise.
        weak_rise_db_max: A median burst rise under this puts the train in the band.
    """

    min_phases: int
    cycle_cv_min: float
    irregular_rise_db_max: float
    weak_rise_db_max: float


@functools.cache
def review_parameters() -> ReviewParameters:
    """The ``review`` parameters of ``data/breath_pattern.yaml``.

    Returns:
        The parameters.
    """
    held = (yaml.safe_load(BREATH_PATTERN_PATH.read_text()) or {})["review"]
    return ReviewParameters(
        min_phases=int(held["min_phases"]),
        cycle_cv_min=float(held["cycle_cv_min"]),
        irregular_rise_db_max=float(held["irregular_rise_db_max"]),
        weak_rise_db_max=float(held["weak_rise_db_max"]),
    )


@dataclass(frozen=True)
class Burst:
    """One breath phase (an inhale or an exhale) the breath train found.

    Attributes:
        peak_s: Its envelope peak, in seconds.
        start_s: Its start.
        end_s: Its end.
        prominence: Its prominence on the combined envelope, in robust-z units.
        coherence: The share of subbands that rose with it.
        rise_db: Its peak's raw broadband level over the recording's floor.
        template_corr: Its spectrum's correlation with the strong bursts' mean spectrum.
        weak: Whether it stood only on the template.
    """

    peak_s: float
    start_s: float
    end_s: float
    prominence: float
    coherence: float
    rise_db: float
    template_corr: float | None = None
    weak: bool = False


@dataclass(frozen=True)
class BreathTrain:
    """The run of bursts that is the breathing task.

    Attributes:
        bursts: The run's bursts, in time order.
        extent_s: The run's hull, padded, trimmed at speech and run on to a following speech onset;
            None with no burst.
        phases: The bursts counted.
        breaths: ``phases`` over two, rounded half up.
        rate_cpm: Breathing cycles per minute, from the median two-phase interval; None under three
            phases.
        cycle_cv: The two-phase intervals' coefficient of variation; None under four phases.
        coherence: The bursts' mean coherence.
        rise_db: The bursts' median rise over the floor.
        bursts_found_n: Every burst found in the file, inside the run or not.
    """

    bursts: tuple[Burst, ...] = ()
    extent_s: tuple[float, float] | None = None
    phases: int = 0
    breaths: int = 0
    rate_cpm: float | None = None
    cycle_cv: float | None = None
    coherence: float | None = None
    rise_db: float | None = None
    bursts_found_n: int = 0

    def record(self) -> dict[str, Any]:
        """The train, as JSON-ready values.

        Returns:
            The fields, keyed by name, with each burst's peak, start and end.
        """
        return {
            "extent_s": list(self.extent_s) if self.extent_s is not None else None,
            "phases": self.phases,
            "breaths": self.breaths,
            "rate_cpm": self.rate_cpm,
            "cycle_cv": self.cycle_cv,
            "coherence": self.coherence,
            "rise_db": self.rise_db,
            "bursts_found_n": self.bursts_found_n,
            "bursts_s": [[b.peak_s, b.start_s, b.end_s] for b in self.bursts],
        }


def in_review_band(train: BreathTrain | None, parameters: ReviewParameters | None = None) -> bool:
    """Whether a breath train is weak or irregular enough to be left for review.

    Args:
        train: The breath train, or None where none was read.
        parameters: The band; ``data/breath_pattern.yaml`` when None.

    Returns:
        True for no train, fewer than ``min_phases`` phases, an irregular cycle with a modest rise, or a
        weak rise.
    """
    p = parameters or review_parameters()
    if train is None or train.rise_db is None or train.phases < p.min_phases:
        return True
    irregular = train.cycle_cv is not None and train.cycle_cv >= p.cycle_cv_min
    return (irregular and train.rise_db < p.irregular_rise_db_max) or train.rise_db < p.weak_rise_db_max


@dataclass(frozen=True)
class BreathExtent:
    """Where in the recording the breathing task was performed.

    Attributes:
        start_s: The extent's start, in seconds.
        end_s: Its end, in seconds.
        source: Where it was read from, one of :data:`EXTENT_SOURCES`.
        phases: The breath train's phases inside the extent; None for a fallback.
        breaths: Those phases over two, rounded half up; None for a fallback.
    """

    start_s: float
    end_s: float
    source: str
    phases: int | None = None
    breaths: int | None = None

    @property
    def bounds(self) -> tuple[float, float]:
        """``(start_s, end_s)``."""
        return self.start_s, self.end_s

    def record(self) -> dict[str, Any]:
        """The extent, as JSON-ready values.

        Returns:
            The fields, keyed by name.
        """
        return {
            "start_s": self.start_s,
            "end_s": self.end_s,
            "source": self.source,
            "phases": self.phases,
            "breaths": self.breaths,
        }


@dataclass(frozen=True)
class VetoParameters:
    """The ``veto`` section of ``data/breath_pattern.yaml``.

    Attributes:
        speech_words_min: Lexical words inside the task extent at or above which the recording is
            speech (:func:`speech_words`).
        active_fraction_min: The active span over the span it was read on, below which there is too
            little active sound.
        noise_labels: The YAMNet labels whose highest score is recorded as background noise, for
            context only.
    """

    speech_words_min: int
    active_fraction_min: float
    noise_labels: tuple[str, ...]


@functools.cache
def veto_parameters() -> VetoParameters:
    """The ``veto`` parameters of ``data/breath_pattern.yaml``.

    Returns:
        The parameters.
    """
    held = (yaml.safe_load(BREATH_PATTERN_PATH.read_text()) or {})["veto"]
    return VetoParameters(
        speech_words_min=int(held["speech_words_min"]),
        active_fraction_min=float(held["active_fraction_min"]),
        noise_labels=tuple(str(label) for label in held["noise_labels"]),
    )


VETO_SPEECH = "speech"
VETO_LITTLE_ACTIVITY = "little_activity"
OVER_TASK_EXTENT = "task_extent"
OVER_FILE = "file"
"""What the active fraction was read over: the task extent, or the whole file where the extent is too
short for the modulation reading or there is none."""


@dataclass(frozen=True)
class BreathVeto:
    """What says a breath the measure found is not breathing, and the readings it was decided on.

    Attributes:
        lexical_words_n: Lexical words inside the task extent (:func:`speech_words`).
        active_fraction: The modulation reading's active span over the span it was read on, or None
            where no modulation reading could be read.
        active_over: :data:`OVER_TASK_EXTENT` or :data:`OVER_FILE`, or None with no active fraction.
        task_extent_s: The task extent, ``(start, end)``, or None where the store holds none.
        speech_mean: The mean YAMNet Speech score over the task extent's windows, for context only.
        silence_mean: The mean YAMNet Silence score there, for context only.
        noise_mean: The mean of the highest noise-label score there, for context only.
        breathing_mean: The mean YAMNet Breathing score there, for context only.
        hear_breathe_max: The highest HeAR Breathe window there, for context only.
        vetoed_by: The first veto that fired, one of ``VETO_*``, or None.
    """

    lexical_words_n: int
    active_fraction: float | None
    active_over: str | None
    task_extent_s: tuple[float, float] | None
    speech_mean: float | None
    silence_mean: float | None
    noise_mean: float | None
    breathing_mean: float | None
    hear_breathe_max: float | None
    vetoed_by: str | None

    def record(self) -> dict[str, Any]:
        """The reading, as JSON-ready values.

        Returns:
            The fields, keyed by name.
        """
        return {
            "lexical_words_n": self.lexical_words_n,
            "active_fraction": self.active_fraction,
            "active_over": self.active_over,
            "task_extent_s": list(self.task_extent_s) if self.task_extent_s is not None else None,
            "speech_mean": self.speech_mean,
            "silence_mean": self.silence_mean,
            "noise_mean": self.noise_mean,
            "breathing_mean": self.breathing_mean,
            "hear_breathe_max": self.hear_breathe_max,
            "vetoed_by": self.vetoed_by,
        }


@dataclass(frozen=True)
class BreathPattern:
    """What :func:`measure_breath_pattern` read.

    Attributes:
        pattern: One of :data:`PATTERNS`.
        events_n: How many breath events it found.
        event_durations_s: Each event's duration.
        intervals_s: Onset-to-onset intervals between consecutive events.
        autocorr_peak: The envelope's autocorrelation peak over the breathing-period range, or None
            where the recording is shorter than that range.
        rhythm: Whether the events repeat at a breathing rhythm.
        modulation: The modulation reading, or None where the recording is shorter than it needs.
        veto: What the stored classifier windows and transcript say against breathing, or None where it
            was not read.
        event_spans_s: Each event's ``(start, end)`` in seconds, trimmed to the frames near its peak.
        extent: The breath-task extent, or None where it was not read.
        train: The breath train, or None where it was not read.
    """

    pattern: str
    events_n: int
    event_durations_s: tuple[float, ...] = field(default_factory=tuple)
    intervals_s: tuple[float, ...] = field(default_factory=tuple)
    autocorr_peak: float | None = None
    rhythm: bool = False
    modulation: ModulationReading | None = None
    veto: BreathVeto | None = None
    event_spans_s: tuple[tuple[float, float], ...] = field(default_factory=tuple)
    extent: BreathExtent | None = None
    train: BreathTrain | None = None

    def record(self) -> dict[str, Any]:
        """The reading, as JSON-ready values.

        Returns:
            The fields, keyed by name.
        """
        return {
            "pattern": self.pattern,
            "events_n": self.events_n,
            "event_durations_s": list(self.event_durations_s),
            "intervals_s": list(self.intervals_s),
            "autocorr_peak": self.autocorr_peak,
            "rhythm": self.rhythm,
            "modulation": self.modulation.record() if self.modulation is not None else None,
            "veto": self.veto.record() if self.veto is not None else None,
            "extent": self.extent.record() if self.extent is not None else None,
            "train": self.train.record() if self.train is not None else None,
        }


def _band_rows(bin_hz: float, band: tuple[float, float]) -> slice:
    return slice(int(np.ceil(band[0] / bin_hz)), int(np.floor(band[1] / bin_hz)) + 1)


def _moving_median(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or len(values) < width:
        return values
    half = width // 2
    padded = np.pad(values, half, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, width), axis=1)[: len(values)]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def measure_breath_pattern(
    power: np.ndarray,
    *,
    hop_s: float,
    bin_hz: float,
    voiced: np.ndarray,
    parameters: BreathPatternParameters,
    modulation: ModulationParameters | None = None,
) -> BreathPattern:
    """Read the breathing pattern off a power spectrogram and a per-frame voicing mask.

    Args:
        power: Power spectrogram, frequency bins by frames, bin 0 at 0 Hz.
        hop_s: Seconds between frames.
        bin_hz: Hertz between frequency bins.
        voiced: Whether each frame is voiced, one per frame.
        parameters: The measure's parameters.
        modulation: The modulation reading's parameters; None skips that reading.

    Returns:
        The pattern and what it was read from.
    """
    p = parameters
    frames = power.shape[1]
    if frames == 0:
        return BreathPattern(pattern=NO_BREATHING, events_n=0)
    tiny = 1e-12
    envelope = 10 * np.log10(power[_band_rows(bin_hz, p.breath_band_hz)].sum(axis=0) + tiny)
    envelope = _moving_median(envelope, max(1, int(round(p.smooth_s / hop_s))) | 1)
    band = power[_band_rows(bin_hz, p.flatness_band_hz)] + tiny
    flatness = np.exp(np.mean(np.log(band), axis=0)) / np.mean(band, axis=0)
    floor = float(np.percentile(envelope, p.floor_percentile))
    active = (envelope >= floor + p.rise_db) & (flatness >= p.flatness_min)

    gap = int(round(p.merge_gap_s / hop_s))
    merged: list[list[int]] = []
    for start, stop in _runs(active):
        if merged and start - merged[-1][1] <= gap:
            merged[-1][1] = stop
        else:
            merged.append([start, stop])
    events: list[tuple[int, float]] = []
    spans: list[tuple[float, float]] = []
    for start, stop in merged:
        duration = (stop - start) * hop_s
        voiced_share = float(voiced[start:stop].mean()) if stop > start else 0.0
        if p.event_s[0] <= duration <= p.event_s[1] and voiced_share <= p.voiced_fraction_max:
            events.append((start, round(duration, 3)))
            near = np.flatnonzero(envelope[start:stop] >= envelope[start:stop].max() - p.event_trim_db)
            spans.append((round((start + near[0]) * hop_s, 3), round((start + near[-1] + 1) * hop_s, 3)))

    onsets = np.array([start * hop_s for start, _ in events])
    intervals = np.diff(onsets) if len(onsets) > 1 else np.array([])
    centred = np.clip(envelope - floor, 0, None)
    centred = centred - centred.mean()
    autocorr_peak: float | None = None
    lag_low, lag_high = int(p.period_s[0] / hop_s), int(p.period_s[1] / hop_s)
    if frames > lag_high and np.any(centred):
        spectrum = np.fft.rfft(centred, 2 * frames)
        autocorr = np.fft.irfft(spectrum * np.conj(spectrum))[:frames]
        if autocorr[0] > 0:
            autocorr_peak = round(float((autocorr / autocorr[0])[lag_low:lag_high].max()), 3)
    in_range = [p.interval_s[0] <= value <= p.interval_s[1] for value in intervals]
    rhythm = (autocorr_peak is not None and autocorr_peak >= p.autocorr_min) or (
        len(in_range) > 0 and float(np.mean(in_range)) >= p.interval_share_min
    )
    count = len(events)
    if count == 0:
        pattern = NO_BREATHING
    elif count == 1:
        pattern = SINGLE_BREATH
    else:
        pattern = ALTERNATING_BREATHS if rhythm else IRREGULAR_EVENTS
    return BreathPattern(
        pattern=pattern,
        events_n=count,
        event_durations_s=tuple(duration for _, duration in events),
        intervals_s=tuple(round(float(value), 2) for value in intervals),
        autocorr_peak=autocorr_peak,
        rhythm=bool(rhythm),
        event_spans_s=tuple(spans),
        modulation=measure_modulation(power, hop_s=hop_s, bin_hz=bin_hz, smooth_s=p.smooth_s, parameters=modulation)
        if modulation is not None
        else None,
    )


def _modulation_psd(values: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    index = np.arange(len(values))
    detrended = values - np.polyval(np.polyfit(index, values, 1), index)
    nfft = 1 << int(np.ceil(np.log2(max(len(values) * 8, 256))))
    spectrum = np.abs(np.fft.rfft(detrended * np.hanning(len(values)), nfft)) ** 2
    return np.fft.rfftfreq(nfft, 1 / fs), spectrum


def _subband_envelopes(
    power: np.ndarray, *, hop_s: float, bin_hz: float, smooth_s: float, parameters: ModulationParameters
) -> np.ndarray:
    """Each subband's dB envelope, smoothed and resampled to ``fs_mod``: bands by envelope frames."""
    p = parameters
    step = max(1, int(round(1 / (p.fs_mod * hop_s))))
    frames = power.shape[1] // step
    width = max(1, int(round(smooth_s / hop_s))) | 1
    envelopes = []
    for low, high in zip(p.band_edges_hz[:-1], p.band_edges_hz[1:]):
        level = 10 * np.log10(power[_band_rows(bin_hz, (low, high))].sum(axis=0) + 1e-12)
        level = _moving_median(level, width)
        envelopes.append(level[: frames * step].reshape(frames, step).mean(axis=1))
    return np.array(envelopes).reshape(len(envelopes), frames)


def _breath_ratio_db(bands: np.ndarray, parameters: ModulationParameters) -> float:
    """Breathing-band over syllabic-band modulation energy, in dB, averaged over the subbands."""
    p = parameters
    tiny = 1e-12
    spectra = []
    for envelope in bands:
        freqs, spectrum = _modulation_psd(envelope, p.fs_mod)
        reference = (freqs >= p.reference_band_hz[0]) & (freqs <= p.reference_band_hz[1])
        spectra.append(spectrum / (spectrum[reference].sum() + tiny))
    mean_spectrum = np.mean(spectra, axis=0)
    breath = (freqs >= p.breath_band_hz[0]) & (freqs <= p.breath_band_hz[1])
    syllabic = (freqs >= p.syllabic_band_hz[0]) & (freqs <= p.syllabic_band_hz[1])
    return float(10 * np.log10((mean_spectrum[breath].sum() + tiny) / (mean_spectrum[syllabic].sum() + tiny)))


def measure_modulation(
    power: np.ndarray, *, hop_s: float, bin_hz: float, smooth_s: float, parameters: ModulationParameters
) -> ModulationReading | None:
    """Read the breathing-over-syllabic modulation ratio and the active span off subband envelopes.

    Args:
        power: Power spectrogram, frequency bins by frames, bin 0 at 0 Hz.
        hop_s: Seconds between frames.
        bin_hz: Hertz between frequency bins.
        smooth_s: The moving-median window on each subband envelope.
        parameters: The reading's parameters.

    Returns:
        The reading, or None where the recording is shorter than ``min_duration_s``.
    """
    p = parameters
    bands = _subband_envelopes(power, hop_s=hop_s, bin_hz=bin_hz, smooth_s=smooth_s, parameters=p)
    if bands.shape[1] / p.fs_mod < p.min_duration_s:
        return None
    ratio_db = _breath_ratio_db(bands, p)

    broadband = 10 * np.log10(np.sum(10 ** (bands / 10), axis=0))
    floor = float(np.percentile(broadband, p.floor_percentile))
    active = np.flatnonzero(broadband >= floor + p.active_rise_db)
    span_s = float((active[-1] - active[0] + 1) / p.fs_mod) if len(active) else 0.0
    return ModulationReading(
        breath_vs_syllabic_db=round(ratio_db, 2),
        active_span_s=round(span_s, 2),
    )


def _denoise(power: np.ndarray, p: TrainParameters) -> tuple[np.ndarray, np.ndarray]:
    """The spectrogram less its per-bin stationary noise, and which bins are tonal lines."""
    noise = np.percentile(power, p.noise_percentile, axis=1)
    noise_db = 10 * np.log10(noise + 1e-12)
    half = p.tonal_halfwidth_bins
    padded = np.pad(noise_db, half, mode="edge")
    neighbours = np.median(np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1), axis=1)
    tonal = noise_db - neighbours > p.tonal_db
    clean = np.maximum(power - p.noise_subtract * noise[:, None], 1e-3 * noise[:, None] + 1e-12)
    return clean, tonal


def _subband_levels(
    power: np.ndarray, tonal: np.ndarray, *, hop_s: float, bin_hz: float, p: TrainParameters
) -> np.ndarray:
    """Each subband's dB level less its tonal bins, smoothed and resampled to ``fs_mod``."""
    step = max(1, int(round(1 / (p.fs_mod * hop_s))))
    frames = power.shape[1] // step
    width = max(1, int(round(p.smooth_s / hop_s))) | 1
    levels = []
    for low, high in zip(p.band_edges_hz[:-1], p.band_edges_hz[1:]):
        rows = np.arange(int(np.ceil(low / bin_hz)), min(power.shape[0], int(np.floor(high / bin_hz)) + 1))
        if len(rows) == 0:
            continue
        rows = rows[~tonal[rows]] if (~tonal[rows]).any() else rows
        level = _moving_median(10 * np.log10(power[rows].sum(axis=0) + 1e-12), width)
        levels.append(level[: frames * step].reshape(frames, step).mean(axis=1))
    return np.array(levels).reshape(len(levels), frames)


def _robust_z(levels: np.ndarray) -> np.ndarray:
    median = np.median(levels, axis=1, keepdims=True)
    mad = np.median(np.abs(levels - median), axis=1, keepdims=True) * 1.4826 + 1e-6
    return (levels - median) / mad


def find_bursts(
    power: np.ndarray,
    *,
    hop_s: float,
    bin_hz: float,
    voiced: np.ndarray,
    family: str | None,
    parameters: TrainParameters,
) -> tuple[list[Burst], np.ndarray, float]:
    """Find breath phases: coherent broadband rises on the denoised subband envelopes.

    Args:
        power: Power spectrogram, frequency bins by frames, bin 0 at 0 Hz.
        hop_s: Seconds between frames.
        bin_hz: Hertz between frequency bins.
        voiced: Whether each frame is voiced, one per frame.
        family: The declared task family, which fixes the least spacing between bursts.
        parameters: The train's parameters.

    Returns:
        The bursts in time order, the raw broadband level at ``fs_mod``, and its floor.
    """
    p = parameters
    fs = p.fs_mod
    clean, tonal = _denoise(power, p)
    z = _robust_z(_subband_levels(clean, tonal, hop_s=hop_s, bin_hz=bin_hz, p=p))
    raw = _subband_levels(power, np.zeros(power.shape[0], dtype=bool), hop_s=hop_s, bin_hz=bin_hz, p=p)
    frames = z.shape[1]
    if frames < 3 or z.shape[0] == 0:
        return [], np.zeros(frames), 0.0
    broadband = 10 * np.log10(np.sum(10 ** (raw / 10), axis=0))
    floor = float(np.percentile(broadband, p.floor_percentile))
    k = max(1, int(round(p.envelope_smooth_s * fs)))
    combined = np.convolve(np.pad(z.mean(axis=0), k, mode="reflect"), np.ones(k) / k, mode="same")[k:-k]
    full = max(
        p.prominence_min, p.prominence_range_frac * float(np.percentile(combined, 95) - np.percentile(combined, 5))
    )
    distance = p.distance_quick_s if family in p.quick_families else p.distance_s
    peaks, props = find_peaks(
        combined, distance=max(1, int(distance * fs)), prominence=full * p.weak_prominence_frac, width=1
    )
    step = max(1, int(round(1 / (fs * hop_s))))
    band = clean[_band_rows(bin_hz, (p.band_edges_hz[0], p.band_edges_hz[-1]))][:, : frames * step] + 1e-12
    flatness = (np.exp(np.mean(np.log(band), axis=0)) / np.mean(band, axis=0)).reshape(frames, step).mean(axis=1)
    held = voiced[: frames * step]
    voicing = held.reshape(frames, step).mean(axis=1) if len(held) == frames * step else np.zeros(frames)
    spectrum = 10 * np.log10(band.reshape(band.shape[0], frames, step).mean(axis=2))
    half = int(round(p.coherence_window_s * fs))
    found: list[Burst] = []
    for index, peak in enumerate(peaks):
        low, high = int(props["left_ips"][index]), int(np.ceil(props["right_ips"][index]))
        if not p.burst_s[0] <= (high - low) / fs <= p.burst_s[1]:
            continue
        if (
            float(flatness[low : high + 1].mean()) < p.flatness_min
            or float(voicing[low : high + 1].mean()) > p.voiced_max
        ):
            continue
        near, before = slice(max(0, peak - half), min(frames, peak + half + 1)), slice(max(0, low - half), low + 1)
        coherence = float(np.mean([(row[near].max() - row[before].min()) >= p.rise_z for row in z]))
        rise = float(broadband[peak] - floor)
        if coherence < p.coherence_min or rise < p.rise_floor_db:
            continue
        prominence = float(props["prominences"][index])
        found.append(
            Burst(
                peak_s=round(peak / fs, 3),
                start_s=round(low / fs, 3),
                end_s=round(high / fs, 3),
                prominence=round(prominence, 3),
                coherence=round(coherence, 3),
                rise_db=round(rise, 2),
                weak=prominence < full,
            )
        )
    strong = [burst for burst in found if not burst.weak]
    if not strong:
        return [], broadband, floor

    def shape(burst: Burst) -> np.ndarray:
        mean = spectrum[:, int(round(burst.start_s * fs)) : int(round(burst.end_s * fs)) + 1].mean(axis=1)
        return (mean - mean.mean()) / (mean.std() + 1e-9)

    template = np.mean([shape(burst) for burst in strong], axis=0)
    scored = [replace(burst, template_corr=round(float(np.mean(shape(burst) * template)), 3)) for burst in found]
    kept = [b for b in scored if not b.weak or (b.template_corr or 0.0) >= p.template_corr_min]
    return kept, broadband, floor


def speech_runs(words: tuple[tuple[float, float], ...], *, words_min: int, gap_s: float) -> list[tuple[float, float]]:
    """Runs of lexical words, split at gaps over ``gap_s``, that hold at least ``words_min`` words.

    Args:
        words: Each lexical word's ``(start, end)``.
        words_min: The fewest words a run holds.
        gap_s: The longest gap inside a run.

    Returns:
        Each run's ``(start, end)``, in time order.
    """
    runs: list[list[tuple[float, float]]] = []
    for word in sorted(words):
        if runs and word[0] - runs[-1][-1][1] <= gap_s:
            runs[-1].append(word)
        else:
            runs.append([word])
    return [(run[0][0], max(end for _, end in run)) for run in runs if len(run) >= words_min]


def _rhu(value: float) -> int:
    return int(np.floor(value + 0.5))


def breath_train(
    bursts: list[Burst],
    words: tuple[tuple[float, float], ...],
    *,
    duration_s: float,
    broadband: np.ndarray,
    floor: float,
    parameters: TrainParameters,
) -> BreathTrain:
    """Group bursts into runs split at long gaps or speech, and take the largest as the task.

    Args:
        bursts: The bursts, in time order (:func:`find_bursts`).
        words: Each lexical word's ``(start, end)``.
        duration_s: The recording's duration.
        broadband: The raw broadband level at ``fs_mod``.
        floor: Its floor.
        parameters: The train's parameters.

    Returns:
        The train; an empty one with no burst.
    """
    p = parameters
    if not bursts:
        return BreathTrain()
    speech = speech_runs(words, words_min=p.speech_words_min, gap_s=p.speech_gap_s)
    gaps = np.diff([b.peak_s for b in bursts])
    cycle = float(np.median(gaps[:-1] + gaps[1:])) if len(gaps) >= 2 else (2 * float(gaps[0]) if len(gaps) else 4.0)
    gap_max = max(p.gap_min_s, p.gap_cycles * cycle / 2)
    runs = [[bursts[0]]]
    for previous, burst in zip(bursts, bursts[1:]):
        split = any(s < burst.start_s and e > previous.end_s for s, e in speech)
        if split or burst.peak_s - previous.peak_s > gap_max:
            runs.append([burst])
        else:
            runs[-1].append(burst)
    run = max(runs, key=lambda r: (len(r), -r[0].start_s))
    start, end = max(0.0, run[0].start_s - p.pad_s), min(duration_s, run[-1].end_s + p.pad_s)
    for s, e in speech:
        if start < s < end and s >= run[-1].peak_s:
            end = s
        if start < e < end and e <= run[0].peak_s:
            start = e
    following = [s for s, _ in speech if s >= end]
    if following and following[0] - end <= gap_max:
        stretch = broadband[int(end * p.fs_mod) : int(following[0] * p.fs_mod)]
        if len(stretch) and float(np.median(stretch)) >= floor + p.continue_db:
            end = following[0]
    gaps = np.diff([b.peak_s for b in run])
    cycles = gaps[:-1] + gaps[1:] if len(gaps) >= 2 else np.array([])
    return BreathTrain(
        bursts=tuple(run),
        extent_s=(round(start, 3), round(end, 3)),
        phases=len(run),
        breaths=_rhu(len(run) / 2),
        rate_cpm=round(60.0 / float(np.median(cycles)), 1) if len(cycles) else None,
        cycle_cv=round(float(np.std(cycles) / np.mean(cycles)), 2) if len(cycles) >= 2 else None,
        coherence=round(float(np.mean([b.coherence for b in run])), 2),
        rise_db=round(float(np.median([b.rise_db for b in run])), 1),
        bursts_found_n=len(bursts),
    )


def measure_breath_train(
    power: np.ndarray,
    *,
    hop_s: float,
    bin_hz: float,
    voiced: np.ndarray,
    words: tuple[tuple[float, float], ...] = (),
    family: str | None = None,
    parameters: TrainParameters | None = None,
) -> BreathTrain:
    """The breath train of a power spectrogram (:func:`find_bursts`, then :func:`breath_train`).

    Args:
        power: Power spectrogram, frequency bins by frames, bin 0 at 0 Hz.
        hop_s: Seconds between frames.
        bin_hz: Hertz between frequency bins.
        voiced: Whether each frame is voiced, one per frame.
        words: Each lexical word's ``(start, end)``.
        family: The declared task family.
        parameters: The train's parameters; ``data/breath_pattern.yaml`` when None.

    Returns:
        The train.
    """
    p = parameters or train_parameters()
    bursts, broadband, floor = find_bursts(
        power, hop_s=hop_s, bin_hz=bin_hz, voiced=voiced, family=family, parameters=p
    )
    return breath_train(
        bursts, words, duration_s=power.shape[1] * hop_s, broadband=broadband, floor=floor, parameters=p
    )


def train_extent(train: BreathTrain, events: tuple[tuple[float, float], ...]) -> BreathExtent:
    """The breath train's extent, widened to hold every measure event that overlaps it.

    Args:
        train: The breath train; its ``extent_s`` is set.
        events: The breathing measure's event spans.

    Returns:
        The extent, sourced :data:`EXTENT_BREATH_TRAIN`.
    """
    assert train.extent_s is not None
    start, end = train.extent_s
    overlapping = [e for e in events if e[1] > start and e[0] < end]
    return BreathExtent(
        start_s=round(min([start, *(e[0] for e in overlapping)]), 3),
        end_s=round(max([end, *(e[1] for e in overlapping)]), 3),
        source=EXTENT_BREATH_TRAIN,
        phases=train.phases,
        breaths=train.breaths,
    )


def breath_extent_fallback(
    events: tuple[tuple[float, float], ...], airway: tuple[float, float] | None, *, duration_s: float, pad_s: float
) -> BreathExtent | None:
    """The extent where the breath train gives none: the measure's events, padded, else AIRWAY's hull.

    Args:
        events: The breathing measure's event spans.
        airway: The hull of AIRWAY's own task-extent spans, or None.
        duration_s: The recording's duration.
        pad_s: Padding added on each side of the events' hull.

    Returns:
        The extent, or None where neither gives one.
    """
    if events:
        start, end = min(e[0] for e in events), max(e[1] for e in events)
        return BreathExtent(
            start_s=round(max(0.0, start - pad_s), 3),
            end_s=round(min(duration_s, end + pad_s), 3),
            source=EXTENT_MEASURE_EVENTS,
        )
    if airway is not None and airway[1] > airway[0]:
        return BreathExtent(start_s=airway[0], end_s=airway[1], source=EXTENT_AIRWAY_EVENTS)
    return None


def _sidecar(store: ProvStore, run_dir: Path, name: str) -> tuple[Mapping[str, Any], Path] | None:
    measurement = find_measurement(store, name)
    if measurement is None:
        return None
    relative = measurement.attributes.get("path")
    if not relative:
        return None
    path = run_dir / str(relative)
    return (measurement.attributes, path) if path.is_file() else None


def _window_scores(
    store: ProvStore, run_dir: Path, name: str, extent: tuple[float, float] | None = None
) -> list[dict[str, float]] | None:
    held = _sidecar(store, run_dir, name)
    if held is None:
        return None
    return [
        {key: float(score) for pair in label_scores(window) for key, score in pair.items()}
        for window in json.loads(held[1].read_text())
        if extent is None or (float(window.get("end", 0.0)) > extent[0] and float(window.get("start", 0.0)) < extent[1])
    ]


def _mean(rows: list[dict[str, float]] | None, label: str) -> float | None:
    return round(float(np.mean([row.get(label, 0.0) for row in rows])), 4) if rows else None


def task_extent_bounds(store: ProvStore) -> tuple[float, float] | None:
    """The hull of the store's live task-extent spans that the branches wrote.

    A span VERDICT wrote to supersede them (it carries ``supersedes``) is left out.

    Args:
        store: The provenance store.

    Returns:
        ``(start, end)`` in seconds, or None where no such live span carries the task-extent role.
    """
    spans = [
        span.extent
        for span in live_entities(store, "span")
        if span.attributes.get("role") == TASK_EXTENT_ROLE
        and span.extent is not None
        and span.attributes.get(SUPERSEDES) is None
    ]
    if not spans:
        return None
    return min(float(span[0]) for span in spans), max(float(span[1]) for span in spans)


def _in_script(text: str, language: str | None) -> bool:
    if language is None or not any(language.lower().startswith(code) for code in LATIN_LANGUAGES):
        return any(ch.isalpha() for ch in text)
    return any(ch.isalpha() and "LATIN" in unicodedata.name(ch, "") for ch in text)


def speech_words(store: ProvStore, extent: tuple[float, float] | None, language: str | None) -> int:
    """The consensus words that are speech, inside the task extent.

    Args:
        store: The provenance store.
        extent: The task extent; None counts the whole file.
        language: The recording's declared language, or None.

    Returns:
        The words that are not bracketed, not a vocalisation or interjection
        (:func:`~senselab.audio.workflows.triage.residue.is_non_lexical`), written in the declared
        language's script, and timed inside the extent.
    """
    count = 0
    for word in lexical_words(store):
        text = str(word.attributes.get("text") or "")
        if is_non_lexical(text) or not _in_script(text, language):
            continue
        start, end = word_hull(word)
        if extent is None or (end > extent[0] and start < extent[1]):
            count += 1
    return count


def breath_veto_of(
    store: ProvStore,
    run_dir: Path,
    *,
    active_fraction: float | None,
    active_over: str | None,
    extent: tuple[float, float] | None,
    language: str | None = None,
    parameters: VetoParameters | None = None,
) -> BreathVeto:
    """Whether the recording shows positive evidence that what the measure found is not breathing.

    Args:
        store: The provenance store, read for the consensus words and the ``yamnet_scores`` and
            ``hear_scores`` measurements.
        run_dir: The run directory their sidecar paths are relative to.
        active_fraction: The active span over the span it was read on, or None where none was read.
        active_over: What ``active_fraction`` was read over, :data:`OVER_TASK_EXTENT` or
            :data:`OVER_FILE`.
        extent: The task extent, or None.
        language: The recording's declared language, or None.
        parameters: The veto parameters; ``data/breath_pattern.yaml`` when None.

    Returns:
        The veto reading. Speech words inside the extent, then too little activity, are the vetoes;
        the classifier scores over the extent are recorded for context and decide nothing.
    """
    p = parameters or veto_parameters()
    yamnet = _window_scores(store, run_dir, YAMNET_SCORES, extent)
    hear = _window_scores(store, run_dir, HEAR_SCORES, extent)
    words = speech_words(store, extent, language)
    vetoed_by: str | None = None
    if words >= p.speech_words_min:
        vetoed_by = VETO_SPEECH
    elif active_fraction is not None and active_fraction < p.active_fraction_min:
        vetoed_by = VETO_LITTLE_ACTIVITY
    noise = (
        round(float(np.mean([max(row.get(label, 0.0) for label in p.noise_labels) for row in yamnet])), 4)
        if yamnet
        else None
    )
    return BreathVeto(
        lexical_words_n=words,
        active_fraction=active_fraction,
        active_over=active_over if active_fraction is not None else None,
        task_extent_s=extent,
        speech_mean=_mean(yamnet, "Speech"),
        silence_mean=_mean(yamnet, "Silence"),
        noise_mean=noise,
        breathing_mean=_mean(yamnet, "Breathing"),
        hear_breathe_max=round(max(row.get("Breathe", 0.0) for row in hear), 4) if hear else None,
        vetoed_by=vetoed_by,
    )


def breath_pattern_of(
    store: ProvStore,
    run_dir: Path,
    *,
    sampling_hz: float,
    parameters: BreathPatternParameters | None = None,
    modulation: ModulationParameters | None = None,
    language: str | None = None,
    family: str | None = None,
) -> BreathPattern | tuple[str, ...]:
    """The recording's breathing pattern, or the stored inputs it could not be read without.

    The pattern and the breath train (:func:`measure_breath_train`) are read over the whole file. The
    breath-task extent is the train's, widened to hold any of the measure's events that overlap it;
    else the measure's events, else AIRWAY's own hull (:func:`breath_extent_fallback`). It is the
    recording's standing task extent; the veto's readings
    (speech, the classifiers, and the active fraction, the file where the hull is shorter than a
    modulation reading) stay over AIRWAY's own hull, as they were fitted.

    Args:
        store: The provenance store, read for the ``spectrogram_narrowband`` and
            ``phonation_tracks`` measurements and the task-extent spans.
        run_dir: The run directory their sidecar paths are relative to.
        sampling_hz: The conditioned stream's sampling rate, which with the spectrogram's own
            ``n_fft`` and ``hop_length`` fixes its bin width and hop.
        parameters: The measure's parameters; ``data/breath_pattern.yaml`` when None.
        modulation: The modulation reading's parameters; ``data/breath_pattern.yaml`` when None.
        language: The recording's declared language, which fixes the script speech words are read in.
        family: The declared task family, which fixes the breath train's least burst spacing.

    Returns:
        The pattern with its veto reading (:func:`breath_veto_of`); or the names of the absent inputs,
        where either derivative or its sidecar is missing.
    """
    spectrogram = _sidecar(store, run_dir, SPECTROGRAM)
    tracks = _sidecar(store, run_dir, PHONATION_TRACKS)
    absent = tuple(name for name, held in ((SPECTROGRAM, spectrogram), (PHONATION_TRACKS, tracks)) if held is None)
    if spectrogram is None or tracks is None:
        return absent
    attributes, path = spectrogram
    n_fft, hop_length = attributes.get("n_fft"), attributes.get("hop_length")
    if not isinstance(n_fft, int) or not isinstance(hop_length, int) or n_fft <= 0 or hop_length <= 0:
        return (SPECTROGRAM,)
    with np.load(path) as held:
        power = np.asarray(held["spectrogram"], dtype=np.float64)
    if power.ndim == 3:
        power = power[0]
    hop_s = hop_length / sampling_hz
    times = np.arange(power.shape[1]) * hop_s
    voiced = np.zeros(power.shape[1], dtype=bool)
    with np.load(tracks[1]) as held:
        track_times, f0, strength = held["times_s"], held["f0_hz"], held["strength"]
    p = parameters or breath_pattern_parameters()
    if len(track_times):
        frame_voiced = (np.nan_to_num(f0) > 0) & (np.nan_to_num(strength) >= p.voicing_strength_min)
        voiced = frame_voiced[np.clip(np.searchsorted(track_times, times), 0, len(track_times) - 1)]
    m = modulation or modulation_parameters()
    bin_hz = sampling_hz / n_fft
    pattern = measure_breath_pattern(power, hop_s=hop_s, bin_hz=bin_hz, voiced=voiced, parameters=p, modulation=m)
    duration_s = power.shape[1] * hop_s
    t = train_parameters()
    airway = task_extent_bounds(store)
    words = tuple(
        word_hull(word) for word in lexical_words(store) if not is_non_lexical(str(word.attributes.get("text") or ""))
    )
    train = measure_breath_train(
        power, hop_s=hop_s, bin_hz=bin_hz, voiced=voiced, words=words, family=family, parameters=t
    )
    breath_extent = (
        train_extent(train, pattern.event_spans_s)
        if train.extent_s is not None
        else breath_extent_fallback(pattern.event_spans_s, airway, duration_s=duration_s, pad_s=t.pad_s)
    )
    active_fraction: float | None = None
    active_over: str | None = None
    if airway is not None:
        first, last = int(np.floor(airway[0] / hop_s)), int(np.ceil(airway[1] / hop_s))
        inside = measure_modulation(power[:, first:last], hop_s=hop_s, bin_hz=bin_hz, smooth_s=p.smooth_s, parameters=m)
        span = (min(last, power.shape[1]) - max(first, 0)) * hop_s
        if inside is not None and span > 0:
            active_fraction, active_over = round(inside.active_span_s / span, 3), OVER_TASK_EXTENT
    if active_fraction is None and pattern.modulation is not None and duration_s > 0:
        active_fraction, active_over = round(pattern.modulation.active_span_s / duration_s, 3), OVER_FILE
    veto = breath_veto_of(
        store, run_dir, active_fraction=active_fraction, active_over=active_over, extent=airway, language=language
    )
    return replace(pattern, veto=veto, extent=breath_extent, train=train)
