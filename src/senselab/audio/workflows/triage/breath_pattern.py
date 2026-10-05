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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml
from scipy.signal import butter, find_peaks, sosfiltfilt

from senselab.audio.tasks.classification.label_scores import label_scores
from senselab.audio.workflows.triage.nodes.common import find_measurement
from senselab.utils.prov_store import ProvStore

BREATH_PATTERN_PATH = Path(__file__).parent / "data" / "breath_pattern.yaml"

SPECTROGRAM = "spectrogram_narrowband"
PHONATION_TRACKS = "phonation_tracks"
YAMNET_SCORES = "yamnet_scores"
HEAR_SCORES = "hear_scores"

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
        peak_min_distance_s: The shortest spacing between two counted peaks.
        peak_prominence_std: The prominence a peak needs, in standard deviations of the component.
        min_duration_s: The shortest recording the reading is taken over.
        peaks_per_breath: Modulation peaks one breath makes (an inhale and an exhale).
        breathing_min_db: At or above this ratio, with an active span, the cycles count.
    """

    band_edges_hz: tuple[float, ...]
    fs_mod: float
    breath_band_hz: tuple[float, float]
    syllabic_band_hz: tuple[float, float]
    reference_band_hz: tuple[float, float]
    floor_percentile: float
    active_rise_db: float
    peak_min_distance_s: float
    peak_prominence_std: float
    min_duration_s: float
    peaks_per_breath: float
    breathing_min_db: float


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
        peak_min_distance_s=float(held["peak_min_distance_s"]),
        peak_prominence_std=float(held["peak_prominence_std"]),
        min_duration_s=float(held["min_duration_s"]),
        peaks_per_breath=float(held["peaks_per_breath"]),
        breathing_min_db=float(held["breathing_min_db"]),
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
        modulation_peaks: Peaks of the breathing-band component inside that span.
        estimated_breaths: ``modulation_peaks`` over the peaks one breath makes, rounded half up.
        breathing: The ratio reaches the breathing minimum over a non-zero active span, so the
            estimated breaths count.
    """

    breath_vs_syllabic_db: float
    active_span_s: float
    modulation_peaks: int
    estimated_breaths: int
    breathing: bool

    def record(self) -> dict[str, Any]:
        """The reading, as JSON-ready values.

        Returns:
            The fields, keyed by name.
        """
        return {
            "breath_vs_syllabic_db": self.breath_vs_syllabic_db,
            "active_span_s": self.active_span_s,
            "modulation_peaks": self.modulation_peaks,
            "estimated_breaths": self.estimated_breaths,
            "breathing": self.breathing,
        }


@dataclass(frozen=True)
class EvidenceParameters:
    """The ``evidence`` section of ``data/breath_pattern.yaml``.

    Attributes:
        yamnet_label: The YAMNet label a breath is heard as.
        yamnet_mean_min: The mean of that label over the file's YAMNet windows that is evidence.
        hear_label: The HeAR label a breath is heard as.
        hear_max_min: The highest HeAR window score of that label that is evidence.
    """

    yamnet_label: str
    yamnet_mean_min: float
    hear_label: str
    hear_max_min: float


@functools.cache
def evidence_parameters() -> EvidenceParameters:
    """The ``evidence`` parameters of ``data/breath_pattern.yaml``.

    Returns:
        The parameters.
    """
    held = (yaml.safe_load(BREATH_PATTERN_PATH.read_text()) or {})["evidence"]
    return EvidenceParameters(
        yamnet_label=str(held["yamnet_label"]),
        yamnet_mean_min=float(held["yamnet_mean_min"]),
        hear_label=str(held["hear_label"]),
        hear_max_min=float(held["hear_max_min"]),
    )


@dataclass(frozen=True)
class BreathEvidence:
    """Whether the stored classifier windows hear a breath anywhere in the recording.

    Attributes:
        yamnet_mean: The mean YAMNet score of the breath label over the file's windows, or None where
            YAMNet's windows are absent.
        hear_max: The highest HeAR window score of the breath label, or None where HeAR's windows
            are absent.
        heard: Either score reaches its minimum.
    """

    yamnet_mean: float | None
    hear_max: float | None
    heard: bool

    def record(self) -> dict[str, Any]:
        """The reading, as JSON-ready values.

        Returns:
            The fields, keyed by name.
        """
        return {"yamnet_mean": self.yamnet_mean, "hear_max": self.hear_max, "heard": self.heard}


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
        evidence: What the stored classifier windows hear, or None where it was not read.
    """

    pattern: str
    events_n: int
    event_durations_s: tuple[float, ...] = field(default_factory=tuple)
    intervals_s: tuple[float, ...] = field(default_factory=tuple)
    autocorr_peak: float | None = None
    rhythm: bool = False
    modulation: ModulationReading | None = None
    evidence: BreathEvidence | None = None

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
            "evidence": self.evidence.record() if self.evidence is not None else None,
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
    for start, stop in merged:
        duration = (stop - start) * hop_s
        voiced_share = float(voiced[start:stop].mean()) if stop > start else 0.0
        if p.event_s[0] <= duration <= p.event_s[1] and voiced_share <= p.voiced_fraction_max:
            events.append((start, round(duration, 3)))

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


def measure_modulation(
    power: np.ndarray, *, hop_s: float, bin_hz: float, smooth_s: float, parameters: ModulationParameters
) -> ModulationReading | None:
    """Read breathing cycles off the modulation spectra of subband envelopes.

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
    step = max(1, int(round(1 / (p.fs_mod * hop_s))))
    frames = power.shape[1] // step
    if frames * step * hop_s < p.min_duration_s:
        return None
    tiny = 1e-12
    width = max(1, int(round(smooth_s / hop_s))) | 1
    envelopes = []
    for low, high in zip(p.band_edges_hz[:-1], p.band_edges_hz[1:]):
        level = 10 * np.log10(power[_band_rows(bin_hz, (low, high))].sum(axis=0) + tiny)
        level = _moving_median(level, width)
        envelopes.append(level[: frames * step].reshape(frames, step).mean(axis=1))
    bands = np.array(envelopes)
    spectra = []
    for envelope in bands:
        freqs, spectrum = _modulation_psd(envelope, p.fs_mod)
        reference = (freqs >= p.reference_band_hz[0]) & (freqs <= p.reference_band_hz[1])
        spectra.append(spectrum / (spectrum[reference].sum() + tiny))
    mean_spectrum = np.mean(spectra, axis=0)
    breath = (freqs >= p.breath_band_hz[0]) & (freqs <= p.breath_band_hz[1])
    syllabic = (freqs >= p.syllabic_band_hz[0]) & (freqs <= p.syllabic_band_hz[1])
    ratio_db = float(10 * np.log10((mean_spectrum[breath].sum() + tiny) / (mean_spectrum[syllabic].sum() + tiny)))

    broadband = 10 * np.log10(np.sum(10 ** (bands / 10), axis=0))
    floor = float(np.percentile(broadband, p.floor_percentile))
    active = np.flatnonzero(broadband >= floor + p.active_rise_db)
    span_s = float((active[-1] - active[0] + 1) / p.fs_mod) if len(active) else 0.0
    sos = butter(2, p.breath_band_hz, btype="band", fs=p.fs_mod, output="sos")
    component = np.mean([sosfiltfilt(sos, envelope - envelope.mean()) for envelope in bands], axis=0)
    peaks, _ = find_peaks(
        component,
        distance=max(1, int(p.peak_min_distance_s * p.fs_mod)),
        prominence=p.peak_prominence_std * (float(np.std(component)) + tiny),
    )
    counted = int(sum(1 for peak in peaks if broadband[peak] >= floor + p.active_rise_db / 2))
    return ModulationReading(
        breath_vs_syllabic_db=round(ratio_db, 2),
        active_span_s=round(span_s, 2),
        modulation_peaks=counted,
        estimated_breaths=int(np.floor(counted / p.peaks_per_breath + 0.5)),
        breathing=ratio_db >= p.breathing_min_db and span_s > 0,
    )


def _sidecar(store: ProvStore, run_dir: Path, name: str) -> tuple[Mapping[str, Any], Path] | None:
    measurement = find_measurement(store, name)
    if measurement is None:
        return None
    relative = measurement.attributes.get("path")
    if not relative:
        return None
    path = run_dir / str(relative)
    return (measurement.attributes, path) if path.is_file() else None


def _label_series(store: ProvStore, run_dir: Path, name: str, label: str) -> list[float] | None:
    held = _sidecar(store, run_dir, name)
    if held is None:
        return None
    values = []
    for window in json.loads(held[1].read_text()):
        scores = {key: score for pair in label_scores(window) for key, score in pair.items()}
        values.append(float(scores.get(label, 0.0)))
    return values


def breath_evidence_of(
    store: ProvStore, run_dir: Path, parameters: EvidenceParameters | None = None
) -> BreathEvidence | tuple[str, ...]:
    """Whether the stored YAMNet or HeAR windows hear a breath, or the inputs it could not be read without.

    Args:
        store: The provenance store, read for the ``yamnet_scores`` and ``hear_scores`` measurements.
        run_dir: The run directory their sidecar paths are relative to.
        parameters: The evidence parameters; ``data/breath_pattern.yaml`` when None.

    Returns:
        The evidence, read from whichever of the two is present; the names of both where neither is.
    """
    p = parameters or evidence_parameters()
    yamnet = _label_series(store, run_dir, YAMNET_SCORES, p.yamnet_label)
    hear = _label_series(store, run_dir, HEAR_SCORES, p.hear_label)
    if yamnet is None and hear is None:
        return (YAMNET_SCORES, HEAR_SCORES)
    yamnet_mean = round(float(np.mean(yamnet)), 3) if yamnet else None
    hear_max = round(float(np.max(hear)), 3) if hear else None
    heard = (yamnet_mean is not None and yamnet_mean >= p.yamnet_mean_min) or (
        hear_max is not None and hear_max >= p.hear_max_min
    )
    return BreathEvidence(yamnet_mean=yamnet_mean, hear_max=hear_max, heard=bool(heard))


def breath_pattern_of(
    store: ProvStore,
    run_dir: Path,
    *,
    sampling_hz: float,
    parameters: BreathPatternParameters | None = None,
    modulation: ModulationParameters | None = None,
) -> BreathPattern | tuple[str, ...]:
    """The recording's breathing pattern, or the stored inputs it could not be read without.

    Args:
        store: The provenance store, read for the ``spectrogram_narrowband`` and
            ``phonation_tracks`` measurements.
        run_dir: The run directory their sidecar paths are relative to.
        sampling_hz: The conditioned stream's sampling rate, which with the spectrogram's own
            ``n_fft`` and ``hop_length`` fixes its bin width and hop.
        parameters: The measure's parameters; ``data/breath_pattern.yaml`` when None.
        modulation: The modulation reading's parameters; ``data/breath_pattern.yaml`` when None.

    Returns:
        The pattern with its classifier evidence (:func:`breath_evidence_of`); or the names of the
        absent inputs, where either derivative or its sidecar is missing, or both classifiers' windows.
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
    evidence = breath_evidence_of(store, run_dir)
    if not isinstance(evidence, BreathEvidence):
        return evidence
    pattern = measure_breath_pattern(
        power,
        hop_s=hop_s,
        bin_hz=sampling_hz / n_fft,
        voiced=voiced,
        parameters=p,
        modulation=modulation or modulation_parameters(),
    )
    return replace(pattern, evidence=evidence)
