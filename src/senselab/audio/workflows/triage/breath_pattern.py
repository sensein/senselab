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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from senselab.audio.workflows.triage.nodes.common import find_measurement
from senselab.utils.prov_store import ProvStore

BREATH_PATTERN_PATH = Path(__file__).parent / "data" / "breath_pattern.yaml"

SPECTROGRAM = "spectrogram_narrowband"
PHONATION_TRACKS = "phonation_tracks"

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
    """

    pattern: str
    events_n: int
    event_durations_s: tuple[float, ...] = field(default_factory=tuple)
    intervals_s: tuple[float, ...] = field(default_factory=tuple)
    autocorr_peak: float | None = None
    rhythm: bool = False

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
) -> BreathPattern:
    """Read the breathing pattern off a power spectrogram and a per-frame voicing mask.

    Args:
        power: Power spectrogram, frequency bins by frames, bin 0 at 0 Hz.
        hop_s: Seconds between frames.
        bin_hz: Hertz between frequency bins.
        voiced: Whether each frame is voiced, one per frame.
        parameters: The measure's parameters.

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


def breath_pattern_of(
    store: ProvStore, run_dir: Path, *, sampling_hz: float, parameters: BreathPatternParameters | None = None
) -> BreathPattern | tuple[str, ...]:
    """The recording's breathing pattern, or the stored inputs it could not be read without.

    Args:
        store: The provenance store, read for the ``spectrogram_narrowband`` and
            ``phonation_tracks`` measurements.
        run_dir: The run directory their sidecar paths are relative to.
        sampling_hz: The conditioned stream's sampling rate, which with the spectrogram's own
            ``n_fft`` and ``hop_length`` fixes its bin width and hop.
        parameters: The measure's parameters; ``data/breath_pattern.yaml`` when None.

    Returns:
        The pattern; or the names of the absent inputs, where either derivative or its sidecar is
        missing.
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
    return measure_breath_pattern(power, hop_s=hop_s, bin_hz=sampling_hz / n_fft, voiced=voiced, parameters=p)
