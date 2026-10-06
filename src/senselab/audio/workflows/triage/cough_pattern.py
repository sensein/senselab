"""The coughs of an airway recording, read off PREPROCESS's stored narrowband spectrogram.

The spectrogram (computed over the pre-emphasised stream) is summed into subbands up to about 8 kHz.
A cough onset is a sharp rise of the broadband envelope that most subbands share, out of a level near
the recording's own floor, reaching well above it and decaying over a fraction of a second. A weaker
rise soon after an onset, or one out of a level still well above the floor, is that cough's second
phase or expiratory tail and is attached to it. One onset is one cough. Onsets beside a speech word
are speech.

Every parameter is in ``data/cough_pattern.yaml``. See ``specs/20261006-cough-pattern/design.md``.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml
from scipy.signal import find_peaks

from senselab.audio.workflows.triage.breath_pattern import SPECTROGRAM, _in_script, _moving_median, _sidecar
from senselab.audio.workflows.triage.nodes.common import lexical_words, word_hull
from senselab.audio.workflows.triage.residue import is_non_lexical
from senselab.utils.prov_store import ProvStore

COUGH_PATTERN_PATH = Path(__file__).parent / "data" / "cough_pattern.yaml"


@dataclass(frozen=True)
class CoughParameters:
    """The measure's parameters, as the ``measure`` section of ``data/cough_pattern.yaml`` holds them.

    Attributes:
        band_edges_hz: The subband edges.
        smooth_s: The per-band moving-median window.
        floor_percentile: The percentile of the broadband envelope taken as the recording's floor.
        digital_silence_db: Summed band power at or below which a frame is digital silence.
        rise_s: The window a rise is read over.
        rise_db: The broadband rise a candidate onset reaches.
        band_rise_db: The rise a subband reaches to count towards coherence.
        coherence_min: The share of subbands rising together that makes a rise broadband.
        min_gap_s: The nearest two candidates may be.
        gap_lookback_s: The window before an onset its pre-onset level is read over.
        gap_db: The pre-onset level above the floor at or below which a rise starts a new cough.
        restart_rise_fraction: The share of the previous onset's rise at or above which a rise starts a
            new cough out of any level.
        peak_db: The level above the floor an onset reaches within ``peak_window_s``.
        peak_window_s: The window after an onset its peak is read over.
        tail_db: The level above the floor an event lasts while it holds.
        min_event_s: The shortest event.
        attach_s: How soon after an onset a weaker rise is that cough's second phase or tail.
        train_drop_db: How far below the median onset peak an onset may fall and stay in the train.
        word_pad_s: How close to a speech word an onset is speech.
        inhale_gap_s: The longest quiet between a preparatory inhale's end and its cough's onset.
        inhale_bridge_s: The longest quiet inside an inhale.
        inhale_max_s: How far before its onset an inhale may start.
        inhale_min_s: The shortest inhale.
        inhale_db: The level above the floor an inhale holds.
    """

    band_edges_hz: tuple[float, ...]
    smooth_s: float
    floor_percentile: float
    digital_silence_db: float
    rise_s: float
    rise_db: float
    band_rise_db: float
    coherence_min: float
    min_gap_s: float
    gap_lookback_s: float
    gap_db: float
    restart_rise_fraction: float
    peak_db: float
    peak_window_s: float
    tail_db: float
    min_event_s: float
    attach_s: float
    train_drop_db: float
    word_pad_s: float
    inhale_gap_s: float
    inhale_bridge_s: float
    inhale_max_s: float
    inhale_min_s: float
    inhale_db: float


@dataclass(frozen=True)
class CoughExtentParameters:
    """The ``extent`` section of ``data/cough_pattern.yaml``.

    Attributes:
        pre_s: How far before the first onset the extent starts.
        post_s: How far after the last event's end it ends.
    """

    pre_s: float
    post_s: float


@dataclass(frozen=True)
class CoughReviewParameters:
    """The ``review`` section of ``data/cough_pattern.yaml``.

    Attributes:
        rise_margin_db: How far ``rise_db`` is raised and lowered for the review band.
        coherence_margin: How far ``coherence_min`` is raised and lowered for it.
    """

    rise_margin_db: float
    coherence_margin: float


def _section(name: str) -> dict[str, Any]:
    return dict((yaml.safe_load(COUGH_PATTERN_PATH.read_text()) or {})[name])


@functools.cache
def cough_parameters() -> CoughParameters:
    """The ``measure`` parameters of ``data/cough_pattern.yaml``.

    Returns:
        The parameters.
    """
    held = _section("measure")
    return CoughParameters(
        band_edges_hz=tuple(float(edge) for edge in held.pop("band_edges_hz")),
        **{key: float(value) for key, value in held.items()},
    )


@functools.cache
def cough_extent_parameters() -> CoughExtentParameters:
    """The ``extent`` parameters of ``data/cough_pattern.yaml``.

    Returns:
        The parameters.
    """
    return CoughExtentParameters(**{key: float(value) for key, value in _section("extent").items()})


@functools.cache
def cough_review_parameters() -> CoughReviewParameters:
    """The ``review`` parameters of ``data/cough_pattern.yaml``.

    Returns:
        The parameters.
    """
    return CoughReviewParameters(**{key: float(value) for key, value in _section("review").items()})


EXTENT_COUGH_ONSETS = "cough_onsets"


@dataclass(frozen=True)
class CoughExtent:
    """Where in the recording the cough task was performed.

    Attributes:
        start_s: The extent's start, in seconds.
        end_s: Its end, in seconds.
    """

    start_s: float
    end_s: float

    def record(self) -> dict[str, Any]:
        """The extent, as JSON-ready values.

        Returns:
            The fields, keyed by name, with ``source`` naming the measure.
        """
        return {"start_s": self.start_s, "end_s": self.end_s, "source": EXTENT_COUGH_ONSETS}


@dataclass(frozen=True)
class CoughPattern:
    """What :func:`measure_cough_pattern` read.

    Attributes:
        onsets_s: Each cough's onset, in seconds.
        event_spans_s: Each cough's ``(start, end)``: its preparatory inhale, onset, second phase and tail.
        rises_db: Each onset's broadband rise.
        peaks_db: Each onset's peak above the recording's floor.
        intervals_s: Onset-to-onset intervals.
        floor_db: The recording's floor, on the summed band-power scale.
        extent: The cough-task extent, or None where no cough was found.
        onsets_strict_n: The onsets with the rise and coherence thresholds raised by the review margins.
        onsets_lenient_n: The onsets with them lowered by the same margins.
    """

    onsets_s: tuple[float, ...] = ()
    event_spans_s: tuple[tuple[float, float], ...] = ()
    rises_db: tuple[float, ...] = ()
    peaks_db: tuple[float, ...] = ()
    intervals_s: tuple[float, ...] = ()
    floor_db: float | None = None
    extent: CoughExtent | None = None
    onsets_strict_n: int = 0
    onsets_lenient_n: int = 0

    @property
    def onsets_n(self) -> int:
        """How many coughs were found."""
        return len(self.onsets_s)

    def record(self) -> dict[str, Any]:
        """The reading, as JSON-ready values.

        Returns:
            The fields, keyed by name.
        """
        return {
            "onsets_n": self.onsets_n,
            "onsets_s": list(self.onsets_s),
            "event_spans_s": [list(span) for span in self.event_spans_s],
            "rises_db": list(self.rises_db),
            "peaks_db": list(self.peaks_db),
            "intervals_s": list(self.intervals_s),
            "floor_db": self.floor_db,
            "onsets_strict_n": self.onsets_strict_n,
            "onsets_lenient_n": self.onsets_lenient_n,
            "extent": self.extent.record() if self.extent is not None else None,
        }


def band_levels_db(power: np.ndarray, *, bin_hz: float, band_edges_hz: Sequence[float]) -> np.ndarray:
    """Summed power per subband, in dB.

    Args:
        power: The ``(bins, frames)`` power spectrogram.
        bin_hz: Its bin width.
        band_edges_hz: The subband edges.

    Returns:
        A ``(bands, frames)`` array.
    """
    rows = [
        power[int(np.ceil(low / bin_hz)) : int(np.floor(high / bin_hz)) + 1].sum(axis=0)
        for low, high in zip(band_edges_hz[:-1], band_edges_hz[1:])
    ]
    return 10.0 * np.log10(np.asarray(rows) + 1e-12)


def _trailing_min(values: np.ndarray, width: int) -> np.ndarray:
    padded = np.concatenate([np.full(width, values[0]), values])
    return np.lib.stride_tricks.sliding_window_view(padded, width + 1).min(axis=-1)


@dataclass(frozen=True)
class _Onsets:
    onsets: list[int]
    starts: list[int]
    ends: list[int]
    rises: list[float]
    peaks: list[float]
    floor: float


def _detect(bands_db: np.ndarray, hop_s: float, p: CoughParameters, words: Sequence[tuple[float, float]]) -> _Onsets:
    frames = bands_db.shape[1]
    smoothed = np.array([_moving_median(band, max(1, int(p.smooth_s / hop_s)) | 1) for band in bands_db])
    k = max(1, int(p.rise_s / hop_s))
    coherent = np.mean([band - _trailing_min(band, k) >= p.band_rise_db for band in smoothed], axis=0)
    broad = 10.0 * np.log10(np.sum(10.0 ** (smoothed / 10.0), axis=0))
    silent = broad <= p.digital_silence_db
    floor = float(np.percentile(broad[~silent] if (~silent).any() else broad, p.floor_percentile))
    rise = broad - _trailing_min(broad, k)
    candidates, _ = find_peaks(rise, height=p.rise_db, distance=max(1, int(p.min_gap_s / hop_s)))
    lookback, peak_w = int(p.gap_lookback_s / hop_s), int(p.peak_window_s / hop_s)
    onsets: list[int] = []
    tails: list[int] = []
    rises: list[float] = []
    peaks: list[float] = []
    for c in candidates:
        if coherent[max(0, c - 2) : c + 3].max() < p.coherence_min:
            continue
        start = max(0, c - k)
        o = start + int(np.argmin(broad[start : c + 1]))
        before = max(0, o - lookback)
        pre = float(broad[before : o + 1].min()) - floor
        peak = float(broad[o : min(frames, o + peak_w)].max()) - floor
        j = int(c)
        while j < frames - 1 and broad[j] >= floor + p.tail_db:
            j += 1
        t = o * hop_s
        from_leading_silence = bool(silent[before : o + 1].any()) and not (~silent[: before + 1]).any()
        spoken = any(w0 - p.word_pad_s <= t <= w1 + p.word_pad_s for w0, w1 in words)
        if peak < p.peak_db or (j - o) * hop_s < p.min_event_s or spoken or from_leading_silence:
            continue
        if onsets and pre > p.gap_db and rise[c] < p.restart_rise_fraction * rises[-1]:
            tails[-1] = max(tails[-1], j)
            continue
        if onsets and (o - onsets[-1]) * hop_s < p.attach_s:
            if rise[c] <= rises[-1]:
                tails[-1] = max(tails[-1], j)
                continue
            onsets.pop(), tails.pop(), rises.pop(), peaks.pop()
        onsets.append(o)
        tails.append(j)
        rises.append(float(rise[c]))
        peaks.append(peak)
    if onsets:
        top = float(np.median(peaks))
        keep = [i for i, peak in enumerate(peaks) if peak >= top - p.train_drop_db]
        onsets, tails = [onsets[i] for i in keep], [tails[i] for i in keep]
        rises, peaks = [rises[i] for i in keep], [peaks[i] for i in keep]
    starts: list[int] = []
    ends: list[int] = []
    inhaling = broad >= floor + p.inhale_db
    for i, o in enumerate(onsets):
        bound = onsets[i + 1] if i + 1 < len(onsets) else frames
        j = o + int(np.argmax(broad[o : min(bound, o + peak_w)]))
        while j < bound - 1 and broad[j] >= floor + p.tail_db:
            j += 1
        ends.append(min(max(j, tails[i]), bound - 1))
        starts.append(_inhale_start(inhaling, o, ends[-2] if i else 0, hop_s, p))
    return _Onsets(onsets=onsets, starts=starts, ends=ends, rises=rises, peaks=peaks, floor=floor)


def _inhale_start(active: np.ndarray, onset: int, bound: int, hop_s: float, p: CoughParameters) -> int:
    """The start of a preparatory inhale ending just before ``onset``, else ``onset`` itself."""
    lead, bridge = int(p.inhale_gap_s / hop_s), int(p.inhale_bridge_s / hop_s)
    earliest = max(bound + 1, onset - int(p.inhale_max_s / hop_s))
    t = onset - 1
    while t >= earliest and onset - t <= lead and not active[t]:
        t -= 1
    if t < earliest or not active[t]:
        return onset
    start, quiet = t, 0
    while t >= earliest and quiet <= bridge:
        if active[t]:
            start, quiet = t, 0
        else:
            quiet += 1
        t -= 1
    return start if (onset - start) * hop_s >= p.inhale_min_s else onset


def cough_extent(
    spans: Sequence[tuple[float, float]],
    *,
    words: Sequence[tuple[float, float]],
    duration_s: float,
    parameters: CoughExtentParameters,
) -> CoughExtent | None:
    """The cough-task extent over the found coughs, padded and kept off speech words.

    Args:
        spans: Each cough's ``(onset, end)``.
        words: The speech words' ``(start, end)``.
        duration_s: The recording's duration.
        parameters: The extent's parameters.

    Returns:
        From ``pre_s`` before the first onset to ``post_s`` after the last end, cut at the nearest
        speech word outside the coughs; None with no cough.
    """
    if not spans:
        return None
    first, last = spans[0][0], spans[-1][1]
    start, end = max(0.0, first - parameters.pre_s), min(duration_s, last + parameters.post_s)
    for w0, w1 in words:
        if w1 <= first:
            start = max(start, w1)
        if w0 >= last:
            end = min(end, w0)
    return CoughExtent(start_s=round(float(start), 3), end_s=round(float(end), 3))


def measure_cough_pattern(
    bands_db: np.ndarray,
    *,
    hop_s: float,
    words: Sequence[tuple[float, float]] = (),
    parameters: CoughParameters | None = None,
    extent: CoughExtentParameters | None = None,
    review: CoughReviewParameters | None = None,
) -> CoughPattern:
    """The coughs in a recording's subband levels.

    Args:
        bands_db: The ``(bands, frames)`` levels of :func:`band_levels_db`.
        hop_s: The frame hop.
        words: The speech words' ``(start, end)``; an onset beside one is speech.
        parameters: The measure's parameters; ``data/cough_pattern.yaml`` when None.
        extent: The extent's parameters; ``data/cough_pattern.yaml`` when None.
        review: The review band's margins; ``data/cough_pattern.yaml`` when None.

    Returns:
        The reading.
    """
    p = parameters or cough_parameters()
    r = review or cough_review_parameters()
    if bands_db.ndim != 2 or bands_db.shape[1] < 2:
        return CoughPattern()
    found = _detect(bands_db, hop_s, p, words)
    strict = replace(p, rise_db=p.rise_db + r.rise_margin_db, coherence_min=p.coherence_min + r.coherence_margin)
    lenient = replace(p, rise_db=p.rise_db - r.rise_margin_db, coherence_min=p.coherence_min - r.coherence_margin)
    spans = tuple((round(float(s * hop_s), 3), round(float(e * hop_s), 3)) for s, e in zip(found.starts, found.ends))
    onsets = tuple(round(float(o * hop_s), 3) for o in found.onsets)
    return CoughPattern(
        onsets_s=onsets,
        event_spans_s=spans,
        rises_db=tuple(round(value, 2) for value in found.rises),
        peaks_db=tuple(round(value, 2) for value in found.peaks),
        intervals_s=tuple(round(b - a, 3) for a, b in zip(onsets, onsets[1:])),
        floor_db=round(found.floor, 2),
        extent=cough_extent(
            spans, words=words, duration_s=bands_db.shape[1] * hop_s, parameters=extent or cough_extent_parameters()
        ),
        onsets_strict_n=len(_detect(bands_db, hop_s, strict, words).onsets),
        onsets_lenient_n=len(_detect(bands_db, hop_s, lenient, words).onsets),
    )


def in_cough_review_band(reading: CoughPattern, needed: int) -> bool:
    """Whether the decision on a cough reading differs inside its review band.

    Args:
        reading: The cough reading.
        needed: The coughs the task needs: its instructed count, else one.

    Returns:
        True where the strict and the lenient counts fall on opposite sides of ``needed``.
    """
    return (reading.onsets_strict_n >= needed) != (reading.onsets_lenient_n >= needed)


def speech_word_spans(store: ProvStore, language: str | None) -> list[tuple[float, float]]:
    """The consensus words that are speech, as ``(start, end)``.

    Args:
        store: The provenance store.
        language: The recording's declared language, or None.

    Returns:
        The hulls of the words that are not bracketed, not a vocalisation or interjection, and written
        in the declared language's script, in time order.
    """
    spans = []
    for word in lexical_words(store):
        text = str(word.attributes.get("text") or "")
        if is_non_lexical(text) or not _in_script(text, language):
            continue
        spans.append(word_hull(word))
    return sorted(spans)


def cough_pattern_of(
    store: ProvStore, run_dir: Path, *, sampling_hz: float, language: str | None = None
) -> CoughPattern | tuple[str, ...]:
    """The recording's coughs, or the stored inputs they could not be read without.

    Args:
        store: The provenance store, read for the ``spectrogram_narrowband`` measurement and the
            consensus words.
        run_dir: The run directory its sidecar path is relative to.
        sampling_hz: The conditioned stream's sampling rate, which with the spectrogram's own
            ``n_fft`` and ``hop_length`` fixes its bin width and hop.
        language: The recording's declared language, which fixes the script speech words are read in.

    Returns:
        The reading of :func:`measure_cough_pattern`; or ``("spectrogram_narrowband",)`` where the
        derivative, its sidecar or its framing is missing.
    """
    held = _sidecar(store, run_dir, SPECTROGRAM)
    if held is None:
        return (SPECTROGRAM,)
    attributes, path = held
    n_fft, hop_length = attributes.get("n_fft"), attributes.get("hop_length")
    if not isinstance(n_fft, int) or not isinstance(hop_length, int) or n_fft <= 0 or hop_length <= 0:
        return (SPECTROGRAM,)
    with np.load(path) as loaded:
        power = np.asarray(loaded["spectrogram"], dtype=np.float64)
    if power.ndim == 3:
        power = power[0]
    p = cough_parameters()
    bands = band_levels_db(power, bin_hz=sampling_hz / n_fft, band_edges_hz=p.band_edges_hz)
    return measure_cough_pattern(
        bands, hop_s=hop_length / sampling_hz, words=speech_word_spans(store, language), parameters=p
    )
