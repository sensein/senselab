"""The phonation attempt of a declared voice task, read off the plain, residual and enhanced streams.

The extent is where the frame level stands over a floor estimated outside the phonation, or the plain
stream is voiced; broken and restarted holds merge into it, the preparatory inhale attaches to its
start, and a microphone shutoff ends it. F0 is tracked on the plain stream, octave-checked against
the harmonic ridge, and stripped of mains lines where the residual carries hum. Holds, breaks, voice
quality and a glide's travel are readings over that extent.

Every parameter is in ``data/voice_phonation.yaml``. The design is
``specs/20261006-voice-phonation/design.md``.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from senselab.utils.prov_store import ProvStore

VOICE_PHONATION_PATH = Path(__file__).parent / "data" / "voice_phonation.yaml"
PLAIN_STREAM = "plain"
RESIDUAL_STREAM = "residual"
ENHANCED_STREAM = "enhanced"

Signal = tuple[np.ndarray, int]
Word = tuple[float, float, str]
Run = tuple[int, int]


@functools.cache
def voice_phonation_parameters() -> dict[str, Any]:
    """The parameters of ``data/voice_phonation.yaml``.

    Returns:
        The parsed mapping.
    """
    return dict(yaml.safe_load(VOICE_PHONATION_PATH.read_text()) or {})


@dataclass(frozen=True)
class Frames:
    """The per-frame series every reading is taken over.

    Attributes:
        times_s: Each frame's centre.
        level_db: The band level, dB re full scale.
        spectrum_db: The magnitude spectrum per frame, dB, frames by bins.
        bin_hz: The spectrum's bin width.
    """

    times_s: np.ndarray
    level_db: np.ndarray
    spectrum_db: np.ndarray
    bin_hz: float


def frames_of(signal: Signal, p: dict[str, Any]) -> Frames:
    """The frame level and spectrum of one stream.

    Args:
        signal: The samples and their rate.
        p: The parameters.

    Returns:
        The frames.
    """
    samples, rate = signal
    frame, hop = int(p["frame_s"] * rate), max(1, int(p["hop_s"] * rate))
    x = samples.astype(np.float64)
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    windows = np.lib.stride_tricks.sliding_window_view(x, frame)[::hop] * np.hanning(frame)
    nfft = int(2 ** np.ceil(np.log2(max(frame, rate / 8.0))))
    power = np.abs(np.fft.rfft(windows, n=nfft, axis=1)) ** 2 / max(np.sum(np.hanning(frame) ** 2), 1e-12)
    bin_hz = rate / nfft
    low, high = p["band_hz"]
    band = slice(int(low / bin_hz), int(min(high, rate / 2) / bin_hz) + 1)
    level = 10.0 * np.log10(power[:, band].sum(axis=1) / max(frame, 1) + 1e-12)
    times = (np.arange(len(windows)) * hop + frame / 2) / rate
    return Frames(times, level, 10.0 * np.log10(power + 1e-12), bin_hz)


def runs_of(mask: np.ndarray) -> list[Run]:
    """The runs of True in a mask, as ``(first, end)`` index pairs, end exclusive."""
    padded = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def bridge(runs: Sequence[Run], gap_frames: int) -> list[Run]:
    """Runs joined across every gap shorter than ``gap_frames``."""
    out: list[Run] = []
    for first, end in runs:
        if out and first - out[-1][1] < gap_frames:
            out[-1] = (out[-1][0], end)
        else:
            out.append((first, end))
    return out


def shutoff_runs(frames: Frames, p: dict[str, Any]) -> list[Run]:
    """Runs where every band dropped abruptly to a flat level well under the rest of the recording.

    Args:
        frames: The plain stream's frames.
        p: The parameters.

    Returns:
        The shutoff runs, in frame indices.
    """
    q = p["shutoff"]
    hop = p["hop_s"]
    level = frames.level_db
    digital = level < p["digital_floor_dbfs"]
    width = max(2, int(round(q["flat_window_s"] / hop)))
    flat = np.zeros(len(level), dtype=bool)
    if len(level) >= width:
        windows = np.lib.stride_tricks.sliding_window_view(level, width)
        spread = np.percentile(windows, 90, axis=1) - np.percentile(windows, 10, axis=1)
        for k in np.flatnonzero(spread <= q["flat_db"]):
            flat[k : k + width] = True
    live = floor_db(level, ~(flat | digital), p)
    dead = digital | (flat & (level <= live - q["below_floor_db"]))
    active = np.flatnonzero(level >= live + p["phonation_db"])
    if active.size == 0:
        return []
    drop_n = max(1, int(round(q["drop_s"] / hop)))
    found: list[Run] = []
    for first, end in bridge(runs_of(dead), int(round(q["bridge_s"] / hop)) + 1):
        if (end - first) * hop < q["min_s"] or first <= active[0]:
            continue
        before = level[max(0, first - drop_n) : first]
        if before.size and before.max() - np.median(level[first:end]) >= q["drop_db"]:
            found.append((first, end))
    return found


def floor_db(level: np.ndarray, valid: np.ndarray, p: dict[str, Any]) -> float:
    """The quietest stretch of the recording outside the excluded frames: its noise floor.

    Args:
        level: The frame level.
        valid: Which frames the floor may be read from.
        p: The parameters.

    Returns:
        The floor, in dB.
    """
    width = max(1, int(round(p["floor_smooth_s"] / p["hop_s"])))
    values = np.where(valid & (level >= p["digital_floor_dbfs"]), level, np.nan)
    if np.isfinite(values).sum() < width:
        finite = values[np.isfinite(values)]
        return float(np.min(finite)) if finite.size else float(np.min(level))
    kernel = np.ones(width)
    sums = np.convolve(np.nan_to_num(values), kernel, mode="valid")
    counts = np.convolve(np.isfinite(values).astype(float), kernel, mode="valid")
    means = np.where(counts >= width, sums / np.maximum(counts, 1), np.nan)
    return float(np.nanmin(means)) if np.isfinite(means).any() else float(np.nanmin(values))


def pitch_of(signal: Signal, times_s: np.ndarray, p: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Praat cc F0 and its strength on the plain stream, sampled at the frame times.

    Args:
        signal: The samples and their rate.
        times_s: The frame times.
        p: The parameters.

    Returns:
        F0 (NaN where unvoiced) and strength, one per frame.
    """
    import parselmouth  # noqa: PLC0415 -- the reading runs only where a voice task is measured

    samples, rate = signal
    low, high = p["f0"]["range_hz"]
    sound = parselmouth.Sound(samples.astype(np.float64), sampling_frequency=float(rate))
    pitch = sound.to_pitch_cc(time_step=p["hop_s"], pitch_floor=low, pitch_ceiling=min(high, rate / 2.0))
    xs = np.asarray(pitch.xs(), dtype=float)
    f0 = np.asarray(pitch.selected_array["frequency"], dtype=float)
    strength = np.asarray(pitch.selected_array["strength"], dtype=float)
    if xs.size == 0:
        return np.full(times_s.shape, np.nan), np.zeros(times_s.shape)
    index = np.clip(np.searchsorted(xs, times_s), 0, xs.size - 1)
    near = np.where(
        (index > 0) & (np.abs(xs[np.maximum(index - 1, 0)] - times_s) < np.abs(xs[index] - times_s)), index - 1, index
    )
    out_f0, out_strength = f0[near].copy(), strength[near].copy()
    out_f0[(out_f0 <= 0.0) | (out_strength < p["f0"]["strength_min"])] = np.nan
    outside = (times_s < xs[0] - p["hop_s"]) | (times_s > xs[-1] + p["hop_s"])
    out_f0[outside] = np.nan
    return out_f0, out_strength


def _harmonic_db(column: np.ndarray, bin_hz: float, f0: float, multiples: Sequence[float], top_hz: float) -> float:
    picks = [column[int(round(m * f0 / bin_hz))] for m in multiples if m * f0 < top_hz]
    return float(np.mean(picks)) if picks else float("nan")


def octave_checked(f0_hz: np.ndarray, frames: Frames, p: dict[str, Any]) -> np.ndarray:
    """F0 moved an octave where the harmonic ridge says the tracker took half or double the pitch.

    A frame whose odd multiples stand well under its even ones is read at half its pitch and
    doubled; one whose half-pitch odd multiples stand level with its own harmonics is halved. A frame
    still far from the running median of the corrected track is then folded toward it by an octave.

    Args:
        f0_hz: F0 per frame, NaN where unvoiced.
        frames: The frames, read for their spectrum.
        p: The parameters.

    Returns:
        The corrected F0.
    """
    q = p["f0"]
    low, high = q["range_hz"]
    top = min(p["band_hz"][1], frames.bin_hz * (frames.spectrum_db.shape[1] - 1))
    k = int(q["harmonics"])
    odd = [2 * i + 1 for i in range(k)]
    even = [2 * i + 2 for i in range(k)]
    out = np.asarray(f0_hz, dtype=float).copy()
    for i in np.flatnonzero(np.isfinite(out)):
        column = frames.spectrum_db[i]
        for _ in range(2):
            f = out[i]
            if 2 * f <= high and _harmonic_db(column, frames.bin_hz, f, odd, top) < (
                _harmonic_db(column, frames.bin_hz, f, even, top) - q["octave_db"]
            ):
                out[i] = 2 * f
                continue
            h = f / 2
            if h >= low and _harmonic_db(column, frames.bin_hz, h, odd, top) >= (
                _harmonic_db(column, frames.bin_hz, h, even, top) - q["octave_db"] / 2
            ):
                out[i] = h
                continue
            break
    width = max(3, int(round(q["fold_window_s"] / p["hop_s"])))
    st = 12.0 * np.log2(out)
    voiced = np.flatnonzero(np.isfinite(st))
    for j, i in enumerate(voiced):
        near = voiced[max(0, j - width // 2) : j + width // 2 + 1]
        centre = float(np.median(st[near]))
        while st[i] - centre > q["fold_semitones"] and out[i] / 2 >= low:
            st[i] -= 12.0
            out[i] /= 2
        while centre - st[i] > q["fold_semitones"] and out[i] * 2 <= high:
            st[i] += 12.0
            out[i] *= 2
    return out


@dataclass(frozen=True)
class Hum:
    """The residual's mains-hum reading.

    Attributes:
        fired: Whether the guard fires.
        lines: Mains fundamental to how many of its multiples stand over their neighbourhood.
        lock_fraction: The voiced plain frames within tolerance of a mains multiple.
        mains_hz: The fundamentals whose multiples are dropped from F0, empty where it did not fire.
    """

    fired: bool
    lines: dict[str, int]
    lock_fraction: float
    mains_hz: tuple[float, ...]

    def record(self) -> dict[str, Any]:
        """The reading, for the store."""
        return {
            "fired": self.fired,
            "lines": dict(self.lines),
            "lock_fraction": round(self.lock_fraction, 4),
            "mains_hz": list(self.mains_hz),
        }


def _near_mains(f0: np.ndarray, mains: Sequence[float], q: dict[str, Any]) -> np.ndarray:
    near = np.zeros(f0.shape, dtype=bool)
    finite = np.isfinite(f0)
    for m in mains:
        for k in range(1, int(q["harmonics_max"]) + 1):
            near |= finite & (np.abs(f0 - k * m) <= q["tolerance_hz"])
    return near


def hum_of(residual: Signal | None, f0_hz: np.ndarray, p: dict[str, Any]) -> Hum:
    """Whether the residual carries mains lines, or the plain F0 is locked onto a mains multiple.

    Args:
        residual: The residual stream, or None where it is absent.
        f0_hz: The plain F0 per frame.
        p: The parameters.

    Returns:
        The hum reading.
    """
    q = p["hum"]
    lines: dict[str, int] = {}
    if residual is not None and len(residual[0]) >= residual[1]:
        from scipy.signal import welch  # noqa: PLC0415

        samples, rate = residual
        freqs, psd = welch(samples.astype(np.float64), fs=rate, nperseg=int(rate))
        db = 10.0 * np.log10(psd + 1e-20)
        lo, hi = q["neighbourhood_hz"]
        for m in q["mains_hz"]:
            count = 0
            for k in range(1, int(q["harmonics_max"]) + 1):
                f = k * m
                at = np.abs(freqs - f) <= 0.5
                around = (np.abs(freqs - f) >= lo) & (np.abs(freqs - f) <= hi)
                if at.any() and around.any() and db[at].max() - np.median(db[around]) >= q["line_db"]:
                    count += 1
            lines[f"{m:g}"] = count
    voiced = np.isfinite(f0_hz)
    lock = float(_near_mains(f0_hz, [50.0, 60.0], {**q, "harmonics_max": 2})[voiced].mean()) if voiced.any() else 0.0
    by_lines = tuple(float(m) for m in q["mains_hz"] if lines.get(f"{m:g}", 0) >= q["lines_min"])
    fired = bool(by_lines) or lock >= q["lock_fraction_min"]
    mains = by_lines or (tuple(float(m) for m in q["mains_hz"]) if fired else ())
    return Hum(fired, lines, lock, mains)


@dataclass(frozen=True)
class Extent:
    """The phonation attempt's extent, with the runs it was merged from.

    Attributes:
        onset: The first phonation frame.
        offset: The frame after the last.
        start: The extent's first frame, the inhale's start where one attached.
        holds: The merged runs, in frame indices.
        outside: The runs left outside the extent.
    """

    onset: int
    offset: int
    start: int
    holds: tuple[Run, ...]
    outside: tuple[Run, ...]


def speech_words(words: Sequence[Word], p: dict[str, Any]) -> list[Word]:
    """The lexical words that are speech rather than the vowel itself transcribed."""
    vowel = re.compile(str(p["vowel_word_pattern"]))
    out: list[Word] = []
    for a, b, text in words:
        letters = re.sub(r"[^a-z]", "", text.lower())
        if letters and not vowel.match(letters) and b - a <= p["speech_word_max_s"]:
            out.append((a, b, text))
    return out


def speech_mask(times: np.ndarray, words: Sequence[Word], p: dict[str, Any]) -> np.ndarray:
    """Which frames a speech word covers."""
    mask = np.zeros(times.shape, dtype=bool)
    for a, b, _ in speech_words(words, p):
        mask |= (times >= a) & (times <= b)
    return mask


def extent_of(
    frames: Frames,
    voiced: np.ndarray,
    floor: float,
    words: Sequence[Word],
    blocked: np.ndarray,
    p: dict[str, Any],
    *,
    phonation_db: float | None = None,
) -> Extent | None:
    """The phonation extent: the longest run off speech words, merged with the further holds around it.

    Args:
        frames: The plain stream's frames.
        voiced: Which frames the plain F0 is voiced in.
        floor: The noise floor.
        words: The lexical consensus words, ``(start, end, text)``.
        blocked: Frames no extent may cover (a shutoff).
        p: The parameters.
        phonation_db: The level over the floor that is phonation; None takes the configured value.

    Returns:
        The extent, or None where no run reaches ``extent_min_s``.
    """
    hop = p["hop_s"]
    margin = p["phonation_db"] if phonation_db is None else phonation_db
    level = frames.level_db
    spoken = speech_mask(frames.times_s, words, p)
    phonation = ((level >= floor + margin) | voiced) & ~blocked & ~spoken
    runs = bridge(runs_of(phonation), int(round(p["break_min_s"] / hop)))
    candidates = [i for i, run in enumerate(runs) if (run[1] - run[0]) * hop >= p["extent_min_s"]]
    if not candidates:
        return None
    core = max(candidates, key=lambda i: runs[i][1] - runs[i][0])
    first = last = core
    gap = int(round(p["merge_gap_max_s"] / hop))
    segment = int(round(p["segment_min_s"] / hop))

    def mergeable(i: int) -> bool:
        a, b = runs[i]
        held = voiced[a:b].mean() >= p["merge_voiced_fraction_min"] or np.median(level[a:b]) >= floor + p["merge_db"]
        return b - a >= segment and bool(held)

    grown = True
    while grown:
        grown = False
        if first > 0 and mergeable(first - 1) and _joins(runs[first - 1], runs[first], gap, spoken):
            first -= 1
            grown = True
        if last < len(runs) - 1 and mergeable(last + 1) and _joins(runs[last], runs[last + 1], gap, spoken):
            last += 1
            grown = True
    onset, offset = runs[first][0], runs[last][1]
    start = _inhale_start(frames, voiced, floor, onset, blocked, p)
    holds = tuple(runs[first : last + 1])
    outside = tuple(run for i, run in enumerate(runs) if i < first or i > last)
    return Extent(onset, offset, start, holds, outside)


def _joins(before: Run, after: Run, gap: int, spoken: np.ndarray) -> bool:
    """Whether two runs merge: the gap between them is short and holds no speech word."""
    return after[0] - before[1] <= gap and not spoken[before[1] : after[0]].any()


def _inhale_start(
    frames: Frames, voiced: np.ndarray, floor: float, onset: int, blocked: np.ndarray, p: dict[str, Any]
) -> int:
    q = p["inhale"]
    hop = p["hop_s"]
    earliest = max(0, onset - int(round(q["max_s"] / hop)))
    rising = (frames.level_db >= floor + q["rise_db"]) & ~voiced & ~blocked
    i = onset - 1
    gap = int(round(q["gap_s"] / hop))
    while i >= earliest and onset - 1 - i < gap and not rising[i]:
        i -= 1
    if i < earliest or not rising[i]:
        return onset
    end = i
    while i - 1 >= earliest and rising[i - 1]:
        i -= 1
    return i if (end - i + 1) * hop >= q["min_s"] else onset


def holds_and_breaks(
    frames: Frames, voiced: np.ndarray, floor: float, extent: Extent, p: dict[str, Any]
) -> tuple[list[Run], list[Run]]:
    """The holds inside the extent and the breaks between them.

    Args:
        frames: The frames.
        voiced: Which frames are voiced.
        floor: The noise floor.
        extent: The extent.
        p: The parameters.

    Returns:
        The holds and the breaks, in frame indices; a break is a fall below phonation of at least
        ``break_min_s``.
    """
    phonation = (frames.level_db >= floor + p["phonation_db"]) | voiced
    inside = phonation[extent.onset : extent.offset]
    gaps = [
        (extent.onset + a, extent.onset + b) for a, b in runs_of(~inside) if (b - a) * p["hop_s"] >= p["break_min_s"]
    ]
    holds: list[Run] = []
    cursor = extent.onset
    for a, b in gaps:
        if a > cursor:
            holds.append((cursor, a))
        cursor = b
    if extent.offset > cursor:
        holds.append((cursor, extent.offset))
    return holds, gaps


@dataclass(frozen=True)
class Glide:
    """A glide's travel over the extent.

    Attributes:
        declared: The declared direction, ``up`` or ``down``.
        travel_declared: Semitones travelled in the declared direction.
        travel_opposite: Semitones travelled against it.
        voiced_s: The voiced time it was read over.
    """

    declared: str
    travel_declared: float
    travel_opposite: float
    voiced_s: float

    def shape(self, bound: float, voiced_min_s: float) -> str:
        """What the glide was: ``declared``, ``opposite``, ``held`` or ``unmeasurable``."""
        if self.voiced_s < voiced_min_s:
            return "unmeasurable"
        if self.travel_declared >= bound:
            return "declared"
        if self.travel_opposite >= bound:
            return "opposite"
        return "held"


def glide_of(f0_hz: np.ndarray, extent: Extent, declared: str, p: dict[str, Any]) -> Glide:
    """The net travel of the octave-checked F0 over the extent, in and against the declared direction.

    Frame-to-frame jumps over ``jump_max_semitones`` are discontinuities, not travel.

    Args:
        f0_hz: The checked F0.
        extent: The extent.
        declared: ``up`` or ``down``.
        p: The parameters.

    Returns:
        The glide reading.
    """
    q = p["glide"]
    piece = f0_hz[extent.onset : extent.offset]
    st = 12.0 * np.log2(piece[np.isfinite(piece)])
    voiced_s = st.size * p["hop_s"]
    if st.size < 2:
        return Glide(declared, 0.0, 0.0, voiced_s)
    width = int(q["smoothing_frames"])
    if width > 1 and st.size >= width:
        st = np.array([np.median(st[max(0, i - width // 2) : i + width // 2 + 1]) for i in range(st.size)])
    steps = np.diff(st)
    steps[np.abs(steps) > q["jump_max_semitones"]] = 0.0
    path = np.concatenate([[0.0], np.cumsum(steps)])
    up = float(np.max(path - np.minimum.accumulate(path)))
    down = float(np.max(np.maximum.accumulate(path) - path))
    return Glide(declared, up if declared == "up" else down, down if declared == "up" else up, voiced_s)


@dataclass(frozen=True)
class PhonationReading:
    """Everything the VOICE measure read over a declared voice task.

    Attributes:
        extent: ``start_s``, ``end_s``, ``onset_s``, ``offset_s`` and ``source``, or None where nothing
            rose above the floor.
        holds: Each hold, ``(start, end)`` seconds.
        breaks: Each break, ``(start, end)`` seconds.
        floor_db: The noise floor.
        voiced_fraction: The voiced share of the phonation frames inside the extent.
        voiced_s: The voiced time inside the extent.
        f0_median_hz: The median checked F0 inside the extent.
        f0_spread_semitones: The typical windowed F0 spread inside the extent.
        glide: The glide reading for a glide family; empty otherwise.
        shape: The glide's shape, for a glide family; None otherwise.
        mismatch: A ``task_mismatch`` description, or None.
        shutoff: ``start_s`` and whether it cut the task, or empty.
        outside_speech: Each speech-like run outside the extent, with the words over it.
        hum: The hum guard's reading.
        enhanced_extent: The enhanced stream's extent where the hum guard fired; None otherwise.
        enhanced_minus_plain_db: The enhanced stream's level against the plain one's.
        review: Why the decision is left for review; empty where it is not.
    """

    extent: dict[str, Any] | None
    holds: tuple[tuple[float, float], ...] = ()
    breaks: tuple[tuple[float, float], ...] = ()
    floor_db: float = float("nan")
    voiced_fraction: float = 0.0
    voiced_s: float = 0.0
    f0_median_hz: float | None = None
    f0_spread_semitones: float | None = None
    glide: dict[str, Any] = field(default_factory=dict)
    shape: str | None = None
    mismatch: str | None = None
    shutoff: dict[str, Any] = field(default_factory=dict)
    outside_speech: tuple[dict[str, Any], ...] = ()
    hum: dict[str, Any] = field(default_factory=dict)
    enhanced_extent: tuple[float, float] | None = None
    enhanced_minus_plain_db: float | None = None
    review: tuple[str, ...] = ()

    @property
    def found(self) -> bool:
        """Whether a phonation attempt was found."""
        return self.extent is not None

    @property
    def capture_cut(self) -> bool:
        """Whether a microphone shutoff cut the task."""
        return bool(self.shutoff.get("during_task"))

    def record(self) -> dict[str, Any]:
        """The reading, for the store."""
        holds = [round(b - a, 3) for a, b in self.holds]
        return {
            "found": self.found,
            "extent": self.extent,
            "holds": [[round(a, 3), round(b, 3)] for a, b in self.holds],
            "breaks": [[round(a, 3), round(b, 3)] for a, b in self.breaks],
            "holds_n": len(self.holds),
            "longest_hold_s": max(holds, default=0.0),
            "floor_db": round(self.floor_db, 2),
            "voiced_fraction": round(self.voiced_fraction, 4),
            "voiced_s": round(self.voiced_s, 3),
            "f0_median_hz": None if self.f0_median_hz is None else round(self.f0_median_hz, 1),
            "f0_spread_semitones": None if self.f0_spread_semitones is None else round(self.f0_spread_semitones, 3),
            "glide": dict(self.glide),
            "shape": self.shape,
            "mismatch": self.mismatch,
            "shutoff": dict(self.shutoff),
            "outside_speech": [dict(run) for run in self.outside_speech],
            "hum": dict(self.hum),
            "enhanced_extent": None if self.enhanced_extent is None else [round(v, 3) for v in self.enhanced_extent],
            "enhanced_minus_plain_db": None
            if self.enhanced_minus_plain_db is None
            else round(self.enhanced_minus_plain_db, 2),
            "review": list(self.review),
        }


def _seconds(frames: Frames, run: Run, hop: float) -> tuple[float, float]:
    return float(frames.times_s[run[0]] - hop / 2), float(frames.times_s[run[1] - 1] + hop / 2)


def _windowed_spread(st: np.ndarray, hop: float, window_s: float) -> float | None:
    width = max(2, int(round(window_s / hop)))
    spreads = [
        float(np.percentile(w[np.isfinite(w)], 95) - np.percentile(w[np.isfinite(w)], 5))
        for w in (st[i : i + width] for i in range(0, max(1, st.size - width + 1)))
        if np.isfinite(w).sum() >= 2
    ]
    return float(np.median(spreads)) if spreads else None


def measure_phonation(
    plain: Signal,
    *,
    residual: Signal | None,
    enhanced: Signal | None,
    words: Sequence[Word],
    glide_direction: str | None,
    p: dict[str, Any] | None = None,
) -> PhonationReading:
    """Read the phonation attempt of a declared voice task.

    Args:
        plain: The plain stream.
        residual: The residual stream, or None.
        enhanced: The enhanced stream, or None.
        words: The lexical consensus words, ``(start, end, text)``.
        glide_direction: ``up`` or ``down`` for a glide family; None for a held one.
        p: The parameters; None reads ``data/voice_phonation.yaml``.

    Returns:
        The reading.
    """
    p = p or voice_phonation_parameters()
    hop = p["hop_s"]
    frames = frames_of(plain, p)
    blocked = np.zeros(len(frames.times_s), dtype=bool)
    shutoffs = shutoff_runs(frames, p)
    for a, b in shutoffs:
        blocked[a:b] = True
    floor = floor_db(frames.level_db, ~blocked, p)
    f0_raw, _ = pitch_of(plain, frames.times_s, p)
    hum = hum_of(residual, f0_raw, p)
    if hum.fired:
        f0_raw = np.where(_near_mains(f0_raw, hum.mains_hz, p["hum"]), np.nan, f0_raw)
    voiced = np.isfinite(f0_raw) & (frames.level_db >= floor + p["voiced_rise_db"])
    f0 = octave_checked(f0_raw, frames, p)
    enhanced_drop = (
        float(np.mean(frames_of(enhanced, p).level_db) - np.mean(frames.level_db)) if enhanced is not None else None
    )
    extent = extent_of(frames, voiced, floor, words, blocked, p)
    if extent is None:
        return PhonationReading(
            None,
            floor_db=floor,
            hum=hum.record(),
            enhanced_minus_plain_db=enhanced_drop,
            review=_review_existence(frames, voiced, floor, words, blocked, p, found=False),
        )
    offset = extent.offset
    shutoff: dict[str, Any] = {}
    gap = int(round(p["shutoff"]["gap_s"] / hop))
    for a, _ in shutoffs:
        during = extent.onset <= a <= offset + gap
        if during or not shutoff:
            shutoff = {"start_s": round(float(frames.times_s[a] - hop / 2), 3), "during_task": during}
        if during:
            break
    holds, breaks = holds_and_breaks(frames, voiced, floor, extent, p)
    inside = slice(extent.onset, offset)
    phonation_frames = ((frames.level_db >= floor + p["phonation_db"]) | voiced)[inside]
    voiced_inside = voiced[inside] & phonation_frames
    voiced_fraction = float(voiced_inside.sum() / max(phonation_frames.sum(), 1))
    checked = f0[inside]
    finite = checked[np.isfinite(checked)]
    f0_median = float(np.median(finite)) if finite.size else None
    spread = _windowed_spread(12.0 * np.log2(checked), hop, p["f0"]["spread_window_s"]) if finite.size else None
    start_s = _seconds(frames, (extent.start, extent.start + 1), hop)[0]
    end_s = _seconds(frames, (offset - 1, offset), hop)[1]
    onset_s = _seconds(frames, (extent.onset, extent.onset + 1), hop)[0]
    record_extent = {
        "start_s": round(start_s, 3),
        "end_s": round(end_s, 3),
        "onset_s": round(onset_s, 3),
        "offset_s": round(end_s, 3),
        "inhale": extent.start < extent.onset,
        "source": "phonation",
    }
    glide: dict[str, Any] = {}
    shape: str | None = None
    mismatch: str | None = None
    review = list(_review_existence(frames, voiced, floor, words, blocked, p, found=True))
    if glide_direction is not None:
        reading = glide_of(f0, extent, glide_direction, p)
        bound = p["glide"]["range_min_semitones"]
        shape = reading.shape(bound, p["glide"]["voiced_min_s"])
        glide = {
            "declared": glide_direction,
            "travel_declared_semitones": round(reading.travel_declared, 2),
            "travel_opposite_semitones": round(reading.travel_opposite, 2),
            "voiced_s": round(reading.voiced_s, 3),
        }
        if shape != "declared":
            mismatch = _glide_mismatch(shape, reading, f0_median, end_s - onset_s)
        margin = p["review"]["glide_margin_semitones"]
        strict = reading.shape(bound + margin, p["glide"]["voiced_min_s"])
        lenient = reading.shape(bound - margin, p["glide"]["voiced_min_s"])
        if (strict == "declared") != (lenient == "declared"):
            review.append("glide_bound")
    enhanced_extent: tuple[float, float] | None = None
    if hum.fired and enhanced is not None:
        enhanced_frames = frames_of(enhanced, p)
        no_voicing = np.zeros(len(enhanced_frames.times_s), dtype=bool)
        found = extent_of(
            enhanced_frames,
            no_voicing,
            floor_db(enhanced_frames.level_db, np.ones_like(no_voicing), p),
            words,
            no_voicing,
            p,
        )
        if found is not None:
            enhanced_extent = (
                _seconds(enhanced_frames, (found.start, found.start + 1), hop)[0],
                _seconds(enhanced_frames, (found.offset - 1, found.offset), hop)[1],
            )
        if (
            enhanced_extent is None
            or max(abs(enhanced_extent[0] - start_s), abs(enhanced_extent[1] - end_s)) > p["hum"]["disagreement_s"]
        ):
            review.append("hum_stream_disagreement")
    outside = _outside_speech(frames, voiced, floor, extent, words, p)
    return PhonationReading(
        record_extent,
        holds=tuple(_seconds(frames, run, hop) for run in holds),
        breaks=tuple(_seconds(frames, run, hop) for run in breaks),
        floor_db=floor,
        voiced_fraction=voiced_fraction,
        voiced_s=float(voiced_inside.sum()) * hop,
        f0_median_hz=f0_median,
        f0_spread_semitones=spread,
        glide=glide,
        shape=shape,
        mismatch=mismatch,
        shutoff=shutoff,
        outside_speech=outside,
        hum=hum.record(),
        enhanced_extent=enhanced_extent,
        enhanced_minus_plain_db=enhanced_drop,
        review=tuple(review),
    )


def _glide_mismatch(shape: str, glide: Glide, f0_median: float | None, duration_s: float) -> str:
    if shape == "opposite":
        return f"glide went {'down' if glide.declared == 'up' else 'up'} {glide.travel_opposite:.1f} st"
    if shape == "held":
        held = f"{f0_median:.0f} Hz" if f0_median is not None else "an unmeasured pitch"
        return f"held near {held} for {duration_s:.1f} s, {glide.travel_declared:.1f} st {glide.declared}"
    return f"no measurable glide over {glide.voiced_s:.2f} s of voicing"


def _review_existence(
    frames: Frames,
    voiced: np.ndarray,
    floor: float,
    words: Sequence[Word],
    blocked: np.ndarray,
    p: dict[str, Any],
    *,
    found: bool,
) -> tuple[str, ...]:
    margin = p["review"]["floor_margin_db"]
    strict = extent_of(frames, voiced, floor, words, blocked, p, phonation_db=p["phonation_db"] + margin)
    lenient = extent_of(frames, voiced, floor, words, blocked, p, phonation_db=p["phonation_db"] - margin)
    unsettled = (strict is not None) != (lenient is not None) or (strict is not None) != found
    return ("phonation_floor",) if unsettled else ()


def _outside_speech(
    frames: Frames, voiced: np.ndarray, floor: float, extent: Extent, words: Sequence[Word], p: dict[str, Any]
) -> tuple[dict[str, Any], ...]:
    hop = p["hop_s"]
    q = p["speech"]
    out: list[dict[str, Any]] = []
    sound = (frames.level_db >= floor + p["phonation_db"]) | voiced
    sound[extent.start : extent.offset] = False
    for run in bridge(runs_of(sound), int(round(p["break_min_s"] / hop))):
        if (run[1] - run[0]) * hop < q["run_min_s"] or voiced[run[0] : run[1]].mean() < q["voiced_fraction_min"]:
            continue
        start, end = _seconds(frames, run, hop)
        said = [text for a, b, text in words if min(end, b) > max(start, a)]
        out.append(
            {
                "start_s": round(start, 3),
                "end_s": round(end, 3),
                "side": "before" if run[1] <= extent.onset else "after",
                "words": said,
            }
        )
    return tuple(out)


def _named_stream(store: ProvStore, run_dir: Path, name: str) -> Signal | None:
    import soundfile  # noqa: PLC0415 -- decoding is only needed where a reading runs

    stream = next(
        (s for s in store.entities("stream") if s.attributes.get("name") == name and not store.is_invalidated(s.id)),
        None,
    )
    path = run_dir / str(stream.attributes.get("path") or "") if stream is not None else None
    if path is None or not path.is_file():
        return None
    samples, rate = soundfile.read(path, dtype="float32", always_2d=True)
    return samples.mean(axis=1), int(rate)


def phonation_reading_of(
    store: ProvStore, run_dir: Path, *, words: Sequence[Word], glide_direction: str | None
) -> PhonationReading | tuple[str, ...]:
    """The phonation reading of a declared voice task, from the store's streams.

    Args:
        store: The provenance store, read for the ``plain``, ``residual`` and ``enhanced`` streams.
        run_dir: The run directory their paths are relative to.
        words: The lexical consensus words, ``(start, end, text)``.
        glide_direction: ``up`` or ``down`` for a glide family; None for a held one.

    Returns:
        The reading, or the names of the inputs that were absent (the ``plain`` stream).
    """
    plain = _named_stream(store, run_dir, PLAIN_STREAM)
    if plain is None:
        return (PLAIN_STREAM,)
    return measure_phonation(
        plain,
        residual=_named_stream(store, run_dir, RESIDUAL_STREAM),
        enhanced=_named_stream(store, run_dir, ENHANCED_STREAM),
        words=words,
        glide_direction=glide_direction,
    )
