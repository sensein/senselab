"""Harmonicity, glottal period marks, formant tracks and phonation spans, through Praat."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.features_extraction.praat_parselmouth import (
    PARSELMOUTH_AVAILABLE,
    get_sound,
    parselmouth,
)


@dataclass(frozen=True)
class PeriodMark:
    """One glottal period, delimited by two consecutive Praat pulses.

    Attributes:
        time_s: The opening pulse, in the recording's clock.
        period_s: Time to the next pulse.
        amplitude: Peak absolute amplitude of the waveform within the period.
    """

    time_s: float
    period_s: float
    amplitude: float


def _require_parselmouth() -> None:
    """Raise when parselmouth is not installed."""
    if not PARSELMOUTH_AVAILABLE:
        raise ModuleNotFoundError(
            "`parselmouth` is not installed. Please install senselab audio dependencies using `pip install senselab`."
        )


def hnr_track(
    audio: Audio,
    *,
    f0_min_hz: float,
    hop_s: float,
    silence_threshold: float,
    periods_per_window: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Harmonics-to-noise ratio over time, in dB, via Praat's cc method.

    Args:
        audio: The recording. ``get_sound`` handles channel merging and resampling.
        f0_min_hz: Lowest F0 the analysis considers. Read it from ``voice.f0_range_hz``.
        hop_s: Praat's ``time_step``. Read it from ``phonation.hop_s``.
        silence_threshold: Praat's silence threshold. Read it from ``phonation.silence_threshold``.
        periods_per_window: Praat's analysis window length, in periods of ``f0_min_hz``. Read it
            from ``phonation.periods_per_window``.

    Returns:
        ``(times_s, hnr_db)``, one value per frame. Frames Praat judges silent carry its floor
        value rather than being dropped, so the track and its times stay aligned.

    Raises:
        ModuleNotFoundError: If parselmouth is not installed.
    """
    _require_parselmouth()
    snd = get_sound(audio)
    harmonicity = snd.to_harmonicity_cc(
        time_step=hop_s,
        minimum_pitch=f0_min_hz,
        silence_threshold=silence_threshold,
        periods_per_window=periods_per_window,
    )
    times = np.asarray(harmonicity.xs(), dtype=np.float64)
    values = np.asarray(harmonicity.values, dtype=np.float64).squeeze(0)
    return times, values


def period_marks(
    audio: Audio,
    start_s: float,
    end_s: float,
    *,
    f0_min_hz: float,
    f0_max_hz: float,
) -> list[PeriodMark]:
    """Glottal period marks inside one span, from Praat's point process.

    Args:
        audio: The recording.
        start_s: Span onset.
        end_s: Span offset.
        f0_min_hz: Lowest F0 to search. Read it from ``voice.f0_range_hz``.
        f0_max_hz: Highest F0 to search. Read it from ``voice.f0_range_hz``.

    Returns:
        One mark per pair of consecutive pulses whose gap is a plausible period — within
        ``[1/f0_max_hz, 1/f0_min_hz]``; a longer gap is a voicing break, not a period. Empty when
        Praat places no pulses — absent, not zero.

    Raises:
        ModuleNotFoundError: If parselmouth is not installed.
    """
    _require_parselmouth()
    snd = get_sound(audio)
    part = snd.extract_part(from_time=start_s, to_time=end_s, preserve_times=True)
    point_process = parselmouth.praat.call(part, "To PointProcess (periodic, cc)", f0_min_hz, f0_max_hz)
    n_points = int(parselmouth.praat.call(point_process, "Get number of points"))
    pulses = [float(parselmouth.praat.call(point_process, "Get time from index", i)) for i in range(1, n_points + 1)]
    x = np.asarray(part.values, dtype=np.float64).squeeze(0)
    t0 = float(part.xmin)
    sr = 1.0 / float(part.dx)
    marks: list[PeriodMark] = []
    for opening, closing in zip(pulses, pulses[1:]):
        period = closing - opening
        if not (1.0 / f0_max_hz <= period <= 1.0 / f0_min_hz):
            continue
        i0 = max(0, int((opening - t0) * sr))
        i1 = min(len(x), int((closing - t0) * sr))
        amplitude = float(np.abs(x[i0:i1]).max()) if i1 > i0 else 0.0
        marks.append(PeriodMark(time_s=opening, period_s=period, amplitude=amplitude))
    return marks


def f0_track(
    audio: Audio,
    *,
    f0_min_hz: float,
    f0_max_hz: float,
    hop_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """F0 and its strength per frame via Praat's cc pitch: ``(times_s, f0_hz, strength)``.

    Unvoiced frames carry NaN in ``f0_hz`` with their ``strength`` retained, so F0 always travels
    with the periodicity that placed it and a reader cannot separate them.

    Args:
        audio: The recording. ``get_sound`` handles channel merging and resampling.
        f0_min_hz: Lowest F0 to search. Read it from ``voice.f0_range_hz``.
        f0_max_hz: Highest F0 to search. Read it from ``voice.f0_range_hz``.
        hop_s: Praat's ``time_step``. Read it from ``phonation.hop_s``.

    Returns:
        ``(times_s, f0_hz, strength)``, one value per frame, all three the same length.

    Raises:
        ModuleNotFoundError: If parselmouth is not installed.
    """
    _require_parselmouth()
    snd = get_sound(audio)
    pitch = snd.to_pitch_cc(time_step=hop_s, pitch_floor=f0_min_hz, pitch_ceiling=f0_max_hz)
    times = np.asarray(pitch.xs(), dtype=np.float64)
    f0 = np.asarray(pitch.selected_array["frequency"], dtype=np.float64)
    strength = np.asarray(pitch.selected_array["strength"], dtype=np.float64)
    f0[f0 == 0.0] = np.nan
    return times, f0, strength


@dataclass(frozen=True)
class FormantTrack:
    """Formant frequencies and bandwidths over one stream, on a fixed hop.

    Attributes:
        times_s: Frame times, in seconds.
        f_hz: Four arrays, F1 to F4, each one value per frame. NaN where Praat placed none.
        bandwidth_hz: Four arrays, the corresponding 3 dB bandwidths. NaN where the formant is NaN.
    """

    times_s: np.ndarray
    f_hz: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    bandwidth_hz: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def formant_track(
    audio: Audio,
    *,
    hop_s: float,
    max_formants: int,
    formant_max_hz: float,
    window_s: float,
    preemphasis_hz: float,
) -> FormantTrack:
    """F1-F4 and their bandwidths over the whole stream, by Praat's Burg method.

    Computed once over the stream so a consumer slices the track rather than re-tracking a fragment.

    Args:
        audio: The recording. ``get_sound`` handles channel merging and resampling.
        hop_s: Praat's ``time_step``. Read it from ``phonation_spans.hop_s``.
        max_formants: Praat's ``max_number_of_formants``. Read it from ``phonation_spans.max_formants``.
        formant_max_hz: Praat's ``maximum_formant``. Read it from ``phonation_spans.formant_max_hz``.
        window_s: Praat's ``window_length``. Read it from ``phonation_spans.formant_window_s``.
        preemphasis_hz: Praat's ``pre_emphasis_from``. Read it from
            ``phonation_spans.formant_preemphasis_hz``.

    Returns:
        The track. A frame where Praat placed no formant carries NaN in both arrays, so a missing
        formant is absent rather than zero.

    Raises:
        ModuleNotFoundError: If parselmouth is not installed.
    """
    _require_parselmouth()
    snd = get_sound(audio)
    formants = snd.to_formant_burg(
        time_step=hop_s,
        max_number_of_formants=max_formants,
        maximum_formant=formant_max_hz,
        window_length=window_s,
        pre_emphasis_from=preemphasis_hz,
    )
    times = np.asarray(formants.xs(), dtype=np.float64)
    values: list[np.ndarray] = []
    bandwidths: list[np.ndarray] = []
    for order in (1, 2, 3, 4):
        values.append(
            np.asarray(
                [formants.get_value_at_time(order, t, unit=parselmouth.FormantUnit.HERTZ) for t in times],
                dtype=np.float64,
            )
        )
        bandwidths.append(
            np.asarray(
                [formants.get_bandwidth_at_time(order, t, unit=parselmouth.FormantUnit.HERTZ) for t in times],
                dtype=np.float64,
            )
        )
    return FormantTrack(
        times_s=times,
        f_hz=(values[0], values[1], values[2], values[3]),
        bandwidth_hz=(bandwidths[0], bandwidths[1], bandwidths[2], bandwidths[3]),
    )


@dataclass(frozen=True)
class PhonationSpan:
    """One proposed sustained-phonation or glide span, with the statistics of its own frames.

    Attributes:
        start: Onset in seconds.
        end: Offset in seconds.
        member: ``"sustained"`` or ``"glide"``.
        production: ``"voiced"``, ``"unvoiced"`` or ``"mixed"``.
        voiced_fraction: Fraction of the span's frames whose pitch strength cleared the floor.
        f0_median_hz: Median F0 over the voiced frames, or None when there are none.
        f0_start_hz: F0 at the first voiced frame, or None.
        f0_end_hz: F0 at the last voiced frame, or None.
        glide_direction: ``"rising"``, ``"falling"``, or None for a sustained span.
        glide_extent_cents: The monotone excursion's magnitude, or None for a sustained span.
        offset_criterion: What closed the span — ``"f0_stability"``, ``"formant_stability"``,
            ``"both"``, ``"monotonicity"`` or ``"stream_end"``.
    """

    start: float
    end: float
    member: str
    production: str
    voiced_fraction: float
    f0_median_hz: float | None
    f0_start_hz: float | None
    f0_end_hz: float | None
    glide_direction: str | None
    glide_extent_cents: float | None
    offset_criterion: str


_CENTS_PER_OCTAVE = 1200.0
_MS_PER_S = 1000.0


def _cents(lower: float, upper: float) -> float:
    """The interval between two frequencies in cents, or NaN when either is not positive."""
    if not (lower > 0.0) or not (upper > 0.0):
        return float("nan")
    return float(_CENTS_PER_OCTAVE * np.log2(upper / lower))


def _on_grid(source_times: np.ndarray, values: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    """Read ``values`` at the frame of ``source_times`` nearest each of ``target_times``."""
    if len(source_times) == 0:
        return np.full(len(target_times), np.nan)
    right = np.clip(np.searchsorted(source_times, target_times), 1, len(source_times) - 1)
    left = right - 1
    take = np.where(
        np.abs(target_times - source_times[left]) <= np.abs(source_times[right] - target_times), left, right
    )
    return np.asarray(values, dtype=np.float64)[take]


def _sustained_runs(continues: np.ndarray, hangover_frames: int) -> list[tuple[int, int]]:
    """Maximal runs of continuing frames, each closed by ``hangover_frames`` of continuous failure."""
    runs: list[tuple[int, int]] = []
    n = len(continues)
    index = 1
    while index < n:
        if not continues[index]:
            index += 1
            continue
        last_ok, cursor, gap = index, index + 1, 0
        while cursor < n:
            if continues[cursor]:
                last_ok, gap = cursor, 0
            else:
                gap += 1
                if gap >= hangover_frames:
                    break
            cursor += 1
        runs.append((index - 1, last_ok))
        index = cursor + 1
    return runs


def _uncovered_runs(n: int, covered: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Maximal runs of frame indices no span in ``covered`` contains."""
    mask = np.ones(n, dtype=bool)
    for first, last in covered:
        mask[first : last + 1] = False
    runs: list[tuple[int, int]] = []
    index = 0
    while index < n:
        if not mask[index]:
            index += 1
            continue
        end = index
        while end + 1 < n and mask[end + 1]:
            end += 1
        runs.append((index, end))
        index = end + 1
    return runs


def _monotone_excursion(values: np.ndarray) -> tuple[str, float] | None:
    """The direction and magnitude of a monotone trajectory, or None when it is not monotone."""
    usable = values[np.isfinite(values) & (values > 0.0)]
    if len(usable) < 2:
        return None
    steps = np.diff(usable)
    if not (np.all(steps >= 0.0) or np.all(steps <= 0.0)):
        return None
    excursion = _cents(float(usable[0]), float(usable[-1]))
    if not np.isfinite(excursion) or excursion == 0.0:
        return None
    return ("rising" if excursion > 0.0 else "falling"), abs(excursion)


def propose_phonation_spans(  # noqa: C901 — one branch per member and per offset criterion
    *,
    times: np.ndarray,
    f0_hz: np.ndarray,
    strength: np.ndarray,
    formants: FormantTrack,
    hop_s: float,
    f0_stability_cents: float,
    formant_stability_hz: float,
    glide_min_excursion_cents: float,
    hangover_ms: float,
    voicing_strength_floor: float,
    mixed_voiced_fraction: float,
    unvoiced_max_formant_bandwidth_hz: float,
) -> list[PhonationSpan]:
    """Sustained-phonation and glide spans over tracks measured once on the whole stream.

    A frame continues the criterion when its F0 moved less than ``f0_stability_cents`` across one hop
    — a frame with no F0 never satisfies this limb — **or** when F1 and F2 both moved less than
    ``formant_stability_hz`` and their fitted bandwidths are at or below
    ``unvoiced_max_formant_bandwidth_hz``. A maximal run of continuing frames, closed by
    ``hangover_ms`` of continuous failure, is a ``"sustained"`` span. A maximal run outside every
    such span whose defined F0 values, or F1 where none is defined, are monotone with an excursion
    at or over ``glide_min_excursion_cents`` is a ``"glide"`` span. No periodicity floor opens or
    closes a span, but broad/unresolved LPC poles cannot by themselves admit non-periodic material.

    Args:
        times: The F0 track's frame times, in seconds. Every span is placed on this grid.
        f0_hz: F0 per frame, NaN where unvoiced, as :func:`f0_track` returns it.
        strength: Praat's pitch strength per frame, the same length as ``times``.
        formants: The stream's formant track. It is read at the frame nearest each of ``times``,
            so the two need not share a grid.
        hop_s: The analysis hop, in seconds. Read it from ``phonation_spans.hop_s``.
        f0_stability_cents: The F0 limb of the continuity criterion. Read it from
            ``phonation_spans.f0_stability_cents``.
        formant_stability_hz: The formant limb. Read it from ``phonation_spans.formant_stability_hz``.
        glide_min_excursion_cents: The excursion separating a glide from drift. Read it from
            ``phonation_spans.glide_min_excursion_cents``.
        hangover_ms: How long the criterion must fail continuously before a span closes. Read it
            from ``phonation_spans.hangover_ms``.
        voicing_strength_floor: The pitch strength above which a frame counts as voiced. Read it
            from ``phonation_spans.voicing_strength_floor``.
        mixed_voiced_fraction: The voiced-frame fraction separating the three productions. Read it
            from ``phonation_spans.mixed_voiced_fraction``.
        unvoiced_max_formant_bandwidth_hz: The widest F1/F2 pole admitted as non-periodic resonant
            evidence. Read it from ``phonation_spans.unvoiced_max_formant_bandwidth_hz``. This is a
            screening condition, not a diagnosis of phonation.

    Returns:
        The spans in time order. Empty when no run satisfies either member.
    """
    n = len(times)
    if n < 2:
        return []
    f0 = np.asarray(f0_hz, dtype=np.float64)
    f1 = _on_grid(formants.times_s, formants.f_hz[0], times)
    f2 = _on_grid(formants.times_s, formants.f_hz[1], times)
    f1_bandwidth = _on_grid(formants.times_s, formants.bandwidth_hz[0], times)
    f2_bandwidth = _on_grid(formants.times_s, formants.bandwidth_hz[1], times)

    f0_ok = np.zeros(n, dtype=bool)
    formant_ok = np.zeros(n, dtype=bool)
    for index in range(1, n):
        moved = _cents(float(f0[index - 1]), float(f0[index]))
        f0_ok[index] = bool(np.isfinite(moved) and abs(moved) < f0_stability_cents)
        formant_ok[index] = bool(
            np.isfinite(f1[index - 1])
            and np.isfinite(f1[index])
            and np.isfinite(f2[index - 1])
            and np.isfinite(f2[index])
            and np.isfinite(f1_bandwidth[index - 1])
            and np.isfinite(f1_bandwidth[index])
            and np.isfinite(f2_bandwidth[index - 1])
            and np.isfinite(f2_bandwidth[index])
            and abs(f1[index] - f1[index - 1]) < formant_stability_hz
            and abs(f2[index] - f2[index - 1]) < formant_stability_hz
            and f1_bandwidth[index - 1] <= unvoiced_max_formant_bandwidth_hz
            and f1_bandwidth[index] <= unvoiced_max_formant_bandwidth_hz
            and f2_bandwidth[index - 1] <= unvoiced_max_formant_bandwidth_hz
            and f2_bandwidth[index] <= unvoiced_max_formant_bandwidth_hz
        )
    continues = f0_ok | formant_ok

    hangover_frames = max(1, int(round(hangover_ms / _MS_PER_S / hop_s)))
    sustained = _sustained_runs(continues, hangover_frames)

    def _statistics(first: int, last: int) -> tuple[float, str, float | None, float | None, float | None]:
        """The span's voiced fraction, production, and F0 statistics over its voiced frames."""
        window = slice(first, last + 1)
        voiced = np.asarray(strength, dtype=np.float64)[window] >= voicing_strength_floor
        fraction = float(np.mean(voiced)) if len(voiced) else 0.0
        if fraction > mixed_voiced_fraction:
            production = "voiced"
        elif fraction < 1.0 - mixed_voiced_fraction:
            production = "unvoiced"
        else:
            production = "mixed"
        placed = f0[window][voiced & np.isfinite(f0[window])]
        if len(placed) == 0:
            return fraction, production, None, None, None
        return fraction, production, float(np.median(placed)), float(placed[0]), float(placed[-1])

    spans: list[PhonationSpan] = []
    for first, last in sustained:
        if last == n - 1:
            criterion = "stream_end"
        elif f0_ok[last] and formant_ok[last]:
            criterion = "both"
        elif f0_ok[last]:
            criterion = "f0_stability"
        else:
            criterion = "formant_stability"
        fraction, production, median, opening, closing = _statistics(first, last)
        spans.append(
            PhonationSpan(
                start=float(times[first]),
                end=float(times[last]) + hop_s,
                member="sustained",
                production=production,
                voiced_fraction=fraction,
                f0_median_hz=median,
                f0_start_hz=opening,
                f0_end_hz=closing,
                glide_direction=None,
                glide_extent_cents=None,
                offset_criterion=criterion,
            )
        )

    for first, last in _uncovered_runs(n, sustained):
        trajectory = f0[first : last + 1]
        if np.count_nonzero(np.isfinite(trajectory)) < 2:
            trajectory = f1[first : last + 1]
        found = _monotone_excursion(trajectory)
        if found is None or found[1] < glide_min_excursion_cents:
            continue
        direction, extent = found
        fraction, production, median, opening, closing = _statistics(first, last)
        spans.append(
            PhonationSpan(
                start=float(times[first]),
                end=float(times[last]) + hop_s,
                member="glide",
                production=production,
                voiced_fraction=fraction,
                f0_median_hz=median,
                f0_start_hz=opening,
                f0_end_hz=closing,
                glide_direction=direction,
                glide_extent_cents=extent,
                offset_criterion="stream_end" if last == n - 1 else "monotonicity",
            )
        )
    spans.sort(key=lambda span: span.start)
    return spans


def propose_word_aligned_phonation_spans(
    *,
    times: np.ndarray,
    f0_hz: np.ndarray,
    strength: np.ndarray,
    formants: FormantTrack,
    word_extents: list[tuple[float, float]],
    voicing_strength_floor: float,
    mixed_voiced_fraction: float,
    unvoiced_max_formant_bandwidth_hz: float,
    min_evidence_fraction: float,
) -> list[PhonationSpan]:
    """Return word-bounded spans supported by acoustic phonation evidence alone.

    Timed words contribute boundaries, not lexical evidence: a segment is positive when the required
    fraction of its frames has either periodic F0 evidence or narrow, resolved F1/F2 resonances.
    This complementary path deliberately does not reject recordings with no consensus words.
    """
    f0 = np.asarray(f0_hz, dtype=np.float64)
    pitch_strength = np.asarray(strength, dtype=np.float64)
    f1 = _on_grid(formants.times_s, formants.f_hz[0], times)
    f2 = _on_grid(formants.times_s, formants.f_hz[1], times)
    f1_bandwidth = _on_grid(formants.times_s, formants.bandwidth_hz[0], times)
    f2_bandwidth = _on_grid(formants.times_s, formants.bandwidth_hz[1], times)
    periodic = np.isfinite(f0) & (pitch_strength >= voicing_strength_floor)
    resonant = (
        np.isfinite(f1)
        & np.isfinite(f2)
        & np.isfinite(f1_bandwidth)
        & np.isfinite(f2_bandwidth)
        & (f1_bandwidth <= unvoiced_max_formant_bandwidth_hz)
        & (f2_bandwidth <= unvoiced_max_formant_bandwidth_hz)
    )
    spans: list[PhonationSpan] = []
    for start, end in word_extents:
        frames = (times >= start) & (times < end)
        if not np.any(frames) or float(np.mean((periodic | resonant)[frames])) < min_evidence_fraction:
            continue
        voiced = periodic[frames]
        fraction = float(np.mean(voiced))
        production = (
            "voiced"
            if fraction > mixed_voiced_fraction
            else "unvoiced"
            if fraction < 1.0 - mixed_voiced_fraction
            else "mixed"
        )
        placed = f0[frames][voiced]
        spans.append(
            PhonationSpan(
                start=start,
                end=end,
                member="word_aligned",
                production=production,
                voiced_fraction=fraction,
                f0_median_hz=float(np.median(placed)) if len(placed) else None,
                f0_start_hz=float(placed[0]) if len(placed) else None,
                f0_end_hz=float(placed[-1]) if len(placed) else None,
                glide_direction=None,
                glide_extent_cents=None,
                offset_criterion="word_boundary",
            )
        )
    return spans
