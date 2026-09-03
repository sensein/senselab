"""The HNR track and the glottal period marks, both through Praat."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.signal import lfilter

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.phonation import (
    FormantTrack,
    PeriodMark,
    f0_track,
    formant_track,
    hnr_track,
    period_marks,
    propose_phonation_spans,
    propose_word_aligned_phonation_spans,
)

SR = 16000


def _buzz(f0: float, seconds: float = 1.0) -> Audio:
    """Return a harmonic buzz at the given F0."""
    t = np.arange(int(seconds * SR)) / SR
    wave = sum((0.3 / (h + 1) * np.sin(2 * np.pi * f0 * (h + 1) * t) for h in range(6)), np.zeros_like(t))
    return Audio(waveform=wave.astype(np.float32)[None, :], sampling_rate=SR)


def _noise(seconds: float = 1.0) -> Audio:
    """Return seeded white noise."""
    rng = np.random.default_rng(0)
    return Audio(waveform=(0.1 * rng.standard_normal(int(seconds * SR))).astype(np.float32)[None, :], sampling_rate=SR)


def _vowel(f0: float, n_samples: int, formants: tuple[float, ...] = (700.0, 1200.0, 2600.0)) -> np.ndarray:
    """A pulse train at ``f0`` through one two-pole resonator per formant, peak-normalised.

    A three-sine fixture will not do here: Praat's Burg analysis has no stable pole to place on a
    line spectrum, so its F1 wanders between the components frame to frame.
    """
    excitation = np.zeros(n_samples)
    excitation[:: max(1, int(round(SR / f0)))] = 1.0
    out = excitation
    for centre in formants:
        r = float(np.exp(-np.pi * 80.0 / SR))
        out = lfilter([1.0 - r], [1.0, -2 * r * np.cos(2 * np.pi * centre / SR), r * r], out)
    return np.asarray(out / (np.abs(out).max() + 1e-12), dtype=np.float64)


def _hnr(audio: Audio) -> np.ndarray:
    """Run hnr_track with Praat's documented cc settings; the fixture supplies the literals."""
    _, hnr_db = hnr_track(audio, f0_min_hz=60.0, hop_s=0.01, silence_threshold=0.1, periods_per_window=4.5)
    return hnr_db


class TestHnr:
    """The gate's measurement is harmonicity in dB, from Praat's cc method."""

    def test_a_buzz_is_harmonic_and_noise_is_not(self) -> None:
        """A 100 Hz harmonic buzz reads far above 20 dB HNR; white noise reads below 0 dB."""
        assert float(np.median(_hnr(_buzz(100.0)))) > 20.0
        assert float(np.median(_hnr(_noise()))) < 0.0

    def test_the_track_carries_its_own_times(self) -> None:
        """Times and values come back paired, equally long and increasing."""
        times, hnr_db = hnr_track(
            _buzz(100.0), f0_min_hz=60.0, hop_s=0.01, silence_threshold=0.1, periods_per_window=4.5
        )
        assert times.shape == hnr_db.shape
        assert np.all(np.diff(times) > 0)

    def test_every_parameter_is_required(self) -> None:
        """No default stands in for a value the caller did not choose."""
        with pytest.raises(TypeError):
            hnr_track(_buzz(100.0))  # type: ignore[call-arg]


class TestPeriodMarks:
    """Pulse times come from Praat's point process, not from integer-lag multiples."""

    def test_marks_are_spaced_by_one_period(self) -> None:
        """A 100 Hz buzz yields pulses one 10 ms period apart across the span."""
        marks = period_marks(_buzz(100.0), 0.2, 0.8, f0_min_hz=60.0, f0_max_hz=400.0)
        assert len(marks) > 40
        gaps = np.diff([m.time_s for m in marks])
        assert float(np.median(gaps)) == pytest.approx(0.01, abs=0.001)

    def test_each_mark_carries_its_period_and_amplitude(self) -> None:
        """Jitter and shimmer read consecutive periods, so each mark carries both quantities."""
        marks = period_marks(_buzz(100.0), 0.2, 0.4, f0_min_hz=60.0, f0_max_hz=400.0)
        m = marks[len(marks) // 2]
        assert isinstance(m, PeriodMark)
        assert m.period_s == pytest.approx(0.01, abs=0.002)
        assert m.amplitude > 0.0

    def test_mark_times_stay_in_the_recordings_clock(self) -> None:
        """Praat places pulses on the waveform inside the span, in absolute time."""
        marks = period_marks(_buzz(100.0), 0.2, 0.8, f0_min_hz=60.0, f0_max_hz=400.0)
        assert 0.2 <= marks[0].time_s <= 0.25
        assert marks[-1].time_s <= 0.8

    def test_noise_yields_no_marks(self) -> None:
        """White noise gives Praat no periodic pulses, so the answer is absent, not zero."""
        assert period_marks(_noise(), 0.2, 0.8, f0_min_hz=60.0, f0_max_hz=400.0) == []


class TestF0Track:
    """F0 travels with the periodicity that placed it: NaN where unvoiced, strength always."""

    def test_f0_track_places_a_steady_tone_and_keeps_strength_with_f0(self) -> None:
        """A synthetic 220 Hz tone reads near 220 Hz where voiced; unvoiced frames are NaN with strength."""
        sr = 16000
        t = np.arange(sr * 2) / sr
        tone = (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        tone[: sr // 2] = 0.0  # half a second of silence first
        audio = Audio(waveform=tone[None, :], sampling_rate=sr)
        times, f0, strength = f0_track(audio, f0_min_hz=100.0, f0_max_hz=400.0, hop_s=0.01)
        assert times.shape == f0.shape == strength.shape
        voiced = ~np.isnan(f0)
        assert np.median(f0[voiced]) == pytest.approx(220.0, rel=0.02)
        assert np.isnan(f0[0]) and not np.isnan(strength[0]), "unvoiced is NaN f0, strength retained"

    def test_every_parameter_is_required(self) -> None:
        """No default stands in for a value the caller did not choose."""
        with pytest.raises(TypeError):
            f0_track(_buzz(100.0))  # type: ignore[call-arg]


class TestFormantTrack:
    """Formants over the whole stream, on the analysis hop, four per frame with bandwidths."""

    def test_a_synthetic_vowel_yields_four_formants_per_frame(self) -> None:
        """Every returned array is the same length and carries F1-F4 with their bandwidths."""
        sr = 16000
        t = np.arange(int(0.5 * sr)) / sr
        wave = sum(np.sin(2 * np.pi * f * t) for f in (120.0, 700.0, 1200.0, 2600.0, 3400.0))
        audio = Audio(waveform=torch.tensor(wave, dtype=torch.float32).unsqueeze(0), sampling_rate=sr)
        track = formant_track(
            audio,
            hop_s=0.01,
            max_formants=5,
            formant_max_hz=5000.0,
            window_s=0.025,
            preemphasis_hz=50.0,
        )
        lengths = {
            len(track.times_s),
            len(track.f_hz[0]),
            len(track.f_hz[3]),
            len(track.bandwidth_hz[0]),
            len(track.bandwidth_hz[3]),
        }
        assert len(lengths) == 1
        assert len(track.f_hz) == 4
        assert len(track.bandwidth_hz) == 4

    def test_tracking_a_slice_and_slicing_the_track_are_not_the_same_measurement(self) -> None:
        """The whole point of tracking once on the stream: a fragment renormalises to its own maximum.

        The fixture is a stream whose second half is 20 dB louder than its first, so a track computed
        on the quiet fragment alone sees a different dynamic range from the same interval sliced out
        of the stream's track. Both are compared here explicitly, which is what makes this test say
        something -- a test that only checked the sliced track against itself would pass under either
        implementation.
        """
        sr = 16000
        n_samples = int(2.0 * sr)
        wave = _vowel(120.0, n_samples)
        wave = np.where(np.arange(n_samples) < n_samples // 2, wave * 0.1, wave)
        audio = Audio(waveform=torch.tensor(wave, dtype=torch.float32).unsqueeze(0), sampling_rate=sr)
        whole = formant_track(
            audio, hop_s=0.01, max_formants=5, formant_max_hz=5000.0, window_s=0.025, preemphasis_hz=50.0
        )
        quiet = Audio(waveform=audio.waveform[:, : n_samples // 2], sampling_rate=sr)
        fragment = formant_track(
            quiet, hop_s=0.01, max_formants=5, formant_max_hz=5000.0, window_s=0.025, preemphasis_hz=50.0
        )
        sliced = whole.f_hz[0][(whole.times_s >= 0.0) & (whole.times_s < 1.0)]
        n = min(len(sliced), len(fragment.f_hz[0]))
        assert n > 50
        assert np.nanmedian(sliced[:n]) == pytest.approx(np.nanmedian(fragment.f_hz[0][:n]), rel=0.15)
        assert len(whole.times_s) > len(fragment.times_s)


def _tracks(
    f0_hz: np.ndarray,
    f1_hz: np.ndarray,
    strength: np.ndarray,
    hop_s: float = 0.01,
    formant_bandwidth_hz: float = 60.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, FormantTrack]:
    """Synthetic tracks on one hop: F2 follows F1 so both stability limbs move together."""
    times = np.arange(len(f0_hz)) * hop_s
    nan = np.full(len(f0_hz), np.nan)
    return (
        times,
        f0_hz,
        strength,
        FormantTrack(
            times_s=times,
            f_hz=(f1_hz, f1_hz * 2.0, nan, nan),
            bandwidth_hz=(
                np.full(len(times), formant_bandwidth_hz),
                np.full(len(times), formant_bandwidth_hz),
                nan,
                nan,
            ),
        ),
    )


_SPAN_RULE = {
    "hop_s": 0.01,
    "f0_stability_cents": 30.0,
    "formant_stability_hz": 20.0,
    "glide_min_excursion_cents": 200.0,
    "hangover_ms": 50.0,
    "voicing_strength_floor": 0.5,
    "mixed_voiced_fraction": 0.6,
    "unvoiced_max_formant_bandwidth_hz": 250.0,
}


class TestProposePhonationSpans:
    """The continuity criterion, the two members, and production across voiced and unvoiced."""

    def test_a_steady_f0_run_is_one_sustained_voiced_span(self) -> None:
        """Stable F0 satisfies its limb on its own, and strength over the floor makes it voiced."""
        n = 100
        times, f0, strength, formants = _tracks(np.full(n, 200.0), np.full(n, 700.0), np.full(n, 0.9))
        [span] = propose_phonation_spans(times=times, f0_hz=f0, strength=strength, formants=formants, **_SPAN_RULE)
        assert span.member == "sustained"
        assert span.production == "voiced"
        assert span.voiced_fraction == pytest.approx(1.0)
        assert span.end - span.start == pytest.approx(1.0, abs=0.02)
        assert span.offset_criterion == "stream_end"

    def test_an_unvoiced_sustain_is_a_span_carried_by_the_formant_limb(self) -> None:
        """No F0 anywhere, stable formants throughout: a span opens and reads unvoiced."""
        n = 100
        times, f0, strength, formants = _tracks(np.full(n, np.nan), np.full(n, 700.0), np.full(n, 0.1))
        [span] = propose_phonation_spans(times=times, f0_hz=f0, strength=strength, formants=formants, **_SPAN_RULE)
        assert span.member == "sustained"
        assert span.production == "unvoiced"
        assert span.f0_median_hz is None

    def test_the_unvoiced_formant_limb_rejects_broadband_structure(self) -> None:
        """Stable LPC poles alone are not phonation when F1/F2 are too broad to be resonant evidence."""
        n = 100
        times, f0, strength, formants = _tracks(
            np.full(n, np.nan),
            np.full(n, 700.0),
            np.full(n, 0.1),
            formant_bandwidth_hz=600.0,
        )
        assert propose_phonation_spans(times=times, f0_hz=f0, strength=strength, formants=formants, **_SPAN_RULE) == []

    def test_a_span_voiced_for_half_its_frames_is_mixed(self) -> None:
        """Between the two cutoffs is its own production, not rounded to the nearer one.

        A disordered voice sustains with little or no periodicity, so `mixed` is the reading the
        detector exists to be able to give; collapsing it into voiced or unvoiced would lose exactly
        the voices worth measuring.
        """
        n = 100
        strength = np.concatenate([np.full(n // 2, 0.9), np.full(n // 2, 0.1)])
        times, f0, strength, formants = _tracks(np.full(n, 200.0), np.full(n, 700.0), strength)
        [span] = propose_phonation_spans(times=times, f0_hz=f0, strength=strength, formants=formants, **_SPAN_RULE)
        assert span.voiced_fraction == pytest.approx(0.5)
        assert span.production == "mixed"

    def test_a_monotone_run_that_fails_both_limbs_is_a_glide(self) -> None:
        """F0 rising 100 cents a hop breaks stability; its monotone excursion makes it a glide."""
        n = 40
        f0 = 150.0 * 2.0 ** (np.arange(n) * 100.0 / 1200.0)
        times, f0, strength, formants = _tracks(f0, f0 * 4.0, np.full(n, 0.9))
        spans = propose_phonation_spans(times=times, f0_hz=f0, strength=strength, formants=formants, **_SPAN_RULE)
        assert [s.member for s in spans] == ["glide"]
        assert spans[0].glide_direction == "rising"
        assert spans[0].glide_extent_cents == pytest.approx(3900.0, rel=0.01)

    def test_a_gap_shorter_than_the_hangover_does_not_close_the_span(self) -> None:
        """One unstable frame inside a stable run leaves one span, not two."""
        n = 100
        f0 = np.full(n, 200.0)
        f0[50] = 260.0
        times, f0, strength, formants = _tracks(f0, np.full(n, 700.0) + (np.arange(n) == 50) * 500.0, np.full(n, 0.9))
        spans = propose_phonation_spans(times=times, f0_hz=f0, strength=strength, formants=formants, **_SPAN_RULE)
        assert len(spans) == 1

    def test_every_parameter_is_required(self) -> None:
        """No default stands in for a value the caller did not choose."""
        times, f0, strength, formants = _tracks(np.full(10, 200.0), np.full(10, 700.0), np.full(10, 0.9))
        with pytest.raises(TypeError):
            propose_phonation_spans(times=times, f0_hz=f0, strength=strength, formants=formants)  # type: ignore[call-arg]


class TestWordAlignedPhonationEvidence:
    """Timed words provide boundaries; their text is not acoustic evidence."""

    def test_an_aperiodic_resonant_word_is_an_unvoiced_span(self) -> None:
        """Narrow, stable resonances inside a timed word are positive acoustic evidence without F0."""
        n = 100
        times, f0, strength, formants = _tracks(np.full(n, np.nan), np.full(n, 700.0), np.full(n, 0.1))
        [span] = propose_word_aligned_phonation_spans(
            times=times,
            f0_hz=f0,
            strength=strength,
            formants=formants,
            word_extents=[(0.1, 0.8)],
            voicing_strength_floor=0.5,
            mixed_voiced_fraction=0.6,
            unvoiced_max_formant_bandwidth_hz=250.0,
            min_evidence_fraction=0.8,
        )
        assert (span.member, span.production, span.offset_criterion) == ("word_aligned", "unvoiced", "word_boundary")

    def test_a_broadband_word_is_not_positive_evidence(self) -> None:
        """A word boundary cannot promote broadband audio when its fitted formants are broad."""
        n = 100
        times, f0, strength, formants = _tracks(
            np.full(n, np.nan), np.full(n, 700.0), np.full(n, 0.1), formant_bandwidth_hz=600.0
        )
        assert (
            propose_word_aligned_phonation_spans(
                times=times,
                f0_hz=f0,
                strength=strength,
                formants=formants,
                word_extents=[(0.1, 0.8)],
                voicing_strength_floor=0.5,
                mixed_voiced_fraction=0.6,
                unvoiced_max_formant_bandwidth_hz=250.0,
                min_evidence_fraction=0.8,
            )
            == []
        )
