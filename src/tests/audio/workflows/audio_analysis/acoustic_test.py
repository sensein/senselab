"""Acoustic presence voters must be absolutely calibrated, not ranked within the file."""

from __future__ import annotations

import numpy as np
import pytest

from senselab.audio.workflows.audio_analysis.acoustic import (
    loudness_confidence,
    lufs_track,
)

SR = 16000


def test_loudness_is_reported_in_lufs_not_a_within_file_rank() -> None:
    """A percentile-normalised "loudness" is rank, not loudness.

    Measured consequence: with a 10th-percentile floor and 75th-percentile ceiling, ~10% of
    frames pin at 0 and ~25% at 1.0 *by construction*, whatever the audio contains — so a
    uniformly quiet recording still reads as loud, and the value cannot be compared to dBFS
    because the dB→[0,1] mapping differs for every file.
    """
    loud = np.full(SR, 0.3)
    quiet = np.full(SR, 0.003)
    _t, loud_lufs = lufs_track(loud, SR, hop_s=0.1)
    _t2, quiet_lufs = lufs_track(quiet, SR, hop_s=0.1)
    assert loud_lufs[2] > quiet_lufs[2] + 20.0, "40 dB of amplitude difference must show"
    assert loud_lufs[2] < 0.0, "LUFS is negative below full scale"


def test_two_recordings_at_the_same_level_report_the_same_loudness() -> None:
    """The property percentile normalisation destroys, and the reason it cannot meet dBFS."""
    a = np.full(SR, 0.1)
    b = np.concatenate([np.full(SR // 2, 0.1), np.full(SR // 2, 0.1)])
    _ta, la = lufs_track(a, SR, hop_s=0.1)
    _tb, lb = lufs_track(b, SR, hop_s=0.1)
    assert la[2] == pytest.approx(lb[2], abs=0.5)


def test_a_quiet_frame_reads_as_low_confidence() -> None:
    """The inversion: a quiet signal must not report high speech confidence."""
    assert loudness_confidence(-70.0) < 0.15


def test_a_speech_level_frame_reads_as_high_confidence() -> None:
    """Conversational speech sits near -23 LUFS, which should read as clearly present."""
    assert loudness_confidence(-23.0) > 0.7


def test_confidence_is_monotonic_in_level() -> None:
    """Anything else means the mapping is not reading level."""
    levels = [-80.0, -60.0, -40.0, -30.0, -20.0, -10.0]
    values = [loudness_confidence(x) for x in levels]
    assert values == sorted(values)


def test_silence_does_not_produce_negative_infinity() -> None:
    """-inf cannot be normalised or plotted; the floor keeps it usable."""
    _t, levels = lufs_track(np.zeros(SR), SR, hop_s=0.1)
    assert np.isfinite(levels).all()
    assert loudness_confidence(float(levels[0])) < 0.1


# ── level above the measured noise floor (D-3, register items 8-9) ────────────


def _tone_in_noise(duration_s: float, *, tone_amp: float, noise_amp: float, sr: int = 16000) -> np.ndarray:
    """A 500 Hz tone over broadband noise; the tone occupies the second half only."""
    t = np.arange(int(duration_s * sr)) / sr
    rng = np.random.default_rng(0)
    noise = noise_amp * rng.standard_normal(t.size)
    tone = np.zeros_like(t)
    half = t.size // 2
    tone[half:] = tone_amp * np.sin(2 * np.pi * 500 * t[half:])
    return (noise + tone).astype(np.float64)


def test_level_above_floor_separates_activity_from_the_noise_floor() -> None:
    """Frames with a source present read well above the recording's own floor."""
    from senselab.audio.workflows.audio_analysis.acoustic import level_above_floor_track

    wav = _tone_in_noise(4.0, tone_amp=0.2, noise_amp=0.01)
    times, excess = level_above_floor_track(wav, 16000, hop_s=0.05)
    assert times.size == excess.size > 0
    half = excess.size // 2
    quiet, active = float(np.median(excess[:half])), float(np.median(excess[half:]))
    assert active - quiet > 10.0, f"tone half only {active - quiet:.1f} dB above the noise half"


def test_level_above_floor_is_gain_invariant_but_lufs_is_not() -> None:
    """Why both signals exist, rather than one standing in for the other.

    Gain scaling changes no signal-to-noise ratio -- it lifts the source and the floor together --
    so an excess-above-floor measure must be unchanged by it. Absolute loudness must *not* be:
    that is what makes LUFS able to say "this recording is quiet". A single signal cannot answer
    both questions, which is what the discarded within-file percentile rank was conflating.
    """
    from senselab.audio.workflows.audio_analysis.acoustic import level_above_floor_track, lufs_track

    wav = _tone_in_noise(4.0, tone_amp=0.2, noise_amp=0.01)
    attenuated = wav * 10.0 ** (-12.0 / 20.0)  # -12 dB

    _t, ex_ref = level_above_floor_track(wav, 16000, hop_s=0.05)
    _t, ex_att = level_above_floor_track(attenuated, 16000, hop_s=0.05)
    assert abs(float(np.median(ex_ref)) - float(np.median(ex_att))) < 1.0, "excess should survive gain"

    _t, lufs_ref = lufs_track(wav, 16000, hop_s=0.05)
    _t, lufs_att = lufs_track(attenuated, 16000, hop_s=0.05)
    assert float(np.median(lufs_ref)) - float(np.median(lufs_att)) == pytest.approx(12.0, abs=0.5)


def test_level_above_floor_handles_digital_silence() -> None:
    """Pure silence has no floor to exceed; the track must not produce inf or nan."""
    from senselab.audio.workflows.audio_analysis.acoustic import level_above_floor_track

    _times, excess = level_above_floor_track(np.zeros(16000 * 2), 16000, hop_s=0.05)
    assert excess.size > 0
    assert np.isfinite(excess).all()
    assert float(np.max(excess)) < 3.0, "silence is not activity"
