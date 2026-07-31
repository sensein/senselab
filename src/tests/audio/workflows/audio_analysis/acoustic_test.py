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
