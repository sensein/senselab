"""The Hilbert envelope in dBFS and its rolling local floor."""

from __future__ import annotations

import numpy as np
import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.envelope import hilbert_envelope_dbfs, rolling_floor_dbfs

SR = 16000


def _tone(seconds: float, amp: float, freq: float = 200.0) -> Audio:
    t = np.arange(int(seconds * SR)) / SR
    return Audio(waveform=(amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)[None, :], sampling_rate=SR)


def _env(audio: Audio) -> np.ndarray:
    """Run the envelope with the config's measured values; the fixture supplies the literals."""
    return hilbert_envelope_dbfs(audio, lowpass_hz=40.0, filter_order=4)


class TestEnvelopeIsAbsolute:
    """The envelope is dBFS, never normalised by the input's own maximum."""

    def test_a_half_scale_tone_sits_near_minus_six_dbfs(self) -> None:
        """A 0.5-amplitude tone reads close to -6 dBFS, pinning the absolute reference."""
        env = _env(_tone(1.0, 0.5))
        mid = env[SR // 4 : -SR // 4]
        assert -7.5 < float(np.median(mid)) < -4.5

    def test_scaling_the_input_shifts_the_envelope_by_the_same_amount(self) -> None:
        """A 20 dB input change moves the envelope 20 dB — absolute, not max-normalised."""
        loud = float(np.median(_env(_tone(1.0, 0.5))[SR // 4 : -SR // 4]))
        quiet = float(np.median(_env(_tone(1.0, 0.05))[SR // 4 : -SR // 4]))
        assert loud - quiet == pytest.approx(20.0, abs=1.0), "dBFS is absolute, not max-normalised"

    def test_a_loud_click_elsewhere_does_not_move_the_rest(self) -> None:
        """One 30 ms click cannot rescale the envelope far from it."""
        quiet = _tone(2.0, 0.05)
        clicked = quiet.waveform.numpy().copy() if hasattr(quiet.waveform, "numpy") else np.array(quiet.waveform)
        clicked[0, SR : SR + 480] += 0.95
        a = _env(quiet)
        b = _env(Audio(waveform=clicked.astype(np.float32), sampling_rate=SR))
        early = slice(SR // 8, SR // 2)
        assert float(np.median(b[early])) == pytest.approx(float(np.median(a[early])), abs=0.5)


class TestRollingFloor:
    """The floor tracks the recording rather than summarising it."""

    def test_the_floor_tracks_a_level_change_rather_than_averaging_it(self) -> None:
        """A -60 to -30 dB step moves the floor to each level, not to their mean."""
        env = np.concatenate([np.full(5 * SR, -60.0), np.full(5 * SR, -30.0)])
        fl = rolling_floor_dbfs(env, SR, window_s=1.0, percentile=10.0, eval_grid_s=0.1)
        assert fl[SR] == pytest.approx(-60.0, abs=1.0)
        assert fl[9 * SR] == pytest.approx(-30.0, abs=1.0)

    def test_the_floor_is_one_value_per_sample(self) -> None:
        """The floor track has the envelope's own shape."""
        env = np.full(3 * SR, -50.0)
        assert rolling_floor_dbfs(env, SR, window_s=3.0, percentile=10.0, eval_grid_s=0.1).shape == env.shape
