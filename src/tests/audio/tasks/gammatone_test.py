"""The gammatone filterbank."""

from __future__ import annotations

import numpy as np
import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.gammatone import erb_space, gammatone_filterbank

SR = 16000


def _tone(freq: float, seconds: float = 1.0) -> Audio:
    t = np.arange(int(seconds * SR)) / SR
    return Audio(waveform=(0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)[None, :], sampling_rate=SR)


class TestErbSpacing:
    """Centre frequencies sit on the ERB-rate scale."""

    def test_centres_span_the_requested_range_and_increase(self) -> None:
        """Forty centres run 80 to 7800 Hz, ascending."""
        cf = erb_space(80.0, 7800.0, 40)
        assert len(cf) == 40
        assert cf[0] == pytest.approx(80.0, abs=1.0)
        assert cf[-1] == pytest.approx(7800.0, abs=10.0)
        assert np.all(np.diff(cf) > 0)

    def test_spacing_is_wider_at_high_frequency(self) -> None:
        """ERB spacing widens with frequency, unlike a linear grid."""
        cf = erb_space(80.0, 7800.0, 40)
        assert (cf[-1] - cf[-2]) > (cf[1] - cf[0]) * 5


class TestFilterbank:
    """The bank resolves frequency into the right channel at the right shape."""

    def test_a_tone_excites_the_channel_nearest_its_frequency(self) -> None:
        """A 1 kHz tone lands in the channel centred nearest 1 kHz."""
        cf, energy = gammatone_filterbank(_tone(1000.0), n_channels=40, low_hz=80.0, high_hz=7800.0, hop_s=0.005)
        loudest = int(np.argmax(energy.mean(axis=1)))
        assert abs(cf[loudest] - 1000.0) < 250.0

    def test_shape_is_channels_by_frames(self) -> None:
        """The output is (n_channels, n_frames) with the centres alongside."""
        cf, energy = gammatone_filterbank(
            _tone(1000.0, seconds=2.0), n_channels=24, low_hz=80.0, high_hz=7800.0, hop_s=0.01
        )
        assert energy.shape[0] == 24 == len(cf)
        assert energy.shape[1] == pytest.approx(2.0 / 0.01, abs=2)
