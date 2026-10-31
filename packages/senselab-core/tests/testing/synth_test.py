"""Synthetic generators are deterministic, amplitude-respecting and shaped (channels, samples)."""

from pathlib import Path

import numpy
import pytest
import soundfile
from senselab_core.kernel.signals import AUDIO_GROUP, Audio, RegularGroup
from senselab_core.testing.synth import chirp, noise, tone, write_wav

pytestmark = pytest.mark.logic


def _x(audio: Audio) -> numpy.ndarray:
    group = audio.groups[AUDIO_GROUP]
    assert isinstance(group, RegularGroup)
    return group.read()


def test_tone_frequency_peak() -> None:
    """The FFT peak lies within one bin of the requested frequency."""
    sr = 16_000
    x = _x(tone(440.0, duration_s=1.0, sampling_rate=sr, channels=2))
    assert x.shape == (2, sr) and x.dtype == numpy.float32
    assert numpy.abs(x).max() <= 0.5 + 1e-6
    spectrum = numpy.abs(numpy.fft.rfft(x[0]))
    assert abs(numpy.argmax(spectrum) * sr / x.shape[1] - 440.0) <= sr / x.shape[1]


def test_noise_is_seeded() -> None:
    """Equal seeds give equal noise, different seeds differ."""
    a, b, c = (_x(noise(duration_s=0.5, seed=s)) for s in (1, 1, 2))
    assert numpy.array_equal(a, b) and not numpy.array_equal(a, c)
    assert a.dtype == numpy.float32 and numpy.abs(a).max() <= 1.0


def test_chirp_sweeps() -> None:
    """The spectral peak rises from the first half to the second."""
    sr = 16_000
    x = _x(chirp(200.0, 2000.0, duration_s=1.0, sampling_rate=sr))[0]
    half = len(x) // 2
    peak = [numpy.argmax(numpy.abs(numpy.fft.rfft(h))) * sr / len(h) for h in (x[:half], x[half:])]
    assert peak[0] < peak[1] and peak[0] < 1100 < peak[1]


def test_write_wav_roundtrip(tmp_path: Path) -> None:
    """A written WAV reads back at the same rate and length."""
    path = write_wav(tone(300.0, duration_s=0.1), tmp_path / "t.wav")
    data, sr = soundfile.read(path, dtype="float32")
    assert sr == 16_000 and data.shape == (1600,)
