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
    """The spectral peak in each of 6 Hann-windowed segments follows `f0 + (f1 - f0) * t / T` to within one FFT bin."""
    sr, f0, f1, dur, windows = 16_000, 200.0, 2000.0, 1.0, 6
    x = _x(chirp(f0, f1, duration_s=dur, sampling_rate=sr))[0]
    size = len(x) // windows
    bin_hz = sr / size
    for k in range(windows):
        seg = x[k * size : (k + 1) * size]
        peak = numpy.argmax(numpy.abs(numpy.fft.rfft(seg * numpy.hanning(size)))) * bin_hz
        centre = (k + 0.5) * size / sr
        assert abs(peak - (f0 + (f1 - f0) * centre / dur)) <= bin_hz


def test_write_wav_roundtrip(tmp_path: Path) -> None:
    """A written WAV reads back at the same rate and length."""
    path = write_wav(tone(300.0, duration_s=0.1), tmp_path / "t.wav")
    data, sr = soundfile.read(path, dtype="float32")
    assert sr == 16_000 and data.shape == (1600,)
