"""Deterministic synthetic audio for tests: tones, noise and chirps, written as PCM WAV."""

from pathlib import Path

import numpy
import soundfile

from senselab_core.kernel.signals import AUDIO_GROUP, Audio, RegularGroup


def _frames(duration_s: float, sampling_rate: int) -> int:
    return round(duration_s * sampling_rate)


def tone(
    freq_hz: float, *, duration_s: float, sampling_rate: int = 16_000, channels: int = 1, amplitude: float = 0.5
) -> Audio:
    """Return a sine tone of `freq_hz` as float32 in `[-1, 1]`, shape `(channels, samples)`, identical in each channel.

    Raises:
        ValueError: if `amplitude` is outside `[0, 1]`.
    """
    if not 0.0 <= amplitude <= 1.0:
        raise ValueError(f"amplitude must lie in [0, 1], got {amplitude}")
    t = numpy.arange(_frames(duration_s, sampling_rate)) / sampling_rate
    wave = (amplitude * numpy.sin(2 * numpy.pi * freq_hz * t)).astype(numpy.float32)
    return Audio.from_array(numpy.tile(wave, (channels, 1)), sampling_rate)


def noise(*, duration_s: float, sampling_rate: int = 16_000, seed: int = 0) -> Audio:
    """Return mono Gaussian noise of standard deviation 0.1, clipped to `[-1, 1]`; deterministic in `seed`."""
    rng = numpy.random.default_rng(seed)
    wave = numpy.clip(rng.standard_normal(_frames(duration_s, sampling_rate)) * 0.1, -1.0, 1.0)
    return Audio.from_array(wave.astype(numpy.float32).reshape(1, -1), sampling_rate)


def chirp(f0_hz: float, f1_hz: float, *, duration_s: float, sampling_rate: int = 16_000) -> Audio:
    """Return a mono linear chirp sweeping from `f0_hz` to `f1_hz` at amplitude 0.5, float32."""
    t = numpy.arange(_frames(duration_s, sampling_rate)) / sampling_rate
    phase = 2 * numpy.pi * (f0_hz * t + (f1_hz - f0_hz) / (2 * duration_s) * t**2)
    wave = 0.5 * numpy.sin(phase)
    return Audio.from_array(wave.astype(numpy.float32).reshape(1, -1), sampling_rate)


def write_wav(audio: Audio, path: Path) -> Path:
    """Write `audio` to `path` as 16-bit PCM WAV and return `path`."""
    group = audio.groups[AUDIO_GROUP]
    if not isinstance(group, RegularGroup):
        raise TypeError(f"audio group must be a RegularGroup, got {type(group).__name__}")
    soundfile.write(path, group.read().T, int(group.sampling_rate), subtype="PCM_16", format="WAV")
    return path
