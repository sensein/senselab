"""YAMNet input-path fidelity (T015, FR-017d / FR-019b).

YAMNet runs in an isolated TensorFlow venv, so senselab hands it audio through a temp
WAV file. That serialization is the one lossy step in the path, and it matters more here
than anywhere else in the feature: this classifier is amplitude-sensitive with an absolute
low-level floor, and the whole point of the background work is to present it faint content
it would otherwise report as silence.

The measured failure with a 16-bit write: a −100 dBFS signal reads back at −93 dBFS,
because 16-bit quantization noise is now louder than the content. What reaches the model
is noise wearing the signal's level. Worse, that quantization noise is statistically
indistinguishable from analog broadband noise, so amplifying it produces exactly the
water-like environmental labels the noise-character guard exists to reject.

These tests pin the fidelity of the write path. They do not need TensorFlow or a
checkpoint — the property under test is that quiet content survives serialization.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from senselab.audio.tasks.classification.yamnet import (
    write_worker_wav,
)
from senselab.utils.portable_audio_io import LOSSLESS_WAV_SUBTYPE

SR = 16000


def _rms_dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))
    return 20.0 * math.log10(max(rms, 1e-30))


def _quiet_noise(level_dbfs: float, seconds: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(int(SR * seconds))
    raw /= float(np.sqrt(np.mean(raw**2)))
    return (raw * (10.0 ** (level_dbfs / 20.0))).astype(np.float32)


def test_lossless_subtype_is_float() -> None:
    """A fixed-point write is what destroys faint content, so the path uses float."""
    assert LOSSLESS_WAV_SUBTYPE == "FLOAT"


@pytest.mark.parametrize("level_dbfs", [-40.0, -70.0, -100.0, -120.0])
def test_quiet_content_survives_the_write(tmp_path: Path, level_dbfs: float) -> None:
    """Level is preserved to a fraction of a dB even far below 16-bit resolution.

    −120 dBFS is the decisive row: a 16-bit write returns exact zeros there, which the
    classifier reports as silence with full confidence.
    """
    import soundfile as sf

    wav = _quiet_noise(level_dbfs)
    path = tmp_path / "q.wav"
    write_worker_wav(path, wav, SR)
    back, sr = sf.read(path, dtype="float32")
    assert sr == SR
    assert _rms_dbfs(back) == pytest.approx(level_dbfs, abs=0.5)


def test_write_is_bit_exact(tmp_path: Path) -> None:
    """No requantization at all — the residual must reach the model unaltered."""
    import soundfile as sf

    wav = _quiet_noise(-90.0)
    path = tmp_path / "e.wav"
    write_worker_wav(path, wav, SR)
    back, _ = sf.read(path, dtype="float32")
    assert np.array_equal(back, wav)


def test_sixteen_bit_would_have_destroyed_it(tmp_path: Path) -> None:
    """Documents the defect being fixed, so a regression is recognizable.

    At −100 dBFS a 16-bit round-trip returns something ~7 dB *louder* than the input,
    because what comes back is quantization noise rather than the signal.
    """
    import soundfile as sf

    wav = _quiet_noise(-100.0)
    lossy = tmp_path / "pcm16.wav"
    sf.write(lossy, wav, SR, subtype="PCM_16")
    back, _ = sf.read(lossy, dtype="float32")
    assert _rms_dbfs(back) > _rms_dbfs(wav) + 3.0, "expected quantization noise to dominate"


def test_clipping_is_reported_not_silently_clamped(tmp_path: Path) -> None:
    """Amplified-past-full-scale input must be visible, not quietly repaired (FR-017d)."""
    wav = (_quiet_noise(-20.0) * 100.0).astype(np.float32)
    report = write_worker_wav(tmp_path / "c.wav", wav, SR)
    assert report["clipped_fraction"] > 0.0
    assert report["requantized"] is False


def test_clean_write_reports_no_clipping(tmp_path: Path) -> None:
    """The common case reports zero rather than omitting the field."""
    report = write_worker_wav(tmp_path / "ok.wav", _quiet_noise(-30.0), SR)
    assert report["clipped_fraction"] == pytest.approx(0.0)
    assert report["subtype"] == LOSSLESS_WAV_SUBTYPE


def test_multichannel_input_is_collapsed_to_mono(tmp_path: Path) -> None:
    """YAMNet expects mono; a stray channel dimension must not change the level."""
    import soundfile as sf

    mono = _quiet_noise(-40.0)
    write_worker_wav(tmp_path / "m.wav", mono.reshape(1, -1), SR)
    back, _ = sf.read(tmp_path / "m.wav", dtype="float32")
    assert back.ndim == 1
    assert _rms_dbfs(back) == pytest.approx(-40.0, abs=0.5)
