"""Tests for the CrisperWhisper 2.0 subprocess-venv backend.

The assembly test is hermetic (worker + venv mocked): it verifies the
worker-output → ScriptLine mapping, including per-word / line ``score`` and the
line span derived from word timestamps. The integration test runs the real
model only when the ``crisperwhisper`` venv is already provisioned (skipped in
default CI).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import senselab.audio.tasks.speech_to_text.crisperwhisper as cw
from senselab.audio.data_structures import Audio
from senselab.audio.tasks.preprocessing import downmix_audios_to_mono, resample_audios
from senselab.utils.data_structures import HFModel

REPO_ROOT = Path(__file__).resolve().parents[5]
FIXTURE_WAV = REPO_ROOT / "src" / "tests" / "data_for_testing" / "audio_48khz_mono_16bits.wav"
CRISPER_VENV = Path.home() / ".cache" / "senselab" / "venvs" / "crisperwhisper"


def test_worker_output_maps_to_scriptlines(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake worker output assembles into a ScriptLine with word chunks + scores."""
    monkeypatch.setattr(cw, "ensure_venv", lambda *a, **k: "/fake/venv")
    monkeypatch.setattr(cw, "venv_python", lambda *a, **k: "/fake/venv/bin/python")
    monkeypatch.setattr(cw.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(
        cw,
        "parse_subprocess_result",
        lambda *a, **k: {
            "results": [
                {
                    "text": "This is Peter",
                    "language": "en",
                    "score": 0.9,
                    "words": [
                        {"text": "This", "start": 0.0, "end": 0.2, "score": 0.95},
                        {"text": "is", "start": 0.2, "end": 0.3, "score": None},
                        {"text": "Peter", "start": 0.3, "end": 0.9, "score": 0.8},
                    ],
                }
            ]
        },
    )
    audio = Audio(waveform=torch.zeros(1, 16000, dtype=torch.float32), sampling_rate=16000)
    # model=None avoids constructing an HFModel (which would validate against the Hub).
    out = cw.CrisperWhisperASR.transcribe_with_crisperwhisper([audio], model=None)

    assert len(out) == 1
    sl = out[0]
    assert sl.text == "This is Peter"
    assert sl.score == 0.9
    assert sl.chunks is not None and len(sl.chunks) == 3
    assert sl.chunks[0].text == "This" and sl.chunks[0].score == 0.95
    assert sl.chunks[1].score is None  # native confidence absent for a word → None
    assert sl.start == 0.0 and sl.end == 0.9


def test_backend_selection_is_platform_appropriate() -> None:
    """CT2 on Linux x86_64, transformers elsewhere (both valid crisperwhisper backends)."""
    assert cw._CRISPER_BACKEND in ("ct2", "transformers")
    if cw._IS_LINUX_X86:
        assert cw._CRISPER_BACKEND == "ct2"
    else:
        assert cw._CRISPER_BACKEND == "transformers"


@pytest.mark.skipif(not CRISPER_VENV.exists(), reason=f"crisperwhisper venv not provisioned at {CRISPER_VENV}")
def test_crisperwhisper_transcribes_when_venv_present() -> None:
    """Integration: real model yields verbatim text + word-level chunks (shape only)."""
    audio = Audio(filepath=str(FIXTURE_WAV))
    audio = downmix_audios_to_mono([audio])[0]
    if audio.sampling_rate != 16000:
        audio = resample_audios([audio], resample_rate=16000)[0]
    out = cw.CrisperWhisperASR.transcribe_with_crisperwhisper(
        [audio], model=HFModel(path_or_uri="nyralabs/CrisperWhisper2.0_turbo")
    )
    assert len(out) == 1
    line = out[0]
    assert isinstance(line.text, str) and line.text
    chunks = line.chunks or []
    assert len(chunks) >= 1
    assert chunks[0].start is not None and chunks[0].end is not None
