"""Smoke tests for the Alibaba Qwen3-ASR subprocess-venv backend.

Skipped automatically when the ``qwen-asr`` venv has not been
provisioned (default CI install does not provision it; first invocation
through ``transcribe_audios`` triggers a one-time install of the
``qwen-asr`` PyPI wrapper plus the model weights).

When the venv is locally available these tests verify only the senselab
API contract for the new backend — return type, ScriptLine shape,
presence/absence of word-level chunks driven by ``return_timestamps`` —
not transcription quality. We use a real-speech fixture from
``src/tests/data_for_testing/`` so the worker subprocess has a valid
WAV to feed to the ASR model.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import senselab.audio.tasks.speech_to_text.qwen as qwen_mod
from senselab.audio.data_structures import Audio
from senselab.audio.tasks.preprocessing import downmix_audios_to_mono, resample_audios
from senselab.audio.tasks.speech_to_text.qwen import QwenASR
from senselab.utils.data_structures import HFModel, Language, ScriptLine

REPO_ROOT = Path(__file__).resolve().parents[5]
FIXTURE_WAV = REPO_ROOT / "src" / "tests" / "data_for_testing" / "audio_48khz_mono_16bits.wav"
SENSELAB_VENV_ROOT = Path.home() / ".cache" / "senselab" / "venvs" / "qwen-asr"

qwen_venv_present = SENSELAB_VENV_ROOT.exists()


def _load_16k_mono_fixture() -> Audio:
    audio = Audio(filepath=str(FIXTURE_WAV))
    audio = downmix_audios_to_mono([audio])[0]
    if audio.sampling_rate != 16000:
        audio = resample_audios([audio], resample_rate=16000)[0]
    return audio


@pytest.mark.skipif(
    not qwen_venv_present,
    reason=f"qwen-asr venv not provisioned at {SENSELAB_VENV_ROOT}",
)
def test_qwen_asr_with_timestamps_populates_chunks() -> None:
    """transcribe_with_qwen with return_timestamps=True yields chunks."""
    audio = _load_16k_mono_fixture()
    model: HFModel = HFModel(path_or_uri="Qwen/Qwen3-ASR-1.7B")

    result = QwenASR.transcribe_with_qwen(audios=[audio], model=model, return_timestamps=True)

    assert isinstance(result, list)
    assert len(result) == 1
    line = result[0]
    text = getattr(line, "text", None)
    assert isinstance(text, str)
    chunks = getattr(line, "chunks", None) or []
    # Shape-only: at least one aligned span came back; do NOT assert content.
    assert len(chunks) >= 1
    first = chunks[0]
    assert getattr(first, "start", None) is not None
    assert getattr(first, "end", None) is not None
    assert float(first.end) >= float(first.start)


@pytest.mark.skipif(
    not qwen_venv_present,
    reason=f"qwen-asr venv not provisioned at {SENSELAB_VENV_ROOT}",
)
def test_qwen_asr_without_timestamps_returns_text_only() -> None:
    """transcribe_with_qwen with return_timestamps=False returns text-only ScriptLines."""
    audio = _load_16k_mono_fixture()
    model: HFModel = HFModel(path_or_uri="Qwen/Qwen3-ASR-1.7B")

    result = QwenASR.transcribe_with_qwen(audios=[audio], model=model, return_timestamps=False)

    assert isinstance(result, list)
    assert len(result) == 1
    line = result[0]
    text = getattr(line, "text", None)
    assert isinstance(text, str)
    chunks = getattr(line, "chunks", None) or []
    assert chunks == [] or chunks is None
    assert getattr(line, "start", None) is None
    assert getattr(line, "end", None) is None


def test_align_with_qwen_maps_worker_output_to_scriptlines(monkeypatch: pytest.MonkeyPatch) -> None:
    """align_with_qwen assembles worker word spans into the align_transcriptions shape."""
    monkeypatch.setattr(qwen_mod, "ensure_venv", lambda *a, **k: "/fake/venv")
    monkeypatch.setattr(qwen_mod, "venv_python", lambda *a, **k: "/fake/venv/bin/python")
    monkeypatch.setattr(qwen_mod.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(
        qwen_mod,
        "parse_subprocess_result",
        lambda *a, **k: {
            "results": [
                {
                    "chunks": [
                        {"text": "This", "start": 0.0, "end": 0.3},
                        {"text": "is", "start": 0.3, "end": 0.5},
                        {"text": "Peter", "start": 0.5, "end": 0.9},
                    ]
                }
            ]
        },
    )
    audio = Audio(waveform=torch.zeros(1, 16000, dtype=torch.float32), sampling_rate=16000)
    out = QwenASR.align_with_qwen([(audio, ScriptLine(text="This is Peter"), Language(language_code="en"))])

    assert isinstance(out, list) and len(out) == 1
    inner = out[0]
    assert isinstance(inner, list) and len(inner) == 1
    utt = inner[0]
    assert utt.text == "This is Peter"
    assert utt.start == 0.0 and utt.end == 0.9
    assert utt.chunks is not None and [c.text for c in utt.chunks] == ["This", "is", "Peter"]
