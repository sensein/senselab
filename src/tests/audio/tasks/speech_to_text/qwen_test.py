"""Smoke tests for the Alibaba Qwen3-ASR subprocess-venv backend.

Skipped automatically when the ``qwen-asr`` venv has not been
provisioned (default CI install does not provision it; first invocation
through ``transcribe_audios`` triggers a one-time install of the
``qwen-asr`` PyPI wrapper plus the model weights).

When the venv is locally available these tests verify only the senselab
API contract for the new backend — return type, ScriptLine shape,
speech_presence/absence of word-level chunks driven by ``return_timestamps`` —
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
from senselab.utils.subprocess_venv import provisioned_venv_dirs

REPO_ROOT = Path(__file__).resolve().parents[5]
FIXTURE_WAV = REPO_ROOT / "src" / "tests" / "data_for_testing" / "audio_48khz_mono_16bits.wav"
QWEN_VENVS = provisioned_venv_dirs("qwen-asr")

qwen_venv_present = bool(QWEN_VENVS)


def _load_16k_mono_fixture() -> Audio:
    audio = Audio(filepath=str(FIXTURE_WAV))
    audio = downmix_audios_to_mono([audio])[0]
    if audio.sampling_rate != 16000:
        audio = resample_audios([audio], resample_rate=16000)[0]
    return audio


@pytest.mark.skipif(
    not qwen_venv_present,
    reason="qwen-asr venv not provisioned for this host's device key",
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
    reason="qwen-asr venv not provisioned for this host's device key",
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
    # align_with_qwen resolves the aligner ref to a commit SHA and stages it via
    # hf_subprocess_env before ever reaching subprocess.run; fake both so this test
    # doesn't depend on network reachability or this host's local HF cache state.
    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda *a, **k: "f" * 40)
    monkeypatch.setattr(qwen_mod, "hf_subprocess_env", lambda *a, **k: {})
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


def _stub_qwen_staging(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda *a, **k: True)
    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda *a, **k: "f" * 40)
    monkeypatch.setattr(qwen_mod, "hf_subprocess_env", lambda *a, **k: {"HF_HUB_OFFLINE": "1"})
    monkeypatch.setattr(qwen_mod, "ensure_venv", lambda *a, **k: "/fake/venv")
    monkeypatch.setattr(qwen_mod, "venv_python", lambda *a, **k: "/fake/venv/bin/python")


def _capture_serve(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    def _serve(identity: tuple, **k: object) -> dict:
        calls.append({"identity": identity, **k})
        return {"results": [{"text": "hello", "chunks": [{"text": "hello", "start": 0.0, "end": 0.5}]}]}

    monkeypatch.setattr(qwen_mod, "serve_in_venv", _serve)
    return calls


def test_mps_loads_half_precision_in_short_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    """On MPS the worker gets float16, 30 s chunks, batch 1 and the allocator limits."""
    from senselab.utils.data_structures import DeviceType

    _stub_qwen_staging(monkeypatch)
    calls = _capture_serve(monkeypatch)
    audio = Audio(waveform=torch.zeros(1, 16000), sampling_rate=16000)
    model: HFModel = HFModel(path_or_uri="Qwen/Qwen3-ASR-1.7B", revision="main")
    QwenASR.transcribe_with_qwen([audio], model=model, device=DeviceType.MPS)

    [call] = calls
    init = call["init"]
    assert init["device"] == "mps" and init["dtype"] == "float16"
    assert init["max_chunk_sec"] == 30.0 and init["max_inference_batch_size"] == 1
    assert call["env"]["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == qwen_mod.MPS_ENV["PYTORCH_MPS_HIGH_WATERMARK_RATIO"]
    assert call["env"]["HF_HUB_OFFLINE"] == "1"
    assert "float16" in call["identity"]


def test_cpu_and_cuda_payloads_are_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Off MPS no dtype/chunk/batch keys reach the worker, so the wrapper keeps its own defaults."""
    from senselab.utils.data_structures import DeviceType

    _stub_qwen_staging(monkeypatch)
    calls = _capture_serve(monkeypatch)
    audio = Audio(waveform=torch.zeros(1, 16000), sampling_rate=16000)
    model: HFModel = HFModel(path_or_uri="Qwen/Qwen3-ASR-1.7B", revision="main")
    for device in (DeviceType.CPU, DeviceType.CUDA):
        QwenASR.transcribe_with_qwen([audio], model=model, device=device)
    for call in calls:
        assert not {"dtype", "max_chunk_sec", "max_inference_batch_size"} & set(call["init"])
        assert "PYTORCH_MPS_HIGH_WATERMARK_RATIO" not in call["env"]
        assert "default" in call["identity"]


def test_chunk_override_and_one_request_per_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """``max_chunk_sec`` overrides the device default; each file is its own request, timeout scaled."""
    from senselab.utils.data_structures import DeviceType

    _stub_qwen_staging(monkeypatch)
    calls = _capture_serve(monkeypatch)
    short = Audio(waveform=torch.zeros(1, 16000), sampling_rate=16000)
    long = Audio(waveform=torch.zeros(1, 16000 * 1000), sampling_rate=16000)
    model: HFModel = HFModel(path_or_uri="Qwen/Qwen3-ASR-1.7B", revision="main")
    out = QwenASR.transcribe_with_qwen([short, long], model=model, device=DeviceType.CPU, max_chunk_sec=45)

    assert len(out) == 2 and len(calls) == 2
    assert all(c["init"]["max_chunk_sec"] == 45.0 for c in calls)
    assert all(len(c["request"]["audio_paths"]) == 1 for c in calls)
    assert calls[1]["request_timeout_s"] == pytest.approx(qwen_mod.REQUEST_SECONDS_PER_AUDIO_SECOND * 1000)
    assert out[1].chunks is not None and out[1].chunks[0].timestamp_model == qwen_mod._DEFAULT_FORCED_ALIGNER
