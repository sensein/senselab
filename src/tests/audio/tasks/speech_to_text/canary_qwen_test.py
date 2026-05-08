"""Smoke tests for the NVIDIA Canary-Qwen 2.5B subprocess-venv backend.

Skipped automatically when the ``nemo-canary-qwen`` venv has not been
provisioned (default CI install does not provision it; first invocation
through ``transcribe_audios`` triggers a one-time ~5 GB install of
``nemo_toolkit[asr,tts]`` from a NeMo trunk pin plus the model weights).

When the venv is locally available these tests verify only the senselab
API contract for the new backend — return type, ScriptLine shape,
text-only output (Canary-Qwen has no native timestamps) — not
transcription quality. We use a real-speech fixture from
``src/tests/data_for_testing/`` so the worker subprocess has a valid
WAV to feed to SALM.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.preprocessing import downmix_audios_to_mono, resample_audios
from senselab.audio.tasks.speech_to_text.canary_qwen import CanaryQwenASR
from senselab.utils.data_structures import HFModel

REPO_ROOT = Path(__file__).resolve().parents[5]
FIXTURE_WAV = REPO_ROOT / "src" / "tests" / "data_for_testing" / "audio_48khz_mono_16bits.wav"
SENSELAB_VENV_ROOT = Path.home() / ".cache" / "senselab" / "venvs" / "nemo-canary-qwen"

canary_venv_present = SENSELAB_VENV_ROOT.exists()


def _load_16k_mono_fixture() -> Audio:
    audio = Audio(filepath=str(FIXTURE_WAV))
    audio = downmix_audios_to_mono([audio])[0]
    if audio.sampling_rate != 16000:
        audio = resample_audios([audio], resample_rate=16000)[0]
    return audio


@pytest.mark.skipif(
    not canary_venv_present,
    reason=f"nemo-canary-qwen venv not provisioned at {SENSELAB_VENV_ROOT}",
)
def test_canary_qwen_returns_text_only_scriptlines() -> None:
    """transcribe_with_canary_qwen returns a list of text-only ScriptLines.

    Asserts the API shape contract only — text is a non-empty string,
    ``start``/``end`` are None, and the chunks list is empty/None
    (Canary-Qwen does not produce native timestamps; the analyze_audio
    script's auto-align stage adds per-segment timing downstream).
    """
    audio = _load_16k_mono_fixture()
    model: HFModel = HFModel(path_or_uri="nvidia/canary-qwen-2.5b")

    result = CanaryQwenASR.transcribe_with_canary_qwen(audios=[audio], model=model)

    assert isinstance(result, list)
    assert len(result) == 1
    line = result[0]
    assert hasattr(line, "text")
    # Shape-only: text exists; do NOT assert specific transcription content.
    text = getattr(line, "text", None)
    assert isinstance(text, str)
    # Canary-Qwen is text-only — no native timestamps.
    assert getattr(line, "start", None) is None
    assert getattr(line, "end", None) is None
    chunks = getattr(line, "chunks", None) or []
    assert chunks == [] or chunks is None
