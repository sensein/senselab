"""Smoke tests for the IBM Granite Speech 3.3 in-process backend.

Skipped automatically when the model weights are not in the local
HuggingFace cache (~16 GB) — we don't trigger downloads in CI. When
the weights are available these tests verify only the senselab API
contract for the new backend — return type, ScriptLine shape,
text-only output (Granite has no native timestamps) — not transcription
quality. We use a real-speech fixture from
``src/tests/data_for_testing/`` so the model has a valid input.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.preprocessing import downmix_audios_to_mono, resample_audios
from senselab.audio.tasks.speech_to_text.granite import GraniteSpeechASR
from senselab.utils.data_structures import HFModel

REPO_ROOT = Path(__file__).resolve().parents[5]
FIXTURE_WAV = REPO_ROOT / "src" / "tests" / "data_for_testing" / "audio_48khz_mono_16bits.wav"
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
GRANITE_CACHE_DIR = HF_CACHE / "models--ibm-granite--granite-speech-3.3-8b"

granite_available = GRANITE_CACHE_DIR.exists()


def _load_16k_mono_fixture() -> Audio:
    audio = Audio(filepath=str(FIXTURE_WAV))
    audio = downmix_audios_to_mono([audio])[0]
    if audio.sampling_rate != 16000:
        audio = resample_audios([audio], resample_rate=16000)[0]
    return audio


@pytest.mark.skipif(
    not granite_available,
    reason=f"ibm-granite/granite-speech-3.3-8b not in HF cache at {GRANITE_CACHE_DIR}",
)
def test_granite_speech_returns_text_only_scriptlines() -> None:
    """transcribe_with_granite returns a list of text-only ScriptLines.

    Asserts the API shape contract only — text is a non-empty string,
    ``start``/``end`` are None, and the chunks list is empty/None
    (Granite does not produce native timestamps; the analyze_audio
    script's auto-align stage adds per-segment timing downstream).
    """
    audio = _load_16k_mono_fixture()
    model = HFModel(path_or_uri="ibm-granite/granite-speech-3.3-8b")

    result = GraniteSpeechASR.transcribe_with_granite(audios=[audio], model=model)

    assert isinstance(result, list)
    assert len(result) == 1
    line = result[0]
    text = getattr(line, "text", None)
    assert isinstance(text, str)
    assert getattr(line, "start", None) is None
    assert getattr(line, "end", None) is None
    chunks = getattr(line, "chunks", None) or []
    assert chunks == [] or chunks is None
