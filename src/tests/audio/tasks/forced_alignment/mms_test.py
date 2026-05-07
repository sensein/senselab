"""Smoke tests for the MMS forced-alignment backend.

Skipped automatically when:
- The local HuggingFace cache does not contain ``facebook/mms-1b-all``
  (the weights are ~1.6 GB; we don't trigger downloads in CI).
- ``uroman`` is not installed (the ja/zh-romanization test only).

When MMS is locally available these tests verify only the API contract
of ``align_transcriptions(..., aligner_model=MMS_MODEL_ID)`` — the
return type, ScriptLine shape, and language-routing — not the
alignment quality. We use a real-speech fixture from
``src/tests/data_for_testing/`` plus a plausible English transcript;
synthetic transcripts on real audio do not align meaningfully but the
API path still produces well-shaped output.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.forced_alignment import align_transcriptions
from senselab.audio.tasks.forced_alignment.constants import MMS_MODEL_ID
from senselab.audio.tasks.preprocessing import downmix_audios_to_mono, resample_audios
from senselab.utils.data_structures import Language, ScriptLine

REPO_ROOT = Path(__file__).resolve().parents[5]
FIXTURE_WAV = REPO_ROOT / "src" / "tests" / "data_for_testing" / "audio_48khz_mono_16bits.wav"
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
MMS_CACHE_DIR = HF_CACHE / "models--facebook--mms-1b-all"

mms_available = MMS_CACHE_DIR.exists()
uroman_available = importlib.util.find_spec("uroman") is not None


def _load_16k_mono_fixture() -> Audio:
    """Read the standard test WAV and prep it as 16 kHz mono."""
    audio = Audio(filepath=str(FIXTURE_WAV))
    audio = downmix_audios_to_mono([audio])[0]
    if audio.sampling_rate != 16000:
        audio = resample_audios([audio], resample_rate=16000)[0]
    return audio


@pytest.mark.skipif(
    not mms_available,
    reason=f"facebook/mms-1b-all not in HF cache at {MMS_CACHE_DIR}",
)
def test_mms_aligner_english() -> None:
    """MMS-via-aligner_model produces well-shaped ScriptLines for an English transcript."""
    audio = _load_16k_mono_fixture()
    transcript = ScriptLine(text="hello world this is a test")

    result = align_transcriptions(
        audios_and_transcriptions_and_language=[(audio, transcript, Language(language_code="en"))],
        aligner_model=MMS_MODEL_ID,
    )

    # Shape contract: List[List[ScriptLine | None]], one outer list entry per input audio.
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], list)
    # At least one segment came back; do NOT assert on alignment quality.
    assert len(result[0]) >= 1


@pytest.mark.skipif(
    not (mms_available and uroman_available),
    reason="MMS weights or uroman not present; install uroman via `uv sync --extra nlp`",
)
def test_mms_aligner_japanese_exercises_romanization() -> None:
    """A Japanese transcript exercises the romanize path; verify the call does not raise."""
    audio = _load_16k_mono_fixture()
    transcript = ScriptLine(text="こんにちは世界")

    result = align_transcriptions(
        audios_and_transcriptions_and_language=[(audio, transcript, Language(language_code="ja"))],
        aligner_model=MMS_MODEL_ID,
    )
    assert isinstance(result, list)
    assert len(result) == 1


def test_unknown_language_for_mms_raises_actionable_error() -> None:
    """A language code with no ISO_1_TO_3 entry yields a clear ValueError pointing at the registry."""
    audio = _load_16k_mono_fixture()
    transcript = ScriptLine(text="text in unsupported language")
    if not mms_available:
        pytest.skip(f"facebook/mms-1b-all not in HF cache at {MMS_CACHE_DIR}")
    with pytest.raises(ValueError, match="ISO_1_TO_3"):
        align_transcriptions(
            audios_and_transcriptions_and_language=[(audio, transcript, Language(language_code="xx"))],
            aligner_model=MMS_MODEL_ID,
        )
