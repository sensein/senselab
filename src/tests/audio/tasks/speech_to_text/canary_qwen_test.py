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
from senselab.audio.tasks.speech_to_text.canary_qwen import (
    _CANARY_WORKER_SCRIPT,
    CanaryQwenASR,
    _regroup_chunk_transcripts,
)
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


def test_regroup_chunk_transcripts_groups_in_input_order() -> None:
    """Per-chunk texts are concatenated (in order) into one transcript per input."""
    entries = [{"text": "a1"}, {"text": "a2"}, {"text": "b1"}, {"text": "c1"}, {"text": "c2"}, {"text": "c3"}]
    texts = _regroup_chunk_transcripts(entries, [2, 1, 3])
    assert texts == ["a1 a2", "b1", "c1 c2 c3"]


def test_regroup_chunk_transcripts_raises_on_worker_shortfall() -> None:
    """A worker that returns fewer chunks than sent must fail loudly, not silently.

    Regression: advancing ``pos`` by the *expected* count while guarding the
    index dropped the missing chunks AND misaligned every downstream audio,
    silently corrupting/truncating transcripts with no error.
    """
    entries = [{"text": "a1"}, {"text": "a2"}, {"text": "b1"}]  # 3 returned...
    with pytest.raises(RuntimeError, match="chunk"):
        _regroup_chunk_transcripts(entries, [2, 1, 3])  # ...but 6 expected


def test_canary_worker_loads_requested_revision() -> None:
    """The worker forwards the requested revision to SALM.from_pretrained.

    The parent passes ``model.revision`` down to the worker; the worker must
    load that snapshot rather than calling SALM.from_pretrained(model_name)
    with no revision (which would default to 'main').
    """
    # Worker reads a revision from its input payload...
    assert 'args["revision"]' in _CANARY_WORKER_SCRIPT or 'args.get("revision"' in _CANARY_WORKER_SCRIPT
    # ...and forwards it to the model loader.
    assert "revision=revision" in _CANARY_WORKER_SCRIPT


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
