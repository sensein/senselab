"""Verify that the senselab HF ASR path supports return_timestamps=False.

Required for non-Whisper non-CTC HF ASR models (e.g., IBM Granite Speech
3.3) whose underlying `transformers.pipeline("automatic-speech-recognition")`
refuses ``return_timestamps`` requests. Without this path, those models
cannot run through senselab at all; with it, they return text-only
ScriptLines that the analyze_audio script then post-aligns via MMS.

This test guards against:
- The senselab dispatcher silently dropping ``return_timestamps=False``.
- The HuggingFaceASR helper rejecting the bool kwarg.

We use a small, well-known CTC model (``facebook/wav2vec2-base-960h``)
that supports both timestamp modes, NOT Granite Speech itself, because
Granite is multi-GB and our goal here is the parameter-passing
contract, not Granite-specific behavior. A local CTC model already in
the HF cache makes the test cheap.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.preprocessing import downmix_audios_to_mono, resample_audios
from senselab.audio.tasks.speech_to_text.huggingface import HuggingFaceASR
from senselab.utils.data_structures import HFModel

REPO_ROOT = Path(__file__).resolve().parents[5]
FIXTURE_WAV = REPO_ROOT / "src" / "tests" / "data_for_testing" / "audio_48khz_mono_16bits.wav"
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
CTC_CACHE_DIR = HF_CACHE / "models--facebook--wav2vec2-base-960h"

ctc_available = CTC_CACHE_DIR.exists()


def _load_16k_mono_fixture() -> Audio:
    audio = Audio(filepath=str(FIXTURE_WAV))
    audio = downmix_audios_to_mono([audio])[0]
    if audio.sampling_rate != 16000:
        audio = resample_audios([audio], resample_rate=16000)[0]
    return audio


def test_hf_pipeline_forwards_revision_without_collision(monkeypatch: pytest.MonkeyPatch) -> None:
    """_get_hf_asr_pipeline forwards the revision to the loader without collision.

    Regression: ``revision`` was placed in ``pipeline_kwargs`` AND passed as the
    reserved ``revision=`` argument of ``load_hf_resilient``, raising
    ``TypeError: got multiple values for keyword argument 'revision'`` for every
    HF-pipeline transcription. This test builds the pipeline with a fake loader
    (no model download) and asserts the revision reaches it exactly once.
    """
    from senselab.audio.tasks.speech_to_text import huggingface as hf_mod
    from senselab.utils import dependencies
    from senselab.utils.data_structures import DeviceType

    # Not cached -> load_hf_resilient just calls the loader once (no offline toggle).
    monkeypatch.setattr(dependencies, "hf_local_files_only", lambda *a, **k: False)

    recorded: dict = {}

    def _fake_pipeline(**kwargs: object) -> object:
        recorded.update(kwargs)
        return object()

    monkeypatch.setattr(hf_mod, "pipeline", _fake_pipeline)

    class _FakeModel:
        path_or_uri = "openai/whisper-tiny"
        revision = "rev-xyz"

    hf_mod.HuggingFaceASR._pipelines.clear()
    out = hf_mod.HuggingFaceASR._get_hf_asr_pipeline(
        model=_FakeModel(),  # type: ignore[arg-type]
        return_timestamps=False,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=1,
        device=DeviceType.CPU,
    )

    assert out is not None
    assert recorded["revision"] == "rev-xyz"  # forwarded to the loader exactly once
    assert recorded["model"] == "openai/whisper-tiny"


@pytest.mark.skipif(
    not ctc_available,
    reason=f"facebook/wav2vec2-base-960h not in HF cache at {CTC_CACHE_DIR}",
)
def test_transcribe_with_return_timestamps_false_returns_text_only() -> None:
    """return_timestamps=False makes the HF ASR helper return text-only ScriptLines."""
    audio = _load_16k_mono_fixture()
    model: HFModel = HFModel(path_or_uri="facebook/wav2vec2-base-960h")

    result = HuggingFaceASR.transcribe_audios_with_transformers(
        audios=[audio],
        model=model,
        return_timestamps=False,
    )

    assert isinstance(result, list)
    assert len(result) == 1
    line = result[0]
    # The pipeline returns text; with return_timestamps=False, no chunks/start/end.
    assert hasattr(line, "text")
    # Shape-only: do not assert on transcription content.
    assert getattr(line, "start", None) is None
    chunks = getattr(line, "chunks", None) or []
    assert chunks == [] or chunks is None or chunks == []
