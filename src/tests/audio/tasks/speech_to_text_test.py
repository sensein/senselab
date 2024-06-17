"""Tests for speech_to_text.py."""

import pytest

from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.speech_to_text.api import transcribe_audios
from senselab.audio.tasks.speech_to_text.huggingface import HuggingFaceASR
from senselab.utils.data_structures.device import DeviceType
from senselab.utils.data_structures.language import Language
from senselab.utils.data_structures.model import HFModel
from senselab.utils.data_structures.script_line import ScriptLine


@pytest.fixture
def sample_audio() -> Audio:
    """Fixture for sample audio."""
    mono_audio = Audio.from_filepath(("src/tests/data_for_testing/audio_48khz_mono_16bits.wav"))
    return mono_audio


@pytest.fixture
def hf_model() -> HFModel:
    """Fixture for Hugging Face model."""
    return HFModel(path_or_uri="openai/whisper-tiny")


def test_scriptline_from_dict() -> None:
    """Test creating ScriptLine from dict."""
    data = {
        "text": "Hello world",
        "chunks": [{"text": "Hello", "timestamps": [0.0, 1.0]}, {"text": "world", "timestamps": [1.0, 2.0]}],
    }
    scriptline = ScriptLine.from_dict(data)

    # Ensure chunks is not None before using it
    assert scriptline.chunks is not None
    assert len(scriptline.chunks) == 2
    assert scriptline.chunks[0].text == "Hello"
    assert scriptline.chunks[0].get_timestamps()[0] == 0.0
    assert scriptline.chunks[0].get_timestamps()[1] == 1.0

    assert scriptline.chunks[1].text == "world"
    assert scriptline.chunks[1].get_timestamps()[0] == 1.0
    assert scriptline.chunks[1].get_timestamps()[1] == 2.0


def test_transcribe_audios(sample_audio: Audio, hf_model: HFModel) -> None:
    """Test transcribing audios."""
    transcripts = transcribe_audios(audios=[sample_audio], model=hf_model)
    assert len(transcripts) == 1
    assert isinstance(transcripts[0], ScriptLine)


def test_transcribe_audios_with_transformers(sample_audio: Audio, hf_model: HFModel) -> None:
    """Test transcribing audios with transformers."""
    transcripts = HuggingFaceASR.transcribe_audios_with_transformers(
        audios=[sample_audio],
        model=hf_model,
        language=Language(language_code="English"),
        return_timestamps="word",
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=1,
        device=DeviceType.CPU,
    )
    assert len(transcripts) == 1
    assert isinstance(transcripts[0], ScriptLine)


def test_asr_pipeline_factory() -> None:
    """Test ASR pipeline factory."""
    hf_model = HFModel(path_or_uri="openai/whisper-tiny")
    pipeline1 = HuggingFaceASR._get_hf_asr_pipeline(
        model=hf_model,
        return_timestamps="word",
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=1,
        device=DeviceType.CPU,
    )
    pipeline2 = HuggingFaceASR._get_hf_asr_pipeline(
        model=hf_model,
        return_timestamps="word",
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=1,
        device=DeviceType.CPU,
    )
    assert pipeline1 is pipeline2  # Check if the same instance is returned
