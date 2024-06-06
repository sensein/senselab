"""Tests for speech_to_text.py."""

import pytest

from senselab.audio.tasks.speech_to_text import (
    ASRPipelineFactory,
    Transcript,
    transcribe_audios,
    transcribe_audios_with_transformers,
)
from senselab.utils.data_structures.audio import Audio
from senselab.utils.data_structures.language import Language
from senselab.utils.device import DeviceType
from senselab.utils.hf import HFModel


@pytest.fixture
def sample_audio() -> Audio:
    """Fixture for sample audio."""
    mono_audio = Audio.from_filepath(("src/tests/data_for_testing/audio_48khz_mono_16bits.wav"))
    return mono_audio


@pytest.fixture
def hf_model() -> HFModel:
    """Fixture for Hugging Face model."""
    return HFModel(path_or_uri="openai/whisper-tiny")


def test_transcript_from_dict() -> None:
    """Test creating Transcript from dict."""
    data = {
        "text": "Hello world",
        "chunks": [{"text": "Hello", "timestamp": [0.0, 1.0]}, {"text": "world", "timestamp": [1.0, 2.0]}],
    }
    transcript = Transcript.from_dict(data)

    # Ensure chunks is not None before using it
    assert transcript.chunks is not None
    assert len(transcript.chunks) == 2
    assert transcript.chunks[0].text == "Hello"
    assert transcript.chunks[0].start == 0.0
    assert transcript.chunks[0].end == 1.0

    assert transcript.chunks[1].text == "world"
    assert transcript.chunks[1].start == 1.0
    assert transcript.chunks[1].end == 2.0


def test_transcript_chunk_from_dict() -> None:
    """Test creating Transcript.Chunk from dict."""
    data = {"text": "Hello", "timestamp": [0.0, 1.0]}
    chunk = Transcript.Chunk.from_dict(data)
    assert chunk.text == "Hello"
    assert chunk.start == 0.0
    assert chunk.end == 1.0


def test_transcribe_audios(sample_audio: Audio, hf_model: HFModel) -> None:
    """Test transcribing audios."""
    transcripts = transcribe_audios(audios=[sample_audio], model=hf_model)
    assert len(transcripts) == 1
    assert isinstance(transcripts[0], Transcript)


def test_transcribe_audios_with_transformers(sample_audio: Audio, hf_model: HFModel) -> None:
    """Test transcribing audios with transformers."""
    transcripts = transcribe_audios_with_transformers(
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
    assert isinstance(transcripts[0], Transcript)


def test_asr_pipeline_factory() -> None:
    """Test ASR pipeline factory."""
    hf_model = HFModel(path_or_uri="openai/whisper-tiny")
    pipeline1 = ASRPipelineFactory._get_hf_asr_pipeline(
        model=hf_model,
        return_timestamps="word",
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=1,
        device=DeviceType.CPU,
    )
    pipeline2 = ASRPipelineFactory._get_hf_asr_pipeline(
        model=hf_model,
        return_timestamps="word",
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=1,
        device=DeviceType.CPU,
    )
    assert pipeline1 is pipeline2  # Check if the same instance is returned
