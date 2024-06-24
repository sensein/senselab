"""Tests for the speech to text task."""

import os

import pytest

from senselab.utils.data_structures.script_line import ScriptLine


def test_scriptline_from_dict() -> None:
    """Test creating ScriptLine from dict."""
    data = {
        "text": "Hello world",
        "chunks": [{"text": "Hello", "timestamps": [0.0, 1.0]},
                   {"text": "world", "timestamps": [1.0, 2.0]}],
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


if os.getenv("GITHUB_ACTIONS") != "true":
    from senselab.audio.data_structures.audio import Audio
    from senselab.audio.tasks.speech_to_text.api import transcribe_audios
    from senselab.audio.tasks.speech_to_text.huggingface import HuggingFaceASR
    from senselab.utils.data_structures.device import DeviceType
    from senselab.utils.data_structures.language import Language
    from senselab.utils.data_structures.model import HFModel

    @pytest.fixture
    def hf_model() -> HFModel:
        """Fixture for Hugging Face model."""
        return HFModel(path_or_uri="openai/whisper-tiny")


    def test_transcribe_audios(mono_audio_sample: Audio, hf_model: HFModel) -> None:
        """Test transcribing audios."""
        transcripts = transcribe_audios(audios=[mono_audio_sample], model=hf_model)
        assert len(transcripts) == 1
        assert isinstance(transcripts[0], ScriptLine)


    def test_transcribe_audios_with_transformers(mono_audio_sample: Audio, hf_model: HFModel) -> None:
        """Test transcribing audios with transformers."""
        transcripts = HuggingFaceASR.transcribe_audios_with_transformers(
            audios=[mono_audio_sample],
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


    def test_asr_pipeline_factory(hf_model: HFModel) -> None:
        """Test ASR pipeline factory."""
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


    def test_transcribe_stereo_audio(resampled_stereo_audio_sample: Audio, 
                                     hf_model: HFModel) -> None:
        """Test transcribing stereo audio."""
        # Create a mock stereo audio sample
        with pytest.raises(ValueError, match="Stereo audio is not supported"):
            transcribe_audios(audios=[resampled_stereo_audio_sample], model=hf_model)
