"""Tests for the speech to text task."""

from typing import Callable

import pytest
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.speech_to_text import transcribe_audios
from senselab.audio.tasks.speech_to_text.huggingface import HuggingFaceASR
from senselab.utils.data_structures import DeviceType, HFModel, Language, ScriptLine

try:
    import torchaudio  # noqa: F401

    TORCHAUDIO_AVAILABLE = True
except ModuleNotFoundError:
    TORCHAUDIO_AVAILABLE = False


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


@pytest.fixture
def hf_model() -> HFModel:
    """Fixture for Hugging Face model."""
    return HFModel(path_or_uri="openai/whisper-tiny")


@pytest.fixture
def hf_model2() -> HFModel:
    """Fixture for Hugging Face model."""
    return HFModel(path_or_uri="facebook/seamless-m4t-unity-small")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available")
@pytest.mark.parametrize("device", [DeviceType.CPU, DeviceType.CUDA])  # MPS is not available for now
def test_hf_asr_pipeline_factory(hf_model: HFModel, device: DeviceType, is_device_available: Callable) -> None:
    """Test ASR pipeline factory."""
    if not is_device_available(device):
        pytest.skip(f"{device} is not available")

    pipeline1 = HuggingFaceASR._get_hf_asr_pipeline(
        model=hf_model,
        return_timestamps="word",
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=1,
        device=device,
    )
    pipeline2 = HuggingFaceASR._get_hf_asr_pipeline(
        model=hf_model,
        return_timestamps="word",
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=1,
        device=device,
    )
    assert pipeline1 is pipeline2  # Check if the same instance is returned (this is the case for serial execution)


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available")
@pytest.mark.parametrize("hf_model", ["hf_model", "hf_model2"], indirect=True)
def test_transcribe_audios(
    resampled_mono_audio_sample: Audio, resampled_mono_audio_sample_x2: Audio, hf_model: HFModel
) -> None:
    """Test transcribing audios."""
    transcripts = transcribe_audios(
        audios=[resampled_mono_audio_sample, resampled_mono_audio_sample_x2], model=hf_model
    )
    assert len(transcripts) == 2
    assert isinstance(transcripts[0], ScriptLine)
    assert (
        transcripts[0].text
        == "This is Peter. This is Johnny. Kenny. And Joe. We just wanted to take a minute to thank you."
    )


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available")
@pytest.mark.parametrize("hf_model", ["hf_model", "hf_model2"], indirect=True)
def test_transcribe_audios_with_params(
    resampled_mono_audio_sample: Audio, resampled_mono_audio_sample_x2: Audio, hf_model: HFModel
) -> None:
    """Test transcribing audios."""
    transcripts = transcribe_audios(
        audios=[resampled_mono_audio_sample, resampled_mono_audio_sample_x2],
        model=hf_model,
        language=Language(language_code="English"),
        return_timestamps=False,
    )
    assert len(transcripts) == 2
    assert isinstance(transcripts[0], ScriptLine)
    # Note: we don't check the transcript because we have noticed that by specifying the language,
    # the transcript is not correct with our sample audio


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available")
def test_transcribe_audios_with_unsupported_params(
    resampled_mono_audio_sample: Audio, resampled_mono_audio_sample_x2: Audio, hf_model: HFModel
) -> None:
    """Test transcribing audios with an unsupported param."""
    with pytest.raises(TypeError, match="got an unexpected keyword argument"):
        transcribe_audios(
            audios=[resampled_mono_audio_sample, resampled_mono_audio_sample_x2],
            model=hf_model,
            unsupported_param="unsupported_param",
        )


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available")
def test_transcribe_stereo_audio(resampled_stereo_audio_sample: Audio, hf_model: HFModel) -> None:
    """Test transcribing stereo audio."""
    # Create a mock stereo audio sample
    with pytest.raises(ValueError, match="Stereo audio is not supported"):
        transcribe_audios(audios=[resampled_stereo_audio_sample], model=hf_model)


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available")
def test_transcribe_audio_with_wrong_sampling_rate(mono_audio_sample: Audio, hf_model: HFModel) -> None:
    """Test transcribing stereo audio."""
    # Create a mock stereo audio sample
    with pytest.raises(ValueError, match="Incorrect sampling rate."):
        transcribe_audios(audios=[mono_audio_sample], model=hf_model)
