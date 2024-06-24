"""Tests for the speech enhancement task."""
import os

import pytest
from speechbrain.inference.separation import SepformerSeparation as separator

from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.preprocessing.preprocessing import resample_audios
from senselab.audio.tasks.speech_enhancement.api import enhance_audios
from senselab.audio.tasks.speech_enhancement.speechbrain import SpeechBrainEnhancer
from senselab.utils.data_structures.device import DeviceType
from senselab.utils.data_structures.model import HFModel


@pytest.fixture
def sample_audio() -> Audio:
    """Fixture for sample audio."""
    mono_audio = Audio.from_filepath(("src/tests/data_for_testing/audio_48khz_mono_16bits.wav"))
    resampled_audios = resample_audios([mono_audio], 16000)
    return resampled_audios[0]

@pytest.fixture
def stereo_audio() -> Audio:
    """Fixture for sample stereo audio."""
    stereo_audio = Audio.from_filepath("src/tests/data_for_testing/audio_48khz_stereo_16bits.wav")
    resampled_audios = resample_audios([stereo_audio], 16000)
    return resampled_audios[0]

@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="Don't want to hit the GitHub API limit")
@pytest.fixture
def hf_model() -> HFModel:
    """Fixture for Hugging Face model."""
    return HFModel(path_or_uri="speechbrain/sepformer-wham16k-enhancement")

@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="Don't want to hit the GitHub API limit")
def test_enhance_audios_stereo_audio(stereo_audio: Audio, hf_model: HFModel) -> None:
    """Test that enhancing stereo audios raises a ValueError."""
    with pytest.raises(ValueError, match="Audio waveform must be mono"):
        SpeechBrainEnhancer.enhance_audios_with_speechbrain(audios=[stereo_audio], model=hf_model)

@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="Don't want to hit the GitHub API limit")
def test_enhance_audios(sample_audio: Audio, hf_model: HFModel) -> None:
    """Test enhancing audios."""
    enhanced_audios = enhance_audios(audios=[sample_audio, sample_audio], model=hf_model)
    assert len(enhanced_audios) == 2
    assert isinstance(enhanced_audios[0], Audio)
    assert enhanced_audios[0].waveform.shape == sample_audio.waveform.shape


@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="Don't want to hit the GitHub API limit")
def test_speechbrain_enhancer_get_model(hf_model: HFModel) -> None:
    """Test getting SpeechBrain model."""
    model = SpeechBrainEnhancer._get_speechbrain_model(model=hf_model, device=DeviceType.CPU)
    assert model is not None
    assert isinstance(model, separator)
    assert model == SpeechBrainEnhancer._models[
        f"{hf_model.path_or_uri}-{hf_model.revision}-{DeviceType.CPU.value}"
        ]

@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="Don't want to hit the GitHub API limit")
def test_enhance_audios_with_speechbrain(sample_audio: Audio, hf_model: HFModel) -> None:
    """Test enhancing audios with SpeechBrain."""
    enhanced_audios = SpeechBrainEnhancer.enhance_audios_with_speechbrain(audios=[sample_audio], model=hf_model)
    assert len(enhanced_audios) == 1
    assert isinstance(enhanced_audios[0], Audio)
    assert enhanced_audios[0].waveform.shape == sample_audio.waveform.shape


@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="Don't want to hit the GitHub API limit")
def test_enhance_audios_incorrect_sampling_rate(sample_audio: Audio, hf_model: HFModel) -> None:
    """Test enhancing audios with incorrect sampling rate."""
    sample_audio.sampling_rate = 8000  # Incorrect sample rate for this model
    with pytest.raises(ValueError, match="Audio sampling rate 8000 does not match expected 16000"):
        SpeechBrainEnhancer.enhance_audios_with_speechbrain(audios=[sample_audio], model=hf_model)
