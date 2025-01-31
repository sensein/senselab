"""Tests for the speech enhancement task."""

from typing import List

import pytest
import torch
from speechbrain.inference.separation import SepformerSeparation as separator

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.speech_enhancement import enhance_audios
from senselab.audio.tasks.speech_enhancement.speechbrain import SpeechBrainEnhancer
from senselab.utils.data_structures import DeviceType, SpeechBrainModel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available")
@pytest.fixture
def speechbrain_model1() -> SpeechBrainModel:
    """Fixture for Hugging Face model."""
    return SpeechBrainModel(path_or_uri="speechbrain/sepformer-wham16k-enhancement", revision="main")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available")
@pytest.fixture
def speechbrain_model2() -> SpeechBrainModel:
    """Fixture for SpeechBrain model."""
    return SpeechBrainModel(path_or_uri="speechbrain/metricgan-plus-voicebank", revision="main")


@pytest.fixture
def speechbrain_model(request: pytest.FixtureRequest) -> SpeechBrainModel:
    """Fixture that dynamically returns test a SpeechBrain model."""
    return request.getfixturevalue(request.param)


@pytest.fixture(autouse=True)
def clear_cache() -> None:
    """Fixture for clearing the cached models between pytest runs."""
    SpeechBrainEnhancer._models = {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available")
@pytest.mark.parametrize("speechbrain_model", ["speechbrain_model1"], indirect=True)
def test_enhance_audios_stereo_audio(resampled_stereo_audio_sample: Audio, speechbrain_model: SpeechBrainModel) -> None:
    """Test that enhancing stereo audios raises a ValueError."""
    with pytest.raises(ValueError, match="Audio waveform must be mono"):
        SpeechBrainEnhancer.enhance_audios_with_speechbrain(
            audios=[resampled_stereo_audio_sample], model=speechbrain_model
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available")
@pytest.mark.parametrize("speechbrain_model", ["speechbrain_model1", "speechbrain_model2"], indirect=True)
def test_enhance_audios(
    resampled_mono_audio_sample: Audio, resampled_mono_audio_sample_x2: Audio, speechbrain_model: SpeechBrainModel
) -> None:
    """Test enhancing audios."""
    enhanced_audios = enhance_audios(
        audios=[resampled_mono_audio_sample, resampled_mono_audio_sample_x2], model=speechbrain_model
    )
    assert len(enhanced_audios) == 2
    assert isinstance(enhanced_audios[0], Audio)
    assert enhanced_audios[0].waveform.shape == resampled_mono_audio_sample.waveform.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available")
@pytest.mark.parametrize("speechbrain_model", ["speechbrain_model1"], indirect=True)
def test_speechbrain_enhancer_get_model(speechbrain_model: SpeechBrainModel) -> None:
    """Test getting SpeechBrain model."""
    # TODO: add tests like these but with multithreading
    model, _, _ = SpeechBrainEnhancer._get_speechbrain_model(model=speechbrain_model, device=DeviceType.CPU)
    assert model is not None
    assert isinstance(model, separator)
    assert (
        model
        == SpeechBrainEnhancer._models[
            f"{speechbrain_model.path_or_uri}-{speechbrain_model.revision}-{DeviceType.CPU.value}"
        ]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available")
@pytest.mark.parametrize("speechbrain_model", ["speechbrain_model1", "speechbrain_model2"], indirect=True)
def test_enhance_audios_with_speechbrain(
    resampled_mono_audio_sample: Audio, resampled_mono_audio_sample_x2: Audio, speechbrain_model: SpeechBrainModel
) -> None:
    """Test enhancing audios with SpeechBrain."""
    enhanced_audios = SpeechBrainEnhancer.enhance_audios_with_speechbrain(
        audios=[resampled_mono_audio_sample, resampled_mono_audio_sample_x2], model=speechbrain_model
    )
    assert len(enhanced_audios) == 2
    assert isinstance(enhanced_audios[0], Audio)
    assert enhanced_audios[0].waveform.shape == resampled_mono_audio_sample.waveform.shape
    assert enhanced_audios[1].waveform.shape == resampled_mono_audio_sample_x2.waveform.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available")
@pytest.mark.parametrize(
    "speechbrain_model",
    ["speechbrain_model1"],
    indirect=True,
)
def test_enhance_audios_incorrect_sampling_rate(mono_audio_sample: Audio, speechbrain_model: SpeechBrainModel) -> None:
    """Test enhancing audios with incorrect sampling rate."""
    mono_audio_sample.sampling_rate = 8000  # Incorrect sample rate for this model
    with pytest.raises(ValueError, match="Audio sampling rate 8000 does not match expected 16000"):
        SpeechBrainEnhancer.enhance_audios_with_speechbrain(audios=[mono_audio_sample], model=speechbrain_model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available")
def test_enhance_audios_with_different_bit_depths(audio_with_different_bit_depths: List[Audio]) -> None:
    """Test enhancing audios with different bit depths."""
    enhanced_audios = SpeechBrainEnhancer.enhance_audios_with_speechbrain(audios=audio_with_different_bit_depths)
    assert len(enhanced_audios) == 2
    for audio in enhanced_audios:
        assert isinstance(audio, Audio)
        assert audio.waveform.shape == audio_with_different_bit_depths[0].waveform.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available")
def test_enhance_audios_with_metadata(audio_with_metadata: Audio) -> None:
    """Test enhancing audios with metadata."""
    enhanced_audios = SpeechBrainEnhancer.enhance_audios_with_speechbrain(audios=[audio_with_metadata])
    assert len(enhanced_audios) == 1
    assert isinstance(enhanced_audios[0], Audio)
    assert enhanced_audios[0].metadata == audio_with_metadata.metadata


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available")
def test_enhance_audios_with_extreme_amplitude(audio_with_extreme_amplitude: Audio) -> None:
    """Test enhancing audios with extreme amplitude values."""
    enhanced_audios = SpeechBrainEnhancer.enhance_audios_with_speechbrain(audios=[audio_with_extreme_amplitude])
    assert len(enhanced_audios) == 1
    assert isinstance(enhanced_audios[0], Audio)
    assert enhanced_audios[0].waveform.shape == audio_with_extreme_amplitude.waveform.shape


def test_model_caching(resampled_mono_audio_sample: Audio) -> None:
    """Test model caching by enhancing audios with the same model multiple times."""
    SpeechBrainEnhancer.enhance_audios_with_speechbrain(audios=[resampled_mono_audio_sample], device=DeviceType.CPU)
    assert len(list(SpeechBrainEnhancer._models.keys())) == 1
    SpeechBrainEnhancer.enhance_audios_with_speechbrain(audios=[resampled_mono_audio_sample], device=DeviceType.CPU)
    assert len(list(SpeechBrainEnhancer._models.keys())) == 1
