"""Tests for the text to speech task."""

from typing import Callable

import pytest
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.preprocessing import extract_segments, resample_audios
from senselab.audio.tasks.text_to_speech import synthesize_texts
from senselab.audio.tasks.text_to_speech.huggingface import HuggingFaceTTS
from senselab.utils.data_structures import DeviceType, HFModel, Language, SenselabModel, TorchModel

try:
    import vocos

    VOCOS_AVAILABLE = True
except ModuleNotFoundError:
    VOCOS_AVAILABLE = False


@pytest.fixture
def hf_model() -> HFModel:
    """Fixture for the HF model."""
    return HFModel(path_or_uri="suno/bark-small", revision="main")


@pytest.fixture
def hf_model2() -> HFModel:
    """Fixture for HF model."""
    return HFModel(path_or_uri="facebook/mms-tts-eng", revision="main")


@pytest.fixture
def mars5_model() -> TorchModel:
    """Fixture for MARS5 model."""
    return TorchModel(path_or_uri="Camb-ai/mars5-tts", revision="master")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available")
@pytest.mark.parametrize("hf_model", ["hf_model", "hf_model2"], indirect=True)
def test_synthesize_texts_with_hf_model(hf_model: HFModel) -> None:
    """Test synthesizing texts."""
    texts = ["Hello world", "Hello world again."]
    audios = synthesize_texts(texts=texts, model=hf_model, device=DeviceType.CUDA)

    assert len(audios) == 2
    assert isinstance(audios[0], Audio)
    assert audios[0].waveform is not None
    assert audios[0].sampling_rate > 0


@pytest.mark.skipif(not VOCOS_AVAILABLE, reason="Vocos is not available (dependency for mars5tts)")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available")
def test_synthesize_texts_with_mars5_model(mars5_model: TorchModel, mono_audio_sample: Audio) -> None:
    """Test synthesizing texts."""
    texts_to_synthesize = ["Hello world", "Hello world again."]
    terget_audio_resampling_rate = 24000
    target_audio_ground_truth = "This is Peter."
    language = Language(language_code="en")

    resampled_mono_audio_sample = resample_audios([mono_audio_sample], terget_audio_resampling_rate)[0]
    target_audio = extract_segments([(resampled_mono_audio_sample, [(0.0, 1.0)])])[0][0]
    audios = synthesize_texts(
        texts=texts_to_synthesize,
        targets=[(target_audio, target_audio_ground_truth), (target_audio, target_audio_ground_truth)],
        model=mars5_model,
        language=language,
        device=DeviceType.CUDA,
    )

    assert len(audios) == 2
    assert isinstance(audios[0], Audio)
    assert audios[0].waveform is not None
    assert audios[0].sampling_rate == terget_audio_resampling_rate


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available")
@pytest.mark.parametrize("device", [DeviceType.CPU, DeviceType.CUDA])  # MPS is not available for now
def test_huggingface_tts_pipeline_factory(hf_model: HFModel, device: DeviceType, is_device_available: Callable) -> None:
    """Test Hugging Face TTS pipeline factory."""
    if not is_device_available(device):
        pytest.skip(f"{device} is not available")

    pipeline1 = HuggingFaceTTS._get_hf_tts_pipeline(model=hf_model, device=device)
    pipeline2 = HuggingFaceTTS._get_hf_tts_pipeline(model=hf_model, device=device)

    assert pipeline1 is pipeline2  # Check if the same instance is returned


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available")
def test_invalid_model() -> None:
    """Test synthesize_texts with invalid model."""
    texts = ["Hello world"]
    model = SenselabModel(path_or_uri="-----", revision="main")

    # TODO Texts like these should be stored in a common utils/constants file such that
    # they only need to be changed in one place
    with pytest.raises(
        NotImplementedError, match="Only Hugging Face models and select Torch models are supported for now."
    ):
        synthesize_texts(texts=texts, model=model)
