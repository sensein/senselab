"""Tests for the text to speech task."""

import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.preprocessing import extract_segments, resample_audios
from senselab.audio.tasks.text_to_speech import synthesize_texts
from senselab.audio.tasks.text_to_speech.huggingface import HuggingFaceTTS
from senselab.utils.data_structures import CoquiTTSModel, DeviceType, HFModel, Language, SenselabModel, TorchModel

# Coqui TTS synthesis still uses direct TTS import (not subprocess venv yet).
# Guard this test until it's migrated.
try:
    from TTS.api import TTS  # noqa: F401

    TTS_AVAILABLE = True
except ModuleNotFoundError:
    TTS_AVAILABLE = False


@pytest.fixture
def hf_model() -> HFModel:
    """Fixture for the HF model."""
    return HFModel(path_or_uri="suno/bark-small", revision="main")


@pytest.fixture
def hf_model2() -> HFModel:
    """Fixture for HF model."""
    return HFModel(path_or_uri="facebook/mms-tts-eng", revision="main")


@pytest.fixture
def coqui_tts_model() -> CoquiTTSModel:
    """Fixture for Coqui TTS model."""
    return CoquiTTSModel(path_or_uri="tts_models/multilingual/multi-dataset/xtts_v2", revision="main")


def test_synthesize_texts_with_mms_tts(hf_model2: HFModel, any_device: DeviceType) -> None:
    """Test synthesizing texts with mms-tts-eng (Tier 1)."""
    texts = ["Hello world", "Hello world again."]
    audios = synthesize_texts(texts=texts, model=hf_model2, device=any_device)

    assert len(audios) == 2
    assert isinstance(audios[0], Audio)
    assert audios[0].waveform is not None
    assert audios[0].sampling_rate > 0


def test_synthesize_texts_with_bark(hf_model: HFModel, gpu_device: DeviceType) -> None:
    """Test synthesizing texts with bark-small (Tier 3)."""
    texts = ["Hello world", "Hello world again."]
    audios = synthesize_texts(texts=texts, model=hf_model, device=gpu_device)

    assert len(audios) == 2
    assert isinstance(audios[0], Audio)
    assert audios[0].waveform is not None
    assert audios[0].sampling_rate > 0


@pytest.mark.skipif(not TTS_AVAILABLE, reason="Coqui TTS synthesis not yet migrated to subprocess venv")
def test_synthesize_texts_with_coqui_model(coqui_tts_model: CoquiTTSModel, gpu_device: DeviceType) -> None:
    """Test synthesizing texts."""
    texts = ["Hello world", "Hello world again."]
    audios = synthesize_texts(
        texts=texts, model=coqui_tts_model, device=gpu_device, language=Language(language_code="en")
    )

    assert len(audios) == 2
    assert isinstance(audios[0], Audio)
    assert audios[0].waveform is not None
    assert audios[0].sampling_rate > 0


def test_huggingface_tts_pipeline_factory(hf_model: HFModel, any_device: DeviceType) -> None:
    """Test Hugging Face TTS pipeline factory."""
    pipeline1 = HuggingFaceTTS._get_hf_tts_pipeline(model=hf_model, device=any_device)
    pipeline2 = HuggingFaceTTS._get_hf_tts_pipeline(model=hf_model, device=any_device)

    assert pipeline1 is pipeline2  # Check if the same instance is returned


def test_invalid_model() -> None:
    """Test synthesize_texts with invalid model."""
    texts = ["Hello world"]
    model: SenselabModel = SenselabModel(path_or_uri="-----", revision="main")

    # TODO Texts like these should be stored in a common utils/constants file such that
    # they only need to be changed in one place
    with pytest.raises(
        NotImplementedError, match="Only Hugging Face models and select Torch models are supported for now."
    ):
        synthesize_texts(texts=texts, model=model)
