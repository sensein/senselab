"""Tests for the text to speech task."""

import os

if os.getenv("GITHUB_ACTIONS") != "true":
    from typing import Callable

    import pytest

    from senselab.audio.data_structures.audio import Audio
    from senselab.audio.tasks.preprocessing.preprocessing import extract_segments, resample_audios
    from senselab.audio.tasks.text_to_speech.api import HuggingFaceTTS, synthesize_texts
    from senselab.utils.data_structures.device import DeviceType
    from senselab.utils.data_structures.language import Language
    from senselab.utils.data_structures.model import HFModel, SenselabModel, TorchModel

    @pytest.fixture
    def hf_model() -> HFModel:
        """Fixture for HF model."""
        return HFModel(path_or_uri="suno/bark-small", revision="main")

    @pytest.fixture
    def hf_model2() -> HFModel:
        """Fixture for HF model."""
        return HFModel(path_or_uri="facebook/mms-tts-eng", revision="main")

    @pytest.fixture
    def mars5_model() -> TorchModel:
        """Fixture for MARS5 model."""
        return TorchModel(path_or_uri="Camb-ai/mars5-tts", revision="master")

    @pytest.mark.parametrize("hf_model", ["hf_model", "hf_model2"], indirect=True)
    def test_synthesize_texts_with_hf_model(hf_model: HFModel) -> None:
        """Test synthesizing texts."""
        texts = ["Hello world", "Hello world again."]
        audios = synthesize_texts(texts=texts, model=hf_model)

        assert len(audios) == 2
        assert isinstance(audios[0], Audio)
        assert audios[0].waveform is not None
        assert audios[0].sampling_rate > 0

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
            target=[(target_audio, target_audio_ground_truth), (target_audio, target_audio_ground_truth)],
            model=mars5_model,
            language=language,
        )

        assert len(audios) == 2
        assert isinstance(audios[0], Audio)
        assert audios[0].waveform is not None
        assert audios[0].sampling_rate == terget_audio_resampling_rate

    @pytest.mark.parametrize("device", [DeviceType.CPU, DeviceType.CUDA])  # MPS is not available for now
    def test_huggingface_tts_pipeline_factory(
        hf_model: HFModel, device: DeviceType, is_device_available: Callable
    ) -> None:
        """Test Hugging Face TTS pipeline factory."""
        if not is_device_available(device):
            pytest.skip(f"{device} is not available")

        pipeline1 = HuggingFaceTTS._get_hf_tts_pipeline(model=hf_model, device=device)
        pipeline2 = HuggingFaceTTS._get_hf_tts_pipeline(model=hf_model, device=device)

        assert pipeline1 is pipeline2  # Check if the same instance is returned

    def test_invalid_model() -> None:
        """Test synthesize_texts with invalid model."""
        texts = ["Hello world"]
        model = SenselabModel(path_or_uri="-----", revision="main")

        with pytest.raises(NotImplementedError, match="Only Hugging Face models are supported for now."):
            synthesize_texts(texts=texts, model=model)
