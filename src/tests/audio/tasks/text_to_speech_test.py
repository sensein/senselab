"""Tests for the text to speech task."""
import os

if os.getenv("GITHUB_ACTIONS") != "true":


    import pytest

    from senselab.audio.data_structures.audio import Audio
    from senselab.audio.tasks.text_to_speech.api import HuggingFaceTTS, synthesize_texts
    from senselab.utils.data_structures.device import DeviceType
    from senselab.utils.data_structures.model import HFModel, SenselabModel


    @pytest.fixture
    def hf_model() -> HFModel:
        """Fixture for HF model."""
        return HFModel(path_or_uri="suno/bark-small", revision="main")
    

    def test_synthesize_texts(hf_model: HFModel) -> None:
        """Test synthesizing texts."""
        texts = ["Hello world"]
        audios = synthesize_texts(texts=texts, model=hf_model)

        assert len(audios) == 1
        assert isinstance(audios[0], Audio)
        assert audios[0].waveform is not None
        assert audios[0].sampling_rate > 0


    def test_huggingface_tts_pipeline_factory(hf_model: HFModel) -> None:
        """Test Hugging Face TTS pipeline factory."""
        device = DeviceType.CPU
        pipeline1 = HuggingFaceTTS._get_hf_tts_pipeline(model=hf_model, device=device)
        pipeline2 = HuggingFaceTTS._get_hf_tts_pipeline(model=hf_model, device=device)

        assert pipeline1 is pipeline2  # Check if the same instance is returned


    def test_invalid_model() -> None:
        """Test synthesize_texts with invalid model."""
        texts = ["Hello world"]
        model = SenselabModel(path_or_uri="-----", revision="main")

        with pytest.raises(NotImplementedError, match="Only Hugging Face models are supported for now."):
            synthesize_texts(texts=texts, model=model)

