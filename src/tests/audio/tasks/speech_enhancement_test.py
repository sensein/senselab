"""Tests for the speech enhancement task."""

import os

if os.getenv("GITHUB_ACTIONS") != "true":
    import pytest
    from speechbrain.inference.separation import SepformerSeparation as separator

    from senselab.audio.data_structures.audio import Audio
    from senselab.audio.tasks.speech_enhancement.api import enhance_audios
    from senselab.audio.tasks.speech_enhancement.speechbrain import SpeechBrainEnhancer
    from senselab.utils.data_structures.device import DeviceType
    from senselab.utils.data_structures.model import SpeechBrainModel

    @pytest.fixture
    def speechbrain_model() -> SpeechBrainModel:
        """Fixture for Hugging Face model."""
        return SpeechBrainModel(path_or_uri="speechbrain/sepformer-wham16k-enhancement")

    def test_enhance_audios_stereo_audio(
        resampled_stereo_audio_sample: Audio, speechbrain_model: SpeechBrainModel
    ) -> None:
        """Test that enhancing stereo audios raises a ValueError."""
        with pytest.raises(ValueError, match="Audio waveform must be mono"):
            SpeechBrainEnhancer.enhance_audios_with_speechbrain(
                audios=[resampled_stereo_audio_sample], model=speechbrain_model
            )

    def test_enhance_audios(resampled_mono_audio_sample: Audio, speechbrain_model: SpeechBrainModel) -> None:
        """Test enhancing audios."""
        enhanced_audios = enhance_audios(
            audios=[resampled_mono_audio_sample, resampled_mono_audio_sample], model=speechbrain_model
        )
        assert len(enhanced_audios) == 2
        assert isinstance(enhanced_audios[0], Audio)
        assert enhanced_audios[0].waveform.shape == resampled_mono_audio_sample.waveform.shape

    def test_speechbrain_enhancer_get_model(speechbrain_model: SpeechBrainModel) -> None:
        """Test getting SpeechBrain model."""
        model = SpeechBrainEnhancer._get_speechbrain_model(model=speechbrain_model, device=DeviceType.CPU)
        assert model is not None
        assert isinstance(model, separator)
        assert (
            model
            == SpeechBrainEnhancer._models[
                f"{speechbrain_model.path_or_uri}-{speechbrain_model.revision}-{DeviceType.CPU.value}"
            ]
        )

    def test_enhance_audios_with_speechbrain(
        resampled_mono_audio_sample: Audio, speechbrain_model: SpeechBrainModel
    ) -> None:
        """Test enhancing audios with SpeechBrain."""
        enhanced_audios = SpeechBrainEnhancer.enhance_audios_with_speechbrain(
            audios=[resampled_mono_audio_sample], model=speechbrain_model
        )
        assert len(enhanced_audios) == 1
        assert isinstance(enhanced_audios[0], Audio)
        assert enhanced_audios[0].waveform.shape == resampled_mono_audio_sample.waveform.shape

    def test_enhance_audios_incorrect_sampling_rate(
        mono_audio_sample: Audio, speechbrain_model: SpeechBrainModel
    ) -> None:
        """Test enhancing audios with incorrect sampling rate."""
        mono_audio_sample.sampling_rate = 8000  # Incorrect sample rate for this model
        with pytest.raises(ValueError, match="Audio sampling rate 8000 does not match expected 16000"):
            SpeechBrainEnhancer.enhance_audios_with_speechbrain(audios=[mono_audio_sample], model=speechbrain_model)

# TODO: add tests
