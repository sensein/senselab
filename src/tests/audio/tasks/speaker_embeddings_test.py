"""Tests for speaker_embeddings.py."""

import os

if os.getenv("GITHUB_ACTIONS") != "true":
    import pytest
    from torch import Tensor

    from senselab.audio.data_structures.audio import Audio
    from senselab.audio.tasks.speaker_embeddings.api import extract_speaker_embeddings_from_audios
    from senselab.utils.data_structures.model import SenselabModel, SpeechBrainModel

    @pytest.fixture
    def ecapa_model() -> SpeechBrainModel:
        """Fixture for the ECAPA-TDNN model."""
        return SpeechBrainModel(path_or_uri="speechbrain/spkrec-ecapa-voxceleb", revision="main")

    @pytest.fixture
    def xvector_model() -> SpeechBrainModel:
        """Fixture for the xvector model."""
        return SpeechBrainModel(path_or_uri="speechbrain/spkrec-xvect-voxceleb", revision="main")

    @pytest.fixture
    def resnet_model() -> SpeechBrainModel:
        """Fixture for the ResNet model."""
        return SpeechBrainModel(path_or_uri="speechbrain/spkrec-resnet-voxceleb", revision="main")

    def test_extract_speaker_embeddings_from_audio(
        resampled_mono_audio_sample: Audio,
        ecapa_model: SpeechBrainModel,
        xvector_model: SpeechBrainModel,
        resnet_model: SpeechBrainModel,
    ) -> None:
        """Test extracting speaker embeddings from audio."""
        embeddings = extract_speaker_embeddings_from_audios(audios=[resampled_mono_audio_sample], model=ecapa_model)
        assert isinstance(embeddings, list) and all(isinstance(embedding, Tensor) for embedding in embeddings)
        assert all(embedding.size(0) == 192 for embedding in embeddings)

        embeddings = extract_speaker_embeddings_from_audios(audios=[resampled_mono_audio_sample], model=xvector_model)
        assert isinstance(embeddings, list) and all(isinstance(embedding, Tensor) for embedding in embeddings)
        assert all(embedding.size(0) == 512 for embedding in embeddings)

        embeddings = extract_speaker_embeddings_from_audios(audios=[resampled_mono_audio_sample], model=resnet_model)
        assert isinstance(embeddings, list) and all(isinstance(embedding, Tensor) for embedding in embeddings)
        assert all(embedding.size(0) == 256 for embedding in embeddings)

    def test_extract_speaker_embeddings_from_multiple_audios(
        resampled_mono_audio_sample: Audio,
        ecapa_model: SpeechBrainModel,
        xvector_model: SpeechBrainModel,
        resnet_model: SpeechBrainModel,
    ) -> None:
        """Test extracting speaker embeddings from multiple audios."""
        embeddings = extract_speaker_embeddings_from_audios(
            audios=[resampled_mono_audio_sample, resampled_mono_audio_sample], model=ecapa_model
        )
        assert isinstance(embeddings, list) and all(isinstance(embedding, Tensor) for embedding in embeddings)
        assert all(embedding.size(0) == 192 for embedding in embeddings)

        embeddings = extract_speaker_embeddings_from_audios(
            audios=[resampled_mono_audio_sample, resampled_mono_audio_sample], model=xvector_model
        )
        assert isinstance(embeddings, list) and all(isinstance(embedding, Tensor) for embedding in embeddings)
        assert all(embedding.size(0) == 512 for embedding in embeddings)

        embeddings = extract_speaker_embeddings_from_audios(
            audios=[resampled_mono_audio_sample, resampled_mono_audio_sample], model=resnet_model
        )
        assert isinstance(embeddings, list) and all(isinstance(embedding, Tensor) for embedding in embeddings)
        assert all(embedding.size(0) == 256 for embedding in embeddings)

    def test_extract_speaker_embeddings_from_multiple_audios_different_sizes(
        resampled_mono_audio_sample: Audio,
        ecapa_model: SpeechBrainModel,
        xvector_model: SpeechBrainModel,
        resnet_model: SpeechBrainModel,
    ) -> None:
        """Test extracting speaker embeddings from multiple audios of differing lengths."""
        # Remove the last 50 samples from the second audio so it has a different length to the first
        resampled_mono_audio_sample_2 = resampled_mono_audio_sample.model_copy()
        resampled_mono_audio_sample_2.waveform = resampled_mono_audio_sample_2.waveform[:, :-50]

        embeddings = extract_speaker_embeddings_from_audios(
            audios=[resampled_mono_audio_sample, resampled_mono_audio_sample_2], model=ecapa_model
        )
        assert isinstance(embeddings, list) and all(isinstance(embedding, Tensor) for embedding in embeddings)
        assert all(embedding.size(0) == 192 for embedding in embeddings)

        embeddings = extract_speaker_embeddings_from_audios(
            audios=[resampled_mono_audio_sample, resampled_mono_audio_sample_2], model=xvector_model
        )
        assert isinstance(embeddings, list) and all(isinstance(embedding, Tensor) for embedding in embeddings)
        assert all(embedding.size(0) == 512 for embedding in embeddings)

        embeddings = extract_speaker_embeddings_from_audios(
            audios=[resampled_mono_audio_sample, resampled_mono_audio_sample_2], model=resnet_model
        )
        assert isinstance(embeddings, list) and all(isinstance(embedding, Tensor) for embedding in embeddings)
        assert all(embedding.size(0) == 256 for embedding in embeddings)

    def test_error_wrong_model(resampled_mono_audio_sample: Audio) -> None:
        """Test raising error when using a non-existent model."""
        with pytest.raises(ValueError):
            extract_speaker_embeddings_from_audios(
                audios=[resampled_mono_audio_sample], model=SpeechBrainModel(path_or_uri="nonexistent---")
            )
        with pytest.raises(NotImplementedError):
            extract_speaker_embeddings_from_audios(
                audios=[resampled_mono_audio_sample], model=SenselabModel(path_or_uri="nonexistent---")
            )

    def test_extract_speechbrain_speaker_embeddings_from_audio_resampled(
        mono_audio_sample: Audio,
        ecapa_model: SpeechBrainModel,
        xvector_model: SpeechBrainModel,
        resnet_model: SpeechBrainModel,
    ) -> None:
        """Test extracting speaker embeddings from audio."""
        # Testing with the ecapa model
        with pytest.raises(ValueError):
            extract_speaker_embeddings_from_audios(audios=[mono_audio_sample], model=ecapa_model)

        # Testing with the xvector model
        with pytest.raises(ValueError):
            extract_speaker_embeddings_from_audios(audios=[mono_audio_sample], model=xvector_model)

        # Testing with the resnet model
        with pytest.raises(ValueError):
            extract_speaker_embeddings_from_audios(audios=[mono_audio_sample], model=resnet_model)

    def test_extract_speechbrain_speaker_embeddings_from_stereo_audio(
        stereo_audio_sample: Audio,
        ecapa_model: SpeechBrainModel,
        xvector_model: SpeechBrainModel,
        resnet_model: SpeechBrainModel,
    ) -> None:
        """Test extracting speaker embeddings from audio."""
        # Testing with the ecapa model
        with pytest.raises(ValueError):
            extract_speaker_embeddings_from_audios(audios=[stereo_audio_sample], model=ecapa_model)

        # Testing with the xvector model
        with pytest.raises(ValueError):
            extract_speaker_embeddings_from_audios(audios=[stereo_audio_sample], model=xvector_model)

        # Testing with the resnet model
        with pytest.raises(ValueError):
            extract_speaker_embeddings_from_audios(audios=[stereo_audio_sample], model=resnet_model)
