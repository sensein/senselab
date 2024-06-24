"""Tests for speaker_embeddings.py."""
import os

if os.getenv("GITHUB_ACTIONS") != "true":

    import pytest
    from torch import Tensor

    from senselab.audio.data_structures.audio import Audio
    from senselab.audio.tasks.speaker_embeddings.api import extract_speaker_embeddings_from_audios
    from senselab.utils.data_structures.model import HFModel

    @pytest.fixture
    def ecapa_model() -> HFModel:
        """Fixture for the ECAPA-TDNN model."""
        return HFModel(path_or_uri="speechbrain/spkrec-ecapa-voxceleb", revision="main")


    @pytest.fixture
    def xvector_model() -> HFModel:
        """Fixture for the xvector model."""
        return HFModel(path_or_uri="speechbrain/spkrec-xvect-voxceleb", revision="main")


    @pytest.fixture
    def resnet_model() -> HFModel:
        """Fixture for the ResNet model."""
        return HFModel(path_or_uri="speechbrain/spkrec-resnet-voxceleb", revision="main")


    def test_extract_speaker_embeddings_from_audio(
        resampled_mono_audio_sample: Audio, ecapa_model: HFModel, xvector_model: HFModel, resnet_model: HFModel
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
        resampled_mono_audio_sample: Audio, ecapa_model: HFModel, xvector_model: HFModel, resnet_model: HFModel
    ) -> None:
        """Test extracting speaker embeddings from multiple audios."""
        embeddings = extract_speaker_embeddings_from_audios(audios=[resampled_mono_audio_sample, 
                                                                    resampled_mono_audio_sample], 
                                                                    model=ecapa_model)
        assert isinstance(embeddings, list) and all(isinstance(embedding, Tensor) for embedding in embeddings)
        assert all(embedding.size(0) == 192 for embedding in embeddings)

        embeddings = extract_speaker_embeddings_from_audios(audios=[resampled_mono_audio_sample, 
                                                                    resampled_mono_audio_sample], 
                                                                    model=xvector_model)
        assert isinstance(embeddings, list) and all(isinstance(embedding, Tensor) for embedding in embeddings)
        assert all(embedding.size(0) == 512 for embedding in embeddings)

        embeddings = extract_speaker_embeddings_from_audios(audios=[resampled_mono_audio_sample, 
                                                                    resampled_mono_audio_sample], 
                                                                    model=resnet_model)
        assert isinstance(embeddings, list) and all(isinstance(embedding, Tensor) for embedding in embeddings)
        assert all(embedding.size(0) == 256 for embedding in embeddings)


    def test_error_wrong_model(resampled_mono_audio_sample: Audio) -> None:
        """Test raising error when using a non-existent model."""
        with pytest.raises(ValueError):
            embeddings = extract_speaker_embeddings_from_audios(
                audios=[resampled_mono_audio_sample], model=HFModel(path_or_uri="nonexistent---")
            )
            assert not embeddings

    def test_extract_speechbrain_speaker_embeddings_from_audio_resampled(
        mono_audio_sample: Audio, ecapa_model: HFModel, xvector_model: HFModel, resnet_model: HFModel
    ) -> None:
        """Test extracting speaker embeddings from audio."""
        # Testing with the ecapa model
        with pytest.raises(ValueError):
            extract_speaker_embeddings_from_audios(
                audios=[mono_audio_sample], model=ecapa_model
            )

        # Testing with the xvector model
        with pytest.raises(ValueError):
            extract_speaker_embeddings_from_audios(
                audios=[mono_audio_sample], model=xvector_model
            )

        # Testing with the resnet model
        with pytest.raises(ValueError):
            extract_speaker_embeddings_from_audios(
                audios=[mono_audio_sample], model=resnet_model
            )

    def test_extract_speechbrain_speaker_embeddings_from_stereo_audio(
        stereo_audio_sample: Audio, ecapa_model: HFModel, xvector_model: HFModel, resnet_model: HFModel
    ) -> None:
        """Test extracting speaker embeddings from audio."""
        # Testing with the ecapa model
        with pytest.raises(ValueError):
            extract_speaker_embeddings_from_audios(
                audios=[stereo_audio_sample], model=ecapa_model
            )

        # Testing with the xvector model
        with pytest.raises(ValueError):
            extract_speaker_embeddings_from_audios(
                audios=[stereo_audio_sample], model=xvector_model
            )

        # Testing with the resnet model
        with pytest.raises(ValueError):
            extract_speaker_embeddings_from_audios(
                audios=[stereo_audio_sample], model=resnet_model
            )
