"""Tests for speaker_embeddings.py."""

import pytest
from torch import Tensor

from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.speaker_embeddings.api import extract_speaker_embeddings_from_audios
from senselab.utils.data_structures.model import HFModel


@pytest.fixture
def sample_audio() -> Audio:
    """Fixture for sample audio."""
    mono_audio = Audio.from_filepath(("src/tests/data_for_testing/audio_48khz_mono_16bits.wav"))
    return mono_audio


@pytest.fixture
def ecapa_model() -> HFModel:
    """Fixture for the ECAPA-TDNN model."""
    return HFModel(path_or_uri="speechbrain/spkrec-ecapa-voxceleb")


@pytest.fixture
def xvector_model() -> HFModel:
    """Fixture for the xvector model."""
    return HFModel(path_or_uri="speechbrain/spkrec-xvect-voxceleb")


@pytest.fixture
def resnet_model() -> HFModel:
    """Fixture for the ResNet model."""
    return HFModel(path_or_uri="speechbrain/spkrec-resnet-voxceleb")


def test_extract_speaker_embeddings_from_audio(
    sample_audio: Audio, ecapa_model: HFModel, xvector_model: HFModel, resnet_model: HFModel
) -> None:
    """Test extracting speaker embeddings from audio."""
    embeddings = extract_speaker_embeddings_from_audios(audios=[sample_audio], model=ecapa_model)
    assert isinstance(embeddings, Tensor)
    assert len(embeddings) == 192
    embeddings = extract_speaker_embeddings_from_audios(audios=[sample_audio], model=xvector_model)
    assert isinstance(embeddings, Tensor)
    assert len(embeddings) == 512
    embeddings = extract_speaker_embeddings_from_audios(audios=[sample_audio], model=resnet_model)
    assert isinstance(embeddings, Tensor)
    assert len(embeddings) == 256


def test_extract_speaker_embeddings_from_multiple_audios(
    sample_audio: Audio, ecapa_model: HFModel, xvector_model: HFModel, resnet_model: HFModel
) -> None:
    """Test extracting speaker embeddings from multiple audios."""
    embeddings = extract_speaker_embeddings_from_audios(audios=[sample_audio, sample_audio], model=ecapa_model)
    assert isinstance(embeddings, Tensor)
    assert embeddings.shape == (2, 192)
    embeddings = extract_speaker_embeddings_from_audios(
        audios=[sample_audio, sample_audio, sample_audio], model=xvector_model
    )
    assert isinstance(embeddings, Tensor)
    assert embeddings.shape == (3, 512)
    embeddings = extract_speaker_embeddings_from_audios(
        audios=[sample_audio, sample_audio, sample_audio, sample_audio], model=resnet_model
    )
    assert isinstance(embeddings, Tensor)
    assert embeddings.shape == (4, 256)


def test_error_wrong_model(sample_audio: Audio) -> None:
    """Test raising error when using a non-existent model."""
    with pytest.raises(ValueError):
        embeddings = extract_speaker_embeddings_from_audios(
            audios=[sample_audio], model=HFModel(path_or_uri="nonexistent---")
        )
        assert not embeddings
