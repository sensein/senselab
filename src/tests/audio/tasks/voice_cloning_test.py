"""This script is for testing the voice cloning API."""

import pytest

from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.preprocessing.preprocessing import resample_audios
from senselab.audio.tasks.voice_cloning.api import clone_voices
from senselab.utils.data_structures.device import DeviceType
from senselab.utils.data_structures.model import TorchModel


@pytest.fixture
def audio_sample() -> Audio:
    """Fixture for sample audio."""
    mono_audio = Audio.from_filepath(("src/tests/data_for_testing/audio_48khz_mono_16bits.wav"))
    resampled_audios = resample_audios([mono_audio], 16000)
    return resampled_audios[0]

@pytest.fixture
def torch_model() -> TorchModel:
    """Fixture for torch model."""
    return TorchModel(path_or_uri="bshall/knn-vc", revision="master")

def test_clone_voices_length_mismatch(audio_sample: Audio, torch_model: TorchModel) -> None:
    """Test length mismatch in source and target audios."""
    source_audios = [audio_sample]
    target_audios = [audio_sample, audio_sample]

    with pytest.raises(ValueError, match="Source and target audios must have the same length."):
        clone_voices(
            source_audios=source_audios,
            target_audios=target_audios,
            model=torch_model,
            device=DeviceType.CPU
        )

def test_clone_voices_invalid_topk(audio_sample: Audio, torch_model: TorchModel) -> None:
    """Test invalid topk value."""
    source_audios = [audio_sample]
    target_audios = [audio_sample]

    with pytest.raises(ValueError, match="topk must be an integer."):
        clone_voices(
            source_audios=source_audios,
            target_audios=target_audios,
            model=torch_model,
            device=DeviceType.CPU,
            topk="invalid"
        )

def test_clone_voices_invalid_prematched_vocoder(audio_sample: Audio, torch_model: TorchModel) -> None:
    """Test invalid prematched_vocoder value."""
    source_audios = [audio_sample]
    target_audios = [audio_sample]

    with pytest.raises(ValueError, match="prematched_vocoder must be a boolean."):
        clone_voices(
            source_audios=source_audios,
            target_audios=target_audios,
            model=torch_model,
            device=DeviceType.CPU,
            prematched_vocoder="invalid"
        )

def test_clone_voices_unsupported_model(audio_sample: Audio) -> None:
    """Test unsupported model."""
    source_audios = [audio_sample]
    target_audios = [audio_sample]
    unsupported_model = TorchModel(path_or_uri="sensein/senselab", revision="main")

    with pytest.raises(NotImplementedError, match="Only KNNVC is supported for now."):
        clone_voices(
            source_audios=source_audios,
            target_audios=target_audios,
            model=unsupported_model,
            device=DeviceType.CPU
        )