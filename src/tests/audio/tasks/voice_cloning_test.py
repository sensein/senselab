"""This script is for testing the voice cloning API."""

import os

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


@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="Don't want to hit the GitHub API limit")
@pytest.fixture
def torch_model() -> TorchModel:
    """Fixture for torch model."""
    return TorchModel(path_or_uri="bshall/knn-vc", revision="master")


@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="Don't want to hit the GitHub API limit")
def test_clone_voices_length_mismatch(audio_sample: Audio, torch_model: TorchModel) -> None:
    """Test length mismatch in source and target audios."""
    source_audios = [audio_sample]
    target_audios = [audio_sample, audio_sample]

    with pytest.raises(ValueError, match="Source and target audios must have the same length."):
        clone_voices(source_audios=source_audios, target_audios=target_audios, model=torch_model, device=DeviceType.CPU)


@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="Don't want to hit the GitHub API limit")
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
            topk="invalid",  # type: ignore[arg-type]
        )


@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="Don't want to hit the GitHub API limit")
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
            prematched_vocoder="invalid",  # type: ignore[arg-type]
        )


def test_clone_voices_unsupported_model(audio_sample: Audio) -> None:
    """Test unsupported model."""
    source_audios = [audio_sample]
    target_audios = [audio_sample]
    # this uri doesn't exist
    unsupported_model = TorchModel(path_or_uri="sensein/senselab", revision="main")

    with pytest.raises(NotImplementedError, match="Only KNNVC is supported for now."):
        clone_voices(
            source_audios=source_audios, target_audios=target_audios, model=unsupported_model, device=DeviceType.CPU
        )


@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="Don't want to hit the GitHub API limit")
def test_clone_voices_valid_input(audio_sample: Audio, torch_model: TorchModel) -> None:
    """Test cloning voices with valid input."""
    source_audios = [audio_sample]
    target_audios = [audio_sample]

    try:
        cloned_output = clone_voices(
            source_audios=source_audios,
            target_audios=target_audios,
            model=torch_model,
            device=DeviceType.CPU,
            topk=5,
            prematched_vocoder=False,
        )
        assert isinstance(cloned_output, list), "Output must be a list."
        assert len(cloned_output) == 1, "Output list should contain exactly one audio sample."
        assert isinstance(cloned_output[0], Audio), "Each item in the output list should be an instance of Audio."
        source_duration = source_audios[0].waveform.shape[1]
        cloned_duration = cloned_output[0].waveform.shape[1]

        # Set tolerance to 1% of source duration
        tolerance = 0.01 * source_duration

        # Check if the absolute difference is within the tolerance
        assert abs(source_duration - cloned_duration) <= tolerance, (
            f"Cloned audio duration is not within acceptable range. Source: {source_duration}, "
            f"Cloned: {cloned_duration}"
        )

    except Exception as e:
        pytest.fail(f"An unexpected exception occurred: {e}")
