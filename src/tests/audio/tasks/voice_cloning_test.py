"""This script is for testing the voice cloning API."""

import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.voice_cloning import clone_voices
from senselab.utils.data_structures import CoquiTTSModel, DeviceType


@pytest.fixture
def vc_model() -> CoquiTTSModel:
    """Fixture for Coqui TTS model."""
    return CoquiTTSModel(path_or_uri="voice_conversion_models/multilingual/multi-dataset/knnvc")


def test_clone_voices_length_mismatch(
    resampled_mono_audio_sample: Audio, vc_model: CoquiTTSModel, gpu_device: DeviceType
) -> None:
    """Test length mismatch in source and target audios."""
    source_audios = [resampled_mono_audio_sample]
    target_audios = [resampled_mono_audio_sample, resampled_mono_audio_sample]

    with pytest.raises(ValueError, match="The list of source and target audios must have the same length"):
        clone_voices(source_audios=source_audios, target_audios=target_audios, model=vc_model, device=gpu_device)


def test_clone_voices_valid_input_sparc(resampled_mono_audio_sample: Audio, gpu_device: DeviceType) -> None:
    """Test cloning voices with valid input using SPARC (subprocess venv)."""
    source_audios = [resampled_mono_audio_sample, resampled_mono_audio_sample]
    target_audios = [resampled_mono_audio_sample, resampled_mono_audio_sample]

    cloned_output = clone_voices(
        source_audios=source_audios, target_audios=target_audios, model=None, device=gpu_device
    )
    assert isinstance(cloned_output, list), "Output must be a list."
    assert len(cloned_output) == 2, "Output list should contain exactly two audio samples."
    assert isinstance(cloned_output[0], Audio), "Each item in the output list should be an instance of Audio."
    source_duration = source_audios[0].waveform.shape[1]
    cloned_duration = cloned_output[0].waveform.shape[1]

    tolerance = 0.01 * source_duration
    assert abs(source_duration - cloned_duration) <= tolerance, (
        f"Cloned audio duration is not within acceptable range. Source: {source_duration}, Cloned: {cloned_duration}"
    )


def test_clone_voices_valid_input(resampled_mono_audio_sample: Audio, vc_model: CoquiTTSModel) -> None:
    """Test cloning voices with valid input via Coqui subprocess venv."""
    source_audios = [resampled_mono_audio_sample, resampled_mono_audio_sample]
    target_audios = [resampled_mono_audio_sample, resampled_mono_audio_sample]

    cloned_output = clone_voices(source_audios=source_audios, target_audios=target_audios, model=vc_model, device=None)
    assert isinstance(cloned_output, list), "Output must be a list."
    assert len(cloned_output) == 2, "Output list should contain exactly two audio samples."
    assert isinstance(cloned_output[0], Audio), "Each item in the output list should be an instance of Audio."
    source_duration = source_audios[0].waveform.shape[1]
    cloned_duration = cloned_output[0].waveform.shape[1]

    tolerance = 0.01 * source_duration
    assert abs(source_duration - cloned_duration) <= tolerance, (
        f"Cloned audio duration is not within acceptable range. Source: {source_duration}, Cloned: {cloned_duration}"
    )


def test_clone_voices_unsupported_model(resampled_mono_audio_sample: Audio) -> None:
    """Test unsupported model."""
    source_audios = [resampled_mono_audio_sample]
    target_audios = [resampled_mono_audio_sample]
    with pytest.raises(ValueError, match="Model sensein/senselab not found. Available models:"):
        unsupported_model: CoquiTTSModel = CoquiTTSModel(path_or_uri="sensein/senselab")
        clone_voices(source_audios=source_audios, target_audios=target_audios, model=unsupported_model, device=None)


def test_clone_voices_stereo_audio(resampled_stereo_audio_sample: Audio, vc_model: CoquiTTSModel) -> None:
    """Test unsupported stereo audio."""
    source_audios = [resampled_stereo_audio_sample]
    target_audios = [resampled_stereo_audio_sample]

    with pytest.raises(ValueError, match="Only mono audio"):
        clone_voices(source_audios=source_audios, target_audios=target_audios, model=vc_model, device=None)


def test_clone_voices_invalid_sampling_rate(mono_audio_sample: Audio, vc_model: CoquiTTSModel) -> None:
    """Test unsupported sampling rate."""
    source_audios = [mono_audio_sample]
    target_audios = [mono_audio_sample]

    with pytest.raises((ValueError, RuntimeError)):
        clone_voices(source_audios=source_audios, target_audios=target_audios, model=vc_model, device=None)
