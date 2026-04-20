"""Tests for voice activity detection."""

import pytest
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.voice_activity_detection import detect_human_voice_activity_in_audios
from senselab.utils.data_structures import DeviceType, PyannoteAudioModel, SenselabModel


@pytest.mark.skip(reason="pyannote-audio is a core dependency and always installed in test environment")
def test_detect_human_voice_activity_in_audios_import_error() -> None:
    """Test that a ModuleNotFoundError is raised when Pyannote is not installed."""
    with pytest.raises(ModuleNotFoundError):
        detect_human_voice_activity_in_audios(audios=[Audio(waveform=torch.rand(1, 16000), sampling_rate=16000)])


def test_detect_human_voice_activity_in_audios_with_invalid_model(mono_audio_sample: Audio) -> None:
    """Test detecting human voice activity with an invalid model."""
    with pytest.raises(NotImplementedError):
        detect_human_voice_activity_in_audios(
            audios=[mono_audio_sample],
            model=SenselabModel(path_or_uri="some/invalid-model"),  # type: ignore
        )


@pytest.fixture
def pyannote_model() -> PyannoteAudioModel:
    """Fixture for Pyannote model."""
    return PyannoteAudioModel(path_or_uri="pyannote/speaker-diarization-community-1")


def test_detect_human_voice_activity_in_audios(
    resampled_mono_audio_sample: Audio, pyannote_model: PyannoteAudioModel, cpu_cuda_device: DeviceType
) -> None:
    """Test detecting human voice activity in audios."""
    results = detect_human_voice_activity_in_audios(
        audios=[resampled_mono_audio_sample], model=pyannote_model, device=cpu_cuda_device
    )
    assert len(results) == 1
    assert all(chunk.speaker == "VOICE" for chunk in results[0])
