"""Tests for voice activity detection."""

import pytest
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.voice_activity_detection import detect_human_voice_activity_in_audios
from senselab.utils.data_structures import PyannoteAudioModel, SenselabModel

try:
    import pyannote.audio

    PYANNOTE_INSTALLED = True
except ImportError:
    PYANNOTE_INSTALLED = False

try:
    import torchaudio

    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False


@pytest.mark.skipif(PYANNOTE_INSTALLED, reason="Pyannote is installed")
def test_detect_human_voice_activity_in_audios_import_error() -> None:
    """Test that an ImportError is raised when Pyannote is not installed."""
    with pytest.raises(ImportError):
        detect_human_voice_activity_in_audios(audios=[Audio(waveform=torch.rand(1, 16000), sampling_rate=16000)])


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="Torchaudio is not available")
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
    return PyannoteAudioModel(path_or_uri="pyannote/speaker-diarization-3.1")


@pytest.mark.skipif(
    not PYANNOTE_INSTALLED or not TORCHAUDIO_AVAILABLE, reason="Pyannote or torchaudio are not installed"
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available")
def test_detect_human_voice_activity_in_audios(
    resampled_mono_audio_sample: Audio, pyannote_model: PyannoteAudioModel
) -> None:
    """Test detecting human voice activity in audios."""
    results = detect_human_voice_activity_in_audios(audios=[resampled_mono_audio_sample], model=pyannote_model)
    assert len(results) == 1
    assert all(chunk.speaker == "VOICE" for chunk in results[0])
