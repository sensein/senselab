"""Tests for speaker diarization."""

import pytest
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.speaker_diarization import diarize_audios
from senselab.audio.tasks.speaker_diarization.pyannote import PyannoteDiarization, diarize_audios_with_pyannote
from senselab.utils.data_structures import DeviceType, PyannoteAudioModel, ScriptLine


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available")
@pytest.fixture
def pyannote_model() -> PyannoteAudioModel:
    """Fixture for Pyannote model."""
    return PyannoteAudioModel(path_or_uri="pyannote/speaker-diarization-3.1")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available")
def test_diarize_audios(resampled_mono_audio_sample: Audio, pyannote_model: PyannoteAudioModel) -> None:
    """Test diarizing audios."""
    results = diarize_audios(audios=[resampled_mono_audio_sample], model=pyannote_model)
    assert len(results) == 1
    assert isinstance(results[0][0], ScriptLine)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available")
def test_diarize_audios_with_pyannote(resampled_mono_audio_sample: Audio, pyannote_model: PyannoteAudioModel) -> None:
    """Test diarizing audios with Pyannote."""
    results = diarize_audios_with_pyannote(
        audios=[resampled_mono_audio_sample], model=pyannote_model, device=DeviceType.CPU, num_speakers=2
    )
    assert len(results) == 1
    assert isinstance(results[0][0], ScriptLine)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available")
def test_pyannote_pipeline_factory(pyannote_model: PyannoteAudioModel) -> None:
    """Test Pyannote pipeline factory."""
    pipeline1 = PyannoteDiarization._get_pyannote_diarization_pipeline(
        model=pyannote_model,
        device=DeviceType.CPU,
    )
    pipeline2 = PyannoteDiarization._get_pyannote_diarization_pipeline(
        model=pyannote_model,
        device=DeviceType.CPU,
    )
    assert pipeline1 is pipeline2  # Check if the same instance is returned


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available")
def test_diarize_audios_with_pyannote_invalid_sampling_rate(
    mono_audio_sample: Audio, pyannote_model: PyannoteAudioModel
) -> None:
    """Test diarizing audios with unsupported sampling_rate."""
    with pytest.raises(ValueError):
        diarize_audios(audios=[mono_audio_sample], model=pyannote_model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available")
def test_diarize_stereo_audios_with_pyannote_invalid(
    resampled_stereo_audio_sample: Audio, pyannote_model: PyannoteAudioModel
) -> None:
    """Test diarizing audios with unsupported number of channels."""
    with pytest.raises(ValueError):
        diarize_audios(audios=[resampled_stereo_audio_sample], model=pyannote_model)
