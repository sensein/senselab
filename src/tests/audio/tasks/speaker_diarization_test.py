"""Tests for speaker diarization."""

import os

import pytest
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.speaker_diarization import diarize_audios
from senselab.audio.tasks.speaker_diarization.pyannote import PyannoteDiarization, diarize_audios_with_pyannote
from senselab.utils.data_structures import DeviceType, PyannoteAudioModel, ScriptLine
from senselab.utils.data_structures.model import HFModel

try:
    import pyannote.audio

    PYANNOTE_INSTALLED = True
except ModuleNotFoundError:
    PYANNOTE_INSTALLED = False

try:
    import torchaudio

    TORCHAUDIO_AVAILABLE = True
except ModuleNotFoundError:
    TORCHAUDIO_AVAILABLE = False

try:
    from nemo.collections.asr.models import SortformerEncLabelModel

    NEMO_SORTFORMER_AVAILABLE = True
except ModuleNotFoundError:
    NEMO_SORTFORMER_AVAILABLE = False


@pytest.fixture
def pyannote_model() -> PyannoteAudioModel:
    """Fixture for Pyannote model."""
    return PyannoteAudioModel(path_or_uri="pyannote/speaker-diarization-community-1")


@pytest.mark.skipif(PYANNOTE_INSTALLED, reason="Pyannote is installed")
def test_pyannote_not_installed(pyannote_model: PyannoteAudioModel) -> None:
    """Test Pyannote not installed."""
    with pytest.raises(ModuleNotFoundError):
        _ = diarize_audios(audios=[Audio(waveform=torch.rand(1, 16000), sampling_rate=16000)], model=pyannote_model)


@pytest.mark.skipif(
    not PYANNOTE_INSTALLED or not TORCHAUDIO_AVAILABLE, reason="Pyannote or torchaudio are not installed"
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available")
def test_diarize_audios(resampled_mono_audio_sample: Audio, pyannote_model: PyannoteAudioModel) -> None:
    """Test diarizing audios."""
    results = diarize_audios(audios=[resampled_mono_audio_sample], model=pyannote_model)
    assert len(results) == 1
    assert isinstance(results[0][0], ScriptLine)


@pytest.mark.skipif(
    not NEMO_SORTFORMER_AVAILABLE or not TORCHAUDIO_AVAILABLE,
    reason="NVIDIA Sortformer or torchaudio are not installed",
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available")
def test_diarize_audios_with_nvidia_sortformer(resampled_mono_audio_sample: Audio) -> None:
    """Test diarizing audios with NVIDIA Sortformer."""
    model: HFModel = HFModel(path_or_uri="nvidia/diar_sortformer_4spk-v1")
    results = diarize_audios(audios=[resampled_mono_audio_sample], model=model)
    assert len(results) == 1
    assert all(isinstance(line, ScriptLine) for line in results[0])
    # Optionally, check that at least one segment is returned
    assert len(results[0]) > 0


@pytest.mark.skipif(
    not PYANNOTE_INSTALLED or not TORCHAUDIO_AVAILABLE, reason="Pyannote or torchaudio are not installed"
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available")
def test_diarize_audios_with_pyannote(resampled_mono_audio_sample: Audio, pyannote_model: PyannoteAudioModel) -> None:
    """Test diarizing audios with Pyannote."""
    results = diarize_audios_with_pyannote(
        audios=[resampled_mono_audio_sample], model=pyannote_model, device=DeviceType.CPU, num_speakers=2
    )
    assert len(results) == 1
    assert isinstance(results[0][0], ScriptLine)


@pytest.mark.skipif(
    not PYANNOTE_INSTALLED or not TORCHAUDIO_AVAILABLE, reason="Pyannote or torchaudio are not installed"
)
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


@pytest.mark.skipif(
    not PYANNOTE_INSTALLED or not TORCHAUDIO_AVAILABLE, reason="Pyannote or torchaudio are not installed"
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available")
def test_diarize_audios_with_pyannote_invalid_sampling_rate(
    mono_audio_sample: Audio, pyannote_model: PyannoteAudioModel
) -> None:
    """Test diarizing audios with unsupported sampling_rate."""
    with pytest.raises(ValueError):
        diarize_audios(audios=[mono_audio_sample], model=pyannote_model)


@pytest.mark.skipif(
    not PYANNOTE_INSTALLED or not TORCHAUDIO_AVAILABLE, reason="Pyannote or torchaudio are not installed"
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available")
def test_diarize_stereo_audios_with_pyannote_invalid(
    resampled_stereo_audio_sample: Audio, pyannote_model: PyannoteAudioModel
) -> None:
    """Test diarizing audios with unsupported number of channels."""
    with pytest.raises(ValueError):
        diarize_audios(audios=[resampled_stereo_audio_sample], model=pyannote_model)
