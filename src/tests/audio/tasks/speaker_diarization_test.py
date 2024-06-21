"""Tests for speaker diarization."""
import pytest

from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.speaker_diarization.api import diarize_audios
from senselab.audio.tasks.speaker_diarization.pyannote import PyannoteDiarization, diarize_audios_with_pyannote
from senselab.utils.data_structures.device import DeviceType
from senselab.utils.data_structures.model import HFModel
from senselab.utils.data_structures.script_line import ScriptLine


@pytest.fixture
def sample_audio() -> Audio:
    """Fixture for sample audio."""
    mono_audio = Audio.from_filepath(("src/tests/data_for_testing/audio_48khz_mono_16bits.wav"))
    return mono_audio


@pytest.fixture
def pyannote_model() -> HFModel:
    """Fixture for Pyannote model."""
    return HFModel(path_or_uri="pyannote/speaker-diarization-3.1")


def test_diarize_audios(sample_audio: Audio, pyannote_model: HFModel) -> None:
    """Test diarizing audios."""
    results = diarize_audios(audios=[sample_audio], model=pyannote_model)
    assert len(results) == 1
    assert isinstance(results[0][0], ScriptLine)


def test_diarize_audios_with_pyannote(sample_audio: Audio, pyannote_model: HFModel) -> None:
    """Test diarizing audios with Pyannote."""
    results = diarize_audios_with_pyannote(
        audios=[sample_audio],
        model=pyannote_model,
        device=DeviceType.CPU,
        num_speakers=2
    )
    assert len(results) == 1
    assert isinstance(results[0][0], ScriptLine)


def test_pyannote_pipeline_factory(pyannote_model: HFModel) -> None:
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
