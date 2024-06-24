"""Tests for voice activity detection."""
import os

import pytest

from senselab.audio.data_structures.audio import Audio
from senselab.utils.data_structures.model import SenselabModel


@pytest.fixture
def sample_audio() -> Audio:
    """Fixture for sample audio."""
    mono_audio = Audio.from_filepath(("src/tests/data_for_testing/audio_48khz_mono_16bits.wav"))
    return mono_audio

def test_detect_human_voice_activity_in_audios_with_invalid_model(sample_audio: Audio) -> None:
    """Test detecting human voice activity with an invalid model."""
    with pytest.raises(NotImplementedError):
        detect_human_voice_activity_in_audios(
            audios=[sample_audio], 
            model=SenselabModel(path_or_uri="some/invalid-model")
        )

if os.getenv("GITHUB_ACTIONS") != "true":
    from senselab.audio.tasks.voice_activity_detection.api import detect_human_voice_activity_in_audios
    from senselab.utils.data_structures.model import HFModel

    @pytest.fixture
    def pyannote_model() -> HFModel:
        """Fixture for Pyannote model."""
        return HFModel(path_or_uri="pyannote/speaker-diarization-3.1")


    def test_detect_human_voice_activity_in_audios(sample_audio: Audio, pyannote_model: HFModel) -> None:
        """Test detecting human voice activity in audios."""
        results = detect_human_voice_activity_in_audios(audios=[sample_audio], model=pyannote_model)
        assert len(results) == 1
        assert all(chunk.speaker == "VOICE" for chunk in results[0])
