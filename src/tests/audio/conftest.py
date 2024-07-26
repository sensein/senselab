"""This script includes some fixtures for pytest unit testing."""

import pytest

from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.preprocessing.preprocessing import concatenate_audios, resample_audios

# Global variables for file paths
MONO_AUDIO_PATH = "src/tests/data_for_testing/audio_48khz_mono_16bits.wav"
STEREO_AUDIO_PATH = "src/tests/data_for_testing/audio_48khz_stereo_16bits.wav"


@pytest.fixture
def mono_audio_sample() -> Audio:
    """Fixture for sample mono audio."""
    return Audio.from_filepath(MONO_AUDIO_PATH)


@pytest.fixture
def stereo_audio_sample() -> Audio:
    """Fixture for sample stereo audio."""
    return Audio.from_filepath(STEREO_AUDIO_PATH)


@pytest.fixture
def resampled_mono_audio_sample(mono_audio_sample: Audio, resampling_rate: int = 16000) -> Audio:
    """Fixture for resampled mono audio sample."""
    return resample_audios([mono_audio_sample], resampling_rate)[0]


@pytest.fixture
def resampled_stereo_audio_sample(stereo_audio_sample: Audio, resampling_rate: int = 16000) -> Audio:
    """Fixture for resampled stereo audio sample."""
    return resample_audios([stereo_audio_sample], resampling_rate)[0]


@pytest.fixture
def resampled_mono_audio_sample_x2(resampled_mono_audio_sample: Audio) -> Audio:
    """Fixture for sample mono audio."""
    return concatenate_audios([resampled_mono_audio_sample, resampled_mono_audio_sample])
