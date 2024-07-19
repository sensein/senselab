"""This script includes some fixtures for pytest unit testing."""

from typing import List

import pytest
import torch

from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.preprocessing.preprocessing import (
    resample_audios,
)

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
def audio_with_metadata() -> Audio:
    """Fixture for generating an audio object with metadata.

    Returns:
        Audio: An audio object with random noise and metadata.
    """
    waveform = torch.randn(1, 16000)  # 1 second of random noise
    metadata = {"source": "test"}
    return Audio(waveform=waveform, sampling_rate=16000, metadata=metadata)


@pytest.fixture
def audio_with_different_bit_depths() -> List[Audio]:
    """Fixture for generating a list of audio objects with different bit depths.

    Returns:
        List[Audio]: A list containing 16-bit and 24-bit audio objects.
    """
    waveform_16bit = torch.randn(1, 16000).short()  # Simulate 16-bit audio
    waveform_24bit = torch.randn(1, 16000).int()  # Simulate 24-bit audio
    return [
        Audio(waveform=waveform_16bit.float(), sampling_rate=16000),
        Audio(waveform=waveform_24bit.float(), sampling_rate=16000),
    ]


@pytest.fixture
def audio_with_extreme_amplitude() -> Audio:
    """Fixture for generating an audio object with extreme amplitude values.

    Returns:
        Audio: An audio object with random noise of extreme amplitude.
    """
    waveform = torch.randn(1, 16000) * 1e6  # 1 second of extreme amplitude noise
    return Audio(waveform=waveform, sampling_rate=16000)
