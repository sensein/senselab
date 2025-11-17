"""This script includes some fixtures for pytest unit testing."""

import os
from typing import Callable, List

import pytest
import torch
from dotenv import find_dotenv, load_dotenv

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.preprocessing import concatenate_audios, resample_audios
from senselab.utils.data_structures import DeviceType

# Load environment variables from .env file if it exists
load_dotenv(find_dotenv(usecwd=True))

# Global variables for file paths
MONO_AUDIO_PATH = os.path.abspath(r"src/tests/data_for_testing/audio_48khz_mono_16bits.wav")
STEREO_AUDIO_PATH = os.path.abspath(r"src/tests/data_for_testing/audio_48khz_stereo_16bits.wav")
HAD_THAT_CURIOSITY_PATH = os.path.abspath(r"src/tests/data_for_testing/had_that_curiosity.wav")


@pytest.fixture
def mono_audio_sample() -> Audio:
    """Fixture for sample mono audio."""
    return Audio(filepath=MONO_AUDIO_PATH)


@pytest.fixture
def stereo_audio_sample() -> Audio:
    """Fixture for sample stereo audio."""
    return Audio(filepath=STEREO_AUDIO_PATH)


@pytest.fixture
def resampled_mono_audio_sample(mono_audio_sample: Audio, resampling_rate: int = 16000) -> Audio:
    """Fixture for resampled mono audio sample."""
    return resample_audios([mono_audio_sample], resampling_rate)[0]


@pytest.fixture
def resampled_stereo_audio_sample(stereo_audio_sample: Audio, resampling_rate: int = 16000) -> Audio:
    """Fixture for resampled stereo audio sample."""
    return resample_audios([stereo_audio_sample], resampling_rate)[0]


@pytest.fixture
def resampled_had_that_curiosity_audio_sample(resampling_rate: int = 16000) -> Audio:
    """Fixture for resampled 'had that curiosity' audio sample."""
    return resample_audios([Audio(filepath=HAD_THAT_CURIOSITY_PATH)], resampling_rate)[0]


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


@pytest.fixture
def resampled_mono_audio_sample_x2(resampled_mono_audio_sample: Audio) -> Audio:
    """Fixture for sample mono audio."""
    return concatenate_audios([resampled_mono_audio_sample, resampled_mono_audio_sample])


@pytest.fixture(scope="session")
def is_device_available() -> Callable[[DeviceType], bool]:
    """Check if a device is available."""

    def _is_device_available(device: DeviceType) -> bool:
        """Check if a device is available."""
        if device == DeviceType.CUDA:
            return torch.cuda.is_available()
        elif device == DeviceType.MPS:
            return torch.backends.mps.is_available()
        return True  # CPU is always available

    return _is_device_available
