"""Module for testing Audio data structures."""

import torch
import torchaudio

from senselab.audio.data_structures.audio import Audio
from tests.audio.conftest import MONO_AUDIO_PATH, STEREO_AUDIO_PATH


def load_audio(file_path: str) -> tuple[torch.Tensor, int]:
    """Loads audio data from the given file path."""
    return torchaudio.load(file_path)

def test_mono_audio_creation(mono_audio_sample: Audio) -> None:
    """Tests mono audio creation."""
    mono_audio_data, mono_sr = load_audio(MONO_AUDIO_PATH)
    mono_audio = Audio(
        waveform=mono_audio_data,
        sampling_rate=mono_sr,
        orig_path_or_id=MONO_AUDIO_PATH,
    )
    assert mono_audio == mono_audio_sample, "Mono audios are not exactly equivalent"

def test_stereo_audio_creation(stereo_audio_sample: Audio) -> None:
    """Tests stereo audio creation."""
    stereo_audio_data, stereo_sr = load_audio(STEREO_AUDIO_PATH)
    stereo_audio = Audio(
        waveform=stereo_audio_data,
        sampling_rate=stereo_sr,
        orig_path_or_id=STEREO_AUDIO_PATH,
    )
    assert stereo_audio == stereo_audio_sample, "Stereo audios are not exactly equivalent"

def test_stereo_audio_uuid_creation(stereo_audio_sample: Audio) -> None:
    """Tests stereo audio creation with different UUID."""
    stereo_audio_data, stereo_sr = load_audio(STEREO_AUDIO_PATH)
    stereo_audio_uuid = Audio(waveform=stereo_audio_data, sampling_rate=stereo_sr)
    assert stereo_audio_sample == stereo_audio_uuid, "Stereo audio with different IDs should still be equivalent"

def test_audio_single_tensor(mono_audio_sample: Audio) -> None:
    """Tests mono audio creation with single tensor."""
    mono_audio_data, mono_sr = load_audio(MONO_AUDIO_PATH)
    audio_single_tensor = Audio(waveform=mono_audio_data[0], sampling_rate=mono_sr)
    assert torch.equal(
        mono_audio_sample.waveform, audio_single_tensor.waveform
    ), "Mono audios of tensor shape (num_samples,) should be reshaped to (1, num_samples)"

def test_audio_from_list(mono_audio_sample: Audio) -> None:
    """Tests mono audio creation from list."""
    mono_audio_data, mono_sr = load_audio(MONO_AUDIO_PATH)
    audio_from_list = Audio(waveform=list(mono_audio_data[0]), sampling_rate=mono_sr)
    assert torch.equal(mono_audio_sample.waveform, audio_from_list.waveform), (
        "List audio should've been converted to Tensor"
    )

def test_audio_from_list_of_lists(mono_audio_sample: Audio) -> None:
    """Tests mono audio creation from list of lists."""
    mono_audio_data, mono_sr = load_audio(MONO_AUDIO_PATH)
    audio_from_list_of_lists = Audio(waveform=[list(mono_audio_data[0])], sampling_rate=mono_sr)
    assert torch.equal(mono_audio_sample.waveform, audio_from_list_of_lists.waveform), (
        "List of lists audio should've been converted to Tensor"
    )

def test_audio_from_numpy(mono_audio_sample: Audio) -> None:
    """Tests mono audio creation from numpy array."""
    mono_audio_data, mono_sr = load_audio(MONO_AUDIO_PATH)
    audio_from_numpy = Audio(waveform=mono_audio_data.numpy(), sampling_rate=mono_sr)
    assert torch.equal(mono_audio_sample.waveform, audio_from_numpy.waveform), (
        "NumPy audio should've been converted to Tensor"
    )
