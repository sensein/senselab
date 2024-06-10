"""Module for testing Audio data structures."""

import torch
import torchaudio

from senselab.audio.data_structures.audio import Audio


def test_audio_creation() -> None:
    """Tests the functionality for creating data instances."""
    mono_audio_data, mono_sr = torchaudio.load("src/tests/data_for_testing/audio_48khz_mono_16bits.wav")
    stereo_audio_data, stereo_sr = torchaudio.load("src/tests/data_for_testing/audio_48khz_stereo_16bits.wav")

    mono_audio = Audio(
        waveform=mono_audio_data,
        sampling_rate=mono_sr,
        orig_path_or_id="src/tests/data_for_testing/audio_48khz_mono_16bits.wav",
    )
    mono_audio_from_file = Audio.from_filepath("src/tests/data_for_testing/audio_48khz_mono_16bits.wav")
    assert mono_audio == mono_audio_from_file, "Mono audios are not exactly equivalent"

    stereo_audio = Audio(
        waveform=stereo_audio_data,
        sampling_rate=stereo_sr,
        orig_path_or_id="src/tests/data_for_testing/audio_48khz_stereo_16bits.wav",
    )
    stereo_audio_uuid = Audio(waveform=stereo_audio_data, sampling_rate=stereo_sr)
    stereo_audio_from_file = Audio.from_filepath("src/tests/data_for_testing/audio_48khz_stereo_16bits.wav")
    assert stereo_audio == stereo_audio_from_file, "Stereo audios are not exactly equivalent"
    assert stereo_audio == stereo_audio_uuid, "Stereo audio with different IDs should still be equivalent"

    audio_single_tensor = Audio(waveform=mono_audio_data[0], sampling_rate=mono_sr)
    assert torch.equal(
        mono_audio.waveform, audio_single_tensor.waveform
    ), "Mono audios of tensor shape (num_samples,) should be reshaped to (1, num_samples)"

    audio_from_list = Audio(waveform=list(mono_audio_data[0]), sampling_rate=mono_sr)
    audio_from_list_of_lists = Audio(waveform=[list(mono_audio_data[0])], sampling_rate=mono_sr)
    audio_from_numpy = Audio(waveform=mono_audio_data.numpy(), sampling_rate=mono_sr)

    assert torch.equal(mono_audio.waveform, audio_from_list.waveform), "List audio should've been converted to Tensor"
    assert torch.equal(
        mono_audio.waveform, audio_from_list_of_lists.waveform
    ), "List of lists audio should've been converted to Tensor"
    assert torch.equal(mono_audio.waveform, audio_from_numpy.waveform), "NumPy audio should've been converted to Tensor"
