"""Module for testing the preprocessing functionality of Audios."""

import math

import pytest
import torch

from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.preprocessing.preprocessing import (
    chunk_audios,
    downmix_audios_to_mono,
    resample_audios,
    select_channel_from_audios,
)


def test_resample_audios() -> None:
    """Tests functionality for resampling Audio objects."""
    resample_rate = 36000
    mono_audio = Audio.from_filepath(("src/tests/data_for_testing/audio_48khz_mono_16bits.wav"))
    resampled_expected_size = mono_audio.waveform.shape[1] / 48000 * resample_rate

    resampled_audio = resample_audios([mono_audio], resample_rate)
    assert math.ceil(resampled_expected_size) == resampled_audio[0].waveform.shape[1]

    stereo_audio = Audio.from_filepath(("src/tests/data_for_testing/audio_48khz_stereo_16bits.wav"))
    resampled_expected_size = stereo_audio.waveform.shape[1] / 48000 * resample_rate

    resampled_audio = resample_audios([stereo_audio], resample_rate)
    assert math.ceil(resampled_expected_size) == resampled_audio[0].waveform.shape[1]


def test_downmix_audios() -> None:
    """Tests functionality for downmixing Audio objects."""
    mono_audio = Audio.from_filepath(("src/tests/data_for_testing/audio_48khz_mono_16bits.wav"))
    down_mixed_audios = downmix_audios_to_mono([mono_audio])
    assert down_mixed_audios[0].waveform.dim() == 2, "Mono audio should maintain the (num_channels, num_samples) shape"
    assert down_mixed_audios[0].waveform.shape[0] == 1, "Mono audio should remain mono after downmixing"
    assert down_mixed_audios[0].waveform.size(1) == mono_audio.waveform.size(
        1
    ), "Downmixed mono audio should have correct number of samples"

    stereo_audio = Audio.from_filepath("src/tests/data_for_testing/audio_48khz_stereo_16bits.wav")
    down_mixed_audios = downmix_audios_to_mono([stereo_audio])
    assert down_mixed_audios[0].waveform.dim() == 2, "Mono audio should maintain the (num_channels, num_samples) shape"
    assert down_mixed_audios[0].waveform.shape[0] == 1, "Stereo audio should become mono after downmixing"
    assert down_mixed_audios[0].waveform.size(1) == stereo_audio.waveform.size(
        1
    ), "Downmixed stereo audio should have correct number of samples"
    assert torch.isclose(
        down_mixed_audios[0].waveform, stereo_audio.waveform.mean(dim=0, keepdim=True)
    ).all(), "Downmixed audio should be the mean of the stereo channels"


def test_select_channel_from_audios() -> None:
    """Tests functionality for selecting a specific channel from Audio objects."""

    def check_selected_channel(audio: Audio, channel_to_select: int) -> None:
        """Checks if the original selected audio channel is the same as the returned selected audio channel."""
        selected_channel_audios = select_channel_from_audios([audio], channel_to_select)
        assert selected_channel_audios[0].waveform.shape[0] == 1, "Selected channel audio should be mono"
        assert (
            selected_channel_audios[0].waveform.shape[1] == audio.waveform.shape[1]
        ), "Selected channel audio should have the correct number of samples"
        assert torch.equal(
            selected_channel_audios[0].waveform[0, :], audio.waveform[channel_to_select, :]
        ), "Selected channel audio should be the same as the selected channel of the original audio"

    channel_to_select = 0
    mono_audio = Audio.from_filepath("src/tests/data_for_testing/audio_48khz_mono_16bits.wav")
    check_selected_channel(mono_audio, channel_to_select)

    stereo_audio = Audio.from_filepath("src/tests/data_for_testing/audio_48khz_stereo_16bits.wav")
    check_selected_channel(stereo_audio, channel_to_select)

    channel_to_select = 1
    check_selected_channel(stereo_audio, channel_to_select)


def test_chunk_audios() -> None:
    """Tests functionality for chunking Audio objects."""
    # Test data setup
    audio_path = "src/tests/data_for_testing/audio_48khz_mono_16bits.wav"
    audio = Audio.from_filepath(audio_path)
    audio_duration = audio.waveform.shape[1] / audio.sampling_rate

    # Test cases
    test_data = [
        (audio, (0.0, 1.0)),  # Normal case within bounds
        (audio, (1.0, 2.0)),  # Normal case within bounds
    ]

    chunked_audios = chunk_audios(test_data)

    # Verify chunked audio lengths
    for i, (original_audio, (start, end)) in enumerate(test_data):
        start_sample = int(start * original_audio.sampling_rate)
        end_sample = int(end * original_audio.sampling_rate)
        expected_length = end_sample - start_sample
        assert chunked_audios[i].waveform.shape[1] == expected_length
    # Test case where start time is negative
    with pytest.raises(ValueError, match="Start time must be greater than or equal to 0."):
        chunk_audios([(audio, (-1.0, 1.0))])

    # Test case where end time exceeds duration
    try:
        chunk_audios([(audio, (0.0, audio_duration + 1.0))])
    except ValueError as e:
        assert str(e) == f"End time must be less than the duration of the audio file ({audio_duration} seconds)."
    else:
        pytest.fail("ValueError not raised")

    # Test case where end time equals duration
    chunked_audio = chunk_audios([(audio, (0.0, audio_duration))])[0]
    assert chunked_audio.waveform.shape[1] == audio.waveform.shape[1]
