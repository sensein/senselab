"""Module for testing the preprocessing functionality of Audios."""

import math

import torch

from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.preprocessing.preprocessing import (
    downmix_audios_to_mono,
    evenly_segment_audios,
    extract_segments,
    pad_audios,
    select_channel_from_audios,
)


def test_resample_audios(
    mono_audio_sample: Audio,
    stereo_audio_sample: Audio,
    resampled_mono_audio_sample: Audio,
    resampled_stereo_audio_sample: Audio,
) -> None:
    """Tests functionality for resampling Audio objects."""
    resample_rate = 16000

    for sample, resampled_sample in zip(
        [mono_audio_sample, stereo_audio_sample], [resampled_mono_audio_sample, resampled_stereo_audio_sample]
    ):
        expected_size = sample.waveform.shape[1] / sample.sampling_rate * resample_rate
        assert (
            math.ceil(expected_size) == resampled_sample.waveform.shape[1]
        ), f"Expected size {math.ceil(expected_size)}, but got {resampled_sample.waveform.shape[1]}"


def test_downmix_audios(mono_audio_sample: Audio, stereo_audio_sample: Audio) -> None:
    """Tests functionality for downmixing Audio objects."""
    for sample in [mono_audio_sample, stereo_audio_sample]:
        down_mixed_audio = downmix_audios_to_mono([sample])[0]
        assert down_mixed_audio.waveform.dim() == 2, "Audio should maintain (num_channels, num_samples) shape"
        assert down_mixed_audio.waveform.shape[0] == 1, "Audio should be mono after downmixing"
        assert down_mixed_audio.waveform.size(1) == sample.waveform.size(
            1
        ), "Downmixed audio should have correct number of samples"
        if sample.waveform.shape[0] == 2:
            assert torch.isclose(
                down_mixed_audio.waveform, sample.waveform.mean(dim=0, keepdim=True)
            ).all(), "Downmixed audio should be the mean of the stereo channels"


def test_select_channel_from_audios(mono_audio_sample: Audio, stereo_audio_sample: Audio) -> None:
    """Tests functionality for selecting a specific channel from Audio objects."""
    for sample in [mono_audio_sample, stereo_audio_sample]:
        for channel in range(sample.waveform.shape[0]):
            selected_channel_audio = select_channel_from_audios([sample], channel)[0]
            assert selected_channel_audio.waveform.shape[0] == 1, "Selected channel audio should be mono"
            assert (
                selected_channel_audio.waveform.shape[1] == sample.waveform.shape[1]
            ), "Selected channel audio should have correct number of samples"
            assert torch.equal(
                selected_channel_audio.waveform[0, :], sample.waveform[channel, :]
            ), "Selected channel audio should match the original selected channel"


def test_extract_segments(resampled_mono_audio_sample: Audio) -> None:
    """Test segment extraction."""
    segments = [(0.0, 2.0), (2.0, 4.0)]
    extracted = extract_segments([(resampled_mono_audio_sample, segments)])
    for i, segment in enumerate(extracted[0]):
        expected_length = int((segments[i][1] - segments[i][0]) * resampled_mono_audio_sample.sampling_rate)
        assert segment.waveform.shape[1] == expected_length
        print(f"Extracted segment {i+1} has correct length: {segment.waveform.shape[1]} samples")


def test_pad_audios(resampled_mono_audio_sample: Audio) -> None:
    """Test audio padding."""
    desired_samples = 1000000
    padded_audio = pad_audios([resampled_mono_audio_sample], desired_samples)[0]
    assert padded_audio.waveform.shape[1] == desired_samples


def test_evenly_segment_audios(resampled_mono_audio_sample: Audio) -> None:
    """Test even audio segmentation."""
    segment_length = 1
    segments = evenly_segment_audios([resampled_mono_audio_sample], segment_length, pad_last_segment=True)
    for i, segment in enumerate(segments[0]):
        if i < len(segments) - 1:
            expected_length = int(segment_length * resampled_mono_audio_sample.sampling_rate)
        else:
            expected_length = int(
                segment_length * resampled_mono_audio_sample.sampling_rate
            )  # Last segment should be padded
        assert segment.waveform.shape[1] == expected_length
        print(f"Segment {i+1} has correct length: {segment.waveform.shape[1]} samples")
