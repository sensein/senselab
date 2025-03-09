"""Module for testing the preprocessing functionality of Audios."""

import math

import pytest
import torch
from pytest import FixtureRequest

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.preprocessing import (
    chunk_audios,
    downmix_audios_to_mono,
    evenly_segment_audios,
    extract_segments,
    pad_audios,
    select_channel_from_audios,
)

try:
    from speechbrain.augment.time_domain import Resample

    SPEECHBRAIN_AVAILABLE = True
except ModuleNotFoundError:
    SPEECHBRAIN_AVAILABLE = False

try:
    import torchaudio

    TORCHAUDIO_AVAILABLE = True
except ModuleNotFoundError:
    TORCHAUDIO_AVAILABLE = False


@pytest.mark.skipif(
    not SPEECHBRAIN_AVAILABLE or not TORCHAUDIO_AVAILABLE,
    reason="SpeechBrain or torchaudio are not available.",
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


@pytest.mark.skipif(
    not TORCHAUDIO_AVAILABLE,
    reason="torchaudio is not available.",
)
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


@pytest.mark.skipif(
    not TORCHAUDIO_AVAILABLE,
    reason="torchaudio is not available.",
)
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


@pytest.mark.skipif(
    not TORCHAUDIO_AVAILABLE,
    reason="torchaudio is not available.",
)
def test_extract_segments(resampled_mono_audio_sample: Audio) -> None:
    """Test segment extraction."""
    segments = [(0.0, 2.0), (2.0, 4.0)]
    extracted = extract_segments([(resampled_mono_audio_sample, segments)])
    for i, segment in enumerate(extracted[0]):
        expected_length = int((segments[i][1] - segments[i][0]) * resampled_mono_audio_sample.sampling_rate)
        assert segment.waveform.shape[1] == expected_length
        print(f"Extracted segment {i+1} has correct length: {segment.waveform.shape[1]} samples")


@pytest.mark.skipif(
    not TORCHAUDIO_AVAILABLE,
    reason="torchaudio is not available.",
)
def test_pad_audios(resampled_mono_audio_sample: Audio, resampled_stereo_audio_sample: Audio) -> None:
    """Test audio padding."""
    desired_samples = 1000000
    padded_mono_audio = pad_audios([resampled_mono_audio_sample], desired_samples)[0]
    padded_stereo_audio = pad_audios([resampled_stereo_audio_sample], desired_samples)[0]
    assert padded_mono_audio.waveform.shape[1] == desired_samples
    assert padded_stereo_audio.waveform.shape[1] == desired_samples
    assert len(padded_stereo_audio.waveform[0]) == desired_samples
    assert len(padded_stereo_audio.waveform[1]) == desired_samples


@pytest.mark.skipif(
    not TORCHAUDIO_AVAILABLE,
    reason="torchaudio is not available.",
)
@pytest.mark.parametrize(
    "audio_sample_fixture, segment_length",
    [
        pytest.param("resampled_mono_audio_sample", 1, id="mono"),
        pytest.param("resampled_stereo_audio_sample", 1, id="stereo"),
    ],
)
def test_evenly_segment_audios(audio_sample_fixture: str, segment_length: int, request: FixtureRequest) -> None:
    """Test even audio segmentation."""
    audio_sample = request.getfixturevalue(audio_sample_fixture)

    segments = evenly_segment_audios([audio_sample], segment_length, pad_last_segment=True)
    for _, segment in enumerate(segments[0]):
        expected_length = int(segment_length * audio_sample.sampling_rate)
        for channel in segment.waveform:
            assert channel.shape[0] == expected_length


@pytest.mark.skipif(
    not TORCHAUDIO_AVAILABLE,
    reason="torchaudio is not available.",
)
def test_chunk_audios(mono_audio_sample: Audio) -> None:
    """Tests functionality for chunking Audio objects."""
    audio_duration = mono_audio_sample.waveform.shape[1] / mono_audio_sample.sampling_rate
    test_data = [
        (mono_audio_sample, (0.0, 1.0)),  # Normal case within bounds
        (mono_audio_sample, (1.0, 2.0)),  # Normal case within bounds
    ]

    chunked_audios = chunk_audios(test_data)
    for i, (original_audio, (start, end)) in enumerate(test_data):
        start_sample = int(start * original_audio.sampling_rate)
        end_sample = int(end * original_audio.sampling_rate)
        expected_length = end_sample - start_sample
        assert chunked_audios[i].waveform.shape[1] == expected_length

    with pytest.raises(ValueError, match="Start time must be greater than or equal to 0."):
        chunk_audios([(mono_audio_sample, (-1.0, 1.0))])

    with pytest.raises(ValueError) as e:
        chunk_audios([(mono_audio_sample, (0.0, audio_duration + 1.0))])
    assert str(e.value) == f"End time must be less than the duration of the audio file ({audio_duration} seconds)."

    chunked_audio = chunk_audios([(mono_audio_sample, (0.0, audio_duration))])[0]
    assert chunked_audio.waveform.shape[1] == mono_audio_sample.waveform.shape[1]


@pytest.mark.skipif(
    not TORCHAUDIO_AVAILABLE,
    reason="torchaudio is not available.",
)
def test_concatenate_audios(resampled_mono_audio_sample: Audio, resampled_mono_audio_sample_x2: Audio) -> None:
    """Tests functionality for concatenating Audio objects."""
    assert torch.equal(
        resampled_mono_audio_sample_x2.waveform,
        torch.cat([resampled_mono_audio_sample.waveform, resampled_mono_audio_sample.waveform], dim=1),
    )
