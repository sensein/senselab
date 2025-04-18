"""Module for testing Audio data structures."""

import tempfile
from pathlib import Path
from typing import List, Tuple
from unittest.mock import MagicMock, patch

import pytest
import torch

from senselab.audio.data_structures import Audio
from tests.audio.conftest import MONO_AUDIO_PATH, STEREO_AUDIO_PATH

try:
    import torchaudio

    TORCHAUDIO_AVAILABLE = True
except ModuleNotFoundError:
    TORCHAUDIO_AVAILABLE = False

try:
    import soundfile

    SOUNDFILE_AVAILABLE = True
except ModuleNotFoundError:
    SOUNDFILE_AVAILABLE = False


def load_audio(file_path: str) -> Tuple[torch.Tensor, int]:
    """Loads audio data from the given file path."""
    return torchaudio.load(file_path)


def check_basic_audio_properties(audio: Audio) -> None:
    """Helper function for testing basic audio properties, based off MONO_AUDIO_PATH."""
    assert audio is not None
    assert audio.waveform is not None
    assert audio.waveform.shape[1] > 0
    assert isinstance(audio.sampling_rate, int)
    assert audio.sampling_rate == 48000


@patch("torchaudio.load")  # Mock torchaudio.load
@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not installed.")
def test_audio_lazy_loading(mock_torchaudio_load: MagicMock) -> None:
    """Test lazy audio loading by mocking torchaudio.load."""
    # Mock `torchaudio.load` to return a fake waveform tensor and sample rate
    fake_waveform = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    fake_sample_rate = 48000
    mock_torchaudio_load.return_value = (fake_waveform, fake_sample_rate)

    audio = Audio(filepath=MONO_AUDIO_PATH)

    mock_torchaudio_load.assert_not_called()
    assert audio.sampling_rate == 48000, "Sampling rate should be set even if the audio is not loaded"

    _ = audio.waveform
    mock_torchaudio_load.assert_called_once_with(MONO_AUDIO_PATH, frame_offset=0, num_frames=-1, backend=None)

    _ = audio.waveform
    mock_torchaudio_load.assert_called_once()

    assert torch.equal(audio.waveform, fake_waveform)


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not installed.")
def test_audio_creation_full_file() -> None:
    """Tests loading the full audio file without offset or duration."""
    audio = Audio(filepath=MONO_AUDIO_PATH)
    check_basic_audio_properties(audio)


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not installed.")
def test_audio_creation_with_offset() -> None:
    """Tests loading audio with a positive offset."""
    test_offset = 1.0

    audio = Audio(filepath=MONO_AUDIO_PATH, offset_in_sec=test_offset)
    check_basic_audio_properties(audio)

    audio_no_offset = Audio(filepath=MONO_AUDIO_PATH)
    manual_audio_offset = int(audio.sampling_rate * test_offset)
    assert torch.equal(
        audio.waveform, audio_no_offset.waveform[:, manual_audio_offset:]
    ), "Audio offset not equivalent to manually offsetting"


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not installed.")
def test_audio_creation_with_duration() -> None:
    """Tests loading a specific duration of an audio file."""
    test_duration = 2.0

    audio = Audio(filepath=MONO_AUDIO_PATH, duration_in_sec=test_duration)
    check_basic_audio_properties(audio)

    audio_no_trunc = Audio(filepath=MONO_AUDIO_PATH)
    manual_audio_duration = int(audio.sampling_rate * test_duration)

    assert torch.equal(
        audio.waveform, audio_no_trunc.waveform[:, :manual_audio_duration]
    ), "Audio with duration not equivalent to manually truncating"


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not installed.")
def test_audio_creation_with_offset_and_duration() -> None:
    """Tests loading audio with both an offset and a duration."""
    test_duration = 2.0
    test_offset = 1.0

    audio = Audio(filepath=MONO_AUDIO_PATH, offset_in_sec=test_offset, duration_in_sec=test_duration)
    check_basic_audio_properties(audio)

    default_audio = Audio(filepath=MONO_AUDIO_PATH)
    audio_start = int(test_offset * audio.sampling_rate)
    audio_end = int((test_duration + test_offset) * audio.sampling_rate)

    assert torch.equal(
        audio.waveform, default_audio.waveform[:, audio_start:audio_end]
    ), "Audio with offset and duration not equivalent to manual version"


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not installed.")
def test_audio_creation_negative_offset() -> None:
    """Tests that a negative offset raises an error."""
    with pytest.raises(ValueError, match="Offset must be a non-negative value"):
        Audio(filepath=MONO_AUDIO_PATH, offset_in_sec=-1.0)


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not installed.")
def test_audio_creation_negative_duration() -> None:
    """Tests that a negative duration (except -1) raises an error."""
    with pytest.raises(ValueError, match="Duration must be -1 .* or a positive value"):
        Audio(filepath=MONO_AUDIO_PATH, duration_in_sec=-0.5)


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not installed.")
def test_audio_creation_full_duration() -> None:
    """Tests loading the full audio file with duration=-1."""
    audio = Audio(filepath=MONO_AUDIO_PATH, duration_in_sec=-1)
    check_basic_audio_properties(audio)

    full_audio = Audio(filepath=MONO_AUDIO_PATH)
    assert audio == full_audio, "Setting duration manually to -1 fails to return full audio"


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not installed.")
def test_audio_creation_stereo_audio() -> None:
    """Tests loading a stereo audio file."""
    audio = Audio(filepath=STEREO_AUDIO_PATH)
    check_basic_audio_properties(audio)
    assert audio.waveform.shape[0] == 2


@pytest.mark.skipif(TORCHAUDIO_AVAILABLE, reason="torchaudio is installed.")
def test_audio_creation_error() -> None:
    """Tests audio creation with missing torchaudio."""
    with pytest.raises(ModuleNotFoundError):
        Audio(filepath=MONO_AUDIO_PATH).waveform


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not installed.")
def test_audio_creation_invalid_backend() -> None:
    """Tests that an invalid backend raises an error."""
    with pytest.raises(ValueError, match="Unsupported backend"):
        Audio(filepath=MONO_AUDIO_PATH, backend="invalid_backend")


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not installed.")
@pytest.mark.parametrize(
    "audio_fixture, audio_path",
    [
        ("mono_audio_sample", MONO_AUDIO_PATH),
        ("stereo_audio_sample", STEREO_AUDIO_PATH),
    ],
)
def test_audio_creation(audio_fixture: str, audio_path: str, request: pytest.FixtureRequest) -> None:
    """Tests mono and stereo audio creation."""
    audio_sample = request.getfixturevalue(audio_fixture)
    audio_data, audio_sr = load_audio(audio_path)
    audio = Audio(
        waveform=audio_data,
        sampling_rate=audio_sr,
    )
    assert audio == audio_sample, "Audios are not exactly equivalent"


@pytest.mark.skipif(not SOUNDFILE_AVAILABLE, reason="soundfile is not installed.")
@pytest.mark.parametrize(
    "audio_path",
    [MONO_AUDIO_PATH, STEREO_AUDIO_PATH],
)
def test_audio_stream(audio_path: str) -> None:
    """Tests mono and stereo audio creation from stream."""
    audio_chunks = Audio.from_stream(audio_path)

    non_streamed_audio = Audio(filepath=audio_path)

    for i, audio_chunk in enumerate(audio_chunks):
        assert isinstance(audio_chunk, Audio), "Audio chunks should be of type Audio"
        assert audio_chunk.sampling_rate == 48000, "Audio chunks should have a sampling rate of 48000"
        assert audio_chunk.waveform.shape[1] <= 48000, "Audio chunks should have a shape of (*, 48000 or less)"

        current_chunk_end = min((i + 1) * 48000, non_streamed_audio.waveform.shape[1])

        assert torch.equal(
            audio_chunk.waveform, non_streamed_audio.waveform[:, i * 48000 : current_chunk_end]
        ), "Audio stream does not match sliding window of equivalent size and step"


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not installed.")
@pytest.mark.parametrize(
    "audio_fixture",
    ["mono_audio_sample", "stereo_audio_sample"],
)
def test_audio_save_to_file(audio_fixture: str, request: pytest.FixtureRequest) -> None:
    """Tests saving audio to file."""
    # Get the audio sample from the fixture
    audio_sample = request.getfixturevalue(audio_fixture)

    # Use a temporary file for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = Path(temp_dir) / "test_audio.wav"

        # Call save_to_file to save the audio
        audio_sample.save_to_file(file_path=temp_file_path, format="wav", bits_per_sample=16)

        # Check if the file was created
        assert temp_file_path.exists(), "The audio file was not saved."

        # Load the saved file and verify its content
        loaded_waveform, loaded_sampling_rate = torchaudio.load(temp_file_path)
        assert torch.allclose(audio_sample.waveform, loaded_waveform, atol=1e-5), "Waveform data does not match."
        assert audio_sample.sampling_rate == loaded_sampling_rate, "Sampling rate does not match."


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not installed.")
@pytest.mark.parametrize(
    "audio_fixture, audio_path",
    [
        ("mono_audio_sample", MONO_AUDIO_PATH),
        ("stereo_audio_sample", STEREO_AUDIO_PATH),
    ],
)
def test_audio_creation_uuid(audio_fixture: str, audio_path: str, request: pytest.FixtureRequest) -> None:
    """Tests audio creation with different UUID."""
    audio_sample = request.getfixturevalue(audio_fixture)
    audio_data, audio_sr = load_audio(audio_path)
    audio_uuid = Audio(waveform=audio_data, sampling_rate=audio_sr)
    assert audio_sample == audio_uuid, "Audio with different IDs should still be equivalent"


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not installed.")
def test_audio_single_tensor(mono_audio_sample: Audio) -> None:
    """Tests mono audio creation with single tensor."""
    mono_audio_data, mono_sr = load_audio(MONO_AUDIO_PATH)
    audio_single_tensor = Audio(waveform=mono_audio_data[0], sampling_rate=mono_sr)
    assert torch.equal(
        mono_audio_sample.waveform, audio_single_tensor.waveform
    ), "Mono audios of tensor shape (num_samples,) should be reshaped to (1, num_samples)"


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not installed.")
def test_audio_no_waveform() -> None:
    """Lazy audio changes allow for no waveform to be passed so test that error is raised."""
    _, mono_sr = load_audio(MONO_AUDIO_PATH)

    with pytest.raises(ValueError, match="Either a waveform or a valid filepath must be provided"):
        _ = Audio(sampling_rate=mono_sr)


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not installed.")
@pytest.mark.parametrize(
    "audio_fixture, audio_path",
    [
        ("mono_audio_sample", MONO_AUDIO_PATH),
    ],
)
def test_audio_from_list(audio_fixture: str, audio_path: str, request: pytest.FixtureRequest) -> None:
    """Tests audio creation from list."""
    audio_sample = request.getfixturevalue(audio_fixture)
    audio_data, audio_sr = load_audio(audio_path)
    audio_from_list = Audio(waveform=list(audio_data[0]), sampling_rate=audio_sr)
    assert torch.equal(audio_sample.waveform, audio_from_list.waveform), "List audio should've been converted to Tensor"


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not installed.")
@pytest.mark.parametrize(
    "audio_fixture, window_size, step_size",
    [
        ("mono_audio_sample", 1024, 512),
        ("stereo_audio_sample", 1024, 512),
    ],
)
def test_window_generator_overlap(
    audio_fixture: str, window_size: int, step_size: int, request: pytest.FixtureRequest
) -> None:
    """Tests window generator with overlapping windows."""
    audio_sample = request.getfixturevalue(audio_fixture)
    audio_length = audio_sample.waveform.size(-1)

    windowed_audios: List[Audio] = list(audio_sample.window_generator(window_size, step_size))

    # Adjust expected windows calculation to handle rounding issues
    expected_windows = (audio_length + step_size - 1) // step_size
    remaining_audio = audio_length - (expected_windows * step_size)
    if remaining_audio > 0:
        expected_windows += 1

    assert len(windowed_audios) == expected_windows, f"Should yield {expected_windows} \
        windows when step size is less than window size. Yielded {len(windowed_audios)}."


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not installed.")
@pytest.mark.parametrize(
    "audio_fixture, window_size, step_size",
    [
        ("mono_audio_sample", 1024, 1024),
        ("stereo_audio_sample", 1024, 1024),
    ],
)
def test_window_generator_exact_fit(
    audio_fixture: str, window_size: int, step_size: int, request: pytest.FixtureRequest
) -> None:
    """Tests window generator when step size equals window size."""
    audio_sample = request.getfixturevalue(audio_fixture)
    audio_length = audio_sample.waveform.size(-1)

    windowed_audios: List[Audio] = list(audio_sample.window_generator(window_size, step_size))

    expected_windows = (audio_length + step_size - 1) // step_size
    # Check if there is any remaining audio for another window
    remaining_audio = audio_length - (expected_windows * step_size)
    if remaining_audio > 0:
        expected_windows += 1

    assert len(windowed_audios) == expected_windows, f"Should yield {expected_windows} \
        windows when step size equals window size. Yielded {len(windowed_audios)}."


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not installed.")
@pytest.mark.parametrize(
    "audio_fixture, window_size, step_size",
    [
        ("mono_audio_sample", 1024, 2048),
        ("stereo_audio_sample", 1024, 2048),
    ],
)
def test_window_generator_step_greater_than_window(
    audio_fixture: str, window_size: int, step_size: int, request: pytest.FixtureRequest
) -> None:
    """Tests window generator when step size is greater than window size."""
    audio_sample = request.getfixturevalue(audio_fixture)
    audio_length = audio_sample.waveform.size(-1)

    windowed_audios: List[Audio] = list(audio_sample.window_generator(window_size, step_size))

    # Refine expected windows calculation
    expected_windows = (audio_length + step_size - 1) // step_size
    assert len(windowed_audios) == expected_windows, f"Should yield {expected_windows} \
        windows when step size is greater than window size. Yielded {len(windowed_audios)}."


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not installed.")
@pytest.mark.parametrize(
    "audio_fixture",
    [
        "mono_audio_sample",
        "stereo_audio_sample",
    ],
)
def test_window_generator_window_greater_than_audio(audio_fixture: str, request: pytest.FixtureRequest) -> None:
    """Tests window generator when window size is greater than the audio length."""
    audio_sample = request.getfixturevalue(audio_fixture)
    audio_length = audio_sample.waveform.size(-1)
    window_size = audio_length + 1000  # Set window size greater than audio length
    step_size = window_size

    windowed_audios: List[Audio] = list(audio_sample.window_generator(window_size, step_size))
    # Expect only 1 window in this case
    assert len(windowed_audios) == 1, f"Should yield 1 window when window size is greater \
                                than audio length. Yielded {len(windowed_audios)}."


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not installed.")
@pytest.mark.parametrize(
    "audio_fixture",
    [
        "mono_audio_sample",
        "stereo_audio_sample",
    ],
)
def test_window_generator_step_greater_than_audio(audio_fixture: str, request: pytest.FixtureRequest) -> None:
    """Tests window generator when step size is greater than the audio length."""
    audio_sample = request.getfixturevalue(audio_fixture)
    audio_length = audio_sample.waveform.size(1)
    window_size = 1024
    step_size = audio_length + 1000  # Step size greater than audio length

    windowed_audios: List[Audio] = list(audio_sample.window_generator(window_size, step_size))

    expected_windows = (audio_length - window_size) // step_size + 1  # This is always 1
    assert len(windowed_audios) == expected_windows, f"Should yield {expected_windows} \
        windows when step size is greater than audio length. Yielded {len(windowed_audios)}."
