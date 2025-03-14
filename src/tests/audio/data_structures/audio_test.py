"""Module for testing Audio data structures."""

import tempfile
from pathlib import Path
from typing import List, Tuple

import pytest
import torch

from senselab.audio.data_structures import Audio
from tests.audio.conftest import MONO_AUDIO_PATH, STEREO_AUDIO_PATH

try:
    import torchaudio

    TORCHAUDIO_AVAILABLE = True
except ModuleNotFoundError:
    TORCHAUDIO_AVAILABLE = False


def load_audio(file_path: str) -> Tuple[torch.Tensor, int]:
    """Loads audio data from the given file path."""
    return torchaudio.load(file_path)


@pytest.mark.skipif(TORCHAUDIO_AVAILABLE, reason="torchaudio is installed.")
def test_audio_creation_error() -> None:
    """Tests audio creation with invalid input."""
    with pytest.raises(ModuleNotFoundError):
        Audio.from_filepath("placeholder.wav")


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
        orig_path_or_id=audio_path,
    )
    assert audio == audio_sample, "Audios are not exactly equivalent"


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


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not installed.")
@pytest.mark.parametrize("audio_fixture", ["mono_audio_sample", "stereo_audio_sample"])
def test_audio_normalize(audio_fixture: str, request: pytest.FixtureRequest) -> None:
    """Tests the normalize method to ensure peak is 1.0 after normalization."""
    audio_sample = request.getfixturevalue(audio_fixture)

    # Check original max amplitude
    original_max = audio_sample.waveform.abs().max()
    assert original_max != 0, "Test assumes audio has non-zero values."

    # Normalize
    audio_sample.normalize()
    new_max = audio_sample.waveform.abs().max()

    assert torch.isclose(new_max, torch.tensor(1.0), atol=1e-6), "Waveform not normalized to peak=1.0."
    assert (
        audio_sample.waveform.shape == audio_sample.waveform.shape
    ), "Normalization should not change the waveform shape."
