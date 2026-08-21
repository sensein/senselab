"""Module for testing Audio data structures."""

import tempfile
from pathlib import Path
from typing import List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
import torch
import torchaudio

from senselab.audio.data_structures import Audio
from tests.audio.conftest import MONO_AUDIO_PATH, STEREO_AUDIO_PATH


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


def test_audio_lazy_loading() -> None:
    """Test lazy audio loading — waveform is not loaded until accessed."""
    audio = Audio(filepath=MONO_AUDIO_PATH)

    # Waveform should not be loaded yet (lazy)
    assert audio._waveform is None, "Waveform should not be loaded on construction"
    assert audio.sampling_rate == 48000, "Sampling rate should be available without loading waveform"

    # Accessing waveform triggers load
    waveform = audio.waveform
    assert waveform is not None
    assert waveform.shape[0] > 0

    # Second access should return same object (cached)
    waveform2 = audio.waveform
    assert waveform is waveform2


def test_audio_creation_full_file() -> None:
    """Tests loading the full audio file without offset or duration."""
    audio = Audio(filepath=MONO_AUDIO_PATH)
    check_basic_audio_properties(audio)


def test_audio_creation_with_offset() -> None:
    """Tests loading audio with a positive offset."""
    test_offset = 1.0

    audio = Audio(filepath=MONO_AUDIO_PATH, offset_in_sec=test_offset)
    check_basic_audio_properties(audio)

    audio_no_offset = Audio(filepath=MONO_AUDIO_PATH)
    manual_audio_offset = int(audio.sampling_rate * test_offset)
    assert torch.equal(audio.waveform, audio_no_offset.waveform[:, manual_audio_offset:]), (
        "Audio offset not equivalent to manually offsetting"
    )


def test_audio_creation_with_duration() -> None:
    """Tests loading a specific duration of an audio file."""
    test_duration = 2.0

    audio = Audio(filepath=MONO_AUDIO_PATH, duration_in_sec=test_duration)
    check_basic_audio_properties(audio)

    audio_no_trunc = Audio(filepath=MONO_AUDIO_PATH)
    manual_audio_duration = int(audio.sampling_rate * test_duration)

    assert torch.equal(audio.waveform, audio_no_trunc.waveform[:, :manual_audio_duration]), (
        "Audio with duration not equivalent to manually truncating"
    )


def test_audio_creation_with_offset_and_duration() -> None:
    """Tests loading audio with both an offset and a duration."""
    test_duration = 2.0
    test_offset = 1.0

    audio = Audio(filepath=MONO_AUDIO_PATH, offset_in_sec=test_offset, duration_in_sec=test_duration)
    check_basic_audio_properties(audio)

    default_audio = Audio(filepath=MONO_AUDIO_PATH)
    audio_start = int(test_offset * audio.sampling_rate)
    audio_end = int((test_duration + test_offset) * audio.sampling_rate)

    assert torch.equal(audio.waveform, default_audio.waveform[:, audio_start:audio_end]), (
        "Audio with offset and duration not equivalent to manual version"
    )


def test_audio_creation_negative_offset() -> None:
    """Tests that a negative offset raises an error."""
    with pytest.raises(ValueError, match="Offset must be a non-negative value"):
        Audio(filepath=MONO_AUDIO_PATH, offset_in_sec=-1.0)


def test_audio_creation_negative_duration() -> None:
    """Tests that a negative duration (except -1) raises an error."""
    with pytest.raises(ValueError, match="Duration must be -1 .* or a positive value"):
        Audio(filepath=MONO_AUDIO_PATH, duration_in_sec=-0.5)


def test_audio_creation_full_duration() -> None:
    """Tests loading the full audio file with duration=-1."""
    audio = Audio(filepath=MONO_AUDIO_PATH, duration_in_sec=-1)
    check_basic_audio_properties(audio)

    full_audio = Audio(filepath=MONO_AUDIO_PATH)
    assert audio == full_audio, "Setting duration manually to -1 fails to return full audio"


def test_audio_creation_stereo_audio() -> None:
    """Tests loading a stereo audio file."""
    audio = Audio(filepath=STEREO_AUDIO_PATH)
    check_basic_audio_properties(audio)
    assert audio.waveform.shape[0] == 2


@pytest.mark.skip(reason="torchaudio is a core dependency and always installed; missing-dep path cannot be tested")
def test_audio_creation_error() -> None:
    """Tests audio creation with missing torchaudio."""
    with pytest.raises(ModuleNotFoundError):
        Audio(filepath=MONO_AUDIO_PATH).waveform


def test_audio_creation_invalid_backend() -> None:
    """Tests that an invalid backend raises an error."""
    with pytest.raises(ValueError, match="Unsupported backend"):
        Audio(filepath=MONO_AUDIO_PATH, backend="invalid_backend")


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

        assert torch.equal(audio_chunk.waveform, non_streamed_audio.waveform[:, i * 48000 : current_chunk_end]), (
            "Audio stream does not match sliding window of equivalent size and step"
        )


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
        audio_sample.save_to_file(file_path=temp_file_path, format="wav", subtype="PCM_16")

        # Check if the file was created
        assert temp_file_path.exists(), "The audio file was not saved."

        # Load the saved file and verify its content
        loaded_waveform, loaded_sampling_rate = torchaudio.load(temp_file_path)
        assert torch.allclose(audio_sample.waveform, loaded_waveform, atol=1e-5), "Waveform data does not match."
        assert audio_sample.sampling_rate == loaded_sampling_rate, "Sampling rate does not match."


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


def test_audio_single_tensor(mono_audio_sample: Audio) -> None:
    """Tests mono audio creation with single tensor."""
    mono_audio_data, mono_sr = load_audio(MONO_AUDIO_PATH)
    audio_single_tensor = Audio(waveform=mono_audio_data[0], sampling_rate=mono_sr)
    assert torch.equal(mono_audio_sample.waveform, audio_single_tensor.waveform), (
        "Mono audios of tensor shape (num_samples,) should be reshaped to (1, num_samples)"
    )


def test_audio_no_waveform() -> None:
    """Lazy audio changes allow for no waveform to be passed so test that error is raised."""
    _, mono_sr = load_audio(MONO_AUDIO_PATH)

    with pytest.raises(ValueError, match="Either a waveform or a valid filepath must be provided"):
        _ = Audio(sampling_rate=mono_sr)


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

    assert len(windowed_audios) == expected_windows, (
        f"Should yield {expected_windows} \
        windows when step size is less than window size. Yielded {len(windowed_audios)}."
    )


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

    assert len(windowed_audios) == expected_windows, (
        f"Should yield {expected_windows} \
        windows when step size equals window size. Yielded {len(windowed_audios)}."
    )


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
    assert len(windowed_audios) == expected_windows, (
        f"Should yield {expected_windows} \
        windows when step size is greater than window size. Yielded {len(windowed_audios)}."
    )


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
    assert len(windowed_audios) == 1, (
        f"Should yield 1 window when window size is greater \
                                than audio length. Yielded {len(windowed_audios)}."
    )


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
    assert len(windowed_audios) == expected_windows, (
        f"Should yield {expected_windows} \
        windows when step size is greater than audio length. Yielded {len(windowed_audios)}."
    )


# --- Windowed reads: the file's timeline and the returned content must agree. ---

STEP_SAMPLING_RATE = 22050
STEP_HALF_SAMPLES = 66150  # 3.0 s of loud content, then 3.0 s of quiet content
STEP_QUIET_SCALE = 1e-3  # the quiet half sits 60 dB below the loud half
STEP_WINDOW_SAMPLES = 4410  # 0.2 s
STEP_WINDOW_LEAD_IN_SAMPLES = 512  # window starts this far inside the quiet half
STEP_FORMATS = [".wav", ".flac", ".mp3"]


def write_amplitude_step_file(path: Path) -> None:
    """Writes a two-half file: uniform noise, then the same noise 60 dB down.

    Args:
        path: Destination path; the container is inferred from its suffix.
    """
    generator = torch.Generator().manual_seed(41)
    loud = torch.rand(STEP_HALF_SAMPLES, generator=generator) * 1.8 - 0.9
    quiet = (torch.rand(STEP_HALF_SAMPLES, generator=generator) * 1.8 - 0.9) * STEP_QUIET_SCALE
    signal = torch.cat([loud, quiet]).unsqueeze(0)
    Audio(waveform=signal, sampling_rate=STEP_SAMPLING_RATE).save_to_file(path)


def read_reference_window(path: Path, start: int, frames: int) -> torch.Tensor:
    """Reads a sample range of the same file with soundfile, as (num_channels, num_samples)."""
    data, _ = sf.read(str(path), dtype="float32", start=start, frames=frames, always_2d=True)
    return torch.from_numpy(np.ascontiguousarray(data.T))


@pytest.mark.parametrize("suffix", STEP_FORMATS)
def test_windowed_read_matches_an_independent_decoder(tmp_path: Path, suffix: str) -> None:
    """A window read by Audio must hold the samples another decoder reads over the same range."""
    path = tmp_path / f"amplitude_step{suffix}"
    write_amplitude_step_file(path)

    start = STEP_HALF_SAMPLES + STEP_WINDOW_LEAD_IN_SAMPLES
    audio = Audio(
        filepath=str(path),
        offset_in_sec=start / STEP_SAMPLING_RATE,
        duration_in_sec=STEP_WINDOW_SAMPLES / STEP_SAMPLING_RATE,
    )
    reference = read_reference_window(path, start, STEP_WINDOW_SAMPLES)

    assert audio.waveform.shape == reference.shape, (
        f"{suffix}: expected {tuple(reference.shape)} samples, got {tuple(audio.waveform.shape)}"
    )
    deviation = (audio.waveform - reference).abs().max().item()
    assert deviation < 1e-3, f"{suffix}: window differs from soundfile over the same range by {deviation:.3e}"


@pytest.mark.parametrize("suffix", STEP_FORMATS)
def test_windowed_read_of_a_quiet_window_holds_no_loud_content(tmp_path: Path, suffix: str) -> None:
    """A window lying wholly inside the quiet half must not come back carrying the loud half."""
    path = tmp_path / f"amplitude_step{suffix}"
    write_amplitude_step_file(path)

    loud_peak = read_reference_window(path, 0, STEP_HALF_SAMPLES).abs().max().item()
    audio = Audio(
        filepath=str(path),
        offset_in_sec=(STEP_HALF_SAMPLES + STEP_WINDOW_LEAD_IN_SAMPLES) / STEP_SAMPLING_RATE,
        duration_in_sec=STEP_WINDOW_SAMPLES / STEP_SAMPLING_RATE,
    )

    window_peak = audio.waveform.abs().max().item()
    assert window_peak < 0.05 * loud_peak, (
        f"{suffix}: quiet window peaks at {window_peak:.6f}, "
        f"{window_peak / loud_peak:.3f} of the loud half's {loud_peak:.6f}"
    )


@pytest.mark.parametrize("suffix", STEP_FORMATS)
def test_offset_only_read_starts_at_the_requested_offset(tmp_path: Path, suffix: str) -> None:
    """Reading from an offset to the end must return exactly the tail beginning at that offset."""
    path = tmp_path / f"amplitude_step{suffix}"
    write_amplitude_step_file(path)

    audio = Audio(filepath=str(path), offset_in_sec=STEP_HALF_SAMPLES / STEP_SAMPLING_RATE)
    reference = read_reference_window(path, STEP_HALF_SAMPLES, -1)

    assert audio.waveform.shape[-1] == reference.shape[-1], (
        f"{suffix}: expected {reference.shape[-1]} samples from the offset, got {audio.waveform.shape[-1]}"
    )
    deviation = (audio.waveform - reference).abs().max().item()
    assert deviation < 1e-3, f"{suffix}: tail differs from soundfile over the same range by {deviation:.3e}"


@pytest.mark.parametrize("suffix", STEP_FORMATS)
def test_consecutive_windows_concatenate_to_the_full_decode(tmp_path: Path, suffix: str) -> None:
    """Windowed reads tiling the file must concatenate back to the whole-file decode."""
    path = tmp_path / f"amplitude_step{suffix}"
    write_amplitude_step_file(path)

    whole = Audio(filepath=str(path)).waveform
    chunk_in_sec = 0.5
    duration_in_sec = whole.shape[-1] / STEP_SAMPLING_RATE
    pieces = []
    start_in_sec = 0.0
    while start_in_sec < duration_in_sec:
        chunk = min(chunk_in_sec, duration_in_sec - start_in_sec)
        pieces.append(Audio(filepath=str(path), offset_in_sec=start_in_sec, duration_in_sec=chunk).waveform)
        start_in_sec += chunk
    tiled = torch.cat(pieces, dim=-1)

    assert tiled.shape == whole.shape, (
        f"{suffix}: tiled windows give {tuple(tiled.shape)} samples against {tuple(whole.shape)} for the full decode"
    )
    assert torch.equal(tiled, whole), (
        f"{suffix}: tiled windows differ from the full decode by {(tiled - whole).abs().max().item():.3e}"
    )
