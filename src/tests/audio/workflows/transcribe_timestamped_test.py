"""Tests the transcribe_timestamped module."""

# TODO: Please double-check this because tests are failing
from senselab.audio.data_structures import Audio
from senselab.audio.workflows.transcribe_timestamped import transcribe_timestamped


def test_transcribe_timestamped_mono(mono_audio_sample: Audio) -> None:
    """Runs the transcribe_timestamped function."""
    assert transcribe_timestamped(audios=[mono_audio_sample])


def test_transcribe_timestamped_stereo(stereo_audio_sample: Audio) -> None:
    """Test transcribe_timestamped with a stereo audio sample."""
    result = transcribe_timestamped(audios=[stereo_audio_sample])
    assert isinstance(result, list), "The result should be a list of ScriptLine lists."
    assert len(result) > 0, "The result should not be empty."
    assert all(isinstance(script_lines, list) for script_lines in result), "Each item in the result should be a list."
    assert all(
        len(script_lines) > 0 for script_lines in result
    ), "Each list in the result should contain ScriptLine objects."


def test_transcribe_timestamped_resampled_mono(
    resampled_mono_audio_sample: Audio,
) -> None:
    """Test transcribe_timestamped with a resampled mono audio sample."""
    result = transcribe_timestamped(audios=[resampled_mono_audio_sample])
    assert isinstance(result, list), "The result should be a list of ScriptLine lists."
    assert len(result) > 0, "The result should not be empty."
    assert all(isinstance(script_lines, list) for script_lines in result), "Each item in the result should be a list."
    assert all(
        len(script_lines) > 0 for script_lines in result
    ), "Each list in the result should contain ScriptLine objects."


def test_transcribe_timestamped_resampled_stereo(
    resampled_stereo_audio_sample: Audio,
) -> None:
    """Test transcribe_timestamped with a resampled stereo audio sample."""
    result = transcribe_timestamped(audios=[resampled_stereo_audio_sample])
    assert isinstance(result, list), "The result should be a list of ScriptLine lists."
    assert len(result) > 0, "The result should not be empty."
    assert all(isinstance(script_lines, list) for script_lines in result), "Each item in the result should be a list."
    assert all(
        len(script_lines) > 0 for script_lines in result
    ), "Each list in the result should contain ScriptLine objects."


def test_transcribe_timestamped_noise(audio_with_metadata: Audio) -> None:
    """Test transcribe_timestamped with a noisy audio sample."""
    result = transcribe_timestamped(audios=[audio_with_metadata])
    assert isinstance(result, list), "The result should be a list of ScriptLine lists."
    assert len(result) > 0, "The result should not be empty."
    assert all(isinstance(script_lines, list) for script_lines in result), "Each item in the result should be a list."
    assert all(
        len(script_lines) > 0 for script_lines in result
    ), "Each list in the result should contain ScriptLine objects."


def test_transcribe_timestamped_different_bit_depths(
    audio_with_different_bit_depths: list[Audio],
) -> None:
    """Test transcribe_timestamped with audio samples of different bit depths."""
    result = transcribe_timestamped(audios=audio_with_different_bit_depths)
    assert isinstance(result, list), "The result should be a list of ScriptLine lists."
    assert len(result) == len(
        audio_with_different_bit_depths
    ), "The result should have the same number of elements as the input audio."
    assert all(isinstance(script_lines, list) for script_lines in result), "Each item in the result should be a list."
    assert all(
        len(script_lines) > 0 for script_lines in result
    ), "Each list in the result should contain ScriptLine objects."
