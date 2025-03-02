"""Tests bioacoustic quality control checks."""

import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.bioacoustic_qc.checks import (
    audio_intensity_positive_check,
    audio_length_positive_check,
)


def test_audio_length_positive_check(mono_audio_sample: Audio) -> None:
    """Tests that `audio_length_positive_check` correctly filters out empty audio files."""
    # Modify existing audio sample for testing an empty case
    empty_audio = Audio(waveform=torch.tensor([]), sampling_rate=16000, metadata={})

    # Check valid audio (should pass)
    results, passed = audio_length_positive_check(mono_audio_sample)
    assert mono_audio_sample in passed, "Valid audio should pass length check."
    assert len(results["exclude"]) == 0, "Valid audio should not be excluded."

    # Check empty audio (should be excluded)
    results, passed = audio_length_positive_check(empty_audio)
    assert empty_audio in results["exclude"], "Empty audio should be excluded."
    assert len(passed) == 0, "Empty audio should not pass."


def test_audio_intensity_positive_check(stereo_audio_sample: Audio) -> None:
    """Tests that `audio_intensity_positive_check` correctly filters out silent audio files."""
    # Modify existing audio sample for testing a silent case
    silent_audio = Audio(waveform=torch.tensor([[0.0, 0.0, 0.0, 0.0]]), sampling_rate=16000, metadata={})

    # Check valid audio (should pass)
    results, passed = audio_intensity_positive_check(stereo_audio_sample)
    assert stereo_audio_sample in passed, "Valid audio should pass intensity check."
    assert len(results["exclude"]) == 0, "Valid audio should not be excluded."

    # Check silent audio (should be excluded)
    results, passed = audio_intensity_positive_check(silent_audio)
    assert silent_audio in results["exclude"], "Silent audio should be excluded."
    assert len(passed) == 0, "Silent audio should not pass."
