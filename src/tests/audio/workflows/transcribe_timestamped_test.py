"""Tests the transcrib_timestamped module."""

from senselab.audio.data_structures.audio import Audio
from senselab.audio.workflows.transcribe_timestamped import transcribe_timestamped


def test_transcribe_timestamped(mono_audio_sample: Audio) -> None:
    """Runs the transcribe_timestamped function."""
    assert transcribe_timestamped(audios=[mono_audio_sample])
