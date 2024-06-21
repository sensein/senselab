"""Unit tests for audio processing functions.

Functions tested:
    - create_clip
    - clip_audios
    - force_align
"""

import pytest
import torch

from senselab.audio.data_structures.audio import Audio
from senselab.audio.workflows.automatic_speech_recognition_draft import clip_audios, create_clip, force_align
from senselab.utils.data_structures.device import DeviceType
from senselab.utils.data_structures.script_line import ScriptLine


@pytest.fixture
def sample_audio_mono() -> Audio:
    """Fixture for sample audio."""
    mono_audio = Audio.from_filepath(("src/tests/data_for_testing/audio_48khz_mono_16bits.wav"))
    return mono_audio


@pytest.fixture
def sample_audio_stereo() -> Audio:
    """Fixture for sample audio."""
    mono_audio = Audio.from_filepath(("src/tests/data_for_testing/audio_48khz_stereo_16bits.wav"))
    return mono_audio


def test_create_clip_valid() -> None:
    """Test create_clip with valid start and end times."""
    audio = Audio(waveform=torch.rand(1, 32000), sampling_rate=16000, orig_path_or_id="test_audio", metadata={})
    start = 0.5  # 0.5 seconds
    end = 1.5  # 1.5 seconds
    clip = create_clip(audio, start, end)
    assert clip is not None
    assert clip.waveform.size(1) == 16000  # 1 second duration


def test_create_clip_start_greater_than_duration() -> None:
    """Test create_clip with start time beyond audio length."""
    audio = Audio(waveform=torch.rand(1, 16000), sampling_rate=16000, orig_path_or_id="test_audio", metadata={})
    start = 2.0  # 2 seconds, beyond audio length
    end = 3.0
    clip = create_clip(audio, start, end)
    assert clip is None


def test_create_clip_end_greater_than_duration() -> None:
    """Test create_clip with end time beyond audio length."""
    audio = Audio(waveform=torch.rand(1, 16000), sampling_rate=16000, orig_path_or_id="test_audio", metadata={})
    start = 0.5
    end = 2.0  # 2 seconds, beyond audio length
    clip = create_clip(audio, start, end)
    assert clip is None


def test_clip_audios_valid() -> None:
    """Test clip_audios with valid diarization segments."""
    audio = Audio(waveform=torch.rand(1, 48000), sampling_rate=16000, orig_path_or_id="test_audio", metadata={})
    diarization = [ScriptLine(start=0.5, end=1.5, speaker="a"), ScriptLine(start=2.0, end=3.0, speaker="b")]
    clips = clip_audios(audio, diarization)
    assert len(clips) == 2


def test_clip_audios_no_start() -> None:
    """Test clip_audios with diarization missing start time."""
    audio = Audio(waveform=torch.rand(1, 32000), sampling_rate=16000, orig_path_or_id="test_audio", metadata={})
    diarization = [ScriptLine(start=None, end=1.5, speaker="a")]
    with pytest.raises(ValueError, match="Diarization must have both start and end times."):
        clip_audios(audio, diarization)


def test_clip_audios_no_end() -> None:
    """Test clip_audios with diarization missing end time."""
    audio = Audio(waveform=torch.rand(1, 32000), sampling_rate=16000, orig_path_or_id="test_audio", metadata={})
    diarization = [ScriptLine(start=0.5, end=None, speaker="a")]
    with pytest.raises(ValueError, match="Diarization must have both start and end times."):
        clip_audios(audio, diarization)


def test_force_align_valid(sample_audio_mono: Audio) -> None:
    """Test force_align with valid inputs."""
    audios = [sample_audio_mono]
    diarization_model_path = "pyannote/speaker-diarization-3.1"
    transcription_model_path = "openai/whisper-tiny"
    device = DeviceType.CPU
    result = force_align(audios, diarization_model_path, transcription_model_path, device)
    assert len(result) == 2


def test_force_align_empty_audio_list() -> None:
    """Test force_align with an empty audio list."""
    audios: list[Audio] = []
    diarization_model_path = "pyannote/speaker-diarization-3.1"
    transcription_model_path = "openai/whisper-tiny"
    device = DeviceType.CPU
    result = force_align(audios, diarization_model_path, transcription_model_path, device)
    assert result == []
