"""Module for testing the Participant, Session, and SenselabDataset classes."""

import numpy as np
import pytest
import torch
from datasets import load_dataset

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import Participant, SenselabDataset, Session
from senselab.video.data_structures import Video
from tests.audio.conftest import MONO_AUDIO_PATH, STEREO_AUDIO_PATH
from tests.video.conftest import VIDEO_PATH

try:
    import torchaudio

    TORCHAUDIO_AVAILABLE = True
except ModuleNotFoundError:
    TORCHAUDIO_AVAILABLE = False

try:
    import librosa

    LIBROSA_AVAILABLE = True
except ModuleNotFoundError:
    LIBROSA_AVAILABLE = False

try:
    import av

    AV_AVAILABLE = True
except ModuleNotFoundError:
    AV_AVAILABLE = False


def test_create_participant() -> None:
    """Test creating a participant."""
    participant = Participant(metadata={"name": "John Doe"})
    assert isinstance(participant, Participant)
    assert participant.metadata["name"] == "John Doe"


def test_create_session() -> None:
    """Test creating a session."""
    session = Session(metadata={"description": "Initial session"})
    assert isinstance(session, Session)
    assert session.metadata["description"] == "Initial session"


def test_add_participant() -> None:
    """Test adding a participant to the dataset."""
    dataset = SenselabDataset()
    participant = Participant()
    dataset.add_participant(participant)
    assert participant.id in dataset.participants


def test_add_duplicate_participant() -> None:
    """Test adding a duplicate participant to the dataset."""
    dataset = SenselabDataset()
    participant = Participant()
    dataset.add_participant(participant)
    with pytest.raises(ValueError):
        dataset.add_participant(participant)


def test_add_session() -> None:
    """Test adding a session to the dataset."""
    dataset = SenselabDataset()
    session = Session()
    dataset.add_session(session)
    assert session.id in dataset.sessions


def test_add_duplicate_session() -> None:
    """Test adding a duplicate session to the dataset."""
    dataset = SenselabDataset()
    session = Session()
    dataset.add_session(session)
    with pytest.raises(ValueError):
        dataset.add_session(session)


def test_get_participants() -> None:
    """Test getting the list of participants."""
    dataset = SenselabDataset()
    participant1 = Participant()
    participant2 = Participant()
    dataset.add_participant(participant1)
    dataset.add_participant(participant2)
    participants = dataset.get_participants()
    assert len(participants) == 2
    assert participant1 in participants
    assert participant2 in participants


def test_get_sessions() -> None:
    """Test getting the list of sessions."""
    dataset = SenselabDataset()
    session1 = Session()
    session2 = Session()
    dataset.add_session(session1)
    dataset.add_session(session2)
    sessions = dataset.get_sessions()
    assert len(sessions) == 2
    assert session1 in sessions
    assert session2 in sessions


@pytest.mark.skipif(TORCHAUDIO_AVAILABLE, reason="torchaudio is installed")
def test_audio_dataset_creation_import_error() -> None:
    """Tests that an ImportError is raised when torchaudio is not installed."""
    with pytest.raises(ModuleNotFoundError):
        dataset = SenselabDataset(audios=[MONO_AUDIO_PATH, STEREO_AUDIO_PATH])
        dataset.audios[0].waveform


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not installed")
def test_audio_dataset_creation() -> None:
    """Tests the creation of AudioDatasets with various ways of generating them."""
    mono_audio_data, mono_sr = torchaudio.load(MONO_AUDIO_PATH)
    stereo_audio_data, stereo_sr = torchaudio.load(STEREO_AUDIO_PATH)
    mono_audio = Audio(
        waveform=mono_audio_data,
        sampling_rate=mono_sr,
    )
    stereo_audio = Audio(
        waveform=stereo_audio_data,
        sampling_rate=stereo_sr,
    )

    audio_dataset_from_paths = SenselabDataset(audios=[MONO_AUDIO_PATH, STEREO_AUDIO_PATH])
    assert audio_dataset_from_paths.audios[0] == mono_audio and audio_dataset_from_paths.audios[1] == stereo_audio, (
        "Audio data generated from paths does not equal creating the individually"
    )

    audio_dataset_from_data = SenselabDataset(
        audios=[
            Audio(waveform=mono_audio_data, sampling_rate=mono_sr),
            Audio(waveform=stereo_audio_data, sampling_rate=stereo_sr),
        ],
    )

    assert audio_dataset_from_paths == audio_dataset_from_data, "Audio datasets should be equivalent"


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not installed")
def test_audio_dataset_splits() -> None:
    """Tests the AudioDataset split functionality."""
    audio_dataset = SenselabDataset(audios=[MONO_AUDIO_PATH, STEREO_AUDIO_PATH])
    mono_audio_data, mono_sr = torchaudio.load(MONO_AUDIO_PATH)
    stereo_audio_data, stereo_sr = torchaudio.load(STEREO_AUDIO_PATH)
    mono_audio = Audio(
        waveform=mono_audio_data,
        sampling_rate=mono_sr,
    )
    stereo_audio = Audio(
        waveform=stereo_audio_data,
        sampling_rate=stereo_sr,
    )

    no_param_cpu_split = audio_dataset.create_audio_split_for_pydra_task()
    assert no_param_cpu_split == [
        [mono_audio],
        [stereo_audio],
    ], "Default split should have been a list of each audio in its own list"

    gpu_split_exact = audio_dataset.create_audio_split_for_pydra_task(2)
    assert gpu_split_exact == [[mono_audio, stereo_audio]], (
        "Exact GPU split should generate a list with one list of all of the audios"
    )

    gpu_excess_split = audio_dataset.create_audio_split_for_pydra_task(4)
    assert gpu_excess_split == [[mono_audio, stereo_audio]], (
        "Excess GPU split should generate a list with one list of all of the audios, unpadded"
    )


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE or not AV_AVAILABLE, reason="torchaudio or av are not installed")
def test_convert_senselab_dataset_to_hf_datasets() -> None:
    """Tests the conversion of Senselab dataset to HuggingFace."""
    dataset = SenselabDataset(
        audios=[STEREO_AUDIO_PATH],
        videos=[VIDEO_PATH],
    )
    # print(dataset)
    # trim the video to 5 frames to speed up unit testing
    dataset.videos[0] = Video(
        frames=dataset.videos[0].frames[:5],
        frame_rate=dataset.videos[0].frame_rate,
        audio=dataset.videos[0].audio,
        metadata=dataset.videos[0].metadata,
    )

    # print(dataset.videos[0].audio.waveform)
    hf_datasets = dataset.convert_senselab_dataset_to_hf_datasets()
    # print(hf_datasets['videos'][0]['audio']['array'])

    # print(torch.max(torch.abs(dataset.videos[0].audio.waveform), dim=1))
    # print(torch.max(torch.abs(torch.tensor(hf_datasets['videos']['audio'][0]['array'])), dim=1))

    # print(torch.min((dataset.videos[0].audio.waveform), dim=1))
    # print(torch.min((torch.tensor(hf_datasets['videos']['audio'][0]['array'])), dim=1))

    audio_data = hf_datasets["audios"]
    video_data = hf_datasets["videos"]
    test_audio = Audio(filepath=STEREO_AUDIO_PATH)
    test_video = Video(filepath=VIDEO_PATH)

    # extracted_audio = extract_audios_from_local_videos('src/tests/data_for_testing/video_48khz_stereo_16bits.mp4')
    # extracted_audio, extract_sr = torchaudio.load(extracted_audio['audio'][0]['path'])

    test_video = Video(
        frames=test_video.frames[:5],
        frame_rate=test_video.frame_rate,
        audio=test_video.audio,
        metadata=test_video.metadata,
    )
    # print(hf_datasets)

    assert video_data.num_rows == 1
    assert audio_data.num_rows == 1
    assert torch.equal(torch.Tensor(audio_data["audio"][0]["array"]), test_audio.waveform)
    assert torch.equal(torch.Tensor(np.array(video_data["frames"][0]["image"])), test_video.frames)

    assert test_video.audio is not None
    assert torch.allclose(torch.Tensor(video_data["audio"][0]["array"]), test_video.audio.waveform, atol=1e-4, rtol=0)

    reconverted_dataset = SenselabDataset.convert_hf_dataset_to_senselab_dataset(hf_datasets)

    # print(torch.max(torch.abs(dataset.videos[0].audio.waveform), dim=1))
    # print(torch.max(torch.abs(torch.tensor(reconverted_dataset.videos[0].audio.waveform)), dim=1))

    # print(torch.min((dataset.videos[0].audio.waveform), dim=1))
    # print(torch.min((torch.tensor(reconverted_dataset.videos[0].audio.waveform)), dim=1))

    assert torch.allclose(reconverted_dataset.audios[0].waveform, dataset.audios[0].waveform, atol=1e-4, rtol=0)
    assert reconverted_dataset.audios[0].sampling_rate == dataset.audios[0].sampling_rate

    assert reconverted_dataset.videos[0].audio is not None
    assert torch.allclose(test_video.audio.waveform, reconverted_dataset.videos[0].audio.waveform, atol=1e-4, rtol=0)

    assert torch.equal(reconverted_dataset.videos[0].frames, dataset.videos[0].frames)
    assert dataset.videos[0].audio is not None
    assert torch.allclose(
        reconverted_dataset.videos[0].audio.waveform, dataset.videos[0].audio.waveform, rtol=0, atol=1e-4
    )
    assert reconverted_dataset.videos[0].frame_rate == dataset.videos[0].frame_rate


@pytest.mark.skipif(not LIBROSA_AVAILABLE, reason="librosa is not installed")
def test_convert_hf_dataset_to_senselab_dataset() -> None:
    """Use an existing HF dataset to show that Senselab properly converts and maintains a HF Dataset."""
    ravdness = load_dataset("xbgoose/ravdess", split="train")
    ravdness_features = list(ravdness.features)
    ravdness_features.remove("audio")
    if "metadata" in ravdness_features:
        ravdness_features.remove("metadata")
    senselab_ravdness = SenselabDataset.convert_hf_dataset_to_senselab_dataset(
        {"audios": ravdness}, transfer_metadata=True
    )

    assert len(senselab_ravdness.audios) == 1440
    assert set(senselab_ravdness.audios[0].metadata.keys()) == set(ravdness_features)

    senselab_ravdness = SenselabDataset.convert_hf_dataset_to_senselab_dataset({"audios": ravdness})

    assert len(senselab_ravdness.audios) == 1440
    assert senselab_ravdness.audios[0].metadata == {}
