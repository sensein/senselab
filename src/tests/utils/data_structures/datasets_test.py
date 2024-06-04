"""Module for testing Datasets for Senselab."""

import torchaudio

from senselab.utils.data_structures.audio import Audio
from senselab.utils.data_structures.datasets import SenselabDataset


def test_audio_dataset_creation() -> None:
    """Tests the creation of AudioDatasets with various ways of generating them."""
    audio_paths = [
        "src/tests/data_for_testing/audio_48khz_mono_16bits.wav",
        "src/tests/data_for_testing/audio_48khz_stereo_16bits.wav",
    ]

    mono_audio_data, mono_sr = torchaudio.load("src/tests/data_for_testing/audio_48khz_mono_16bits.wav")
    stereo_audio_data, stereo_sr = torchaudio.load("src/tests/data_for_testing/audio_48khz_stereo_16bits.wav")
    mono_audio = Audio(
        waveform=mono_audio_data,
        sampling_rate=mono_sr,
        orig_path_or_id="src/tests/data_for_testing/audio_48khz_mono_16bits.wav",
    )
    stereo_audio = Audio(
        waveform=stereo_audio_data,
        sampling_rate=stereo_sr,
        orig_path_or_id="src/tests/data_for_testing/audio_48khz_stereo_16bits.wav",
    )

    audio_dataset_from_paths = SenselabDataset(audios=audio_paths)
    assert (
        audio_dataset_from_paths.audios[0] == mono_audio and audio_dataset_from_paths.audios[1] == stereo_audio
    ), "Audio data generated from paths does not equal creating the individually"

    audio_dataset_from_data = SenselabDataset(
        audios=[
            Audio(waveform=mono_audio_data, sampling_rate=mono_sr),
            Audio(waveform=stereo_audio_data, sampling_rate=stereo_sr),
        ],
    )

    assert audio_dataset_from_paths == audio_dataset_from_data, "Audio datasets should be equivalent"


def test_audio_dataset_splits() -> None:
    """Tests the AudioDataset split functionality."""
    audio_paths = [
        "src/tests/data_for_testing/audio_48khz_mono_16bits.wav",
        "src/tests/data_for_testing/audio_48khz_stereo_16bits.wav",
    ]
    audio_dataset = SenselabDataset(audios=audio_paths)
    mono_audio_data, mono_sr = torchaudio.load("src/tests/data_for_testing/audio_48khz_mono_16bits.wav")
    stereo_audio_data, stereo_sr = torchaudio.load("src/tests/data_for_testing/audio_48khz_stereo_16bits.wav")
    mono_audio = Audio(
        waveform=mono_audio_data,
        sampling_rate=mono_sr,
        orig_path_or_id="src/tests/data_for_testing/audio_48khz_mono_16bits.wav",
    )
    stereo_audio = Audio(
        waveform=stereo_audio_data,
        sampling_rate=stereo_sr,
        orig_path_or_id="src/tests/data_for_testing/audio_48khz_stereo_16bits.wav",
    )

    no_param_cpu_split = audio_dataset.create_audio_split_for_pydra_task()
    assert no_param_cpu_split == [
        [mono_audio],
        [stereo_audio],
    ], "Default split should have been a list of each audio in its own list"

    gpu_split_exact = audio_dataset.create_audio_split_for_pydra_task(2)
    assert gpu_split_exact == [
        [mono_audio, stereo_audio]
    ], "Exact GPU split should generate a list with one list of all of the audios"

    gpu_excess_split = audio_dataset.create_audio_split_for_pydra_task(4)
    assert gpu_excess_split == [
        [mono_audio, stereo_audio]
    ], "Excess GPU split should generate a list with one list of all of the audios, unpadded"
