"""Module for testing Audio data structures."""

import torch
import torchaudio

from senselab.utils.data_structures.audio import Audio, AudioDataset


def test_audio_creation() -> None:
    """Tests the functionality for creating data instances."""
    mono_audio_data, mono_sr = torchaudio.load("src/tests/data_for_testing/audio_48khz_mono_16bits.wav")
    stereo_audio_data, stereo_sr = torchaudio.load("src/tests/data_for_testing/audio_48khz_stereo_16bits.wav")

    mono_audio = Audio(
        waveform=mono_audio_data,
        sampling_rate=mono_sr,
        path_or_id="src/tests/data_for_testing/audio_48khz_mono_16bits.wav",
    )
    mono_audio_from_file = Audio.from_filepath("src/tests/data_for_testing/audio_48khz_mono_16bits.wav")
    assert mono_audio == mono_audio_from_file, "Mono audios are not exactly equivalent"

    stereo_audio = Audio(
        waveform=stereo_audio_data,
        sampling_rate=stereo_sr,
        path_or_id="src/tests/data_for_testing/audio_48khz_stereo_16bits.wav",
    )
    stereo_audio_uuid = Audio(waveform=stereo_audio_data, sampling_rate=stereo_sr)
    stereo_audio_from_file = Audio.from_filepath("src/tests/data_for_testing/audio_48khz_stereo_16bits.wav")
    assert stereo_audio == stereo_audio_from_file, "Stereo audios are not exactly equivalent"
    assert stereo_audio != stereo_audio_uuid, "Stereo audio with different IDs were equivalent when they shouldn't be"

    audio_single_tensor = Audio(waveform=mono_audio_data[0], sampling_rate=mono_sr)
    assert torch.equal(
        mono_audio.waveform, audio_single_tensor.waveform
    ), "Mono audios of tensor shape (num_samples,) should be reshaped to (1, num_samples)"

    audio_from_list = Audio(waveform=list(mono_audio_data[0]), sampling_rate=mono_sr)
    audio_from_list_of_lists = Audio(waveform=[list(mono_audio_data[0])], sampling_rate=mono_sr)
    audio_from_numpy = Audio(waveform=mono_audio_data.numpy(), sampling_rate=mono_sr)

    assert torch.equal(mono_audio.waveform, audio_from_list.waveform), "List audio should've been converted to Tensor"
    assert torch.equal(
        mono_audio.waveform, audio_from_list_of_lists.waveform
    ), "List of lists audio should've been converted to Tensor"
    assert torch.equal(mono_audio.waveform, audio_from_numpy.waveform), "NumPy audio should've been converted to Tensor"


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
        path_or_id="src/tests/data_for_testing/audio_48khz_mono_16bits.wav",
    )
    stereo_audio = Audio(
        waveform=stereo_audio_data,
        sampling_rate=stereo_sr,
        path_or_id="src/tests/data_for_testing/audio_48khz_stereo_16bits.wav",
    )

    audio_dataset_from_paths = AudioDataset.generate_dataset_from_filepaths(audio_filepaths=audio_paths)
    assert (
        audio_dataset_from_paths.audios[0] == mono_audio and audio_dataset_from_paths.audios[1] == stereo_audio
    ), "Audio data generated from paths does not equal creating the individually"

    audio_dataset_from_data = AudioDataset.generate_dataset_from_audio_data(
        audios_data=[mono_audio_data, stereo_audio_data],
        sampling_rates=[mono_sr, stereo_sr],
        audio_paths_or_ids=audio_paths,
    )

    assert audio_dataset_from_paths == audio_dataset_from_data, "Audio datasets should be equivalent"


def test_audio_dataset_splits() -> None:
    """Tests the AudioDataset split functionality."""
    audio_paths = [
        "src/tests/data_for_testing/audio_48khz_mono_16bits.wav",
        "src/tests/data_for_testing/audio_48khz_stereo_16bits.wav",
    ]
    audio_dataset = AudioDataset.generate_dataset_from_filepaths(audio_paths)
    mono_audio_data, mono_sr = torchaudio.load("src/tests/data_for_testing/audio_48khz_mono_16bits.wav")
    stereo_audio_data, stereo_sr = torchaudio.load("src/tests/data_for_testing/audio_48khz_stereo_16bits.wav")
    mono_audio = Audio(
        waveform=mono_audio_data,
        sampling_rate=mono_sr,
        path_or_id="src/tests/data_for_testing/audio_48khz_mono_16bits.wav",
    )
    stereo_audio = Audio(
        waveform=stereo_audio_data,
        sampling_rate=stereo_sr,
        path_or_id="src/tests/data_for_testing/audio_48khz_stereo_16bits.wav",
    )

    no_param_cpu_split = audio_dataset.create_split_for_pydra_task()
    assert no_param_cpu_split == [
        [mono_audio],
        [stereo_audio],
    ], "Default split should have been a list of each audio in its own list"

    gpu_split_exact = audio_dataset.create_split_for_pydra_task(True, 2)
    assert gpu_split_exact == [
        [mono_audio, stereo_audio]
    ], "Exact GPU split should generate a list with one list of all of the audios"

    gpu_excess_split = audio_dataset.create_split_for_pydra_task(True, 4)
    assert gpu_excess_split == [
        [mono_audio, stereo_audio]
    ], "Excess GPU split should generate a list with one list of all of the audios, unpadded"

    audio_dataset_preset_gpu = AudioDataset.generate_dataset_from_filepaths(audio_paths, use_gpu=True, batch_size=2)
    no_param_gpu_split = audio_dataset_preset_gpu.create_split_for_pydra_task()
    assert no_param_gpu_split == [
        [mono_audio, stereo_audio]
    ], "Exact GPU split from dataset presets should generate a list with one list of all of the audios"
