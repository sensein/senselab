"""Module for testing data augmentation on audios."""

import torch
from torch_audiomentations import Compose, PolarityInversion

from senselab.audio.tasks.data_augmentation import augment_audio_dataset
from senselab.utils.data_structures.audio import AudioDataset


def test_audio_data_augmentation() -> None:
    """Test data augmentations using the new Audio data types."""
    apply_augmentation = Compose(transforms=[PolarityInversion(p=1, output_type="dict")], output_type="dict")

    audio_paths = [
        "src/tests/data_for_testing/audio_48khz_mono_16bits.wav",
        "src/tests/data_for_testing/audio_48khz_stereo_16bits.wav",
    ]
    audio_dataset_from_paths = AudioDataset.generate_dataset_from_filepaths(audio_filepaths=audio_paths)
    mono_audio, stereo_audio = audio_dataset_from_paths.create_split_for_pydra_task()
    mono_inverted = augment_audio_dataset(mono_audio, apply_augmentation, batched=False)
    stereo_inverted = augment_audio_dataset(stereo_audio, apply_augmentation, batched=False)
    assert torch.equal(
        mono_audio[0].waveform, -1 * mono_inverted[0].waveform
    ), "Audio should have been inverted by the augmentation"
    assert torch.equal(
        stereo_audio[0].waveform, -1 * stereo_inverted[0].waveform
    ), "Audio should have been inverted by the augmentation and not affected by stereo audio"

    batched_audio = AudioDataset.generate_dataset_from_audio_data(
        [stereo_audio[0].waveform[0], stereo_audio[0].waveform[1]], stereo_audio[0].sampling_rate
    ).create_split_for_pydra_task(True, 2)
    batch_inverted = augment_audio_dataset(batched_audio[0], apply_augmentation, batched=True)
    assert torch.equal(batched_audio[0][0].waveform, -1 * batch_inverted[0].waveform) and torch.equal(
        batched_audio[0][1].waveform, -1 * batch_inverted[1].waveform
    )
