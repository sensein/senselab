"""Module for testing data augmentation on audios."""

import torch
from torch_audiomentations import Compose, PolarityInversion

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.data_augmentation import augment_audios
from senselab.utils.data_structures import SenselabDataset
from tests.audio.conftest import MONO_AUDIO_PATH, STEREO_AUDIO_PATH


def test_audio_data_augmentation() -> None:
    """Test data augmentations using the new Audio data types."""
    apply_augmentation = Compose(transforms=[PolarityInversion(p=1, output_type="dict")], output_type="dict")

    audio_paths = [
        MONO_AUDIO_PATH,
        STEREO_AUDIO_PATH,
    ]
    audio_dataset_from_paths = SenselabDataset(audios=audio_paths)
    mono_audio, stereo_audio = audio_dataset_from_paths.create_audio_split_for_pydra_task()
    mono_inverted = augment_audios(mono_audio, apply_augmentation)
    stereo_inverted = augment_audios(stereo_audio, apply_augmentation)
    assert torch.equal(
        mono_audio[0].waveform, -1 * mono_inverted[0].waveform
    ), "Audio should have been inverted by the augmentation"
    assert torch.equal(
        stereo_audio[0].waveform, -1 * stereo_inverted[0].waveform
    ), "Audio should have been inverted by the augmentation and not affected by stereo audio"

    batched_audio = SenselabDataset(
        audios=[
            Audio(waveform=stereo_audio[0].waveform[0], sampling_rate=stereo_audio[0].sampling_rate),
            Audio(waveform=stereo_audio[0].waveform[1], sampling_rate=stereo_audio[0].sampling_rate),
        ]
    ).create_audio_split_for_pydra_task(2)
    batch_inverted = augment_audios(batched_audio[0], apply_augmentation)
    assert torch.equal(batched_audio[0][0].waveform, -1 * batch_inverted[0].waveform) and torch.equal(
        batched_audio[0][1].waveform, -1 * batch_inverted[1].waveform
    )
