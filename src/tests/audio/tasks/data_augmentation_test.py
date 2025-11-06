"""Module for testing data augmentation on audios."""

import pytest
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.data_augmentation import augment_audios
from senselab.utils.data_structures import SenselabDataset
from tests.audio.conftest import MONO_AUDIO_PATH, STEREO_AUDIO_PATH

try:
    from audiomentations import Compose as AudiomentationsCompose
    from audiomentations import Gain

    AUDIOMENTATIONS_AVAILABLE = True
except ModuleNotFoundError:
    AUDIOMENTATIONS_AVAILABLE = False

try:
    from torch_audiomentations import Compose, PolarityInversion

    TORCH_AUDIOMENTATIONS_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AUDIOMENTATIONS_AVAILABLE = False


@pytest.mark.skipif(AUDIOMENTATIONS_AVAILABLE, reason="audiomentations is installed.")
def test_audio_data_augmentation_without_audiomentations() -> None:
    """Test data augmentations without audiomentations."""
    with pytest.raises(NameError):
        _ = AudiomentationsCompose(transforms=[Gain(min_gain_db=14.99, max_gain_db=15, p=1.0)])


@pytest.mark.skipif(TORCH_AUDIOMENTATIONS_AVAILABLE, reason="torch-audiomentations is installed.")
def test_audio_data_augmentation_without_torch_audiomentations() -> None:
    """Test data augmentations without torch-audiomentations."""
    with pytest.raises(NameError):
        _ = Compose(transforms=[PolarityInversion(p=1, output_type="dict")], output_type="dict")


@pytest.mark.skipif(not TORCH_AUDIOMENTATIONS_AVAILABLE, reason="audiomentations is not installed.")
def test_audio_data_augmentation_with_torch_audiomentations() -> None:
    """Test data augmentations using the new Audio data types with torch-audiomentations."""
    apply_augmentation = Compose(transforms=[PolarityInversion(p=1, output_type="dict")], output_type="dict")

    audio_paths = [
        MONO_AUDIO_PATH,
        STEREO_AUDIO_PATH,
    ]
    audio_dataset_from_paths = SenselabDataset(audios=audio_paths)
    mono_audio, stereo_audio = audio_dataset_from_paths.create_audio_split_for_pydra_task()
    mono_inverted = augment_audios(mono_audio, apply_augmentation)
    stereo_inverted = augment_audios(stereo_audio, apply_augmentation)
    assert torch.allclose(
        mono_audio[0].waveform, -1 * mono_inverted[0].waveform, atol=1e-3
    ), "Audio should have been inverted by the augmentation"
    assert torch.allclose(
        stereo_audio[0].waveform, -1 * stereo_inverted[0].waveform, atol=1e-3
    ), "Audio should have been inverted by the augmentation and not affected by stereo audio"

    batched_audio = SenselabDataset(
        audios=[
            Audio(waveform=stereo_audio[0].waveform[0], sampling_rate=stereo_audio[0].sampling_rate),
            Audio(waveform=stereo_audio[0].waveform[1], sampling_rate=stereo_audio[0].sampling_rate),
        ]
    ).create_audio_split_for_pydra_task(2)
    batch_inverted = augment_audios(batched_audio[0], apply_augmentation)
    assert torch.allclose(batched_audio[0][0].waveform, -1 * batch_inverted[0].waveform, atol=1e-3) and torch.allclose(
        batched_audio[0][1].waveform, -1 * batch_inverted[1].waveform, atol=1e-3
    )

    # Test error when augmenting mixed channel audios in a batch
    if torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="All audios must have the same number of channels."):
            augment_audios([mono_audio[0], stereo_audio[0]], apply_augmentation)


@pytest.mark.skipif(not AUDIOMENTATIONS_AVAILABLE, reason="audiomentations is not installed.")
def test_audio_data_augmentation_with_audiomentations(mono_audio_sample: Audio, stereo_audio_sample: Audio) -> None:
    """Test data augmentations using the new Audio data types with audiomentations."""
    apply_augmentation = AudiomentationsCompose(transforms=[Gain(min_gain_db=14.99, max_gain_db=15, p=1.0)])

    # Augmenting mono and stereo audio clips
    augmented_audios = augment_audios([mono_audio_sample, stereo_audio_sample], apply_augmentation)
    assert len(augmented_audios) == 2
