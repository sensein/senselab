"""Module for testing data augmentation on audios."""

import torch
from torch_audiomentations import AddColoredNoise, Compose, Gain, PitchShift, PolarityInversion

from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.data_augmentation.data_augmentation import augment_audios


def test_audio_data_augmentation(resampled_mono_audio_sample: Audio, resampled_stereo_audio_sample: Audio) -> None:
    """Test data augmentations using the new Audio data types."""
    apply_augmentation = Compose(transforms=[PolarityInversion(p=1, output_type="dict")], output_type="dict")

    mono_inverted = augment_audios([resampled_mono_audio_sample], apply_augmentation)
    stereo_inverted = augment_audios([resampled_stereo_audio_sample], apply_augmentation)
    assert torch.equal(
        resampled_mono_audio_sample.waveform, -1 * mono_inverted[0].waveform
    ), "Audio should have been inverted by the augmentation"
    assert torch.equal(
        resampled_stereo_audio_sample.waveform, -1 * stereo_inverted[0].waveform
    ), "Audio should have been inverted by the augmentation and not affected by stereo audio"

    batched_audio = [
        Audio(
            waveform=resampled_stereo_audio_sample.waveform[0],
            sampling_rate=resampled_stereo_audio_sample.sampling_rate,
        ),
        Audio(
            waveform=resampled_stereo_audio_sample.waveform[1],
            sampling_rate=resampled_stereo_audio_sample.sampling_rate,
        ),
    ]
    batch_inverted = augment_audios(batched_audio, apply_augmentation)
    assert torch.equal(batched_audio[0].waveform, -1 * batch_inverted[0].waveform) and torch.equal(
        batched_audio[1].waveform, -1 * batch_inverted[1].waveform
    )


def test_more_augmentations(resampled_mono_audio_sample: Audio, resampled_stereo_audio_sample: Audio) -> None:
    """Test more augmentations for audiomentations."""
    apply_augmentation = Compose(
        transforms=[
            Gain(p=1, output_type="dict"),
            PolarityInversion(p=1, output_type="dict"),
            PitchShift(p=1, output_type="dict", sample_rate=resampled_mono_audio_sample.sampling_rate),
        ],
        output_type="dict",
    )

    mono_augmented = augment_audios([resampled_mono_audio_sample], apply_augmentation)
    assert not torch.equal(
        resampled_mono_audio_sample.waveform, mono_augmented[0].waveform
    ), "Mono audio should have been augmented by the composed augmentations"

    stereo_augmented = augment_audios([resampled_stereo_audio_sample], apply_augmentation)
    assert not torch.equal(
        resampled_stereo_audio_sample.waveform, stereo_augmented[0].waveform
    ), "Stereo audio should have been augmented by the composed augmentations"


def test_silence_audio(silent_audio_sample: Audio) -> None:
    """Test with silence-only audio."""
    apply_augmentation = Compose(
        transforms=[PolarityInversion(p=1, output_type="dict"), Gain(p=1, output_type="dict")], output_type="dict"
    )
    silence_inverted = augment_audios([silent_audio_sample], apply_augmentation)
    assert torch.equal(
        silent_audio_sample.waveform, -1 * silence_inverted[0].waveform
    ), "Silence audio should have been inverted by the augmentation"


def test_noise_audio(noise_audio_sample: Audio) -> None:
    """Test with noise-only audio."""
    apply_augmentation = Compose(
        transforms=[PolarityInversion(p=1, output_type="dict"), AddColoredNoise()], output_type="dict"
    )
    noise_augmented = augment_audios([noise_audio_sample], apply_augmentation)
    assert not torch.equal(
        noise_audio_sample.waveform, noise_augmented[0].waveform
    ), "Noise audio should have been augmented by the composed augmentations"


def test_empty_and_short_audio(empty_audio_sample: Audio, short_audio_sample: Audio) -> None:
    """Test an empty audio file and a very short audio file."""
    apply_augmentation = Compose(transforms=[PolarityInversion(p=1, output_type="dict")], output_type="dict")

    # Test with empty audio
    empty_augmented = augment_audios([empty_audio_sample], apply_augmentation)
    assert (
        empty_audio_sample.waveform.shape == empty_augmented[0].waveform.shape
    ), "Empty audio should remain unchanged by the augmentation"

    # Test with short audio
    short_augmented = augment_audios([short_audio_sample], apply_augmentation)
    assert torch.equal(
        short_audio_sample.waveform, -1 * short_augmented[0].waveform
    ), "Short audio should have been inverted by the augmentation"
