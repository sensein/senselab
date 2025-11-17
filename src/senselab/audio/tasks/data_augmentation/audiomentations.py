"""This module contains functions for applying data augmentation using audiomentations."""

from __future__ import annotations

from typing import List

from senselab.audio.data_structures import Audio

try:
    from audiomentations import Compose

    AUDIOMENTATIONS_AVAILABLE = True
except ModuleNotFoundError:
    AUDIOMENTATIONS_AVAILABLE = False


def augment_audios_with_audiomentations(audios: List[Audio], augmentation: "Compose") -> List[Audio]:
    """Apply data augmentation using **audiomentations** by looping over the inputs.

    The provided `audiomentations.Compose` pipeline is applied sequentially to each
    `Audio` object.

    Notes:
        - This function expects a CPU-only pipeline (audiomentations is NumPy-based).
        - For reproducibility, construct your `Compose` with your own random seed
          strategy (e.g., seeding your RNG before creating the pipeline).
        - The returned `Audio` objects preserve sampling rate and copy metadata.

    Args:
        audios (list[Audio]):
            List of `Audio` objects to augment.
        augmentation (Compose):
            An `audiomentations.Compose` pipeline (CPU).

    Returns:
        list[Audio]: Augmented `Audio` objects in the same order as input.

    Raises:
        ModuleNotFoundError: If `audiomentations` is not installed.
        Exception: Any error raised during augmentation.

    Example:
        >>> from audiomentations import Compose, AddGaussianNoise, Gain
        >>> from senselab.audio.data_structures import Audio
        >>> from senselab.audio.tasks.data_augmentation.audiomentations import (
        ...     augment_audios_with_audiomentations
        ... )
        >>> aug = Compose([
        ...     AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=1.0),
        ...     Gain(min_gain_in_db=-6, max_gain_in_db=6, p=1.0),
        ... ])
        >>> a1 = Audio(filepath="/abs/path/sample1.wav")
        >>> a2 = Audio(filepath="/abs/path/sample2.wav")
        >>> out = augment_audios_with_audiomentations([a1, a2], aug)
        >>> len(out)
        2
    """
    if not AUDIOMENTATIONS_AVAILABLE:
        raise ModuleNotFoundError(
            "`audiomentations` is not installed. "
            "Please install senselab audio dependencies using `pip install senselab`."
        )

    augmented_audios: List[Audio] = []
    for audio in audios:
        augmented = augmentation(samples=audio.waveform, sample_rate=audio.sampling_rate)
        augmented_audios.append(
            Audio(
                waveform=augmented,
                sampling_rate=audio.sampling_rate,
                metadata=audio.metadata.copy(),
            )
        )
    return augmented_audios
