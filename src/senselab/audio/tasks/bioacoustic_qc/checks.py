"""Audio Quality Control Checks.

Each check function processes a list of Audio objects and returns a dictionary
indicating which files should be excluded or reviewed, along with a list of
audio files that passed the check.

All checks take a list of Audio objects as input and return:
    - A dictionary with the structure:
        {
            "exclude": [Audio],  # List of Audio objects that failed the check
            "review": [Audio]    # List of Audio objects that need manual review
        }
    - A list of Audio objects that passed the check.
"""

from typing import Dict, List, Tuple

import torch

from senselab.audio.data_structures import Audio


def audio_length_positive_check(audio: Audio) -> Tuple[Dict[str, List[Audio] | None], List[Audio]]:
    """Checks if an Audio object has a positive length.

    Args:
        audio (Audio): The Audio object to validate.

    Returns:
        Tuple[Dict[str, List[Audio] | None], List[Audio]]: A tuple containing:
            - A dictionary with:
                - "exclude": A list of Audio objects that failed the check.
                - "review": Always None (not used in this check).
            - A list of Audio objects that passed the check.
    """
    exclude = []
    passed = []

    if audio.waveform.numel() == 0:  # No samples
        exclude.append(audio)
    else:
        passed.append(audio)

    return {"exclude": exclude, "review": None}, passed


def audio_intensity_positive_check(audio: Audio) -> bool:
    """Checks if an Audio object has nonzero intensity.

    Args:
        audio (Audio): The Audio object to validate.

    Returns:
        bool: True if intensity is greater than zero, False otherwise.

    Raises:
        ValueError: If the waveform is empty or not a 2D tensor.
    """
    if not isinstance(audio.waveform, torch.Tensor) or audio.waveform.ndim != 2:
        raise ValueError("Waveform must be a 2D torch.Tensor with shape (num_channels, num_samples).")

    return torch.mean(torch.abs(audio.waveform)) > 0
