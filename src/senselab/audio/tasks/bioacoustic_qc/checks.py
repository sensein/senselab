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


def audio_intensity_positive_check(audio: Audio) -> Tuple[Dict[str, List[Audio] | None], List[Audio]]:
    """Checks if an Audio object has nonzero intensity.

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

    if torch.sum(torch.abs(audio.waveform)) == 0:  # Use sum instead of mean to check for silence
        exclude.append(audio)  # Exclude silent or empty audio
    else:
        passed.append(audio)

    return {"exclude": exclude, "review": None}, passed
