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

from typing import Callable, Dict, List

import torch

from senselab.audio.data_structures import Audio


def audio_quality_check(audios: List[Audio], activity_audios: List[Audio], condition: Callable[[Audio], bool]) -> Dict:
    """Generic function to check audio quality based on a given condition.

    Args:
        audios (List[Audio]): The complete dataset of Audio objects.
        activity_audios (List[Audio]): The subset of Audio objects to check.
        condition (Callable[[Audio], bool]): A function that returns True if the
            audio fails the check (should be excluded).

    Returns:
        Dict[str, List[Audio]]: A dictionary containing:
            - "exclude": List of Audio objects that failed the check.
            - "review": Always an empty list (not used in these checks).
            - "passed": List of Audio objects that passed the check.
    """
    exclude = []
    passed = []

    for audio in activity_audios:
        if audio in audios:
            if condition(audio):
                exclude.append(audio)
                audios.remove(audio)
            else:
                passed.append(audio)

    return {"exclude": exclude, "review": [], "passed": passed}


def audio_length_positive_check(audios: List[Audio], activity_audios: List[Audio]) -> Dict:
    """Checks if an Audio object has a positive length."""
    return audio_quality_check(audios, activity_audios, lambda audio: audio.waveform.numel() == 0)


def audio_intensity_positive_check(audios: List[Audio], activity_audios: List[Audio]) -> Dict:
    """Checks if an Audio object has nonzero intensity."""
    return audio_quality_check(audios, activity_audios, lambda audio: torch.sum(torch.abs(audio.waveform)) == 0)
