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

from typing import Dict, List

import torch

from senselab.audio.data_structures import Audio


def audio_length_positive_check(audios: List[Audio], task_audios: List[Audio]) -> Dict:
    """Checks if an Audio object has a positive length.

    Args:
        audios (Audio): The Audio object to validate.
        task_audios (List[Audio]): The subset of Audio objects to check.

    Returns:
        Dict[str, List[Audio]]: A tuple containing:
            - A dictionary with:
                - "exclude": A list of Audio objects that failed the check.
                - "review": Always None (not used in this check).
                - "passed": List of Audio objects that passed the check.
    """
    exclude = []
    passed = []

    for audio in task_audios:  # iterate over task_audios
        if audio in audios:
            if audio.waveform.numel() == 0:  # No samples
                exclude.append(audio)
                audios.remove(audio)  # remove from all audios
            else:
                passed.append(audio)

    return {"exclude": exclude, "review": [], "passed": passed}


def audio_intensity_positive_check(audios: List[Audio], task_audios: List[Audio]) -> Dict:
    """Checks if each Audio object has nonzero intensity.

    Args:
        audios (List[Audio]): The complete dataset of Audio objects.
        task_audios (List[Audio]): The subset of Audio objects to check.

    Returns:
        Dict[str, List[Audio]]: A tuple containing:
            - A dictionary with:
                - "exclude": List of Audio objects that have zero intensity.
                - "review": Always an empty list (not used in this check).
                - "passed": List of Audio objects that passed the check.
    """
    exclude = []
    passed = []

    for audio in task_audios:
        if audio in audios:
            if torch.sum(torch.abs(audio.waveform)) == 0:
                exclude.append(audio)
                audios.remove(audio)
            else:
                passed.append(audio)

    return {"exclude": exclude, "review": [], "passed": passed}
