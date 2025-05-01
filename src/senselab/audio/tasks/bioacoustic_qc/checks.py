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

import pandas as pd
import torch

from senselab.audio.data_structures import Audio


def audio_length_positive_check(audio: Audio) -> bool:
    """Checks if an Audio object has a positive length."""
    return audio.waveform.numel() != 0


def audio_intensity_positive_check(audio: Audio) -> bool:
    """Checks if an Audio object has nonzero intensity."""
    return torch.sum(torch.abs(audio.waveform)).item() != 0
