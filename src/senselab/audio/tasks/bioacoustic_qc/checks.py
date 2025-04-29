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
from senselab.audio.tasks.bioacoustic_qc.metrics import (
    amplitude_headroom_metric,
    proportion_clipped_metric,
    proportion_silent_metric,
)


def audio_length_positive_check(audio: Audio) -> bool:
    """Checks if an Audio object has a positive length."""
    return audio.waveform.numel() != 0


def proportion_clipped_check(audio: Audio, threshold: float = 0.0001) -> bool:
    """Checks if an Audio object has less than 0.01% clipped samples."""
    return proportion_clipped_metric(audio) < threshold


def completely_silent_check(audio: Audio, silence_threshold: float = 0.01) -> bool:
    """Checks if an Audio object is completely silent."""
    return proportion_silent_metric(audio, silence_threshold=silence_threshold) < 1.0


def mostly_silent_check(audio: Audio, silence_threshold: float = 0.01, max_silent_proportion: float = 0.95) -> bool:
    """Checks if an Audio object is completely silent."""
    return proportion_silent_metric(audio, silence_threshold=silence_threshold) < max_silent_proportion


def very_low_headroom_check(audio: Audio, headroom_threshold: float = 0.005) -> bool:
    """Checks if an Audio object has very low headroom."""
    return amplitude_headroom_metric(audio) < headroom_threshold


def very_high_headroom_check(audio: Audio, headroom_threshold: float = 0.95) -> bool:
    """Checks if an Audio object has very low headroom."""
    return amplitude_headroom_metric(audio) > headroom_threshold
