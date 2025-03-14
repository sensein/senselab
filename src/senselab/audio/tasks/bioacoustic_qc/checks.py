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
from pandas import pd

from senselab.audio.data_structures import Audio


def apply_audio_quality_check(
    df: pd.DataFrame, activity_audios: List[Audio], condition: Callable[[Audio], bool]
) -> pd.DataFrame:
    """Applies a condition to each audio and stores results in a new column with the function name.

    Args:
        df (pd.DataFrame): DataFrame containing audio metadata with an 'audio_path_or_id' column.
        activity_audios (List[Audio]): List of Audio objects to check.
        condition (Callable[[Audio], bool]): Function that evaluates an Audio object and returns a bool.

    Returns:
        pd.DataFrame: Updated DataFrame with a new column for the check results.
    """
    column_name = condition.__name__
    audio_dict = {audio.orig_path_or_id: condition(audio) for audio in activity_audios}
    df[column_name] = df["audio_path_or_id"].map(audio_dict).fillna(float("nan"))
    return df


def audio_length_positive_check(audio: Audio) -> bool:
    """Checks if an Audio object has a positive length."""
    return audio.waveform.numel() == 0


def audio_intensity_positive_check(audio: Audio) -> bool:
    """Checks if an Audio object has nonzero intensity."""
    return torch.sum(torch.abs(audio.waveform)) == 0
