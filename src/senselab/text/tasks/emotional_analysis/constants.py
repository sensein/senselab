"""This module defines the Emotion enum and related functions for emotional analysis."""

from enum import Enum


class Emotion(Enum):
    """Enum class representing different emotions."""

    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"

    @classmethod
    def from_string(cls, emotion_string: str) -> "Emotion":
        """Convert a string to an Emotion enum value.

        Args:
            emotion_string: The string representation of the emotion.

        Returns:
            The corresponding Emotion enum value.

        Raises:
            ValueError: If the input string doesn't match any Emotion enum value.
        """
        for emotion in cls:
            if emotion.value == emotion_string.lower():
                return emotion
        raise ValueError(f"'{emotion_string}' is not a valid Emotion")
