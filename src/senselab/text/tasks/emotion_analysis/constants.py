"""This module defines the Emotion enum and related constants for emotional analysis."""

from enum import Enum
from typing import Any, Dict

from senselab.utils.data_structures.model import HFModel
from senselab.utils.model_utils import HFUtils


class Emotion(Enum):
    """Enum class representing different emotions."""

    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"
    DISAPPOINTMENT = "disappointment"
    ANNOYANCE = "annoyance"
    DISAPPROVAL = "disapproval"
    REALIZATION = "realization"
    NERVOUSNESS = "nervousness"
    EMBARRASSMENT = "embarrassment"
    CARING = "caring"
    REMORSE = "remorse"
    GRIEF = "grief"
    CONFUSION = "confusion"
    RELIEF = "relief"
    DESIRE = "desire"
    ADMIRATION = "admiration"
    OPTIMISM = "optimism"
    LOVE = "love"
    EXCITEMENT = "excitement"
    CURIOSITY = "curiosity"
    AMUSEMENT = "amusement"
    GRATITUDE = "gratitude"
    PRIDE = "pride"
    APPROVAL = "approval"


MODEL_TYPE_TO_UTILS: Dict[Any, Any] = {HFModel: HFUtils}
