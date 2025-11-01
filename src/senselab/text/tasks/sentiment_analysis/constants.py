"""This module defines the Sentiment enum and related constants for sentiment analysis."""

from enum import Enum
from typing import Any, Dict

from senselab.utils.data_structures.model import HFModel
from senselab.utils.model_utils import HFUtils


class Sentiment(Enum):
    """Enum class representing sentiment labels."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


MODEL_TYPE_TO_UTILS: Dict[Any, Any] = {HFModel: HFUtils}
