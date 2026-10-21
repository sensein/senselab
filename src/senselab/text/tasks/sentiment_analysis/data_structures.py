"""Result types for text sentiment and emotion classification."""

from typing import Dict, Optional

from pydantic import BaseModel


class TextClassificationResult(BaseModel):
    """Full label distribution for one input text.

    Attributes:
        text: The input text, unchanged.
        scores: Probability per label, in the model's label-index order. Empty for an empty text.
        label: The highest-probability label; None for an empty text.
        confidence: Probability of ``label``; None for an empty text.
        n_windows: Number of token windows averaged (1 unless windowing split the text; 0 for an empty text).
        truncated: True when text beyond the model's maximum length was discarded.
        model_id: Hub id of the model that produced the scores.
        commit_sha: Commit the weights were loaded from.
    """

    text: str
    scores: Dict[str, float]
    label: Optional[str] = None
    confidence: Optional[float] = None
    n_windows: int = 1
    truncated: bool = False
    model_id: str = ""
    commit_sha: str = ""

    @property
    def is_empty(self) -> bool:
        """True when the input had no content and no scores were computed."""
        return not self.scores


class SentimentResult(TextClassificationResult):
    """Sentiment distribution with a signed valence.

    Attributes:
        valence: ``p_positive - p_negative`` in [-1, 1]; None for an empty text.
    """

    valence: Optional[float] = None


class EmotionResult(TextClassificationResult):
    """Emotion distribution over the model's emotion labels."""
