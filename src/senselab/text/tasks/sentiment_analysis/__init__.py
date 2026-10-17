""".. include:: ./doc.md"""  # noqa: D415

from .api import analyze_sentiment, classify_text_emotions  # noqa: F401
from .data_structures import EmotionResult, SentimentResult  # noqa: F401

__all__ = ["EmotionResult", "SentimentResult", "analyze_sentiment", "classify_text_emotions"]
