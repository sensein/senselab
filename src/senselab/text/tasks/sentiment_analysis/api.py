"""Public API for text sentiment and emotion classification."""

from typing import List, Optional

from senselab.text.tasks.sentiment_analysis.data_structures import EmotionResult, SentimentResult
from senselab.text.tasks.sentiment_analysis.huggingface import LongText, classify_texts
from senselab.utils.compatibility import requires_compatibility
from senselab.utils.data_structures import DeviceType, HFModel

DEFAULT_SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DEFAULT_EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"


def _valence(scores: dict[str, float]) -> float:
    """``p_positive - p_negative`` from a label distribution."""
    pos = [v for k, v in scores.items() if k.lower().startswith("pos")]
    neg = [v for k, v in scores.items() if k.lower().startswith("neg")]
    if len(pos) != 1 or len(neg) != 1:
        raise ValueError(
            f"Cannot compute valence: model labels {list(scores)} lack one positive and one negative label."
        )
    return pos[0] - neg[0]


@requires_compatibility("text.tasks.sentiment_analysis.analyze_sentiment")
def analyze_sentiment(
    texts: List[str],
    model: Optional[HFModel] = None,
    device: Optional[DeviceType] = None,
    batch_size: int = 32,
    long_text: LongText = "truncate",
    window_stride: int = 64,
) -> List[SentimentResult]:
    """Score texts for sentiment.

    Args:
        texts: Texts to score. Empty or whitespace-only texts yield a result with no scores.
        model: A sequence-classification model with negative and positive labels. Defaults to
            ``cardiffnlp/twitter-roberta-base-sentiment-latest``.
        device: Device to run on. Defaults to the fastest available (CUDA, then MPS, then CPU).
        batch_size: Number of token windows per forward pass.
        long_text: ``"truncate"`` keeps the first model-length tokens; ``"window"`` splits the text into
            overlapping windows and averages their distributions, weighted by window length.
        window_stride: Token overlap between consecutive windows when ``long_text="window"``.

    Returns:
        One ``SentimentResult`` per input, in input order, with the full label distribution and
        ``valence = p_positive - p_negative``.
    """
    hf_model = model or HFModel(path_or_uri=DEFAULT_SENTIMENT_MODEL)
    base = classify_texts(texts, hf_model, device, batch_size, long_text, window_stride)
    return [SentimentResult(**r.model_dump(), valence=None if r.is_empty else _valence(r.scores)) for r in base]


@requires_compatibility("text.tasks.sentiment_analysis.classify_text_emotions")
def classify_text_emotions(
    texts: List[str],
    model: Optional[HFModel] = None,
    device: Optional[DeviceType] = None,
    batch_size: int = 32,
    long_text: LongText = "truncate",
    window_stride: int = 64,
) -> List[EmotionResult]:
    """Classify texts into emotions.

    Args:
        texts: Texts to classify. Empty or whitespace-only texts yield a result with no scores.
        model: A sequence-classification emotion model. Defaults to
            ``j-hartmann/emotion-english-distilroberta-base`` (anger, disgust, fear, joy, neutral, sadness, surprise).
        device: Device to run on. Defaults to the fastest available (CUDA, then MPS, then CPU).
        batch_size: Number of token windows per forward pass.
        long_text: ``"truncate"`` or ``"window"``; see ``analyze_sentiment``.
        window_stride: Token overlap between consecutive windows when ``long_text="window"``.

    Returns:
        One ``EmotionResult`` per input, in input order, with the full label distribution.
    """
    hf_model = model or HFModel(path_or_uri=DEFAULT_EMOTION_MODEL)
    return [
        EmotionResult(**r.model_dump())
        for r in classify_texts(texts, hf_model, device, batch_size, long_text, window_stride)
    ]
