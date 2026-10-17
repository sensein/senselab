"""Tests for text sentiment and emotion classification."""

import pytest

from senselab.text.tasks.sentiment_analysis import (
    EmotionResult,
    SentimentResult,
    analyze_sentiment,
    classify_text_emotions,
)
from senselab.text.tasks.sentiment_analysis.huggingface import TextClassifierFactory
from senselab.utils.data_structures import DeviceType, HFModel

SENTIMENT_LABELS = ["negative", "neutral", "positive"]
EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
POSITIVE = "I am so happy, this is wonderful news and I love it."
NEGATIVE = "This is terrible, I hate it and I am deeply disappointed."


def test_empty_input_list() -> None:
    """An empty list yields an empty list without loading a model."""
    assert analyze_sentiment([]) == []


@pytest.mark.large_model
def test_sentiment_distribution_and_valence() -> None:
    """Scores are ordered, sum to one, and valence has the right sign."""
    pos, neg = analyze_sentiment([POSITIVE, NEGATIVE], device=DeviceType.CPU)
    for r in (pos, neg):
        assert isinstance(r, SentimentResult)
        assert list(r.scores) == SENTIMENT_LABELS
        assert sum(r.scores.values()) == pytest.approx(1.0, abs=1e-4)
        assert r.valence == pytest.approx(r.scores["positive"] - r.scores["negative"])
    assert pos.label == "positive" and pos.valence is not None and pos.valence > 0.5
    assert neg.label == "negative" and neg.valence is not None and neg.valence < -0.5


@pytest.mark.large_model
def test_empty_and_whitespace_texts_keep_position() -> None:
    """Empty texts get a scoreless result in the right slot."""
    out = analyze_sentiment(["", POSITIVE, "   "], device=DeviceType.CPU)
    assert [r.is_empty for r in out] == [True, False, True]
    assert out[0].valence is None and out[0].label is None and out[0].n_windows == 0


@pytest.mark.large_model
def test_batching_equivalence() -> None:
    """Batch size does not change the scores."""
    texts = [POSITIVE, NEGATIVE, "The meeting is at noon.", "Okay."]
    a = analyze_sentiment(texts, device=DeviceType.CPU, batch_size=1)
    b = analyze_sentiment(texts, device=DeviceType.CPU, batch_size=4)
    for x, y in zip(a, b):
        for k in SENTIMENT_LABELS:
            assert x.scores[k] == pytest.approx(y.scores[k], abs=1e-4)


@pytest.mark.large_model
def test_long_text_truncate_and_window() -> None:
    """Truncation is reported; windowing averages several windows."""
    long = " ".join([POSITIVE] * 120)
    (cut,) = analyze_sentiment([long], device=DeviceType.CPU)
    (win,) = analyze_sentiment([long], device=DeviceType.CPU, long_text="window")
    assert cut.truncated and cut.n_windows == 1
    assert not win.truncated and win.n_windows > 1
    assert win.label == "positive"
    assert sum(win.scores.values()) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.large_model
def test_model_cache_reused() -> None:
    """Repeated calls do not add cache entries."""
    analyze_sentiment([POSITIVE], device=DeviceType.CPU)
    n = len(TextClassifierFactory._models)
    analyze_sentiment([NEGATIVE], device=DeviceType.CPU)
    assert len(TextClassifierFactory._models) == n


@pytest.mark.large_model
def test_emotions() -> None:
    """Emotion distribution covers the seven labels and picks joy / anger."""
    joy, anger = classify_text_emotions(
        ["I just won the lottery, I am so happy!", "How dare you, I am furious with you!"], device=DeviceType.CPU
    )
    for r in (joy, anger):
        assert isinstance(r, EmotionResult)
        assert list(r.scores) == EMOTION_LABELS
        assert sum(r.scores.values()) == pytest.approx(1.0, abs=1e-4)
    assert joy.label == "joy"
    assert anger.label == "anger"


def test_invalid_arguments() -> None:
    """Bad batch_size or long_text values are rejected."""
    with pytest.raises(ValueError):
        analyze_sentiment(["x"], batch_size=0, model=HFModel.model_construct(path_or_uri="x", revision="main"))
    with pytest.raises(ValueError):
        analyze_sentiment(["x"], long_text="nope", model=HFModel.model_construct(path_or_uri="x", revision="main"))  # type: ignore[arg-type]
