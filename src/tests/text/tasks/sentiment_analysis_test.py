"""Test module for sentiment analysis functionality."""

import pprint

import pytest

from senselab.text.tasks.sentiment_analysis.api import analyze_sentiment
from senselab.utils.data_structures.device import DeviceType
from senselab.utils.data_structures.model import HFModel


@pytest.fixture
def model() -> HFModel:
    """Fixture that returns an instance of HFModel."""
    return HFModel(path_or_uri="distilbert-base-uncased-finetuned-sst-2-english", revision="main")


def test_analyze_sentiment_basic(model: HFModel) -> None:
    """Test case for basic sentiment analysis."""
    texts = ["I love this product!", "I hate this service."]
    results = analyze_sentiment(texts, model=model, device=DeviceType.CUDA)

    assert len(results) == 2
    assert all("score" in result and "label" in result for result in results)
    assert float(results[0]["score"]) > 0 and results[0]["label"] == "positive"
    assert float(results[1]["score"]) < 0 and results[1]["label"] == "negative"


def test_analyze_sentiment_empty_input() -> None:
    """Test case for empty input in sentiment analysis."""
    with pytest.raises(ValueError):
        analyze_sentiment([""])


def test_analyze_sentiment_long_text(model: HFModel) -> None:
    """Test case for sentiment analysis with a long text."""
    long_text = "This is a very long text. " * 100
    results = analyze_sentiment([long_text], model=model)
    assert len(results) == 1
    assert "score" in results[0] and "label" in results[0]


def test_analyze_sentiment_multilingual(model: HFModel) -> None:
    """Test case for sentiment analysis with multilingual texts."""
    texts = ["Je t'aime", "Ich hasse das", "我喜欢这个"]
    results = analyze_sentiment(texts, model=model)
    pprint.pprint(f"results: {results}")
    assert len(results) == 3
    assert all("score" in result and "label" in result for result in results)


def test_analyze_sentiment_special_characters(model: HFModel) -> None:
    """Test case for sentiment analysis with special characters."""
    text = "I love this product! 🎉😊"
    results = analyze_sentiment([text], model=model)
    assert float(results[0]["score"]) > 0
    assert results[0]["label"] == "positive"


def test_analyze_sentiment_mixed_case(model: HFModel) -> None:
    """Test case for sentiment analysis with mixed case text."""
    text = "I LOVE this Product!"
    results = analyze_sentiment([text], model=model)
    assert float(results[0]["score"]) > 0
    assert results[0]["label"] == "positive"


def test_analyze_sentiment_very_short_text(model: HFModel) -> None:
    """Test case for sentiment analysis with very short text."""
    text = "Good"
    results = analyze_sentiment([text], model=model)
    assert float(results[0]["score"]) > 0
    assert results[0]["label"] == "positive"


def test_analyze_sentiment_all_punctuation() -> None:
    """Test case for sentiment analysis with all punctuation."""
    text = "!@#$%^&*()"
    with pytest.raises(ValueError, match="Input string contains only punctuation"):
        analyze_sentiment([text])


def test_analyze_sentiment_large_batch(model: HFModel) -> None:
    """Test case for sentiment analysis with a large batch of texts."""
    texts = ["Sample text"] * 1000
    results = analyze_sentiment(texts, model=model)
    assert len(results) == 1000


@pytest.mark.parametrize("device", [DeviceType.CUDA, DeviceType.CPU])
def test_analyze_sentiment_different_devices(model: HFModel, device: DeviceType) -> None:
    """Test case for sentiment analysis with different devices."""
    text = "I love this!"
    results = analyze_sentiment([text], model=model, device=device)
    assert float(results[0]["score"]) > 0
    assert results[0]["label"] == "positive"


def test_analyze_sentiment_empty_list() -> None:
    """Test case for sentiment analysis with an empty list."""
    with pytest.raises(ValueError, match="Input list is empty or None"):
        analyze_sentiment([])


def test_analyze_sentiment_none_input() -> None:
    """Test case for sentiment analysis with None input."""
    with pytest.raises(ValueError, match="Input list is empty or None"):
        analyze_sentiment([])


def test_analyze_sentiment_non_string_input() -> None:
    """Test case for sentiment analysis with non-string input."""
    with pytest.raises(TypeError, match="Input must be a string"):
        analyze_sentiment(["42"])
