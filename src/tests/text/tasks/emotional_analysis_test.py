"""Unit tests for the emotional analysis module."""

from typing import Any, Dict, List

import pytest

from senselab.text.tasks.emotion_analysis.api import analyze_emotion
from senselab.text.tasks.emotion_analysis.constants import Emotion
from senselab.utils.data_structures.device import DeviceType
from senselab.utils.data_structures.model import HFModel


@pytest.fixture
def model() -> HFModel:
    """Fixture that returns an instance of HFModel."""
    return HFModel(path_or_uri="j-hartmann/emotion-english-distilroberta-base", revision="main")


def test_analyze_emotion_basic(model: HFModel) -> None:
    """Test case for basic emotional analysis."""
    texts = ["I'm so happy today!", "I'm feeling very sad and lonely."]
    results: List[Dict[str, Any]] = analyze_emotion(texts, model=model, device=DeviceType.CUDA)

    assert len(results) == 2
    assert all("scores" in result and "dominant_emotion" in result for result in results)
    assert results[0]["dominant_emotion"] == Emotion.JOY.value
    assert results[1]["dominant_emotion"] == Emotion.SADNESS.value


def test_analyze_emotion_empty_input() -> None:
    """Test case for empty input in emotional analysis."""
    with pytest.raises(ValueError):
        analyze_emotion([""])


def test_analyze_emotion_long_text(model: HFModel) -> None:
    """Test case for emotional analysis with a long text."""
    long_text = "I'm feeling really excited about this new project. " * 100
    results: List[Dict[str, Any]] = analyze_emotion([long_text], model=model)
    assert len(results) == 1
    assert "scores" in results[0] and "dominant_emotion" in results[0]


def test_analyze_emotion_mixed_emotions(model: HFModel) -> None:
    """Test case for emotional analysis with mixed emotions."""
    text = "I'm happy about my promotion but scared about the new responsibilities."
    results: List[Dict[str, Any]] = analyze_emotion([text], model=model)
    scores: Dict[str, float] = results[0]["scores"]
    assert scores[Emotion.JOY.value] > 0 and scores[Emotion.FEAR.value] > 0
    assert len(scores) == 7


def test_analyze_emotion_neutral_text(model: HFModel) -> None:
    """Test case for emotional analysis with neutral text."""
    text = "The sky is blue and the grass is green."
    results: List[Dict[str, Any]] = analyze_emotion([text], model=model)
    assert results[0]["dominant_emotion"] == Emotion.NEUTRAL.value


def test_analyze_emotion_all_punctuation() -> None:
    """Test case for emotional analysis with all punctuation."""
    text = "!@#$%^&*()"
    with pytest.raises(ValueError, match="Input text cannot contain only punctuation."):
        analyze_emotion([text])


@pytest.mark.parametrize("device", [DeviceType.CUDA, DeviceType.CPU])
def test_analyze_emotion_different_devices(model: HFModel, device: DeviceType) -> None:
    """Test case for emotional analysis with different devices."""
    text = "I'm so excited!"
    results: List[Dict[str, Any]] = analyze_emotion([text], model=model, device=device)
    assert results[0]["dominant_emotion"] == Emotion.JOY.value


def test_analyze_emotion_empty_list() -> None:
    """Test case for emotional analysis with an empty list."""
    with pytest.raises(ValueError, match="Input list is empty or None"):
        analyze_emotion([])


def test_analyze_emotion_scores_sum_to_one(model: HFModel) -> None:
    """Test case to ensure emotion scores sum to approximately 1."""
    text = "I'm feeling a mix of emotions today."
    results: List[Dict[str, Any]] = analyze_emotion([text], model=model)
    scores: Dict[str, float] = results[0]["scores"]
    total_score = sum(scores.values())
    tolerance = 2e-4  # Allowing for a deviation of up to 0.0002
    assert abs(total_score - 1.0) <= tolerance, f"Total score {total_score} deviates from 1.0 by more than {tolerance}"


def test_analyze_emotion_consistent_labels(model: HFModel) -> None:
    """Test case to ensure consistent emotion labels across results."""
    texts = ["I'm happy", "I'm sad", "I'm angry", "I'm scared", "I'm disgusted", "I'm surprised", "I'm neutral"]
    results: List[Dict[str, Any]] = analyze_emotion(texts, model=model)
    all_labels = set()
    for result in results:
        all_labels.update(result["scores"].keys())
    assert len(all_labels) == 7
    assert all_labels == set(emotion.value for emotion in Emotion)
