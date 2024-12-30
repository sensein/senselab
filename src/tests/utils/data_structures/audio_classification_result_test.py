"""Tests the AudioClassificationResult data structure."""

import pytest
from pydantic import ValidationError

from senselab.utils.data_structures import AudioClassificationResult


def test_audio_classification_result_sorts() -> None:
    """Test AudioClassificationResult creation and sorting."""
    labels = ["a", "b", "c"]
    scores = [0.2, 0.3, 0.5]
    classification_result = AudioClassificationResult(labels=labels, scores=scores)

    assert classification_result.get_labels() == labels[::-1]
    assert classification_result.get_scores() == scores[::-1]


def test_from_hf_classification_pipeline() -> None:
    """Test AudioClassificationResult from a HuggingFace pipeline output."""
    labels = ["a", "b", "c"]
    scores = [0.2, 0.3, 0.5]

    pipeline_output_example = []
    for i in range(len(labels)):
        pipeline_output_example.append({"label": labels[i], "score": scores[i]})

    classification_result = AudioClassificationResult.from_hf_classification_pipeline(pipeline_output_example)

    assert classification_result.get_labels() == labels[::-1]
    assert classification_result.get_scores() == scores[::-1]


def test_improper_classification_inputs() -> None:
    """Tests AudioClassificationResult when invalid data is inputted, like misaligned lists."""
    labels = ["a", "b", "c"]
    scores = [0.2, 0.3, 0.5]
    with pytest.raises(ValidationError):
        AudioClassificationResult(labels=labels[0:2], scores=scores)

    with pytest.raises(ValidationError):
        AudioClassificationResult(labels=labels, scores=scores[0:2])

    with pytest.raises(ValueError):
        AudioClassificationResult.from_hf_classification_pipeline(
            [{"labels": labels[i], "scores": scores[i]} for i in range(len(labels))]
        )
