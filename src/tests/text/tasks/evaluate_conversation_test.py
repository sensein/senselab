"""Unit tests for evaluating chat functionality."""

import pytest

from senselab.text.tasks.evaluate_conversation.deep_eval import evaluate_transcript_output
from senselab.utils.data_structures.transcript_output import TranscriptOutput


@pytest.fixture
def transcript_output() -> TranscriptOutput:
    """Fixture for providing a sample TranscriptOutput.

    Returns:
        TranscriptOutput: A sample transcript output.
    """
    return TranscriptOutput(
        temp=0.5,
        model="test_model",
        prompt="test_prompt.txt",
        transcript="test.json",
        data=[
            {"speaker": "Tutor", "text": "Mazen speaks Arabic"},
            {"speaker": "Child", "text": "Mazen speaks Arabic"},
            {"speaker": "Tutor", "text": "I live in USA"},
            {"speaker": "AI", "text": "I live in KSA"},
            {"speaker": "Child", "text": "I like playing soccer"},
            {"speaker": "Tutor", "text": "Soccer is a great sport"},
            {"speaker": "AI", "text": "I prefer basketball"},
            {"speaker": "Child", "text": "I have a pet dog"},
            {"speaker": "Tutor", "text": "Dogs are wonderful pets"},
            {"speaker": "AI", "text": "I have a virtual pet"},
            {"speaker": "Child", "text": "I enjoy reading books"},
            {"speaker": "Tutor", "text": "Reading is very beneficial"},
            {"speaker": "AI", "text": "I read digital books"},
        ],
    )


def test_evaluate_transcript_output(transcript_output: TranscriptOutput) -> None:
    """Test the evaluate_transcript_output function.

    Args:
        transcript_output (TranscriptOutput): A sample transcript output.

    Asserts:
        The evaluation result is not None and contains overall score and metrics.
    """
    metrics = ["TextStatistics"]
    result = evaluate_transcript_output(transcript_output, metrics)

    assert result is not None, "The evaluation result should not be None."
    assert result.data is not None, "The evaluation result should contain 'data'."
    assert isinstance(result.data, list), "'data' should be a list."
    for response in result.data[1:]:
        if response["speaker"] == "AI" or response["speaker"] == "Tutor":
            assert "metrics" in response, "Each response should contain 'metrics'."
            assert isinstance(response["metrics"], dict), "Metrics should be a dictionary."
            assert "word_count" in response["metrics"], "Metrics should contain 'word_count'."
            assert isinstance(response["metrics"]["word_count"], int), "'word_count' should be an integer."
            assert "sentence_count" in response["metrics"], "Metrics should contain 'sentence_count'."
            assert isinstance(response["metrics"]["sentence_count"], int), "'sentence_count' should be an integer."
