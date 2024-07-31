"""Unit tests for evaluating chat functionality."""

from typing import List

import pytest

from senselab.text.tasks.evaluate_conversation.api import evaluate_chat
from senselab.utils.data_structures.script_line import ScriptLine


@pytest.fixture
def script_lines() -> List[ScriptLine]:
    """Fixture for providing sample script lines.

    Returns:
        List[ScriptLine]: A list of sample script lines.
    """
    return [
        ScriptLine(text="Mazen speaks Arabic", speaker="agent"),
        ScriptLine(text="Mazen speaks Arabic", speaker="user"),
        ScriptLine(text="I live in USA", speaker="agent"),
        ScriptLine(text="I live in KSA", speaker="user"),
    ]


def test_evaluate_chat(script_lines: List[ScriptLine]) -> None:
    """Test the evaluate_chat function.

    Args:
        script_lines (List[ScriptLine]): A list of script lines to evaluate.

    Asserts:
        The evaluation result is not None and contains overall score and metrics.
    """
    result = evaluate_chat(script_lines)
    assert result is not None
    assert "metrics" in result
