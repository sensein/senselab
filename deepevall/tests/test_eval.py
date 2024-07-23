import pytest
from deepevall.api import evaluate_chat
from deepevall.script_line import ScriptLine
from typing import List

@pytest.fixture
def script_lines() -> List[ScriptLine]:
    """Fixture to provide script lines for testing."""
    return [
        ScriptLine(text="Mazen speaks Arabic", speaker="agent"),
        ScriptLine(text="Mazen reads Arabic", speaker="user"),
        ScriptLine(text="I live in USA", speaker="agent"),
        ScriptLine(text="I live in KSA", speaker="user")
    ]

def test_evaluate_chat(script_lines: List[ScriptLine]) -> None:
    """Test the evaluate_chat function."""
    result = evaluate_chat(script_lines)
    print(result)
    assert result is not None
    assert "overall_score" in result
    assert "metrics" in result

if __name__ == "__main__":
    pytest.main(["-s"])
