import pytest
from deepevall.api import evaluate_chat
from deepevall.script_line import ScriptLine

@pytest.fixture
def script_lines():
    return [
        ScriptLine(text="Mazen speaks Arabic", speaker="agent"),
        ScriptLine(text="Mazen reads Arabic", speaker="user"),
        ScriptLine(text="I live in USA", speaker="agent"),
        ScriptLine(text="I live in KSA", speaker="user")
    ]

def test_evaluate_chat(script_lines: list[ScriptLine]):
    result = evaluate_chat(script_lines)
    print(result)
    assert result is not None
    assert "overall_score" in result
    assert "metrics" in result


if __name__ == "__main__":
    pytest.main(["-s"])
