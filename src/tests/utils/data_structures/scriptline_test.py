"""Tests functionality for the ScriptLine class."""

from senselab.utils.data_structures.script_line import ScriptLine


def test_scriptline_from_dict() -> None:
    """Test creating ScriptLine from dict."""
    data = {
        "text": "Hello world",
        "chunks": [{"text": "Hello", "timestamps": [0.0, 1.0]}, {"text": "world", "timestamps": [1.0, 2.0]}],
    }
    scriptline = ScriptLine.from_dict(data)

    # Ensure chunks is not None before using it
    assert scriptline.chunks is not None
    assert len(scriptline.chunks) == 2
    assert scriptline.chunks[0].text == "Hello"
    assert scriptline.chunks[0].get_timestamps()[0] == 0.0
    assert scriptline.chunks[0].get_timestamps()[1] == 1.0

    assert scriptline.chunks[1].text == "world"
    assert scriptline.chunks[1].get_timestamps()[0] == 1.0
    assert scriptline.chunks[1].get_timestamps()[1] == 2.0


def test_null_scriptline() -> None:
    """Tests creation of a ScriptLine with no text or speaker."""
    # Create a ScriptLine with no text or speaker
    null_scriptline = ScriptLine()

    # Assert that the ScriptLine fields are None by default
    assert null_scriptline.text is None
    assert null_scriptline.speaker is None
    assert null_scriptline.start is None
    assert null_scriptline.end is None
    assert null_scriptline.chunks is None
