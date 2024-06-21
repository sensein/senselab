"""Tests for the app hello_world function."""

from senselab.app import hello_world


def test_hello_world() -> None:
    """A test function for hello_world.

    This function uses the capsys fixture to capture stdout and stderr.
    """
    result = hello_world()
    assert "Hello World!" == result["output"]
