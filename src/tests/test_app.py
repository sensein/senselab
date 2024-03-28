"""Tests for the app hello_world function."""
from _pytest.capture import CaptureFixture

from YOUR_PACKAGE_NAME.app import hello_world


def test_hello_world(capsys: CaptureFixture) -> None:  
    """A test function for hello_world.

    This function uses the capsys fixture to capture stdout and stderr.
    """
    hello_world()
    captured = capsys.readouterr()  # Captures the output of hello_world()
    assert "Hello World!\n" == captured.out