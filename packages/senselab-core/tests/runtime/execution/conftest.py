"""Shared fixtures of the job engine tests."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def fixture_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let host workers import the fixture backends as `backends.<name>`."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]))
