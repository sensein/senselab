"""Shared fixtures of the job engine tests."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from lease_clock_fixture import ManualClock

CLOCK_START = 1000.0


@pytest.fixture(autouse=True)
def fixture_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let host workers import the fixture backends as `backends.<name>`."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]))


@pytest.fixture
def clock(lease_clock: Callable[..., ManualClock]) -> ManualClock:
    """A clock that starts at `CLOCK_START` and is moved by hand."""
    return lease_clock(start=CLOCK_START)


@pytest.fixture
def fake_poll_s(slack_tests: dict[str, Any]) -> float:
    """The real seconds a fake worker thread waits between looks at its queue."""
    return float(slack_tests["fake_worker"]["poll_s"])
