"""Shared fixtures for the runtime tests."""

import importlib
import os
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from huggingface_hub import errors
from huggingface_hub.errors import HfHubHTTPError
from lease_clock_fixture import ManualClock


def _hub_client() -> ModuleType:
    """Return the HTTP client module `huggingface_hub` builds its errors on (`httpx` in 1.x, `httpx2` in 2.x)."""
    return importlib.import_module(vars(errors)["HTTPError"].__module__.partition(".")[0])


@pytest.fixture
def hub_client() -> ModuleType:
    """The HTTP client module `huggingface_hub` builds its errors on (`httpx` in 1.x, `httpx2` in 2.x)."""
    return _hub_client()


@pytest.fixture
def hub_error() -> Callable[..., Exception]:
    """Build a real ``huggingface_hub`` HTTP error over a response of the Hub's own HTTP client."""

    def _make(status: int, cls: type = HfHubHTTPError) -> Exception:
        client = _hub_client()
        response = client.Response(status, request=client.Request("GET", "https://huggingface.co/org/model"))
        error: Exception = cls(f"{status} from the Hub", response=response)
        return error

    return _make


@pytest.fixture
def backends_importable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let worker processes import the fixture backends as `backends.<name>`."""
    here = str(Path(__file__).parent)
    existing = os.environ.get("PYTHONPATH")
    monkeypatch.setenv("PYTHONPATH", here if not existing else f"{here}{os.pathsep}{existing}")


@pytest.fixture
def lease_clock(slack_tests: dict[str, Any], tier_bound: float) -> Callable[..., ManualClock]:
    """Build a `ManualClock` (or a subclass `kind`): polls from the test slack file, hang bound from the tier."""

    def make(kind: type[ManualClock] = ManualClock, **kwargs: Any) -> ManualClock:  # noqa: ANN401 - `kind`'s own arguments
        return kind(poll_s=slack_tests["lease_clock"]["poll_s"], hang_s=tier_bound, **kwargs)

    return make
