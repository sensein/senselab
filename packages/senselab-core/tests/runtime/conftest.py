"""Shared fixtures for the runtime tests."""

import os
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from huggingface_hub.errors import HfHubHTTPError


@pytest.fixture
def hub_error() -> Callable[..., Exception]:
    """Build a real ``huggingface_hub`` HTTP error whose status is on ``.response.status_code``."""

    def _make(status: int, cls: type = HfHubHTTPError) -> Exception:
        response = httpx.Response(status, request=httpx.Request("GET", "https://huggingface.co/org/model"))
        return cls(f"{status} from the Hub", response=response)  # type: ignore[call-arg]

    return _make


@pytest.fixture
def backends_importable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let worker processes import the fixture backends as `backends.<name>`."""
    here = str(Path(__file__).parent)
    existing = os.environ.get("PYTHONPATH")
    monkeypatch.setenv("PYTHONPATH", here if not existing else f"{here}{os.pathsep}{existing}")
