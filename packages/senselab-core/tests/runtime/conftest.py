"""Shared fixtures for the runtime tests."""

import importlib
import os
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest
from huggingface_hub import errors
from huggingface_hub.errors import HfHubHTTPError


def _hub_client() -> ModuleType:
    """Return the HTTP client module `huggingface_hub` builds its errors on (`httpx` in 1.x, `httpx2` in 2.x)."""
    return importlib.import_module(errors.HTTPError.__module__.partition(".")[0])


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
        return cls(f"{status} from the Hub", response=response)  # type: ignore[call-arg]

    return _make


@pytest.fixture
def backends_importable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let worker processes import the fixture backends as `backends.<name>`."""
    here = str(Path(__file__).parent)
    existing = os.environ.get("PYTHONPATH")
    monkeypatch.setenv("PYTHONPATH", here if not existing else f"{here}{os.pathsep}{existing}")
