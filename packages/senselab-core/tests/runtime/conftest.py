"""Shared fixtures for the runtime tests."""

from collections.abc import Callable

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
