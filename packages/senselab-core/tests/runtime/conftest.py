"""Shared helpers for the runtime tests."""

import httpx
from huggingface_hub.errors import HfHubHTTPError


def hub_error(status: int, cls: type = HfHubHTTPError) -> Exception:
    """Build a real ``huggingface_hub`` HTTP error whose status is on ``.response.status_code``."""
    response = httpx.Response(status, request=httpx.Request("GET", "https://huggingface.co/org/model"))
    return cls(f"{status} from the Hub", response=response)
