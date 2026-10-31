"""A backend that writes to stdout and fd 1 at import time and during load."""

import os
from typing import Any

print("noise at import")
os.write(1, b"noise on fd 1 at import\n")


def load(init: dict[str, Any]) -> object:
    """Print during load and return no state."""
    print("noise from load")
    return None


def handle(state: object, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Return empty outputs for every item."""
    return [{"outputs": {}} for _ in items]
