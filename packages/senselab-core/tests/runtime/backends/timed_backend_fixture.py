"""A host backend that sleeps the seconds each item's input names, and records the items of each request."""

import time
from typing import Any

SIZES: list[int] = []
"""The number of items of each request, in order."""


def load(init: dict[str, Any]) -> object:
    """Return `init` as the state."""
    return init


def handle(state: object, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Sleep the seconds the items name, summed, and return each item's input as its value."""
    SIZES.append(len(items))
    values = [next(iter(item["inputs"].values())) for item in items]
    time.sleep(sum(float(v) for v in values))
    return [{"outputs": {"value": str(v)}} for v in values]
