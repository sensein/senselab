"""A backend whose output for each item is its `audio` input's identity."""

from typing import Any


def load(init: dict[str, Any]) -> object:
    """Return `init` as the state."""
    return init


def handle(state: object, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Return each item's `audio` identity as its `value`."""
    return [{"outputs": {"value": item["inputs"]["audio"].identity}} for item in items]
