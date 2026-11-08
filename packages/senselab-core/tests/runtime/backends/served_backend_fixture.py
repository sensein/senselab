"""A server backend's worker: its state is its `init`, which carries the server it answers over."""

from typing import Any


def load(init: dict[str, Any]) -> object:
    """Return `init` as the state."""
    return init


def handle(state: object, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Answer each item with a constant."""
    return [{"outputs": {"value": "served"}} for _ in items]
