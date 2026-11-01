"""A backend whose outputs for each item are that item's decoded inputs."""

from typing import Any


def load(init: dict[str, Any]) -> object:
    """Return `init` as the state."""
    return init


def handle(state: object, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Return each item's inputs as its outputs."""
    return [{"outputs": item["inputs"]} for item in items]
