"""A host backend that advances a fake clock by the seconds each item's input names, so its requests take no time."""

from typing import Any

NOW: list[float] = [0.0]
"""The fake clock's reading, which only a request advances."""


def now() -> float:
    """Return the fake clock's reading."""
    return NOW[0]


def load(init: dict[str, Any]) -> object:
    """Return `init` as the state."""
    return init


def handle(state: object, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Advance the fake clock by the seconds the items name, summed, and return each item's input as its value."""
    values = [next(iter(item["inputs"].values())) for item in items]
    NOW[0] += sum(float(v) for v in values)
    return [{"outputs": {"value": str(v)}} for v in values]
