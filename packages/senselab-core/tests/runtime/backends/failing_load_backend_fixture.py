"""A host backend whose load always fails, counting each attempt, as a broken env does."""

from typing import Any

LOADS: list[dict[str, Any]] = []
"""The init of each load attempted, in order."""


def load(init: dict[str, Any]) -> object:
    """Record the attempt and fail."""
    LOADS.append(init)
    raise RuntimeError("this env cannot load its backend")


def handle(state: object, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Never reached: no load succeeds."""
    raise AssertionError("handle without a load")
