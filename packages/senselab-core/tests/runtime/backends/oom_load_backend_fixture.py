"""A host backend whose load always fails with the exception `RAISE` holds, counting each attempt."""

from typing import Any

LOADS: list[dict[str, Any]] = []
"""The init of each load attempted, in order."""

RAISE: list[BaseException] = []
"""The exception each load raises."""


def load(init: dict[str, Any]) -> object:
    """Record the attempt and fail with the exception in `RAISE`."""
    LOADS.append(init)
    raise RAISE[0]


def handle(state: object, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Never reached: no load succeeds."""
    raise AssertionError("handle without a load")
