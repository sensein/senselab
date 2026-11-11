"""A host backend whose load fails with the exception `RAISE` holds, after `SUCCEED` loads, counting each attempt."""

from typing import Any

LOADS: list[dict[str, Any]] = []
"""The init of each load attempted, in order."""

RAISE: list[BaseException] = []
"""The exception each load raises."""

SUCCEED = [0]
"""How many of the first loads succeed before every later one raises."""


def load(init: dict[str, Any]) -> object:
    """Record the attempt and fail with the exception in `RAISE`."""
    LOADS.append(init)
    if len(LOADS) <= SUCCEED[0]:
        return init
    raise RAISE[0]


def handle(state: object, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Answer each item with its own identity."""
    return [{"outputs": {"value": str(item["id"])}} for item in items]
