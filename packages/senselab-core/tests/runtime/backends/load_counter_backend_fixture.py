"""A host backend whose state is the ordinal of its load in this process, returned by every item."""

import itertools
from typing import Any

_LOADS = itertools.count()


def load(init: dict[str, Any]) -> object:
    """Return this load's ordinal."""
    return next(_LOADS)


def handle(state: object, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the state as every item's output `state`."""
    return [{"outputs": {"state": state}} for _ in items]
