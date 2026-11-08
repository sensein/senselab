"""A host backend whose state records its load ordinal in `RELEASED` once the state is freed."""

import itertools
import weakref
from typing import Any

RELEASED: list[int] = []
"""The ordinals of the states freed, in order."""
_LOADS = itertools.count()


class _State:
    def __init__(self, ordinal: int) -> None:
        self.ordinal = ordinal


def load(init: dict[str, Any]) -> object:
    """Return a state that records its ordinal in `RELEASED` when freed."""
    state = _State(next(_LOADS))
    weakref.finalize(state, RELEASED.append, state.ordinal)
    return state


def handle(state: _State, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the state's ordinal as every item's output `state`."""
    return [{"outputs": {"state": state.ordinal}} for _ in items]
