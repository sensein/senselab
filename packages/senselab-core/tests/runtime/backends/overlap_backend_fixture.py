"""A host backend whose handle waits at a two-party barrier and records whether a second handle met it there."""

import threading
from typing import Any

from backends import trials_backend_fixture

BARRIER = threading.Barrier(2)
"""Passed only when two handles run at once."""
MET: list[bool] = []
"""One entry per handle: whether another handle reached `BARRIER` within `WAIT_S`."""
WAIT_S = 2.0

load = trials_backend_fixture.load


def handle(state: object, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Wait at `BARRIER`, record the outcome in `MET`, and return the trials fixture's outputs."""
    try:
        BARRIER.wait(timeout=WAIT_S)
        MET.append(True)
    except threading.BrokenBarrierError:
        MET.append(False)
    return trials_backend_fixture.handle(state, items, params)
