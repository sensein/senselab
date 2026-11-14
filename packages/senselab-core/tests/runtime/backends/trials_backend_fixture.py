"""A backend whose output for each item is one fixed `Trials` value."""

from typing import Any

import numpy
from senselab_core.kernel.results import Trials


def load(init: dict[str, Any]) -> object:
    """Return `init` as the state."""
    return init


def handle(state: object, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the same two trials for every item."""
    rows = numpy.array([[1.0, 0.0], [0.0, 1.0]], dtype=numpy.float32)
    trials = Trials(first=rows, second=rows[::-1], same=[True, False], space="m")
    return [{"outputs": {"trials": trials}} for _ in items]
