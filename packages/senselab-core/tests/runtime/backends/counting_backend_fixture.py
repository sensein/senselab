"""The fake backend, recording each `load` in `LOADS`."""

from typing import Any

from senselab_worker import fake

LOADS: list[dict[str, Any]] = []
"""The `init` of every load, in order."""


def load(init: dict[str, Any]) -> object:
    """Record `init` and return the fake's state for it."""
    LOADS.append(init)
    return fake.load(init)


handle = fake.handle
