"""A host backend whose requests of more than two items run out of memory, as a model framework raises it.

In a smaller request, an input `oom` runs out of memory alone and `fail` is a content fault.
"""

from typing import Any

from senselab_worker.fake import OutOfMemoryError
from senselab_worker.faults import ContentFault

MOST = 2
"""The most items a request holds before it runs out of memory."""


def load(init: dict[str, Any]) -> object:
    """Return `init` as the state."""
    return init


def handle(state: object, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Raise `OutOfMemoryError` for a request of more than `MOST` items, else each item's own fault or identity."""
    if len(items) > MOST:
        raise OutOfMemoryError(f"out of memory at a batch of {len(items)}")
    entries: list[dict[str, Any]] = []
    for item in items:
        value = next(iter(item["inputs"].values()))
        if value == "oom":
            entries.append({"error": OutOfMemoryError("out of memory")})
        elif value == "fail":
            entries.append({"error": ContentFault("asked to fail")})
        else:
            entries.append({"outputs": {"value": str(getattr(value, "identity", value))}})
    return entries
