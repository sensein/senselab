"""A host backend that records the size of each request it gets and runs out of memory above `FITS` items.

A text longer than `HUGE` characters runs out alone; a text starting with `b` fails its item with a content fault.
"""

from typing import Any

from senselab_worker.faults import ContentFault

FITS = 5
"""The most items a request holds before it runs out of memory."""
HUGE = 70
"""The length of text no request holds, alone or not."""
BATCHES: list[int] = []
"""The item count of each request, in the order they arrived."""


class OutOfMemoryError(RuntimeError):
    """Out of memory, as a model framework raises it: no senselab fault base."""


def _text(item: dict[str, Any]) -> str:
    value = next(iter(item["inputs"].values()))
    return value if isinstance(value, str) else getattr(value, "content", "")


def load(init: dict[str, Any]) -> object:
    """Return `init` as the state."""
    return init


def handle(state: object, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Record the request; raise `OutOfMemoryError` above `FITS` items or for a `HUGE` text, else answer each item."""
    BATCHES.append(len(items))
    texts = [_text(i) for i in items]
    if len(items) > FITS or any(len(t) > HUGE for t in texts):
        raise OutOfMemoryError(f"out of memory at a batch of {len(items)}")
    return [{"error": ContentFault("bad item")} if t.startswith("b") else {"outputs": {"value": t}} for t in texts]
