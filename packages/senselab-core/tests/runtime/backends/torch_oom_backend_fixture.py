"""A host backend that runs out of memory as torch does: a plain exception named `OutOfMemoryError`.

A request of more than `MOST` items, or one holding a text longer than `HUGE`, raises it for the whole request.
"""

from typing import Any

MOST = 5
"""The most items a request holds before it runs out of memory."""
HUGE = 70
"""The length of text no request holds, alone or not."""


class OutOfMemoryError(RuntimeError):
    """Out of memory, as a model framework raises it: no senselab fault base."""


def _size(value: object) -> int:
    return len(value) if isinstance(value, str) else len(getattr(value, "content", ""))


def load(init: dict[str, Any]) -> object:
    """Return `init` as the state."""
    return init


def handle(state: object, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Raise `OutOfMemoryError` for a request over `MOST` items or holding a `HUGE` text, else each item's identity."""
    values = [next(iter(item["inputs"].values())) for item in items]
    if len(items) > MOST or any(_size(v) > HUGE for v in values):
        raise OutOfMemoryError(f"out of memory at a batch of {len(items)}")
    return [{"outputs": {"value": str(getattr(v, "identity", v))}} for v in values]
