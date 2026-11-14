"""A host backend that runs out of memory as torch does: a plain exception named `OutOfMemoryError`.

A request whose texts hold more than `BUDGET` characters in all, or one text longer than `HUGE`, raises it for the
whole request.
"""

from typing import Any

BUDGET = 300
"""The most characters a request holds before it runs out of memory."""
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
    """Raise `OutOfMemoryError` for a request over `BUDGET` characters or holding a `HUGE` text, else each identity."""
    values = [next(iter(item["inputs"].values())) for item in items]
    sizes = [_size(v) for v in values]
    if sum(sizes) > BUDGET or any(n > HUGE for n in sizes):
        raise OutOfMemoryError(f"out of memory at a batch of {len(items)}")
    return [{"outputs": {"value": str(getattr(v, "identity", v))}} for v in values]
