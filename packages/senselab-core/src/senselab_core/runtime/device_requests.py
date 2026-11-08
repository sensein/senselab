"""The device memory each request a thread ran allocated: `(peak, allocated after, items)`, collected per block.

A worker client records what its worker's `done` reports, and an in-process host worker what it measured
(`torch_meter`); `collected()` gathers the records of the requests the calling thread runs inside it.
"""

import threading
from collections.abc import Iterator
from contextlib import contextmanager

_LOCAL = threading.local()


def record(peak: int, allocated: int, items: int) -> None:
    """Record one request's device peak and allocation after it, over `items` items, for the thread's collector."""
    found: list[tuple[int, int, int]] | None = getattr(_LOCAL, "found", None)
    if found is not None:
        found.append((peak, allocated, items))


@contextmanager
def collected() -> Iterator[list[tuple[int, int, int]]]:
    """Yield the list the requests this thread runs inside the block are recorded in."""
    previous = getattr(_LOCAL, "found", None)
    found: list[tuple[int, int, int]] = []
    _LOCAL.found = found
    try:
        yield found
    finally:
        _LOCAL.found = previous
