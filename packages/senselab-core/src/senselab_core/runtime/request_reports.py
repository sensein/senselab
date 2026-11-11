"""What each worker request reported, collected for the thread that ran it: `RequestReport`, one per request.

A worker client records what its worker's `done` reports, and an in-process host worker what it measured; `request`
opens the report of the request a thread is about to send, tagged with the jobs it carries, and `collected()` gathers
the reports of the requests the calling thread runs inside it, timed by the clock it is given.
"""

import threading
from collections.abc import Hashable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

from senselab_worker.faults import FaultClass

from senselab_core.runtime.clock import Clock, SystemClock

Failure = Literal["bisected", "failed"]
"""How a request failed: `bisected` where its jobs were sent again in halves, `failed` where it was not split."""

FaultKind = Literal["content", "resource", "infrastructure"]
"""The class of a failed request's fault."""

_LOCAL = threading.local()


@dataclass
class RequestReport:
    """One request: its jobs' tags, its wall seconds, what its model state reported and its device figures.

    Attributes:
        tags: The tags of the jobs it carried; empty when the sender tagged none.
        items: The items it held.
        wall_s: Seconds from the request being sent to its replies being read; None outside `request`.
        since_load: The requests its model state answered before it; None where the worker reports none.
        load_s: The seconds the state's load took, on the request that used the load; else None.
        device: `(peak, allocated after)` device bytes; None where the worker measured none.
        failure: Set by the bisection engine on a request it split or retried; None for a request that settled its jobs.
        fault: The class of the fault `failure` names.
    """

    tags: tuple[Hashable, ...] = ()
    items: int = 0
    wall_s: float | None = None
    since_load: int | None = None
    load_s: float | None = None
    device: tuple[int, int] | None = None
    failure: Failure | None = None
    fault: FaultKind | None = None


def _open() -> list[RequestReport]:
    if not hasattr(_LOCAL, "opened"):
        _LOCAL.opened = []
    opened: list[RequestReport] = _LOCAL.opened
    return opened


def _collector() -> list[RequestReport] | None:
    found: list[RequestReport] | None = getattr(_LOCAL, "found", None)
    return found


def record(
    items: int, *, since_load: int | None = None, load_s: float | None = None, device: tuple[int, int] | None = None
) -> None:
    """Record what the request this thread is sending reported, over `items` items.

    Without an open `request` it is a request of its own, tagged with nothing; without a collector it is dropped.
    """
    opened = _open()
    if opened:
        report = opened[-1]
        report.items, report.since_load, report.load_s, report.device = items, since_load, load_s, device
    elif (found := _collector()) is not None:
        found.append(RequestReport(items=items, since_load=since_load, load_s=load_s, device=device))


@contextmanager
def request(tags: tuple[Hashable, ...]) -> Iterator[RequestReport]:
    """Open the report of a request carrying the jobs `tags`; kept, with its wall, once the block ends and recorded.

    Its wall is read on the clock of the enclosing `collected`, else the system clock.

    A block that raises or records nothing (a request every job of which the store served) keeps no report, unless
    `mark` keeps it; the report is yielded for that.
    """
    report = RequestReport(tags=tags)
    opened = _open()
    opened.append(report)
    clock: Clock = getattr(_LOCAL, "clock", None) or SystemClock()
    began = clock.now()
    try:
        yield report
    finally:
        report.wall_s = clock.now() - began
        opened.pop()
    if (found := _collector()) is not None and report.items:
        found.append(report)


def mark(report: RequestReport, failure: Failure, fault: FaultClass, items: int) -> None:
    """Mark `report` as a request that failed, kept in the enclosing `collected` even if it raised and recorded none.

    `items` is the number of jobs the request carried, taken as its items where it recorded none.
    """
    report.failure, report.fault = failure, fault.value
    found = _collector()
    if found is not None and not any(r is report for r in found):
        report.items = report.items or items
        found.append(report)


@contextmanager
def collected(clock: Clock = SystemClock()) -> Iterator[list[RequestReport]]:
    """Yield the list the requests this thread runs inside the block are reported in, their walls read on `clock`."""
    previous = getattr(_LOCAL, "found", None), getattr(_LOCAL, "clock", None)
    found: list[RequestReport] = []
    _LOCAL.found, _LOCAL.clock = found, clock
    try:
        yield found
    finally:
        _LOCAL.found, _LOCAL.clock = previous
