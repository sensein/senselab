"""The line queue a worker client reads: a line put while the reader looks is never waited out."""

import queue
import threading
from collections.abc import Callable

import pytest
from lease_clock_fixture import Called, ManualClock
from senselab_core.runtime.worker_pipes import Lines
from senselab_worker.codec import Deadline

LINE = b"reply\n"
DEADLINE_S = 8.0


class _PutAtClear(threading.Event):
    """An event that, at its `at`-th `clear`, first has the pump put `LINE`, as a pump thread may at that point."""

    def __init__(self, at: int) -> None:
        super().__init__()
        self.at, self.cleared, self.fired = at, 0, False
        self.lines: Lines | None = None

    def clear(self) -> None:
        self.cleared += 1
        if self.cleared == self.at:
            assert self.lines is not None
            self.fired = True
            self.lines.put(LINE)
        super().clear()


@pytest.mark.contract
@pytest.mark.parametrize("at", [1, 2])
def test_a_line_put_as_the_reader_clears_its_flag_is_returned_without_waiting(
    at: int, lease_clock: Callable[..., ManualClock]
) -> None:
    """A line put just before any `clear` of the wake flag is returned at once, never after the deadline."""
    clock = lease_clock()
    event = _PutAtClear(at)
    lines = Lines(clock, event)
    event.lines = lines
    call = Called(lambda: lines.get(Deadline(DEADLINE_S, clock.now)), hang_s=clock.hang_s)
    assert not (clock.parked(call.thread) and event.fired), "a line was put and the reader still waits for the deadline"
    if not event.fired:
        lines.put(LINE)
    assert call.result() == LINE


@pytest.mark.contract
def test_an_empty_queue_raises_empty_once_the_clock_passes_the_deadline(
    lease_clock: Callable[..., ManualClock],
) -> None:
    """With no line, `get` waits on the clock and raises `queue.Empty` when the deadline has passed, not before."""
    clock = lease_clock()
    call = Called(lambda: Lines(clock).get(Deadline(DEADLINE_S, clock.now)), hang_s=clock.hang_s)
    assert clock.parked(call.thread)
    clock.advance(DEADLINE_S / 2)
    assert clock.parked(call.thread)
    clock.advance_to(DEADLINE_S)
    with pytest.raises(queue.Empty):
        call.result()
