"""The line queue a worker client reads: a line put while the reader looks is never waited out."""

import os
import queue
import subprocess
import sys
import threading
from collections.abc import Callable

import pytest
from lease_clock_fixture import Called, ManualClock
from senselab_core.runtime.worker_pipes import Lines, WakeEvent
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
    at: int, lease_clock: Callable[..., ManualClock], request: pytest.FixtureRequest
) -> None:
    """A line put just before any `clear` of the wake flag is returned at once, never after the deadline."""
    clock = lease_clock()
    event = _PutAtClear(at)
    lines = Lines(clock, event)
    event.lines = lines
    request.addfinalizer(lines.close)
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
    lines = Lines(clock)
    call = Called(lambda: lines.get(Deadline(DEADLINE_S, clock.now)), hang_s=clock.hang_s)
    assert clock.parked(call.thread)
    clock.advance(DEADLINE_S / 2)
    assert clock.parked(call.thread)
    clock.advance_to(DEADLINE_S)
    with pytest.raises(queue.Empty):
        call.result()
    lines.close()


_HANDLER_PUTS = """
import os, signal, sys, threading
from signal_in_lock_fixture import deliver_in
from senselab_core.runtime.clock import SystemClock
from senselab_core.runtime.worker_pipes import Lines
from senselab_worker.codec import Deadline

lines = Lines(SystemClock())
signal.signal(signal.SIGUSR1, lambda *_: lines.put(None))
threading.Timer(0.3, os.kill, (os.getpid(), signal.SIGUSR1)).start()
deliver_in(signal.SIGUSR1, lambda f: f.f_code.co_name == "wait" and isinstance(f.f_locals.get("self"), threading.Event))
print("woken" if lines.get(Deadline(60.0, SystemClock().now)) is None else "line", flush=True)
sys.settrace(None)
lines.close()
"""


@pytest.mark.contract
def test_a_put_from_a_signal_handler_wakes_a_reader_waiting_in_get() -> None:
    """A SIGUSR1 handler that puts while the main thread waits in `get`, even inside an `Event.wait`, wakes it."""
    proc = subprocess.run(
        [sys.executable, "-c", _HANDLER_PUTS],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip() == "woken"


@pytest.mark.contract
def test_a_wake_event_is_a_threading_event_that_sets_clears_and_times_out() -> None:
    """`WakeEvent` waits until set from another thread, returns False at its timeout and is clear after `clear`."""
    event = WakeEvent()
    assert isinstance(event, threading.Event) and not event.is_set() and event.wait(0.0) is False
    threading.Timer(0.05, event.set).start()
    assert event.wait(30.0) is True and event.is_set() and event.wait(0.0) is True
    event.clear()
    assert not event.is_set() and event.wait(0.01) is False
