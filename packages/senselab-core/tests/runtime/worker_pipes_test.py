"""The line queue a worker client reads: a line put while the reader looks is never waited out."""

import os
import queue
import subprocess
import sys
import threading
import types
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


def _open_fds() -> set[int]:
    found = set()
    for name in os.listdir("/dev/fd"):
        try:
            os.fstat(int(name))
        except OSError:
            continue  # the descriptor of the listing itself
        found.add(int(name))
    return found


def _paused_at(call: str, caller: str) -> tuple[Callable[..., None], threading.Event, threading.Event]:
    """A profile hook that holds its thread at its first C call named `call` made within a function named `caller`."""
    paused, resume = threading.Event(), threading.Event()

    def within(frame: types.FrameType | None) -> bool:
        while frame is not None and frame.f_code.co_name != caller:
            frame = frame.f_back
        return frame is not None

    def hook(frame: types.FrameType, event: str, arg: object) -> None:
        if event == "c_call" and getattr(arg, "__name__", "") == call and within(frame):
            if not paused.is_set():
                paused.set()
                resume.wait(DEADLINE_S)

    return hook, paused, resume


def _profiled(hook: Callable[..., None], body: Callable[[], object], out: list[object]) -> threading.Thread:
    def target() -> None:
        sys.setprofile(hook)
        try:
            out.append(body())
        finally:
            sys.setprofile(None)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread


@pytest.mark.contract
def test_a_close_while_a_setter_holds_its_descriptor_never_lets_it_write_into_a_reused_one() -> None:
    """A setter held between reading the write end and writing it, then closed under, writes into no later pipe.

    Once the setter has returned the event holds no descriptor.
    """
    before = _open_fds()
    event = WakeEvent()
    hook, paused, resume = _paused_at("write", "set")
    setter = _profiled(hook, event.set, [])
    assert paused.wait(DEADLINE_S) and event.is_set()
    event.close()
    probes = [os.pipe() for _ in range(4)]  # the lowest free numbers: the event's, if it released them
    resume.set()
    setter.join(DEADLINE_S)
    assert not setter.is_alive()
    for r, w in probes:
        os.set_blocking(r, False)
        with pytest.raises(BlockingIOError):
            os.read(r, 1)
        os.close(r)
        os.close(w)
    assert _open_fds() == before


@pytest.mark.contract
def test_a_wait_in_flight_when_the_event_is_closed_returns_at_once() -> None:
    """A waiter held at its selector's wait while another thread closes the event returns the flag, not its timeout.

    Once the waiter has returned the event holds no descriptor.
    """
    before = _open_fds()
    event = WakeEvent()
    hook, paused, resume = _paused_at("control" if sys.platform == "darwin" else "poll", "select")
    out: list[object] = []
    waiter = _profiled(hook, lambda: event.wait(10 * DEADLINE_S), out)
    assert paused.wait(DEADLINE_S)
    event.close()
    resume.set()
    waiter.join(DEADLINE_S)
    assert not waiter.is_alive() and out == [False]
    assert _open_fds() == before


@pytest.mark.contract
def test_a_closed_wake_event_keeps_its_flag_and_never_blocks() -> None:
    """After `close`, `set`, `clear` and `wait` work on the flag alone and the event holds no descriptor."""
    before = _open_fds()
    event = WakeEvent()
    event.close()
    event.close()
    assert event.wait(None) is False
    event.set()
    assert event.is_set() and event.wait(None) is True
    event.clear()
    assert not event.is_set() and event.wait(DEADLINE_S) is False
    assert _open_fds() == before
