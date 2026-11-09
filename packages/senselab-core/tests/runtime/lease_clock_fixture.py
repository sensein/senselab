"""A clock a test advances, for leases: monotonic time, the storage clock and waits all move only when told to.

`ManualClock` is passed as `Lease(clock=...)`. `sleep` advances it (or only yields, when built with `spends=False`),
`wait` on an event returns when the event is set or the clock has been advanced past the timeout, and `settle`
returns once every thread parked in `wait` has seen the latest advance, so a test that advanced the clock reads the
beats that advance caused. `age` backdates the stamps in a lease's directory, which is how a test makes storage time
pass for a lease nobody is renewing.
"""

import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Optional

from senselab_core.runtime.storage import StorageDir


class ManualClock:
    """Monotonic time starting at 0, advanced by the test; storage time is the real filesystem's unless pinned.

    Args:
        poll_s: Real seconds between a parked thread's looks at its event.
        hang_s: Real seconds after which a wait or a `settle` that has not finished fails; a failure bound only.
        storage_origin: Pins the storage clock to `storage_origin` plus the monotonic time; leases planted with
            that stamp then age exactly as the clock advances. None reads the real filesystem's stamps.
        spends: Whether `sleep` advances the clock by the time slept.
        on_sleep: Called with each `sleep`'s duration before it advances the clock.

    Attributes:
        slept: The duration of each `sleep`.
    """

    def __init__(
        self,
        *,
        poll_s: float,
        hang_s: float,
        storage_origin: Optional[float] = None,
        spends: bool = True,
        on_sleep: Optional[Callable[[float], None]] = None,
    ) -> None:
        """Start at time 0 with nothing parked."""
        self._now = 0.0
        self._version = 0
        self._cond = threading.Condition()
        self._parked: dict[threading.Thread, int] = {}
        self._origin, self._spends, self.on_sleep = storage_origin, spends, on_sleep
        self._poll_s, self._hang_s = poll_s, hang_s
        self._guard = time.monotonic() + hang_s
        self.slept: list[float] = []

    def now(self) -> float:
        """Return the monotonic time."""
        return self._now

    def storage_now(self, directory: StorageDir) -> float:
        """Return the pinned storage time, or `directory`'s filesystem's."""
        return directory.dir_now() if self._origin is None else self._origin + self._now

    def advance(self, seconds: float) -> None:
        """Move the clock ahead `seconds` and wake the threads waiting on it."""
        with self._cond:
            self._now += seconds
            self._version += 1
            self._cond.notify_all()

    def advance_to(self, when: float) -> None:
        """Move the clock to `when`, which is not in the past."""
        self.advance(max(0.0, when - self.now()))

    def sleep(self, seconds: float) -> None:
        """Record the sleep, call `on_sleep`, and advance the clock unless it was built with `spends=False`."""
        self.slept.append(seconds)
        if self.on_sleep is not None:
            self.on_sleep(seconds)
        if self._spends:
            self.advance(seconds)
        else:
            assert time.monotonic() < self._guard, f"a wait under a clock that never advances outlived {self._hang_s}s"
            time.sleep(0)

    def wait(self, event: threading.Event, timeout: float) -> bool:
        """Return True once `event` is set, False once the clock is `timeout` past now; park until one happens."""
        deadline = self.now() + timeout
        thread = threading.current_thread()
        while True:
            with self._cond:
                if event.is_set():
                    self._parked.pop(thread, None)
                    return True
                if self.now() >= deadline:
                    return False
                self._parked[thread] = self._version
                self._cond.notify_all()
            event.wait(self._poll_s)

    def settle(self, parked: int = 1) -> None:
        """Return once `parked` live threads are parked in `wait` and each has seen the latest advance."""
        deadline = time.monotonic() + self._hang_s
        with self._cond:
            while not self._settled(parked):
                assert time.monotonic() < deadline, "a thread did not park after the clock advanced"
                self._cond.wait(self._poll_s)

    def parked(self, thread: threading.Thread) -> bool:
        """Return once `thread` is parked in `wait` at the latest advance, or False once it has ended."""
        deadline = time.monotonic() + self._hang_s
        with self._cond:
            while thread.is_alive() and self._parked.get(thread) != self._version:
                assert time.monotonic() < deadline, "the thread did not park after the clock advanced"
                self._cond.wait(self._poll_s)
            return thread.is_alive()

    def until(self, predicate: Callable[[], bool], what: str, step: float = 0.01) -> None:
        """Advance the clock by `step`, once the threads parked in `wait` have seen it, until `predicate` holds.

        Fails if `predicate` has not held after `hang_s` real seconds; that bound is a failure only.
        """
        deadline = time.monotonic() + self._hang_s
        while not predicate():
            assert time.monotonic() < deadline, f"not within {self._hang_s}s: {what}"
            self.advance(step)
            self.settle(0)

    def drive(self, thread: threading.Thread, step: float) -> None:
        """Advance the clock by `step` each time `thread` is parked in `wait` at the latest advance, until it ends.

        The thread is never advanced past while it works, so what it reads between two waits is read on one time.
        """
        while self.parked(thread):
            self.advance(step)

    def _settled(self, parked: int) -> bool:
        live = {t: v for t, v in self._parked.items() if t.is_alive()}
        return len(live) >= parked and all(v == self._version for v in live.values())


class SpendingClock(ManualClock):
    """A `ManualClock` for one thread: a `wait` not already satisfied spends its whole timeout at once."""

    def wait(self, event: threading.Event, timeout: float) -> bool:
        """Return True if `event` is set; else move the clock `timeout` ahead and return whether it is set now."""
        if not event.is_set():
            self.advance(timeout)
        return event.is_set()


def newest(path: Path) -> float:
    """Return the newest modification time of `path` and of everything below it."""
    return max(os.lstat(p).st_mtime for p in (path, *path.rglob("*")))


def gate(directory: str, takers: int, *, hang_s: float) -> None:
    """Join a barrier of `takers` directory entries in `directory`; fail if the others have not come in `hang_s`."""
    os.mkdir(os.path.join(directory, os.urandom(8).hex()))
    guard = time.monotonic() + hang_s
    while len(os.listdir(directory)) < takers:
        assert time.monotonic() < guard, f"only {len(os.listdir(directory))} of {takers} takers came in {hang_s}s"
        time.sleep(0)


def age(parent: Path, by: float) -> None:
    """Backdate by `by` seconds the stamps of every lease directory in `parent` and every entry directly in them."""
    for lease in parent.glob("*.lease"):
        for path in (*lease.iterdir(), lease):
            stamp = os.lstat(path).st_mtime - by
            os.utime(path, (stamp, stamp), follow_symlinks=False)
