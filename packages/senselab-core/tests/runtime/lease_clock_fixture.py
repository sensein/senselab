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

HANG_GUARD_S = 10.0
"""Real seconds `settle` waits before it reports a hang; it is a failure bound, never a passing condition."""


class ManualClock:
    """Monotonic time starting at 0, advanced by the test; storage time is the real filesystem's unless pinned.

    Args:
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
        self.advance(max(0.0, when - self._now))

    def sleep(self, seconds: float) -> None:
        """Record the sleep, call `on_sleep`, and advance the clock unless it was built with `spends=False`."""
        self.slept.append(seconds)
        if self.on_sleep is not None:
            self.on_sleep(seconds)
        if self._spends:
            self.advance(seconds)
        else:
            time.sleep(0)

    def wait(self, event: threading.Event, timeout: float) -> bool:
        """Return True once `event` is set, False once the clock is `timeout` past now; park until one happens."""
        deadline = self._now + timeout
        thread = threading.current_thread()
        while True:
            if event.is_set():
                self._parked.pop(thread, None)
                return True
            with self._cond:
                if self._now >= deadline:
                    return False
                self._parked[thread] = self._version
                self._cond.notify_all()
            event.wait(0.0005)

    def settle(self, parked: int = 1) -> None:
        """Return once `parked` live threads are parked in `wait` and each has seen the latest advance."""
        deadline = time.monotonic() + HANG_GUARD_S
        with self._cond:
            while not self._settled(parked):
                assert time.monotonic() < deadline, "a thread did not park after the clock advanced"
                self._cond.wait(0.0005)

    def _settled(self, parked: int) -> bool:
        live = {t: v for t, v in self._parked.items() if t.is_alive()}
        return len(live) >= parked and all(v == self._version for v in live.values())


def age(parent: Path, by: float) -> None:
    """Backdate by `by` seconds the stamps of every lease directory in `parent` and every entry directly in them."""
    for lease in parent.glob("*.lease"):
        for path in (*lease.iterdir(), lease):
            stamp = os.lstat(path).st_mtime - by
            os.utime(path, (stamp, stamp), follow_symlinks=False)
