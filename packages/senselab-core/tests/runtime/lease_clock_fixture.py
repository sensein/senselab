"""A clock a test advances, for leases: monotonic time, the storage clock and waits all move only when told to.

`ManualClock` is passed as `Lease(clock=...)`. `sleep` advances it (or only yields, when built with `spends=False`),
`wait` on an event returns when the event is set or the clock has been advanced past the timeout (or, built with
`waits="spend"`, spends the timeout at once), and `settle` returns once every thread parked in `wait` has seen the
latest advance, so a test that advanced the clock reads the beats that advance caused. Every wait and loop that
a real event ends keeps a real failure bound, `hang_s`.

`age` backdates the stamps in a lease's directory, which is how a test makes storage time
pass for a lease nobody is renewing.
"""

import os
import selectors
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Optional, TypeVar, Union

from senselab_core.runtime.storage import StorageDir

_T = TypeVar("_T")


class ManualClock:
    """Monotonic time starting at 0, advanced by the test; storage time is the real filesystem's unless pinned.

    Args:
        poll_s: Real seconds between a parked thread's looks at its event.
        hang_s: Real seconds after which a wait, `settle` or drive that has not finished fails; a failure bound only.
        step_s: The default step `until` advances by.
        start: The time the clock starts at.
        waits: `park` to hold a wait until the clock is advanced; `spend` to spend its timeout when it is called.
        cell: The one-element list that holds the time, which a fake backend may advance too.
        storage_origin: Pins the storage clock to `storage_origin` plus the monotonic time; leases planted with
            that stamp then age exactly as the clock advances. None reads the real filesystem's stamps.
        spends: `True` for a `sleep` that advances the clock by the time slept, `False` for one that only yields,
            `"park"` for one that parks, as a `wait` does, until the clock has been advanced past it.
        on_sleep: Called with each `sleep`'s duration before it advances the clock.

    Attributes:
        slept: The duration of each `sleep`.
    """

    def __init__(  # noqa: PLR0913 - the clock's real bounds, then its keyword-only variants
        self,
        *,
        poll_s: float,
        hang_s: float,
        step_s: float = 0.0,
        start: float = 0.0,
        waits: Literal["park", "spend"] = "park",
        cell: Optional[list[float]] = None,
        storage_origin: Optional[float] = None,
        spends: Union[bool, Literal["park"]] = True,
        on_sleep: Optional[Callable[[float], None]] = None,
    ) -> None:
        """Start at `start` with nothing parked."""
        self._cell = cell if cell is not None else [start]
        self._waits, self.step_s = waits, step_s
        self._version = 0
        self._cond = threading.Condition()
        self._parked: dict[threading.Thread, int] = {}
        self._origin, self._spends, self.on_sleep = storage_origin, spends, on_sleep
        self.poll_s, self.hang_s = poll_s, hang_s
        self._guard = time.monotonic() + hang_s
        self.slept: list[float] = []

    def now(self) -> float:
        """Return the monotonic time."""
        return self._cell[0]

    def storage_now(self, directory: StorageDir) -> float:
        """Return the pinned storage time, or `directory`'s filesystem's."""
        return directory.dir_now() if self._origin is None else self._origin + self.now()

    def local_now(self) -> float:
        """Return the pinned storage time, or the real wall clock."""
        return time.time() if self._origin is None else self._origin + self.now()

    def advance(self, seconds: float) -> None:
        """Move the clock ahead `seconds` and wake the threads waiting on it."""
        with self._cond:
            self._cell[0] += seconds
            self._version += 1
            self._cond.notify_all()

    def advance_to(self, when: float) -> None:
        """Move the clock to `when`, which is not in the past."""
        self.advance(max(0.0, when - self.now()))

    def sleep(self, seconds: float) -> None:
        """Record the sleep, call `on_sleep`, and advance the clock unless it was built with `spends=False`."""
        self.slept.append(seconds)
        with self._cond:
            self._parked.pop(threading.current_thread(), None)
        if self.on_sleep is not None:
            self.on_sleep(seconds)
        if self._spends == "park":
            self._park(seconds, lambda real_s: time.sleep(real_s) if real_s else None)
        elif self._spends:
            self.advance(seconds)
        else:
            assert time.monotonic() < self._guard, f"a wait under a clock that never advances outlived {self.hang_s}s"
            time.sleep(0)

    def wait(self, event: threading.Event, timeout: float) -> bool:
        """Return True once `event` is set, False once the clock is `timeout` past now; park until one happens."""
        if self._waits == "spend":
            if not event.is_set():
                self.advance(timeout)
            return event.is_set()
        return self._park(timeout, lambda real_s: True if event.wait(real_s) else None) is True

    def select(
        self, selector: selectors.BaseSelector, timeout: Optional[float]
    ) -> list[tuple[selectors.SelectorKey, int]]:
        """Return the ready files of `selector`, or none once the clock is `timeout` past now; park until then."""
        ready = self._park(float("inf") if timeout is None else timeout, lambda real_s: selector.select(real_s) or None)
        return [] if ready is None else ready

    def _park(self, timeout: float, look: Callable[[float], Optional[_T]]) -> Optional[_T]:
        """Return what `look` finds, or None once the clock is `timeout` past now; `look(s)` blocks up to `s` real s."""
        deadline, guard = self.now() + timeout, time.monotonic() + self.hang_s
        thread = threading.current_thread()
        while True:
            found = look(0.0)
            with self._cond:
                if found is not None:
                    self._parked.pop(thread, None)
                    return found
                if self.now() >= deadline:
                    return None
                assert time.monotonic() < guard, f"a wait on a clock nobody advanced outlived {self.hang_s}s"
                self._parked[thread] = self._version
                self._cond.notify_all()
            look(self.poll_s)

    def settle(self, parked: int = 1) -> None:
        """Return once `parked` live threads are parked in `wait` and each has seen the latest advance."""
        deadline = time.monotonic() + self.hang_s
        with self._cond:
            while not self._settled(parked):
                assert time.monotonic() < deadline, "a thread did not park after the clock advanced"
                self._cond.wait(self.poll_s)

    def parked(self, thread: threading.Thread) -> bool:
        """Return once `thread` is parked in `wait` at the latest advance, or False once it has ended."""
        deadline = time.monotonic() + self.hang_s
        with self._cond:
            while thread.is_alive() and self._parked.get(thread) != self._version:
                assert time.monotonic() < deadline, "the thread did not park after the clock advanced"
                self._cond.wait(self.poll_s)
            return thread.is_alive()

    def is_parked(self, thread: threading.Thread) -> bool:
        """Return whether `thread` is parked in `wait` at the latest advance now; never waits."""
        with self._cond:
            return thread.is_alive() and self._parked.get(thread) == self._version

    def until(self, predicate: Callable[[], bool], what: str, step: Optional[float] = None) -> None:
        """Advance the clock by `step` (default `step_s`), once the parked threads saw it, until `predicate` holds.

        Fails if `predicate` has not held after `hang_s` real seconds; that bound is a failure only.
        """
        deadline = time.monotonic() + self.hang_s
        while not predicate():
            assert time.monotonic() < deadline, f"not within {self.hang_s}s: {what}"
            self.advance(self.step_s if step is None else step)
            self.settle(0)

    def drive(self, thread: threading.Thread, step: float) -> None:
        """Advance the clock by `step` each time `thread` is parked in `wait` at the latest advance, until it ends.

        The thread is never advanced past while it works, so what it reads between two waits is read on one time.
        Fails if `thread` has not ended after `hang_s` real seconds.
        """
        deadline = time.monotonic() + self.hang_s
        while self.parked(thread):
            assert time.monotonic() < deadline, f"the driven thread outlived {self.hang_s}s"
            self.advance(step)

    def _settled(self, parked: int) -> bool:
        live = {t: v for t, v in self._parked.items() if t.is_alive()}
        return len(live) >= parked and all(v == self._version for v in live.values())


class Called:
    """`fn()` running on a daemon thread; `result` is what it returned, or raises what it raised.

    Attributes:
        thread: The thread running `fn`.
    """

    def __init__(self, fn: Callable[[], object], *, hang_s: float) -> None:
        """Start `fn`; `result` fails if it has not ended `hang_s` real seconds after it is asked."""
        self._hang_s = hang_s
        self._value: object = None
        self._error: Optional[BaseException] = None

        def run() -> None:
            try:
                self._value = fn()
            except BaseException as exc:  # noqa: BLE001 - handed to `result`
                self._error = exc

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def result(self) -> object:
        """Return what `fn` returned, or raise what it raised; fail if it has not ended within the hang bound."""
        self.thread.join(self._hang_s)
        assert not self.thread.is_alive(), f"the call outlived {self._hang_s}s"
        if self._error is not None:
            raise self._error
        return self._value


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
