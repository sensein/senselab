"""The injected clock: monotonic time, the storage clock and waits, with the real one."""

import selectors
import threading
import time
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from senselab_core.runtime.storage import StorageDir


class Clock(Protocol):
    """How time is read and spent: monotonic time, the storage clock, and waits. `SystemClock` is the real one."""

    def now(self) -> float:
        """Return monotonic seconds."""
        ...

    def storage_now(self, directory: "StorageDir") -> float:
        """Return the current time as `directory`'s filesystem stamps it."""
        ...

    def local_now(self) -> float:
        """Return the local wall-clock time, for a judgement of filesystem stamps no storage clock can be read for."""
        ...

    def sleep(self, seconds: float) -> None:
        """Wait `seconds`."""
        ...

    def wait(self, event: threading.Event, timeout: float) -> bool:
        """Wait up to `timeout` seconds for `event`; return whether it is set."""
        ...

    def select(
        self, selector: selectors.BaseSelector, timeout: float | None
    ) -> list[tuple[selectors.SelectorKey, int]]:
        """Wait up to `timeout` seconds (forever when None) for `selector`'s files; return those ready, else none."""
        ...


class SystemClock:
    """The real clock: `time.monotonic`, `StorageDir.dir_now`, `time.sleep` and `threading.Event.wait`."""

    def now(self) -> float:
        """Return `time.monotonic()`."""
        return time.monotonic()

    def storage_now(self, directory: "StorageDir") -> float:
        """Return `directory.dir_now()`."""
        return directory.dir_now()

    def local_now(self) -> float:
        """Return `time.time()`."""
        return time.time()

    def sleep(self, seconds: float) -> None:
        """Wait `seconds` in `time.sleep`."""
        time.sleep(seconds)

    def wait(self, event: threading.Event, timeout: float) -> bool:
        """Wait in `event.wait(timeout)`."""
        return event.wait(timeout)

    def select(
        self, selector: selectors.BaseSelector, timeout: float | None
    ) -> list[tuple[selectors.SelectorKey, int]]:
        """Wait in `selector.select(timeout)`."""
        return selector.select(timeout)
