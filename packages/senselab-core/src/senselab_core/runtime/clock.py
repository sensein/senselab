"""The injected clock: monotonic time, the storage clock and waits, with the real one."""

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

    def sleep(self, seconds: float) -> None:
        """Wait `seconds`."""
        ...

    def wait(self, event: threading.Event, timeout: float) -> bool:
        """Wait up to `timeout` seconds for `event`; return whether it is set."""
        ...


class SystemClock:
    """The real clock: `time.monotonic`, `StorageDir.dir_now`, `time.sleep` and `threading.Event.wait`."""

    def now(self) -> float:
        """Return `time.monotonic()`."""
        return time.monotonic()

    def storage_now(self, directory: "StorageDir") -> float:
        """Return `directory.dir_now()`."""
        return directory.dir_now()

    def sleep(self, seconds: float) -> None:
        """Wait `seconds` in `time.sleep`."""
        time.sleep(seconds)

    def wait(self, event: threading.Event, timeout: float) -> bool:
        """Wait in `event.wait(timeout)`."""
        return event.wait(timeout)
