"""The process clock the runtime waits and measures intervals on."""

import time
from typing import Protocol


class Clock(Protocol):
    """A monotonic clock that can also wait."""

    def monotonic(self) -> float:
        """Return seconds on a clock that never goes backwards."""
        ...

    def sleep(self, seconds: float) -> None:
        """Wait `seconds`."""
        ...


class SystemClock:
    """The process's own clock: `time.monotonic` and `time.sleep`."""

    def monotonic(self) -> float:
        """Return `time.monotonic()`."""
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        """Call `time.sleep(seconds)`."""
        time.sleep(seconds)


SYSTEM_CLOCK: Clock = SystemClock()
"""The default clock of every runtime component that takes one."""
