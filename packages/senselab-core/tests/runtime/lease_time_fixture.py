"""Storage time advanced at the `os` boundary, and the beats lease renewers make, for tests of beating leases.

`install(monkeypatch)` wraps `os.lstat` relative to a directory so a stamp made after an advance reads that much
later, and wraps `os.mkdir` to count beat directories. `outlast` advances the storage clock past a lease's stale
bound in steps, waits for a beat after each, and tries to break the lease each time.
"""

import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from senselab_core.runtime.locks import Lease, LeaseTimeout

BEAT = re.compile(r"beat-[0-9]+")
WAIT_S = 10.0


class _Shifted:
    """A stat result whose mtime reads `by` seconds later."""

    def __init__(self, st: os.stat_result, by: float) -> None:
        self._st, self._by = st, by

    def __getattr__(self, name: str) -> object:
        if name == "st_mtime":
            return self._st.st_mtime + self._by
        if name == "st_mtime_ns":
            return self._st.st_mtime_ns + int(self._by * 1e9)
        return getattr(self._st, name)


class LeaseTime:
    """The storage clock a test advances, the beats made so far, and the breaks `outlast` achieved."""

    def __init__(self) -> None:
        """Start with no advance, no beat and no break."""
        self.advances: list[tuple[float, float]] = []
        self.beats = 0
        self.breaks = 0
        self.attempts = 0

    def advance(self, seconds: float) -> None:
        """Move the storage clock `seconds` ahead from now on, and let the storage's stamps pass the moment."""
        time.sleep(0.05)
        self.advances.append((time.time(), seconds))
        time.sleep(0.05)

    def offset(self, mtime: float) -> float:
        """Return how far the storage clock was advanced before the stamp `mtime`."""
        return sum(by for at, by in self.advances if mtime >= at)

    def wait_beat(self) -> None:
        """Wait for the next beat, up to `WAIT_S`."""
        seen, deadline = self.beats, time.monotonic() + WAIT_S
        while self.beats <= seen and time.monotonic() < deadline:
            time.sleep(0.01)
        assert self.beats > seen, f"no beat within {WAIT_S}s"

    def outlast(self, parent: Path, stale_after: float, steps: int = 4) -> None:
        """Advance the storage clock by a third of `stale_after` `steps` times; after each beat, try to break.

        The one lease in `parent` is the target; each attempt that took it is counted in `breaks`.
        """
        for _ in range(steps):
            self.advance(stale_after / 3)
            self.wait_beat()
            (lease,) = parent.glob("*.lease")
            self.attempts += 1
            try:
                with Lease(lease.with_suffix(""), timeout=0, stale_after=stale_after):
                    self.breaks += 1
            except LeaseTimeout:
                pass


def install(monkeypatch: pytest.MonkeyPatch) -> LeaseTime:
    """Wrap `os.lstat` and `os.mkdir` so the test controls storage time and sees each beat; return the clock."""
    clock, real_lstat, real_mkdir = LeaseTime(), os.lstat, os.mkdir

    def lstat(path: Any, *, dir_fd: int | None = None) -> object:  # noqa: ANN401
        st = real_lstat(path, dir_fd=dir_fd)
        return st if dir_fd is None else _Shifted(st, clock.offset(st.st_mtime))

    def mkdir(path: Any, mode: int = 0o777, *, dir_fd: int | None = None) -> None:  # noqa: ANN401
        real_mkdir(path, mode, dir_fd=dir_fd)
        if BEAT.fullmatch(os.path.basename(os.fsdecode(path))):
            clock.beats += 1

    monkeypatch.setattr(os, "lstat", lstat)
    monkeypatch.setattr(os, "mkdir", mkdir)
    return clock


def on_create(monkeypatch: pytest.MonkeyPatch, prefix: str, action: Callable[[], None]) -> list[str]:
    """Run `action` once, at the first file created whose name starts with `prefix`; return the names it ran at."""
    real_open, ran = os.open, list[str]()

    def opened(path: Any, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:  # noqa: ANN401
        name = os.path.basename(os.fsdecode(path))
        if flags & os.O_CREAT and name.startswith(prefix) and not ran:
            ran.append(name)
            action()
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", opened)
    return ran
