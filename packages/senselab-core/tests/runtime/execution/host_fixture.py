"""Run a resume as another process, or as one on another host: what a Slurm requeue or a second node does.

A resume is a new process, so its pid is new in both shapes; in `another host` the name the host reports is new too. A
lock whose stale break reads the holder's pid and host name (as `filelock` does) is breakable from the same host and
not from another, and a test that never changes the name never meets the second case.
"""

import contextlib
import os
import platform
import socket
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass

import pytest

SHAPES = ("same host", "another host")
"""The two places a resume runs: this host's next process, and a process of a host with another name."""
_PID_OFFSET = 1_000_003


def dead_pid() -> int:
    """Return the pid of a process of this host that has ended and been reaped."""
    child = subprocess.Popen([sys.executable, "-I", "-c", "pass"])
    child.wait()
    return child.pid


@dataclass(frozen=True)
class HostShape:
    """Where a resume runs relative to the process that started the run."""

    name: str
    monkeypatch: pytest.MonkeyPatch

    @contextlib.contextmanager
    def moved(self) -> Iterator[None]:
        """Within the block this process reports a new pid, and in `another host` a new host name."""
        pid = os.getpid() + _PID_OFFSET
        with self.monkeypatch.context() as patch:
            patch.setattr(os, "getpid", lambda: pid)
            if self.name == "another host":
                elsewhere = f"{socket.gethostname()}.elsewhere"
                patch.setattr(socket, "gethostname", lambda: elsewhere)
                patch.setattr(platform, "node", lambda: elsewhere)
            yield


@pytest.fixture(params=SHAPES)
def host_shape(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> HostShape:
    """The shape a resume runs in; the test wraps its resume in `host_shape.moved()`."""
    return HostShape(request.param, monkeypatch)
