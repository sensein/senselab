"""Probes the worker lifecycle tests share: process liveness, process groups and a gate backends block on."""

import json
import os
import socket
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

RUNTIME_TESTS = Path(__file__).parent
GATE_TIMEOUT_S = 30.0


def can_run(stat: str) -> bool:
    """Whether a `ps` state string names a process that can still run user code: not a zombie, not exiting.

    The first letter is the run state (`Z` zombie, `X` dead); macOS adds an `E` flag while a process exits.
    """
    return bool(stat) and stat[0] not in "ZX" and "E" not in stat[1:]


def alive(pid: int) -> bool:
    """Whether `pid` names a process that can still run user code."""
    return can_run(state(pid))


def state(pid: int) -> str:
    """The `ps` state letters of `pid`, empty if it has gone."""
    out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, check=False)
    return out.stdout.strip()


def group_alive(pgid: int) -> bool:
    """Whether any process of group `pgid` can still run user code, by `ps` over every process."""
    out = subprocess.run(["ps", "-A", "-o", "pgid=,stat="], capture_output=True, text=True, check=True)
    rows = (line.split() for line in out.stdout.splitlines())
    return any(len(row) == 2 and int(row[0]) == pgid and can_run(row[1]) for row in rows)


def recorded_pids(pid_file: Path) -> list[int]:
    """The pids a backend appended to `pid_file`, none if it wrote nothing."""
    if not pid_file.exists():
        return []
    return [int(json.loads(line)) for line in pid_file.read_text().splitlines()]


def interpreter(directory: Path, body: str) -> Path:
    """Write an executable shell script that stands in for the worker's interpreter and ignores its arguments."""
    path = directory / "interpreter"
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)
    return path


def listing(directory: Path) -> set[str]:
    """Every path under `directory`, relative to it, without following symlinks."""
    return {str(p.relative_to(directory)) for p in directory.rglob("*")}


class Gate:
    """A listening socket a `gate`-mode backend connects to; `entered` waits for it, `release` frees it."""

    def __init__(self, server: socket.socket) -> None:
        """Wrap a listening server socket."""
        self._server = server
        self._conn: socket.socket | None = None
        self.port: int = server.getsockname()[1]

    def entered(self) -> None:
        """Block until the backend is inside `handle`, at most `GATE_TIMEOUT_S`."""
        self._conn, _ = self._server.accept()
        self._conn.settimeout(GATE_TIMEOUT_S)
        if self._conn.recv(2) != b"in":
            raise RuntimeError("the backend closed the gate without entering it")

    def release(self) -> None:
        """Let the backend return from `handle`."""
        assert self._conn is not None
        self._conn.sendall(b"go")


@contextmanager
def gate() -> Iterator[Gate]:
    """Yield a `Gate` on a fresh localhost port; close it on exit."""
    server = socket.create_server(("127.0.0.1", 0))
    server.settimeout(GATE_TIMEOUT_S)
    g = Gate(server)
    try:
        yield g
    finally:
        if g._conn is not None:
            g._conn.close()
        server.close()
