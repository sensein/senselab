"""Probes the worker lifecycle tests share: process liveness, process groups and a gate backends block on."""

import json
import os
import shlex
import socket
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

RUNTIME_TESTS = Path(__file__).parent
GATE_TIMEOUT_S = 30.0


def recorded_pids(pid_file: Path) -> list[int]:
    """The pids a backend appended to `pid_file`, none if it wrote nothing."""
    if not pid_file.exists():
        return []
    return [int(json.loads(line)) for line in pid_file.read_text().splitlines()]


def interpreter(directory: Path, body: str) -> Path:
    """Write an executable shell script that stands in for the worker's interpreter and ignores its arguments.

    The script imports from this process's `sys.path`, which it sets as its `PYTHONPATH`.
    """
    path = directory / "interpreter"
    paths = shlex.quote(os.pathsep.join(p for p in sys.path if p))
    path.write_text(f"#!/bin/sh\nexport PYTHONPATH={paths}\n{body}\n")
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
