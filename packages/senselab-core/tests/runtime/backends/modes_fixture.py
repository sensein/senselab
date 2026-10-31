"""Behaviour shared by the lifecycle fixture backends, chosen per request by `params["mode"]`.

- `ok` (default): return normally.
- `hang`: block forever.
- `deaf`: ignore SIGTERM, then block forever.
- `crash`: exit at once with status 70.
- `gate`: connect to `127.0.0.1:init["gate_port"]`, send `in`, and block until two bytes come back.
"""

import json
import os
import signal
import socket
import threading
from collections.abc import Iterable
from typing import Any

EXIT_CRASH = 70


def apply(init: dict[str, Any], params: dict[str, Any]) -> None:
    """Act out `params["mode"]`."""
    mode = params.get("mode", "ok")
    if mode == "deaf":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if mode in ("hang", "deaf"):
        threading.Event().wait()
    elif mode == "crash":
        os._exit(EXIT_CRASH)
    elif mode == "gate":
        with socket.create_connection(("127.0.0.1", init["gate_port"])) as conn:
            conn.sendall(b"in")
            conn.recv(2)


def record(init: dict[str, Any], pids: Iterable[int]) -> None:
    """Append `pids` to `init["pid_file"]`, one JSON number per line, if a pid file was given."""
    if "pid_file" in init:
        with open(init["pid_file"], "a") as f:
            f.write("".join(f"{json.dumps(pid)}\n" for pid in pids))
