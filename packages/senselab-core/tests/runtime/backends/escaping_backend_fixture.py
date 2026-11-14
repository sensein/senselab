"""A backend that leaves, on each request, a forked child in its group and a `setsid` escapee whose parent has exited.

`handle` returns the pids `[member, escapee]` and appends both to `init["pid_file"]` when one is given.
"""

import os
import time
from typing import Any

from backends import modes_fixture as _modes

SLEEP_S = 3600


def _sleep_forever() -> None:
    time.sleep(SLEEP_S)
    os._exit(0)


def _spawn() -> list[int]:
    member = os.fork()
    if member == 0:
        _sleep_forever()
    read, write = os.pipe()
    middle = os.fork()
    if middle == 0:
        os.close(read)
        escapee = os.fork()
        if escapee == 0:
            os.close(write)
            os.setsid()
            _sleep_forever()
        os.write(write, str(escapee).encode())
        os._exit(0)
    os.close(write)
    with os.fdopen(read, "rb") as f:
        escapee = int(f.read())
    os.waitpid(middle, 0)
    return [member, escapee]


def load(init: dict[str, Any]) -> object:
    """Return `init` as the state."""
    return init


def handle(state: Any, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:  # noqa: ANN401
    """Spawn the member and the escapee, record them, then act out the mode."""
    pids = _spawn()
    _modes.record(state, pids)
    _modes.apply(state, params)
    return [{"outputs": {"pids": pids}} for _ in items]
