"""A backend that leaves two grandchildren sleeping, one exec'd, one forked: per request, or once in `load`.

Both inherit stderr; the forked one also holds every descriptor of the worker, the channel included.
"""

import os
import subprocess
import sys
import time
from typing import Any

from backends import modes_fixture as _modes

SLEEP_S = 3600


def _spawn(init: dict[str, Any]) -> list[int]:
    child = subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({SLEEP_S})"])
    forked = os.fork()
    if forked == 0:
        time.sleep(SLEEP_S)
        os._exit(0)
    _modes.record(init, [child.pid, forked])
    return [child.pid, forked]


def load(init: dict[str, Any]) -> object:
    """Return `init` as the state; with `init["spawn_at_load"]`, spawn and record the grandchildren now.

    With `init["exit_at_load"]`, spawn them and then exit with `modes_fixture.EXIT_CRASH` before ready.
    """
    if init.get("exit_at_load"):
        _spawn(init)
        os._exit(_modes.EXIT_CRASH)
    if init.get("spawn_at_load"):
        _spawn(init)
    return init


def handle(state: Any, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:  # noqa: ANN401
    """Spawn and record the grandchildren unless `load` did, then act out the mode."""
    pids = [] if state.get("spawn_at_load") else _spawn(state)
    _modes.apply(state, params)
    return [{"outputs": {"pids": pids}} for _ in items]
