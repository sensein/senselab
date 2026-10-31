"""A backend whose every request leaves two grandchildren sleeping: one exec'd, one forked.

Both inherit stderr; the forked one also holds every descriptor of the worker, the channel included.
"""

import os
import subprocess
import sys
import time
from typing import Any

from backends import modes_fixture as _modes

SLEEP_S = 3600


def load(init: dict[str, Any]) -> object:
    """Return `init` as the state."""
    return init


def handle(state: Any, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:  # noqa: ANN401
    """Spawn the grandchildren, record their pids, then act out the mode."""
    child = subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({SLEEP_S})"])
    forked = os.fork()
    if forked == 0:
        time.sleep(SLEEP_S)
        os._exit(0)
    _modes.record(state, [child.pid, forked])
    _modes.apply(state, params)
    return [{"outputs": {"pids": [child.pid, forked]}} for _ in items]
