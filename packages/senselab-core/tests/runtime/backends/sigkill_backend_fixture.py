"""A backend whose worker process kills itself with SIGKILL on any request, as the kernel's OOM killer does."""

import os
import signal
from typing import Any


def load(init: dict[str, Any]) -> object:
    """Return `init` as the state."""
    return init


def handle(state: object, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Kill this process with SIGKILL."""
    os.kill(os.getpid(), signal.SIGKILL)
    return []
