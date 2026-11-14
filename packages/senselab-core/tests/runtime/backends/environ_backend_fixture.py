"""A backend whose output for each item is how its worker process was started: isolated, cwd and environment."""

import os
import sys
from typing import Any


def load(init: dict[str, Any]) -> object:
    """Return `init` as the state."""
    return init


def handle(state: object, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Return `sys.flags.isolated`, the working directory and the environment's variables."""
    started = {"isolated": sys.flags.isolated, "cwd": os.getcwd(), "environ": dict(os.environ)}
    return [{"outputs": started} for _ in items]
