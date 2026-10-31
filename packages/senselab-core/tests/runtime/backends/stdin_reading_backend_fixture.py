"""A backend that runs children with the default stdin, as `subprocess.run(["ffmpeg", ...])` does.

`load` leaves one child reading stdin to its end; each request runs another to completion and
reports how many bytes it read.
"""

import subprocess
import sys
from typing import Any

from backends import modes_fixture as _modes

READ_STDIN = "import sys; print(len(sys.stdin.buffer.read()))"


def load(init: dict[str, Any]) -> object:
    """Start a child reading stdin and return `init` as the state."""
    reader = subprocess.Popen([sys.executable, "-c", READ_STDIN], stdout=subprocess.DEVNULL)
    _modes.record(init, [reader.pid])
    return init


def handle(state: Any, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:  # noqa: ANN401
    """Run a stdin-reading child to completion, then act out the mode."""
    child = subprocess.run([sys.executable, "-c", READ_STDIN], capture_output=True, check=True)
    _modes.apply(state, params)
    return [{"outputs": {"stdin_bytes": int(child.stdout)}} for _ in items]
