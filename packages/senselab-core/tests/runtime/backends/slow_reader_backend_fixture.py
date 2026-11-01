"""A backend that, asked to with `params["mode"] == "stop_reading"`, stops reading its channel.

After answering that request it stops itself (SIGSTOP) as soon as the serve loop is back waiting
for the next line, so a later request is never read.
"""

import os
import signal
import sys
import threading
import time
from typing import Any

from backends import modes_fixture as _modes

WATCH_S = 30.0
_POLL_S = 0.001


def load(init: dict[str, Any]) -> object:
    """Return `init` as the state."""
    return init


def _stop_when_waiting(main: int) -> None:
    end = time.monotonic() + WATCH_S
    while time.monotonic() < end:
        frame = sys._current_frames().get(main)
        if frame is not None and frame.f_code.co_name == "_receive":
            os.kill(os.getpid(), signal.SIGSTOP)
            return
        time.sleep(_POLL_S)


def handle(state: Any, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:  # noqa: ANN401
    """Act out the mode; for `stop_reading`, answer and then stop."""
    if params.get("mode") == "stop_reading":
        main = threading.main_thread().ident
        assert main is not None
        threading.Thread(target=_stop_when_waiting, args=(main,), daemon=True).start()
    else:
        _modes.apply(state, params)
    return [{"outputs": {"ok": True}} for _ in items]
