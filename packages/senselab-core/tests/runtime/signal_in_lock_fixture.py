"""Deliver a signal to the main thread while it holds the lock of a `threading` primitive, deterministically.

`deliver_in(signum, where)` installs a trace on the calling thread. The first time a frame of `threading.py` that
`where(frame)` accepts executes its second line, which is inside the `with self._cond:` of `Event.wait` and
`Event.set`, `signal.raise_signal(signum)` runs there, so the handler interrupts a frame that holds the lock.
"""

import signal
import sys
import threading
from collections.abc import Callable
from types import FrameType
from typing import Any

THREADING = threading.__file__


def deliver_in(
    signum: int,
    where: Callable[[FrameType], bool],
    files: tuple[str, ...] = (THREADING,),
    at: Callable[[FrameType], bool] | None = None,
) -> dict[str, bool]:
    """Trace this thread; deliver `signum` once inside the first frame of one of `files` that `where` accepts.

    The signal is raised on the frame's second line, or, with `at`, on its first line for which `at(frame)` holds.
    """
    state = {"fired": False}
    lines: dict[FrameType, int] = {}

    def local(frame: FrameType, event: str, _arg: Any) -> Any:  # noqa: ANN401 - the trace-function protocol
        if event == "line" and not state["fired"]:
            lines[frame] = lines.get(frame, 0) + 1
            if (lines[frame] == 2) if at is None else at(frame):
                state["fired"] = True
                signal.raise_signal(signum)
        return local

    def trace(frame: FrameType, event: str, _arg: Any) -> Any:  # noqa: ANN401 - the trace-function protocol
        if event == "call" and not state["fired"] and frame.f_code.co_filename in files and where(frame):
            return local
        return None

    sys.settrace(trace)
    return state
