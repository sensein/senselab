"""Worker entry point: `python -m senselab_worker.launch <backend_module>`.

First, `claim_standard_fds` opens `/dev/null` on each of fds 0-2 that was closed when the interpreter
started (`sys.__stdin__`, `__stdout__` or `__stderr__` is `None`), replacing whatever a module opened
on it since. The protocol then moves to private, non-inheritable duplicates of fd 0 and fd 1; fd 0
and `sys.stdin` read `/dev/null` and fd 1 and `sys.stdout` write to stderr, so nothing a backend or
its children read or print touches the channel. A worker started with fd 0 closed reads end of file
and exits 0 after its hello. The backend module defines `load(init) -> state` and
`handle(state, items, params) -> list[dict]` (see `serve`); it is imported during load, so an import
failure is reported as a fatal error.
"""

import importlib
import os
import sys
from typing import Any, BinaryIO

STANDARD_STREAMS = ((0, "__stdin__"), (1, "__stdout__"), (2, "__stderr__"))
"""Each standard fd and the `sys` attribute the interpreter sets to `None` when that fd is closed at start."""


def claim_standard_fds() -> None:
    """Point each of fds 0-2 that was closed at interpreter start at `/dev/null`; give `sys.stderr` a writer."""
    for fd, name in STANDARD_STREAMS:
        if getattr(sys, name) is None:
            devnull = os.open(os.devnull, os.O_RDWR)
            if devnull != fd:
                os.dup2(devnull, fd)
                os.close(devnull)
    if sys.stderr is None:
        sys.stderr = open(2, "w", closefd=False, errors="backslashreplace")  # noqa: SIM115 - the process's stderr


def _claim_channel() -> tuple[BinaryIO, BinaryIO]:
    if sys.stdout is not None:
        sys.stdout.flush()
    rx = os.fdopen(os.dup(0), "rb")
    tx = os.fdopen(os.dup(1), "wb")
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.close(devnull)
    os.dup2(2, 1)
    sys.stdin = open(os.devnull)  # noqa: SIM115 - replaces the interpreter's stdin for the process lifetime
    sys.stdout = sys.stderr
    return rx, tx


def main(argv: list[str]) -> int:
    """Serve the backend named by `argv[1]` on the original stdin and stdout; return the exit code."""
    claim_standard_fds()
    rx, tx = _claim_channel()
    from senselab_worker.serve import serve

    if len(argv) != 2:
        print("usage: python -m senselab_worker.launch <backend_module>", file=sys.stderr)
        return 2
    backend: dict[str, Any] = {}

    def load(init: dict[str, Any]) -> object:
        module = importlib.import_module(argv[1])  # noqa: TID251
        backend["handle"] = module.handle
        return module.load(init)

    def handle(state: object, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = backend["handle"](state, items, params)
        return result

    return serve(load, handle, rx, tx)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
