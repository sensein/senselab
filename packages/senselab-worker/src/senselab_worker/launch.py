"""Worker entry point: `python -m senselab_worker.launch <backend_module>`.

Before anything else, the protocol moves to private, non-inheritable duplicates of fd 0 and fd 1;
fd 0 and `sys.stdin` then read `/dev/null` and fd 1 and `sys.stdout` write to stderr, so nothing a
backend or its children read or print touches the channel. If fd 0 is closed at start it is first
opened on `/dev/null`, so the channel reads end of file and the worker exits 0 after its hello. The
backend module defines
`load(init) -> state` and `handle(state, items, params) -> list[dict]` (see `serve`); it is
imported during load, so an import failure is reported as a fatal error.
"""

import importlib
import os
import sys
from typing import Any, BinaryIO

from senselab_worker.serve import serve


def _claim_channel() -> tuple[BinaryIO, BinaryIO]:
    sys.stdout.flush()
    try:
        os.fstat(0)
    except OSError:
        os.open(os.devnull, os.O_RDONLY)
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
    rx, tx = _claim_channel()
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
