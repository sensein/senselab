"""Worker entry point: `python -m senselab_worker.launch <backend_module>`.

Before anything else, fd 1 and `sys.stdout` are pointed at stderr and the protocol keeps a private
duplicate of fd 1, so nothing a backend prints reaches the host. The backend module defines
`load(init) -> state` and `handle(state, items, params) -> list[dict]` (see `serve`); it is
imported during load, so an import failure is reported as a fatal error.
"""

import importlib
import os
import sys
from typing import Any, BinaryIO

from senselab_worker.serve import serve


def _claim_stdout() -> BinaryIO:
    sys.stdout.flush()
    protocol = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return protocol


def main(argv: list[str]) -> int:
    """Serve the backend named by `argv[1]` on stdin and the original stdout; return the exit code."""
    protocol = _claim_stdout()
    if len(argv) != 2:
        print("usage: python -m senselab_worker.launch <backend_module>", file=sys.stderr)
        return 2
    backend: dict[str, Any] = {}

    def load(init: dict[str, Any]) -> object:
        module = importlib.import_module(argv[1])
        backend["handle"] = module.handle
        return module.load(init)

    def handle(state: object, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = backend["handle"](state, items, params)
        return result

    return serve(load, handle, sys.stdin.buffer, protocol)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
