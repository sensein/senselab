"""A backend that reports which descriptors its children would inherit and where fd 0 and fd 1 point."""

import os
import sys
from typing import Any

MAX_FD = 256


def _same(a: os.stat_result, b: os.stat_result) -> bool:
    return (a.st_dev, a.st_ino, a.st_rdev) == (b.st_dev, b.st_ino, b.st_rdev)


def _inheritable(fd: int) -> bool:
    try:
        return os.get_inheritable(fd)
    except OSError:
        return False


def load(init: dict[str, Any]) -> object:
    """Return no state."""
    return None


def handle(state: object, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Report the descriptor layout once per item."""
    devnull = os.stat(os.devnull)
    report = {
        "fd0_is_devnull": _same(os.fstat(0), devnull),
        "fd1_is_fd2": _same(os.fstat(1), os.fstat(2)),
        "sys_stdin_is_devnull": _same(os.fstat(sys.stdin.fileno()), devnull),
        "sys_stdout_is_stderr": sys.stdout is sys.stderr,
        "inheritable": [fd for fd in range(MAX_FD) if _inheritable(fd)],
    }
    return [{"outputs": report} for _ in items]
