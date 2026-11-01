"""Faults injected at the operating-system boundary, where the storage itself would raise them.

Each helper wraps one `os` function through `monkeypatch`, so the code under test runs unchanged and
meets the error at the call that real storage fails at.
"""

import errno
import os
import stat
from collections.abc import Callable
from typing import Any

import pytest


def fail_open(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    err: OSError,
    *,
    times: int = 1,
    creating: bool | None = None,
) -> list[str]:
    """Make the next `times` `os.open` calls whose last path component is `name` raise `err`.

    With `creating` True only opens that pass `O_CREAT` count, with False only those that do not.
    Returns the list the failed names are appended to.
    """
    real = os.open
    failed: list[str] = []

    def opened(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:  # noqa: ANN401
        matches = os.path.basename(os.fsdecode(path)) == name
        if creating is not None:
            matches = matches and bool(flags & os.O_CREAT) == creating
        if matches and len(failed) < times:
            failed.append(name)
            raise err
        return real(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", opened)
    return failed


def spy_open(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the last path component of every `os.open`; return the list they are appended to."""
    real = os.open
    names: list[str] = []

    def opened(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:  # noqa: ANN401
        names.append(os.path.basename(os.fsdecode(path)))
        return real(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", opened)
    return names


def fail_fsync(monkeypatch: pytest.MonkeyPatch, when: Callable[[os.stat_result], bool], err: OSError) -> list[int]:
    """Make `os.fsync` raise `err` on every descriptor whose `fstat` satisfies `when`; return the fds it failed."""
    real = os.fsync
    failed: list[int] = []

    def fsync(fd: int) -> None:
        if when(os.fstat(fd)):
            failed.append(fd)
            raise err
        real(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    return failed


def regular_of_size(size: int) -> Callable[[os.stat_result], bool]:
    """A `fail_fsync` condition: a regular file holding exactly `size` bytes."""
    return lambda st: stat.S_ISREG(st.st_mode) and st.st_size == size


def refuse_fchmod(monkeypatch: pytest.MonkeyPatch, inode: int) -> None:
    """Make `os.fchmod` on the file with `inode` raise `EPERM`, as it does on a file another user owns."""
    real = os.fchmod

    def fchmod(fd: int, mode: int) -> None:
        if os.fstat(fd).st_ino == inode:
            raise PermissionError(errno.EPERM, "Operation not permitted")
        real(fd, mode)

    monkeypatch.setattr(os, "fchmod", fchmod)
