"""Faults injected at the operating-system boundary, where the storage itself would raise them.

Each helper wraps one `os` function through `monkeypatch`, so the code under test runs unchanged and
meets the error at the call that real storage fails at.
"""

import errno
import os
import stat
from collections.abc import Callable
from pathlib import Path
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
    """Record the last path component of every `os.open` that does not create; return the list they go to."""
    real = os.open
    names: list[str] = []

    def opened(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:  # noqa: ANN401
        if not flags & os.O_CREAT:
            names.append(os.path.basename(os.fsdecode(path)))
        return real(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", opened)
    return names


def substitute_once(monkeypatch: pytest.MonkeyPatch, name: str, replacement: Path) -> list[str]:
    """Make the first `os.open` of `name` (last path component) open `replacement` instead, as a torn read would.

    Returns the list the substituted name is appended to.
    """
    real = os.open
    done: list[str] = []

    def opened(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:  # noqa: ANN401
        if not done and os.path.basename(os.fsdecode(path)) == name:
            done.append(name)
            return real(replacement, flags)
        return real(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", opened)
    return done


def on_rename(
    monkeypatch: pytest.MonkeyPatch, matches: Callable[[str, str], bool], action: Callable[[], object], *, after: bool
) -> list[object]:
    """Run `action` once, just before (or just `after`) the first `os.rename(old, new)` that `matches`.

    `matches` gets the last path components of `old` and `new`. Renames made by `action` itself pass
    through unchanged. Returns the list `action`'s result is put in.
    """
    real = os.rename
    results: list[object] = []

    def rename(old: Any, new: Any, **dir_fds: Any) -> None:  # noqa: ANN401
        fire = not results and matches(os.path.basename(os.fsdecode(old)), os.path.basename(os.fsdecode(new)))
        if fire and not after:
            results.append(None)
            results[0] = action()
        real(old, new, **dir_fds)
        if fire and after:
            results.append(None)
            results[0] = action()

    monkeypatch.setattr(os, "rename", rename)
    return results


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
