"""Faults injected at the operating-system boundary, where the storage itself would raise them.

Each helper wraps one `os` function through `monkeypatch`, so the code under test runs unchanged and
meets the error at the call that real storage fails at.
"""

import errno
import io
import os
import shutil
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import pytest

Step = Literal["open", "read", "write", "fsync"]
"""Where `fail_at` makes a file's I/O fail."""


def fail_at(  # noqa: C901, PLR0913 - one patcher over the five syscalls; the keyword-only options are the cases
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    at: Step,
    err: OSError,
    *,
    times: int = 1,
    creating: bool | None = None,
) -> list[str]:
    """Make the next `times` `at` steps on a file whose last path component is `name` raise `err`.

    `open` fails the `os.open` itself. `read` and `write` fail the first read or write that reaches the
    descriptor, through a file object `os.fdopen` returns over it (duplicates included). `fsync` fails
    `os.fsync` on it. With `creating` True only opens that pass `O_CREAT` count, with False only those
    that do not. Returns the list each failure appends `name` to.
    """
    real_open, real_fdopen, real_fsync = os.open, os.fdopen, os.fsync
    inodes: set[tuple[int, int]] = set()
    failed: list[str] = []

    def due() -> bool:
        return len(failed) < times

    def fail() -> None:
        failed.append(name)
        raise err

    def ours(fd: int) -> bool:
        st = os.fstat(fd)
        return (st.st_dev, st.st_ino) in inodes

    def opened(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:  # noqa: ANN401
        matches = os.path.basename(os.fsdecode(path)) == name
        if creating is not None:
            matches = matches and bool(flags & os.O_CREAT) == creating
        if matches and at == "open" and due():
            fail()
        fd = real_open(path, flags, *args, **kwargs)
        if matches:
            st = os.fstat(fd)
            inodes.add((st.st_dev, st.st_ino))
        return fd

    class FaultyIO(io.FileIO):
        def readinto(self, buffer: Any) -> int | None:  # noqa: ANN401
            if at == "read" and due():
                fail()
            return super().readinto(buffer)

        def readall(self) -> bytes:
            if at == "read" and due():
                fail()
            return super().readall()

        def read(self, size: int | None = -1) -> bytes:
            if at == "read" and due():
                fail()
            return super().read(size)

        def write(self, data: Any) -> int:  # noqa: ANN401
            if at == "write" and due():
                fail()
            return super().write(data)

    def fdopen(fd: int, mode: str = "r", *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        if at not in ("read", "write") or "b" not in mode or not ours(fd):
            return real_fdopen(fd, mode, *args, **kwargs)
        raw = FaultyIO(fd, mode.replace("b", ""), closefd=True)
        if "+" in mode:
            return io.BufferedRandom(raw)
        return io.BufferedReader(raw) if "r" in mode else io.BufferedWriter(raw)

    def fsync(fd: int) -> None:
        if at == "fsync" and due() and ours(fd):
            fail()
        real_fsync(fd)

    monkeypatch.setattr(os, "open", opened)
    monkeypatch.setattr(os, "fdopen", fdopen)
    monkeypatch.setattr(os, "fsync", fsync)
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


def refuse_fchmod(monkeypatch: pytest.MonkeyPatch, inode: int) -> None:
    """Make `os.fchmod` on the file with `inode` raise `EPERM`, as it does on a file another user owns."""
    real = os.fchmod

    def fchmod(fd: int, mode: int) -> None:
        if os.fstat(fd).st_ino == inode:
            raise PermissionError(errno.EPERM, "Operation not permitted")
        real(fd, mode)

    monkeypatch.setattr(os, "fchmod", fchmod)


def pause_replaces(
    monkeypatch: pytest.MonkeyPatch, new: str, count: int
) -> list[tuple[threading.Event, threading.Event]]:
    """Hold each of the first `count` `os.replace` calls onto a name `new` (last path component) before it runs.

    The i-th such call sets the first event of the i-th pair and waits (60 s at most) for the second.
    """
    real = os.replace
    gates = [(threading.Event(), threading.Event()) for _ in range(count)]
    seen: list[str] = []
    lock = threading.Lock()

    def replace(old: Any, dst: Any, **dir_fds: Any) -> None:  # noqa: ANN401
        if os.path.basename(os.fsdecode(dst)) == new:
            with lock:
                index = len(seen)
                seen.append(new)
            if index < count:
                reached, go = gates[index]
                reached.set()
                assert go.wait(60)
        real(old, dst, **dir_fds)

    monkeypatch.setattr(os, "replace", replace)
    return gates


def displace_after_take(monkeypatch: pytest.MonkeyPatch, name: str) -> list[str]:
    """Remove the lease directory `name` at the first file created after each take of it, as a breaker would.

    A take is the `os.mkdir` of `name` and then of a token in it; the removal renames `name` aside in its parent
    and removes the tree. Returns the list each displacement appends `name` to.
    """
    real_mkdir, real_open = os.mkdir, os.open
    parent: list[int] = []
    displaced: list[str] = []

    def mkdir(path: Any, mode: int = 0o777, *, dir_fd: int | None = None) -> None:  # noqa: ANN401
        real_mkdir(path, mode, dir_fd=dir_fd)
        if os.fsdecode(path) == name and dir_fd is not None:
            parent[:] = [dir_fd]

    def opened(path: Any, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:  # noqa: ANN401
        if parent and flags & os.O_CREAT:
            aside = uuid.uuid4().hex
            os.rename(name, aside, src_dir_fd=parent[0], dst_dir_fd=parent[0])
            shutil.rmtree(aside, dir_fd=parent.pop())
            displaced.append(name)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "mkdir", mkdir)
    monkeypatch.setattr(os, "open", opened)
    return displaced


def displace_before_proof(monkeypatch: pytest.MonkeyPatch, name: str) -> list[str]:
    """Remove the lease directory just before each `Lease.prove` makes its proof directory, as a breaker would.

    The lease is the last directory made whose name ends with `name`. Returns the list each displacement appends the
    lease's name to.
    """
    real_mkdir = os.mkdir
    parent: list[tuple[int, str]] = []
    displaced: list[str] = []

    def mkdir(path: Any, mode: int = 0o777, *, dir_fd: int | None = None) -> None:  # noqa: ANN401
        if os.fsdecode(path).startswith("proof-") and parent:
            fd, lease = parent[0]
            aside = uuid.uuid4().hex
            os.rename(lease, aside, src_dir_fd=fd, dst_dir_fd=fd)
            shutil.rmtree(aside, dir_fd=fd)
            displaced.append(lease)
        real_mkdir(path, mode, dir_fd=dir_fd)
        if os.fsdecode(path).endswith(name) and dir_fd is not None:
            parent[:] = [(dir_fd, os.fsdecode(path))]

    monkeypatch.setattr(os, "mkdir", mkdir)
    return displaced


COUNTED = (
    *("open", "close", "read", "write", "mkdir", "rmdir", "rename", "replace", "link", "symlink", "unlink"),
    *("lstat", "stat", "fstat", "listdir", "scandir", "readlink", "utime", "fsync", "fchmod", "dup"),
)
"""The `os` functions `count_calls` counts."""


def count_calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Count calls to the `COUNTED` functions and the bytes they return; `calls` and `bytes` keep the totals.

    The bytes are those `read` returns, the target `readlink` returns and the names `listdir` returns.
    """
    totals = {"calls": 0, "bytes": 0}

    def size(result: object) -> int:
        if isinstance(result, (bytes, str)):
            return len(result)
        if isinstance(result, list):
            return sum(len(n) for n in result if isinstance(n, (bytes, str)))
        return 0

    def wrap(name: str, real: Callable[..., Any]) -> Callable[..., Any]:
        def counted(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            result = real(*args, **kwargs)
            totals["calls"] += 1
            if name in ("read", "readlink", "listdir"):
                totals["bytes"] += size(result)
            return result

        return counted

    for name in COUNTED:
        if hasattr(os, name):
            monkeypatch.setattr(os, name, wrap(name, getattr(os, name)))
    return totals


def stall(function: str, prefix: str, seconds: float, *, times: int | None = None, after: bool = False) -> None:
    """In this process, make `os.<function>` on a name starting with `prefix` sleep `seconds`, without `monkeypatch`.

    The sleep comes before the call, or after it with `after`; only the first `times` matching calls sleep (all with
    None). For a spawned child, which ends with its patch.
    """
    real = getattr(os, function)
    left = [times]

    def stalled(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        names = [os.path.basename(os.fsdecode(a)) for a in args[:2] if isinstance(a, (str, bytes, os.PathLike))]
        due = any(n.startswith(prefix) for n in names) and left[0] != 0
        if due and left[0] is not None:
            left[0] -= 1
        if due and not after:
            time.sleep(seconds)
        result = real(*args, **kwargs)
        if due and after:
            time.sleep(seconds)
        return result

    setattr(os, function, stalled)


def exit_at(function: str, prefix: str) -> None:
    """In this process, make `os.<function>` on a name starting with `prefix` end the process at once, as a crash."""
    real = getattr(os, function)

    def crashed(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        names = [os.path.basename(os.fsdecode(a)) for a in args[:2] if isinstance(a, (str, bytes, os.PathLike))]
        if any(n.startswith(prefix) for n in names):
            os._exit(9)
        return real(*args, **kwargs)

    setattr(os, function, crashed)
