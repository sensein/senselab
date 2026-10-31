"""A file lock that several users can share on one directory (ADR-003).

`SharedFileLock` over a resource `<dir>/<name>` is an exclusive `flock` on `<name>.lock`, with three
sidecars beside it: `<name>.holder` (who holds it), `<name>.heartbeat` (refreshed while held) and
`<name>.clock` (the storage clock). Everything is done relative to the directory's descriptor:

- the lock file is opened with `O_NOFOLLOW` and must be a regular file with one link; anything else at
  its name (a symlink, a hard link, a directory, a fifo) is moved aside beside it;
- the sidecars are only ever replaced whole, by a fresh file renamed over the name
  (`StorageDir.write_atomically`), so a symlink or hard link there is replaced and its target never touched;
- modes are set only through descriptors.

A holder that is not a whole identity is corrupt: it is moved aside and the lock is taken once its age
passes `stale_after`. Every wait is bounded by `timeout` and ends in `LockTimeout`. Ages are read on the
storage's own clock. The lock is correct only where `flock` is coherent across clients (local disks,
NFSv4).
"""

import fcntl
import getpass
import json
import logging
import os
import re
import socket
import stat
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from senselab_core.runtime.storage import TEMPORARY_NAME, StorageDir, WrongKindError, path_component

logger = logging.getLogger("senselab")

LOCK_NAME = re.compile(r"^.+\.lock$")
"""The file a lock `flock`s: `<name>.lock`."""
SIDECAR_NAME = re.compile(r"^.+\.(heartbeat|holder|clock)$")
"""The heartbeat, holder and clock files a lock writes beside the guarded path."""
WRITTEN_NAMES = frozenset({LOCK_NAME, SIDECAR_NAME})
"""The grammars of the file names this module writes (besides `storage.WRITTEN_NAMES`)."""
LOCK_FILE_MODE = 0o664
"""The default mode of the lock file and its sidecars."""
LOCK_DIR_MODE = 0o2775
"""The mode of a lock directory this module manages."""
HOLDER_LIMIT = 1 << 16
"""The most bytes a holder file is read to."""
POLL_S = 0.05
"""Seconds between two attempts to take the lock."""
_HOLDER_FIELDS: dict[str, Any] = {"user": str, "host": str, "pid": int, "taken_at": (int, float)}


class LockTimeout(TimeoutError):
    """A lock wait reached its bound: a holder kept the lock, or a corrupt holder was not yet stale."""


def _holder_record(data: Optional[bytes]) -> Optional[dict[str, Any]]:
    """Return the holder bytes `data` as a JSON object, None if they are absent or not one."""
    if not data or not data.strip():
        return None
    try:
        record = json.loads(data.decode("utf-8"))
    except ValueError:
        return None
    return record if isinstance(record, dict) else None


def _whole(record: Optional[dict[str, Any]]) -> bool:
    """Return whether `record` is a whole holder identity: `user`, `host`, `pid` and `taken_at` of their types."""
    return record is not None and all(
        isinstance(record.get(k), t) and not isinstance(record.get(k), bool) for k, t in _HOLDER_FIELDS.items()
    )


def lock_holder(lock_path: Path) -> Optional[dict]:
    """Return the JSON object recorded in the holder file `lock_path`, or None if none is legible there.

    The file is read through its directory's descriptor without following or blocking. None covers a
    missing file, anything but a regular file, and content that is not a JSON object. It is not evidence
    that the lock is free: mutual exclusion is the `flock`, not this file.
    """
    path = Path(lock_path)
    try:
        with StorageDir.open_root(path.parent) as directory:
            return _holder_record(directory.read_regular(path.name, limit=HOLDER_LIMIT))
    except (OSError, ValueError):
        return None


def _same_file(directory: StorageDir, name: str, fd: int) -> bool:
    """Return whether the file open on `fd` is the regular file now at `directory/name` (device and inode).

    The name is opened and `fstat`ed, never `stat`ed, so NFS revalidates it (close-to-open).
    """
    held = os.fstat(fd)
    try:
        probe = directory.open_regular(name)
    except (OSError, WrongKindError):
        return False
    if probe is None:
        return False
    try:
        current = os.fstat(probe)
    finally:
        os.close(probe)
    return (current.st_dev, current.st_ino) == (held.st_dev, held.st_ino)


def lock_fd_matches_path(fd: int, path: "os.PathLike[str] | str") -> bool:
    """Return whether the file locked through `fd` is the regular file now at `path` (same device and inode)."""
    target = Path(path)
    try:
        with StorageDir.open_root(target.parent) as directory:
            return _same_file(directory, target.name, fd)
    except OSError:
        return False


class SharedFileLock:
    """A cross-user file lock over the resource `path`, with a heartbeat that stale detection reads.

    The lock file is `path` with `.lock` appended and the sidecars `.holder`, `.heartbeat` and `.clock`
    likewise (appended, so dotted resource names never collide). The module docstring says how each is
    written.

    Args:
        path: The resource being guarded (not the lock file itself).
        timeout: Seconds any wait in `__enter__` may take before `LockTimeout`.
        heartbeat_interval: Seconds between heartbeat refreshes while held.
        stale_after: Seconds of heartbeat silence after which a previous holder's leftovers count as stale.
        manage_dir_mode: Whether the lock's directory is made (or set, if this user owns it) `LOCK_DIR_MODE`;
            False leaves a caller's directory as it is and makes a missing one under the umask.
        file_mode: The mode of the lock file and sidecars this lock creates.
        directory: The directory holding `path`, already open; it is duplicated and `path`'s parent is not
            opened again.

    Attributes:
        took_over: Whether, once the `flock` was held, a previous holder's identity (whole or corrupt) was
            still there: a holder that never released.
        set_aside: The paths this lock moved aside: objects of the wrong kind at its names, corrupt holders.
    """

    def __init__(
        self,
        path: Path,
        *,
        timeout: float = 600.0,
        heartbeat_interval: float = 30.0,
        stale_after: float = 120.0,
        manage_dir_mode: bool = True,
        file_mode: int = LOCK_FILE_MODE,
        directory: Optional[StorageDir] = None,
    ) -> None:
        """Configure a lock over `path` without acquiring it."""
        self._path = Path(path)
        name = path_component(self._path.name)
        self._lock_name = name + ".lock"
        self._holder_name = name + ".holder"
        self._heartbeat_name = name + ".heartbeat"
        self._clock_name = name + ".clock"
        self._lock_path = self._path.with_name(self._lock_name)
        self._heartbeat_path = self._path.with_name(self._heartbeat_name)
        self._holder_path = self._path.with_name(self._holder_name)
        self._clock_path = self._path.with_name(self._clock_name)
        self._timeout = timeout
        self._heartbeat_interval = heartbeat_interval
        self._stale_after = stale_after
        self._manage_dir_mode = manage_dir_mode
        self._file_mode = file_mode
        self._given = directory
        self._dir: Optional[StorageDir] = None
        self._fd: Optional[int] = None
        self._stop_event = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._token: Optional[str] = None
        self.took_over = False
        self.set_aside: list[Path] = []

    def _open_dir(self) -> StorageDir:
        if self._given is not None:
            return self._given.dup()
        parent = self._path.parent
        if not self._manage_dir_mode:
            return StorageDir.open_root(parent, create=True)
        directory = StorageDir.open_root(parent, create=True, mode=LOCK_DIR_MODE)
        if os.fstat(directory.fd).st_uid == os.geteuid():
            try:
                directory.set_mode(LOCK_DIR_MODE)
            except OSError:
                pass
        return directory

    def _directory(self) -> StorageDir:
        if self._dir is None:
            self._dir = self._open_dir()
        return self._dir

    def _move_aside(self, name: str, why: str) -> None:
        moved = self._directory().set_aside(name, None)
        self.set_aside.append(moved)
        logger.warning("lock %s: moved %s aside to %s (%s)", self._lock_path, name, moved, why)

    def _open_lock(self) -> Optional[int]:
        """Return a descriptor on the lock file, made if missing; None after moving a wrong object aside."""
        directory = self._directory()
        fd, failure = self._open_lock_fd(directory)
        if failure is not None:
            kind = directory.kind(self._lock_name)
            if kind in (None, "file"):
                raise failure
            self._move_aside(self._lock_name, f"a {kind}")
            return None
        if fd is None:
            return None
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
            os.close(fd)
            self._move_aside(self._lock_name, "a hard link" if stat.S_ISREG(st.st_mode) else "not a regular file")
            return None
        return fd

    def _open_lock_fd(self, directory: StorageDir) -> tuple[Optional[int], Optional[OSError]]:
        """Open the lock file, creating it if missing: `(descriptor, error)`; both None if it vanished."""
        try:
            return directory.open_or_create(self._lock_name, self._file_mode, grammar=LOCK_NAME)[0], None
        except FileNotFoundError:
            return None, None
        except OSError as err:
            return None, err

    def _try_lock(self) -> Optional[int]:
        fd = self._open_lock()
        if fd is None:
            return None
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return None
        if not _same_file(self._directory(), self._lock_name, fd):
            os.close(fd)
            return None
        return fd

    def _now(self) -> float:
        return self._directory().fs_now(self._clock_name)

    def _heartbeat_age(self) -> Optional[float]:
        """Return seconds since the heartbeat was last written, on the storage's clock; None if it cannot be read."""
        try:
            beat = self._directory().lstat(self._heartbeat_name)
            if beat is None:
                return None
            return self._now() - beat.st_mtime
        except (OSError, WrongKindError):
            return None

    def _staleness(self, holder: dict) -> tuple[float, str, float]:
        """Return `(age, basis, bound)` for a whole holder identity left behind.

        The heartbeat's age against `stale_after` when it can be read, otherwise how long ago the holder
        recorded taking the lock against `timeout`.
        """
        hb_age = self._heartbeat_age()
        if hb_age is not None:
            return hb_age, "heartbeat", self._stale_after
        return max(0.0, time.time() - float(holder["taken_at"])), "taken_at", self._timeout

    def _write_holder(self) -> None:
        self._token = uuid.uuid4().hex
        payload = {
            "user": getpass.getuser(),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "taken_at": time.time(),
            "token": self._token,
        }
        self._directory().write_atomically(
            self._holder_name, json.dumps(payload).encode("utf-8"), grammar=SIDECAR_NAME, mode=self._file_mode
        )

    def _beat(self) -> None:
        self._directory().write_atomically(self._heartbeat_name, b"", grammar=SIDECAR_NAME, mode=self._file_mode)

    def _read_holder(self) -> tuple[str, Optional[dict[str, Any]], float]:
        """Return `("absent" | "whole" | "corrupt", record, mtime)` for the holder file."""
        directory = self._directory()
        info = directory.lstat(self._holder_name)
        if info is None:
            return "absent", None, 0.0
        try:
            data = directory.read_regular(self._holder_name, limit=HOLDER_LIMIT)
        except (WrongKindError, ValueError):
            data = b""
        record = _holder_record(data)
        return ("whole" if _whole(record) else "corrupt"), record, info.st_mtime

    def owns(self) -> bool:
        """Return whether this instance is still the identity recorded in the holder file."""
        current = lock_holder(self._holder_path)
        return current is not None and self._token is not None and current.get("token") == self._token

    def _warn_stale_takeover(self, holder: dict, age: float, basis: str, bound: float) -> None:
        logger.warning(
            "Stale lock at %s: %s is %.1fs old (> %s threshold %.1fs). "
            "Previous holder was user=%s host=%s pid=%s. Breaking lock and taking over.",
            self._lock_path,
            "heartbeat" if basis == "heartbeat" else "time since previous holder recorded acquiring the lock",
            age,
            basis,
            bound,
            holder.get("user"),
            holder.get("host"),
            holder.get("pid"),
        )

    def _timeout_message(self) -> str:
        holder = lock_holder(self._holder_path)
        age = self._heartbeat_age()
        age_display = f"{age:.1f}s old" if age is not None else "unreadable"
        if holder is None:
            return f"held by an unknown process (heartbeat {age_display}, no identity on disk)"
        taken_at = holder.get("taken_at")
        held_for = f"{time.time() - taken_at:.1f}s" if isinstance(taken_at, (int, float)) else "an unknown time"
        return (
            f"held by user={holder.get('user')} host={holder.get('host')} pid={holder.get('pid')} "
            f"for {held_for} (heartbeat {age_display})"
        )

    def _acquire(self, deadline: float) -> int:
        while True:
            fd = self._try_lock()
            if fd is not None:
                return fd
            if time.monotonic() >= deadline:
                raise LockTimeout(
                    f"Timed out after {self._timeout:.1f}s waiting for lock at {self._lock_path}: "
                    f"{self._timeout_message()}. Another process held its flock for the whole wait, so it is "
                    "not broken automatically."
                )
            time.sleep(min(POLL_S, max(0.0, deadline - time.monotonic())))

    def _remove_dead_temporaries(self) -> None:
        """Remove sidecar temporaries a dead holder left; only a holder writes them, and this one holds the flock."""
        directory = self._directory()
        prefixes = tuple(f".{n}." for n in (self._holder_name, self._heartbeat_name, self._clock_name))
        for name in directory.listdir():
            if name.startswith(prefixes) and TEMPORARY_NAME.fullmatch(name):
                directory.remove_owned(name, pattern=TEMPORARY_NAME)

    def _settle_leftover(self, deadline: float) -> None:
        """Judge what a previous holder left, once the `flock` is held."""
        self._remove_dead_temporaries()
        state, record, mtime = self._read_holder()
        if state == "absent":
            return
        self.took_over = True
        if state == "whole" and record is not None:
            age, basis, bound = self._staleness(record)
            if age > bound:
                self._warn_stale_takeover(record, age, basis, bound)
            return
        self._move_aside(self._holder_name, "not a whole holder identity")
        while True:
            beat = self._directory().lstat(self._heartbeat_name)
            age = self._now() - max(mtime, beat.st_mtime if beat is not None else 0.0)
            if age > self._stale_after:
                logger.warning("lock %s: taking over after a corrupt holder %.1fs old", self._lock_path, age)
                return
            if time.monotonic() >= deadline:
                raise LockTimeout(
                    f"Timed out after {self._timeout:.1f}s at {self._lock_path}: a corrupt holder was moved aside "
                    f"and its last sign of life is {age:.1f}s old, under the {self._stale_after:.1f}s stale bound"
                )
            time.sleep(min(POLL_S, max(0.0, deadline - time.monotonic())))

    def __enter__(self) -> "SharedFileLock":
        """Acquire the lock, taking over a dead holder's leftovers.

        Returns:
            This instance, for use as a context manager.

        Raises:
            LockTimeout: the lock was held for the whole `timeout`, or a corrupt holder did not become
                stale within it.
            WrongKindError: an object of the wrong kind at one of the lock's names cannot be moved aside.
        """
        self.took_over = False
        self.set_aside = []
        deadline = time.monotonic() + self._timeout
        try:
            self._fd = self._acquire(deadline)
            self._settle_leftover(deadline)
            self._write_holder()
            self._beat()
        except BaseException:
            self._unlock()
            raise
        self._stop_event.clear()
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()
        return self

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self._heartbeat_interval):
            try:
                self._beat()
            except (OSError, WrongKindError, ValueError) as err:
                logger.warning("lock %s: cannot refresh the heartbeat: %s", self._lock_path, err)

    def _unlock(self) -> None:
        """Release the `flock` and close the descriptors; the sidecars stay, so the next holder sees them."""
        directory, fd = self._dir, self._fd
        self._dir, self._fd = None, None
        try:
            if fd is not None:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    os.close(fd)
        finally:
            if directory is not None:
                directory.close()

    def _remove_sidecars(self) -> None:
        if self._dir is None or self._fd is None:
            return
        for name in (self._heartbeat_name, self._holder_name, self._clock_name):
            try:
                self._dir.remove_owned(name, pattern=SIDECAR_NAME)
            except OSError as err:
                logger.warning("lock %s: cannot remove %s: %s", self._lock_path, name, err)

    def __exit__(self, *exc: object) -> None:
        """Stop the heartbeat, remove the heartbeat, holder and clock sidecars, and release the lock."""
        self._stop_event.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=5)
            self._heartbeat_thread = None
        try:
            self._remove_sidecars()
        finally:
            self._unlock()


__all__ = [
    "HOLDER_LIMIT",
    "LOCK_DIR_MODE",
    "LOCK_FILE_MODE",
    "LOCK_NAME",
    "SIDECAR_NAME",
    "WRITTEN_NAMES",
    "LockTimeout",
    "SharedFileLock",
    "lock_fd_matches_path",
    "lock_holder",
]
