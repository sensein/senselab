"""A file lock that several users can share on one directory.

Two invisible assumptions:

- **Staleness is a local-clock comparison.** ``stale_after`` is checked as
  ``time.time() - heartbeat_mtime``: ``time.time()`` is this host's clock,
  ``heartbeat_mtime`` was set by whichever host holds the lock. On a cluster the
  two are different machines, so ``stale_after`` must exceed any plausible clock
  skew between nodes — set it too tight and a live holder on a fast/behind clock
  gets misread as dead and taken over mid-work.
- **This still relies on ``fcntl`` semantics underneath.** ``filelock`` uses
  ``fcntl.flock``/``lockf`` on POSIX, which is only correctly serialising when
  the underlying filesystem implements those locks coherently across clients —
  true for local disks and NFSv4 (or equivalent), not guaranteed for NFSv3 or
  other network filesystems that treat locking as advisory-only or per-client.
  On such a filesystem two hosts can both believe they hold the lock.
"""

import getpass
import json
import logging
import os
import socket
import threading
import time
from pathlib import Path
from typing import Optional

from filelock import FileLock, Timeout

logger = logging.getLogger("senselab")

# rw-rw-r--: owner and group can both write the lock/heartbeat files, so a
# second user's heartbeat touch (and lock takeover) does not fail silently.
LOCK_FILE_MODE = 0o664
# rwxrwsr-x with the setgid bit: new files/directories created underneath
# inherit the parent directory's group, so the *next* user's files land in the
# same group as the first, not their own primary group.
LOCK_DIR_MODE = 0o2775


def _ensure_dir(path: Path) -> None:
    """Create ``path`` (with parents) and force it group-writable and setgid.

    A failed ``chmod`` is ignored: on a shared tree the directory may already
    belong to another user with the mode already correct, and raising here
    would break exactly the multi-user case this module exists for.
    """
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, LOCK_DIR_MODE)
    except OSError:
        pass


def _touch_shared(path: Path) -> None:
    """Create ``path`` if it does not exist, then force it group-writable.

    Used for both the lock file and the heartbeat file. As with
    :func:`_ensure_dir`, a failed ``chmod`` is ignored rather than raised.
    """
    path.touch(exist_ok=True)
    try:
        os.chmod(path, LOCK_FILE_MODE)
    except OSError:
        pass


def lock_holder(lock_path: Path) -> Optional[dict]:
    """Return the identity recorded in ``lock_path``, or ``None`` if unheld.

    ``None`` covers a missing file, an empty file (the window between
    ``_touch_shared`` creating it and the holder's payload write landing), and
    a file that fails to parse as JSON — a partially-written read must never
    raise, only report "unheld".

    Args:
        lock_path: Path to the ``.lock`` file to inspect.

    Returns:
        The decoded holder payload (``user``, ``host``, ``pid``, ``taken_at``),
        or ``None`` if the file does not name a current holder.
    """
    try:
        text = lock_path.read_text()
    except OSError:
        return None
    if not text.strip():
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


class SharedFileLock:
    """A cross-user file lock with a heartbeat that stale-detection actually reads.

    ``path`` names the resource being guarded; the lock itself lives at
    ``path.with_suffix(".lock")`` and the heartbeat at
    ``path.with_suffix(".heartbeat")``, matching the convention both prior
    lock implementations (``dependencies.py``, ``subprocess_venv.py``) used.
    """

    def __init__(
        self,
        path: Path,
        *,
        timeout: float = 600.0,
        heartbeat_interval: float = 30.0,
        stale_after: float = 120.0,
    ) -> None:
        """Configure a lock over ``path`` without acquiring it.

        Args:
            path: The resource being guarded (not the lock file itself).
            timeout: Seconds to wait for the underlying ``filelock`` before
                checking whether the current holder's heartbeat is stale.
                Default 600s matches ``subprocess_venv``'s longest existing
                lock use (a model/venv install), which can legitimately take
                minutes.
            heartbeat_interval: Seconds between heartbeat touches while held.
                Default 30s: frequent enough that ``stale_after``'s default
                (4x this) tolerates a couple of missed beats from scheduler
                jitter without false-negatives, infrequent enough not to be a
                meaningful I/O load on a network filesystem.
            stale_after: Seconds since the last heartbeat touch after which a
                holder is presumed dead and taken over. Default 120s (4x
                ``heartbeat_interval``) leaves headroom for both a couple of
                missed beats and plausible cross-node clock skew (see the
                module docstring) without waiting for the full ``timeout``.
        """
        self._path = path
        self._lock_path = path.with_suffix(".lock")
        self._heartbeat_path = path.with_suffix(".heartbeat")
        self._timeout = timeout
        self._heartbeat_interval = heartbeat_interval
        self._stale_after = stale_after
        self._lock = FileLock(str(self._lock_path))
        self._stop_event = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self._heartbeat_interval):
            try:
                self._heartbeat_path.touch()
            except OSError:
                # Swallowed deliberately, but this is exactly the failure mode
                # this module exists to fix: if the heartbeat file is not
                # group-writable, a permission error here silently stops a
                # *live* holder's heartbeat from refreshing, and the next
                # waiter reads it as stale and breaks a lock that is very
                # much alive. LOCK_FILE_MODE is what prevents that in
                # practice; this except is only a backstop against unrelated
                # I/O errors (e.g. the underlying directory being removed).
                pass

    def _heartbeat_age(self) -> float:
        try:
            return time.time() - self._heartbeat_path.stat().st_mtime
        except OSError:
            # No heartbeat file at all reads as infinitely stale.
            return float("inf")

    def _write_holder(self) -> None:
        payload = {
            "user": getpass.getuser(),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "taken_at": time.time(),
        }
        self._lock_path.write_text(json.dumps(payload))

    def _warn_stale_takeover(self, holder: Optional[dict], age: float) -> None:
        """Log the "we are taking over a dead holder's lock" warning.

        Named identity (user/host/pid) is what lets someone on a cluster check
        whether the job that held this lock actually died, rather than just
        being told "stale lock detected".
        """
        if holder is not None:
            logger.warning(
                "Stale lock at %s: heartbeat is %.1fs old (> stale_after=%.1fs). "
                "Previous holder was user=%s host=%s pid=%s. Breaking lock and taking over.",
                self._lock_path,
                age,
                self._stale_after,
                holder.get("user"),
                holder.get("host"),
                holder.get("pid"),
            )
        else:
            logger.warning(
                "Stale lock at %s: heartbeat is %.1fs old (> stale_after=%.1fs), "
                "and no holder identity could be read. Breaking lock and taking over.",
                self._lock_path,
                age,
                self._stale_after,
            )

    def __enter__(self) -> "SharedFileLock":
        """Acquire the lock, taking over a stale holder if one is found.

        Returns:
            This instance, for use as a context manager.

        Raises:
            TimeoutError: The lock is held and its heartbeat is still fresh.
        """
        _ensure_dir(self._lock_path.parent)
        _touch_shared(self._lock_path)
        # Read whatever identity is on disk *before* we touch anything else.
        # A crash releases the OS-level flock immediately (the kernel drops it
        # when the holding process exits), but leaves this recorded identity
        # and heartbeat behind — so the far more common "stale lock" case
        # below is an *uncontested* acquire that still needs to report a
        # takeover, not a `filelock.Timeout`.
        previous_holder: Optional[dict] = lock_holder(self._lock_path)
        try:
            self._lock.acquire(timeout=self._timeout)
        except Timeout:
            age = self._heartbeat_age()
            if age <= self._stale_after:
                raise TimeoutError(
                    f"Lock at {self._lock_path} is held and its heartbeat is fresh ({age:.1f}s old); not taking over."
                ) from None
            self._warn_stale_takeover(previous_holder, age)
            self._lock_path.unlink(missing_ok=True)
            _touch_shared(self._lock_path)
            self._lock.acquire(timeout=self._timeout)
        else:
            if previous_holder is not None:
                age = self._heartbeat_age()
                if age > self._stale_after:
                    self._warn_stale_takeover(previous_holder, age)

        self._write_holder()
        # The lock file's mode survives write_text (it rewrites content, not the
        # inode), but re-assert it: some filesystems/implementations recreate
        # the inode on write, which would silently drop the group-writable bit.
        try:
            os.chmod(self._lock_path, LOCK_FILE_MODE)
        except OSError:
            pass
        self._stop_event.clear()
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()
        _touch_shared(self._heartbeat_path)
        return self

    def __exit__(self, *exc: object) -> None:
        """Stop the heartbeat, clear holder identity, and release the lock."""
        self._stop_event.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=5)
        self._heartbeat_path.unlink(missing_ok=True)
        try:
            # Truncate rather than unlink: the file (and its group-writable
            # mode) stays in place for the next acquirer's _touch_shared to
            # find already correctly permissioned; only the holder payload
            # is cleared so lock_holder() reports unheld.
            self._lock_path.write_text("")
        except OSError:
            pass
        self._lock.release()
