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
    """Return the identity recorded in ``lock_path``, if one is legible there.

    **``None`` does not mean the lock is free.** This reads file *content*,
    which is advisory; mutual exclusion is governed by the ``flock`` on the
    descriptor, and the two disagree under contention. ``filelock``'s
    ``UnixFileLock._acquire`` opens the lock file with ``O_TRUNC`` on *every*
    poll attempt including the ones that fail to take the lock, so a waiter's
    own polling erases the live holder's payload within one poll interval.
    Never use this as a liveness probe: use it for logging, and to capture the
    holder's identity **once, before** any contended acquire begins — which is
    what ``SharedFileLock.__enter__`` does, and why its timeout message can
    still name a holder whose payload it is about to wipe.

    ``None`` therefore covers: a missing file, a file truncated by a
    contender's failed poll, the window between ``_touch_shared`` creating the
    file and the holder's payload write landing, and a file that fails to parse
    as JSON. A partially-written read must never raise, only report ``None``.

    Args:
        lock_path: Path to the ``.lock`` file to inspect.

    Returns:
        The decoded holder payload (``user``, ``host``, ``pid``, ``taken_at``),
        or ``None`` if no identity is currently legible in the file — which is
        not evidence that the lock is unheld (see above).
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
    ``path`` with ``.lock`` appended and the heartbeat at ``path`` with
    ``.heartbeat`` appended -- string concatenation, not ``Path.with_suffix``.

    ``with_suffix`` *replaces* everything from the resource name's last dot
    onward rather than appending after it, and both prior lock implementations
    (``dependencies.py``, ``subprocess_venv.py``) guard resources whose names
    can legitimately contain a dot -- a HuggingFace revision or a venv path,
    for instance. Two distinct resources differing only after such a dot, e.g.
    ``org--model--v1.5--main`` and ``org--model--v1.6--main``, both reduce
    under ``with_suffix(".lock")`` to the identical ``org--model--v1.lock``:
    silently merging two callers' locks onto one file, each unaware the other
    exists. Concatenation (``Path(str(path) + ".lock")``) is injective -- two
    distinct ``path`` values can never produce the same lock file -- so no
    caller needs to invent its own workaround (an earlier version of
    ``dependencies.py`` did, appending a synthetic no-dot marker before
    calling this class; that workaround is gone now that the primitive itself
    cannot collide).
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
        # Append, don't replace -- see the class docstring for the concrete collision
        # (two dotted resource names reducing to the same path) that using
        # Path.with_suffix here used to produce.
        self._lock_path = Path(str(path) + ".lock")
        self._heartbeat_path = Path(str(path) + ".heartbeat")
        self._timeout = timeout
        self._heartbeat_interval = heartbeat_interval
        self._stale_after = stale_after
        self._lock = FileLock(str(self._lock_path))
        self._stop_event = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self._heartbeat_interval):
            try:
                _touch_shared(self._heartbeat_path)
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
        """Acquire the lock, taking over a dead holder's leftovers if found.

        The two branches below are deliberately asymmetric, and that
        asymmetry is the safety property this class provides:

        - **Uncontended acquire, stale identity on disk** (the ``else``
          branch): the OS-level ``flock`` was free — a crashed process's
          ``flock`` is released by the kernel the instant it exits — so what
          is left behind is a leftover: a lock file naming a dead holder.
          Safe to clean up and log.
        - **``filelock.Timeout``** (the ``except`` branch): reaching this
          proves the ``flock`` was held *continuously* for the entire
          ``timeout`` window. A crashed process cannot do that — so a timeout
          always means a live process holds the lock, no matter how stale its
          heartbeat looks (heartbeat writes can stall or die independently of
          the holder, e.g. an uncaught exception in the heartbeat thread, or
          the thread simply not being scheduled under load — the holder's
          real work continues regardless). Unlinking here would hand a new
          claimant a lock on a **fresh inode** while the original holder still
          holds the ``flock`` on the now-orphaned one: both then believe they
          hold the lock, which is precisely the concurrent clobber this class
          exists to prevent. So this branch never unlinks or retries; it only
          raises, naming the holder so a human can decide whether to kill it.

        Returns:
            This instance, for use as a context manager.

        Raises:
            TimeoutError: The lock is held by a live process (see above).
        """
        _ensure_dir(self._lock_path.parent)
        _touch_shared(self._lock_path)
        # Read whatever identity is on disk *before* we touch anything else,
        # for both branches below: the uncontended-acquire check needs it to
        # decide whether to log a takeover, and the Timeout branch needs it
        # purely to name the current holder in the error message.
        previous_holder: Optional[dict] = lock_holder(self._lock_path)
        try:
            self._lock.acquire(timeout=self._timeout)
        except Timeout:
            age = self._heartbeat_age()
            if previous_holder is not None:
                # `.get`, not `[...]`, like every neighbouring field: lock_holder returns whatever
                # JSON is on disk, including a holder written by an older senselab that predates
                # `taken_at` (or a hand-edited/partial file that still parses). A bare subscript
                # would raise KeyError (or TypeError on a non-numeric value) out of __enter__, and
                # the retry loops in ensure_hf_model / ensure_venv / record_resolution only catch
                # TimeoutError — so it would propagate as an unrelated crash instead of a timeout.
                taken_at = previous_holder.get("taken_at")
                held_for = f"{time.time() - taken_at:.1f}s" if isinstance(taken_at, (int, float)) else "an unknown time"
                detail = (
                    f"held by user={previous_holder.get('user')} host={previous_holder.get('host')} "
                    f"pid={previous_holder.get('pid')} for {held_for} "
                    f"(heartbeat {age:.1f}s old)"
                )
            else:
                detail = f"held by an unknown process (heartbeat {age:.1f}s old, no identity on disk)"
            raise TimeoutError(
                f"Timed out after {self._timeout:.1f}s waiting for lock at {self._lock_path}: {detail}. "
                "The OS-level lock was held for the entire wait, which a crashed process cannot do, so this "
                "is a live holder even though its heartbeat may look stale -- it is not broken automatically. "
                "If that process is confirmed dead, remove the .lock file by hand."
            ) from None
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
            # `filelock.UnixFileLock._release()` (called below) unconditionally
            # attempts to unlink the lock file itself, so this content does not
            # normally survive past this call -- the next acquirer's
            # `_touch_shared` recreates the file from nothing, not from what we
            # leave here. Truncating first is a backstop for the case where
            # that unlink silently fails (`_release()` suppresses `OSError`,
            # e.g. a permission or network hiccup on a shared tree): even then,
            # `lock_holder()` reads the leftover file as unheld rather than
            # reporting our identity indefinitely.
            self._lock_path.write_text("")
        except OSError:
            pass
        self._lock.release()
