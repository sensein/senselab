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
import uuid
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


def _ensure_dir(path: Path, *, manage_mode: bool = True) -> None:
    """Create ``path`` (with parents); when ``manage_mode`` force it group-writable and setgid.

    ``manage_mode`` is True for senselab's own cache dirs, where the shared-tree modes are the
    whole point. It is False when the lock guards a caller-supplied path — e.g. a ``FileRef`` over
    an input file sitting in the caller's *own* private directory.

    The harm this guards against is self-inflicted, not inflicted on a stranger. ``chmod(2)``
    returns ``EPERM`` unless the effective UID owns the directory, and the ``except OSError``
    below swallows that — so a directory belonging to another user is never modified and never
    raises. The chmod only lands on directories the invoking user owns, and there
    ``LOCK_DIR_MODE`` is a widening: measured, a ``0o700`` directory comes back ``0o2775``, which
    is not merely setgid + group-write but also **other-read and other-execute** — world traversal
    of a directory its owner deliberately made private, as a side effect of dropping a ``.lock``
    file in it.

    Skipping the chmod also stops the lock from defeating a deliberately read-only directory.
    Measured on a ``0o500`` directory: with ``manage_mode`` the chmod widens it to ``0o2775`` and
    the lock file is then written successfully; with ``manage_mode`` False the write raises
    ``PermissionError``. That is a behaviour change — a caller-supplied path under a read-only
    directory now fails loudly instead of being silently made writable — and failing is the
    intended outcome.

    A failed ``chmod`` is ignored: on a shared tree the directory may already
    belong to another user with the mode already correct, and raising here
    would break exactly the multi-user case this module exists for.
    """
    path.mkdir(parents=True, exist_ok=True)
    if not manage_mode:
        return
    try:
        os.chmod(path, LOCK_DIR_MODE)
    except OSError:
        pass


def _touch_shared(path: Path) -> None:
    """Create ``path`` if it does not exist, then force it group-writable.

    Used for both the lock file and the heartbeat file. As with
    :func:`_ensure_dir`, a failed ``chmod`` is ignored rather than raised.

    This stays unconditional while :func:`_ensure_dir`'s became opt-out, and the boundary is
    narrower than it looks: these are files senselab itself creates, so widening them alters
    nothing the caller made, whereas the directory chmod alters a directory the caller did.
    Group-write on the heartbeat is also load-bearing — a second user must be able to refresh it,
    or a live holder reads as stale (see :meth:`SharedFileLock._heartbeat_loop`).
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

    # Above this, a successful acquire is treated as "we waited for a live holder to
    # release", never as a stale takeover -- see `__enter__`. Comfortably above
    # filelock's own 0.05s poll interval, comfortably below any real contention in this
    # codebase (seconds to minutes).
    _UNCONTENDED_ACQUIRE_THRESHOLD = 0.5

    def __init__(
        self,
        path: Path,
        *,
        timeout: float = 600.0,
        heartbeat_interval: float = 30.0,
        stale_after: float = 120.0,
        manage_dir_mode: bool = True,
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
            manage_dir_mode: When True (default, for senselab's own cache dirs),
                the lock file's parent directory is chmod'ed to ``LOCK_DIR_MODE`` so a
                later user's files inherit the shared group. Set False when the guarded
                path is caller-supplied (e.g. a ``FileRef`` over an input file in the
                caller's own private directory): the chmod can only succeed on a
                directory the invoking user owns, and there it *widens* — a ``0o700``
                directory becomes ``0o2775``, world-traversable. See :func:`_ensure_dir`
                for the measurements and for the read-only-directory behaviour change.
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
        self._manage_dir_mode = manage_dir_mode
        self._lock = FileLock(str(self._lock_path))
        self._stop_event = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._token: Optional[str] = None

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

    def _heartbeat_age(self) -> Optional[float]:
        """Return seconds since the heartbeat file was last touched, or None if it cannot be read.

        ``None`` covers a missing file and every other ``OSError`` from ``stat`` alike --
        see ``specs/20260907-shared-lock-heartbeat-inf/`` for why this must not be coerced
        to ``float("inf")``.
        """
        try:
            return time.time() - self._heartbeat_path.stat().st_mtime
        except OSError:
            return None

    def _staleness(self, holder: dict) -> tuple[Optional[float], str, float]:
        """Return ``(age, basis, bound)`` for judging whether ``holder``'s lock is stale.

        ``basis`` is ``"heartbeat"`` when the heartbeat file's own age decides it, or
        ``"taken_at"`` when falling back to how long ago ``holder`` recorded acquiring the
        lock because the heartbeat could not be read. ``age`` is ``None`` when neither
        source is available, and callers must not treat that as stale.
        """
        hb_age = self._heartbeat_age()
        if hb_age is not None:
            return hb_age, "heartbeat", self._stale_after
        taken_at = holder.get("taken_at")
        if isinstance(taken_at, (int, float)):
            return max(0.0, time.time() - taken_at), "taken_at", self._timeout
        return None, "unknown", self._timeout

    def _write_holder(self) -> None:
        self._token = uuid.uuid4().hex
        payload = {
            "user": getpass.getuser(),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "taken_at": time.time(),
            "token": self._token,
        }
        self._lock_path.write_text(json.dumps(payload))

    def owns(self) -> bool:
        """Return whether this instance is still the identity recorded at the lock path.

        A takeover elsewhere overwrites that identity without contacting this process, so a
        caller doing long-running work under the lock should call this before an
        irreversible step (e.g. declaring a build complete) rather than assume that holding
        this Python object still means holding the resource.
        """
        current = lock_holder(self._lock_path)
        return current is not None and self._token is not None and current.get("token") == self._token

    def _warn_stale_takeover(self, holder: dict, age: float, basis: str, bound: float) -> None:
        """Log the "we are taking over a dead holder's lock" warning.

        Named identity (user/host/pid), the age actually read, and the basis it came from
        are what let someone on a cluster check whether the job that held this lock really
        died, rather than just being told "stale lock detected".
        """
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

    def __enter__(self) -> "SharedFileLock":
        """Acquire the lock, taking over a dead holder's leftovers if found.

        See ``specs/20260907-shared-lock-heartbeat-inf/`` for why the ``else`` branch
        gates on *how long the acquire took* rather than re-reading holder content
        afterward: ``filelock`` truncates the lock file on every attempt (including a
        contender's failed polls), so content read post-acquire cannot distinguish a
        clean release from a crash leftover, but elapsed wait time can.

        Returns:
            This instance, for use as a context manager.

        Raises:
            TimeoutError: The lock is held by a live process, or staleness cannot be
                determined for a holder identity left behind by an uncontended acquire.
        """
        _ensure_dir(self._lock_path.parent, manage_mode=self._manage_dir_mode)
        _touch_shared(self._lock_path)
        # Read whatever identity is on disk *before* any acquire attempt: `filelock`
        # truncates the lock file with O_TRUNC on every `_acquire()` call it makes,
        # including a failed poll, so this is the only read guaranteed to precede that.
        previous_holder: Optional[dict] = lock_holder(self._lock_path)
        acquire_started = time.monotonic()
        try:
            self._lock.acquire(timeout=self._timeout)
        except Timeout:
            age = self._heartbeat_age()
            age_display = f"{age:.1f}s old" if age is not None else "unreadable"
            if previous_holder is not None:
                # `.get`, not `[...]`: guards against a lock file senselab did not write --
                # a hand-edited one, or a foreign `<resource>.lock` met on the FileRef path
                # in subprocess_venv.call_in_venv.
                taken_at = previous_holder.get("taken_at")
                held_for = f"{time.time() - taken_at:.1f}s" if isinstance(taken_at, (int, float)) else "an unknown time"
                detail = (
                    f"held by user={previous_holder.get('user')} host={previous_holder.get('host')} "
                    f"pid={previous_holder.get('pid')} for {held_for} "
                    f"(heartbeat {age_display})"
                )
            else:
                detail = f"held by an unknown process (heartbeat {age_display}, no identity on disk)"
            raise TimeoutError(
                f"Timed out after {self._timeout:.1f}s waiting for lock at {self._lock_path}: {detail}. "
                "The OS-level lock was held for the entire wait, which a crashed process cannot do, so this "
                "is a live holder even though its heartbeat may look stale -- it is not broken automatically. "
                "If that process is confirmed dead, remove the .lock file by hand."
            ) from None
        else:
            # A measurable wait means the flock was actually held until moments ago: a
            # live holder finished and released normally during it (`__exit__` clears the
            # heartbeat and holder identity before releasing the flock, so `previous_holder`
            # is now a snapshot of a resource that no longer needs taking over from anyone).
            # Only a genuinely uncontended (near-instant) acquire can mean `previous_holder`,
            # if present, is an untouched leftover from a holder that never released at all.
            contended = (time.monotonic() - acquire_started) > self._UNCONTENDED_ACQUIRE_THRESHOLD
            if previous_holder is not None and not contended:
                age, basis, bound = self._staleness(previous_holder)
                if age is None:
                    # Release the OS-level lock we just took: raising here must leave this
                    # process holding nothing, matching the Timeout branch's contract.
                    self._lock.release()
                    raise TimeoutError(
                        f"Cannot determine whether the lock at {self._lock_path} is stale: "
                        f"held by user={previous_holder.get('user')} host={previous_holder.get('host')} "
                        f"pid={previous_holder.get('pid')}, no heartbeat and no recorded acquire time. "
                        "Not taking over."
                    )
                if age > bound:
                    self._warn_stale_takeover(previous_holder, age, basis, bound)

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
