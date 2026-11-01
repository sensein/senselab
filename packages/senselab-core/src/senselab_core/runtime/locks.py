"""A lease on a resource in a shared directory, by `O_EXCL` link and rename (ADR-003).

`Lease` over a resource `<dir>/<name>` is the file `<name>.lease`. Everything is done relative to the
directory's descriptor and never follows a symlink:

- **take:** the holder's identity is written to a fresh private temporary (`O_EXCL | O_NOFOLLOW`, its mode set
  through its descriptor) and hard-linked at `<name>.lease`; the link fails if anything is at the name, so one
  taker wins. The lease's mtime, stamped by the storage, is when it was taken.
- **hold:** the holder owns the lease while the file at `<name>.lease` carries the token it wrote. A token, not
  an inode number, because a filesystem may give a freed inode number to the next file (ext4 does at once).
- **judge:** a lease is stale when its age on the storage clock (`StorageDir.fs_now`) passes `stale_after`.
  Anything but a regular file at `<name>.lease` is of the wrong kind and is moved aside (quarantined).
- **break and release:** the lease is renamed to a fresh uuid name, which is atomic, and the moved file is
  compared with the one judged (device, inode and mtime, which rename keeps) or with the holder's token; a
  lease moved by mistake is linked back. So two breakers cannot both remove one lease, and a releaser never
  removes another holder's lease.

Every wait ends at `timeout` in `LeaseTimeout`. The lease is an efficiency hint: a holder that outlives
`stale_after` can be broken, and what it guards stays correct by idempotence, not by the lease.
"""

import errno
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
from collections.abc import Callable
from typing import Optional

from senselab_worker.ownership import UUID_HEX

from senselab_core.runtime.storage import TEMPORARY_NAME, StorageDir, WrongKindError, owner_remedy, path_component

logger = logging.getLogger("senselab")

LEASE_SUFFIX = ".lease"
"""Appended to a resource's name to name its lease."""
LEASE_NAME = re.compile(r"^.+\.lease$")
"""The lease file: `<name>.lease`."""
LEASE_FILE_MODE = 0o664
"""The default mode of a lease file."""
TIMEOUT_S = 600.0
"""The default bound, in seconds, on a wait for a lease."""
STALE_AFTER_S = 120.0
"""The default age, in seconds on the storage clock, after which a lease is stale."""
POLL_S = 0.05
"""Seconds between two attempts to take a lease."""
IDENTITY_LIMIT = 1 << 16
"""The most bytes of a lease file read for its holder's identity."""


class LeaseTimeout(TimeoutError):
    """A wait for a lease reached its bound while a live lease was held."""


class LeaseUnavailable(RuntimeError):
    """A lease cannot be taken in its directory, or a stale lease cannot be moved; the message names the remedy.

    Attributes:
        path: The lease file.
    """

    def __init__(self, path: Path, message: str) -> None:
        """Record `path`; `message` is the whole text."""
        super().__init__(message)
        self.path = path


def _temporary_name(lease: str) -> str:
    return f".{lease}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"


class Lease:
    """A lease over the resource `path`, held for the duration of a `with` block.

    Args:
        path: The resource being guarded (not the lease file itself).
        timeout: Seconds a wait in `__enter__` may take before `LeaseTimeout`; not negative.
        stale_after: Seconds on the storage clock after which another taker breaks the lease.
        file_mode: The mode of the lease file.
        directory: The directory holding `path`, already open; it is duplicated and `path`'s parent is not
            opened again.

    Attributes:
        broke: The paths of the stale leases this lease broke before it was taken.
        set_aside: The paths this lease moved aside: objects of the wrong kind at its name.
    """

    def __init__(
        self,
        path: Path,
        *,
        timeout: float = TIMEOUT_S,
        stale_after: float = STALE_AFTER_S,
        file_mode: int = LEASE_FILE_MODE,
        directory: Optional[StorageDir] = None,
    ) -> None:
        """Configure a lease over `path` without taking it.

        Raises:
            ValueError: `timeout` or `stale_after` is negative, or `path`'s name is not one component.
        """
        if timeout < 0 or stale_after < 0:
            raise ValueError(f"a lease's timeout and stale_after are not negative (got {timeout}, {stale_after})")
        self._path = Path(path)
        self._name = path_component(self._path.name) + LEASE_SUFFIX
        self._lease_path = self._path.with_name(self._name)
        self._timeout = timeout
        self._stale_after = stale_after
        self._file_mode = file_mode
        self._given = directory
        self._dir: Optional[StorageDir] = None
        self._held: Optional[str] = None
        self._seen: Optional[tuple[tuple[int, int, int], float, float]] = None
        self.broke: list[Path] = []
        self.set_aside: list[Path] = []

    @property
    def path(self) -> Path:
        """The lease file."""
        return self._lease_path

    def _directory(self) -> StorageDir:
        if self._dir is None:
            self._dir = self._given.dup() if self._given is not None else StorageDir.open_root(self._path.parent)
        return self._dir

    def _identity(self, token: str) -> bytes:
        record = {"user": getpass.getuser(), "host": socket.gethostname(), "pid": os.getpid(), "token": token}
        return json.dumps(record).encode("utf-8")

    def _try_take(self) -> Optional[str]:
        """Link a fresh identity at the lease name; return its token, None if the name is taken.

        A link whose reply is lost (NFS) may fail although it was made, so the outcome is read from the
        temporary's link count, not from the error.
        """
        directory = self._directory()
        temporary = _temporary_name(self._name)
        try:
            fd = directory.open_new(temporary, self._file_mode, grammar=TEMPORARY_NAME)
        except PermissionError as err:
            raise LeaseUnavailable(
                self._lease_path,
                f"cannot take the lease {self._lease_path}: {err.strerror}; {owner_remedy(directory.path)}",
            ) from err
        token = uuid.uuid4().hex
        try:
            os.write(fd, self._identity(token))
            os.fsync(fd)
            try:
                directory.link(temporary, directory, self._name, grammar=LEASE_NAME)
            except OSError as err:
                if os.fstat(fd).st_nlink < 2:
                    if err.errno == errno.EEXIST:
                        return None
                    raise
            return token
        finally:
            os.close(fd)
            directory.remove_owned(temporary, pattern=TEMPORARY_NAME)

    def _record_at(self, name: str) -> Optional[dict]:
        """Return the identity recorded in the regular file `name`, None if none is legible there."""
        try:
            data = self._directory().read_regular(name, limit=IDENTITY_LIMIT)
            record = json.loads(data.decode("utf-8")) if data else None
        except (OSError, ValueError, WrongKindError):
            return None
        return record if isinstance(record, dict) else None

    def _carries(self, token: str) -> Callable[[str], bool]:
        """Return a test of whether the file at a name carries `token`."""
        return lambda name: (self._record_at(name) or {}).get("token") == token

    def _is(self, judged: os.stat_result) -> Callable[[str], bool]:
        """Return a test of whether the object at a name is the file `judged`: device, inode and mtime."""
        key = (judged.st_dev, judged.st_ino, judged.st_mtime_ns)

        def test(name: str) -> bool:
            st = self._directory().lstat(name)
            return st is not None and (st.st_dev, st.st_ino, st.st_mtime_ns) == key

        return test

    def _take_aside(self, expected: Callable[[str], bool]) -> bool:
        """Rename the lease name aside and remove it if `expected` holds of it; else link it back.

        Returns whether the expected file was removed. Rename is atomic, so of two callers expecting one file
        only one moves it. A caller that moved another file links it back; if a new lease took the name in
        between, the moved lease is removed and its holder finds it no longer owns it (`owns`).
        """
        directory = self._directory()
        if not expected(self._name):
            return False
        aside = uuid.uuid4().hex
        try:
            directory.rename(self._name, directory, aside, grammar=UUID_HEX)
        except FileNotFoundError:
            return False
        if expected(aside):
            directory.remove_owned(aside, pattern=UUID_HEX)
            return True
        moved = directory.lstat(aside)
        if moved is None or not stat.S_ISREG(moved.st_mode):
            self.set_aside.append(directory.path / aside)
            return False
        try:
            directory.link(aside, directory, self._name, grammar=LEASE_NAME)
        except FileExistsError:
            logger.warning("lease %s: displaced a newer lease that another taker replaced", self._lease_path)
        directory.remove_owned(aside, pattern=UUID_HEX)
        return False

    def _age(self, st: os.stat_result) -> float:
        """Return the age of the lease `st` on the storage clock.

        The storage clock is read once per lease seen; while the same lease stays, its age grows by the
        local monotonic time elapsed since, which is a duration and needs no clock agreement.
        """
        seen = (st.st_dev, st.st_ino, st.st_mtime_ns)
        if self._seen is None or self._seen[0] != seen:
            self._seen = (seen, self._directory().fs_now() - st.st_mtime, time.monotonic())
        return self._seen[1] + time.monotonic() - self._seen[2]

    def _judge(self) -> None:
        """Clear the lease name if what is there is of the wrong kind or a stale lease."""
        directory = self._directory()
        try:
            st = directory.lstat(self._name)
            if st is None:
                return
            if not stat.S_ISREG(st.st_mode):
                self.set_aside.append(directory.set_aside(self._name, None))
                logger.warning("lease %s: moved an object of the wrong kind aside", self._lease_path)
                return
            age = self._age(st)
            if age <= self._stale_after:
                return
            holder = self.holder() or "an unknown holder"
            if self._take_aside(self._is(st)):
                self.broke.append(self._lease_path)
                logger.warning(
                    "lease %s: broke a stale lease %.1fs old (> %.1fs), held by %s",
                    self._lease_path,
                    age,
                    self._stale_after,
                    holder,
                )
        except (WrongKindError, PermissionError) as err:
            raise LeaseUnavailable(
                self._lease_path,
                f"{self._lease_path} is stale or of the wrong kind and cannot be judged or moved ({err}); "
                f"{owner_remedy(self._lease_path)}",
            ) from err

    def holder(self) -> Optional[dict]:
        """Return the identity recorded in the lease file, None if none is legible there."""
        return self._record_at(self._name)

    def owns(self) -> bool:
        """Return whether the file at the lease name carries the token this lease wrote."""
        return self._held is not None and self._carries(self._held)(self._name)

    def __enter__(self) -> "Lease":
        """Take the lease, breaking a stale one and moving a wrong kind aside.

        Raises:
            LeaseTimeout: a live lease was held for the whole `timeout`.
            LeaseUnavailable: a lease cannot be created in the directory, or a stale or wrong-kind object at
                the lease name cannot be moved.
        """
        self.broke, self.set_aside, self._seen = [], [], None
        deadline = time.monotonic() + self._timeout
        try:
            while True:
                self._judge()
                self._held = self._try_take()
                if self._held is not None:
                    return self
                if time.monotonic() >= deadline:
                    raise LeaseTimeout(
                        f"Timed out after {self._timeout:.1f}s waiting for the lease {self._lease_path}, "
                        f"held by {self.holder() or 'an unknown holder'}; it is broken once older than "
                        f"{self._stale_after:.1f}s on the storage clock"
                    )
                time.sleep(min(POLL_S, max(0.0, deadline - time.monotonic())))
        except BaseException:
            self._close()
            raise

    def __exit__(self, *exc: object) -> None:
        """Release the lease if this lease still holds it; a lease broken while held is logged and left."""
        try:
            if self._held is not None and not self._take_aside(self._carries(self._held)):
                logger.warning("lease %s: was broken while held; the work it guarded may run twice", self._lease_path)
        finally:
            self._held = None
            self._close()

    def _close(self) -> None:
        directory, self._dir = self._dir, None
        if directory is not None:
            directory.close()


__all__ = [
    "IDENTITY_LIMIT",
    "LEASE_FILE_MODE",
    "LEASE_NAME",
    "LEASE_SUFFIX",
    "POLL_S",
    "STALE_AFTER_S",
    "TIMEOUT_S",
    "Lease",
    "LeaseTimeout",
    "LeaseUnavailable",
]
