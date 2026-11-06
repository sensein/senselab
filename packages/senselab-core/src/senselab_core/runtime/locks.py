"""A lease on a resource in a shared directory: a directory taken by `mkdir` and renewed by beats (ADR-003).

`Lease` over a resource `<dir>/<name>` is the directory `<name>.lease`, which holds one directory named by its
holder's uuid token. Every operation is relative to the directory's descriptor, never follows a symlink, and
stays inside `<dir>`; no file is written under the lease:

- **take:** `mkdir <name>.lease`, which fails if anything is at the name, so one taker wins; then
  `mkdir <name>.lease/<token>`. An `EEXIST` on the token is this holder's own (a lost NFS reply). A taker that
  finds another token beside its own withdraws.
- **renew:** `keep_alive` runs a renewer thread, owned by the lease and independent of the work, that creates
  `<token>/beat-<n+1>` and removes `<token>/beat-<n>` every `renew_fraction × stale_after`, so the storage
  stamps `<token>/` at each beat. A beat that finds the lease gone (`ENOENT`) marks its `LeaseState` lost.
- **judge:** a lease's age is the storage clock (`StorageDir.dir_now`) less the newest mtime of `<name>.lease`
  and its tokens, read by a fresh `lstat`; a lease is broken only when that age, read after a fresh clock
  probe, passes `stale_after`. Anything but a directory at the name is moved aside (quarantined).
- **break and release:** `<name>.lease` is renamed to a fresh uuid name beside it, which is atomic, and removed
  whatever it holds. A breaker moves only what holds the tokens it judged stale, re-read after a fresh clock
  probe; a newer lease moved in the window between that read and the rename is removed too, and its holder
  learns it at its next beat or `owns` (overlap is duplicate work, within idempotence). A taker checks its
  token afresh after the take, so a lease moved during the take is not held.

Every wait ends at `timeout` in `LeaseTimeout`. The lease is an efficiency hint: what it guards stays correct by
idempotence, not by the lease.
"""

import contextlib
import logging
import pwd
import re
import stat
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Optional

from senselab_worker.ownership import UUID_HEX

from senselab_core.runtime.storage import StorageDir, WrongKindError, owner_remedy, path_component

logger = logging.getLogger("senselab")

LEASE_SUFFIX = ".lease"
"""Appended to a resource's name to name its lease."""
LEASE_NAME = re.compile(r"^.+\.lease$")
"""The lease directory: `<name>.lease`."""
BEAT_NAME = re.compile(r"^beat-[0-9]+$")
"""A beat directory in a holder's token directory."""
LEASE_DIR_MODE = 0o2770
"""The default mode of a lease directory and of its token directory."""
TIMEOUT_S = 600.0
"""The default bound, in seconds, on a wait for a lease."""
STALE_AFTER_S = 120.0
"""The default age, in seconds on the storage clock, after which a lease is stale."""
POLL_S = 0.05
"""Seconds between two attempts to take a lease."""
REMOVE_ATTEMPTS = 3
"""How many times a lease moved aside is removed while a taker that reached it before the move races the removal."""
RENEWER_PREFIX = "lease-renewer:"
"""The name of a renewer thread, before its lease's name."""


class LeaseTimeout(TimeoutError):
    """A wait for a lease reached its bound while a live lease was held."""


class LeaseUnavailable(RuntimeError):
    """A lease cannot be taken in its directory, or a stale lease cannot be moved; the message names the remedy.

    Attributes:
        path: The lease directory.
    """

    def __init__(self, path: Path, message: str) -> None:
        """Record `path`; `message` is the whole text."""
        super().__init__(message)
        self.path = path


class LeaseState:
    """What a lease's renewer reports while `Lease.keep_alive` runs.

    Attributes:
        beats: The beats made so far.
    """

    def __init__(self, lost: threading.Event) -> None:
        """Start with no beat; the lease is lost once its renewer sets `lost`."""
        self.beats = 0
        self._lost = lost

    @property
    def lost(self) -> bool:
        """Whether the lease was broken while held: the holder stops and does not commit as its owner."""
        return self._lost.is_set()

    def wait_lost(self, timeout: float) -> bool:
        """Wait up to `timeout` seconds for the lease to be lost; return whether it was."""
        return self._lost.wait(timeout)


class Lease:
    """A lease over the resource `path`, held for the duration of a `with` block.

    Args:
        path: The resource being guarded (not the lease itself).
        timeout: Seconds a wait in `__enter__` may take before `LeaseTimeout`; not negative.
        stale_after: Seconds on the storage clock after which another taker breaks the lease.
        mode: The mode of the lease directory and of the token directory.
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
        mode: int = LEASE_DIR_MODE,
        directory: Optional[StorageDir] = None,
    ) -> None:
        """Configure a lease over `path` without taking it.

        Raises:
            ValueError: `timeout` or `stale_after` is negative, or `path`'s name or its lease's is not one component.
        """
        if timeout < 0 or stale_after < 0:
            raise ValueError(f"a lease's timeout and stale_after are not negative (got {timeout}, {stale_after})")
        self._path = Path(path)
        self._name = path_component(path_component(self._path.name) + LEASE_SUFFIX)
        self._lease_path = self._path.with_name(self._name)
        self._timeout = timeout
        self._stale_after = stale_after
        self._mode = mode
        self._given = directory
        self._dir: Optional[StorageDir] = None
        self._held: Optional[str] = None
        self._clock: Optional[tuple[float, float]] = None
        self._renewer: Optional[tuple[threading.Thread, threading.Event]] = None
        self.broke: list[Path] = []
        self.set_aside: list[Path] = []

    @property
    def path(self) -> Path:
        """The lease directory."""
        return self._lease_path

    def _directory(self) -> StorageDir:
        if self._dir is None:
            self._dir = self._given.dup() if self._given is not None else StorageDir.open_root(self._path.parent)
        return self._dir

    def _now(self, *, fresh: bool) -> float:
        """Return the storage clock: probed now when `fresh`, else the last probe plus the monotonic time since."""
        if fresh or self._clock is None:
            self._clock = (self._directory().dir_now(), time.monotonic())
        return self._clock[0] + time.monotonic() - self._clock[1]

    def _tokens_in(self, name: str) -> Optional[dict[str, float]]:
        """Return the mtime of each token directory in the directory `name`; empty if unlistable, None if gone."""
        try:
            with self._directory().child(name) as lease:
                found = {t: lease.lstat(t) for t in lease.listdir() if UUID_HEX.fullmatch(t)}
        except (FileNotFoundError, WrongKindError):
            return None
        except PermissionError:
            return {}
        return {t: st.st_mtime for t, st in found.items() if st is not None and stat.S_ISDIR(st.st_mode)}

    def _seen(self) -> Optional[tuple[frozenset[str], float]]:
        """Return the tokens of the lease and the newest mtime of it and them; None if no lease is at the name.

        Anything but a directory at the name is moved aside.
        """
        directory = self._directory()
        st = directory.lstat(self._name)
        if st is None:
            return None
        if not stat.S_ISDIR(st.st_mode):
            self.set_aside.append(directory.set_aside(self._name, None))
            logger.warning("lease %s: moved an object of the wrong kind aside", self._lease_path)
            return None
        tokens = self._tokens_in(self._name)
        if tokens is None:
            return None
        return frozenset(tokens), max([st.st_mtime, *tokens.values()])

    def _remove(self, aside: str) -> None:
        """Remove the moved lease `aside`, again while a taker that reached it before the move races the removal."""
        directory = self._directory()
        failure: Optional[OSError] = None
        for _ in range(REMOVE_ATTEMPTS):
            try:
                directory.remove_owned(aside, pattern=UUID_HEX)
            except OSError as err:
                failure = err
            if directory.lstat(aside) is None:
                return
        self.set_aside.append(directory.path / aside)
        logger.warning("lease %s: moved aside to %s, which cannot be removed: %s", self._lease_path, aside, failure)

    def _take_aside(self, expected: Callable[[frozenset[str]], bool]) -> bool:
        """Rename the lease aside and remove it; return whether its tokens were those `expected` holds of.

        Rename is atomic, so of two callers expecting one lease only one moves it. A lease moved by mistake is not
        put back, because its taker may have judged its take failed; its holder finds it lost.
        """
        directory = self._directory()
        aside = uuid.uuid4().hex
        try:
            directory.rename(self._name, directory, aside, grammar=UUID_HEX)
        except FileNotFoundError:
            return False
        found = self._tokens_in(aside)
        if found is None:
            self.set_aside.append(directory.path / aside)
            return False
        self._remove(aside)
        if expected(frozenset(found)):
            return True
        if found:
            logger.warning("lease %s: displaced a newer lease; its holder finds it lost", self._lease_path)
        return False

    def _judge(self) -> None:
        """Clear the lease name if what is there is of the wrong kind or a stale lease."""
        try:
            seen = self._seen()
            if seen is None or self._now(fresh=False) - seen[1] <= self._stale_after:
                return
            now = self._now(fresh=True)
            seen = self._seen()
            if seen is None or now - seen[1] <= self._stale_after:
                return
            holder = self.holder() or "an unknown holder"
            judged = seen[0]
            if self._take_aside(lambda found: found == judged):
                self.broke.append(self._lease_path)
                logger.warning(
                    "lease %s: broke a stale lease %.1fs old (> %.1fs), held by %s",
                    self._lease_path,
                    now - seen[1],
                    self._stale_after,
                    holder,
                )
        except (WrongKindError, PermissionError) as err:
            raise LeaseUnavailable(
                self._lease_path,
                f"{self._lease_path} is stale or of the wrong kind and cannot be judged or moved ({err}); "
                f"{owner_remedy(self._lease_path)}",
            ) from err

    def _try_take(self) -> Optional[str]:
        """Make the lease directory and this holder's token in it; return the token, None if another holds it."""
        directory = self._directory()
        try:
            directory.make_dir(self._name, grammar=LEASE_NAME, mode=self._mode)
        except (FileExistsError, FileNotFoundError):
            return None
        except PermissionError as err:
            raise LeaseUnavailable(
                self._lease_path,
                f"cannot take the lease {self._lease_path}: {err.strerror}; {owner_remedy(directory.path)}",
            ) from err
        token = uuid.uuid4().hex
        try:
            with directory.child(self._name) as lease:
                lease.make_dir(token, grammar=UUID_HEX, mode=self._mode, exist_ok=True)
                if any(t != token and UUID_HEX.fullmatch(t) for t in lease.listdir()):
                    lease.remove_dir(token, grammar=UUID_HEX)
                    return None
        except (FileNotFoundError, WrongKindError):
            return None
        return token if self._has(token) else None

    def holder(self) -> Optional[dict[str, Any]]:
        """Return the token and owning user of the lease's token directory, None if no token is there."""
        try:
            with self._directory().child(self._name) as lease:
                for token in sorted(t for t in lease.listdir() if UUID_HEX.fullmatch(t)):
                    st = lease.lstat(token)
                    if st is not None:
                        return {"token": token, "user": _user(st.st_uid)}
        except (OSError, WrongKindError):
            return None
        return None

    def owns(self) -> bool:
        """Return whether this lease's token directory is in the directory at the lease name."""
        return self._held is not None and self._has(self._held)

    def _has(self, token: str) -> bool:
        try:
            with self._directory().child(self._name) as lease:
                return lease.kind(token) == "directory"
        except (FileNotFoundError, WrongKindError):
            return False

    @contextlib.contextmanager
    def keep_alive(self, renew_fraction: float) -> Iterator[LeaseState]:
        """Renew the held lease every `renew_fraction × stale_after` seconds from a daemon thread until the block ends.

        Raises:
            ValueError: `renew_fraction` is not in (0, 1).
            RuntimeError: the lease is not held, or is already kept alive.
        """
        if not 0 < renew_fraction < 1:
            raise ValueError(f"a renew fraction is in (0, 1) (got {renew_fraction})")
        if self._held is None or self._renewer is not None:
            raise RuntimeError(f"lease {self._lease_path}: keep_alive needs a held lease without a renewer")
        lost, stop = threading.Event(), threading.Event()
        state = LeaseState(lost)
        thread = threading.Thread(
            target=self._renew,
            args=(self._held, renew_fraction * self._stale_after, state, lost, stop),
            name=RENEWER_PREFIX + self._name,
            daemon=True,
        )
        self._renewer = (thread, stop)
        thread.start()
        try:
            yield state
        finally:
            self._stop_renewer()

    def _stop_renewer(self) -> None:
        renewer, self._renewer = self._renewer, None
        if renewer is not None:
            renewer[1].set()
            renewer[0].join()

    def _beat(self, token: str, n: int) -> None:
        """Make `beat-<n+1>` in this holder's token directory, reached afresh, and remove `beat-<n>`."""
        with self._directory().child(self._name) as lease, lease.child(token) as mine:
            mine.make_dir(f"beat-{n + 1}", grammar=BEAT_NAME, exist_ok=True)
            if n:
                with contextlib.suppress(FileNotFoundError):
                    mine.remove_dir(f"beat-{n}", grammar=BEAT_NAME)

    def _renew(  # noqa: PLR0913 - the token, the interval, and the state with its two events
        self, token: str, interval: float, state: LeaseState, lost: threading.Event, stop: threading.Event
    ) -> None:
        """Beat every `interval` seconds until `stop`; a lease found gone, or unrenewed past its bound, is lost."""
        last = time.monotonic()
        while not stop.wait(interval):
            try:
                self._beat(token, state.beats)
            except (FileNotFoundError, WrongKindError):
                logger.warning("lease %s: was broken while held; its holder stops", self._lease_path)
                lost.set()
                return
            except OSError as err:
                logger.warning("lease %s: a beat failed: %s", self._lease_path, err)
                if time.monotonic() - last > self._stale_after:
                    lost.set()
                    return
                continue
            state.beats += 1
            last = time.monotonic()

    def __enter__(self) -> "Lease":
        """Take the lease, breaking a stale one and moving a wrong kind aside.

        Raises:
            LeaseTimeout: a live lease was held for the whole `timeout`.
            LeaseUnavailable: a lease cannot be created in the directory, or a stale or wrong-kind object at
                the lease name cannot be moved.
            StorageUnsupported: the directory has no atomic `mkdir`.
        """
        self.broke, self.set_aside, self._clock = [], [], None
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
        """Stop the renewer and release the lease if this lease still holds it; one broken while held is logged."""
        try:
            self._stop_renewer()
            token = self._held
            if token is not None and not (self.owns() and self._take_aside(lambda found: token in found)):
                logger.warning("lease %s: was broken while held; the work it guarded may run twice", self._lease_path)
        finally:
            self._held = None
            self._close()

    def _close(self) -> None:
        directory, self._dir = self._dir, None
        if directory is not None:
            directory.close()


def _user(uid: int) -> str:
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return str(uid)


__all__ = [
    "BEAT_NAME",
    "LEASE_DIR_MODE",
    "LEASE_NAME",
    "LEASE_SUFFIX",
    "POLL_S",
    "REMOVE_ATTEMPTS",
    "RENEWER_PREFIX",
    "STALE_AFTER_S",
    "TIMEOUT_S",
    "Lease",
    "LeaseState",
    "LeaseTimeout",
    "LeaseUnavailable",
]
