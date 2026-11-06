"""A lease on a resource in a shared directory: a directory taken by `mkdir` and renewed by beats (ADR-003).

`Lease` over a resource `<dir>/<name>` is the directory `<name>.lease`, which holds one directory named by its
holder's uuid token. Every operation is relative to the directory's descriptor, never follows a symlink, and
stays inside `<dir>`; no file is written under the lease:

- **take:** `mkdir <name>.lease`, which fails if anything is at the name, so one taker wins; then
  `mkdir <name>.lease/<token>`. An `EEXIST` on the token is this holder's own (a lost NFS reply). A taker that
  finds another token beside its own withdraws, and removes the lease directory if that leaves it empty, so two
  takers that both withdraw leave nothing at the name. A taker checks its token afresh after the take.
- **renew:** `keep_alive` runs a renewer thread, owned by the lease and independent of the work, that creates
  `<token>/beat-<n+1>` and removes `<token>/beat-<n>` on a cadence of `renew_fraction × stale_after`, so the
  storage stamps `<token>/` at each beat. A beat that finds the lease gone (`ENOENT`, or `ESTALE` on NFS), or a
  renewer that ends for any other reason, marks its `LeaseState` lost.
- **prove:** before committing anything under the lease, a holder calls `prove`, a `mkdir` in its token
  directory that the storage answers itself.
- **judge:** a lease's age is the storage clock (`StorageDir.dir_now`) less the newest mtime of `<name>.lease`
  and its tokens, read by a fresh `lstat`; a lease is broken only when that age, read after a fresh clock
  probe, passes `stale_after`. Anything but a directory at the name is moved aside (quarantined).
- **break and release:** the breaker or releaser claims the lease by renaming an entry inside it: its token to
  `gone-<token>`, or, in a lease with no token, another entry (a dead claimant's) to `claim-<uuid4>`. An entry is
  one directory's, so one claim succeeds, and only the claimant renames `<name>.lease` to a fresh uuid name beside
  it and removes it. A lease with no entry is removed by `rmdir`, which fails once a taker's token is in it; one
  that cannot be listed is not broken. A holder whose token was claimed learns it at its next beat, `owns` or
  `prove`.

Every wait ends at `timeout` in `LeaseTimeout`. The lease is an efficiency hint: what it guards stays correct by
idempotence, not by the lease.
"""

import contextlib
import errno
import logging
import os
import pwd
import re
import stat
import threading
import time
import uuid
from collections.abc import Iterator
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
GONE_PREFIX = "gone-"
"""Prefixed to a token by the one caller that claims its lease for removal."""
GONE_NAME = re.compile(r"^gone-[0-9a-f]{32}$")
"""A claimed token in a lease being removed."""
CLAIM_PREFIX = "claim-"
"""Names the entry a breaker renames to claim a lease that has no token."""
CLAIM_NAME = re.compile(r"^claim-[0-9a-f]{32}$")
"""A claimed lease with no token, being removed."""
PROOF_NAME = re.compile(r"^proof-[0-9a-f]{32}$")
"""A directory `Lease.prove` makes and removes in a holder's token directory."""
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
GONE_ERRNOS = frozenset({errno.ENOENT, errno.ESTALE})
"""What the storage answers for a lease directory that was moved or removed: gone, or a stale NFS handle to it."""
CLAIM_LOST_ERRNOS = GONE_ERRNOS | {errno.EINVAL}
"""What a claim's rename answers when the entry was claimed first or its lease is being torn down (`EINVAL` on APFS)."""
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
        self._renewer: Optional[tuple[threading.Thread, threading.Event, float]] = None
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

    def _listing(self) -> Optional[tuple[dict[str, float], Optional[dict[str, float]]]]:
        """Return the mtime of each token directory in the lease and of each other entry; None if the lease is gone.

        The other entries are None when the lease cannot be listed.
        """
        try:
            with self._directory().child(self._name) as lease:
                found = {n: lease.lstat(n) for n in lease.listdir()}
        except (FileNotFoundError, WrongKindError):
            return None
        except PermissionError:
            return {}, None
        tokens = {
            n: st.st_mtime
            for n, st in found.items()
            if st is not None and UUID_HEX.fullmatch(n) and stat.S_ISDIR(st.st_mode)
        }
        return tokens, {n: st.st_mtime for n, st in found.items() if st is not None and n not in tokens}

    def _seen(self) -> Optional[tuple[frozenset[str], float, Optional[frozenset[str]]]]:
        """Return the lease's tokens, the newest mtime of it, its tokens and claims, and its other entries.

        None when no lease is at the name. The lease is read again after its entries, so a claim renamed inside it
        meanwhile shows in its mtime. The other entries are None when the lease cannot be listed. Anything but a
        directory at the name is moved aside.
        """
        directory = self._directory()
        st = directory.lstat(self._name)
        if st is None:
            return None
        if not stat.S_ISDIR(st.st_mode):
            self.set_aside.append(directory.set_aside(self._name, None))
            logger.warning("lease %s: moved an object of the wrong kind aside", self._lease_path)
            return None
        listing = self._listing()
        if listing is None:
            return None
        tokens, others = listing
        after = directory.lstat(self._name)
        if after is None:
            return None
        claims = [m for n, m in (others or {}).items() if GONE_NAME.fullmatch(n) or CLAIM_NAME.fullmatch(n)]
        newest = max([st.st_mtime, after.st_mtime, *tokens.values(), *claims])
        return frozenset(tokens), newest, None if others is None else frozenset(others)

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

    def _break(self, tokens: frozenset[str], others: Optional[frozenset[str]]) -> bool:
        """Claim the lease holding `tokens` (or, with none, the entries `others`), move it aside and remove it.

        Returns whether this caller removed it. The caller first claims the lease by renaming one entry inside it, and
        stamps the claimed entry so other judges see the lease as fresh while it is removed: its
        first token to `gone-<token>`, or, with no token, its first other entry (a dead claimant's `gone-<token>`) to
        `claim-<uuid4>`. An entry name is one directory's, so of all callers that judged the lease only one claim
        succeeds, and only the claimant moves `<name>.lease`. A further token is claimed too; it cannot be a live
        holder's, since the lease's age is the newest of its tokens' and it was judged stale. A lease with no entry
        at all is removed by `rmdir`, which succeeds only while it is empty.

        Raises:
            LeaseUnavailable: the lease cannot be listed, so it cannot be claimed.
        """
        if others is None:
            raise LeaseUnavailable(
                self._lease_path,
                f"{self._lease_path} is stale and cannot be listed, so it cannot be claimed; "
                f"{owner_remedy(self._lease_path)}",
            )
        if not tokens and not others:
            return self._remove_empty()
        claims = [(t, GONE_PREFIX + t, GONE_NAME) for t in sorted(tokens)] or [
            (min(others), f"{CLAIM_PREFIX}{uuid.uuid4().hex}", CLAIM_NAME)
        ]
        claimed = self._claim(claims)
        if claimed is None:
            return False
        directory = self._directory()
        current = directory.lstat(self._name)
        if current is None or (current.st_dev, current.st_ino) != (claimed.st_dev, claimed.st_ino):
            logger.warning("lease %s: claimed, then found another directory at its name", self._lease_path)
            return False
        aside = uuid.uuid4().hex
        try:
            directory.rename(self._name, directory, aside, grammar=UUID_HEX)
        except FileNotFoundError:
            return False
        self._remove(aside)
        return True

    def _claim(self, claims: list[tuple[str, str, re.Pattern[str]]]) -> Optional[os.stat_result]:
        """Rename each `(entry, claim)` in the lease, the first deciding; return the claimed lease's stat, else None."""
        try:
            with self._directory().child(self._name) as lease:
                (name, claim, grammar), *rest = claims
                lease.rename(name, lease, claim, grammar=grammar)
                with contextlib.suppress(OSError):
                    lease.stamp(claim)
                for name, claim, grammar in rest:
                    with contextlib.suppress(OSError):
                        lease.rename(name, lease, claim, grammar=grammar)
                return os.fstat(lease.fd)
        except WrongKindError:
            return None
        except OSError as err:
            if err.errno in CLAIM_LOST_ERRNOS:
                return None
            raise

    def _remove_empty(self) -> bool:
        """Remove the lease by `rmdir`, which succeeds only while it holds nothing; return whether it did."""
        try:
            self._directory().remove_dir(self._name, grammar=LEASE_NAME)
        except OSError as err:
            if err.errno in (*GONE_ERRNOS, errno.ENOTEMPTY, errno.EEXIST):
                return False
            raise
        return True

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
            judged = seen[0]
            if self._break(judged, seen[2]):
                self.broke.append(self._lease_path)
                logger.warning(
                    "lease %s: broke a stale lease %.1fs old (> %.1fs), held by %s",
                    self._lease_path,
                    now - seen[1],
                    self._stale_after,
                    ", ".join(sorted(judged)) or "no token",
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
                    with contextlib.suppress(OSError):
                        directory.remove_dir(self._name, grammar=LEASE_NAME)
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
        except WrongKindError:
            return False
        except OSError as err:
            if err.errno in GONE_ERRNOS:
                return False
            raise

    def prove(self) -> bool:
        """Return whether this lease is still held, proven by a `mkdir` (and `rmdir`) in its token directory.

        A `mkdir` is answered by the storage itself, never from a client's attribute cache, so a holder calls this
        immediately before it commits anything under the lease; False means the lease was lost.
        """
        if self._held is None:
            return False
        proof = f"proof-{uuid.uuid4().hex}"
        try:
            with self._directory().child(self._name) as lease, lease.child(self._held) as mine:
                mine.make_dir(proof, grammar=PROOF_NAME, exist_ok=True)
                with contextlib.suppress(OSError):
                    mine.remove_dir(proof, grammar=PROOF_NAME)
        except WrongKindError:
            return False
        except OSError as err:
            if err.errno in GONE_ERRNOS:
                return False
            raise
        return True

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
        interval = renew_fraction * self._stale_after
        thread = threading.Thread(
            target=self._renew,
            args=(self._directory().dup(), self._held, interval, state, lost, stop),
            name=RENEWER_PREFIX + self._name,
            daemon=True,
        )
        self._renewer = (thread, stop, interval)
        thread.start()
        try:
            yield state
        finally:
            self._stop_renewer()

    def _stop_renewer(self) -> None:
        """Stop the renewer and wait for it for one interval; one held in a hung beat is left to end on its own."""
        renewer, self._renewer = self._renewer, None
        if renewer is not None:
            thread, stop, interval = renewer
            stop.set()
            thread.join(interval)
            if thread.is_alive():
                logger.warning("lease %s: its renewer is still in a beat; it ends when the beat returns", self._name)

    def _beat(self, directory: StorageDir, token: str, n: int) -> None:
        """Make `beat-<n+1>` in this holder's token directory, reached afresh, and remove `beat-<n>`."""
        with directory.child(self._name) as lease, lease.child(token) as mine:
            mine.make_dir(f"beat-{n + 1}", grammar=BEAT_NAME, exist_ok=True)
            if n:
                with contextlib.suppress(FileNotFoundError):
                    mine.remove_dir(f"beat-{n}", grammar=BEAT_NAME)

    def _renew(  # noqa: PLR0913 - its own directory handle, the token, the interval, the state and its two events
        self,
        directory: StorageDir,
        token: str,
        interval: float,
        state: LeaseState,
        lost: threading.Event,
        stop: threading.Event,
    ) -> None:
        """Beat every `interval` seconds until `stop`; a lease found gone, or unrenewed past its bound, is lost.

        Each beat is due `interval` after the last one was due, whatever the beats took, and a beat that is already
        overdue runs at once. The renewer marks the lease lost whenever it ends for any reason but `stop`.
        """
        last = due = time.monotonic()
        try:
            while True:
                due += interval
                if stop.wait(max(0.0, due - time.monotonic())):
                    return
                try:
                    self._beat(directory, token, state.beats)
                except WrongKindError:
                    logger.warning("lease %s: was broken while held; its holder stops", self._lease_path)
                    return
                except OSError as err:
                    if err.errno in GONE_ERRNOS:
                        logger.warning("lease %s: was broken while held; its holder stops", self._lease_path)
                        return
                    logger.warning("lease %s: a beat failed: %s", self._lease_path, err)
                    if time.monotonic() - last > self._stale_after:
                        return
                    continue
                state.beats += 1
                last = time.monotonic()
                due = max(due, last - interval)
        except Exception:
            logger.exception("lease %s: the renewer failed; the lease is lost", self._lease_path)
        finally:
            directory.close()
            if not stop.is_set():
                lost.set()

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
            if token is not None and not self._break(frozenset({token}), frozenset()):
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
    "CLAIM_LOST_ERRNOS",
    "CLAIM_NAME",
    "CLAIM_PREFIX",
    "GONE_ERRNOS",
    "GONE_NAME",
    "GONE_PREFIX",
    "PROOF_NAME",
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
