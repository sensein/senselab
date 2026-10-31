"""Storage faults, the store policy, object modes, the storage clock and the address contract.

- `storage_fault` is the one mapping from an `OSError` to a retryable `StoreUnavailable` (or
  `StorageFull`), used by the result store and the data-fetch cache.
- `StorePolicy` holds the modes and retention ages, read from `data/store_policy.yaml`.
- Committed objects (store entries, fetch blobs and completion markers) are read-only:
  `committed_modes` gives their file and directory modes. Staging objects are `STAGING_FILE_MODE`
  and `STAGING_DIR_MODE`, and directories that hold objects are `shared_dir_mode`.
- `StorageDir` is the one way to reach and write below a storage root (ADR-003): no symlink below the root
  is followed, a mutable file is only ever replaced whole by a fresh file renamed over its name, every file
  name written matches a grammar its module lists in `WRITTEN_NAMES`, and modes are set only through a
  descriptor (`set_mode`).
- `StorageDir.fs_now` reads the current time from the filesystem itself, from a fresh sentinel file.
- `write_atomically` replaces a caller's file so a reader sees the old or the new bytes, never a mix.
- An `AddressedFormat` is a persisted artifact whose location names its content. `ADDRESSED_FORMATS`
  lists every one: the store entry, the fetch blob and the fetch dataset marker.
"""

import errno
import logging
import os
import pwd
import re
import stat
import threading
import uuid
from collections.abc import Callable
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Protocol, Self, runtime_checkable

import yaml
from pydantic import Field, StrictBool
from senselab_worker.ownership import UUID_HEX, remove_owned

from senselab_core.kernel._base import Frozen

logger = logging.getLogger("senselab")

POLICY_FORMAT = "store-policy@1.0"
STAGING_FILE_MODE = 0o660
STAGING_DIR_MODE = 0o2770
CLOCK_NAME = ".clock"
"""The sentinel `StorageDir.fs_now` replaces in a directory."""
CLOCK = re.compile(r".*\.clock")
"""A storage clock: `.clock`, or a lock's `<name>.clock`."""

_FULL_ERRNOS = frozenset({errno.ENOSPC, errno.EDQUOT})
_WORLD_READ = 0o004
_WORLD_READ_SEARCH = 0o005


class StoreUnavailable(OSError):
    """Storage could not be read or written; nothing was changed or judged corrupt, and a retry may succeed.

    `errno` and `filename` are those of the original error, which is the `__cause__`.
    """


class StorageFull(StoreUnavailable):
    """The storage is out of space or over quota (`ENOSPC`, `EDQUOT`)."""


def storage_fault(err: OSError, path: Path, *, hint: str | None = None) -> StoreUnavailable:
    """Return the storage fault for `err`, raised while working on `path`; a `StoreUnavailable` is returned as is.

    The fault names the path `err` was raised on by a `StorageDir`, otherwise `path`. The message names the
    cause and the fix: `hint` when given, otherwise one chosen by errno.
    """
    if isinstance(err, StoreUnavailable):
        return err
    path = getattr(err, "storage_path", None) or path
    full = err.errno in _FULL_ERRNOS
    fix = hint or (
        "out of space or over quota; free space or move the store elsewhere"
        if full
        else "check the path exists, is a directory you can read and write, and retry"
    )
    cause = err.strerror or str(err) or type(err).__name__
    fault = (StorageFull if full else StoreUnavailable)(err.errno, f"{cause}: {fix}", str(path))
    fault.__cause__ = err
    return fault


def _value(raw: object, name: str) -> object:
    if not isinstance(raw, dict) or set(raw) != {"value", "derivation"}:
        raise ValueError(f"{POLICY_FORMAT} entry {name!r} must be a mapping of exactly 'value' and 'derivation'")
    if not isinstance(raw["derivation"], str) or not raw["derivation"].strip():
        raise ValueError(f"{POLICY_FORMAT} entry {name!r} needs a non-empty derivation")
    return raw["value"]


class StorePolicy(Frozen):
    """How the result store and the fetch cache set modes and age out leftovers.

    Attributes:
        world_readable: Committed objects and object directories also grant read (and search) to others.
        quarantine_retention_s: Seconds a quarantined entry is kept before `ResultStore.collect` removes it.
        staging_stale_s: Seconds after which a staging directory or download belongs to a crashed writer.
        collect_interval_s: Least seconds between two collections that `ResultStore.put` starts.
    """

    world_readable: StrictBool = False
    quarantine_retention_s: Annotated[int, Field(strict=True, gt=0)]
    staging_stale_s: Annotated[int, Field(strict=True, gt=0)]
    collect_interval_s: Annotated[int, Field(strict=True, gt=0)]

    @classmethod
    def load_default(cls) -> Self:
        """Return the policy in `data/store_policy.yaml`.

        Raises:
            ValueError: if the file lacks the format, an entry, or an entry's derivation.
        """
        text = resources.files("senselab_core.runtime").joinpath("data", "store_policy.yaml").read_text("utf-8")
        raw = yaml.safe_load(text)
        if not isinstance(raw, dict) or raw.pop("format", None) != POLICY_FORMAT:
            raise ValueError(f"store_policy.yaml must carry format {POLICY_FORMAT!r}")
        return cls.model_validate({name: _value(entry, name) for name, entry in raw.items()})


def committed_modes(policy: StorePolicy) -> tuple[int, int]:
    """Return `(file mode, directory mode)` of committed objects: `0o440` and `0o2550`, plus world read if allowed."""
    if policy.world_readable:
        return 0o440 | _WORLD_READ, 0o2550 | _WORLD_READ_SEARCH
    return 0o440, 0o2550


def shared_dir_mode(policy: StorePolicy) -> int:
    """Return the mode of a directory that holds objects: `0o2770`, plus world read and search if allowed."""
    return STAGING_DIR_MODE | (_WORLD_READ_SEARCH if policy.world_readable else 0)


def set_mode(fd: int, mode: int) -> None:
    """Set `mode` on the open descriptor `fd`; if that raises `PermissionError`, set it without setgid.

    Raises:
        OSError: if the mode cannot be set even without setgid.
    """
    for attempt in (mode, mode & ~stat.S_ISGID):
        try:
            os.fchmod(fd, attempt)
            return
        except PermissionError:
            if attempt == mode & ~stat.S_ISGID:
                raise


_UMASK_LOCK = threading.Lock()


def umask_default_file_mode() -> int:
    """Return the mode `open(..., 0o666)` would give a new file under the process's current umask."""
    with _UMASK_LOCK:
        mask = os.umask(0o077)
        os.umask(mask)
    return 0o666 & ~mask


TEMPORARY_NAME = re.compile(r"^\..+\.[0-9]+\.[0-9]+\.[0-9a-f]{32}\.tmp$")
"""A file being written atomically: `.<name>.<pid>.<thread id>.<uuid4 hex>.tmp`."""
ASIDE_NAME = UUID_HEX
"""An object moved aside: a uuid4 hex."""
WRITTEN_NAMES = frozenset({CLOCK, TEMPORARY_NAME, ASIDE_NAME})
"""The grammars of the file names this module writes under a storage root."""
_CALLER_FILE = re.compile(r".+")
"""A file a caller names, outside any storage root (`write_atomically`)."""


class WrongKindError(ValueError):
    """An object of the wrong kind is at a path the runtime owns: a symlink, or a file where a directory belongs.

    Attributes:
        path: The offending path.
        found: What is there: `"symlink"`, `"directory"`, `"file"` or `"other"`.
    """

    def __init__(self, path: Path, found: str, message: str) -> None:
        """Record `path` and `found`; `message` is the whole text."""
        super().__init__(message)
        self.path = path
        self.found = found


def path_component(name: str) -> str:
    """Return `name` if it is one path component: non-empty, not `.` or `..`, with no slash, backslash or NUL.

    Raises:
        ValueError: otherwise.
    """
    if not isinstance(name, str) or name in ("", ".", "..") or any(c in name for c in "/\\\x00"):
        raise ValueError(f"{name!r} is not a single path component")
    return name


def owner_remedy(path: Path) -> str:
    """Return who owns `path` and what they must run to remove it."""
    try:
        uid = os.lstat(path).st_uid
    except OSError:
        return f"{path} could not be examined; remove it as its owner"
    try:
        owner = pwd.getpwuid(uid).pw_name
    except KeyError:
        owner = f"uid {uid}"
    return (
        f"{path} is owned by {owner} (uid {uid}); as {owner}, run "
        f"`chmod -R u+w '{path}' && rm -rf '{path}'` to remove it"
    )


def _kind(mode: int) -> str:
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISREG(mode):
        return "file"
    return "other"


_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_NEW_FILE_FLAGS = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
_READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_NOCTTY


def _owned(name: str, grammar: re.Pattern[str]) -> str:
    """Return `name` if it is one path component matching `grammar` in full."""
    if not grammar.fullmatch(path_component(name)):
        raise ValueError(f"{name!r} is not a name this writer owns ({grammar.pattern})")
    return name


class _Naming:
    """Re-raise an `OSError` naming `path`: as `storage_path`, and as `filename` when it carries an errno."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> None:
        return None

    def __exit__(self, kind: object, err: object, tb: object) -> None:
        if isinstance(err, OSError) and not isinstance(err, StoreUnavailable):
            if err.errno is not None:
                err.filename = str(self.path)
            err.storage_path = self.path  # type: ignore[attr-defined]


class StorageDir:
    """A directory under a storage root, held open and reached without following any symlink below the root.

    Every name this class takes is one path component (`path_component`); every name it writes a file at
    also matches the `grammar` the caller passes. Every operation is relative to the directory's
    descriptor, and a name is never followed when it is a symlink: `child` and `walk` refuse a symlink or
    a non-directory with `WrongKindError`, files are created with `O_EXCL | O_NOFOLLOW`, and a file that
    changes is replaced whole by a fresh one renamed over its name (`write_atomically`, `fs_now`), so a
    symlink or hard link at the name is replaced and its target never touched. Modes are set only through
    descriptors. The root itself is opened as given, so a root that is a symlink is followed. A
    `StorageDir` is a context manager that closes its descriptor; errors name the full path.

    Attributes:
        path: The directory's path, for messages and for readers that take a path.
        fd: The open descriptor.
    """

    def __init__(self, path: Path, fd: int) -> None:
        """Wrap an open directory descriptor `fd` for `path`; use `open_root`, `child` or `walk` instead."""
        self.path = path
        self.fd = fd

    @classmethod
    def open_root(cls, root: Path, *, create: bool = False, mode: int | None = None) -> "StorageDir":
        """Open the storage root `root`, following it if it is a symlink.

        With `create`, a missing root is made with `mode` (set through its descriptor) as `child` makes a
        directory, and its missing parents, which are outside the root, are made as `os.makedirs` makes them.

        Raises:
            FileNotFoundError: the root is missing and `create` is False.
            WrongKindError: `create` found something other than a directory at a missing root's name.
            OSError: the root cannot be created or opened.
        """
        root = Path(root)
        with _Naming(root):
            try:
                return cls(root, os.open(root, os.O_RDONLY | os.O_DIRECTORY))
            except FileNotFoundError:
                if not create:
                    raise
            os.makedirs(root.parent, exist_ok=True)
        with cls.open_root(root.parent) as parent:
            made = parent.child(root.name, create=True, mode=mode)
        made.path = root
        return made

    @classmethod
    def open_unfollowed(cls, path: Path) -> "StorageDir":
        """Open the directory `path`, refusing it if its last component is a symlink or not a directory.

        Raises:
            OSError: it is missing, a symlink or not a directory, or cannot be opened.
        """
        with _Naming(Path(path)):
            return cls(Path(path), os.open(path, _DIR_FLAGS))

    def close(self) -> None:
        """Close the descriptor."""
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __enter__(self) -> "StorageDir":
        """Return this directory."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the descriptor."""
        self.close()

    def dup(self) -> "StorageDir":
        """Return a second handle on this directory, with its own descriptor."""
        return StorageDir(self.path, os.dup(self.fd))

    def lstat(self, name: str) -> os.stat_result | None:
        """Return the `lstat` of `name`, None if it is missing."""
        with _Naming(self.path / name):
            try:
                return os.lstat(path_component(name), dir_fd=self.fd)
            except FileNotFoundError:
                return None

    def kind(self, name: str) -> str | None:
        """Return what `name` is (`"symlink"`, `"directory"`, `"file"` or `"other"`), None if it is missing."""
        st = self.lstat(name)
        return None if st is None else _kind(st.st_mode)

    def wrong_kind(self, name: str, expected: str) -> WrongKindError:
        """Return the error for `name`, which is not `expected`."""
        found = self.kind(name) or "nothing"
        path = self.path / name
        return WrongKindError(path, found, f"{path} is a {found} where a {expected} belongs; {owner_remedy(path)}")

    def child(self, name: str, *, create: bool = False, mode: int | None = None) -> "StorageDir":
        """Open the subdirectory `name`; with `create`, make it if it is missing.

        A directory this call makes gets `mode`, set through its descriptor; with `mode` None it keeps the
        mode `mkdir` gives under the umask.

        Raises:
            WrongKindError: `name` is a symlink or not a directory.
            FileNotFoundError: `name` is missing and `create` is False.
            OSError: it cannot be created or opened.
        """
        path_component(name)
        path = self.path / name
        for _ in range(3):
            with _Naming(path):
                try:
                    return StorageDir(path, os.open(name, _DIR_FLAGS, dir_fd=self.fd))
                except FileNotFoundError:
                    if not create:
                        raise
                except OSError:
                    if self.kind(name) in (None, "directory"):
                        raise
                    raise self.wrong_kind(name, "directory") from None
                try:
                    os.mkdir(name, 0o700 if mode is not None else 0o777, dir_fd=self.fd)
                except FileExistsError:
                    continue
                made = StorageDir(path, os.open(name, _DIR_FLAGS, dir_fd=self.fd))
            if mode is not None:
                try:
                    made.set_mode(mode)
                except BaseException:
                    made.close()
                    raise
            return made
        raise self.wrong_kind(name, "directory")

    def walk(
        self,
        *parts: str,
        create: bool = False,
        mode: int | None = None,
        aside: Callable[[], "StorageDir | None"] | None = None,
    ) -> "StorageDir":
        """Open `parts` below this directory, one component at a time, as `child` does.

        With `aside`, a component of the wrong kind is moved out of the way (`set_aside` into the
        directory `aside()` returns, which is then closed, or beside itself when it returns None) and
        the component is opened or created again; without it, `WrongKindError` is raised. Intermediate
        descriptors are closed; the returned directory is new even when `parts` is empty.

        Raises:
            WrongKindError: a component is of the wrong kind and `aside` is None, or it cannot be moved.
            FileNotFoundError: a component is missing and `create` is False.
            OSError: a component cannot be created or opened.
        """
        current = self.dup()
        try:
            for part in parts:
                try:
                    nxt = current.child(part, create=create, mode=mode)
                except WrongKindError:
                    if aside is None:
                        raise
                    into = aside()
                    try:
                        current.set_aside(part, into)
                    finally:
                        if into is not None:
                            into.close()
                    nxt = current.child(part, create=create, mode=mode)
                current.close()
                current = nxt
        except BaseException:
            current.close()
            raise
        return current

    def set_aside(self, name: str, into: "StorageDir | None") -> Path:
        """Move `name` to `into/<uuid4 hex>`, or to `<uuid4 hex>` beside it when `into` is None or refuses it.

        The moved name is stamped with the current time (never following it) so retention ages count from
        the move. A symlink or hard link is moved as a name; what it points at is not touched.

        Raises:
            WrongKindError: it cannot be moved at all; the message names its owner and the remedy.
        """
        found = self.kind(name) or "nothing"
        aside = uuid.uuid4().hex
        failure: OSError | None = None
        for target in (into, self) if into is not None else (self,):
            try:
                os.rename(path_component(name), aside, src_dir_fd=self.fd, dst_dir_fd=target.fd)
            except FileNotFoundError:
                return target.path / aside
            except OSError as err:
                failure = err
                continue
            if found == "directory":
                try:
                    target.stamp(aside)
                except OSError:
                    pass
            return target.path / aside
        path = self.path / name
        message = f"{path} is a {found} where it does not belong and cannot be moved: {failure}; {owner_remedy(path)}"
        raise WrongKindError(path, found, message) from failure

    def listdir(self) -> list[str]:
        """Return the names in this directory."""
        with _Naming(self.path):
            return os.listdir(self.fd)

    def readlink(self, name: str) -> str:
        """Return the target of the symlink `name`."""
        with _Naming(self.path / name):
            return os.readlink(path_component(name), dir_fd=self.fd)

    def open_regular(self, name: str) -> int | None:
        """Open the regular file `name` for reading, never following it and never blocking; None if it is missing.

        Raises:
            WrongKindError: `name` is a symlink, a directory, a fifo or anything but a regular file.
        """
        with _Naming(self.path / name):
            try:
                fd = os.open(path_component(name), _READ_FLAGS, dir_fd=self.fd)
            except FileNotFoundError:
                return None
            except OSError:
                if self.kind(name) in (None, "file"):
                    raise
                raise self.wrong_kind(name, "file") from None
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise self.wrong_kind(name, "file")
        return fd

    def read_regular(self, name: str, *, limit: int | None = None) -> bytes | None:
        """Return the bytes of the regular file `name` (`open_regular`), None if it is missing.

        Raises:
            WrongKindError: `name` is not a regular file.
            ValueError: it holds more than `limit` bytes.
        """
        fd = self.open_regular(name)
        if fd is None:
            return None
        with os.fdopen(fd, "rb") as handle, _Naming(self.path / name):
            data = handle.read() if limit is None else handle.read(limit + 1)
        if limit is not None and len(data) > limit:
            raise ValueError(f"{self.path / name} holds more than {limit} bytes")
        return data

    def open_new(self, name: str, mode: int, *, grammar: re.Pattern[str]) -> int:
        """Create the file `name`, which must not exist, with `mode` whatever the umask; return a read-write descriptor.

        Raises:
            ValueError: `name` does not match `grammar`.
            FileExistsError: something, a symlink included, is at `name`.
        """
        _owned(name, grammar)
        with _Naming(self.path / name):
            fd = os.open(name, _NEW_FILE_FLAGS, 0o600, dir_fd=self.fd)
        try:
            set_mode(fd, mode)
        except BaseException:
            os.close(fd)
            raise
        return fd

    def open_or_create(self, name: str, mode: int, *, grammar: re.Pattern[str]) -> tuple[int, bool]:
        """Open `name` read-write without following or blocking, creating it with `mode` if missing.

        Returns `(descriptor, created)`. An existing file is opened as it is; its mode is not changed.

        Raises:
            ValueError: `name` does not match `grammar`.
            FileNotFoundError: it vanished between the two attempts.
            OSError: it is a symlink, a directory or cannot be opened; the caller judges it by `kind`.
        """
        _owned(name, grammar)
        flags = os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_NOCTTY | os.O_CLOEXEC
        with _Naming(self.path / name):
            try:
                fd = os.open(name, flags | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=self.fd)
            except FileExistsError:
                return os.open(name, flags, dir_fd=self.fd), False
        try:
            set_mode(fd, mode)
        except OSError:
            pass
        return fd, True

    def stamp(self, name: str) -> None:
        """Set the times of the directory `name` to now, never following it; anything else is left alone."""
        if self.kind(name) == "directory":
            with _Naming(self.path / name):
                os.utime(path_component(name), dir_fd=self.fd, follow_symlinks=False)

    def write_new(self, name: str, data: bytes, mode: int, *, grammar: re.Pattern[str]) -> None:
        """Create the file `name` holding `data`, fsynced, with `mode` (`open_new`)."""
        with os.fdopen(self.open_new(name, mode, grammar=grammar), "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())

    def symlink(self, target: str, name: str, *, grammar: re.Pattern[str]) -> None:
        """Create the symlink `name` pointing at `target`."""
        _owned(name, grammar)
        with _Naming(self.path / name):
            os.symlink(target, name, dir_fd=self.fd)

    def link(self, name: str, dst: "StorageDir", dst_name: str, *, grammar: re.Pattern[str]) -> None:
        """Hard-link `name` (never followed) at `dst/dst_name`, which must not exist and must match `grammar`."""
        _owned(dst_name, grammar)
        with _Naming(dst.path / dst_name):
            os.link(path_component(name), dst_name, src_dir_fd=self.fd, dst_dir_fd=dst.fd, follow_symlinks=False)

    def rename(
        self, name: str, dst: "StorageDir", dst_name: str, *, grammar: re.Pattern[str], replace: bool = False
    ) -> None:
        """Rename `name` to `dst/dst_name`, which must match `grammar`.

        With `replace` an existing file there is replaced; replacing a symlink or hard link replaces the
        name and never touches its target.
        """
        _owned(dst_name, grammar)
        with _Naming(self.path / name):
            (os.replace if replace else os.rename)(
                path_component(name), dst_name, src_dir_fd=self.fd, dst_dir_fd=dst.fd
            )

    def remove_owned(self, name: str, *, pattern: re.Pattern[str]) -> bool:
        """Remove `name` through `ownership.remove_owned` relative to this descriptor; return whether it did."""
        with _Naming(self.path / name):
            return remove_owned(self.path, self.path / path_component(name), pattern=pattern, root_fd=self.fd)

    def remove_older(self, cutoff: float, *, pattern: re.Pattern[str] = UUID_HEX) -> int:
        """Remove the children named by `pattern` whose mtime is before `cutoff`; return how many.

        A directory among them is made writable by its owner first (through its descriptor), so a sealed
        entry's files can be unlinked. Failures are logged and skip that child.
        """
        removed = 0
        try:
            names = self.listdir()
        except OSError as err:
            logger.warning("cannot list %s to collect it: %s", self.path, err)
            return 0
        for name in names:
            if not pattern.fullmatch(name):
                continue
            try:
                info = self.lstat(name)
                if info is None or info.st_mtime >= cutoff:
                    continue
                if stat.S_ISDIR(info.st_mode):
                    with self.child(name) as child:
                        child.set_mode(STAGING_DIR_MODE)
                removed += self.remove_owned(name, pattern=pattern)
            except (OSError, WrongKindError) as err:
                logger.warning("cannot collect %s: %s", self.path / name, err)
        return removed

    def set_mode(self, mode: int) -> None:
        """Set this directory's mode through its descriptor, as `set_mode` does."""
        with _Naming(self.path):
            set_mode(self.fd, mode)

    def fsync(self) -> None:
        """Sync this directory."""
        with _Naming(self.path):
            os.fsync(self.fd)

    def write_atomically(
        self, name: str, data: bytes, *, grammar: re.Pattern[str], mode: int | None = None
    ) -> os.stat_result:
        """Replace `name` with a fresh file holding `data`, fsynced, renamed over the name; return the new file's stat.

        The fresh file is a temporary (`TEMPORARY_NAME`) created `0o600` with `O_EXCL | O_NOFOLLOW` and then
        given `mode` through its descriptor (`umask_default_file_mode()` when None), before any byte is
        written. A symlink or hard link at `name` is replaced and its target never touched; a directory or
        other non-file there is set aside beside itself first. The temporary is removed if the write fails.

        Raises:
            ValueError: `name` does not match `grammar`.
            WrongKindError: a directory at `name` cannot be moved aside.
            OSError: if the temporary cannot be written or renamed.
        """
        _owned(name, grammar)
        temporary = f".{name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
        with _Naming(self.path / temporary):
            descriptor = os.open(temporary, _NEW_FILE_FLAGS, 0o600, dir_fd=self.fd)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                set_mode(handle.fileno(), mode if mode is not None else umask_default_file_mode())
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
                written = os.fstat(handle.fileno())
            if self.kind(name) not in (None, "file", "symlink"):
                self.set_aside(name, None)
            self.rename(temporary, self, name, grammar=grammar, replace=True)
        except BaseException:
            self.remove_owned(temporary, pattern=TEMPORARY_NAME)
            raise
        return written

    def fs_now(self, name: str = CLOCK_NAME) -> float:
        """Return the current time as the filesystem stamps it: the mtime of a fresh `name` (`write_atomically`).

        On NFS the server sets it, so ages compared with this clock do not depend on the local clock.
        """
        return self.write_atomically(name, b"", grammar=CLOCK, mode=STAGING_FILE_MODE).st_mtime


def write_atomically(target: Path, data: bytes, *, mode: int | None = None) -> None:
    """Replace the caller's file `target` with `data` as `StorageDir.write_atomically` does, beside it.

    The new file gets `mode`, or the mode a regular file at `target` had (read with `lstat`), or the umask's
    default.

    Raises:
        OSError: if the temporary cannot be written or renamed.
    """
    target = Path(target)
    with StorageDir.open_root(target.parent) as directory:
        if mode is None:
            existing = directory.lstat(target.name)
            if existing is not None and stat.S_ISREG(existing.st_mode):
                mode = stat.S_IMODE(existing.st_mode)
        directory.write_atomically(target.name, data, grammar=_CALLER_FILE, mode=mode)


@runtime_checkable
class AddressedFormat(Protocol):
    """A persisted artifact whose location is derived from its content.

    Attributes:
        name: The format's name.
    """

    name: str

    def address_of(self, path: Path) -> str:
        """Return the address re-derived from the bytes at `path`."""
        ...

    def address_at(self, path: Path) -> str:
        """Return the address the location `path` names."""
        ...

    def read(self, path: Path) -> object:
        """Read the artifact at `path` as its readers do; raise the format's error when it is not at its address."""
        ...


if TYPE_CHECKING:
    ADDRESSED_FORMATS: tuple[AddressedFormat, ...]


def __getattr__(name: str) -> object:
    if name == "ADDRESSED_FORMATS":
        from senselab_core.runtime import fetch, store

        return (store.STORE_ENTRY, fetch.BLOB, fetch.DATASET_MARKER)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ADDRESSED_FORMATS",
    "ASIDE_NAME",
    "CLOCK",
    "WRITTEN_NAMES",
    "StorageDir",
    "WrongKindError",
    "owner_remedy",
    "path_component",
    "CLOCK_NAME",
    "TEMPORARY_NAME",
    "STAGING_DIR_MODE",
    "STAGING_FILE_MODE",
    "AddressedFormat",
    "StorageFull",
    "StorePolicy",
    "StoreUnavailable",
    "committed_modes",
    "set_mode",
    "shared_dir_mode",
    "storage_fault",
    "write_atomically",
]
