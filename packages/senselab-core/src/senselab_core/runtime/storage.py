"""Storage faults, the store policy, object modes, the storage clock and the address contract.

- `storage_fault` is the one mapping from an `OSError` to a `StoreUnavailable` (or `StorageFull`), used by
  the result store and the data-fetch cache; whether it is transient is `faults.is_transient`.
- `load_policy_values` reads a packaged policy file.
- `StorePolicy` holds the modes and retention ages, read from `data/store_policy.yaml`.
- Committed objects (store entries, fetch blobs and completion markers) are read-only:
  `committed_modes` gives their file and directory modes. Staging objects are `STAGING_FILE_MODE`
  and `STAGING_DIR_MODE`, and directories that hold objects are `shared_dir_mode`.
- `StorageDir` is the one way to reach and write below a storage root (ADR-003): no symlink below the root
  is followed, a mutable file is only ever replaced whole by a fresh file renamed over its name, every file
  name written matches a grammar the caller passes, and modes are set only through a
  descriptor (`set_mode`).
- `StorageDir.publish` makes a write-once file appear whole (staged, fsynced, linked, directory fsynced), and
  `StorageDir.append` adds one whole record to a log or leaves it as it was. `FileOps` is the seam their system
  calls go through, so a test can fail any of them (`using_file_ops`). Run state is written through these and
  `write_atomically` only; `tests/runtime/storage_writes_contract_test.py` refuses any other write under `runtime/`.
- `StorageDir.dir_now` reads the current time from the filesystem itself, from a fresh probe directory, and writes
  no file.
- `StorageUnsupported` is a root without a primitive the runtime needs, such as an atomic `mkdir`.
- `probe_capabilities` measures the primitives the lease protocol needs on a root itself.
- `write_atomically` replaces a caller's file so a reader sees the old or the new bytes, never a mix.
- An `AddressedFormat` is a persisted artifact whose location names its content. `ADDRESSED_FORMATS`
  lists every one: the store entry, the fetch blob and the fetch dataset marker.
"""

import contextlib
import errno
import logging
import os
import pwd
import re
import shutil
import stat
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final, Literal, Protocol, Self, runtime_checkable

from pydantic import Field, StrictBool
from senselab_worker.ownership import UUID_HEX, remove_owned

from senselab_core.kernel._base import Frozen
from senselab_core.runtime.config import load_yaml
from senselab_core.runtime.faults import is_transient

logger = logging.getLogger("senselab")

POLICY_FORMAT = "store-policy@2.0"
STAGING_FILE_MODE = 0o660
STAGING_DIR_MODE = 0o2770
PRIVATE_DIR_MODE: Final = 0o700  # invariant: owner-only access, whatever the site's group policy
"""The mode of a `Staging` directory that only this process's user can write."""
PRIVATE_DIR = "private"
"""A run directory's one root for state that holds paths: substrate state, manifests, placements, worker logs."""
"""The grammar of the storage clock's sentinel."""

_FULL_ERRNOS = frozenset({errno.ENOSPC, errno.EDQUOT})
_UNSUPPORTED_ERRNOS = frozenset({errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS})
_WORLD_READ = 0o004
_WORLD_READ_SEARCH = 0o005


class StoreUnavailable(OSError):
    """Storage could not be read or written; nothing was changed or judged corrupt.

    `errno` and `filename` are those of the original error, which is the `__cause__`; whether a retry may
    succeed is `faults.is_transient` of it.
    """


class StorageFull(StoreUnavailable):
    """The storage is out of space or over quota (`ENOSPC`, `EDQUOT`)."""


class StorageUnsupported(OSError):
    """The storage root lacks a primitive the runtime needs; the message names the primitive and the root.

    `errno` and `filename` are those of the original error, which is the `__cause__`.
    """


def storage_fault(err: OSError, path: Path, *, hint: str | None = None) -> StoreUnavailable:
    """Return the storage fault for `err`, raised while working on `path`; a `StoreUnavailable` is returned as is.

    The fault names the path `err` was raised on by a `StorageDir`, otherwise `path`. The message names the
    cause and the fix: `hint` when given, otherwise one chosen by errno.
    """
    if isinstance(err, StoreUnavailable):
        return err
    path = getattr(err, "storage_path", None) or path
    full = err.errno in _FULL_ERRNOS
    if hint:
        fix = hint
    elif full:
        fix = "out of space or over quota; free space or move the store elsewhere"
    elif is_transient(err):
        fix = "a transient fault; retry"
    else:
        fix = "check the path exists and is a directory you can read and write"
    cause = err.strerror or str(err) or type(err).__name__
    fault = (StorageFull if full else StoreUnavailable)(err.errno, f"{cause}: {fix}", str(path))
    fault.__cause__ = err
    return fault


def load_policy_values(package: str, file_name: str, policy_format: str) -> dict[str, object]:
    """Return `{name: value}` from the packaged `data/<file_name>` of `package`.

    Raises:
        ValueError: if the file lacks `format: <policy_format>`.
    """
    raw = load_yaml(resources.files(package).joinpath("data", file_name), file_name)
    if raw.pop("format", None) != policy_format:
        raise ValueError(f"{file_name} must carry format {policy_format!r}")
    return dict(raw)


class StorePolicy(Frozen):
    """How the result store and the fetch cache set modes and age out leftovers.

    Attributes:
        world_readable: Committed objects and object directories also grant read (and search) to others.
        quarantine_retention_s: Seconds a quarantined entry is kept before `ResultStore.collect` removes it.
        staging_stale_s: Seconds after which a staging directory or download belongs to a crashed writer.
        collect_interval_s: Least seconds between two collections that `ResultStore.put` starts.
        measures_file_bytes_max: Most bytes of one measures file a reader takes; a larger file is malformed.
    """

    world_readable: StrictBool = False
    quarantine_retention_s: Annotated[int, Field(strict=True, gt=0)]
    staging_stale_s: Annotated[int, Field(strict=True, gt=0)]
    collect_interval_s: Annotated[int, Field(strict=True, gt=0)]
    measures_file_bytes_max: Annotated[int, Field(strict=True, gt=0)]

    @classmethod
    def load_default(cls) -> Self:
        """Return the policy in `data/store_policy.yaml`.

        Raises:
            ValueError: if the file lacks the format or an entry.
        """
        return cls.model_validate(load_policy_values("senselab_core.runtime", "store_policy.yaml", POLICY_FORMAT))


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


NAME_MAX = 255
"""The most bytes of one path component: POSIX `NAME_MAX` on ext4, XFS, APFS and NFS."""


def path_component(name: str) -> str:
    """Return `name` if it is one path component: non-empty, not `.` or `..`, no slash, backslash or NUL.

    It is at most `NAME_MAX` bytes in the filesystem encoding.

    Raises:
        ValueError: otherwise.
    """
    if not isinstance(name, str) or name in ("", ".", "..") or any(c in name for c in "/\\\x00"):
        raise ValueError(f"{name!r} is not a single path component")
    if len(os.fsencode(name)) > NAME_MAX:
        raise ValueError(f"{name[:32]!r}... is longer than {NAME_MAX} bytes, the most one path component can hold")
    return name


def temporary_name(name: str) -> str:
    """Return a fresh `TEMPORARY_NAME` for a file being written in place of `name`.

    Raises:
        ValueError: the temporary would not be one path component.
    """
    return path_component(f".{name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp")


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


class FileOps:
    """The system calls `StorageDir` makes to write: `write`, `fsync`, `link` and `ftruncate`.

    This is the one seam a fault is injected through. A test subclasses it and enters `using_file_ops`; the
    methods here call the operating system.
    """

    def write(self, fd: int, data: bytes) -> int:
        """Write `data` to `fd`; return how many bytes it took."""
        return os.write(fd, data)

    def fsync(self, fd: int) -> None:
        """Flush `fd` to stable storage."""
        os.fsync(fd)

    def link(self, src: str, dst: str, *, src_dir_fd: int, dst_dir_fd: int) -> None:
        """Hard-link `src` at `dst`, never following a symlink at `src`."""
        os.link(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd, follow_symlinks=False)

    def ftruncate(self, fd: int, length: int) -> None:
        """Cut `fd` to `length` bytes."""
        os.ftruncate(fd, length)


_FILE_OPS: list[FileOps] = [FileOps()]


@contextlib.contextmanager
def using_file_ops(ops: FileOps) -> Iterator[None]:
    """Make every `StorageDir` write through `ops` for the block."""
    _FILE_OPS.append(ops)
    try:
        yield
    finally:
        _FILE_OPS.pop()


def _write_all(fd: int, data: bytes) -> None:
    """Write all of `data` to `fd`, continuing after a short write.

    Raises:
        OSError: a write failed, or took no bytes (`ENOSPC`).
    """
    view = memoryview(data)
    while view:
        taken = _FILE_OPS[-1].write(fd, bytes(view))
        if taken <= 0:
            raise OSError(errno.ENOSPC, "a write took no bytes")
        view = view[taken:]


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
    changes is replaced whole by a fresh one renamed over its name (`write_atomically`), so a
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

    def stamp(self, name: str) -> None:
        """Set the times of the directory `name` to now, never following it; anything else is left alone."""
        if self.kind(name) == "directory":
            with _Naming(self.path / name):
                os.utime(path_component(name), dir_fd=self.fd, follow_symlinks=False)

    def make_dir(self, name: str, *, grammar: re.Pattern[str], mode: int | None = None, exist_ok: bool = False) -> None:
        """Create the directory `name`; with `mode`, set through its descriptor whatever the umask.

        With `exist_ok`, a directory already at `name` is taken as made (and given `mode`).

        Raises:
            ValueError: `name` does not match `grammar`.
            FileExistsError: something is at `name`, and it is not a directory or `exist_ok` is False.
            StorageUnsupported: the filesystem refuses `mkdir` as an operation it does not support.
        """
        _owned(name, grammar)
        try:
            with _Naming(self.path / name):
                os.mkdir(name, 0o700 if mode is not None else 0o777, dir_fd=self.fd)
        except FileExistsError:
            if not exist_ok or self.kind(name) != "directory":
                raise
        except OSError as err:
            if err.errno not in _UNSUPPORTED_ERRNOS:
                raise
            unsupported = StorageUnsupported(err.errno, f"no atomic mkdir: {err.strerror}", str(self.path / name))
            raise unsupported from err
        if mode is not None:
            with self.child(name) as made:
                made.set_mode(mode)

    def remove_dir(self, name: str, *, grammar: re.Pattern[str]) -> None:
        """Remove the empty directory `name`, never following it."""
        _owned(name, grammar)
        with _Naming(self.path / name):
            os.rmdir(name, dir_fd=self.fd)

    def write_new(self, name: str, data: bytes, mode: int, *, grammar: re.Pattern[str]) -> None:
        """Create the file `name` holding `data`, fsynced, with `mode` (`open_new`)."""
        with os.fdopen(self.open_new(name, mode, grammar=grammar), "wb") as handle:
            _write_all(handle.fileno(), data)
            _FILE_OPS[-1].fsync(handle.fileno())

    def publish(
        self,
        name: str,
        data: bytes,
        *,
        grammar: re.Pattern[str],
        mode: int | None = None,
        on_exists: Literal["keep", "raise"] = "keep",
    ) -> bool:
        """Make `name` hold `data`, complete or absent, and never replace a file already there; return whether it did.

        `data` is written to a temporary and fsynced, linked at `name`, and the directory is fsynced: a reader
        sees no name before its bytes are on disk, and a kill leaves the name absent or whole. The temporary is
        removed. With `on_exists="raise"` a file at `name` raises `FileExistsError`; with `"keep"` it is left and
        False is returned. `mode` is `umask_default_file_mode()` when None.

        Raises:
            ValueError: `name` does not match `grammar`.
            FileExistsError: `on_exists` is `"raise"` and `name` exists.
            OSError: the temporary cannot be written or linked.
        """
        _owned(name, grammar)
        temporary = temporary_name(name)
        self.write_new(temporary, data, mode if mode is not None else umask_default_file_mode(), grammar=TEMPORARY_NAME)
        try:
            self.link(temporary, self, name, grammar=grammar)
        except FileExistsError:
            if on_exists == "raise":
                raise
            return False
        finally:
            self.remove_owned(temporary, pattern=TEMPORARY_NAME)
        self.fsync()
        return True

    def append(self, name: str, data: bytes, *, grammar: re.Pattern[str], mode: int | None = None) -> None:
        """Append `data` to the file `name`, made if missing, as one whole record by this directory's one writer of it.

        A write that takes fewer bytes is continued. If the append cannot finish, the file is cut back to the
        length it had, so no part of `data` stays, and the error is raised. A file whose last byte is not a newline
        (a record a kill cut short) is first ended with one, so `data` starts a line of its own.

        Raises:
            ValueError: `name` does not match `grammar`.
            OSError: the append failed; the file holds only what it held before.
        """
        _owned(name, grammar)
        flags = os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW
        with _Naming(self.path / name):
            fd = os.open(name, flags, mode if mode is not None else 0o644, dir_fd=self.fd)
        try:
            size = os.fstat(fd).st_size
            if size and os.pread(fd, 1, size - 1) != b"\n":
                data = b"\n" + data
            try:
                _write_all(fd, data)
            except BaseException as err:
                try:
                    _FILE_OPS[-1].ftruncate(fd, size)
                except OSError as cut:
                    logger.warning("cannot cut %s back to %d after a failed append: %s", self.path / name, size, cut)
                    err.add_note(f"the file could not be cut back to {size} bytes: {cut}")
                raise
        finally:
            os.close(fd)

    def symlink(self, target: str, name: str, *, grammar: re.Pattern[str]) -> None:
        """Create the symlink `name` pointing at `target`."""
        _owned(name, grammar)
        with _Naming(self.path / name):
            os.symlink(target, name, dir_fd=self.fd)

    def link(self, name: str, dst: "StorageDir", dst_name: str, *, grammar: re.Pattern[str]) -> None:
        """Hard-link `name` (never followed) at `dst/dst_name`, which must not exist and must match `grammar`."""
        _owned(dst_name, grammar)
        with _Naming(dst.path / dst_name):
            _FILE_OPS[-1].link(path_component(name), dst_name, src_dir_fd=self.fd, dst_dir_fd=dst.fd)

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
            _FILE_OPS[-1].fsync(self.fd)

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
        temporary = temporary_name(name)
        with _Naming(self.path / temporary):
            descriptor = os.open(temporary, _NEW_FILE_FLAGS, 0o600, dir_fd=self.fd)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                set_mode(handle.fileno(), mode if mode is not None else umask_default_file_mode())
                _write_all(handle.fileno(), data)
                _FILE_OPS[-1].fsync(handle.fileno())
                written = os.fstat(handle.fileno())
            if self.kind(name) not in (None, "file", "symlink"):
                self.set_aside(name, None)
            self.rename(temporary, self, name, grammar=grammar, replace=True)
            self.fsync()
        except BaseException:
            self.remove_owned(temporary, pattern=TEMPORARY_NAME)
            raise
        return written

    def dir_now(self) -> float:
        """Return the current time as the filesystem stamps it: the mtime of a fresh `ASIDE_NAME` directory, removed.

        Raises:
            StorageUnsupported: the filesystem has no `mkdir` here.
            OSError: the probe cannot be made or read.
        """
        probe = uuid.uuid4().hex
        self.make_dir(probe, grammar=ASIDE_NAME)
        try:
            made = self.lstat(probe)
        finally:
            self.remove_dir(probe, grammar=ASIDE_NAME)
        if made is None:
            raise FileNotFoundError(errno.ENOENT, "the clock probe was removed before it was read", str(self.path))
        return made.st_mtime


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


def publish(
    target: Path, data: bytes, *, mode: int | None = None, on_exists: Literal["keep", "raise"] = "keep"
) -> bool:
    """Publish `data` at the caller's file `target` as `StorageDir.publish` does, beside it; return whether it did.

    Raises:
        FileExistsError: `on_exists` is `"raise"` and `target` exists.
        OSError: the temporary cannot be written or linked.
    """
    target = Path(target)
    with StorageDir.open_root(target.parent) as directory:
        return directory.publish(target.name, data, grammar=_CALLER_FILE, mode=mode, on_exists=on_exists)


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


class Staging:
    """`root/<uuid4 hex>/`, made with `mode` and removed with everything in it on exit.

    `__enter__` returns the directory and `name` is its name under `root`. With `PRIVATE_DIR_MODE` the
    directory must then be owned by this process's user with exactly that mode, so a file staged there
    keeps the bytes written through its descriptor until it is linked or renamed out. A directory that
    cannot be removed on exit is logged, not raised.

    Raises:
        WrongKindError: on enter, a private directory that is not owned by this user with mode `0o700`.
    """

    def __init__(self, root: StorageDir, *, mode: int) -> None:
        """Name a fresh staging directory under `root`, made with `mode` on enter."""
        self.root = root
        self.mode = mode
        self.name = uuid.uuid4().hex
        self._dir: StorageDir | None = None

    def __enter__(self) -> StorageDir:
        """Create the directory and return it open."""
        made = self.root.child(self.name, create=True, mode=self.mode)
        if self.mode == PRIVATE_DIR_MODE:
            try:
                _check_private(made.fd, made.path)
            except WrongKindError:
                made.close()
                raise
        self._dir = made
        return made

    def __exit__(self, *exc: object) -> None:
        """Close the directory and remove it with everything left in it."""
        if self._dir is not None:
            self._dir.close()
        try:
            self.root.remove_owned(self.name, pattern=UUID_HEX)
        except OSError as err:
            logger.warning("cannot remove staging directory %s: %s", self.root.path / self.name, err)


def _check_private(fd: int, path: Path) -> None:
    """Refuse the directory open at `fd` unless this process's user owns it with mode `PRIVATE_DIR_MODE`."""
    st = os.fstat(fd)
    if st.st_uid != os.geteuid() or stat.S_IMODE(st.st_mode) != PRIVATE_DIR_MODE:
        raise WrongKindError(path, "directory", f"{path} is not a private directory")


def private_dir(path: Path) -> Path:
    """Create `path` owner-only, its missing parents under the umask; return it once it is this user's, 0700.

    An existing directory this user owns is set to `PRIVATE_DIR_MODE` whatever the umask; no parent's mode changes.

    Raises:
        WrongKindError: `path` is a symlink or not a directory, or another user owns it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(FileExistsError):
        path.mkdir(mode=PRIVATE_DIR_MODE)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as err:
        if err.errno not in (errno.ELOOP, errno.ENOTDIR):
            raise
        raise WrongKindError(path, "symlink" if path.is_symlink() else "file", f"{path} is not a directory") from err
    try:
        if os.fstat(fd).st_uid == os.geteuid():
            os.fchmod(fd, PRIVATE_DIR_MODE)
        _check_private(fd, path)
    finally:
        os.close(fd)
    return path


PROBE_BOUND_S = 5.0
"""Seconds `probe_capabilities` waits for a directory's mtime to show a change before it judges the clock."""
PROBE_POLL_S = 0.005
"""Seconds between the probe's reads of a directory's mtime."""
_CLOCK_TOLERANCE_S = 3600.0


@dataclass(frozen=True)
class StorageCapabilities:
    """What a root does, as measured: the primitives its leases and queues need, and its attribute delay.

    Attributes:
        atomic_mkdir: `mkdir` of a name that exists fails, so exactly one of two makers wins.
        directory_rename: A directory is renamed whole and a rename onto a non-empty directory fails.
        dir_mtime_server_clock: A directory's mtime moves to the present when an entry is made in it.
        atomic_rename: A rename replaces a file whole.
        exclusive_link: `link` of a name that exists fails, so exactly one of two linkers wins.
        attribute_delay_s: Seconds from an entry being made to the directory's mtime showing it, as this client saw.
    """

    atomic_mkdir: bool
    directory_rename: bool
    dir_mtime_server_clock: bool
    atomic_rename: bool
    exclusive_link: bool
    attribute_delay_s: float

    def missing(self) -> list[str]:
        """Return the primitives this root lacks, by field name."""
        names = ("atomic_mkdir", "directory_rename", "dir_mtime_server_clock", "atomic_rename", "exclusive_link")
        return [n for n in names if not getattr(self, n)]

    def require(self, root: Path) -> None:
        """Raise `StorageUnsupported` naming each primitive of `root` that is missing."""
        lacking = self.missing()
        if lacking:
            raise StorageUnsupported(errno.ENOTSUP, f"no {', '.join(lacking)}", str(root))


def _refused(call: Callable[[], object], *errnos: int) -> bool:
    """Return whether `call` raised an `OSError` of one of `errnos`: the refusal an exclusive operation must give."""
    try:
        call()
    except OSError as err:
        return err.errno in errnos
    return False


def _mtime_after_entry(work: Path) -> tuple[bool, float]:
    """Return whether a new entry moves `work`'s mtime to the present, and the seconds until it showed."""
    old = time.time() - 86400
    os.utime(work, (old, old))
    before = work.stat().st_mtime_ns
    (work / "entry").write_bytes(b"")
    began = time.monotonic()
    while True:
        now = work.stat()
        if now.st_mtime_ns != before:
            return abs(now.st_mtime - time.time()) < _CLOCK_TOLERANCE_S, time.monotonic() - began
        if time.monotonic() - began >= PROBE_BOUND_S:
            return False, PROBE_BOUND_S
        time.sleep(PROBE_POLL_S)


def probe_capabilities(root: Path) -> StorageCapabilities:
    """Measure `root` itself: `mkdir`, directory and file rename, `link` and the directory mtime, in a probe directory.

    A root that does not exist yet is judged by its nearest existing ancestor. The probe directory is removed.

    Raises:
        StorageUnsupported: `mkdir` is refused as an operation the filesystem does not support.
        OSError: the root cannot be written.
    """
    base = Path(root)
    while not base.exists():
        base = base.parent
    work = base / f".senselab-probe-{uuid.uuid4().hex}"
    try:
        try:
            os.mkdir(work)
        except OSError as err:
            if err.errno not in _UNSUPPORTED_ERRNOS:
                raise
            raise StorageUnsupported(err.errno, f"no atomic mkdir: {err.strerror}", str(root)) from err
        os.mkdir(work / "made")
        atomic_mkdir = _refused(lambda: os.mkdir(work / "made"), errno.EEXIST)
        os.mkdir(work / "from")
        os.mkdir(work / "full")
        (work / "full" / "x").write_bytes(b"")
        os.rename(work / "from", work / "to")
        directory_rename = (work / "to").is_dir() and not (work / "from").exists()
        os.mkdir(work / "again")
        directory_rename = directory_rename and _refused(
            lambda: os.rename(work / "again", work / "full"), errno.ENOTEMPTY, errno.EEXIST
        )
        (work / "a").write_bytes(b"new")
        (work / "b").write_bytes(b"old")
        os.rename(work / "a", work / "b")
        replaced = (work / "b").read_bytes() == b"new" and not (work / "a").exists()
        try:
            os.link(work / "b", work / "c")
            exclusive_link = _refused(lambda: os.link(work / "b", work / "c"), errno.EEXIST)
        except OSError:
            exclusive_link = False
        server_clock, delay = _mtime_after_entry(work)
        return StorageCapabilities(atomic_mkdir, directory_rename, server_clock, replaced, exclusive_link, delay)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if TYPE_CHECKING:
    ADDRESSED_FORMATS: tuple[AddressedFormat, ...]


def __getattr__(name: str) -> object:
    if name == "ADDRESSED_FORMATS":
        from senselab_core.runtime import fetch, store  # noqa: PLC0415 - both import this module

        return (store.STORE_ENTRY, fetch.BLOB, fetch.DATASET_MARKER)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ADDRESSED_FORMATS",
    "ASIDE_NAME",
    "FileOps",
    "StorageDir",
    "WrongKindError",
    "owner_remedy",
    "path_component",
    "private_dir",
    "publish",
    "probe_capabilities",
    "NAME_MAX",
    "TEMPORARY_NAME",
    "STAGING_DIR_MODE",
    "STAGING_FILE_MODE",
    "PRIVATE_DIR",
    "PRIVATE_DIR_MODE",
    "Staging",
    "AddressedFormat",
    "StorageCapabilities",
    "StorageFull",
    "StorageUnsupported",
    "StorePolicy",
    "StoreUnavailable",
    "committed_modes",
    "set_mode",
    "shared_dir_mode",
    "storage_fault",
    "temporary_name",
    "using_file_ops",
    "write_atomically",
]
