"""Storage faults, the store policy, object modes, the storage clock and the address contract.

- `storage_fault` is the one mapping from an `OSError` to a retryable `StoreUnavailable` (or
  `StorageFull`), used by the result store and the data-fetch cache.
- `StorePolicy` holds the modes and retention ages, read from `data/store_policy.yaml`.
- Committed objects (store entries, fetch blobs and completion markers) are read-only:
  `committed_modes` gives their file and directory modes. Staging objects are `STAGING_FILE_MODE`
  and `STAGING_DIR_MODE`, and directories that hold objects are `shared_dir_mode`.
- `fs_now` reads the current time from the filesystem itself, by touching a sentinel file.
- `write_atomically` replaces a small file so a reader sees the old or the new bytes, never a mix.
- An `AddressedFormat` is a persisted artifact whose location names its content. `ADDRESSED_FORMATS`
  lists every one: the store entry, the fetch blob and the fetch dataset marker.
"""

import errno
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
from senselab_worker.ownership import remove_owned

from senselab_core.kernel._base import Frozen

POLICY_FORMAT = "store-policy@1.0"
STAGING_FILE_MODE = 0o660
STAGING_DIR_MODE = 0o2770
CLOCK_NAME = ".clock"
"""The sentinel `fs_now` touches in a directory; it is never removed."""

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


def set_mode(path: Path | str, mode: int, *, fd: int | None = None) -> None:
    """Set `mode` on `path` (or on `fd` when given); if that raises `PermissionError`, set it without setgid.

    Raises:
        OSError: if the mode cannot be set even without setgid.
    """
    for attempt in (mode, mode & ~stat.S_ISGID):
        try:
            if fd is not None:
                os.fchmod(fd, attempt)
            else:
                os.chmod(path, attempt)
            return
        except PermissionError:
            if attempt == mode & ~stat.S_ISGID:
                raise


def fs_now(sentinel: Path) -> float:
    """Return the current time as the filesystem holding `sentinel` stamps it.

    `sentinel` is created if needed (group-writable, so other users can touch it) and its times are
    set to now by the filesystem (on NFS, the server), then its mtime is read. Ages compared with
    this clock do not depend on the local clock.

    Raises:
        OSError: if the sentinel cannot be created or touched.
    """
    fd = os.open(sentinel, os.O_WRONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), STAGING_FILE_MODE)
    try:
        if os.fstat(fd).st_uid == os.geteuid():
            os.fchmod(fd, STAGING_FILE_MODE)
        os.utime(fd)
        return os.fstat(fd).st_mtime
    finally:
        os.close(fd)


TEMPORARY_NAME = re.compile(r"^\..+\.[0-9]+\.[0-9]+\.[0-9a-f]{32}\.tmp$")
"""A file being written atomically: `.<name>.<pid>.<thread id>.<uuid4 hex>.tmp`."""


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
_NEW_FILE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW


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

    Every name this class takes is one path component (`path_component`). Every operation is done relative
    to the directory's descriptor, and a name is never followed when it is a symlink: `child` and `walk`
    refuse a symlink or a non-directory with `WrongKindError`, and files are created with `O_NOFOLLOW`.
    The root itself is opened as given, so a root that is a symlink is followed. A `StorageDir` is a
    context manager that closes its descriptor; errors name the full path.

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

        With `create`, a missing root and its missing parents are created, each with `mode`.

        Raises:
            FileNotFoundError: the root is missing and `create` is False.
            OSError: the root cannot be created or opened.
        """
        root = Path(root)
        if create:
            missing: list[Path] = []
            current = root
            while not os.path.lexists(current):
                missing.append(current)
                current = current.parent
            for directory in reversed(missing):
                with _Naming(directory):
                    try:
                        os.mkdir(directory, 0o700)
                    except FileExistsError:
                        continue
                    if mode is not None:
                        set_mode(directory, mode)
        with _Naming(root):
            fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        return cls(root, fd)

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
        """Open the subdirectory `name`; with `create`, make it with `mode` (or 0o700) if it is missing.

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
                    os.mkdir(name, 0o700, dir_fd=self.fd)
                except FileExistsError:
                    continue
                made = StorageDir(path, os.open(name, _DIR_FLAGS, dir_fd=self.fd))
            try:
                made.set_mode(mode if mode is not None else 0o700)
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
        the component is opened or created again; without it, `WrongKindError` is raised. Intermediate descriptors are
        closed; the returned directory is new even when `parts` is empty.

        Raises:
            WrongKindError: a component is of the wrong kind and `aside` is None, or it cannot be moved.
            FileNotFoundError: a component is missing and `create` is False.
            OSError: a component cannot be created or opened.
        """
        current = StorageDir(self.path, os.dup(self.fd))
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

        The moved object is stamped with the current time so retention ages count from the move. A
        symlink is moved, never followed.

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
            try:
                os.utime(aside, dir_fd=target.fd, follow_symlinks=False)
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

    def open_new(self, name: str, mode: int) -> int:
        """Create the file `name`, which must not exist, with `mode` whatever the umask; return its descriptor."""
        with _Naming(self.path / name):
            fd = os.open(path_component(name), _NEW_FILE_FLAGS, 0o600, dir_fd=self.fd)
        try:
            set_mode(self.path / name, mode, fd=fd)
        except BaseException:
            os.close(fd)
            raise
        return fd

    def write_new(self, name: str, data: bytes, mode: int) -> None:
        """Create the file `name` holding `data`, fsynced, with `mode`."""
        with os.fdopen(self.open_new(name, mode), "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())

    def symlink(self, target: str, name: str) -> None:
        """Create the symlink `name` pointing at `target`."""
        with _Naming(self.path / name):
            os.symlink(target, path_component(name), dir_fd=self.fd)

    def link(self, name: str, dst: "StorageDir", dst_name: str) -> None:
        """Hard-link `name` (never followed) at `dst/dst_name`, which must not exist."""
        with _Naming(dst.path / dst_name):
            os.link(
                path_component(name),
                path_component(dst_name),
                src_dir_fd=self.fd,
                dst_dir_fd=dst.fd,
                follow_symlinks=False,
            )

    def rename(self, name: str, dst: "StorageDir", dst_name: str, *, replace: bool = False) -> None:
        """Rename `name` to `dst/dst_name`; with `replace` an existing file there is replaced atomically."""
        with _Naming(self.path / name):
            (os.replace if replace else os.rename)(
                path_component(name), path_component(dst_name), src_dir_fd=self.fd, dst_dir_fd=dst.fd
            )

    def remove_owned(self, name: str, *, pattern: re.Pattern[str]) -> bool:
        """Remove `name` through `ownership.remove_owned` relative to this descriptor; return whether it did."""
        with _Naming(self.path / name):
            return remove_owned(self.path, self.path / path_component(name), pattern=pattern, root_fd=self.fd)

    def set_mode(self, mode: int) -> None:
        """Set this directory's mode, as `set_mode` does."""
        with _Naming(self.path):
            set_mode(self.path, mode, fd=self.fd)

    def fsync(self) -> None:
        """Sync this directory."""
        with _Naming(self.path):
            os.fsync(self.fd)

    def write_atomically(self, name: str, data: bytes, *, mode: int | None = None) -> None:
        """Replace `name` with `data` through a fsynced temporary file in this directory.

        The new file gets `mode`, or the mode the file had, or the umask's default. The temporary is
        removed if the write fails.

        Raises:
            OSError: if the temporary cannot be written or renamed.
        """
        if mode is None:
            st = self.lstat(name)
            mode = stat.S_IMODE(st.st_mode) if st is not None and stat.S_ISREG(st.st_mode) else None
        temporary = f".{path_component(name)}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
        with _Naming(self.path / temporary):
            descriptor = os.open(temporary, _NEW_FILE_FLAGS, 0o666, dir_fd=self.fd)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                if mode is not None:
                    os.fchmod(handle.fileno(), mode)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            self.rename(temporary, self, name, replace=True)
        except BaseException:
            self.remove_owned(temporary, pattern=TEMPORARY_NAME)
            raise


def write_atomically(target: Path, data: bytes, *, mode: int | None = None) -> None:
    """Replace `target` with `data` as `StorageDir.write_atomically` does, in the directory holding `target`.

    Raises:
        OSError: if the temporary cannot be written or renamed.
    """
    target = Path(target)
    with StorageDir.open_root(target.parent) as directory:
        directory.write_atomically(target.name, data, mode=mode)


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
    "fs_now",
    "set_mode",
    "shared_dir_mode",
    "storage_fault",
    "write_atomically",
]
