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
import re
import stat
import threading
import uuid
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

    The message names the cause and the fix: `hint` when given, otherwise one chosen by errno.
    """
    if isinstance(err, StoreUnavailable):
        return err
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
    """

    world_readable: StrictBool = False
    quarantine_retention_s: Annotated[int, Field(strict=True, gt=0)]
    staging_stale_s: Annotated[int, Field(strict=True, gt=0)]

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


def write_atomically(target: Path, data: bytes, *, mode: int | None = None) -> None:
    """Replace `target` with `data` through a fsynced temporary file in the same directory.

    The new file gets `mode`, or the mode `target` had, or the umask's default. The temporary is
    removed if the write fails.

    Raises:
        OSError: if the temporary cannot be written or renamed.
    """
    if mode is None:
        try:
            mode = stat.S_IMODE(target.stat().st_mode)
        except FileNotFoundError:
            mode = None
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            if mode is not None:
                os.fchmod(handle.fileno(), mode)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        remove_owned(target.parent, temporary, pattern=TEMPORARY_NAME)
        raise


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
