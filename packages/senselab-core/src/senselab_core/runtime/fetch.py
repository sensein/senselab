"""Versioned data manifests and a hash-verified, content-addressed download cache.

Layout under `cache_root`:

- `blobs/<sha256>`: each verified file once, read-only (`committed_modes`);
- `tmp/<uuid>`: where a download streams before its size and hash are checked;
- `datasets/<name>@<version>/files/<path>`: per-name symlinks into `blobs/`;
- `datasets/<name>@<version>/.complete`: the completion marker, read-only.

Manifest paths live only under `files/`, so no manifest path can name the marker. Directories are
`shared_dir_mode`, whatever the umask; world bits are set only when the `StorePolicy` allows them.
Every directory below `cache_root` is reached through `StorageDir`, so no symlink below the root is
followed: a symlink or a non-directory where `blobs/`, `tmp/`, `datasets/`, a dataset or its `files/`
belongs, and anything but a regular file at `blobs/<sha256>`, is moved into `tmp/<uuid>` (and removed
with the stale downloads); one under `files/` is a `ManifestConflictError`. An `OSError` is raised as
`StoreUnavailable` (`StorageFull` when out of space), naming the path it was raised on. Only `https`
URLs are fetched, redirects included.
"""

import errno
import json
import logging
import os
import re
import stat
import unicodedata
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Mapping
from http.client import HTTPMessage
from pathlib import Path
from typing import IO, Annotated, Any, BinaryIO, Literal, NamedTuple

import yaml
from pydantic import Field, ValidationError, field_validator, model_validator
from senselab_worker.canonical import is_portable, jcs_dumps
from senselab_worker.digest import file_digest, identity_digest
from senselab_worker.ownership import UUID_HEX, remove_owned

from senselab_core.kernel._base import Frozen, NonEmptyStr
from senselab_core.kernel.license import License
from senselab_core.runtime.storage import (
    CLOCK_NAME,
    StorageDir,
    StorePolicy,
    StoreUnavailable,
    WrongKindError,
    committed_modes,
    path_component,
    shared_dir_mode,
    storage_fault,
)

MANIFEST_FORMAT = "data-manifest@2.0"
MARKER_FORMAT = "dataset-marker@2.0"
CACHE_ENV = "SENSELAB_DATA_CACHE"
ALLOWED_SCHEMES = frozenset({"https"})
FILES_DIR = "files"
COMPLETE_MARKER = ".complete"
_CHUNK = 1 << 20
logger = logging.getLogger("senselab")

LINK_STAGING_NAME = re.compile(r"^\..+\.[0-9a-f]{32}$")
"""A symlink being placed: a dot, the link's name, a dot and a uuid4 hex."""
MANIFEST_PART = re.compile(r"[^/\\\x00]+")
"""A link under `files/`, named by the last part of a manifest path."""
STAGED_BLOB = "blob"
STAGED_MARKER = "marker"
STAGED_NAME = re.compile(r"blob|marker")
"""A download or a marker, written in a private staging directory `tmp/<uuid>/`."""
COMMITTED_NAME = re.compile(r"[0-9a-f]{64}|\.complete")
"""A blob `blobs/<sha256>` or a dataset's completion marker, linked or renamed from private staging."""
WRITTEN_NAMES = frozenset({LINK_STAGING_NAME, MANIFEST_PART, STAGED_NAME, COMMITTED_NAME})
"""The grammars of the file names this module writes (besides `storage.WRITTEN_NAMES`)."""
MARKER_LIMIT = 1 << 24
"""The most bytes a completion marker is read to."""
_NO_HARDLINK_ERRNOS = frozenset({errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EXDEV})
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ManifestError(ValueError):
    """A data manifest is unreadable or does not validate, or a cached artifact does not match it."""


class FetchError(RuntimeError):
    """No URL for a manifest file served bytes with the expected size and sha256."""


class ManifestConflictError(ManifestError):
    """The dataset directory holds another manifest's completion marker or files."""


class BlobMismatchError(ManifestError):
    """A blob's bytes do not hash to the name it is stored under."""


class DataFile(Frozen):
    """One file of a dataset: its path inside the dataset, its URL, size, sha256 and optional mirrors.

    `path` is `/`-separated; every part is non-empty, not `.` or `..`, and not a link staging name.
    """

    path: NonEmptyStr
    url: NonEmptyStr
    sha256: str
    size: Annotated[int, Field(strict=True, ge=0)]
    mirrors: tuple[str, ...] = ()

    @field_validator("path")
    @classmethod
    def _relative(cls, v: str) -> str:
        try:
            parts = [path_component(p) for p in v.split("/")]
        except ValueError:
            parts = []
        if not parts or any(LINK_STAGING_NAME.fullmatch(p) for p in parts):
            raise ValueError(
                f"file path must be relative, '/'-separated, with no empty, '.', '..' or staging-name part, got {v!r}"
            )
        return v

    @field_validator("sha256")
    @classmethod
    def _hex(cls, v: str) -> str:
        if not _SHA256.fullmatch(v):
            raise ValueError(f"sha256 must be 64 lowercase hex digits, got {v!r}")
        return v


def path_key(path: str) -> tuple[str, ...]:
    """Return how a filesystem may compare `path`: part by part, by Unicode canonical caseless match.

    Each part whose code points are all assigned in Unicode 13.0 (`canonical.is_portable`) is decomposed
    (NFD), upper-cased (`str.upper`), fully case-folded (`str.casefold`) and recomposed (NFC), so it joins
    what full case folding joins and what a filesystem comparing by upper case joins (`ı` with `I` and
    `i`). Any other part is kept as given, so the key is the same on every supported Python. Two manifest
    paths equal under this key are one path.
    """
    return tuple(_part_key(part) for part in path.split("/"))


def _part_key(part: str) -> str:
    if not is_portable(part):
        return part
    return unicodedata.normalize("NFC", unicodedata.normalize("NFD", part).upper().casefold())


class DataManifest(Frozen):
    """A versioned dataset: its license and terms, and the hash-pinned files that make it up.

    The file paths are prefix-free under `path_key`: no two are equal, and none is a directory
    prefix of another.
    """

    format: Literal["data-manifest@2.0"]
    name: NonEmptyStr
    version: NonEmptyStr
    license: License
    terms: NonEmptyStr
    files: tuple[DataFile, ...]

    @field_validator("name", "version")
    @classmethod
    def _plain(cls, v: str) -> str:
        try:
            return path_component(v)
        except ValueError:
            raise ValueError(f"must be one path component, with no path separators, got {v!r}") from None

    @model_validator(mode="after")
    def _prefix_free(self) -> "DataManifest":
        keys = [path_key(f.path) for f in self.files]
        seen = set(keys)
        if len(seen) != len(keys):
            dupes = sorted({f.path for f, k in zip(self.files, keys, strict=True) if keys.count(k) > 1})
            raise ValueError(f"duplicate file paths: {dupes}")
        nested = sorted(
            f.path for f, k in zip(self.files, keys, strict=True) if any(k[:i] in seen for i in range(1, len(k)))
        )
        if nested:
            raise ValueError(f"file paths inside another file's path: {nested}")
        return self

    def digest(self) -> str:
        """Return `identity_digest("data-manifest@3", <the manifest's JSON form>)`."""
        return identity_digest("data-manifest@3", self.model_dump(mode="json"))

    @classmethod
    def load(cls, path: Path) -> "DataManifest":
        """Read and validate the manifest at `path`; raise `ManifestError` naming the file and the cause."""
        try:
            raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        except OSError as err:
            raise ManifestError(f"cannot read manifest {path}: {err}") from err
        except yaml.YAMLError as err:
            raise ManifestError(f"manifest {path} is not valid YAML: {err}") from err
        if not isinstance(raw, dict):
            raise ManifestError(f"manifest {path} must be a mapping with format {MANIFEST_FORMAT!r}")
        try:
            return cls.model_validate(raw)
        except ValidationError as err:
            first = err.errors()[0]
            where = ".".join(str(p) for p in first["loc"])
            raise ManifestError(f"manifest {path} is invalid at {where}: {first['msg']}") from err


def default_cache_root() -> Path:
    """Return `$SENSELAB_DATA_CACHE`, or `~/.cache/senselab/data`."""
    env = os.environ.get(CACHE_ENV)
    return Path(env) if env else Path.home() / ".cache" / "senselab" / "data"


class _SourceError(FetchError):
    """One URL failed to serve the declared bytes; the next mirror may still work."""


def _require_https(url: str) -> None:
    scheme = urllib.parse.urlsplit(url).scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        raise _SourceError(f"refusing {scheme or 'scheme-less'} URL {url!r}; only {sorted(ALLOWED_SCHEMES)} is fetched")


class _HttpsOnlyRedirect(urllib.request.HTTPRedirectHandler):
    """Follows a redirect only to an `https` URL."""

    def redirect_request(
        self, req: urllib.request.Request, fp: IO[bytes], code: int, msg: str, headers: HTTPMessage, newurl: str
    ) -> urllib.request.Request | None:
        _require_https(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _https_opener() -> urllib.request.OpenerDirector:
    """Return an opener with only the proxy, HTTPS, redirect and HTTP-error handlers: no file, ftp, data or http."""
    opener = urllib.request.OpenerDirector()
    for handler in (
        urllib.request.ProxyHandler(),
        urllib.request.HTTPSHandler(),
        _HttpsOnlyRedirect(),
        urllib.request.HTTPDefaultErrorHandler(),
        urllib.request.HTTPErrorProcessor(),
    ):
        opener.add_handler(handler)
    return opener


def _urlopen(url: str) -> BinaryIO:
    """Open `url` over `https`; raise `FetchError` before connecting if its scheme is not allowed."""
    _require_https(url)
    return _https_opener().open(url, timeout=60)  # type: ignore[no-any-return]


def _remove_stale_downloads(top: StorageDir, tmp: StorageDir, stale_s: int) -> None:
    """Remove downloads in `tmp` older than `stale_s` on the cache's own clock; failures are logged."""
    try:
        cutoff = top.fs_now() - stale_s
    except (OSError, WrongKindError) as err:
        logger.warning("cannot clear stale downloads in %s: %s", tmp.path, err)
        return
    tmp.remove_older(cutoff)


def _download(url: str, opener: Callable[[str], BinaryIO], out: BinaryIO, *, size: int) -> None:
    """Stream `url` into the open file `out`, reading at most `size + 1` bytes.

    Raises:
        FetchError: the URL is not `https`, the opener or a read failed, or the source served more
            or fewer than `size` bytes.
        OSError: the file could not be written.
    """
    _require_https(url)
    try:
        source = opener(url)
    except Exception as err:  # noqa: BLE001 - any opener failure is this source's failure
        raise _SourceError(f"{type(err).__name__}: {err}") from err
    received = 0
    with source:
        while True:
            try:
                chunk = source.read(min(_CHUNK, size + 1 - received))
            except Exception as err:  # noqa: BLE001 - a dropped connection is this source's failure
                raise _SourceError(f"{type(err).__name__}: {err}") from err
            if not chunk:
                break
            received += len(chunk)
            if received > size:
                raise _SourceError(f"served more than the declared {size} bytes")
            out.write(chunk)
        if received < size:
            raise _SourceError(f"ended after {received} of the declared {size} bytes")
        out.flush()
        os.fsync(out.fileno())


class _PrivateStaging:
    """`tmp/<uuid4 hex>/`, made `0o700` by this process and removed with everything in it on exit.

    Only this process's user can create, rename or replace a file in it, so a file staged there keeps
    the bytes written and hashed through its descriptor until it is linked or renamed out.
    """

    def __init__(self, tmp: StorageDir) -> None:
        self.tmp = tmp
        self.name = uuid.uuid4().hex
        self.dir: StorageDir | None = None

    def __enter__(self) -> StorageDir:
        made = self.tmp.child(self.name, create=True, mode=0o700)
        st = os.fstat(made.fd)
        if st.st_uid != os.geteuid() or stat.S_IMODE(st.st_mode) != 0o700:
            made.close()
            raise WrongKindError(made.path, "directory", f"{made.path} is not a private staging directory")
        self.dir = made
        return made

    def __exit__(self, *exc: object) -> None:
        if self.dir is not None:
            self.dir.close()
        self.tmp.remove_owned(self.name, pattern=UUID_HEX)


def _commit(
    source: StorageDir,
    name: str,
    target: StorageDir,
    final: str,
    *,
    replace: bool,
    keeps: Callable[[], bool],
) -> bool:
    """Move verified `source/name` (in a private staging directory) into `target/final`, then sync `target`.

    With `replace` False the first writer wins (a hard link): a `final` that appeared meanwhile is
    kept if `keeps()` verifies it, and otherwise atomically replaced. With `replace` True `final` is
    atomically replaced, so a reader sees the old or the new bytes, never a missing path. A symlink or
    hard link at `final` is replaced as a name. Return whether the staged file is what `final` now names.

    Raises:
        StoreUnavailable: the filesystem has no hard links.
        OSError: the move failed.
    """
    placed = True
    if not replace:
        if not _link_first(source, name, target, final):
            placed = replace = not keeps()
    if replace:
        if target.kind(final) not in (None, "file", "symlink"):
            target.set_aside(final, None)
        source.rename(name, target, final, grammar=COMMITTED_NAME, replace=True)
    target.fsync()
    return placed


def _link_first(source: StorageDir, name: str, target: StorageDir, final: str) -> bool:
    """Hard-link `source/name` at `target/final`; return False if `final` already exists.

    Raises:
        StoreUnavailable: the filesystem has no hard links.
        OSError: the link failed otherwise.
    """
    try:
        source.link(name, target, final, grammar=COMMITTED_NAME)
    except FileExistsError:
        return False
    except OSError as err:
        if err.errno in _NO_HARDLINK_ERRNOS:
            raise storage_fault(
                err,
                target.path,
                hint="this filesystem has no hard links; use a cache root on a filesystem that supports them",
            ) from err
        raise
    return True


class BlobStamp(NamedTuple):
    """What `verify="size"` trusts a blob by: its size, mtime, inode and ctime when its dataset was completed."""

    size: int
    mtime_ns: int
    inode: int
    ctime_ns: int


_STAMP_FIELDS = frozenset(BlobStamp._fields)


def _stamp(st: os.stat_result) -> BlobStamp:
    return BlobStamp(st.st_size, st.st_mtime_ns, st.st_ino, st.st_ctime_ns)


def _regular(path: Path) -> os.stat_result | None:
    """Return the `lstat` of `path` if it is a regular file, None if it is absent or anything else."""
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return None
    return st if stat.S_ISREG(st.st_mode) else None


class _BlobFormat:
    """A blob as an addressed format: `blobs/<sha256 of its bytes>`."""

    name = "fetch-blob"

    def address_of(self, path: Path) -> str:
        if _regular(path) is None:
            raise BlobMismatchError(f"blob {path} is not a regular file")
        return file_digest("file@1", path)

    def address_at(self, path: Path) -> str:
        return path.name

    def read(self, path: Path) -> object:
        actual = self.address_of(path)
        if actual != self.address_at(path):
            raise BlobMismatchError(f"blob {path} holds bytes with sha256 {actual}")
        return path


BLOB = _BlobFormat()


def _hashed(fd: int) -> str:
    """Return the `file@1` digest of the file open on `fd`, read from its start through a second descriptor."""
    with os.fdopen(os.dup(fd), "rb") as handle:
        handle.seek(0)
        return file_digest("file@1", handle)


def _verified_stamp(blobs: StorageDir, sha256: str, recorded: Mapping[str, BlobStamp], verify: str) -> BlobStamp | None:
    """Return the stamp of `blobs/<sha256>` if it is a regular file whose bytes hash to its name, else None.

    It is opened without following or blocking and judged through that descriptor. With `verify="size"`
    a stamp equal to the recorded one (which was taken from a hashed inode) stands for the hash.
    """
    try:
        fd = blobs.open_regular(sha256)
    except WrongKindError:
        return None
    if fd is None:
        return None
    try:
        stamp = _stamp(os.fstat(fd))
        if verify == "size" and recorded.get(sha256) == stamp:
            return stamp
        return stamp if _hashed(fd) == sha256 else None
    finally:
        os.close(fd)


def _ensure_blob(
    file: DataFile,
    blobs: StorageDir,
    tmp: StorageDir,
    opener: Callable[[str], BinaryIO],
    recorded: Mapping[str, BlobStamp],
    verify: str,
    file_mode: int,
) -> tuple[Path, BlobStamp | None]:
    """Return `blobs/<sha256>` holding the file's bytes, and the stamp of the inode this call verified there.

    A download is written and hashed through one descriptor in a private staging directory and linked from
    there. The stamp is None when the name no longer holds the verified inode, so no marker trusts it.
    """
    blob = blobs.path / file.sha256
    stamp = _verified_stamp(blobs, file.sha256, recorded, verify)
    if stamp is not None:
        return blob, stamp
    tried: list[str] = []
    for url in (file.url, *file.mirrors):
        try:
            with _PrivateStaging(tmp) as private:
                with os.fdopen(private.open_new(STAGED_BLOB, file_mode, grammar=STAGED_NAME), "w+b") as out:
                    _download(url, opener, out, size=file.size)
                    actual = _hashed(out.fileno())
                    if actual != file.sha256:
                        tried.append(f"{url} (sha256 {actual[:12]}..., expected {file.sha256[:12]}...)")
                        continue
                    if blobs.kind(file.sha256) not in (None, "file", "symlink"):
                        moved = blobs.set_aside(file.sha256, tmp)
                        logger.warning("moved the object at blob %s aside to %s", blob, moved)
                    kept: list[BlobStamp | None] = []

                    def keeps() -> bool:
                        kept.append(_verified_stamp(blobs, file.sha256, {}, "full"))
                        return kept[-1] is not None

                    placed = _commit(private, STAGED_BLOB, blobs, file.sha256, replace=False, keeps=keeps)
                    if not placed:
                        return blob, kept[-1]
                    private.remove_owned(STAGED_BLOB, pattern=STAGED_NAME)
                    written = os.fstat(out.fileno())
                    now = blobs.lstat(file.sha256)
            same = now is not None and (now.st_dev, now.st_ino) == (written.st_dev, written.st_ino)
            return blob, _stamp(written) if same else None
        except _SourceError as err:
            tried.append(f"{url} ({err})")
    raise FetchError(
        f"could not fetch {file.path!r}: no source served {file.size} bytes with sha256 {file.sha256}; tried "
        + "; ".join(tried)
        + ". Check the network and the URLs, or fix the manifest's size and sha256."
    )


def _link(files: StorageDir, path: str, blob: Path, dir_mode: int) -> Path:
    """Point `files/path` at `blob` through a staged symlink and an atomic rename; return the link's path.

    The directories of `path` are made below `files` with `dir_mode`.

    Raises:
        ManifestConflictError: a directory occupies the link's path, or a symlink or non-directory
            occupies a directory of it.
    """
    *dirs, name = path.split("/")
    try:
        parent = files.walk(*dirs, create=True, mode=dir_mode)
    except WrongKindError as err:
        raise ManifestConflictError(
            f"{err.path} is a {err.found} where a directory of the manifest's {path!r} belongs; remove it"
        ) from err
    with parent:
        link = parent.path / name
        target = os.path.relpath(blob, parent.path)
        kind = parent.kind(name)
        if kind == "symlink" and parent.readlink(name) == target:
            return link
        if kind == "directory":
            raise ManifestConflictError(f"{link} is a directory where a file of the manifest belongs; remove it")
        staged = f".{name}.{uuid.uuid4().hex}"
        parent.symlink(target, staged, grammar=LINK_STAGING_NAME)
        try:
            parent.rename(staged, parent, name, grammar=MANIFEST_PART, replace=True)
        finally:
            parent.remove_owned(staged, pattern=LINK_STAGING_NAME)
    return link


def _stamps(value: object) -> dict[str, BlobStamp] | None:
    if not isinstance(value, dict):
        return None
    stamps: dict[str, BlobStamp] = {}
    for key, raw in value.items():
        if not isinstance(raw, dict) or set(raw) != _STAMP_FIELDS:
            return None
        size, counters = raw["size"], [raw[name] for name in ("mtime_ns", "inode", "ctime_ns")]
        if type(size) is not int or size < 0 or not all(isinstance(v, str) and v.isdecimal() for v in counters):
            return None
        stamps[key] = BlobStamp(size, *(int(v) for v in counters))
    return stamps


def _marker_record(data: bytes) -> dict[str, Any] | None:
    """Return the marker bytes `data` as a JSON object, None if they are not one."""
    try:
        record = json.loads(data.decode("utf-8"))
    except ValueError:
        return None
    return record if isinstance(record, dict) else None


def _marker_address(record: dict[str, Any] | None, path: Path) -> str:
    if record is None or not isinstance(record.get("name"), str) or not isinstance(record.get("version"), str):
        raise ManifestConflictError(f"completion marker {path} names no dataset")
    return f"{record['name']}@{record['version']}"


class _DatasetMarkerFormat:
    """The completion marker as an addressed format: `datasets/<name>@<version>/.complete`."""

    name = "fetch-dataset-marker"

    def _bytes(self, path: Path) -> bytes:
        try:
            with StorageDir.open_root(path.parent) as directory:
                data = directory.read_regular(path.name, limit=MARKER_LIMIT)
        except (WrongKindError, ValueError) as err:
            raise ManifestConflictError(f"{path} is not a regular marker; remove {path.parent} to refetch") from err
        if data is None:
            raise ManifestConflictError(f"completion marker {path} is missing")
        return data

    def address_of(self, path: Path) -> str:
        return _marker_address(_marker_record(self._bytes(path)), path)

    def address_at(self, path: Path) -> str:
        return path.parent.name

    def read(self, path: Path) -> object:
        record = _marker_record(self._bytes(path))
        found = _marker_address(record, path)
        if found != self.address_at(path):
            raise ManifestConflictError(f"completion marker {path} belongs to {found}; remove {path.parent} to refetch")
        return record


DATASET_MARKER = _DatasetMarkerFormat()


def _foreign_files(files_dir: Path, manifest: DataManifest) -> list[str]:
    """Return the paths under `files_dir` that are neither a manifest file nor a directory above one."""
    expected = {f.path for f in manifest.files}
    above = {"/".join(f.path.split("/")[:i]) for f in manifest.files for i in range(1, f.path.count("/") + 1)}
    foreign: list[str] = []
    for current, dirs, names in os.walk(files_dir):
        rel = Path(current).relative_to(files_dir).as_posix()
        prefix = "" if rel == "." else rel + "/"
        foreign.extend(prefix + d for d in dirs if prefix + d not in above)
        dirs[:] = [d for d in dirs if prefix + d in above]
        foreign.extend(prefix + n for n in names if prefix + n not in expected and not LINK_STAGING_NAME.fullmatch(n))
    return sorted(foreign)


def _read_marker(ds: StorageDir, manifest: DataManifest, tmp: StorageDir) -> tuple[dict[str, BlobStamp], str]:
    """Return `(recorded blob stamps, state)`; state is `"valid"`, `"absent"` or `"corrupt"`.

    The marker is read through `ds` without following or blocking. Anything but a regular file there is
    moved into `tmp` and the state is `"corrupt"`. A marker that is not a whole `dataset-marker@2.0`
    record is `"corrupt"` and contributes no stamps, but a dataset or manifest digest it still names must
    be this one.

    Raises:
        ManifestConflictError: the marker belongs to another dataset or names another manifest.
        WrongKindError: an object at the marker cannot be moved aside.
    """
    path = ds.path / COMPLETE_MARKER
    try:
        data = ds.read_regular(COMPLETE_MARKER, limit=MARKER_LIMIT)
    except (WrongKindError, ValueError) as err:
        logger.warning("moving aside the completion marker %s: %s", path, err)
        if ds.kind(COMPLETE_MARKER) not in (None, "file"):
            ds.set_aside(COMPLETE_MARKER, tmp)
        return {}, "corrupt"
    if data is None:
        return {}, "absent"
    record = _marker_record(data)
    if record is None:
        logger.warning("ignoring unreadable completion marker %s", path)
        return {}, "corrupt"
    recorded_digest = record.get("manifest_sha256")
    if isinstance(recorded_digest, str) and recorded_digest != manifest.digest():
        raise ManifestConflictError(
            f"{ds.path.name} was already fetched from a different manifest (hash {recorded_digest[:12]}...); "
            f"bump the manifest version, or remove {ds.path} to refetch"
        )
    try:
        found = _marker_address(record, path)
    except ManifestConflictError:
        logger.warning("ignoring malformed completion marker %s", path)
        return {}, "corrupt"
    if found != ds.path.name:
        raise ManifestConflictError(f"completion marker {path} belongs to {found}; remove {ds.path} to refetch")
    stamps = _stamps(record.get("stamps"))
    if record.get("format") != MARKER_FORMAT or stamps is None or not isinstance(recorded_digest, str):
        logger.warning("ignoring malformed completion marker %s", path)
        return {}, "corrupt"
    return stamps, "valid"


def fetch(
    manifest: DataManifest,
    *,
    cache_root: Path,
    opener: Callable[[str], BinaryIO] | None = None,
    verify: Literal["size", "full"] = "size",
    policy: StorePolicy | None = None,
) -> dict[str, Path]:
    """Download every file of `manifest` into the cache and return `{file path: path under datasets/.../files/}`.

    A present blob is reused with no network call. `verify="size"` trusts a regular-file blob whose
    `BlobStamp` (size, mtime, inode, ctime) equals the one recorded when its dataset was completed; `"full"`,
    and any other blob, re-hashes it. A download streams to `tmp/<uuid>`, is bounded by the declared
    size, is hashed, and is linked into `blobs/<sha256>` only on a match (first writer wins); on a
    mismatch or failed transfer the next mirror is tried. When every link is in place
    `datasets/<name>@<version>/.complete` records the dataset, the manifest digest and the blob stamps;
    it is rewritten when a stamp has changed, so a repaired blob is trusted again by size.
    A dataset without a valid marker must hold no file outside the manifest.

    Raises:
        FetchError: no URL served the expected bytes; the message names the file and every URL tried.
        ManifestConflictError: another manifest's marker or files are under this `name@version`, or a
            symlink or non-directory is where a directory of a manifest path belongs.
        StoreUnavailable: the cache could not be read or written; `StorageFull` when out of space.
        WrongKindError: an object of the wrong kind in the cache could not be moved aside.
    """
    policy = policy if policy is not None else StorePolicy.load_default()
    file_mode, _ = committed_modes(policy)
    dir_mode = shared_dir_mode(policy)
    open_url = opener if opener is not None else _urlopen
    root = Path(cache_root)
    try:
        with StorageDir.open_root(root, create=True, mode=dir_mode) as top:
            with top.walk("tmp", create=True, mode=dir_mode, aside=lambda: None) as tmp:
                return _fetch_into(manifest, top, tmp, open_url, verify, policy)
    except OSError as err:
        raise storage_fault(err, root) from err


def _fetch_into(
    manifest: DataManifest,
    top: StorageDir,
    tmp: StorageDir,
    open_url: Callable[[str], BinaryIO],
    verify: str,
    policy: StorePolicy,
) -> dict[str, Path]:
    file_mode, _ = committed_modes(policy)
    dir_mode = shared_dir_mode(policy)

    def aside() -> StorageDir:
        return StorageDir(tmp.path, os.dup(tmp.fd))

    _remove_stale_downloads(top, tmp, policy.staging_stale_s)
    with (
        top.walk("blobs", create=True, mode=dir_mode, aside=aside) as blobs,
        top.walk("datasets", f"{manifest.name}@{manifest.version}", create=True, mode=dir_mode, aside=aside) as ds,
    ):
        recorded, marker_state = _read_marker(ds, manifest, tmp)
        with ds.walk(FILES_DIR, create=True, mode=dir_mode, aside=aside) as files:
            if marker_state != "valid" and (foreign := _foreign_files(files.path, manifest)):
                raise ManifestConflictError(
                    f"{ds.path} holds files that are not in this manifest ({', '.join(foreign[:3])}); "
                    "remove it to refetch"
                )
            result: dict[str, Path] = {}
            stamps: dict[str, BlobStamp] = {}
            for file in manifest.files:
                blob, stamp = _ensure_blob(file, blobs, tmp, open_url, recorded, verify, file_mode)
                result[file.path] = _link(files, file.path, blob, dir_mode)
                if stamp is not None:
                    stamps[file.sha256] = stamp
        if marker_state != "valid" or stamps != recorded:
            record = {
                "format": MARKER_FORMAT,
                "name": manifest.name,
                "version": manifest.version,
                "manifest_sha256": manifest.digest(),
                "stamps": {
                    sha: {
                        "size": st.size,
                        "mtime_ns": str(st.mtime_ns),
                        "inode": str(st.inode),
                        "ctime_ns": str(st.ctime_ns),
                    }
                    for sha, st in stamps.items()
                },
            }
            with _PrivateStaging(tmp) as private:
                private.write_new(STAGED_MARKER, jcs_dumps(record), file_mode, grammar=STAGED_NAME)
                _commit(
                    private,
                    STAGED_MARKER,
                    ds,
                    COMPLETE_MARKER,
                    replace=marker_state != "absent",
                    keeps=lambda: _read_marker(ds, manifest, tmp)[1] == "valid",
                )
        return result


__all__ = [
    "ALLOWED_SCHEMES",
    "BLOB",
    "DATASET_MARKER",
    "BlobMismatchError",
    "BlobStamp",
    "DataFile",
    "DataManifest",
    "FetchError",
    "ManifestConflictError",
    "ManifestError",
    "StoreUnavailable",
    "default_cache_root",
    "fetch",
    "path_key",
]
