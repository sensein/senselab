"""Versioned data manifests and a hash-verified, content-addressed download cache.

Layout under `cache_root`:

- `blobs/<sha256>`: each verified file once, read-only (`committed_modes`);
- `tmp/<uuid>`: where a download streams before its size and hash are checked;
- `datasets/<name>@<version>/files/<path>`: per-name symlinks into `blobs/`;
- `datasets/<name>@<version>/.complete`: the completion marker, read-only.

Manifest paths live only under `files/`, so no manifest path can name the marker. Directories are
`shared_dir_mode`; world bits are set only when the `StorePolicy` allows them. An `OSError` is
raised as `StoreUnavailable` (`StorageFull` when out of space). Only `https` URLs are fetched,
redirects included.
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
from senselab_worker.canonical import is_portable, jcs_dumps, tagged_digest
from senselab_worker.hashing import sha256_file
from senselab_worker.ownership import UUID_HEX, remove_owned

from senselab_core.kernel._base import Frozen, NonEmptyStr
from senselab_core.kernel.license import License
from senselab_core.runtime.locks import ensure_dirs
from senselab_core.runtime.storage import (
    CLOCK_NAME,
    StorePolicy,
    StoreUnavailable,
    committed_modes,
    fs_now,
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
        parts = v.split("/")
        if "\\" in v or "\x00" in v or any(p in ("", ".", "..") or LINK_STAGING_NAME.fullmatch(p) for p in parts):
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
        if "/" in v or v in (".", "..") or "\\" in v or "\x00" in v:
            raise ValueError(f"must not contain path separators, got {v!r}")
        return v

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
        """Return `tagged_digest("data-manifest@3", <the manifest's JSON form>)`."""
        return tagged_digest("data-manifest@3", self.model_dump(mode="json"))

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


def _remove_stale_downloads(root: Path, stale_s: int) -> None:
    """Remove downloads in `root/tmp` older than `stale_s` on the cache's own clock; failures are logged."""
    tmp_root = root / "tmp"
    try:
        cutoff = fs_now(root / CLOCK_NAME) - stale_s
        children = list(tmp_root.iterdir())
    except OSError as err:
        logger.warning("cannot clear stale downloads in %s: %s", tmp_root, err)
        return
    for child in children:
        try:
            if os.lstat(child).st_mtime < cutoff:
                remove_owned(tmp_root, child, pattern=UUID_HEX)
        except FileNotFoundError:
            continue
        except OSError as err:
            logger.warning("cannot remove stale download %s: %s", child, err)


def _download(url: str, opener: Callable[[str], BinaryIO], staged: Path, *, size: int, mode: int = 0o440) -> None:
    """Stream `url` into `staged` (created with `mode`), reading at most `size + 1` bytes.

    Raises:
        FetchError: the URL is not `https`, the opener or a read failed, or the source served more
            or fewer than `size` bytes.
        OSError: `staged` could not be written.
    """
    _require_https(url)
    try:
        source = opener(url)
    except Exception as err:  # noqa: BLE001 - any opener failure is this source's failure
        raise _SourceError(f"{type(err).__name__}: {err}") from err
    received = 0
    with source, open(staged, "xb") as out:
        os.fchmod(out.fileno(), mode)
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


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _commit(staged: Path, final: Path, *, replace: bool, keeps: Callable[[Path], bool]) -> None:
    """Move verified `staged` into `final`, then sync the directory.

    With `replace` False the first writer wins (`os.link`): a `final` that appeared meanwhile is
    kept if `keeps(final)` verifies it, and otherwise atomically replaced by `staged`. With `replace`
    True `final` is atomically replaced, so a reader sees the old or the new bytes, never a missing path.

    Raises:
        StoreUnavailable: the filesystem has no hard links.
        OSError: the move failed.
    """
    try:
        if not replace:
            replace = not _link_first(staged, final) and not keeps(final)
        if replace:
            os.replace(staged, final)
    finally:
        remove_owned(staged.parent, staged, pattern=UUID_HEX)
    _fsync_dir(final.parent)


def _link_first(staged: Path, final: Path) -> bool:
    """Hard-link `staged` at `final`; return False if `final` already exists.

    Raises:
        StoreUnavailable: the filesystem has no hard links.
        OSError: the link failed otherwise.
    """
    try:
        os.link(staged, final)
    except FileExistsError:
        return False
    except OSError as err:
        if err.errno in _NO_HARDLINK_ERRNOS:
            raise storage_fault(
                err,
                final.parent,
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
        return sha256_file(path)

    def address_at(self, path: Path) -> str:
        return path.name

    def read(self, path: Path) -> object:
        actual = self.address_of(path)
        if actual != self.address_at(path):
            raise BlobMismatchError(f"blob {path} holds bytes with sha256 {actual}")
        return path


BLOB = _BlobFormat()


def _blob_state(
    blob: Path, sha256: str, recorded: Mapping[str, BlobStamp], verify: str
) -> Literal["absent", "present", "verified"]:
    """Return whether `blob` is absent (missing, a symlink or not a regular file), present but unverified, or verified.

    With `verify="size"` a blob whose `BlobStamp` equals the recorded one is verified without reading
    it; otherwise its bytes are hashed.
    """
    st = _regular(blob)
    if st is None:
        return "absent"
    if verify == "size" and recorded.get(sha256) == _stamp(st):
        return "verified"
    try:
        BLOB.read(blob)
    except BlobMismatchError:
        return "present"
    return "verified"


def _ensure_blob(
    file: DataFile,
    root: Path,
    opener: Callable[[str], BinaryIO],
    recorded: Mapping[str, BlobStamp],
    verify: str,
    file_mode: int,
) -> Path:
    blob = root / "blobs" / file.sha256
    if _blob_state(blob, file.sha256, recorded, verify) == "verified":
        return blob
    tried: list[str] = []
    for url in (file.url, *file.mirrors):
        staged = root / "tmp" / uuid.uuid4().hex
        try:
            _download(url, opener, staged, size=file.size, mode=file_mode)
            actual = sha256_file(staged)
            if actual != file.sha256:
                tried.append(f"{url} (sha256 {actual[:12]}..., expected {file.sha256[:12]}...)")
                continue
            _commit(
                staged,
                blob,
                replace=os.path.lexists(blob),
                keeps=lambda existing: _blob_state(existing, file.sha256, {}, "full") == "verified",
            )
            return blob
        except _SourceError as err:
            tried.append(f"{url} ({err})")
        finally:
            remove_owned(staged.parent, staged, pattern=UUID_HEX)
    raise FetchError(
        f"could not fetch {file.path!r}: no source served {file.size} bytes with sha256 {file.sha256}; tried "
        + "; ".join(tried)
        + ". Check the network and the URLs, or fix the manifest's size and sha256."
    )


def _link(link: Path, blob: Path, dir_mode: int) -> None:
    """Point `link` at `blob` through a staged symlink and an atomic rename.

    Raises:
        ManifestConflictError: a directory occupies the link's path.
    """
    target = os.path.relpath(blob, link.parent)
    if link.is_symlink() and os.readlink(link) == target:
        return
    if link.is_dir() and not link.is_symlink():
        raise ManifestConflictError(f"{link} is a directory where a file of the manifest belongs; remove it")
    ensure_dirs(link.parent, mode=dir_mode)
    staged = link.with_name(f".{link.name}.{uuid.uuid4().hex}")
    os.symlink(target, staged)
    try:
        os.replace(staged, link)
    finally:
        remove_owned(link.parent, staged, pattern=LINK_STAGING_NAME)


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


def _marker_record(path: Path) -> dict[str, Any] | None:
    """Return the marker at `path` as a JSON object, None if it is not one."""
    try:
        record = json.loads(path.read_bytes().decode("utf-8"))
    except ValueError:
        return None
    return record if isinstance(record, dict) else None


class _DatasetMarkerFormat:
    """The completion marker as an addressed format: `datasets/<name>@<version>/.complete`."""

    name = "fetch-dataset-marker"

    def address_of(self, path: Path) -> str:
        record = _marker_record(path)
        if record is None or not isinstance(record.get("name"), str) or not isinstance(record.get("version"), str):
            raise ManifestConflictError(f"completion marker {path} names no dataset")
        return f"{record['name']}@{record['version']}"

    def address_at(self, path: Path) -> str:
        return path.parent.name

    def read(self, path: Path) -> object:
        if _regular(path) is None:
            raise ManifestConflictError(f"{path} is not a regular file; remove {path.parent} to refetch")
        found = self.address_of(path)
        if found != self.address_at(path):
            raise ManifestConflictError(f"completion marker {path} belongs to {found}; remove {path.parent} to refetch")
        return _marker_record(path)


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


def _read_marker(dataset: Path, manifest: DataManifest) -> tuple[dict[str, BlobStamp], str]:
    """Return `(recorded blob stamps, state)`; state is `"valid"`, `"absent"` or `"corrupt"`.

    A marker that is not a whole `dataset-marker@2.0` record is `"corrupt"` and contributes no
    stamps, but a dataset or manifest digest it still names must be this one.

    Raises:
        ManifestConflictError: the marker is not a regular file, belongs to another dataset, or names
            another manifest.
    """
    path = dataset / COMPLETE_MARKER
    if not os.path.lexists(path):
        return {}, "absent"
    if _regular(path) is None:
        raise ManifestConflictError(f"{path} is not a regular file; remove {dataset} to refetch")
    record = _marker_record(path)
    if record is None:
        logger.warning("ignoring unreadable completion marker %s", path)
        return {}, "corrupt"
    recorded_digest = record.get("manifest_sha256")
    if isinstance(recorded_digest, str) and recorded_digest != manifest.digest():
        raise ManifestConflictError(
            f"{dataset.name} was already fetched from a different manifest (hash {recorded_digest[:12]}...); "
            f"bump the manifest version, or remove {dataset} to refetch"
        )
    try:
        DATASET_MARKER.read(path)
    except ManifestConflictError:
        if isinstance(record.get("name"), str) and isinstance(record.get("version"), str):
            raise
        logger.warning("ignoring malformed completion marker %s", path)
        return {}, "corrupt"
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
        ManifestConflictError: another manifest's marker or files are under this `name@version`.
        StoreUnavailable: the cache could not be read or written; `StorageFull` when out of space.
    """
    policy = policy if policy is not None else StorePolicy.load_default()
    file_mode, _ = committed_modes(policy)
    dir_mode = shared_dir_mode(policy)
    open_url = opener if opener is not None else _urlopen
    root = Path(cache_root)
    dataset = root / "datasets" / f"{manifest.name}@{manifest.version}"
    files_dir = dataset / FILES_DIR
    try:
        for part in ("blobs", "tmp", "datasets"):
            ensure_dirs(root / part, mode=dir_mode)
        _remove_stale_downloads(root, policy.staging_stale_s)
        recorded, marker_state = _read_marker(dataset, manifest)
        ensure_dirs(files_dir, mode=dir_mode)
        if marker_state != "valid" and (foreign := _foreign_files(files_dir, manifest)):
            raise ManifestConflictError(
                f"{dataset} holds files that are not in this manifest ({', '.join(foreign[:3])}); remove it to refetch"
            )
        result: dict[str, Path] = {}
        stamps: dict[str, BlobStamp] = {}
        for file in manifest.files:
            blob = _ensure_blob(file, root, open_url, recorded, verify, file_mode)
            link = files_dir / file.path
            _link(link, blob, dir_mode)
            result[file.path] = link
            stamps[file.sha256] = _stamp(os.lstat(blob))
        if marker_state != "valid" or stamps != recorded:
            staged = root / "tmp" / uuid.uuid4().hex
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
            with open(staged, "xb") as out:
                os.fchmod(out.fileno(), file_mode)
                out.write(jcs_dumps(record))
                out.flush()
                os.fsync(out.fileno())
            _commit(
                staged,
                dataset / COMPLETE_MARKER,
                replace=marker_state != "absent",
                keeps=lambda _: _read_marker(dataset, manifest)[1] == "valid",
            )
        return result
    except OSError as err:
        raise storage_fault(err, root) from err


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
