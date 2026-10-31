"""Versioned data manifests and a hash-verified, content-addressed download cache.

Layout under `cache_root`: `blobs/<sha256>` holds each verified file once, `tmp/<uuid>` is where a download
streams before its hash is checked, and `datasets/<name>@<version>/<path>` holds per-name symlinks into
`blobs/`. Directories are mode `LOCK_DIR_MODE` and files `LOCK_FILE_MODE` whatever the umask.
"""

import errno
import json
import logging
import os
import re
import time
import urllib.request
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Literal

import yaml
from pydantic import ValidationError, field_validator, model_validator
from senselab_worker.canonical import jcs_dumps
from senselab_worker.hashing import sha256_bytes, sha256_file

from senselab_core.kernel._base import Frozen, NonEmptyStr
from senselab_core.kernel.license import License
from senselab_core.runtime.locks import LOCK_FILE_MODE, ensure_dirs

MANIFEST_FORMAT = "data-manifest@1.0"
CACHE_ENV = "SENSELAB_DATA_CACHE"
DOWNLOAD_STALE_SECONDS = 24 * 60 * 60
_CHUNK = 1 << 20
COMPLETE_MARKER = ".complete"
logger = logging.getLogger("senselab")
_STORAGE_ERRNOS = (errno.ENOSPC, errno.EDQUOT)
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ManifestError(ValueError):
    """A data manifest is unreadable or does not validate."""


class FetchError(RuntimeError):
    """No URL for a manifest file served bytes with the expected sha256."""


class StorageError(RuntimeError):
    """The cache directory is full, over quota or otherwise unwritable."""


class ManifestConflictError(ManifestError):
    """A different manifest was already completed under the same `name@version`."""


class DataFile(Frozen):
    """One file of a dataset: its path inside the dataset, its URL, the expected sha256 and optional mirrors."""

    path: NonEmptyStr
    url: NonEmptyStr
    sha256: str
    mirrors: tuple[str, ...] = ()

    @field_validator("path")
    @classmethod
    def _relative(cls, v: str) -> str:
        parts = PurePosixPath(v).parts
        if PurePosixPath(v).is_absolute() or ".." in parts or "\\" in v:
            raise ValueError(f"file path must be relative and stay inside the dataset, got {v!r}")
        return v

    @field_validator("sha256")
    @classmethod
    def _hex(cls, v: str) -> str:
        if not _SHA256.fullmatch(v):
            raise ValueError(f"sha256 must be 64 lowercase hex digits, got {v!r}")
        return v


class DataManifest(Frozen):
    """A versioned dataset: its license and terms, and the hash-pinned files that make it up."""

    format: Literal["data-manifest@1.0"]
    name: NonEmptyStr
    version: NonEmptyStr
    license: License
    terms: NonEmptyStr
    files: tuple[DataFile, ...]

    @field_validator("name", "version")
    @classmethod
    def _plain(cls, v: str) -> str:
        if "/" in v or v in (".", "..") or "\\" in v:
            raise ValueError(f"must not contain path separators, got {v!r}")
        return v

    @model_validator(mode="after")
    def _unique(self) -> "DataManifest":
        paths = [f.path for f in self.files]
        dupes = sorted({p for p in paths if paths.count(p) > 1})
        if dupes:
            raise ValueError(f"duplicate file paths: {dupes}")
        return self

    def digest(self) -> str:
        """Return the sha256 of the manifest's canonical JSON form."""
        return sha256_bytes(jcs_dumps(self.model_dump(mode="json")))

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


def _urlopen(url: str) -> BinaryIO:
    return urllib.request.urlopen(url, timeout=60)  # type: ignore[no-any-return]


class _SourceError(Exception):
    """One URL failed to serve bytes (opener or read error); the next mirror may still work."""


def _remove_stale_downloads(tmp_root: Path) -> None:
    cutoff = time.time() - DOWNLOAD_STALE_SECONDS
    try:
        children = list(tmp_root.iterdir())
    except OSError as err:
        logger.warning("cannot list %s to clear stale downloads: %s", tmp_root, err)
        return
    for child in children:
        try:
            if child.stat().st_mtime < cutoff:
                child.unlink()
        except FileNotFoundError:
            continue
        except OSError as err:
            logger.warning("cannot remove stale download %s: %s", child, err)


def _download(url: str, opener: Callable[[str], BinaryIO], staged: Path) -> None:
    try:
        source = opener(url)
    except Exception as err:  # noqa: BLE001 - any opener failure is this source's failure
        raise _SourceError(f"{type(err).__name__}: {err}") from err
    with source, open(staged, "xb") as out:
        os.fchmod(out.fileno(), LOCK_FILE_MODE)
        while True:
            try:
                chunk = source.read(_CHUNK)
            except Exception as err:  # noqa: BLE001 - a dropped connection is this source's failure
                raise _SourceError(f"{type(err).__name__}: {err}") from err
            if not chunk:
                break
            out.write(chunk)
        out.flush()
        os.fsync(out.fileno())


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _commit(staged: Path, final: Path) -> None:
    """Link `staged` to `final`, first writer wins, then drop `staged` and sync the directory."""
    try:
        os.link(staged, final)
    except FileExistsError:
        pass
    finally:
        staged.unlink(missing_ok=True)
    _fsync_dir(final.parent)


def _blob_ok(blob: Path, sha256: str, recorded_sizes: Mapping[str, int], verify: str) -> bool:
    try:
        size = blob.stat().st_size
    except FileNotFoundError:
        return False
    if verify == "size" and recorded_sizes.get(sha256) == size:
        return True
    return sha256_file(blob) == sha256


def _ensure_blob(
    file: DataFile, root: Path, opener: Callable[[str], BinaryIO], recorded: Mapping[str, int], verify: str
) -> Path:
    blob = root / "blobs" / file.sha256
    present = blob.exists()
    if _blob_ok(blob, file.sha256, recorded, verify):
        return blob
    if present:
        blob.unlink(missing_ok=True)  # the only reason to discard a blob: it does not match its name
    tried: list[str] = []
    for url in (file.url, *file.mirrors):
        staged = root / "tmp" / uuid.uuid4().hex
        try:
            _download(url, opener, staged)
            actual = sha256_file(staged)
            if actual != file.sha256:
                tried.append(f"{url} (sha256 {actual[:12]}..., expected {file.sha256[:12]}...)")
                continue
            _commit(staged, blob)
            return blob
        except _SourceError as err:
            tried.append(f"{url} ({err})")
        finally:
            staged.unlink(missing_ok=True)
    raise FetchError(
        f"could not fetch {file.path!r}: no source matched sha256 {file.sha256}; tried "
        + "; ".join(tried)
        + ". Check the network and the URLs, or fix the manifest's sha256."
    )


def _link(link: Path, blob: Path) -> None:
    target = os.path.relpath(blob, link.parent)
    if link.is_symlink() and os.readlink(link) == target:
        return
    ensure_dirs(link.parent)
    staged = link.with_name(f".{link.name}.{uuid.uuid4().hex}")
    os.symlink(target, staged)
    try:
        os.replace(staged, link)
    finally:
        staged.unlink(missing_ok=True)


def _read_marker(dataset: Path, digest: str, subject: str) -> dict[str, int]:
    """Return the blob sizes recorded by a matching completion marker; `{}` when there is none."""
    try:
        marker = json.loads((dataset / COMPLETE_MARKER).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except ValueError:
        logger.warning("ignoring unreadable completion marker in %s", dataset)
        return {}
    if marker.get("manifest_sha256") != digest:
        raise ManifestConflictError(
            f"{subject} was already fetched from a different manifest "
            f"(hash {str(marker.get('manifest_sha256'))[:12]}...); "
            "bump the manifest version, or remove that datasets/ directory to refetch"
        )
    sizes = marker.get("sizes", {})
    return {str(k): int(v) for k, v in sizes.items()}


def fetch(
    manifest: DataManifest,
    *,
    cache_root: Path,
    opener: Callable[[str], BinaryIO] | None = None,
    verify: Literal["size", "full"] = "size",
) -> dict[str, Path]:
    """Download every file of `manifest` into the cache and return `{file path: path under datasets/}`.

    A present blob is reused with no network call. `verify="size"` trusts a blob whose size equals the one
    recorded when its dataset was completed; `"full"`, and any blob without a recorded size, re-hashes it.
    A download streams to `tmp/<uuid>`, is hashed, and is linked into `blobs/<sha256>` only on a match
    (first writer wins); on a mismatch or failed transfer the next mirror is tried. When every link is in
    place `datasets/<name>@<version>/.complete` records the manifest hash and blob sizes.

    Raises:
        FetchError: no URL served the expected bytes; the message names the file and every URL tried.
        ManifestConflictError: a different manifest was already completed under this `name@version`.
        StorageError: the cache directory is full or over quota.
    """
    open_url = opener if opener is not None else _urlopen
    root = Path(cache_root)
    dataset = root / "datasets" / f"{manifest.name}@{manifest.version}"
    digest = manifest.digest()
    try:
        for part in ("blobs", "tmp", "datasets"):
            ensure_dirs(root / part)
        _remove_stale_downloads(root / "tmp")
        recorded = _read_marker(dataset, digest, f"{manifest.name}@{manifest.version}")
        ensure_dirs(dataset)
        result: dict[str, Path] = {}
        sizes: dict[str, int] = {}
        for file in manifest.files:
            blob = _ensure_blob(file, root, open_url, recorded, verify)
            _link(dataset / file.path, blob)
            result[file.path] = dataset / file.path
            sizes[file.sha256] = blob.stat().st_size
        staged = root / "tmp" / uuid.uuid4().hex
        with open(staged, "xb") as out:
            os.fchmod(out.fileno(), LOCK_FILE_MODE)
            out.write(jcs_dumps({"manifest_sha256": digest, "sizes": sizes}))
            out.flush()
            os.fsync(out.fileno())
        _commit(staged, dataset / COMPLETE_MARKER)
        return result
    except OSError as err:
        if err.errno in _STORAGE_ERRNOS:
            raise StorageError(
                f"cache {root} is out of space or over quota ({err.strerror}); free space or pass --cache DIR elsewhere"
            ) from err
        raise


__all__ = [
    "DataFile",
    "DataManifest",
    "FetchError",
    "ManifestConflictError",
    "ManifestError",
    "StorageError",
    "default_cache_root",
    "fetch",
]
