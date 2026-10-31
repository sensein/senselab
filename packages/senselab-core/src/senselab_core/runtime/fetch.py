"""Versioned data manifests and a hash-verified, content-addressed download cache.

Layout under `cache_root`: `blobs/<sha256>` holds each verified file once, `tmp/<uuid>` is where a download
streams before its hash is checked, and `datasets/<name>@<version>/<path>` holds per-name symlinks into
`blobs/`. Directories are mode `LOCK_DIR_MODE` and files `LOCK_FILE_MODE` whatever the umask.
"""

import os
import re
import time
import urllib.request
import uuid
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Literal

import yaml
from pydantic import ValidationError, field_validator, model_validator
from senselab_worker.hashing import sha256_file

from senselab_core.kernel._base import Frozen, NonEmptyStr
from senselab_core.kernel.license import License
from senselab_core.runtime.locks import LOCK_FILE_MODE, _ensure_dir

MANIFEST_FORMAT = "data-manifest@1.0"
CACHE_ENV = "SENSELAB_DATA_CACHE"
DOWNLOAD_STALE_SECONDS = 24 * 60 * 60
_CHUNK = 1 << 20
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ManifestError(ValueError):
    """A data manifest is unreadable or does not validate."""


class FetchError(RuntimeError):
    """No URL for a manifest file served bytes with the expected sha256."""


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


def _remove_stale_downloads(tmp_root: Path) -> None:
    cutoff = time.time() - DOWNLOAD_STALE_SECONDS
    for child in tmp_root.iterdir():
        try:
            if child.stat().st_mtime < cutoff:
                child.unlink()
        except FileNotFoundError:
            continue


def _download(url: str, opener: Callable[[str], BinaryIO], staged: Path) -> None:
    with opener(url) as source, open(staged, "xb") as out:
        os.fchmod(out.fileno(), LOCK_FILE_MODE)
        while chunk := source.read(_CHUNK):
            out.write(chunk)
        out.flush()
        os.fsync(out.fileno())


def _blob_is_valid(blob: Path, sha256: str) -> bool:
    if not blob.is_file():
        return False
    return sha256_file(blob) == sha256


def _ensure_blob(file: DataFile, root: Path, opener: Callable[[str], BinaryIO]) -> Path:
    blob = root / "blobs" / file.sha256
    if _blob_is_valid(blob, file.sha256):
        return blob
    tried: list[str] = []
    for url in (file.url, *file.mirrors):
        staged = root / "tmp" / uuid.uuid4().hex
        try:
            _download(url, opener, staged)
            actual = sha256_file(staged)
            if actual != file.sha256:
                tried.append(f"{url} (sha256 {actual[:12]}..., expected {file.sha256[:12]}...)")
                continue
            os.replace(staged, blob)
            return blob
        except Exception as err:  # noqa: BLE001 - any failure of one source moves on to the next
            tried.append(f"{url} ({type(err).__name__}: {err})")
        finally:
            staged.unlink(missing_ok=True)
    raise FetchError(
        f"could not fetch {file.path!r}: no source matched sha256 {file.sha256}; tried "
        + "; ".join(tried)
        + ". Check the network and the URLs, or fix the manifest's sha256."
    )


def _link(link: Path, blob: Path) -> None:
    if link.is_symlink() and Path(os.readlink(link)) == Path(os.path.relpath(blob, link.parent)):
        return
    _ensure_dir(link.parent)
    staged = link.with_name(f".{link.name}.{uuid.uuid4().hex}")
    os.symlink(os.path.relpath(blob, link.parent), staged)
    try:
        os.replace(staged, link)
    finally:
        staged.unlink(missing_ok=True)


def fetch(
    manifest: DataManifest,
    *,
    cache_root: Path,
    opener: Callable[[str], BinaryIO] | None = None,
) -> dict[str, Path]:
    """Download every file of `manifest` into the cache and return `{file path: path under datasets/}`.

    A present blob whose sha256 matches is reused with no network call. A download streams to
    `tmp/<uuid>`, is hashed, and is renamed into `blobs/<sha256>` only on a match; on a mismatch or a
    failed transfer the next mirror is tried. Concurrent fetches of one blob are safe: each verifies its own
    staged file and the rename of identical, verified bytes is atomic.

    Raises:
        FetchError: no URL served the expected bytes; the message names the file and every URL tried.
    """
    open_url = opener if opener is not None else _urlopen
    root = Path(cache_root)
    for part in ("", "blobs", "tmp", "datasets"):
        _ensure_dir(root / part)
    _remove_stale_downloads(root / "tmp")
    dataset = root / "datasets" / f"{manifest.name}@{manifest.version}"
    _ensure_dir(dataset)
    result: dict[str, Path] = {}
    for file in manifest.files:
        blob = _ensure_blob(file, root, open_url)
        link = dataset / file.path
        _link(link, blob)
        result[file.path] = link
    return result


__all__ = ["DataFile", "DataManifest", "FetchError", "ManifestError", "default_cache_root", "fetch"]
