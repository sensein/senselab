"""The data manifest, cache paths and source stubs the fetch tests share."""

import hashlib
import io
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO

import pytest
from senselab_core.runtime.fetch import DataManifest

PAYLOAD = b"hello senselab data" * 1000
SHA = hashlib.sha256(PAYLOAD).hexdigest()
LICENSE = {
    "expression": "CC-BY-4.0",
    "commercial_use": "yes",
    "copyleft": "none",
    "attribution": True,
    "gated": False,
    "source": "https://example.org/license",
    "verified": {"date": "2026-10-01", "how": "manual"},
}


def file_entry(path: str, url: str, data: bytes = PAYLOAD, **extra: object) -> dict[str, Any]:
    """A manifest file entry serving `data` at `url`."""
    return {"path": path, "url": url, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data), **extra}


def demo_manifest(files: list[dict[str, Any]] | None = None, version: str = "1", name: str = "demo") -> DataManifest:
    """The `demo` manifest of `files` (one file at `a/voice.wav` when None)."""
    return DataManifest.model_validate(
        {
            "format": "data-manifest@2.0",
            "name": name,
            "version": version,
            "license": LICENSE,
            "terms": "Cite the source.",
            "files": files or [file_entry("a/voice.wav", "https://x/a.wav")],
        }
    )


def dataset_dir(root: Path, name: str = "demo", version: str = "1") -> Path:
    """The directory of dataset `name@version` under the cache `root`."""
    return root / "datasets" / f"{name}@{version}"


def unseal(path: Path) -> Path:
    """Make the read-only cache file at `path` writable and return it."""
    path.chmod(0o660)
    return path


def rehashed(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The messages in which fetch re-hashed a cached blob."""
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == "senselab" and r.levelname == "DEBUG" and r.getMessage().startswith("verifying cached blob")
    ]


def opener_of(routes: dict[str, bytes], calls: list[str] | None = None) -> Callable[[str], BinaryIO]:
    """An opener serving each URL of `routes`, recording the URLs asked in `calls`."""

    def opener(url: str) -> BinaryIO:
        if calls is not None:
            calls.append(url)
        if url not in routes:
            raise OSError(f"no route to {url}")
        return io.BytesIO(routes[url])

    return opener


def change_blob(root: Path, change: str) -> None:
    """Change the completed blob under `root`: its times (mtime and ctime), its size, its inode, or its bytes."""
    blob = root / "blobs" / SHA
    if change == "times":
        st = blob.stat()
        os.utime(blob, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    elif change == "size":
        unseal(blob).write_bytes(PAYLOAD + b"x")
    elif change == "inode":
        (root / "copy").write_bytes(PAYLOAD)
        os.replace(root / "copy", blob)
    else:
        unseal(blob).write_bytes(b"POISONED!!!!" + PAYLOAD[12:])
    blob.chmod(0o440)
