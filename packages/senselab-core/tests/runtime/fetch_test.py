"""Data manifests and the hash-verified, content-addressed fetch cache."""

import hashlib
import io
import multiprocessing as mp
import os
import stat
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO

import pytest
from senselab_core.cli import main as cli_main
from senselab_core.runtime import fetch as fetch_module
from senselab_core.runtime.fetch import (
    DOWNLOAD_STALE_SECONDS,
    DataFile,
    DataManifest,
    FetchError,
    ManifestError,
    default_cache_root,
    fetch,
)
from senselab_core.runtime.locks import LOCK_DIR_MODE, LOCK_FILE_MODE

pytestmark = pytest.mark.logic

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


def _manifest(files: list[dict[str, Any]] | None = None) -> DataManifest:
    return DataManifest.model_validate(
        {
            "format": "data-manifest@1.0",
            "name": "demo",
            "version": "1",
            "license": LICENSE,
            "terms": "Cite the source.",
            "files": files or [{"path": "a/voice.wav", "url": "https://x/a.wav", "sha256": SHA}],
        }
    )


def _opener(routes: dict[str, bytes], calls: list[str] | None = None) -> Callable[[str], BinaryIO]:
    def opener(url: str) -> BinaryIO:
        if calls is not None:
            calls.append(url)
        if url not in routes:
            raise OSError(f"no route to {url}")
        return io.BytesIO(routes[url])

    return opener


class _Truncated(io.BytesIO):
    def read(self, size: int | None = -1) -> bytes:
        data = super().read(16)
        if not data:
            raise ConnectionResetError("connection dropped midway")
        return data


def test_fetch_verifies_and_caches(tmp_path: Path) -> None:
    """Fetch verifies and caches."""
    paths = fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))
    link = paths["a/voice.wav"]
    assert link.read_bytes() == PAYLOAD
    assert link.parent == tmp_path / "datasets" / "demo@1" / "a"
    assert (tmp_path / "blobs" / SHA).read_bytes() == PAYLOAD
    assert list((tmp_path / "tmp").iterdir()) == []


def test_partial_download_not_cached(tmp_path: Path) -> None:
    """Partial download not cached."""

    def opener(url: str) -> BinaryIO:
        return _Truncated(PAYLOAD)

    with pytest.raises(FetchError) as err:
        fetch(_manifest(), cache_root=tmp_path, opener=opener)
    assert "a/voice.wav" in str(err.value) and "https://x/a.wav" in str(err.value)
    assert not (tmp_path / "blobs").exists() or list((tmp_path / "blobs").iterdir()) == []
    assert list((tmp_path / "tmp").iterdir()) == []


def test_mirror_hash_mismatch_refused(tmp_path: Path) -> None:
    """Mirror hash mismatch refused."""
    manifest = _manifest(
        [{"path": "v.wav", "url": "https://x/a", "sha256": SHA, "mirrors": ["https://m/a"]}],
    )
    routes = {"https://x/a": b"tampered", "https://m/a": b"also wrong"}
    with pytest.raises(FetchError) as err:
        fetch(manifest, cache_root=tmp_path, opener=_opener(routes))
    message = str(err.value)
    assert "v.wav" in message and "https://x/a" in message and "https://m/a" in message
    assert not (tmp_path / "blobs" / SHA).exists()


def test_mirror_with_matching_hash_accepted(tmp_path: Path) -> None:
    """Mirror with matching hash accepted."""
    manifest = _manifest(
        [{"path": "v.wav", "url": "https://x/a", "sha256": SHA, "mirrors": ["https://m/a"]}],
    )
    routes = {"https://x/a": b"tampered", "https://m/a": PAYLOAD}
    paths = fetch(manifest, cache_root=tmp_path, opener=_opener(routes))
    assert paths["v.wav"].read_bytes() == PAYLOAD


def test_second_fetch_is_offline(tmp_path: Path) -> None:
    """Second fetch is offline."""
    first = fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))

    def never(url: str) -> BinaryIO:
        raise AssertionError(f"network call to {url}")

    assert fetch(_manifest(), cache_root=tmp_path, opener=never) == first


def test_corrupt_blob_is_refetched(tmp_path: Path) -> None:
    """Corrupt blob is refetched."""
    fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))
    (tmp_path / "blobs" / SHA).write_bytes(b"bit rot")
    paths = fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))
    assert paths["a/voice.wav"].read_bytes() == PAYLOAD


def test_transient_read_error_does_not_discard_blob(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Transient read error does not discard blob."""
    fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))

    def flaky(path: object) -> str:
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(fetch_module, "sha256_file", flaky)
    with pytest.raises(OSError):
        fetch(_manifest(), cache_root=tmp_path, opener=_opener({}))
    assert (tmp_path / "blobs" / SHA).read_bytes() == PAYLOAD


def test_permissions_independent_of_umask(tmp_path: Path) -> None:
    """Permissions independent of umask."""
    old = os.umask(0o077)
    try:
        paths = fetch(_manifest(), cache_root=tmp_path / "c", opener=_opener({"https://x/a.wav": PAYLOAD}))
    finally:
        os.umask(old)
    root = tmp_path / "c"
    for directory in (root, root / "blobs", root / "tmp", root / "datasets", paths["a/voice.wav"].parent):
        assert stat.S_IMODE(directory.stat().st_mode) == LOCK_DIR_MODE, directory
    assert stat.S_IMODE((root / "blobs" / SHA).stat().st_mode) == LOCK_FILE_MODE


def test_stale_download_removed_but_live_one_kept(tmp_path: Path) -> None:
    """Stale download removed but live one kept."""
    (tmp_path / "tmp").mkdir()
    stale = tmp_path / "tmp" / "stale"
    live = tmp_path / "tmp" / "live"
    stale.write_bytes(b"x")
    live.write_bytes(b"y")
    old = time.time() - DOWNLOAD_STALE_SECONDS - 60
    os.utime(stale, (old, old))
    fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))
    assert not stale.exists()
    assert live.exists()


def _concurrent_fetch(root: str, barrier: "mp.synchronize.Barrier", queue: "mp.Queue[str]") -> None:
    def opener(url: str) -> BinaryIO:
        barrier.wait(60)  # both processes are mid-download before either finishes
        return io.BytesIO(PAYLOAD)

    paths = fetch(_manifest(), cache_root=Path(root), opener=opener)
    queue.put(str(paths["a/voice.wav"]))


def test_concurrent_fetches_one_blob(tmp_path: Path) -> None:
    """Concurrent fetches one blob."""
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(2)
    queue: "mp.Queue[str]" = ctx.Queue()
    procs = [ctx.Process(target=_concurrent_fetch, args=(str(tmp_path), barrier, queue)) for _ in range(2)]
    for p in procs:
        p.start()
    results = [queue.get(timeout=120) for _ in procs]
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    assert results[0] == results[1]
    assert [b.name for b in (tmp_path / "blobs").iterdir()] == [SHA]
    assert Path(results[0]).read_bytes() == PAYLOAD
    assert list((tmp_path / "tmp").iterdir()) == []


MANIFEST_YAML = """\
format: data-manifest@1.0
name: demo
version: "1"
license:
  expression: CC-BY-4.0
  commercial_use: "yes"
  copyleft: none
  attribution: true
  gated: false
  source: https://example.org/license
  verified: {date: "2026-10-01", how: manual}
terms: Cite the source.
files:
  - {path: a.wav, url: "https://x/a.wav", sha256: "%s"}
"""


def test_manifest_loads(tmp_path: Path) -> None:
    """Manifest loads."""
    path = tmp_path / "m.yaml"
    path.write_text(MANIFEST_YAML % SHA)
    manifest = DataManifest.load(path)
    assert manifest.files == (DataFile(path="a.wav", url="https://x/a.wav", sha256=SHA),)
    assert manifest.license.expression == "CC-BY-4.0"


@pytest.mark.parametrize(
    "text",
    [
        "- a list\n",
        "{unclosed: [",
        MANIFEST_YAML.replace("data-manifest@1.0", "data-manifest@9.9") % SHA,
        MANIFEST_YAML % "abc",
        MANIFEST_YAML.replace("a.wav,", "../escape.wav,") % SHA,
        MANIFEST_YAML.replace("a.wav,", "/abs.wav,") % SHA,
        "\n".join(line for line in (MANIFEST_YAML % SHA).splitlines() if not line.startswith("terms")),
    ],
)
def test_malformed_manifest_is_typed_error(tmp_path: Path, text: str) -> None:
    """Malformed manifest is typed error."""
    path = tmp_path / "m.yaml"
    path.write_text(text)
    with pytest.raises(ManifestError):
        DataManifest.load(path)


def test_duplicate_paths_refused() -> None:
    """Duplicate paths refused."""
    entry = {"path": "a.wav", "url": "https://x/a", "sha256": SHA}
    with pytest.raises(ValueError):
        _manifest([entry, entry])


def test_default_cache_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Default cache root."""
    monkeypatch.setenv("SENSELAB_DATA_CACHE", str(tmp_path))
    assert default_cache_root() == tmp_path
    monkeypatch.delenv("SENSELAB_DATA_CACHE")
    assert default_cache_root() == Path.home() / ".cache" / "senselab" / "data"


def test_cli_prints_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Cli prints paths."""
    manifest_path = tmp_path / "m.yaml"
    manifest_path.write_text(MANIFEST_YAML % SHA)
    monkeypatch.setattr(fetch_module, "_urlopen", _opener({"https://x/a.wav": PAYLOAD}))
    code = cli_main(["data", "fetch", str(manifest_path), "--cache", str(tmp_path / "cache")])
    assert code == 0
    assert str(tmp_path / "cache" / "datasets" / "demo@1" / "a.wav") in capsys.readouterr().out


def test_cli_error_is_one_line_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Cli error is one line nonzero."""
    manifest_path = tmp_path / "m.yaml"
    manifest_path.write_text(MANIFEST_YAML % SHA)
    monkeypatch.setattr(fetch_module, "_urlopen", _opener({}))
    assert cli_main(["data", "fetch", str(manifest_path), "--cache", str(tmp_path / "c")]) != 0
    err = capsys.readouterr().err.strip()
    assert len(err.splitlines()) == 1 and "a.wav" in err
    assert cli_main(["data", "fetch", str(tmp_path / "missing.yaml")]) != 0
    assert len(capsys.readouterr().err.strip().splitlines()) == 1


@pytest.mark.network
def test_live_fetch(tmp_path: Path) -> None:
    """Live fetch."""
    url = "https://raw.githubusercontent.com/sensein/senselab/main/LICENSE"
    manifest = _manifest([{"path": "LICENSE", "url": url, "sha256": "0" * 64}])
    with pytest.raises(FetchError):  # a live server answers, but the pinned hash cannot match
        fetch(manifest, cache_root=tmp_path)
