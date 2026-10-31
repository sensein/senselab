"""Data manifests and the hash-verified, content-addressed fetch cache."""

import errno
import hashlib
import io
import json
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
    ManifestConflictError,
    ManifestError,
    StorageError,
    default_cache_root,
    fetch,
)
from senselab_core.runtime.locks import LOCK_DIR_MODE, LOCK_FILE_MODE

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


def _manifest(files: list[dict[str, Any]] | None = None, version: str = "1") -> DataManifest:
    return DataManifest.model_validate(
        {
            "format": "data-manifest@1.0",
            "name": "demo",
            "version": version,
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


@pytest.mark.logic
def test_fetch_verifies_and_caches(tmp_path: Path) -> None:
    """Fetch verifies and caches."""
    paths = fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))
    link = paths["a/voice.wav"]
    assert link.read_bytes() == PAYLOAD
    assert link.parent == tmp_path / "datasets" / "demo@1" / "a"
    assert (tmp_path / "blobs" / SHA).read_bytes() == PAYLOAD
    assert list((tmp_path / "tmp").iterdir()) == []


@pytest.mark.logic
def test_partial_download_not_cached(tmp_path: Path) -> None:
    """Partial download not cached."""

    def opener(url: str) -> BinaryIO:
        return _Truncated(PAYLOAD)

    with pytest.raises(FetchError) as err:
        fetch(_manifest(), cache_root=tmp_path, opener=opener)
    assert "a/voice.wav" in str(err.value) and "https://x/a.wav" in str(err.value)
    assert not (tmp_path / "blobs").exists() or list((tmp_path / "blobs").iterdir()) == []
    assert list((tmp_path / "tmp").iterdir()) == []


@pytest.mark.logic
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


@pytest.mark.logic
def test_mirror_with_matching_hash_accepted(tmp_path: Path) -> None:
    """Mirror with matching hash accepted."""
    manifest = _manifest(
        [{"path": "v.wav", "url": "https://x/a", "sha256": SHA, "mirrors": ["https://m/a"]}],
    )
    routes = {"https://x/a": b"tampered", "https://m/a": PAYLOAD}
    paths = fetch(manifest, cache_root=tmp_path, opener=_opener(routes))
    assert paths["v.wav"].read_bytes() == PAYLOAD


@pytest.mark.logic
def test_second_fetch_is_offline(tmp_path: Path) -> None:
    """Second fetch is offline."""
    first = fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))

    def never(url: str) -> BinaryIO:
        raise AssertionError(f"network call to {url}")

    assert fetch(_manifest(), cache_root=tmp_path, opener=never) == first


@pytest.mark.logic
def test_corrupt_blob_is_refetched(tmp_path: Path) -> None:
    """Corrupt blob is refetched."""
    fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))
    (tmp_path / "blobs" / SHA).write_bytes(b"bit rot")
    paths = fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))
    assert paths["a/voice.wav"].read_bytes() == PAYLOAD


@pytest.mark.logic
def test_transient_read_error_does_not_discard_blob(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Transient read error does not discard blob."""
    fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))

    def flaky(path: object) -> str:
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(fetch_module, "sha256_file", flaky)
    with pytest.raises(OSError):
        fetch(_manifest(), cache_root=tmp_path, opener=_opener({}), verify="full")
    assert (tmp_path / "blobs" / SHA).read_bytes() == PAYLOAD


@pytest.mark.logic
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


@pytest.mark.logic
def test_stale_download_removed_but_live_one_kept(tmp_path: Path) -> None:
    """Stale download removed but live one kept."""
    (tmp_path / "tmp").mkdir()
    stale = tmp_path / "tmp" / ("5" * 32)
    live = tmp_path / "tmp" / ("1" * 32)
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


@pytest.mark.logic
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


@pytest.mark.logic
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
@pytest.mark.logic
def test_malformed_manifest_is_typed_error(tmp_path: Path, text: str) -> None:
    """Malformed manifest is typed error."""
    path = tmp_path / "m.yaml"
    path.write_text(text)
    with pytest.raises(ManifestError):
        DataManifest.load(path)


@pytest.mark.logic
def test_duplicate_paths_refused() -> None:
    """Duplicate paths refused."""
    entry = {"path": "a.wav", "url": "https://x/a", "sha256": SHA}
    with pytest.raises(ValueError):
        _manifest([entry, entry])


@pytest.mark.logic
def test_default_cache_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Default cache root."""
    monkeypatch.setenv("SENSELAB_DATA_CACHE", str(tmp_path))
    assert default_cache_root() == tmp_path
    monkeypatch.delenv("SENSELAB_DATA_CACHE")
    assert default_cache_root() == Path.home() / ".cache" / "senselab" / "data"


@pytest.mark.logic
def test_cli_prints_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Cli prints paths."""
    manifest_path = tmp_path / "m.yaml"
    manifest_path.write_text(MANIFEST_YAML % SHA)
    monkeypatch.setattr(fetch_module, "_urlopen", _opener({"https://x/a.wav": PAYLOAD}))
    code = cli_main(["data", "fetch", str(manifest_path), "--cache", str(tmp_path / "cache")])
    assert code == 0
    assert str(tmp_path / "cache" / "datasets" / "demo@1" / "a.wav") in capsys.readouterr().out


@pytest.mark.logic
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


@pytest.mark.logic
def test_nested_dirs_are_group_writable_under_restrictive_umask(tmp_path: Path) -> None:
    """Every directory created under the cache, at every level, is 0o2775."""
    manifest = _manifest([{"path": "sub-01/ses-01/a.wav", "url": "https://x/a.wav", "sha256": SHA}])
    old = os.umask(0o077)
    try:
        paths = fetch(manifest, cache_root=tmp_path / "deep" / "cache", opener=_opener({"https://x/a.wav": PAYLOAD}))
    finally:
        os.umask(old)
    leaf = paths["sub-01/ses-01/a.wav"].parent
    directory = leaf
    while directory != tmp_path / "deep":
        assert stat.S_IMODE(directory.stat().st_mode) == LOCK_DIR_MODE, directory
        directory = directory.parent
    marker = tmp_path / "deep" / "cache" / "datasets" / "demo@1" / ".complete"
    assert stat.S_IMODE(marker.stat().st_mode) == LOCK_FILE_MODE


@pytest.mark.logic
def test_size_mode_skips_rehash_and_full_mode_catches_same_size_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache hit checks size only by default; verify='full' re-hashes."""
    fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))
    hashed: list[object] = []
    real = fetch_module.sha256_file

    def spy(path: Path) -> str:
        hashed.append(path)
        return real(path)

    monkeypatch.setattr(fetch_module, "sha256_file", spy)
    fetch(_manifest(), cache_root=tmp_path, opener=_opener({}))
    assert hashed == []
    (tmp_path / "blobs" / SHA).write_bytes(b"x" * len(PAYLOAD))
    paths = fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}), verify="full")
    assert paths["a/voice.wav"].read_bytes() == PAYLOAD
    assert hashed


@pytest.mark.logic
def test_first_writer_wins_blob_commit(tmp_path: Path) -> None:
    """An existing blob is never replaced by a later commit; the staged file is dropped."""
    staged = tmp_path / ("5" * 32)
    final = tmp_path / "blob"
    final.write_bytes(b"first")
    staged.write_bytes(b"second")
    fetch_module._commit(staged, final)
    assert final.read_bytes() == b"first"
    assert not staged.exists()


@pytest.mark.logic
def test_completion_marker_records_manifest_hash(tmp_path: Path) -> None:
    """The dataset directory carries a .complete marker naming the manifest hash."""
    manifest = _manifest()
    fetch(manifest, cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))
    marker = json.loads((tmp_path / "datasets" / "demo@1" / ".complete").read_text())
    assert marker["manifest_sha256"] == manifest.digest()


@pytest.mark.logic
def test_dataset_without_marker_is_incomplete_and_completed_by_fetch(tmp_path: Path) -> None:
    """A dataset whose fetch died before the marker is refetched and then marked complete."""
    calls: list[str] = []
    fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))
    (tmp_path / "datasets" / "demo@1" / ".complete").unlink()
    fetch(_manifest(), cache_root=tmp_path, opener=_opener({}, calls))
    assert calls == []  # blob already verified by hash, so no network
    assert (tmp_path / "datasets" / "demo@1" / ".complete").exists()


@pytest.mark.logic
def test_same_name_version_different_manifest_raises(tmp_path: Path) -> None:
    """A different manifest under a completed name@version is refused, links untouched."""
    first = fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))
    other = b"other bytes"
    changed = _manifest([{"path": "a/voice.wav", "url": "https://x/b", "sha256": hashlib.sha256(other).hexdigest()}])
    with pytest.raises(ManifestConflictError):
        fetch(changed, cache_root=tmp_path, opener=_opener({"https://x/b": other}))
    assert first["a/voice.wav"].read_bytes() == PAYLOAD


@pytest.mark.logic
def test_disk_full_is_storage_error_not_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ENOSPC while staging is reported as a storage error."""

    def full(path: object) -> str:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(fetch_module, "sha256_file", full)
    with pytest.raises(StorageError) as err:
        fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))
    assert "network" not in str(err.value) and "space" in str(err.value)


@pytest.mark.logic
@pytest.mark.skipif(os.geteuid() == 0, reason="root removes a directory whatever its mode")
def test_stale_cleanup_tolerates_unremovable_entries(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A stale entry that cannot be removed does not abort the fetch; the failure is logged."""
    odd = tmp_path / "tmp" / ("5" * 32)
    odd.mkdir(parents=True)
    (odd / "inner").write_bytes(b"x")
    odd.chmod(0o500)
    old = time.time() - DOWNLOAD_STALE_SECONDS - 60
    os.utime(odd, (old, old))
    try:
        with caplog.at_level("WARNING", logger="senselab"):
            fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))
    finally:
        odd.chmod(0o700)
    assert "cannot remove stale download" in caplog.text


@pytest.mark.logic
def test_cli_oserror_names_cause_and_fix(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """An unwritable cache prints the cause and the fix on one line."""
    manifest_path = tmp_path / "m.yaml"
    manifest_path.write_text(MANIFEST_YAML % SHA)
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert cli_main(["data", "fetch", str(manifest_path), "--cache", str(blocker / "c")]) == 1
    err = capsys.readouterr().err.strip()
    assert len(err.splitlines()) == 1 and "--cache" in err


@pytest.mark.logic
def test_cli_verify_maps_to_full(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--verify passes verify='full' to fetch."""
    manifest_path = tmp_path / "m.yaml"
    manifest_path.write_text(MANIFEST_YAML % SHA)
    seen: list[str] = []

    def spy(manifest: DataManifest, *, cache_root: Path, verify: str) -> dict[str, Path]:
        seen.append(verify)
        return {}

    monkeypatch.setattr("senselab_core.cli.fetch", spy)
    cli_main(["data", "fetch", str(manifest_path), "--cache", str(tmp_path)])
    cli_main(["data", "fetch", str(manifest_path), "--cache", str(tmp_path), "--verify"])
    assert seen == ["size", "full"]


def _repair_fetch(root: str, started: "mp.synchronize.Event", release: "mp.synchronize.Event") -> None:
    def opener(url: str) -> BinaryIO:
        started.set()
        release.wait(60)
        return io.BytesIO(PAYLOAD)

    fetch(_manifest(), cache_root=Path(root), opener=opener, verify="full")


@pytest.mark.logic
def test_blob_path_exists_throughout_repair(tmp_path: Path) -> None:
    """While a corrupt blob is re-downloaded, a concurrent reader's path always exists."""
    fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))
    blob = tmp_path / "blobs" / SHA
    blob.write_bytes(b"bit rot")
    ctx = mp.get_context("spawn")
    started, release = ctx.Event(), ctx.Event()
    proc = ctx.Process(target=_repair_fetch, args=(str(tmp_path), started, release))
    proc.start()
    try:
        assert started.wait(120)
        assert blob.exists() and blob.read_bytes() == b"bit rot"
    finally:
        release.set()
        proc.join(60)
    assert proc.exitcode == 0
    assert blob.read_bytes() == PAYLOAD


@pytest.mark.logic
def test_failed_repair_leaves_corrupt_blob_untouched(tmp_path: Path) -> None:
    """If every mirror fails, the corrupt blob stays and FetchError is raised."""
    fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))
    blob = tmp_path / "blobs" / SHA
    blob.write_bytes(b"bit rot")
    with pytest.raises(FetchError):
        fetch(_manifest(), cache_root=tmp_path, opener=_opener({}))
    assert blob.read_bytes() == b"bit rot"


@pytest.mark.logic
def test_corrupt_marker_is_replaced_once_dataset_verifies(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A corrupt marker is rewritten, so later fetches stop warning."""
    fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))
    marker = tmp_path / "datasets" / "demo@1" / ".complete"
    marker.write_text("{not json")
    fetch(_manifest(), cache_root=tmp_path, opener=_opener({}))
    assert json.loads(marker.read_text())["manifest_sha256"] == _manifest().digest()
    caplog.clear()
    with caplog.at_level("WARNING", logger="senselab"):
        fetch(_manifest(), cache_root=tmp_path, opener=_opener({}))
    assert caplog.text == ""


@pytest.mark.logic
@pytest.mark.parametrize(
    "text",
    [
        "[]",
        "null",
        "{}",
        '{"manifest_sha256": 1, "sizes": {}}',
        '{"manifest_sha256": "x", "sizes": []}',
        '{"manifest_sha256": "x", "sizes": {"a": "big"}}',
        '{"manifest_sha256": "x", "sizes": {"a": true}}',
        '{"manifest_sha256": "x", "sizes": {"a": -1}}',
        "\xff",
    ],
)
def test_malformed_marker_is_treated_as_absent(tmp_path: Path, text: str) -> None:
    """Any malformed marker is logged and ignored, never raised."""
    fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))
    marker = tmp_path / "datasets" / "demo@1" / ".complete"
    marker.write_bytes(text.encode("latin-1"))
    paths = fetch(_manifest(), cache_root=tmp_path, opener=_opener({}))
    assert paths["a/voice.wav"].read_bytes() == PAYLOAD
    assert json.loads(marker.read_text())["manifest_sha256"] == _manifest().digest()


@pytest.mark.logic
def test_no_hardlink_support_is_storage_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A filesystem without hard links gets a StorageError naming the fix."""

    def refuse(src: object, dst: object) -> None:
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(os, "link", refuse)
    with pytest.raises(StorageError) as err:
        fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))
    assert "hard links" in str(err.value)


LIVE_URL = "https://raw.githubusercontent.com/sensein/senselab/5e2f8755de12c256f80437e7f32216b47afa91f8/LICENSE.md"
LIVE_SHA = "75009e0aaa9374223af0c954d175a91d39407acf13bc8e2ae2c882f6a1904f8d"


@pytest.mark.network
def test_live_fetch(tmp_path: Path) -> None:
    """A real download of an immutable public file lands in blobs/ and verifies."""
    manifest = _manifest([{"path": "LICENSE.md", "url": LIVE_URL, "sha256": LIVE_SHA}])
    paths = fetch(manifest, cache_root=tmp_path)
    blob = tmp_path / "blobs" / LIVE_SHA
    assert blob.is_file()
    assert hashlib.sha256(blob.read_bytes()).hexdigest() == LIVE_SHA
    assert paths["LICENSE.md"].read_bytes() == blob.read_bytes()
