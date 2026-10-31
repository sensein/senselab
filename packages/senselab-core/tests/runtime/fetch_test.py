"""Data manifests and the hash-verified, content-addressed fetch cache."""

import errno
import hashlib
import http.client
import io
import json
import multiprocessing as mp
import os
import stat
import tempfile
import time
import unicodedata
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO

import pytest
from hypothesis import event, given, settings
from hypothesis import strategies as st
from senselab_core.cli import main as cli_main
from senselab_core.runtime import fetch as fetch_module
from senselab_core.runtime.fetch import (
    DataFile,
    DataManifest,
    FetchError,
    ManifestConflictError,
    ManifestError,
    default_cache_root,
    fetch,
)
from senselab_core.runtime.storage import StorageFull, StorePolicy, StoreUnavailable
from senselab_worker.canonical import jcs_dumps
from senselab_worker.digest import identity_digest

STALE_SECONDS = StorePolicy.load_default().staging_stale_s

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


def _file(path: str, url: str, data: bytes = PAYLOAD, **extra: object) -> dict[str, Any]:
    return {"path": path, "url": url, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data), **extra}


def _manifest(files: list[dict[str, Any]] | None = None, version: str = "1", name: str = "demo") -> DataManifest:
    return DataManifest.model_validate(
        {
            "format": "data-manifest@2.0",
            "name": name,
            "version": version,
            "license": LICENSE,
            "terms": "Cite the source.",
            "files": files or [_file("a/voice.wav", "https://x/a.wav")],
        }
    )


def _dataset(root: Path, name: str = "demo", version: str = "1") -> Path:
    return root / "datasets" / f"{name}@{version}"


def _unseal(path: Path) -> Path:
    path.chmod(0o660)
    return path


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
    assert link.parent == _dataset(tmp_path) / "files" / "a"
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
    manifest = _manifest([_file("v.wav", "https://x/a", mirrors=["https://m/a"])])
    routes = {"https://x/a": b"tampered", "https://m/a": b"also wrong"}
    with pytest.raises(FetchError) as err:
        fetch(manifest, cache_root=tmp_path, opener=_opener(routes))
    message = str(err.value)
    assert "v.wav" in message and "https://x/a" in message and "https://m/a" in message
    assert not (tmp_path / "blobs" / SHA).exists()


@pytest.mark.logic
def test_mirror_with_matching_hash_accepted(tmp_path: Path) -> None:
    """Mirror with matching hash accepted."""
    manifest = _manifest([_file("v.wav", "https://x/a", mirrors=["https://m/a"])])
    routes = {"https://x/a": b"x" * len(PAYLOAD), "https://m/a": PAYLOAD}
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
    _unseal(tmp_path / "blobs" / SHA).write_bytes(b"bit rot")
    paths = fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))
    assert paths["a/voice.wav"].read_bytes() == PAYLOAD


@pytest.mark.logic
def test_transient_read_error_does_not_discard_blob(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Transient read error does not discard blob."""
    fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))

    def flaky(tag: str, path: object) -> str:
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(fetch_module, "blob_digest", flaky)
    with pytest.raises(StoreUnavailable):
        fetch(_manifest(), cache_root=tmp_path, opener=_opener({}), verify="full")
    assert (tmp_path / "blobs" / SHA).read_bytes() == PAYLOAD


@pytest.mark.logic
@pytest.mark.parametrize("umask", [0o077, 0o022, 0o000])
def test_permissions_independent_of_umask(tmp_path: Path, umask: int) -> None:
    """Directories are 0o2770 and blobs and the marker 0o440 whatever the umask; nothing grants others access."""
    old = os.umask(umask)
    try:
        paths = fetch(_manifest(), cache_root=tmp_path / "c", opener=_opener({"https://x/a.wav": PAYLOAD}))
    finally:
        os.umask(old)
    root = tmp_path / "c"
    for directory in (root, root / "blobs", root / "tmp", root / "datasets", paths["a/voice.wav"].parent):
        assert stat.S_IMODE(directory.stat().st_mode) & ~stat.S_ISGID == 0o770, directory
    assert stat.S_IMODE((root / "blobs" / SHA).stat().st_mode) == 0o440
    assert stat.S_IMODE((_dataset(root) / ".complete").stat().st_mode) == 0o440
    assert all(p.lstat().st_mode & 0o007 == 0 for p in root.rglob("*") if not p.is_symlink())


@pytest.mark.logic
def test_stale_download_removed_but_live_one_kept(tmp_path: Path) -> None:
    """Stale download removed but live one kept."""
    (tmp_path / "tmp").mkdir()
    stale = tmp_path / "tmp" / ("5" * 32)
    live = tmp_path / "tmp" / ("1" * 32)
    stale.write_bytes(b"x")
    live.write_bytes(b"y")
    old = time.time() - STALE_SECONDS - 60
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
format: data-manifest@2.0
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
  - {path: a.wav, url: "https://x/a.wav", sha256: "%s", size: 19000}
"""


@pytest.mark.logic
def test_manifest_loads(tmp_path: Path) -> None:
    """Manifest loads."""
    path = tmp_path / "m.yaml"
    path.write_text(MANIFEST_YAML % SHA)
    manifest = DataManifest.load(path)
    assert manifest.files == (DataFile(path="a.wav", url="https://x/a.wav", sha256=SHA, size=len(PAYLOAD)),)
    assert manifest.license.expression == "CC-BY-4.0"


@pytest.mark.parametrize(
    "text",
    [
        "- a list\n",
        "{unclosed: [",
        MANIFEST_YAML.replace("data-manifest@2.0", "data-manifest@9.9") % SHA,
        MANIFEST_YAML.replace("data-manifest@2.0", "data-manifest@1.0") % SHA,
        MANIFEST_YAML.replace(", size: 19000", "") % SHA,
        MANIFEST_YAML.replace("size: 19000", "size: -1") % SHA,
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
    entry = _file("a.wav", "https://x/a")
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
    assert str(_dataset(tmp_path / "cache") / "files" / "a.wav") in capsys.readouterr().out


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
    """Every directory created under the cache, at every level, is 0o2770."""
    manifest = _manifest([_file("sub-01/ses-01/a.wav", "https://x/a.wav")])
    old = os.umask(0o077)
    try:
        paths = fetch(manifest, cache_root=tmp_path / "deep" / "cache", opener=_opener({"https://x/a.wav": PAYLOAD}))
    finally:
        os.umask(old)
    leaf = paths["sub-01/ses-01/a.wav"].parent
    directory = leaf
    while directory != tmp_path / "deep":
        assert stat.S_IMODE(directory.stat().st_mode) & ~stat.S_ISGID == 0o770, directory
        directory = directory.parent
    marker = _dataset(tmp_path / "deep" / "cache") / ".complete"
    assert stat.S_IMODE(marker.stat().st_mode) == 0o440


@pytest.mark.logic
def test_size_mode_skips_rehash_and_full_mode_catches_same_size_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache hit checks size only by default; verify='full' re-hashes."""
    fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))
    hashed: list[object] = []
    real = fetch_module.blob_digest

    def spy(tag: str, path: Path) -> str:
        hashed.append(path)
        return real(tag, path)

    monkeypatch.setattr(fetch_module, "blob_digest", spy)
    fetch(_manifest(), cache_root=tmp_path, opener=_opener({}))
    assert hashed == []
    _unseal(tmp_path / "blobs" / SHA).write_bytes(b"x" * len(PAYLOAD))
    paths = fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}), verify="full")
    assert paths["a/voice.wav"].read_bytes() == PAYLOAD
    assert hashed


@pytest.mark.contract
@pytest.mark.parametrize("existing", ["good", "other_bytes", "truncated", "empty", "symlink_to_good", "dangling"])
def test_link_race_keeps_an_existing_blob_only_if_it_verifies(tmp_path: Path, existing: str) -> None:
    """A blob that appeared before the link is kept only if it verifies; else the verified staged copy replaces it."""
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    final = blobs / SHA
    good = tmp_path / "good"
    good.write_bytes(PAYLOAD)
    content = {"good": PAYLOAD, "other_bytes": b"y" * len(PAYLOAD), "truncated": PAYLOAD[:-1], "empty": b""}
    if existing in content:
        final.write_bytes(content[existing])
    else:
        final.symlink_to(good if existing == "symlink_to_good" else tmp_path / "nowhere")
    before = os.lstat(final).st_ino
    staged = tmp_path / ("5" * 32)
    staged.write_bytes(PAYLOAD)
    fetch_module._commit(
        staged,
        final,
        replace=False,
        keeps=lambda p: fetch_module._blob_state(p, SHA, {}, "full") == "verified",
    )
    assert not os.path.islink(final) and final.read_bytes() == PAYLOAD and not staged.exists()
    assert (os.lstat(final).st_ino == before) is (existing == "good")


@pytest.mark.logic
def test_completion_marker_records_manifest_hash(tmp_path: Path) -> None:
    """The dataset directory carries a .complete marker naming the manifest hash."""
    manifest = _manifest()
    fetch(manifest, cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))
    marker = json.loads((_dataset(tmp_path) / ".complete").read_text())
    assert marker["manifest_sha256"] == manifest.digest()
    assert (marker["format"], marker["name"], marker["version"]) == ("dataset-marker@2.0", "demo", "1")


@pytest.mark.logic
def test_dataset_without_marker_is_incomplete_and_completed_by_fetch(tmp_path: Path) -> None:
    """A dataset whose fetch died before the marker is refetched and then marked complete."""
    calls: list[str] = []
    fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))
    (_dataset(tmp_path) / ".complete").unlink()
    fetch(_manifest(), cache_root=tmp_path, opener=_opener({}, calls))
    assert calls == []  # blob already verified by hash, so no network
    assert (_dataset(tmp_path) / ".complete").exists()


@pytest.mark.logic
def test_same_name_version_different_manifest_raises(tmp_path: Path) -> None:
    """A different manifest under a completed name@version is refused, links untouched."""
    first = fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))
    other = b"other bytes"
    changed = _manifest([_file("a/voice.wav", "https://x/b", other)])
    with pytest.raises(ManifestConflictError):
        fetch(changed, cache_root=tmp_path, opener=_opener({"https://x/b": other}))
    assert first["a/voice.wav"].read_bytes() == PAYLOAD


@pytest.mark.logic
def test_disk_full_is_storage_error_not_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ENOSPC while staging is reported as StorageFull."""

    def full(tag: str, path: object) -> str:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(fetch_module, "blob_digest", full)
    with pytest.raises(StorageFull) as err:
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
    old = time.time() - STALE_SECONDS - 60
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
    _unseal(blob).write_bytes(b"bit rot")
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
    _unseal(blob).write_bytes(b"bit rot")
    with pytest.raises(FetchError):
        fetch(_manifest(), cache_root=tmp_path, opener=_opener({}))
    assert blob.read_bytes() == b"bit rot"


@pytest.mark.logic
def test_corrupt_marker_is_replaced_once_dataset_verifies(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A corrupt marker is rewritten, so later fetches stop warning."""
    fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))
    marker = _unseal(_dataset(tmp_path) / ".complete")
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
        '{"manifest_sha256": 1, "stamps": {}}',
        '{MARKER, "stamps": []}',
        '{MARKER, "stamps": {"a": {"size": "big", "mtime_ns": "1", "inode": "1", "ctime_ns": "1"}}}',
        '{MARKER, "stamps": {"a": {"size": true, "mtime_ns": "1", "inode": "1", "ctime_ns": "1"}}}',
        '{MARKER, "stamps": {"a": {"size": -1, "mtime_ns": "1", "inode": "1", "ctime_ns": "1"}}}',
        '{MARKER, "stamps": {"a": {"size": 1, "mtime_ns": 1, "inode": "1", "ctime_ns": "1"}}}',
        '{MARKER, "stamps": {"a": {"size": 1, "mtime_ns": "1", "inode": "1"}}}',
        '{MARKER, "format": "dataset-marker@1.0", "stamps": {}}',
        "\xff",
    ],
)
def test_malformed_marker_is_treated_as_absent(tmp_path: Path, text: str) -> None:
    """Any malformed marker naming this manifest, or none, is logged and rewritten, never raised."""
    fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))
    marker = _unseal(_dataset(tmp_path) / ".complete")
    fields = f'"manifest_sha256": "{_manifest().digest()}", "name": "demo", "version": "1"'
    marker.write_bytes(text.replace("{MARKER", "{" + fields).encode("latin-1"))
    paths = fetch(_manifest(), cache_root=tmp_path, opener=_opener({}))
    assert paths["a/voice.wav"].read_bytes() == PAYLOAD
    assert json.loads(marker.read_text())["manifest_sha256"] == _manifest().digest()


@pytest.mark.logic
def test_no_hardlink_support_is_storage_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A filesystem without hard links gets a StorageError naming the fix."""

    def refuse(src: object, dst: object) -> None:
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(os, "link", refuse)
    with pytest.raises(StoreUnavailable) as err:
        fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))
    assert "hard links" in str(err.value)


LIVE_URL = "https://raw.githubusercontent.com/sensein/senselab/5e2f8755de12c256f80437e7f32216b47afa91f8/LICENSE.md"
LIVE_SHA = "75009e0aaa9374223af0c954d175a91d39407acf13bc8e2ae2c882f6a1904f8d"
LIVE_SIZE = 16423


@pytest.mark.env
@pytest.mark.network
def test_live_fetch(tmp_path: Path) -> None:
    """A real download of an immutable public file lands in blobs/ and verifies."""
    manifest = _manifest([{"path": "LICENSE.md", "url": LIVE_URL, "sha256": LIVE_SHA, "size": LIVE_SIZE}])
    paths = fetch(manifest, cache_root=tmp_path)
    blob = tmp_path / "blobs" / LIVE_SHA
    assert blob.is_file()
    assert hashlib.sha256(blob.read_bytes()).hexdigest() == LIVE_SHA
    assert paths["LICENSE.md"].read_bytes() == blob.read_bytes()


@pytest.mark.logic
def test_manifest_digest_is_tagged() -> None:
    """The manifest digest is the registered `data-manifest@3` tagged digest of its JSON form."""
    manifest = _manifest()
    assert manifest.digest() == identity_digest("data-manifest@3", manifest.model_dump(mode="json"))
    assert manifest.digest() != hashlib.sha256(jcs_dumps(manifest.model_dump(mode="json"))).hexdigest()


@pytest.mark.logic
def test_a18_manifest_file_named_like_the_marker_stays_payload(tmp_path: Path) -> None:
    """The probe: a manifest file `.complete` was overwritten by the marker; it now lives under files/."""
    other = PAYLOAD + b"2"
    manifest = _manifest([_file(".complete", "https://x/c"), _file("x.wav", "https://x/x", other)])
    routes = {"https://x/c": PAYLOAD, "https://x/x": other}
    for _ in range(2):
        paths = fetch(manifest, cache_root=tmp_path, opener=_opener(routes))
        assert paths[".complete"].read_bytes() == PAYLOAD and paths["x.wav"].read_bytes() == other
    assert json.loads((_dataset(tmp_path) / ".complete").read_bytes())["manifest_sha256"] == manifest.digest()


@pytest.mark.logic
@pytest.mark.parametrize(
    "paths",
    [
        ["."],
        ["a/./b"],
        ["a//b"],
        ["/a"],
        ["a/"],
        ["a/../b"],
        ["a\\b"],
        ["a/b", "a"],
        ["a", "a/b/c"],
        ["A", "a"],
        ["e\u0301", "\u00e9"],
        ["a/.b." + "0" * 32],
    ],
    ids=lambda p: repr(p),
)
def test_a18_paths_that_are_not_prefix_free_are_refused(paths: list[str]) -> None:
    """The probe: `.` and `a/b` with `a` validated; every non-normal or nested path set is refused."""
    with pytest.raises(ValueError):
        _manifest([_file(p, f"https://x/{i}") for i, p in enumerate(paths)])


_PLAIN_PARTS = ["a", "A", "b", "c", "\u00e9", "e\u0301", ".complete", "files", "tmp", "blobs"]
_PLAIN_PARTS += [
    "\u03c3",
    "\u03c2",
    "\u03a3",
    "\u00b5",
    "\u03bc",
    "\u017f",
    "s",
    "\u00df",
    "ss",
    "\u1e9e",
    "\u212a",
    "k",
    "\u0131",
    "I",
    "i",
]


def _oracle_fold(part: str) -> str:
    """Canonical caseless match (NFD, casefold, NFD), plus the one join simple upper case adds: dotless ı is i."""
    return unicodedata.normalize("NFD", unicodedata.normalize("NFD", part).casefold()).replace("\u0131", "i")


_BAD_PARTS = [".", "..", "", ".a." + "0" * 32, "x\\y"]


@st.composite
def manifest_path_sets(draw: st.DrawFn) -> list[str]:
    """Sets of one to five paths of one to three parts drawn from names that collide, nest or escape."""
    part = st.sampled_from(_PLAIN_PARTS * 4 + _BAD_PARTS)
    path = st.lists(part, min_size=1, max_size=3).map("/".join)
    return draw(st.lists(path, min_size=1, max_size=5))


def _acceptable(paths: list[str]) -> bool:
    """The oracle: every part is a plain name, and no path equals or contains another under canonical caseless match."""
    keys = []
    for p in paths:
        parts = p.split("/")
        if "\\" in p or any(x in ("", ".", "..") or (x.startswith(".a.") and x.endswith("0" * 32)) for x in parts):
            return False
        keys.append(tuple(_oracle_fold(x) for x in parts))
    return all(a != b and a[: len(b)] != b for a in keys for b in keys if a is not b)


_FOLD_CLASSES = [
    ["\u03c3", "\u03c2", "\u03a3"],
    ["\u00b5", "\u03bc", "\u039c"],
    ["\u017f", "s", "S"],
    ["\u00df", "ss", "SS", "\u1e9e", "s\u017f"],
    ["\u212a", "k", "K"],
    ["\u00e9", "e\u0301", "\u00c9", "E\u0301"],
    ["\u212b", "\u00c5", "\u00e5", "a\u030a"],
    ["\u1e69", "s\u0323\u0307", "s\u0307\u0323", "\u1e68"],
    ["\u0131", "i", "I"],
]


@pytest.mark.contract
@settings(max_examples=300, deadline=None)
@given(
    picks=st.lists(st.tuples(st.integers(0, len(_FOLD_CLASSES) - 1), st.integers(0, 9), st.integers(0, 9)), max_size=6)
)
def test_path_key_joins_every_spelling_of_one_caseless_name(picks: list[tuple[int, int, int]]) -> None:
    """Names built from spellings one filesystem may store as one file share a key; other names never do."""
    left = "".join(_FOLD_CLASSES[c][i % len(_FOLD_CLASSES[c])] for c, i, _ in picks)
    right = "".join(_FOLD_CLASSES[c][j % len(_FOLD_CLASSES[c])] for c, _, j in picks)
    assert fetch_module.path_key(f"d/{left}") == fetch_module.path_key(f"D/{right}")
    other = "".join(_FOLD_CLASSES[(c + 1) % len(_FOLD_CLASSES)][0] for c, _, _ in picks)
    assert (fetch_module.path_key(left) == fetch_module.path_key(other)) is (not picks)


@pytest.mark.logic
@settings(max_examples=150, deadline=None)
@given(paths=manifest_path_sets())
def test_no_manifest_path_addresses_store_owned_name(paths: list[str]) -> None:
    """Accepted path sets are exactly the plain prefix-free ones; each fetched path lands in files/ with its bytes."""
    files = [_file(p, f"https://x/{i}", f"{i}:{p}".encode()) for i, p in enumerate(paths)]
    try:
        manifest = _manifest(files)
    except ValueError:
        event("refused")
        assert not _acceptable(paths)
        return
    event("accepted")
    assert _acceptable(paths)
    routes = {f["url"]: f"{i}:{f['path']}".encode() for i, f in enumerate(files)}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        result = fetch(manifest, cache_root=root, opener=_opener(routes))
        files_dir = _dataset(root) / "files"
        for i, f in enumerate(files):
            link = result[f["path"]]
            assert os.path.commonpath([os.path.normpath(link), files_dir]) == str(files_dir)
            assert link.read_bytes() == routes[f["url"]], f["path"]
        marker = json.loads((_dataset(root) / ".complete").read_bytes())
        assert marker["manifest_sha256"] == manifest.digest()
        assert sorted(p.name for p in _dataset(root).iterdir()) == [".complete", "files"]


@pytest.mark.logic
@pytest.mark.parametrize("kind", ["dangling", "to_good_bytes", "directory"])
def test_symlinked_or_dangling_blob_is_absent(tmp_path: Path, kind: str) -> None:
    """A blob path that is not a regular file is absent: never trusted, and replaced by a verified regular file."""
    blob = tmp_path / "blobs" / SHA
    blob.parent.mkdir(parents=True)
    if kind == "dangling":
        blob.symlink_to(tmp_path / "elsewhere")
    elif kind == "to_good_bytes":
        (tmp_path / "outside").write_bytes(PAYLOAD)
        blob.symlink_to(tmp_path / "outside")
    else:
        blob.mkdir()
    assert fetch_module._blob_state(blob, SHA, {}, "full") == "absent"
    calls: list[str] = []
    if kind == "directory":
        with pytest.raises(StoreUnavailable):
            fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}, calls))
        return
    paths = fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}, calls))
    assert calls == ["https://x/a.wav"]
    assert stat.S_ISREG(blob.lstat().st_mode) and stat.S_IMODE(blob.lstat().st_mode) == 0o440
    assert paths["a/voice.wav"].read_bytes() == PAYLOAD


@pytest.mark.logic
@pytest.mark.skipif(os.geteuid() == 0, reason="root writes a file whatever its mode")
def test_a19_blob_cannot_be_written_through_the_returned_path(tmp_path: Path) -> None:
    """The probe: a same-size write through the dataset path poisoned the shared blob; the blob is now read-only."""
    paths = fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))
    with pytest.raises(PermissionError):
        open(paths["a/voice.wav"], "r+b")


@pytest.mark.logic
@pytest.mark.parametrize("change", ["rewrite", "replace"])
def test_same_size_rewrite_is_not_trusted(tmp_path: Path, change: str) -> None:
    """Under verify='size' a blob whose mtime or inode moved since completion is re-hashed and refetched."""
    fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))
    blob = tmp_path / "blobs" / SHA
    poison = b"POISONED!!!!" + PAYLOAD[12:]
    if change == "rewrite":
        _unseal(blob).write_bytes(poison)
    else:
        (tmp_path / "p").write_bytes(poison)
        os.replace(tmp_path / "p", blob)
        os.utime(blob, ns=(0, 0))
    calls: list[str] = []
    paths = fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}, calls))
    assert calls == ["https://x/a.wav"]
    assert paths["a/voice.wav"].read_bytes() == PAYLOAD


class _Endless(io.RawIOBase):
    def __init__(self) -> None:
        self.served = 0

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        n = size if size >= 0 else _ENDLESS_CHUNK
        self.served += n
        return b"\0" * n


_ENDLESS_CHUNK = 1 << 20


@pytest.mark.logic
@pytest.mark.parametrize("size", [0, 1, 100, 3 * _ENDLESS_CHUNK + 5])
def test_download_aborts_past_declared_size(tmp_path: Path, size: int) -> None:
    """The probe: an endless source streamed 3.86 GB in 3 s; it now stops after size + 1 bytes."""
    source = _Endless()
    staged = tmp_path / ("5" * 32)
    with pytest.raises(FetchError, match="more than the declared"):
        fetch_module._download("https://x/a", lambda url: source, staged, size=size)  # type: ignore[arg-type,return-value]
    assert source.served == size + 1
    assert staged.stat().st_size <= size


@pytest.mark.logic
def test_download_short_of_declared_size_is_refused(tmp_path: Path) -> None:
    """A source that ends before the declared size fails that source; the next mirror is tried."""
    manifest = _manifest([_file("v.wav", "https://x/a", mirrors=["https://m/a"])])
    routes = {"https://x/a": PAYLOAD[:-1], "https://m/a": PAYLOAD}
    assert fetch(manifest, cache_root=tmp_path, opener=_opener(routes))["v.wav"].read_bytes() == PAYLOAD
    with pytest.raises(FetchError, match="ended after"):
        fetch_module._download("https://x/a", _opener(routes), tmp_path / ("6" * 32), size=len(PAYLOAD))


@pytest.mark.logic
@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://x/y", "http://x/y", "data:,hello", "x/y"])
def test_non_https_refused(tmp_path: Path, url: str) -> None:
    """A non-https URL raises FetchError before any opener is called, through fetch or the default opener."""
    with pytest.raises(FetchError):
        fetch_module._urlopen(url)
    calls: list[str] = []
    with pytest.raises(FetchError) as err:
        fetch(_manifest([_file("a", url)]), cache_root=tmp_path, opener=_opener({url: PAYLOAD}, calls))
    assert calls == [] and "https" in str(err.value)


@pytest.mark.logic
def test_a20_file_url_is_not_read(tmp_path: Path) -> None:
    """The probe: a file:// manifest URL was read by the default opener; nothing is opened now."""
    secret = tmp_path / "secret.txt"
    secret.write_text("participant notes")
    manifest = _manifest([_file("a", f"file://{secret}")])
    with pytest.raises(FetchError, match="refusing file URL"):
        fetch(manifest, cache_root=tmp_path / "c")


@pytest.mark.logic
@pytest.mark.parametrize("target", ["http://x/y", "file:///etc/passwd", "ftp://x/y"])
def test_redirect_to_non_https_refused(target: str) -> None:
    """The default opener follows a redirect only to https, and carries no file, ftp, data or http handler."""
    opener = fetch_module._https_opener()
    handlers = opener.handlers  # type: ignore[attr-defined]
    kinds = {type(h) for h in handlers}
    for banned in (
        urllib.request.FileHandler,
        urllib.request.FTPHandler,
        urllib.request.DataHandler,
        urllib.request.HTTPHandler,
        urllib.request.UnknownHandler,
    ):
        assert banned not in kinds
    (redirect,) = [h for h in handlers if isinstance(h, urllib.request.HTTPRedirectHandler)]
    req = urllib.request.Request("https://x/start")
    with pytest.raises(FetchError):
        redirect.redirect_request(req, io.BytesIO(), 302, "Found", http.client.HTTPMessage(), target)
    assert (
        redirect.redirect_request(req, io.BytesIO(), 302, "Found", http.client.HTTPMessage(), "https://y/n") is not None
    )


@pytest.mark.logic
@pytest.mark.parametrize("err", [errno.ENOSPC, errno.EDQUOT])
def test_enospc_is_storage_full(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, err: int) -> None:
    """Out of space while writing a download is StorageFull, with no staged file left behind."""

    def full(url: str, opener: object, staged: Path, *, size: int, mode: int) -> None:
        staged.write_bytes(b"partial")
        raise OSError(err, os.strerror(err))

    monkeypatch.setattr(fetch_module, "_download", full)
    with pytest.raises(StorageFull) as caught:
        fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))
    assert caught.value.errno == err
    assert list((tmp_path / "tmp").iterdir()) == []


@pytest.mark.logic
def test_marker_that_is_a_directory_is_a_conflict(tmp_path: Path) -> None:
    """A `.complete` that is a directory is reported as a conflict, never a raw OSError from os.replace."""
    fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))
    marker = _dataset(tmp_path) / ".complete"
    marker.unlink()
    marker.mkdir()
    with pytest.raises(ManifestConflictError, match="not a regular file"):
        fetch(_manifest(), cache_root=tmp_path, opener=_opener({}))


@pytest.mark.logic
def test_misfiled_marker_is_a_conflict(tmp_path: Path) -> None:
    """Another dataset's marker copied into this dataset is refused, not rewritten."""
    routes = {"https://x/a.wav": PAYLOAD}
    fetch(_manifest(), cache_root=tmp_path, opener=_opener(routes))
    fetch(_manifest(name="other"), cache_root=tmp_path, opener=_opener(routes))
    target = _unseal(_dataset(tmp_path) / ".complete")
    target.write_bytes((_dataset(tmp_path, "other") / ".complete").read_bytes())
    with pytest.raises(ManifestConflictError):
        fetch(_manifest(), cache_root=tmp_path, opener=_opener(routes))


@pytest.mark.logic
def test_malformed_marker_naming_another_manifest_is_a_conflict(tmp_path: Path) -> None:
    """A malformed marker is rewritten only if the manifest digest it still names is this manifest's."""
    fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))
    marker = _unseal(_dataset(tmp_path) / ".complete")
    marker.write_text(json.dumps({"manifest_sha256": "f" * 64, "stamps": "garbled"}))
    with pytest.raises(ManifestConflictError):
        fetch(_manifest(), cache_root=tmp_path, opener=_opener({}))
    assert json.loads(marker.read_text())["manifest_sha256"] == "f" * 64


@pytest.mark.logic
def test_unmarked_dataset_with_foreign_files_is_a_conflict(tmp_path: Path) -> None:
    """Without a valid marker, a dataset holding files outside the manifest is refused rather than marked complete."""
    fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))
    (_dataset(tmp_path) / ".complete").unlink()
    (_dataset(tmp_path) / "files" / "stale.wav").write_bytes(b"x")
    with pytest.raises(ManifestConflictError, match="stale.wav"):
        fetch(_manifest(), cache_root=tmp_path, opener=_opener({}))
    assert not (_dataset(tmp_path) / ".complete").exists()


@pytest.mark.logic
def test_stale_download_age_uses_the_storage_clock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A local clock running ahead does not make a live download look stale."""
    live = tmp_path / "tmp" / ("1" * 32)
    live.parent.mkdir(parents=True)
    live.write_bytes(b"in flight")
    real = time.time
    monkeypatch.setattr(time, "time", lambda: real() + 2 * STALE_SECONDS)
    fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))
    assert live.exists()


@pytest.mark.logic
def test_a19_dangling_blob_symlink_is_not_trusted(tmp_path: Path) -> None:
    """The probe: a dangling `blobs/<sha>` symlink was trusted as present; it is now replaced by a verified blob."""
    (tmp_path / "blobs").mkdir()
    os.symlink("/nonexistent/elsewhere", tmp_path / "blobs" / SHA)
    for _ in range(2):
        paths = fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))
        assert paths["a/voice.wav"].read_bytes() == PAYLOAD
    assert not (tmp_path / "blobs" / SHA).is_symlink()


@pytest.mark.logic
def test_a19_poisoned_blob_is_not_served_under_size_verification(tmp_path: Path) -> None:
    """The probe: a same-size overwrite through the returned path was served by `verify="size"` with no network call."""
    paths = fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))
    _unseal(tmp_path / "blobs" / SHA)
    with open(paths["a/voice.wav"], "r+b") as f:
        f.write(b"POISONED!!!!!!!!!!")
    calls: list[str] = []
    paths = fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}, calls))
    assert paths["a/voice.wav"].read_bytes() == PAYLOAD and calls == ["https://x/a.wav"]


@pytest.mark.logic
def test_a20_endless_source_is_cut_at_declared_size(tmp_path: Path) -> None:
    """The probe: an endless source wrote 3.86 GB to tmp/ before any check; fetch now stops at size + 1 bytes."""
    source = _Endless()
    with pytest.raises(FetchError, match="more than the declared"):
        fetch(_manifest(), cache_root=tmp_path, opener=lambda url: source)  # type: ignore[arg-type,return-value]
    assert source.served == len(PAYLOAD) + 1
    assert list((tmp_path / "tmp").iterdir()) == []


@pytest.mark.logic
def test_repaired_blob_is_trusted_again_by_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """After a blob is repaired its new stamp is recorded, so the next size-verified fetch reads nothing."""
    fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))
    _unseal(tmp_path / "blobs" / SHA).write_bytes(b"x" * len(PAYLOAD))
    fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))
    hashed: list[object] = []
    real = fetch_module.blob_digest

    def spy(tag: str, path: Path) -> str:
        hashed.append(path)
        return real(tag, path)

    monkeypatch.setattr(fetch_module, "blob_digest", spy)
    fetch(_manifest(), cache_root=tmp_path, opener=_opener({}))
    assert hashed == []


@pytest.mark.logic
@pytest.mark.skipif(os.geteuid() == 0, reason="root writes whatever the mode")
def test_complete_read_only_cache_still_serves(tmp_path: Path) -> None:
    """A completed cache whose directories are no longer writable still serves its dataset offline."""
    paths = fetch(_manifest(), cache_root=tmp_path / "c", opener=_opener({"https://x/a.wav": PAYLOAD}))
    dirs = [p for p in [tmp_path / "c", *(tmp_path / "c").rglob("*")] if p.is_dir() and not p.is_symlink()]
    for d in dirs:
        d.chmod(0o555)
    try:
        again = fetch(_manifest(), cache_root=tmp_path / "c", opener=_opener({}))
    finally:
        for d in dirs:
            d.chmod(0o770)
    assert again == paths and again["a/voice.wav"].read_bytes() == PAYLOAD


@pytest.mark.logic
def test_mtime_restored_rewrite_is_caught_by_ctime(tmp_path: Path) -> None:
    """A same-size rewrite whose mtime is put back with utime still changes ctime, so it is re-hashed and refetched."""
    fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))
    blob = tmp_path / "blobs" / SHA
    before = blob.stat()
    _unseal(blob).write_bytes(b"POISONED!!!!" + PAYLOAD[12:])
    blob.chmod(0o440)
    os.utime(blob, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert blob.stat().st_mtime_ns == before.st_mtime_ns and blob.stat().st_ino == before.st_ino
    calls: list[str] = []
    paths = fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}, calls))
    assert calls == ["https://x/a.wav"] and paths["a/voice.wav"].read_bytes() == PAYLOAD


@pytest.mark.logic
def test_directory_at_a_link_path_is_a_conflict(tmp_path: Path) -> None:
    """A directory where a manifest file's link belongs is a typed conflict, not a retryable storage fault."""
    fetch(_manifest(), cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))
    link = _dataset(tmp_path) / "files" / "a" / "voice.wav"
    link.unlink()
    link.mkdir()
    with pytest.raises(ManifestConflictError, match="is a directory"):
        fetch(_manifest(), cache_root=tmp_path, opener=_opener({}))


@pytest.mark.logic
def test_path_key_keeps_a_part_with_post_13_code_points_as_given() -> None:
    """A part holding a code point unassigned in Unicode 13.0 is its own key, on every interpreter; others fold."""
    raw = "é\U0001e08f̣.wav"
    assert fetch_module.path_key(f"Data/{raw}") == ("data", raw)
