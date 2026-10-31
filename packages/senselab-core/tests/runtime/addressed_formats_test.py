"""The address contract, generated over every registered addressed format: a misfiled artifact never reads."""

import hashlib
import io
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO

import pytest
from senselab_core.kernel.identity import ExecutionContext, LogicalKey, ModelIdentity
from senselab_core.runtime.fetch import DataManifest, ManifestError, fetch
from senselab_core.runtime.storage import ADDRESSED_FORMATS, AddressedFormat
from senselab_core.runtime.store import STORE_DIR, CorruptEntryError, ResultStore

LICENSE = {
    "expression": "CC-BY-4.0",
    "commercial_use": "yes",
    "copyleft": "none",
    "attribution": True,
    "gated": False,
    "source": "https://example.org/license",
    "verified": {"date": "2026-10-01", "how": "manual"},
}


def _key(audio: str) -> LogicalKey:
    return LogicalKey(
        task="asr",
        task_version="1",
        backend="hf",
        backend_version="1",
        model=ModelIdentity(source="hf", id="org/asr", identity="0" * 40),
        params={"beam": 1},
        inputs={"audio": "content:" + audio * 64},
    )


def _ctx(platform: str = "linux") -> ExecutionContext:
    return ExecutionContext(
        env_lock="lock",
        platform=platform,
        wheel_set="wheels",
        numerics={"device": "cpu", "precision": "fp32"},
        decoder=None,
        driver={},
        host="node1",
        protocol="p1",
    )


def _store_entries(root: Path) -> tuple[Path, Path]:
    s, c = ResultStore(root), _ctx()
    paths = []
    for audio in ("a", "b"):
        k = _key(audio)
        s.put(k, {"text": f"transcript of {audio}"}, c)
        d = k.digest()
        paths.append(root / STORE_DIR / d[:2] / d / c.fingerprint())
    return paths[0], paths[1]


def _store_contexts(root: Path) -> tuple[Path, Path]:
    s, k = ResultStore(root), _key("a")
    paths = []
    for platform in ("linux", "macos"):
        c = _ctx(platform)
        s.put(k, {"text": platform}, c)
        d = k.digest()
        paths.append(root / STORE_DIR / d[:2] / d / c.fingerprint())
    return paths[0], paths[1]


def _manifest(name: str, data: bytes) -> DataManifest:
    return DataManifest.model_validate(
        {
            "format": "data-manifest@2.0",
            "name": name,
            "version": "1",
            "license": LICENSE,
            "terms": "Cite the source.",
            "files": [
                {
                    "path": "f.bin",
                    "url": f"https://x/{name}",
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size": len(data),
                }
            ],
        }
    )


def _fetched(root: Path) -> tuple[DataManifest, DataManifest]:
    data = {"one": b"first dataset bytes", "two": b"second dataset bytes"}

    def opener(url: str) -> BinaryIO:
        return io.BytesIO(data[url.rsplit("/", 1)[1]])

    manifests = (_manifest("one", data["one"]), _manifest("two", data["two"]))
    for m in manifests:
        fetch(m, cache_root=root, opener=opener)
    return manifests


def _blobs(root: Path) -> tuple[Path, Path]:
    a, b = _fetched(root)
    return root / "blobs" / a.files[0].sha256, root / "blobs" / b.files[0].sha256


def _markers(root: Path) -> tuple[Path, Path]:
    _fetched(root)
    return root / "datasets" / "one@1" / ".complete", root / "datasets" / "two@1" / ".complete"


def _writable(path: Path) -> None:
    for p in [path, *path.rglob("*")] if path.is_dir() else [path]:
        p.chmod(0o770 if p.is_dir() else 0o660)


def _swap(a: Path, b: Path) -> None:
    """Put a copy of `b`'s bytes at `a`'s address, as a restore or rsync mix-up would."""
    _writable(a)
    if b.is_dir():
        shutil.rmtree(a)
        shutil.copytree(b, a)
    else:
        a.write_bytes(b.read_bytes())


HARNESSES: dict[str, list[Callable[[Path], tuple[Path, Path]]]] = {
    "store-entry": [_store_entries, _store_contexts],
    "fetch-blob": [_blobs],
    "fetch-dataset-marker": [_markers],
}
"""For each registered format name, makers of two valid artifacts at their own addresses."""

CASES: list[Any] = [
    pytest.param(fmt, make, id=f"{fmt.name}-{make.__name__.strip('_')}")
    for fmt in ADDRESSED_FORMATS
    for make in HARNESSES.get(fmt.name, [])
]


@pytest.mark.contract
def test_every_addressed_format_has_a_harness() -> None:
    """Each registered format is an `AddressedFormat` with a harness, so a new format cannot skip the contract."""
    assert {f.name for f in ADDRESSED_FORMATS} == {"store-entry", "fetch-blob", "fetch-dataset-marker"}
    for fmt in ADDRESSED_FORMATS:
        assert isinstance(fmt, AddressedFormat)
        assert HARNESSES.get(fmt.name), fmt.name


@pytest.mark.contract
@pytest.mark.parametrize(("fmt", "make"), CASES)
def test_artifact_at_its_address_reads(
    fmt: AddressedFormat, make: Callable[[Path], tuple[Path, Path]], tmp_path: Path
) -> None:
    """A valid artifact's re-derived address equals the one its location names, and it reads."""
    for path in make(tmp_path):
        assert fmt.address_of(path) == fmt.address_at(path)
        fmt.read(path)


@pytest.mark.contract
@pytest.mark.parametrize(("fmt", "make"), CASES)
def test_artifact_at_wrong_address_never_reads(
    fmt: AddressedFormat, make: Callable[[Path], tuple[Path, Path]], tmp_path: Path
) -> None:
    """Another artifact's bytes copied onto an address re-derive a different address and are refused by the reader."""
    a, b = make(tmp_path)
    _swap(a, b)
    assert fmt.address_of(a) != fmt.address_at(a)
    with pytest.raises((CorruptEntryError, ManifestError)):
        fmt.read(a)
