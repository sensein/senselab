"""Packaging contracts of senselab-core."""

import importlib
import subprocess
import sys
from importlib import metadata

import pytest
import senselab_core

FORBIDDEN = {"torch", "transformers", "speechbrain", "pyannote", "senselab"}


@pytest.mark.logic
def test_core_imports_without_heavy_frameworks() -> None:
    """Importing senselab_core loads no model framework and no senselab."""
    code = f"import senselab_core, sys; print(sorted(m for m in sys.modules if m.split('.')[0] in {FORBIDDEN!r}))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


@pytest.mark.logic
def test_versions_are_pinned_equal() -> None:
    """Core and worker share one version and core pins the worker exactly."""
    assert metadata.version("senselab-core") == metadata.version("senselab-worker")
    assert f"senselab-worker=={metadata.version('senselab-worker')}" in (metadata.requires("senselab-core") or [])


@pytest.mark.logic
def test_version_derives_from_a_release_tag() -> None:
    """A non-release tag must not set the version."""
    assert not metadata.version("senselab-core").startswith("31.")


@pytest.mark.logic
@pytest.mark.parametrize(
    ("core", "worker"), [("2.0.0", "2.0.1"), ("2.0.0", "1.9.0"), ("2.0.0", "2.0.0+local"), ("2.0.0", "2.0.0.dev1")]
)
def test_a32_core_refuses_mismatched_worker(core: str, worker: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing core raises ``ImportError`` naming both versions when the worker differs from core's own."""
    versions = {"senselab-core": core, "senselab-worker": worker}
    monkeypatch.setattr(metadata, "version", versions.__getitem__)
    with pytest.raises(ImportError) as raised:
        importlib.reload(senselab_core)
    assert core in str(raised.value) and worker in str(raised.value)


@pytest.mark.logic
def test_a32_core_refuses_a_missing_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """A worker that is not installed raises ``ImportError`` too, naming core's version."""

    def version(name: str) -> str:
        if name == "senselab-worker":
            raise metadata.PackageNotFoundError(name)
        return "2.0.0"

    monkeypatch.setattr(metadata, "version", version)
    with pytest.raises(ImportError, match="2.0.0"):
        importlib.reload(senselab_core)


@pytest.mark.logic
def test_a32_core_imports_when_the_versions_match(monkeypatch: pytest.MonkeyPatch) -> None:
    """Equal versions import cleanly."""
    monkeypatch.setattr(metadata, "version", lambda name: "2.0.0")
    importlib.reload(senselab_core)
