"""Packaging contracts of senselab-core."""

import importlib
import json
import subprocess
import sys
from importlib import metadata
from typing import Any

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


class _Dist:
    """A stand-in installed distribution: a version and, for an editable one, a PEP 610 ``direct_url.json``."""

    def __init__(self, version: str, kind: str) -> None:
        self.version = version
        table: dict[str, dict[str, Any] | None] = {
            "editable": {"dir_info": {"editable": True}},
            "wheel": {"dir_info": {}},
            "none": None,
        }
        self._direct = table[kind]

    def read_text(self, name: str) -> str | None:
        assert name == "direct_url.json"
        return None if self._direct is None else json.dumps({"url": "file:///x", **self._direct})


def _install(monkeypatch: pytest.MonkeyPatch, core: _Dist, worker: _Dist | None) -> None:
    def distribution(name: str) -> _Dist:
        found = {"senselab-core": core, "senselab-worker": worker}[name]
        if found is None:
            raise metadata.PackageNotFoundError(name)
        return found

    monkeypatch.setattr(metadata, "distribution", distribution)


KINDS = ("editable", "wheel", "none")


@pytest.mark.logic
@pytest.mark.parametrize("core_kind", KINDS)
@pytest.mark.parametrize("worker_kind", KINDS)
@pytest.mark.parametrize("match", [True, False])
def test_a32_core_checks_the_worker_version_only_between_non_editable_installs(
    core_kind: str, worker_kind: str, match: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Over every install kind of each distribution, equal or not, only a pair of non-editable installs is checked."""
    _install(monkeypatch, _Dist("2.0.0", core_kind), _Dist("2.0.0" if match else "2.0.1", worker_kind))
    refused = not match and "editable" not in (core_kind, worker_kind)
    if refused:
        with pytest.raises(ImportError) as raised:
            importlib.reload(senselab_core)
        assert "2.0.0" in str(raised.value) and "2.0.1" in str(raised.value)
    else:
        importlib.reload(senselab_core)


@pytest.mark.logic
@pytest.mark.parametrize("core_kind", KINDS)
def test_a32_core_refuses_a_missing_worker(core_kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A worker that is not installed raises ``ImportError`` whatever kind of install core is."""
    _install(monkeypatch, _Dist("2.0.0", core_kind), None)
    with pytest.raises(ImportError, match="not installed"):
        importlib.reload(senselab_core)


@pytest.mark.logic
@pytest.mark.parametrize("text", ["", "{", "[]", '{"dir_info": []}', '{"dir_info": {"editable": "yes"}}'])
def test_a32_malformed_direct_url_never_raises(text: str) -> None:
    """A malformed direct_url.json never raises; only a true ``dir_info.editable`` marks an install editable."""

    class Dist:
        def read_text(self, name: str) -> str:
            return text

    assert senselab_core._is_editable(Dist()) is (text == '{"dir_info": {"editable": "yes"}}')  # type: ignore[arg-type]
