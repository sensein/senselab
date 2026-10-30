"""Packaging contracts of senselab-core."""

import subprocess
import sys
from importlib import metadata

import pytest

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
