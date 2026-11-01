"""Packaging contracts of senselab-worker."""

import subprocess
import sys

import pytest

FORBIDDEN = {"pydantic", "senselab_core", "torch", "transformers", "speechbrain", "pyannote"}


@pytest.mark.logic
def test_worker_imports_without_pydantic_core_or_frameworks() -> None:
    """Importing senselab_worker loads no pydantic, senselab_core or model framework."""
    code = f"import senselab_worker, sys; print(sorted(m for m in sys.modules if m.split('.')[0] in {FORBIDDEN!r}))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
