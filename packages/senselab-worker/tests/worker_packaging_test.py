"""Packaging contracts of senselab-worker."""

import subprocess
import sys

import pytest


@pytest.mark.logic
def test_worker_imports_without_pydantic_or_core() -> None:
    """Importing senselab_worker loads neither pydantic nor senselab_core."""
    code = (
        "import senselab_worker, sys; "
        "print(sorted(m for m in sys.modules if m.split('.')[0] in {'pydantic', 'senselab_core'}))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
