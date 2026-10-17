"""The reproduce package stays generic: no site paths, partitions or project names, no backend venv calls."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
SCANNED = (
    ROOT / "src" / "senselab" / "utils" / "reproduce",
    ROOT / "src" / "tests" / "utils" / "reproduce",
    ROOT / "specs" / "20261007-reproduce-driver",
)
# The forbidden patterns live outside the repository: SENSELAB_HYGIENE_DENY_FILE names one or more
# files (os.pathsep-separated) with one regular expression per line; without it the scan is skipped.
DENY_FILE_ENV = "SENSELAB_HYGIENE_DENY_FILE"


def _forbidden() -> "list[re.Pattern[str]]":
    value = os.environ.get(DENY_FILE_ENV, "")
    if not value:
        pytest.skip(f"{DENY_FILE_ENV} is not set")
    lines = [
        line
        for name in value.split(os.pathsep)
        if name
        for line in Path(name).read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    return [re.compile(line) for line in lines]


BACKEND_CALL = "ensure" + "_venv("


def _files() -> list[Path]:
    return [p for d in SCANNED if d.exists() for p in d.rglob("*") if p.is_file() and "__pycache__" not in p.parts]


def test_no_site_or_project_names() -> None:
    """No file names a cluster path, a partition or the project the package was built for."""
    forbidden = _forbidden()
    hits = [
        str(p.relative_to(ROOT)) for p in _files() if any(f.search(p.read_text(errors="replace")) for f in forbidden)
    ]
    assert hits == []


def test_package_never_calls_the_backend_venv_builder() -> None:
    """The package installs its own locks; it never goes through ``ensure_venv``."""
    package = SCANNED[0]
    hits = [str(p.relative_to(ROOT)) for p in package.rglob("*.py") if BACKEND_CALL in p.read_text()]
    assert hits == []
