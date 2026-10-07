"""The evaluation and events files carry no bare commit hashes outside the provenance note."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SPECS = ROOT / "specs" / "20261007-evaluation-and-events"

# Every path the change adds. Directories are scanned recursively; a listed file is scanned once it exists.
SCANNED = (
    ROOT / "src" / "senselab" / "utils" / "evaluation",
    ROOT / "src" / "senselab" / "utils" / "data_structures" / "events.py",
    ROOT / "src" / "senselab" / "utils" / "data_structures" / "events_io.py",
    ROOT / "src" / "tests" / "utils" / "evaluation",
    ROOT / "src" / "tests" / "utils" / "data_structures" / "events_test.py",
    ROOT / "src" / "tests" / "utils" / "data_structures" / "events_io_test.py",
    SPECS,
)
# Files that must exist; reading them raises FileNotFoundError when they do not.
REQUIRED = (
    SPECS / "design.md",
    SPECS / "provenance.md",
    ROOT / "src" / "senselab" / "utils" / "evaluation" / "__init__.py",
    ROOT / "src" / "senselab" / "utils" / "evaluation" / "doc.md",
)

HEX40 = re.compile(r"\b[0-9a-f]{40}\b", re.IGNORECASE)
# Paths relative to the repository root where a full commit hash is the point of the file.
EXEMPT_40HEX = {"specs/20261007-evaluation-and-events/provenance.md"}


def _files() -> list[Path]:
    found = set(REQUIRED)
    for path in SCANNED:
        if path.is_dir():
            found.update(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
        elif path.is_file():
            found.add(path)
    return sorted(found)


def _texts() -> dict[str, str]:
    return {p.relative_to(ROOT).as_posix(): p.read_text(errors="replace") for p in _files()}


def test_no_bare_commit_hashes() -> None:
    """A 40-hex commit appears only in the provenance note, where origins are cited by commit."""
    hits = [name for name, text in _texts().items() if name not in EXEMPT_40HEX and HEX40.search(text)]
    assert hits == []
