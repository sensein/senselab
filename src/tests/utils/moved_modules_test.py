"""Guard: modules moved into ``senselab_core`` leave no reference to their old paths."""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCAN_ROOTS = ("src", "scripts")
SELF = Path(__file__).resolve()

MOVED = {
    "senselab.utils.file_lock",
    "senselab.utils.model_revision",
    "senselab/utils/file_lock",
    "senselab/utils/model_revision",
}

# Extended by later tasks: patterns (not paths) that must not appear in any scanned file.
FORBIDDEN_PATTERNS: tuple[re.Pattern[str], ...] = ()


def _scanned_files() -> list[Path]:
    """Every text source under the scan roots, excluding this file."""
    files: list[Path] = []
    for root in SCAN_ROOTS:
        for suffix in ("*.py", "*.md"):
            files.extend(p for p in (REPO / root).rglob(suffix) if p.resolve() != SELF)
    return files


@pytest.mark.logic
def test_no_moved_module_paths_remain() -> None:
    """No scanned file mentions an old module path, worker-script strings included."""
    hits = []
    for path in _scanned_files():
        text = path.read_text(errors="replace")
        hits += [f"{path.relative_to(REPO)}: {m}" for m in sorted(MOVED) if m in text]
        hits += [f"{path.relative_to(REPO)}: {p.pattern}" for p in FORBIDDEN_PATTERNS if p.search(text)]
    assert hits == []
