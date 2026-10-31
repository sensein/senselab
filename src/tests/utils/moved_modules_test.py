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

# Moved module basenames, for the `from senselab.utils import <name>` form.
MOVED_NAMES = {"file_lock", "model_revision"}

# Names that now live in senselab_core and must not be imported from `senselab.utils.dependencies`.
MOVED_HELPERS = {"retry_on_transient_error", "_is_transient", "_http_status", "get_cached_commit_hash"}

_IMPORT_FROM = re.compile(r"from\s+{module}\s+import\s+(\(?[^)\n]*(?:\n[^)\n]*)*?\)|[^\n]*)", re.M)


def _imported_names(text: str, module: str) -> set[str]:
    """Names brought in by every ``from <module> import ...`` statement in ``text``."""
    names: set[str] = set()
    for m in re.finditer(_IMPORT_FROM.pattern.format(module=re.escape(module)), text):
        names.update(n.strip().split(" as ")[0].strip("() \n") for n in m.group(1).replace("\n", ",").split(","))
    return {n for n in names if n}


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
        hits += [
            f"{path.relative_to(REPO)}: from senselab.utils import {n}"
            for n in sorted(MOVED_NAMES & _imported_names(text, "senselab.utils"))
        ]
        hits += [
            f"{path.relative_to(REPO)}: {n} imported from senselab.utils.dependencies"
            for n in sorted(MOVED_HELPERS & _imported_names(text, "senselab.utils.dependencies"))
        ]
    assert hits == []


_SHA_LITERAL = re.compile(r"\b\w*SHA256\s*=\s*[\"'][0-9a-f]{64}[\"']")


@pytest.mark.logic
def test_no_weight_sha256_literals_remain() -> None:
    """Weight digests live in `weights.yaml`; no `*_SHA256 = "<64 hex>"` literal is left outside `reproduce/`."""
    hits = [
        str(path.relative_to(REPO))
        for path in (REPO / "src" / "senselab").rglob("*.py")
        if "reproduce" not in path.relative_to(REPO / "src" / "senselab").parts
        and _SHA_LITERAL.search(path.read_text())
    ]
    assert hits == []
