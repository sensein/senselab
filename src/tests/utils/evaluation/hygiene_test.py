"""The evaluation and events files carry no bare commit hashes outside the provenance note."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

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


# The reviewer's evasions of a line regex, plus a comment that must not count.
_EVASIONS = {
    "multiline": "from senselab.utils.evaluation.splits import (\n    Split,\n    _cells,\n)\n",
    "relative": "from .splits import _concat\n",
    "relative-alias": "from .bootstrap import _defined as defined\n",
    "module-alias-attribute": "import senselab.utils.evaluation.splits as sp\nx = sp._sorted\n",
    "package-import-attribute": "from . import splits\nx = splits._majority\n",
}
_ALLOWED = {
    "comment": "# from senselab.utils.evaluation.splits import _cells\nfrom senselab.utils.evaluation import _ids\n",
    "ids": "from ._ids import key_array\nfrom senselab.utils.evaluation._ids import is_missing\n",
    "dunder": "from . import splits\nx = splits.__name__\n",
}


def _private_uses(source: str) -> list[str]:
    """Private names a module takes from a sibling evaluation module (other than `_ids`)."""
    package = "senselab.utils.evaluation"
    hits: list[str] = []
    aliases: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = package + ("." + module if module else "")
            if not module.startswith(package):
                continue
            for alias in node.names:
                target = module if module != package else f"{package}.{alias.name}"
                if module == package and not alias.name.startswith("_"):
                    aliases.add(alias.asname or alias.name)
                    continue
                if target.endswith("._ids"):
                    continue
                if alias.name.startswith("_") and not alias.name.startswith("__"):
                    hits.append(f"{module}:{alias.name}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(package + ".") and not alias.name.endswith("._ids"):
                    aliases.add(alias.asname or alias.name)
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in aliases
            and node.attr.startswith("_")
            and not node.attr.startswith("__")
        ):
            hits.append(f"{node.value.id}.{node.attr}")
    return hits


@pytest.mark.parametrize("name", sorted(_EVASIONS))
def test_private_use_detector_catches_evasions(name: str) -> None:
    """Each evasion of a line-based check is caught."""
    assert _private_uses(_EVASIONS[name])


@pytest.mark.parametrize("name", sorted(_ALLOWED))
def test_private_use_detector_allows_legitimate_code(name: str) -> None:
    """Comments, `_ids` imports and dunder attributes are not flagged."""
    assert _private_uses(_ALLOWED[name]) == []


def test_package_takes_no_private_names_from_siblings() -> None:
    """No evaluation module uses another's private names."""
    package = ROOT / "src" / "senselab" / "utils" / "evaluation"
    hits = {p.name: _private_uses(p.read_text()) for p in package.glob("*.py")}
    assert {name: found for name, found in hits.items() if found} == {}
