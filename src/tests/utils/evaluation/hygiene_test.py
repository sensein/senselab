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


# Evasions of a line regex and of a flat import walk, as (source, the
# importing module); each must be reported. Legitimate code that must not be reported follows.
_EVAL = "senselab.utils.evaluation"
_EVASIONS = {
    "multiline": ("from senselab.utils.evaluation.splits import (\n    Split,\n    _cells,\n)\n", f"{_EVAL}.x"),
    "relative": ("from .splits import _concat\n", f"{_EVAL}.x"),
    "relative-alias": ("from .bootstrap import _defined as defined\n", f"{_EVAL}.x"),
    "module-alias-attribute": ("import senselab.utils.evaluation.splits as sp\nx = sp._sorted\n", f"{_EVAL}.x"),
    "package-import-attribute": ("from . import splits\nx = splits._majority\n", f"{_EVAL}.x"),
    "dotted-module": (
        "import senselab.utils.evaluation.splits\nx = senselab.utils.evaluation.splits._sorted\n",
        f"{_EVAL}.x",
    ),
    "package-alias": ("import senselab.utils.evaluation as ev\nx = ev.bootstrap._defined\n", f"{_EVAL}.x"),
    "from-package-alias": ("from senselab.utils import evaluation as ev\nx = ev.splits._concat\n", f"{_EVAL}.x"),
    "getattr": ("from . import splits\nx = getattr(splits, '_majority')\n", f"{_EVAL}.x"),
    "relative-level-2": ("from ..data_structures.events import _require_yaml_string\n", f"{_EVAL}.x"),
}
_ALLOWED = {
    "comment": (
        "# from senselab.utils.evaluation.splits import _cells\nfrom senselab.utils.evaluation import _ids\n",
        f"{_EVAL}.x",
    ),
    "ids": ("from ._ids import key_array\nfrom senselab.utils.evaluation._ids import is_missing\n", f"{_EVAL}.x"),
    "dunder": ("from . import splits\nx = splits.__name__\n", f"{_EVAL}.x"),
    "public-getattr": ("from . import splits\nx = getattr(splits, 'grouped_splits')\n", f"{_EVAL}.x"),
    "third-party": ("import numpy as np\nx = np._NoValue\n", f"{_EVAL}.x"),
}


def _dotted(node: ast.expr) -> str | None:
    """`a.b.c` for a Name/Attribute chain, else None."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    return ".".join([node.id, *reversed(parts)])


def _private_uses(source: str, module: str) -> list[str]:
    """Private names `module` (a dotted module name) takes from another senselab module, other than `_ids`.

    Covers `from` imports (absolute, or relative resolved by level against `module`'s package), module
    aliases from `import a.b.c [as x]` and `from pkg import mod [as x]`, dotted attribute chains on
    them (`ev.bootstrap._x`), and `getattr(mod, "_x")` with a literal name. Reported as `module:name`.
    """

    def private(name: str) -> bool:
        return name.startswith("_") and not name.startswith("__")

    def foreign(target: str) -> bool:
        return target.startswith("senselab.") and target != module and not target.endswith("._ids")

    package = module.rsplit(".", 1)[0]
    tree = ast.parse(source)
    hits: list[str] = []
    bound: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = package.split(".")
                anchor = anchor[: len(anchor) - (node.level - 1)]
                base = ".".join(anchor + ([base] if base else []))
            for alias in node.names:
                if private(alias.name) and foreign(base) and foreign(f"{base}.{alias.name}"):
                    hits.append(f"{base}:{alias.name}")
                bound[alias.asname or alias.name] = f"{base}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    bound[alias.asname] = alias.name
                else:
                    head = alias.name.split(".")[0]
                    bound[head] = head

    def resolve(expr: ast.expr) -> str | None:
        dotted = _dotted(expr)
        if dotted is None:
            return None
        head, _, rest = dotted.partition(".")
        if head not in bound:
            return None
        return bound[head] + ("." + rest if rest else "")

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and private(node.attr):
            target = resolve(node.value)
            if target is not None and foreign(target):
                hits.append(f"{target}:{node.attr}")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
            and private(node.args[1].value)
        ):
            target = resolve(node.args[0])
            if target is not None and foreign(target):
                hits.append(f"{target}:{node.args[1].value}")
    return hits


@pytest.mark.parametrize("name", sorted(_EVASIONS))
def test_private_use_detector_catches_evasions(name: str) -> None:
    """Each evasion is caught."""
    assert _private_uses(*_EVASIONS[name])


def test_relative_imports_are_reported_under_their_real_module() -> None:
    """`from ..data_structures.events import _x` in an evaluation module names the data_structures module."""
    assert _private_uses(*_EVASIONS["relative-level-2"]) == [
        "senselab.utils.data_structures.events:_require_yaml_string"
    ]


@pytest.mark.parametrize("name", sorted(_ALLOWED))
def test_private_use_detector_allows_legitimate_code(name: str) -> None:
    """Comments, `_ids` imports, dunders, public getattr and third-party privates are not flagged."""
    assert _private_uses(*_ALLOWED[name]) == []


def test_package_takes_no_private_names_from_other_modules() -> None:
    """No evaluation module uses another senselab module's private names (`_ids` excepted)."""
    package = ROOT / "src" / "senselab" / "utils" / "evaluation"
    hits = {p.name: _private_uses(p.read_text(), f"{_EVAL}.{p.stem}") for p in package.glob("*.py")}
    assert {name: found for name, found in hits.items() if found} == {}
