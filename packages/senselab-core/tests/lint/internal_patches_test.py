"""No test substitutes a senselab internal, except the declared shutdown boundaries in a child process.

The sweep reads every test module under ``packages/*/tests``. A substitution is a ``monkeypatch`` set or delete, a
``setattr`` or ``delattr``, or an assignment whose target is reached from a name bound by importing a senselab
package. The shutdown harness substitutes only through ``shutdown_schedule_fixture._owner``, which refuses a
boundary outside ``BOUNDARIES``; a ``setattr`` with an unpacked target is allowed in that module alone.
"""

import ast
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.logic

PACKAGES = Path(__file__).resolve().parents[3]
HARNESS = "shutdown_schedule_fixture.py"
MONKEYPATCH_WRITES = frozenset({"setattr", "setitem", "delattr", "delitem"})


def _senselab_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0].startswith("senselab"):
                    names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0].startswith("senselab"):
            names.update(alias.asname or alias.name for alias in node.names)
    return names


def _root(node: ast.expr) -> str | None:
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _is_internal(node: ast.expr, names: set[str]) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.startswith("senselab")
    return _root(node) in names


def _refused_writes(tree: ast.Module) -> set[int]:
    """The ids of the nodes inside a ``with pytest.raises(...)`` block: writes a test expects to be refused."""
    inside: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.With) and any(
            isinstance(item.context_expr, ast.Call) and ast.unparse(item.context_expr.func) == "pytest.raises"
            for item in node.items
        ):
            for statement in node.body:
                inside.update(id(child) for child in ast.walk(statement))
    return inside


def substitutions(source: str, filename: str) -> list[str]:
    """``<line>: <what>`` for each substitution of a senselab internal in the test module `source`."""
    tree = ast.parse(source)
    names = _senselab_names(tree)
    refused = _refused_writes(tree)
    found = []
    for node in ast.walk(tree):
        if id(node) in refused:
            continue
        if isinstance(node, ast.Call) and node.args:
            func, target = node.func, node.args[0]
            if isinstance(func, ast.Attribute) and func.attr in MONKEYPATCH_WRITES and _root(func) == "monkeypatch":
                if _is_internal(target, names):
                    found.append(f"{node.lineno}: monkeypatch.{func.attr} of {ast.unparse(target)}")
            elif isinstance(func, ast.Name) and func.id in ("setattr", "delattr"):
                if isinstance(target, ast.Starred) and filename != HARNESS:
                    found.append(f"{node.lineno}: {func.id} with an unpacked target")
                elif _is_internal(target, names):
                    found.append(f"{node.lineno}: {func.id} of {ast.unparse(target)}")
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for target in targets:
            if isinstance(target, (ast.Attribute, ast.Subscript)) and _root(target) in names:
                found.append(f"{target.lineno}: assignment to {ast.unparse(target)}")
    return found


def test_no_test_module_substitutes_a_senselab_internal() -> None:
    """Every test module under ``packages/`` is free of substitutions of senselab names."""
    found = {
        str(path.relative_to(PACKAGES)): hits
        for path in sorted(PACKAGES.glob("*/tests/**/*.py"))
        if (hits := substitutions(path.read_text(), path.name))
    }
    assert found == {}


def test_every_declared_shutdown_boundary_exists_and_nothing_else_is_accepted() -> None:
    """Each name in ``BOUNDARIES`` resolves; a name outside it is refused before anything is substituted."""
    sys.path.insert(0, str(PACKAGES / "senselab-core" / "tests" / "runtime"))
    try:
        import shutdown_schedule_fixture as harness
    finally:
        sys.path.pop(0)
    for boundary in harness.BOUNDARIES:
        owner, name = harness._owner(boundary)
        assert callable(getattr(owner, name)), boundary
    with pytest.raises(ValueError, match="not a declared shutdown boundary"):
        harness._owner("module:_live_members")


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("from senselab_core.runtime import store\nmonkeypatch.setattr(store, 'x', 1)\n", 1),
        ("monkeypatch.setattr('senselab_core.runtime.store.x', 1)\n", 1),
        ("import senselab_core.runtime.revision as mr\nmr._MEMO = {}\n", 1),
        ("from senselab_core.runtime.store import ResultStore\nsetattr(ResultStore, 'get', None)\n", 1),
        ("import os\nmonkeypatch.setattr(os, 'open', None)\nos.lstat = None\n", 0),
        ("from senselab_core.runtime import audio_io\naudio_io.VERIFIED.clear()\n", 0),
        ("setattr(*owner, value)\n", 1),
        ("from senselab_core.runtime.models import WEIGHTS\nwith pytest.raises(TypeError):\n    WEIGHTS['x'] = 1\n", 0),
    ],
    ids=["module", "dotted-string", "assignment", "builtin-setattr", "stdlib", "call", "unpacked", "refused-write"],
)
def test_the_sweep_finds_each_form_of_substitution(source: str, expected: int) -> None:
    """Planted modules: each form of substituting a senselab name is found; stdlib patches and calls are not."""
    assert len(substitutions(source, "planted_test.py")) == expected
