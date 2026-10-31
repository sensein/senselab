"""Package sources delete files only through `senselab_worker.ownership.remove_owned`."""

import ast
from pathlib import Path

import pytest

PACKAGES = Path(__file__).resolve().parents[3]
SOURCES = sorted(PACKAGES.glob("*/src/**/*.py"))
DELETING_ATTRIBUTES = {"unlink", "rmtree", "rmdir", "removedirs"}
DELETING_NAMES = {("os", "remove"), ("os", "unlink"), ("os", "rmdir"), ("os", "removedirs"), ("shutil", "rmtree")}
ALLOWED = {
    ("senselab_worker/ownership.py", "remove_owned"): None,
    ("senselab_core/runtime/worker_client.py", "WorkerClient._shutdown"): {"os.rmdir(self._array_dir)"},
}
"""Per (module, qualname), the calls allowed there as `<resolved target>(<argument source>)`; None allows any."""


def _imports(tree: ast.AST) -> dict[str, str]:
    """Map each name a module binds by `import` to the module it resolves to."""
    bound: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )
    return bound


def _call_name(call: ast.Call, bound: dict[str, str]) -> str | None:
    """Return the deletion call's resolved target, the base module through the module's imports, or None."""
    func = call.func
    if isinstance(func, ast.Attribute):
        name = func.value.id if isinstance(func.value, ast.Name) else "?"
        base = bound.get(name, f"<unbound {name}>")
        if func.attr in DELETING_ATTRIBUTES or {(base, func.attr), (name, func.attr)} & DELETING_NAMES:
            return f"{base}.{func.attr}"
    if isinstance(func, ast.Name) and any(func.id == name for _, name in DELETING_NAMES):
        return func.id
    return None


def deletions(source: str, module: str) -> list[str]:
    """Return `module:line qualname call` for every deletion call or import that the allowlist does not cover."""
    found: list[str] = []
    tree = ast.parse(source)
    bound = _imports(tree)

    def visit(node: ast.AST, scope: tuple[str, ...]) -> None:
        for child in ast.iter_child_nodes(node):
            inner = scope
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                inner = (*scope, child.name)
            if isinstance(child, ast.ImportFrom) and child.module in ("os", "shutil"):
                for alias in child.names:
                    if (child.module, alias.name) in DELETING_NAMES:
                        found.append(f"{module}:{child.lineno} import {child.module}.{alias.name}")
            if isinstance(child, ast.Call) and (name := _call_name(child, bound)) is not None:
                call = f"{name}({', '.join(ast.unparse(a) for a in [*child.args, *child.keywords])})"
                if not _allowed(module, inner, call):
                    found.append(f"{module}:{child.lineno} {'.'.join(scope) or '<module>'} {call}")
            visit(child, inner)

    visit(tree, ())
    return found


def _allowed(module: str, scope: tuple[str, ...], call: str) -> bool:
    for (where, qualname), calls in ALLOWED.items():
        parts = tuple(qualname.split("."))
        if module.endswith(where) and scope[: len(parts)] == parts and (calls is None or call in calls):
            return True
    return False


@pytest.mark.logic
def test_no_deletion_outside_remove_owned() -> None:
    """No package source unlinks, removes or deletes a tree except through `remove_owned`."""
    assert SOURCES
    found = [hit for path in SOURCES for hit in deletions(path.read_text(), path.relative_to(PACKAGES).as_posix())]
    assert found == []


@pytest.mark.logic
@pytest.mark.parametrize(
    "line",
    [
        "os.unlink(p)",
        "os.remove(p)",
        "os.rmdir(p)",
        "os.removedirs(p)",
        "shutil.rmtree(p)",
        "Path(p).unlink(missing_ok=True)",
        "self.path.rmdir()",
    ],
)
def test_the_lint_finds_every_deletion_form(line: str) -> None:
    """Each form of deletion is reported in an ordinary module, and an allowed scope covers only its own call."""
    assert deletions(f"def f(p):\n    {line}\n", "senselab_core/other.py")
    assert not deletions(f"def remove_owned(p):\n    {line}\n", "senselab_worker/ownership.py")
    in_shutdown = f"class WorkerClient:\n    def _shutdown(self, p):\n        {line}\n"
    assert deletions(in_shutdown, "senselab_core/runtime/worker_client.py")


@pytest.mark.contract
@pytest.mark.parametrize(
    ("imports", "line", "allowed"),
    [
        ("import os", "os.rmdir(self._array_dir)", True),
        ("import os as o", "o.rmdir(self._array_dir)", True),
        ("import shutil as os", "os.rmdir(self._array_dir)", False),
        ("import pathlib as os", "os.rmdir(self._array_dir)", False),
        ("", "os.rmdir(self._array_dir)", False),
        ("import os", "os.rmdir(p)", False),
        ("import os", "os.rmdir(self._array_dir.parent)", False),
        ("import os", "os.rmdir(self._array_dir, dir_fd=3)", False),
        ("import os", "os.removedirs(self._array_dir)", False),
        ("import os", "self._array_dir.rmdir()", False),
    ],
)
def test_shutdown_allowance_is_one_resolved_call_on_one_argument(imports: str, line: str, allowed: bool) -> None:
    """`WorkerClient._shutdown` may call only `os.rmdir(self._array_dir)`, with `os` resolved through the imports."""
    source = f"{imports}\nclass WorkerClient:\n    def _shutdown(self, p):\n        {line}\n"
    assert (deletions(source, "senselab_core/runtime/worker_client.py") == []) is allowed


@pytest.mark.logic
@pytest.mark.parametrize("line", ["from os import remove", "from os import unlink as u", "from shutil import rmtree"])
def test_importing_a_deletion_function_is_reported_anywhere(line: str) -> None:
    """A deletion function imported by name is reported, even inside an allowed scope."""
    assert deletions(f"def remove_owned(p):\n    {line}\n", "senselab_worker/ownership.py")
