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
    ("senselab_core/runtime/worker_client.py", "WorkerClient._shutdown"): {"os.rmdir"},
}


def _call_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Attribute):
        base = func.value.id if isinstance(func.value, ast.Name) else "?"
        if func.attr in DELETING_ATTRIBUTES or (base, func.attr) in DELETING_NAMES:
            return f"{base}.{func.attr}"
    if isinstance(func, ast.Name) and any(func.id == name for _, name in DELETING_NAMES):
        return func.id
    return None


def deletions(source: str, module: str) -> list[str]:
    """Return `module:line qualname call` for every deletion call or import that the allowlist does not cover."""
    found: list[str] = []

    def visit(node: ast.AST, scope: tuple[str, ...]) -> None:
        for child in ast.iter_child_nodes(node):
            inner = scope
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                inner = (*scope, child.name)
            if isinstance(child, ast.ImportFrom) and child.module in ("os", "shutil"):
                for alias in child.names:
                    if (child.module, alias.name) in DELETING_NAMES:
                        found.append(f"{module}:{child.lineno} import {child.module}.{alias.name}")
            if isinstance(child, ast.Call) and (name := _call_name(child)) is not None:
                if not _allowed(module, inner, name):
                    found.append(f"{module}:{child.lineno} {'.'.join(scope) or '<module>'} {name}")
            visit(child, inner)

    visit(ast.parse(source), ())
    return found


def _allowed(module: str, scope: tuple[str, ...], name: str) -> bool:
    for (where, qualname), names in ALLOWED.items():
        parts = tuple(qualname.split("."))
        if module.endswith(where) and scope[: len(parts)] == parts and (names is None or name in names):
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
    assert bool(deletions(in_shutdown, "senselab_core/runtime/worker_client.py")) == (line != "os.rmdir(p)")


@pytest.mark.logic
@pytest.mark.parametrize("line", ["from os import remove", "from os import unlink as u", "from shutil import rmtree"])
def test_importing_a_deletion_function_is_reported_anywhere(line: str) -> None:
    """A deletion function imported by name is reported, even inside an allowed scope."""
    assert deletions(f"def remove_owned(p):\n    {line}\n", "senselab_worker/ownership.py")
