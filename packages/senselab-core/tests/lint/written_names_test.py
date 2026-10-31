"""Every file the store, fetch cache, locks and run manifests write is named by a grammar their modules export.

The modules' `WRITTEN_NAMES` (with `storage.WRITTEN_NAMES`) are the list. Every call to a `StorageDir`
write method in them passes `grammar=` naming one of those grammars, and nothing outside `StorageDir`
writes, renames, links, makes directories, changes modes or times by any other API (H-R40).
"""

import ast
import re
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from senselab_core.runtime import fetch, locks, revision, storage, store

RUNTIME = Path(__file__).resolve().parents[2] / "src" / "senselab_core" / "runtime"
MODULES: dict[str, ModuleType] = {
    "storage": storage,
    "store": store,
    "fetch": fetch,
    "locks": locks,
    "revision": revision,
}
WRITE_METHODS = frozenset({"open_new", "write_new", "write_atomically", "symlink", "link", "rename", "open_or_create"})
RAW_OS = frozenset(
    {
        "open",
        "rename",
        "replace",
        "link",
        "symlink",
        "mkdir",
        "makedirs",
        "chmod",
        "fchmod",
        "lchmod",
        "chown",
        "utime",
        "truncate",
        "ftruncate",
        "mkfifo",
        "mknod",
        "remove",
        "unlink",
        "rmdir",
    }
)
RAW_METHODS = frozenset({"write_text", "write_bytes", "touch", "mkdir", "chmod", "symlink_to", "hardlink_to"})
ALLOWED_GRAMMARS = {("storage", "write_atomically"): {"_CALLER_FILE"}}
"""Module-level writers outside any storage root: `storage.write_atomically` writes the caller's own file."""
ALLOWED_RAW = {("storage", "set_mode"): {"os.fchmod"}}
"""Raw calls outside `StorageDir`, by function: `set_mode` changes a mode through the descriptor it is given."""


def written_names() -> frozenset[re.Pattern[str]]:
    """Return the union of the modules' `WRITTEN_NAMES`."""
    found: set[re.Pattern[str]] = set()
    for module in MODULES.values():
        found |= module.WRITTEN_NAMES
    return frozenset(found)


def _functions(tree: ast.Module) -> dict[ast.AST, tuple[str, ...]]:
    """Map every node to the qualified name of the function or class it is in."""
    where: dict[ast.AST, tuple[str, ...]] = {}

    def visit(node: ast.AST, scope: tuple[str, ...]) -> None:
        for child in ast.iter_child_nodes(node):
            inner = (*scope, child.name) if isinstance(child, (ast.FunctionDef, ast.ClassDef)) else scope
            where[child] = inner
            visit(child, inner)

    visit(tree, ())
    return where


def violations(source: str, module: str, grammars: frozenset[re.Pattern[str]], namespace: object) -> list[str]:
    """Return `module:line message` for each write whose name is not a listed grammar, or that bypasses `StorageDir`."""
    tree = ast.parse(source)
    where = _functions(tree)
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
                if where.get(node, ())[:1] != ("StorageDir",):
                    found.append(f"{module}:{node.lineno} builtin open")
            continue
        scope = where.get(node, ())
        inside = scope[:1] == ("StorageDir",)
        attr = node.func.attr
        base = node.func.value
        if isinstance(base, ast.Name) and base.id in ("os", "shutil") and (base.id == "shutil" or attr in RAW_OS):
            if not inside and f"{base.id}.{attr}" not in ALLOWED_RAW.get((module, ".".join(scope)), set()):
                found.append(f"{module}:{node.lineno} {base.id}.{attr} outside StorageDir")
            continue
        if attr in RAW_METHODS and not (isinstance(base, ast.Name) and base.id == "self" and inside):
            found.append(f"{module}:{node.lineno} .{attr}")
            continue
        if attr not in WRITE_METHODS or inside or (isinstance(base, ast.Name) and base.id in ("os", "fcntl")):
            continue
        grammar = next((k.value for k in node.keywords if k.arg == "grammar"), None)
        if not isinstance(grammar, ast.Name):
            found.append(f"{module}:{node.lineno} .{attr} without a named grammar=")
            continue
        if grammar.id in ALLOWED_GRAMMARS.get((module, ".".join(scope)), set()):
            continue
        if getattr(namespace, grammar.id, None) not in grammars:
            found.append(f"{module}:{node.lineno} .{attr} grammar={grammar.id} is not in WRITTEN_NAMES")
    return found


@pytest.mark.logic
def test_every_write_names_a_listed_grammar_and_goes_through_storage_dir() -> None:
    """In the store, fetch, lock and run-manifest modules, every write passes a listed grammar, and none bypasses."""
    grammars = written_names()
    found = [
        hit
        for name, module in MODULES.items()
        for hit in violations((RUNTIME / f"{name}.py").read_text(), name, grammars, module)
    ]
    assert found == []


@pytest.mark.logic
@pytest.mark.parametrize(
    "line",
    [
        "d.write_new(n, b'', 0o600)",
        "d.write_new(n, b'', 0o600, grammar=re.compile('.*'))",
        "d.rename(a, d, b, grammar=UNLISTED)",
        "d.write_atomically(n, b'', grammar=UNLISTED)",
        "os.rename(a, b)",
        "os.chmod(p, 0o600)",
        "os.utime(p)",
        "os.open(p, os.O_CREAT)",
        "shutil.copy(a, b)",
        "Path(p).write_text('x')",
        "Path(p).touch()",
        "open(p, 'w')",
    ],
)
def test_the_check_finds_every_unlisted_or_raw_write(line: str) -> None:
    """A write without a grammar, with an unlisted one, or through a raw API is reported."""
    listed_grammar = re.compile("y")
    namespace = SimpleNamespace(UNLISTED=re.compile("x"), LISTED=listed_grammar)
    source = f"import os, re, shutil\nfrom pathlib import Path\ndef f(d, n, a, b, p):\n    {line}\n"
    assert violations(source, "m", frozenset({listed_grammar}), namespace)
    listed = "def f(d, n, a, b):\n    d.write_new(n, b'', 0o600, grammar=LISTED)\n"
    assert violations(listed, "m", frozenset({listed_grammar}), namespace) == []
