"""No destructive call is reachable from a handler that catches `OSError`: a failed read never deletes or moves data."""

import ast
import builtins
import importlib
from pathlib import Path

import pytest

PACKAGES = Path(__file__).resolve().parents[3]
SOURCES = sorted(PACKAGES.glob("*/src/**/*.py"))
DESTRUCTIVE_MODULE_CALLS = {
    ("os", "rename"),
    ("os", "replace"),
    ("os", "unlink"),
    ("os", "remove"),
    ("os", "rmdir"),
    ("shutil", "rmtree"),
    ("shutil", "move"),
}
DESTRUCTIVE_METHODS = {"unlink", "rename", "rmdir", "_quarantine"}
DESTRUCTIVE_FUNCTIONS = {"remove_owned"}
BROAD = {"OSError", "Exception", "BaseException"}
SCRATCH_CLEANUP = {
    ("senselab_core.runtime.storage", "write_atomically", "remove_owned(pattern=TEMPORARY_NAME)"): "its temporary",
    ("senselab_worker.codec", "_write_array", "remove_owned(pattern=ARRAY_TEMP_NAME)"): "its temporary array file",
    ("senselab_core.runtime.worker_client", "WorkerClient._remove_arrays", "remove_owned(pattern=ARRAY_NAME)"): (
        "an array file in the worker's private channel"
    ),
    ("senselab_core.runtime.worker_client", "WorkerClient._remove_arrays", "remove_owned(pattern=ARRAY_TEMP_NAME)"): (
        "a temporary array file in the worker's private channel"
    ),
    (
        "senselab_core.runtime.worker_client",
        "WorkerClient._shutdown",
        "os.rmdir",
    ): "the worker's emptied private channel",
}
"""Call sites, by module, enclosing function and call, whose deletions touch only scratch the same object created.

A `remove_owned` call is identified with its `pattern=` argument, so a site deleting under another
grammar, or any other destructive call added to these functions, is reported.
"""


def _module_name(path: Path) -> str:
    rel = path.relative_to(PACKAGES).parts
    return ".".join(rel[2:]).removesuffix(".py").removesuffix(".__init__")


class _Module:
    """One parsed module: its functions by qualified name, its classes' bases, and its imported names."""

    def __init__(self, source: str, name: str) -> None:
        self.name = name
        self.tree = ast.parse(source)
        self.functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
        self.bases: dict[str, list[ast.expr]] = {}
        self.imported: dict[str, tuple[str, str]] = {}
        self.aliases: dict[str, ast.expr] = {}
        for node in self.tree.body:
            self._index(node, ())

    def _index(self, node: ast.stmt, scope: tuple[str, ...]) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self.functions[".".join((*scope, node.name))] = node
        elif isinstance(node, ast.ClassDef):
            self.bases[node.name] = node.bases
            for child in node.body:
                self._index(child, (*scope, node.name))
        elif isinstance(node, ast.ImportFrom) and node.module and not scope:
            for alias in node.names:
                self.imported[alias.asname or alias.name] = (node.module, alias.name)
        elif isinstance(node, ast.Assign) and not scope and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                self.aliases[target.id] = node.value

    def catches_oserror(self, expr: ast.expr | None, seen: frozenset[str] = frozenset()) -> bool:
        """Return whether an `except` clause of type `expr` catches some `OSError` (bare and broad clauses do)."""
        if expr is None:
            return True
        if isinstance(expr, ast.Tuple):
            return any(self.catches_oserror(e, seen) for e in expr.elts)
        if isinstance(expr, ast.Attribute):
            return self._resolved(_dotted(expr))
        if not isinstance(expr, ast.Name) or expr.id in seen:
            return True
        name = expr.id
        if name in BROAD:
            return True
        builtin = getattr(builtins, name, None)
        if isinstance(builtin, type):
            return issubclass(builtin, OSError) or issubclass(OSError, builtin)
        if name in self.bases:
            return any(self.catches_oserror(b, seen | {name}) for b in self.bases[name])
        if name in self.aliases:
            return self.catches_oserror(self.aliases[name], seen | {name})
        if name in self.imported:
            module, attr = self.imported[name]
            return self._resolved(f"{module}.{attr}")
        return True

    @staticmethod
    def _resolved(dotted: str | None) -> bool:
        if dotted is None:
            return True
        module, _, attr = dotted.rpartition(".")
        try:
            value = getattr(importlib.import_module(module), attr)
        except (ImportError, AttributeError, ValueError):
            return True
        if isinstance(value, tuple):
            return any(not isinstance(v, type) or issubclass(v, OSError) or issubclass(OSError, v) for v in value)
        return not isinstance(value, type) or issubclass(value, OSError) or issubclass(OSError, value)


def _dotted(expr: ast.expr) -> str | None:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        base = _dotted(expr.value)
        return None if base is None else f"{base}.{expr.attr}"
    return None


def _destructive(call: ast.Call) -> str | None:
    """Return the call's signature if it is destructive: `os.rename`, `.unlink`, `remove_owned(pattern=X)`, ..."""
    func = call.func
    name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
    if name in DESTRUCTIVE_FUNCTIONS:
        pattern = next((ast.unparse(k.value) for k in call.keywords if k.arg == "pattern"), "?")
        return f"{name}(pattern={pattern})"
    if isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name) and (func.value.id, func.attr) in DESTRUCTIVE_MODULE_CALLS:
            return f"{func.value.id}.{func.attr}"
        if func.attr in DESTRUCTIVE_METHODS or func.attr in DESTRUCTIVE_FUNCTIONS:
            return f".{func.attr}"
    if isinstance(func, ast.Name) and (
        func.id in DESTRUCTIVE_FUNCTIONS or func.id in {n for _, n in DESTRUCTIVE_MODULE_CALLS}
    ):
        return func.id
    return None


def _callee(module: _Module, call: ast.Call, scope: tuple[str, ...]) -> str | None:
    """Resolve a call to a function of `module`: a module-level name, or `self.`/`cls.`/`Class.` method."""
    func = call.func
    if isinstance(func, ast.Name) and func.id in module.functions:
        return func.id
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        owner = func.value.id
        cls = scope[0] if scope and owner in ("self", "cls") else owner
        name = f"{cls}.{func.attr}"
        if name in module.functions:
            return name
    return None


def _reachable(module: _Module, body: list[ast.stmt], scope: tuple[str, ...], where: str, seen: set[str]) -> list[str]:
    """Return the destructive calls in `body` (inside function `where`) or in same-module functions it calls."""
    found: list[str] = []
    for stmt in body:
        for node in ast.walk(stmt):
            if not isinstance(node, ast.Call):
                continue
            if (name := _destructive(node)) is not None and (module.name, where, name) not in SCRATCH_CLEANUP:
                found.append(f"{name} (line {node.lineno})")
            callee = _callee(module, node, scope)
            if callee is not None and callee not in seen:
                seen.add(callee)
                fn = module.functions[callee]
                inner = tuple(callee.split(".")[:-1])
                found.extend(f"{hit} via {callee}" for hit in _reachable(module, fn.body, inner, callee, seen))
    return found


def violations(source: str, name: str) -> list[str]:
    """Return `module:line handler -> call` for each destructive call reachable from an `OSError`-catching handler."""
    module = _Module(source, name)
    found: list[str] = []

    def visit(node: ast.AST, scope: tuple[str, ...], where: tuple[str, ...]) -> None:
        for child in ast.iter_child_nodes(node):
            inner, inner_where = scope, where
            if isinstance(child, ast.ClassDef):
                inner, inner_where = (*scope, child.name), (*where, child.name)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                inner_where = (*where, child.name)
            if isinstance(child, ast.ExceptHandler) and module.catches_oserror(child.type):
                for hit in _reachable(module, child.body, scope, ".".join(where), set()):
                    found.append(f"{name}:{child.lineno} -> {hit}")
            visit(child, inner, inner_where)

    visit(module.tree, (), ())
    return found


@pytest.mark.logic
def test_no_destructive_call_under_oserror_handler() -> None:
    """No package source renames, replaces, removes or quarantines from a handler that catches an OSError."""
    assert SOURCES
    found = [hit for path in SOURCES for hit in violations(path.read_text(), _module_name(path))]
    assert found == []


@pytest.mark.logic
@pytest.mark.parametrize(
    "call",
    [
        "os.rename(a, b)",
        "os.replace(a, b)",
        "os.unlink(a)",
        "os.remove(a)",
        "os.rmdir(a)",
        "shutil.rmtree(a)",
        "shutil.move(a, b)",
        "Path(a).unlink()",
        "Path(a).rename(b)",
        "self.path.rmdir()",
        "remove_owned(r, a, pattern=p)",
        "self._quarantine(a)",
    ],
)
@pytest.mark.parametrize(
    "clause",
    [
        "OSError",
        "FileNotFoundError",
        "PermissionError",
        "Exception",
        "BaseException",
        "",
        "(ValueError, OSError)",
        "StoreUnavailable",
        "_ERRORS",
    ],
)
def test_the_lint_finds_every_destructive_call_under_every_oserror_clause(call: str, clause: str) -> None:
    """Every destructive form is reported under every clause that catches an OSError, directly and through helpers."""
    header = (
        "import os, shutil\nfrom senselab_core.runtime.storage import StoreUnavailable\n_ERRORS = (KeyError, IOError)\n"
    )
    direct = f"{header}def f(a, b):\n    try:\n        pass\n    except {clause}:\n        {call}\n"
    assert violations(direct, "m"), direct
    helper = (
        f"{header}def g(a, b):\n    {call}\n\n"
        f"class C:\n    def h(self, a, b):\n        g(a, b)\n\n"
        f"    def f(self, a, b):\n        try:\n            pass\n        except {clause}:\n            self.h(a, b)\n"
    )
    assert violations(helper, "m"), helper


@pytest.mark.logic
def test_scratch_cleanup_allowance_names_existing_functions() -> None:
    """Each allowed scratch cleanup names a call that exists in its function, with the same `pattern=` argument."""
    modules = {_module_name(p): _Module(p.read_text(), _module_name(p)) for p in SOURCES}
    for module, function, call in SCRATCH_CLEANUP:
        fn = modules[module].functions[function]
        sites = {_destructive(node) for node in ast.walk(fn) if isinstance(node, ast.Call)}
        assert call in sites, (module, function, call)


@pytest.mark.logic
@pytest.mark.parametrize(
    "call",
    ["remove_owned(r, a, pattern=OTHER_NAME)", "remove_owned(r, a, TEMPORARY_NAME)", "os.unlink(a)", "a.rename(b)"],
)
def test_scratch_allowance_does_not_cover_other_calls_in_the_same_function(call: str) -> None:
    """Inside an allowed function, a deletion under another grammar, or any other destructive call, is reported."""
    source = f"import os\ndef write_atomically(r, a, b):\n    try:\n        pass\n    except OSError:\n        {call}\n"
    assert violations(source, "senselab_core.runtime.storage")
    allowed = source.replace(call, "remove_owned(r, a, pattern=TEMPORARY_NAME)")
    assert violations(allowed, "senselab_core.runtime.storage") == []


@pytest.mark.logic
@pytest.mark.parametrize("clause", ["ValueError", "CorruptEntryError", "(KeyError, TypeError)", "FetchError"])
def test_handlers_that_cannot_catch_oserror_are_not_reported(clause: str) -> None:
    """A destructive call under a handler for content errors, or in a try body or finally, is not reported."""
    source = (
        "import os\nfrom senselab_core.runtime.store import CorruptEntryError\n"
        "from senselab_core.runtime.fetch import FetchError\n"
        f"def f(a, b):\n    try:\n        os.rename(a, b)\n    except {clause}:\n        os.unlink(a)\n"
        "    finally:\n        os.unlink(b)\n"
    )
    assert violations(source, "m") == []
