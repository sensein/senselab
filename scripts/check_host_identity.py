"""Fail when 2.x source or a benchmark reads the host name or the account name.

Refuses calls to ``socket.gethostname``, ``socket.getfqdn``, ``platform.node``, ``getpass.getuser``,
``os.getlogin`` and ``pwd.getpwuid``, under any import spelling, and reads of ``.nodename`` and of
``platform.uname().node``. A record other users read names a host by ``host_token()`` and its hardware by
``HardwareClass``. ``ALLOWED`` lists the places that show the name only to the user running the code, render it out
of a record, or are 1.x.

Usage::

    uv run python scripts/check_host_identity.py            # packages/*/src and benchmarks/
    uv run python scripts/check_host_identity.py FILE ...   # just these (pre-commit)
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANNED_CALLS = {
    ("socket", "gethostname"),
    ("socket", "getfqdn"),
    ("platform", "node"),
    ("getpass", "getuser"),
    ("os", "getlogin"),
    ("pwd", "getpwuid"),
}
ALLOWED = {
    "packages/senselab-core/src/senselab_core/runtime/profiles.py",
    "packages/senselab-core/src/senselab_core/runtime/execution/executor.py",
    "packages/senselab-core/src/senselab_core/runtime/locks.py",
    "packages/senselab-core/src/senselab_core/runtime/storage.py",
    "benchmarks/crisperwhisper_confidence_cost.py",
}


class _Imports:
    """The local names a module's imports bind: module aliases and functions imported by name."""

    def __init__(self, tree: ast.AST) -> None:
        self.modules: dict[str, str] = {}
        self.functions: dict[str, tuple[str, str]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.modules.update({a.asname or a.name: a.name for a in node.names})
            elif isinstance(node, ast.ImportFrom) and node.module:
                self.functions.update({a.asname or a.name: (node.module, a.name) for a in node.names})

    def target(self, func: ast.AST) -> tuple[str, str] | None:
        """Return `(module, name)` for a call of `module.name` under any alias, or of a name imported from a module."""
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.value.id in self.functions:
                return None
            return (self.modules.get(func.value.id, func.value.id), func.attr)
        if isinstance(func, ast.Name):
            return self.functions.get(func.id)
        return None


def _banned(node: ast.AST, imports: _Imports) -> str | None:
    target = imports.target(node.func) if isinstance(node, ast.Call) else None
    if target is not None and target in BANNED_CALLS:
        return ".".join(target)
    if isinstance(node, ast.Attribute) and node.attr == "nodename":
        return "uname().nodename"
    if (
        isinstance(node, ast.Attribute)
        and node.attr == "node"
        and isinstance(node.value, ast.Call)
        and imports.target(node.value.func) == ("platform", "uname")
    ):
        return "platform.uname().node"
    return None


def findings(paths: list[Path], root: Path = ROOT) -> list[str]:
    """Return one ``path:line: call`` finding per banned read in `paths`, none for an allowed path."""
    found = []
    for path in paths:
        rel = path.relative_to(root).as_posix() if path.is_absolute() else path.as_posix()
        if rel in ALLOWED or not path.is_file():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = _Imports(tree)
        for node in ast.walk(tree):
            if (call := _banned(node, imports)) is not None:
                found.append(f"{rel}:{getattr(node, 'lineno', 0)}: {call}")
    return found


def default_paths(root: Path = ROOT) -> list[Path]:
    """Return every Python file of the 2.x packages' source and of ``benchmarks/``."""
    return sorted([*root.glob("packages/*/src/**/*.py"), *root.glob("benchmarks/**/*.py")])


def main(argv: list[str]) -> int:
    """Check `argv` paths, or the default set when none are given."""
    hits = findings([Path(a) for a in argv] if argv else default_paths())
    for hit in hits:
        print(hit)
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
