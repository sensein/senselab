"""Fail when 2.x source or a benchmark reads the host name or the account name.

Refuses calls to ``socket.gethostname``, ``platform.node``, ``getpass.getuser`` and the ``nodename`` of
``os.uname()``. A record other users read names a host by ``host_token()`` and its hardware by ``HardwareClass``.
``ALLOWED`` lists the places that show the name only to the user running the code, or are 1.x.

Usage::

    uv run python scripts/check_host_identity.py            # packages/*/src and benchmarks/
    uv run python scripts/check_host_identity.py FILE ...   # just these (pre-commit)
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANNED_CALLS = {("socket", "gethostname"), ("platform", "node"), ("getpass", "getuser")}
ALLOWED = {
    "packages/senselab-core/src/senselab_core/runtime/profiles.py",
    "benchmarks/crisperwhisper_confidence_cost.py",
}


def _banned(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        owner = node.func.value
        if isinstance(owner, ast.Name) and (owner.id, node.func.attr) in BANNED_CALLS:
            return f"{owner.id}.{node.func.attr}"
    if isinstance(node, ast.Attribute) and node.attr == "nodename":
        return "uname().nodename"
    return None


def findings(paths: list[Path], root: Path = ROOT) -> list[str]:
    """Return one ``path:line: call`` finding per banned read in `paths`, none for an allowed path."""
    found = []
    for path in paths:
        rel = path.relative_to(root).as_posix() if path.is_absolute() else path.as_posix()
        if rel in ALLOWED or not path.is_file():
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (call := _banned(node)) is not None:
                found.append(f"{rel}:{node.lineno}: {call}")
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
