"""Fail when package source fixes a parameter that belongs in config (P-34).

Two forms are refused under ``packages/*/src``:

- a module-level numeric assignment whose name ends in one of ``SUFFIXES``;
- a numeric literal other than ``0`` (no wait) passed to ``sleep(...)``, ``wait(...)``, ``join(...)`` or as
  ``timeout=``.

An invariant is declared ``NAME: Final = ...  # invariant: <why it cannot vary>`` and listed under ``invariants`` in
``scripts/data/invariants.yaml`` with its reason. A site listed under ``pending`` names the register entry that moves
it to config. A listed entry that matches no site is refused too.

Usage::

    uv run python scripts/check_fixed_parameters.py
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = Path(__file__).resolve().parent / "data" / "invariants.yaml"
ALLOWED_FORMAT = "invariants@1.0"
SOURCE = re.compile(r"^packages/[^/]+/src/.+\.py$")
SUFFIXES = ("_S", "_MS", "_BYTES", "_ATTEMPTS", "_RETRIES", "_WORKERS", "_ITEMS", "_POLLS", "_BUDGET")
CALLS = ("sleep", "wait", "join")
DECLARED = re.compile(r"#\s*invariant:\s*\S")


class Site(NamedTuple):
    """A fixed parameter: its file, line and name (a constant, a call, or ``timeout=``)."""

    path: str
    line: int
    name: str


def _numeric(node: ast.AST) -> bool:
    """Whether `node` is a number built only from literals: ``5``, ``-1``, ``10**9``, ``60 * 60``."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
    if isinstance(node, ast.UnaryOp):
        return _numeric(node.operand)
    if isinstance(node, ast.BinOp):
        return _numeric(node.left) and _numeric(node.right)
    return False


def _waits(node: ast.AST) -> bool:
    """Whether `node` is a numeric literal wait other than zero."""
    return _numeric(node) and not (isinstance(node, ast.Constant) and node.value == 0)


def _callee(node: ast.Call) -> str:
    func = node.func
    return func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""


def sites(path: str, source: str) -> list[tuple[Site, bool]]:
    """Each fixed parameter in `source`, with whether its line declares it an invariant."""
    lines = source.splitlines()
    tree = ast.parse(source)
    found: list[Site] = []
    for stmt in tree.body:
        target, value = None, None
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target, value = stmt.targets[0], stmt.value
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            target, value = stmt.target, stmt.value
        if isinstance(target, ast.Name) and target.id.endswith(SUFFIXES) and _numeric(value):
            found.append(Site(path, stmt.lineno, target.id))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _callee(node) in CALLS and node.args and _waits(node.args[0]):
            found.append(Site(path, node.lineno, f"{_callee(node)}()"))
        found += [Site(path, node.lineno, "timeout=") for k in node.keywords if k.arg == "timeout" and _waits(k.value)]
    return [(s, bool(DECLARED.search(lines[s.line - 1]))) for s in found]


def load_allowed(path: Path = ALLOWED) -> tuple[frozenset[tuple[str, str]], frozenset[tuple[str, str]]]:
    """Return the ``(path, name)`` pairs listed as invariants and as pending; an entry without its reason is refused."""
    raw = yaml.safe_load(path.read_text())
    if raw.get("format") != ALLOWED_FORMAT:
        raise ValueError(f"{path}: format is {raw.get('format')!r}, not {ALLOWED_FORMAT!r}")
    out = []
    for section, why in (("invariants", "reason"), ("pending", "register")):
        pairs = set()
        for entry in raw.get(section) or []:
            if not str(entry.get(why, "")).strip():
                raise ValueError(f"{path}: {section} {entry.get('path')}::{entry.get('name')} needs a {why}")
            pairs.add((entry["path"], entry["name"]))
        out.append(frozenset(pairs))
    return out[0], out[1]


def problems(
    found: list[tuple[Site, bool]], invariants: frozenset[tuple[str, str]], pending: frozenset[tuple[str, str]]
) -> list[str]:
    """One message per site that is neither a declared, listed invariant nor pending, and per unmatched entry."""
    seen = {(site.path, site.name) for site, _ in found}
    out = [
        f"{p}::{n} is listed in {ALLOWED.name} but not found; remove it"
        for p, n in sorted((invariants | pending) - seen)
    ]
    for site, declared in found:
        key = (site.path, site.name)
        if key in pending:
            continue
        if key in invariants and declared:
            continue
        if key in invariants:
            out.append(f"{site.path}:{site.line}: {site.name} is listed; declare it `: Final = ...  # invariant: ...`")
        else:
            out.append(f"{site.path}:{site.line}: {site.name} is a fixed parameter; make it config (P-34)")
    return out


def tracked_source(root: Path = ROOT) -> list[str]:
    """Every Python file git tracks under ``packages/*/src``."""
    out = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "packages"], capture_output=True, check=True)
    return [p for p in out.stdout.decode().split("\0") if SOURCE.match(p)]


def main(root: Path = ROOT) -> int:
    """Check every tracked file under ``packages/*/src`` of `root`; print the problems; return 1 if any."""
    invariants, pending = load_allowed()
    found = [s for p in tracked_source(root) for s in sites(p, (root / p).read_text())]
    out = problems(found, invariants, pending)
    for line in out:
        print(line)
    return 1 if out else 0


if __name__ == "__main__":
    sys.exit(main())
