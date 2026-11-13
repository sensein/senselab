"""Fail when package source fixes a parameter that belongs in config (P-34).

Two forms are refused under ``packages/*/src``:

- a module-level numeric assignment (in the module body or under its ``if``/``try``/``with`` blocks, not in a class
  or function) whose name ends in one of ``SUFFIXES``;
- a numeric literal other than ``0`` (no wait) passed to ``sleep(...)``, ``wait(...)``, ``join(...)`` or as
  ``timeout=``;
- a numeric literal other than ``0`` and ``1`` multiplying an expression that names a lease bound
  (``renew_fraction``, ``stale_after``, ``stale_after_s``): a beat factor is ``lease.missed_beats``.

An invariant is declared ``NAME: Final = ...  # invariant: <why it cannot vary>`` and listed under ``invariants`` in
``scripts/data/invariants.yaml`` with its reason. An entry under ``pending`` names the register entry that moves it to
config and the ``count`` of sites it admits (default 1); a file whose number of sites differs is refused, and so is a
listed entry that matches no site.

Usage::

    uv run python scripts/check_fixed_parameters.py
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path
from collections.abc import Iterator
from typing import NamedTuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = Path(__file__).resolve().parent / "data" / "invariants.yaml"
ALLOWED_FORMAT = "invariants@1.0"
SOURCE = re.compile(r"^packages/[^/]+/src/.+\.py$")
SUFFIXES = ("_S", "_MS", "_BYTES", "_ATTEMPTS", "_RETRIES", "_WORKERS", "_ITEMS", "_POLLS", "_BUDGET", "_MODE")
CALLS = ("sleep", "wait", "join")
BEAT_NAMES = frozenset({"renew_fraction", "stale_after", "stale_after_s"})
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


def _module_level(body: list[ast.stmt]) -> Iterator[ast.stmt]:
    """The statements of `body`, and of the ``if``, ``try`` and ``with`` blocks in it, outside classes and functions."""
    for stmt in body:
        yield stmt
        blocks: list[list[ast.stmt]] = []
        if isinstance(stmt, (ast.If, ast.With)):
            blocks = [stmt.body, getattr(stmt, "orelse", [])]
        elif isinstance(stmt, ast.Try):
            blocks = [stmt.body, stmt.orelse, stmt.finalbody, *(h.body for h in stmt.handlers)]
        for block in blocks:
            yield from _module_level(block)


def sites(path: str, source: str) -> list[tuple[Site, bool]]:
    """Each fixed parameter in `source`, with whether its line declares it an invariant."""
    lines = source.splitlines()
    tree = ast.parse(source)
    found: list[Site] = []
    for stmt in _module_level(tree.body):
        target, value = None, None
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target, value = stmt.targets[0], stmt.value
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            target, value = stmt.target, stmt.value
        if isinstance(target, ast.Name) and target.id.endswith(SUFFIXES) and value is not None and _numeric(value):
            found.append(Site(path, stmt.lineno, target.id))
    found += [Site(path, n.lineno, "beat factor") for n in ast.walk(tree) if isinstance(n, ast.BinOp) and _beat_factor(n)]
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _callee(node) in CALLS and node.args and _waits(node.args[0]):
            found.append(Site(path, node.lineno, f"{_callee(node)}()"))
        found += [Site(path, node.lineno, "timeout=") for k in node.keywords if k.arg == "timeout" and _waits(k.value)]
    return [(s, bool(DECLARED.search(lines[s.line - 1]))) for s in found]


def _names(node: ast.AST) -> set[str]:
    """The identifiers and attribute names in the expression `node`."""
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)
    }


def _beat_factor(node: ast.AST) -> bool:
    """Whether `node` multiplies a lease bound by a numeric literal other than 0 and 1."""
    if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)):
        return False
    for literal, other in ((node.left, node.right), (node.right, node.left)):
        if _numeric(literal) and not (isinstance(literal, ast.Constant) and literal.value in (0, 1)):
            if _names(other) & BEAT_NAMES:
                return True
    return False


Key = tuple[str, str]


def load_allowed(path: Path = ALLOWED) -> tuple[frozenset[Key], dict[Key, int]]:
    """Return the ``(path, name)`` pairs listed as invariants, and the pending ones with the count each admits.

    Raises:
        ValueError: the format is wrong, an entry lacks its reason or register entry, or a count is not positive.
    """
    raw = yaml.safe_load(path.read_text())
    if raw.get("format") != ALLOWED_FORMAT:
        raise ValueError(f"{path}: format is {raw.get('format')!r}, not {ALLOWED_FORMAT!r}")
    entries = {}
    for section, why in (("invariants", "reason"), ("pending", "register")):
        for entry in raw.get(section) or []:
            where = f"{path}: {section} {entry.get('path')}::{entry.get('name')}"
            if not str(entry.get(why, "")).strip():
                raise ValueError(f"{where} needs a {why}")
            count = entry.get("count", 1)
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                raise ValueError(f"{where}: count is not a positive integer")
            entries[section, entry["path"], entry["name"]] = count
    invariants = frozenset((p, n) for s, p, n in entries if s == "invariants")
    return invariants, {(p, n): c for (s, p, n), c in entries.items() if s == "pending"}


def problems(found: list[tuple[Site, bool]], invariants: frozenset[Key], pending: dict[Key, int]) -> list[str]:
    """One message per site that is neither a declared, listed invariant nor pending, and per unmatched entry.

    A pending entry whose number of sites differs from its count is reported.
    """
    counts: dict[Key, int] = {}
    for site, _ in found:
        counts[site.path, site.name] = counts.get((site.path, site.name), 0) + 1
    out = [
        f"{p}::{n} is listed in {ALLOWED.name} but not found; remove it"
        for p, n in sorted((invariants | set(pending)) - set(counts))
    ]
    out += [
        f"{p}::{n} has {counts[p, n]} sites; {ALLOWED.name} admits {c}"
        for (p, n), c in sorted(pending.items())
        if (p, n) in counts and counts[p, n] != c
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
