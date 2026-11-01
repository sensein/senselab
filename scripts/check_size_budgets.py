"""Fail when a Python file or function tracked under ``packages/`` is over its length budget.

Four limits, each with its derivation, are in ``scripts/data/size_budget.yaml``: lines per source file, lines per
test file, and lines per function in each. Every run prints the longest files and functions, so a change that
brings one near its limit is visible before it passes it.

Usage::

    uv run python scripts/check_size_budgets.py
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
BUDGET = Path(__file__).resolve().parent / "data" / "size_budget.yaml"
BUDGET_FORMAT = "size-budget@1.0"
BUDGETED = "packages/"
LIMITS = ("src_file_max_lines", "test_file_max_lines", "src_function_max_lines", "test_function_max_lines")
REPORTED = 5


class Budget(NamedTuple):
    """The four limits and the ``(path, function)`` pairs allowed over them."""

    limits: dict[str, int]
    exempt: frozenset[tuple[str, str]]


class Measure(NamedTuple):
    """One file or function: where it is, its name (``""`` for a file) and its length in lines."""

    path: str
    name: str
    lines: int


def load_budget(path: Path = BUDGET) -> Budget:
    """Return the budget file at ``path``; an entry without a value and a derivation is refused."""
    raw = yaml.safe_load(path.read_text())
    if raw.get("format") != BUDGET_FORMAT:
        raise ValueError(f"{path}: format is {raw.get('format')!r}, not {BUDGET_FORMAT!r}")
    limits = {}
    for name in LIMITS:
        entry = raw.get(name)
        if not isinstance(entry, dict) or not str(entry.get("derivation", "")).strip():
            raise ValueError(f"{path}: {name} needs a value and a derivation")
        if not isinstance(entry.get("value"), int) or isinstance(entry["value"], bool) or entry["value"] <= 0:
            raise ValueError(f"{path}: {name} is not a positive integer")
        limits[name] = entry["value"]
    exempt = set()
    for entry in raw.get("exempt") or []:
        if not str(entry.get("reason", "")).strip():
            raise ValueError(f"{path}: exempt {entry.get('path')}::{entry.get('function')} needs a reason")
        exempt.add((entry["path"], entry.get("function", "")))
    return Budget(limits, frozenset(exempt))


def tracked_python(root: Path = ROOT) -> list[str]:
    """Every Python file git tracks under ``packages/``."""
    out = subprocess.run(["git", "-C", str(root), "ls-files", "-z", BUDGETED], capture_output=True, check=True)
    return [p for p in out.stdout.decode().split("\0") if p.endswith(".py")]


def kind(path: str) -> str:
    """``test`` for a file under a ``tests`` directory or named ``conftest.py``, else ``src``."""
    parts = path.split("/")
    return "test" if "tests" in parts or parts[-1] == "conftest.py" else "src"


def measure(path: str, source: str) -> tuple[Measure, list[Measure]]:
    """The length of the file `source` and of each function in it, nested ones included."""
    file = Measure(path, "", len(source.splitlines()))
    functions = [
        Measure(path, node.name, (node.end_lineno or node.lineno) - node.lineno + 1)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    return file, functions


def violations(files: list[Measure], functions: list[Measure], budget: Budget) -> list[str]:
    """One message per file or function over its limit and not exempt."""
    out = []
    for m in files:
        limit = budget.limits[f"{kind(m.path)}_file_max_lines"]
        if m.lines > limit and (m.path, "") not in budget.exempt:
            out.append(f"{m.path}: {m.lines} lines exceeds the {limit} line file budget; split the module")
    for m in functions:
        limit = budget.limits[f"{kind(m.path)}_function_max_lines"]
        if m.lines > limit and (m.path, m.name) not in budget.exempt:
            out.append(f"{m.path}::{m.name}: {m.lines} lines exceeds the {limit} line function budget; split it")
    return out


def report(files: list[Measure], functions: list[Measure]) -> list[str]:
    """The longest files and functions of each kind, one line each."""
    lines = []
    for which, items in (("file", files), ("function", functions)):
        for k in ("src", "test"):
            top = sorted((m for m in items if kind(m.path) == k), key=lambda m: -m.lines)[:REPORTED]
            lines.append(
                f"longest {k} {which}s: "
                + ", ".join(f"{m.path.rsplit('/', 1)[-1]}::{m.name or '*'} {m.lines}" for m in top)
            )
    return lines


def main(root: Path = ROOT) -> int:
    """Run the check over the files tracked under ``packages/`` of ``root``; return the exit status."""
    budget = load_budget()
    files: list[Measure] = []
    functions: list[Measure] = []
    for path in tracked_python(root):
        file, found = measure(path, (root / path).read_text())
        files.append(file)
        functions += found
    for line in report(files, functions):
        print(line)
    found_violations = violations(files, functions, budget)
    for v in found_violations:
        print(v)
    return 1 if found_violations else 0


if __name__ == "__main__":
    sys.exit(main())
