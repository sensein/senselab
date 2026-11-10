"""Which refs and pushes the limits tier of 2.x CI covers, derived from the tree and from STATE.md.

Standard library only, so a workflow runs it before any environment exists.

- ``scope``: print ``run=true`` or ``run=false`` for the push or pull request in the environment (``GITHUB_EVENT_NAME``,
  ``GITHUB_REF_NAME``, ``BEFORE``). The limits job runs on a pull request, on every push to the phase branch, and on a
  push that changes a limits-marked test, a module such a test imports, a ``conftest.py`` or the test data.
- ``phase-branch``: print the code branch of the current phase, the first branch named in STATE.md's ``Code (current)``
  row.
- ``refs INTEGRATION``: print the JSON list of refs the nightly limits matrix runs, the integration branch and the
  phase branch.

Usage::

    python3 scripts/limits_scope.py scope [--root DIR]
    python3 scripts/limits_scope.py phase-branch [--root DIR]
    python3 scripts/limits_scope.py refs 2.x [--root DIR]
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = "packages/senselab-core/tests"
STATE = "specs/20261030-140022-senselab-2x-api/STATE.md"
MARK = re.compile(r"\bpytest\.mark\.limits\b")
CODE_ROW = re.compile(r"^\|\s*Code \(current\)\s*\|\s*`([^`]+)`", re.MULTILINE)
THIS_SCRIPT = "scripts/limits_scope.py"


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def limits_tests(root: Path = ROOT) -> list[str]:
    """The test files under the core tests that carry the ``limits`` marker, relative to `root`, sorted."""
    return sorted(
        _rel(path, root) for path in (root / TESTS).rglob("*_test.py") if MARK.search(path.read_text(encoding="utf-8"))
    )


def _imports(source: str) -> set[str]:
    """Every dotted module name an absolute import in `source` could name, including ``from a import b`` as ``a.b``."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module)
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return found


def subjects(test: Path, root: Path = ROOT) -> set[str]:
    """The source files of the packages that `test` imports, relative to `root`.

    A dotted name ``a.b`` resolves, with each package it imports through (``a``), to ``<package>/src/<a>/<b>.py`` or
    ``<package>/src/<a>/<b>/__init__.py`` in any package.
    """
    out: set[str] = set()
    for name in _imports(test.read_text(encoding="utf-8")):
        parts = name.split(".")
        for src in (root / "packages").glob("*/src"):
            for end in range(1, len(parts) + 1):
                for candidate in (
                    src.joinpath(*parts[:end]).with_suffix(".py"),
                    src.joinpath(*parts[:end], "__init__.py"),
                ):
                    if candidate.is_file():
                        out.add(_rel(candidate, root))
    return out


def scope(root: Path = ROOT) -> frozenset[str]:
    """The files whose change makes the limits tier run: the marked tests, their subjects, conftests and this script."""
    paths = {THIS_SCRIPT}
    for test in limits_tests(root):
        paths.add(test)
        paths |= subjects(root / test, root)
    paths.update(_rel(path, root) for path in (root / TESTS).rglob("conftest.py"))
    return frozenset(paths)


def affected(changed: list[str], root: Path = ROOT) -> bool:
    """Whether any of the `changed` paths is in the scope or under the core tests' data directory."""
    covered = scope(root)
    return any(path in covered or path.startswith(f"{TESTS}/data/") for path in changed)


def phase_branch(root: Path = ROOT) -> str:
    """The code branch of the current phase, from STATE.md; raises when the row is missing."""
    found = CODE_ROW.search((root / STATE).read_text(encoding="utf-8"))
    if not found:
        raise ValueError(f"{STATE} has no `Code (current)` row naming a branch")
    return found.group(1)


def changed_since(before: str, root: Path) -> list[str] | None:
    """The paths changed between `before` and HEAD, or None when `before` is not a commit the clone has."""
    done = subprocess.run(
        ["git", "diff", "--name-only", before, "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return done.stdout.split() if done.returncode == 0 else None


def runs(event: str, ref_name: str, before: str, root: Path = ROOT) -> bool:
    """Whether the limits job runs for this event: a pull request or the phase branch always, else on a change in scope.

    A base the clone lacks (a new branch, a force push) counts as a change.
    """
    if event != "push" or ref_name == phase_branch(root):
        return True
    changed = changed_since(before, root)
    return changed is None or affected(changed, root)


def main(argv: list[str] | None = None) -> int:
    """Run one subcommand and print its answer."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["scope", "phase-branch", "refs"])
    parser.add_argument("integration", nargs="?", help="the integration branch, for `refs`")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    if args.command == "scope":
        env = os.environ
        print(f"run={str(runs(env['GITHUB_EVENT_NAME'], env['GITHUB_REF_NAME'], env['BEFORE'], args.root)).lower()}")
    elif args.command == "phase-branch":
        print(phase_branch(args.root))
    else:
        if not args.integration:
            parser.error("`refs` takes the integration branch")
        print(json.dumps([args.integration, phase_branch(args.root)]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
