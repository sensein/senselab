"""Which refs and pushes the limits tier of 2.x CI covers, derived from the tree and from the remote's branches.

Standard library only, so a workflow runs it before any environment exists.

- ``scope``: print ``run=true`` or ``run=false`` for the push or pull request in the environment (``GITHUB_EVENT_NAME``,
  ``GITHUB_REF_NAME``, ``BEFORE``). The limits job runs on a pull request, on every push to the phase branch, and on a
  push that changes a file in the limits scope at ``BEFORE`` or at HEAD: a limits-marked test, every in-repo module it
  imports directly or transitively (source and test-side helpers), a ``conftest.py`` or the test data.
- ``phase-branch``: print the phase branch, the highest ``2x/phase-<N>`` head on ``origin``.
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
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
TESTS = "packages/senselab-core/tests"
THIS_SCRIPT = "scripts/limits_scope.py"
PHASE_HEAD = re.compile(r"\srefs/heads/(2x/phase-(\d+))\s*$")


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return None


def _is_limits_mark(node: ast.AST) -> bool:
    """Whether `node` is ``pytest.mark.limits`` or ``mark.limits``, however it is applied."""
    if not (isinstance(node, ast.Attribute) and node.attr == "limits"):
        return False
    inner = node.value
    if isinstance(inner, ast.Name):
        return inner.id == "mark"
    return isinstance(inner, ast.Attribute) and inner.attr == "mark"


def has_limits_mark(path: Path) -> bool:
    """Whether the file applies the ``limits`` marker (decorator, ``pytestmark``, call); a file that does not parse counts."""
    tree = _parse(path)
    return tree is None or any(_is_limits_mark(node) for node in ast.walk(tree))


def limits_tests(root: Path = ROOT) -> list[str]:
    """The test files under the core tests that carry the ``limits`` marker, relative to `root`, sorted."""
    return sorted(_rel(path, root) for path in (root / TESTS).rglob("*_test.py") if has_limits_mark(path))


def _names(tree: ast.Module) -> list[tuple[int, list[str]]]:
    """``(relative level, dotted parts)`` for every module an import in `tree` could name, ``from a import b`` as both."""
    found: list[tuple[int, list[str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(0, alias.name.split(".")) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module.split(".") if node.module else []
            found.append((node.level, base))
            found += [(node.level, [*base, alias.name]) for alias in node.names]
    return found


def _module_files(parts: list[str], base: Path) -> list[Path]:
    """The files a dotted name resolves to under `base`, with each package it imports through."""
    out = []
    for end in range(1, len(parts) + 1):
        for candidate in (base.joinpath(*parts[:end]).with_suffix(".py"), base.joinpath(*parts[:end], "__init__.py")):
            if candidate.is_file():
                out.append(candidate)
    return out


def _roots(file: Path, root: Path) -> list[Path]:
    """Where a top-level import in `file` can resolve: every package's ``src``, and the file's directory and each parent
    up to the core tests root, which pytest puts on ``sys.path`` for a test directory without an ``__init__``."""
    roots = list((root / "packages").glob("*/src"))
    tests = root / TESTS
    if tests in file.parents:
        roots += [d for d in (file.parent, *file.parent.parents) if d == tests or tests in d.parents]
    return roots


def subjects(test: Path, root: Path = ROOT) -> set[str]:
    """The in-repo files `test` depends on through imports, transitively, relative to `root`.

    A dotted name resolves under ``packages/*/src`` and under the importing file's test directories, with the packages
    it imports through; a relative import resolves from the importing file.
    """
    seen: set[Path] = set()
    todo = [test]
    while todo:
        file = todo.pop()
        tree = _parse(file)
        if tree is None:
            continue
        for level, parts in _names(tree):
            if level:
                bases = [file.parent.parents[level - 2] if level > 1 else file.parent]
            else:
                bases = _roots(file, root)
            for base in bases:
                for found in _module_files(parts, base):
                    if found not in seen and root in found.parents and found != test:
                        seen.add(found)
                        todo.append(found)
    return {_rel(path, root) for path in seen}


def scope(root: Path = ROOT) -> frozenset[str]:
    """The files whose change makes the limits tier run: the marked tests, their dependencies, conftests, this script."""
    paths = {THIS_SCRIPT}
    for test in limits_tests(root):
        paths.add(test)
        paths |= subjects(root / test, root)
    paths.update(_rel(path, root) for path in (root / TESTS).rglob("conftest.py"))
    return frozenset(paths)


def scope_at(rev: str, root: Path) -> frozenset[str]:
    """The scope of the tree at `rev`, empty when the clone lacks `rev`; only ``packages`` Python files are unpacked."""
    done = subprocess.run(
        ["git", "archive", "--format=tar", rev, "--", "packages"], cwd=root, capture_output=True, timeout=300, check=False
    )
    if done.returncode != 0:
        return frozenset()
    with tempfile.TemporaryDirectory() as tmp, tarfile.open(fileobj=io.BytesIO(done.stdout)) as tar:
        for member in tar:
            parts = PurePosixPath(member.name).parts
            if member.isfile() and member.name.endswith(".py") and ".." not in parts and not member.name.startswith("/"):
                target = Path(tmp, *parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                assert source is not None
                target.write_bytes(source.read())
        return scope(Path(tmp))


def affected(changed: list[str], covered: frozenset[str]) -> bool:
    """Whether any of the `changed` paths is in `covered` or under the core tests' data directory."""
    return any(path in covered or path.startswith(f"{TESTS}/data/") for path in changed)


def phase_branch_from(ls_remote: str) -> str:
    """The highest ``2x/phase-<N>`` head in the output of ``git ls-remote --heads``; ``-spec`` branches do not count."""
    heads = [(int(m.group(2)), m.group(1)) for line in ls_remote.splitlines() if (m := PHASE_HEAD.search(line))]
    if not heads:
        raise ValueError("the remote has no 2x/phase-<N> branch")
    return max(heads)[1]


def phase_branch(root: Path = ROOT) -> str:
    """The phase branch: the highest ``2x/phase-<N>`` head on ``origin``."""
    done = subprocess.run(
        ["git", "ls-remote", "--heads", "origin", "2x/phase-*"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    return phase_branch_from(done.stdout)


def changed_since(before: str, root: Path) -> list[str] | None:
    """The paths changed between `before` and HEAD, renames as a delete and an add, or None when `before` is unknown."""
    done = subprocess.run(
        ["git", "diff", "--name-only", "--no-renames", before, "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return done.stdout.split() if done.returncode == 0 else None


def runs(event: str, ref_name: str, before: str, root: Path = ROOT, phase: str | None = None) -> bool:
    """Whether the limits job runs: a pull request or the phase branch always, else on a change in scope at either end.

    A base the clone lacks (a new branch, a force push) counts as a change. The scope at `before` is included so a
    deleted or renamed dependency, or a removed mark, is seen.
    """
    if event != "push" or ref_name == (phase or phase_branch(root)):
        return True
    changed = changed_since(before, root)
    return changed is None or affected(changed, scope(root) | scope_at(before, root))


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
