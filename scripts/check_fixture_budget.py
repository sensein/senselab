"""Fail when committed test fixtures exceed the data-policy budget.

Scans every tracked non-code file under `src/tests/` and `packages/*/tests/`. Allowlisted
(grandfathered) files are listed as `path bytes` in `fixture_budget_allowlist.txt`; they do not count
toward the budgets, and may not grow past the recorded size. Usage::

    uv run python scripts/check_fixture_budget.py
"""

from __future__ import annotations

import fnmatch
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = Path(__file__).resolve().parent / "fixture_budget_allowlist.txt"
TEST_TREES = ("src/tests/*", "packages/*/tests/*")
SOURCE_SUFFIXES = (".py", ".pyi", ".mjs")
"""Code suffixes present under the test trees; every other tracked file, data formats included, is scanned."""
PER_FILE = 256 * 1024
TOTAL = 2 * 1024 * 1024
_FIX = "generate it with senselab_core.testing.synth, or pull it through a data manifest"


def is_fixture(rel: str) -> bool:
    """Whether the repo-relative path is a tracked non-source file under a test tree."""
    return any(fnmatch.fnmatch(rel, p) for p in TEST_TREES) and not rel.endswith(SOURCE_SUFFIXES)


def parse_allowlist(text: str) -> dict[str, int]:
    """Parse `path bytes` lines (blank lines and `#` comments skipped) into `{path: recorded size}`.

    Raises:
        ValueError: naming the line, if a line has no size or a malformed one.
    """
    out: dict[str, int] = {}
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.rsplit(None, 1)
        if len(parts) != 2 or not parts[1].isdigit():
            raise ValueError(f"allowlist line {n}: expected 'path bytes', got {line!r}")
        out[parts[0].strip()] = int(parts[1])
    return out


def check(
    paths: Sequence[Path],
    *,
    per_file: int = PER_FILE,
    total: int = TOTAL,
    allow: Mapping[str, int],
    root: Path = ROOT,
) -> list[str]:
    """Return one violation message per problem among `paths`, naming the file, its size and the limit.

    `allow` maps each grandfathered repo-relative path (relative to `root`) to its recorded size.
    """
    out: list[str] = []
    counted = 0
    for path in paths:
        rel = path.resolve().relative_to(root.resolve()).as_posix()
        size = path.stat().st_size
        if rel in allow:
            if size > allow[rel]:
                out.append(f"{rel}: {size} bytes, grew past its grandfathered {allow[rel]} bytes; {_FIX}")
            continue
        counted += size
        if size > per_file:
            out.append(f"{rel}: {size} bytes exceeds the {per_file} byte per-file limit; {_FIX}")
    if counted > total:
        out.append(f"total of non-allowlisted fixtures: {counted} bytes exceeds the {total} byte limit; {_FIX}")
    return out


def _tracked(root: Path) -> list[str]:
    result = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True)
    return [f for f in result.stdout.splitlines() if is_fixture(f)]


def main(root: Path = ROOT, allowlist: Path = ALLOWLIST) -> int:
    """Run the check over the tracked fixtures; return the exit status."""
    try:
        allow = parse_allowlist(allowlist.read_text())
    except ValueError as e:
        print(e)
        return 1
    files = [root / f for f in _tracked(root) if (root / f).is_file()]
    violations = check(files, allow=allow, root=root)
    for v in violations:
        print(v)
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
