"""Fail when committed test fixtures exceed the data-policy budget.

Scans tracked files under the fixture locations. Allowlisted (grandfathered) files do not count toward
the budgets, but may not grow. Usage::

    uv run python scripts/check_fixture_budget.py
"""

from __future__ import annotations

import fnmatch
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = Path(__file__).resolve().parent / "fixture_budget_allowlist.txt"
FIXTURE_PATTERNS = ("src/tests/data_for_testing/*", "packages/*/tests/*/data/*", "packages/*/tests/data/*")
PER_FILE = 256 * 1024
TOTAL = 2 * 1024 * 1024
_FIX = "generate it with senselab_core.testing.synth, or pull it through a data manifest"


def check(
    paths: Sequence[Path],
    *,
    per_file: int = PER_FILE,
    total: int = TOTAL,
    allow: set[str],
    root: Path = ROOT,
    sizes: dict[str, int] | None = None,
) -> list[str]:
    """Return one violation message per problem among `paths`, naming the file, its size and the limit.

    `allow` holds repo-relative paths (relative to `root`) that are grandfathered; `sizes` optionally
    maps an allowlisted path to its recorded maximum size.
    """
    out: list[str] = []
    counted = 0
    for path in paths:
        try:
            rel = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            rel = path.as_posix()
        size = path.stat().st_size
        if rel in allow:
            cap = (sizes or {}).get(rel)
            if cap is not None and size > cap:
                out.append(f"{rel}: {size} bytes, grew past its grandfathered {cap} bytes; {_FIX}")
            continue
        counted += size
        if size > per_file:
            out.append(f"{rel}: {size} bytes exceeds the {per_file} byte per-file limit; {_FIX}")
    if counted > total:
        out.append(f"total of non-allowlisted fixtures: {counted} bytes exceeds the {total} byte limit; {_FIX}")
    return out


def _tracked(root: Path) -> list[str]:
    result = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True)
    return [f for f in result.stdout.splitlines() if any(fnmatch.fnmatch(f, p) for p in FIXTURE_PATTERNS)]


def _allowlist(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")}


def main() -> int:
    """Run the check over the tracked fixtures; return the exit status."""
    allow = _allowlist(ALLOWLIST)
    files = [ROOT / f for f in _tracked(ROOT) if (ROOT / f).is_file()]
    sizes = {rel: (ROOT / rel).stat().st_size for rel in allow if (ROOT / rel).is_file()}
    # Growth is judged against the size recorded in git HEAD, so the allowlist stays a plain path list.
    for rel in list(sizes):
        head = subprocess.run(["git", "cat-file", "-s", f"HEAD:{rel}"], cwd=ROOT, capture_output=True, text=True)
        if head.returncode == 0:
            sizes[rel] = int(head.stdout)
    violations = check(files, allow=allow, root=ROOT, sizes=sizes)
    for v in violations:
        print(v)
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
