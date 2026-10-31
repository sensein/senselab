"""Fail when tracked files exceed the data-policy budget, bind the allowlist to the 1.x freeze, refuse symlinks.

Every `git ls-files` path is scanned:

- a non-code file over `PER_FILE` bytes, or a code file (`.py`, `.pyi`, `.ipynb`) over `code_file_max_bytes`
  from `data/fixture_budget.yaml`, fails unless it is on the allowlist;
- the files under the test trees that are not allowlisted may total at most `TOTAL` bytes;
- a tracked symlink (mode 120000) fails, without exception;
- the allowlist lines are `<path>\t<sha256>\t<bytes>`. Each path must exist in the freeze branch with the same
  content hash, the recorded size may not exceed the frozen size, and the current file must still hash to the
  recorded value, so an entry can neither be added after the freeze nor be replaced by other bytes.

Usage::

    uv run python scripts/check_fixture_budget.py
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = Path(__file__).resolve().parent / "fixture_budget_allowlist.txt"
POLICY = Path(__file__).resolve().parent / "data" / "fixture_budget.yaml"
FREEZE_BRANCH = "freeze-1x-2026-10-30"
TEST_TREES = ("src/tests/*", "packages/*/tests/*")
SOURCE_SUFFIXES = (".py", ".pyi", ".mjs")
"""Code suffixes present under the test trees; every other tracked file, data formats included, is scanned."""
CODE_SUFFIXES = (".py", ".pyi", ".ipynb")
SYMLINK_MODE = "120000"
PER_FILE = 256 * 1024
TOTAL = 2 * 1024 * 1024
_FIX = "generate it with senselab_core.testing.synth, or pull it through a data manifest"
_SHA = re.compile(r"[0-9a-f]{64}")


class Entry(NamedTuple):
    """One allowlist line: the content hash and byte size recorded for a grandfathered path."""

    sha256: str
    size: int


class Policy(NamedTuple):
    """The byte policy read from `data/fixture_budget.yaml`."""

    code_file_max_bytes: int
    generated_files: tuple[str, ...]


def load_policy(path: Path = POLICY) -> Policy:
    """Read the policy file.

    Raises:
        ValueError: if `code_file_max_bytes` is not a positive integer.
    """
    data = yaml.safe_load(path.read_text())
    cap = data.get("code_file_max_bytes")
    if not isinstance(cap, int) or isinstance(cap, bool) or cap <= 0:
        raise ValueError(f"{path}: code_file_max_bytes must be a positive integer, got {cap!r}")
    return Policy(cap, tuple(data.get("generated_files", ())))


def is_fixture(rel: str) -> bool:
    """Whether the repo-relative path is a tracked non-source file under a test tree."""
    return any(fnmatch.fnmatch(rel, p) for p in TEST_TREES) and not rel.endswith(SOURCE_SUFFIXES)


def parse_allowlist(text: str) -> dict[str, Entry]:
    """Parse `path<TAB>sha256<TAB>bytes` lines (blank lines and `#` comments skipped).

    Raises:
        ValueError: naming the line, if a line is not three tab-separated columns with a 64-hex hash and a size.
    """
    out: dict[str, Entry] = {}
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 3 or not _SHA.fullmatch(parts[1]) or not parts[2].isdigit():
            raise ValueError(f"allowlist line {n}: expected '<path>\\t<sha256>\\t<bytes>', got {line!r}")
        out[parts[0]] = Entry(parts[1], int(parts[2]))
    return out


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, check=True).stdout


def resolve_freeze_ref(root: Path = ROOT) -> str:
    """The ref of the freeze branch: its remote-tracking ref, and the local branch only when that is absent.

    Raises:
        LookupError: if neither exists, naming the fetch that creates it.
    """
    for ref in (f"origin/{FREEZE_BRANCH}", FREEZE_BRANCH):
        probe = subprocess.run(
            ["git", "rev-parse", "--verify", "-q", f"{ref}^{{commit}}"], cwd=root, capture_output=True
        )
        if probe.returncode == 0:
            return ref
    raise LookupError(
        f"the freeze branch {FREEZE_BRANCH} is not available; run: "
        f"git fetch origin {FREEZE_BRANCH}:refs/remotes/origin/{FREEZE_BRANCH}"
    )


def frozen_tree(root: Path, ref: str) -> dict[str, tuple[str, int]]:
    """`{path: (blob id, size)}` of every file in `ref`."""
    out: dict[str, tuple[str, int]] = {}
    for line in _git(root, "ls-tree", "-r", "-l", "-z", ref).decode().split("\0"):
        if line:
            meta, _, path = line.partition("\t")
            _mode, _kind, blob, size = meta.split()
            out[path] = (blob, int(size) if size.isdigit() else 0)
    return out


def check_allowlist(allow: Mapping[str, Entry], root: Path, ref: str) -> list[str]:
    """One message per allowlist entry that is not in the freeze with its hash, is larger, or no longer matches."""
    frozen = frozen_tree(root, ref)
    out: list[str] = []
    for path, entry in sorted(allow.items()):
        if path not in frozen:
            out.append(f"{path}: not in {ref}; the allowlist may only grandfather files that existed at the freeze")
            continue
        blob, size = frozen[path]
        if hashlib.sha256(_git(root, "cat-file", "blob", blob)).hexdigest() != entry.sha256:
            out.append(f"{path}: recorded sha256 is not the content of the file in {ref}")
        elif entry.size > size:
            out.append(f"{path}: recorded {entry.size} bytes is larger than the {size} bytes in {ref}")
        current = root / path
        if current.is_file() and not current.is_symlink():
            data = current.read_bytes()
            if len(data) > entry.size:
                out.append(f"{path}: {len(data)} bytes, grew past its grandfathered {entry.size} bytes; {_FIX}")
            elif hashlib.sha256(data).hexdigest() != entry.sha256:
                out.append(f"{path}: content differs from the grandfathered sha256; {_FIX}")
    return out


def check(
    paths: Sequence[Path],
    *,
    per_file: int = PER_FILE,
    total: int = TOTAL,
    allow: Mapping[str, int],
    root: Path = ROOT,
    code_max: int | None = None,
    generated: Sequence[str] = (),
) -> list[str]:
    """Return one violation message per problem among `paths`, naming the file, its size and the limit.

    `allow` maps each grandfathered repo-relative path (relative to `root`) to its recorded size. A symlink
    among `paths` is a violation. `code_max` caps code files
    (`CODE_SUFFIXES`); `generated` are globs exempt from the per-file limit.
    """
    out: list[str] = []
    counted = 0
    for path in paths:
        rel = Path(os.path.relpath(os.path.abspath(path), os.path.abspath(root))).as_posix()
        if path.is_symlink():
            out.append(f"{rel}: tracked symlink; commit the file itself")
            continue
        size = path.stat().st_size
        if rel in allow:
            if size > allow[rel]:
                out.append(f"{rel}: {size} bytes, grew past its grandfathered {allow[rel]} bytes; {_FIX}")
            continue
        if is_fixture(rel):
            counted += size
        if rel.endswith(CODE_SUFFIXES):
            if code_max is not None and size > code_max:
                out.append(
                    f"{rel}: {size} bytes exceeds the {code_max} byte code-file limit; code is not a data carrier"
                )
        elif size > per_file and not any(fnmatch.fnmatch(rel, g) for g in generated):
            out.append(f"{rel}: {size} bytes exceeds the {per_file} byte per-file limit; {_FIX}")
    if counted > total:
        out.append(f"total of non-allowlisted fixtures: {counted} bytes exceeds the {total} byte limit; {_FIX}")
    return out


def _tracked(root: Path) -> list[tuple[str, str]]:
    """`(mode, path)` of every tracked path."""
    result = _git(root, "ls-files", "-s", "-z").decode()
    out = []
    for line in result.split("\0"):
        if line:
            meta, _, path = line.partition("\t")
            out.append((meta.split()[0], path))
    return out


def main(root: Path = ROOT, allowlist: Path = ALLOWLIST, policy: Path = POLICY) -> int:
    """Run the check over every tracked path; return the exit status."""
    try:
        allow = parse_allowlist(allowlist.read_text())
        limits = load_policy(policy)
        ref = resolve_freeze_ref(root)
    except (ValueError, LookupError) as e:
        print(e)
        return 1
    violations = check_allowlist(allow, root, ref)
    files = [root / p for _mode, p in _tracked(root) if (root / p).is_symlink() or (root / p).is_file()]
    violations += check(
        files,
        allow={p: e.size for p, e in allow.items()},
        root=root,
        code_max=limits.code_file_max_bytes,
        generated=limits.generated_files,
    )
    for v in violations:
        print(v)
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
