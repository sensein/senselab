"""Fail when a ruling id, a finding id or a review-round name appears under ``packages/``.

Those ids name rows in ``specs/``; the code says what it does and the spec says why. The patterns are exact:

- rulings: ``H-R12``, ``R12``;
- register and finding ids: ``R-12``, ``F-12``, ``A-12``, ``XA-12`` (a letter after ``X``);
- round names: ``r12``, ``H9c`` (a hardening task, one digit), ``full-review``, ``reattack``, ``rereview``.

Output names the file, line and kind, never more. Principle and ADR references (``P-23``, ``ADR-002``) are
allowed: they point at the design, they do not record a review.

Usage::

    uv run python scripts/check_spec_ids.py            # every tracked file under packages/
    uv run python scripts/check_spec_ids.py FILE ...   # just these (pre-commit)
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKED = "packages/"
SUFFIXES = (".py", ".pyi", ".md", ".rst", ".yaml", ".yml", ".toml", ".txt", ".cfg", ".ini", ".json")

PATTERNS: dict[str, re.Pattern[str]] = {
    "ruling id": re.compile(r"\b(?:H-R\d+|R\d+)\b"),
    "register or finding id": re.compile(r"\b(?:R-\d+|F-\d+|A-\d+|X[A-Z]-\d+)\b"),
    "review-round name": re.compile(r"\b(?:r\d{2}|H[1-9][a-z]?)\b|(?i:full-review|re-?attack|re-?review)"),
}


def tracked_files(root: Path = ROOT) -> list[str]:
    """Every path git tracks under ``packages/`` of ``root`` that is text of a checked suffix."""
    out = subprocess.run(["git", "-C", str(root), "ls-files", "-z", CHECKED], capture_output=True, check=True)
    return [p for p in out.stdout.decode().split("\0") if p.endswith(SUFFIXES)]


def kinds(line: str) -> list[str]:
    """The kinds of id `line` carries, each once."""
    return [kind for kind, pattern in PATTERNS.items() if pattern.search(line)]


def findings(paths: list[str], root: Path = ROOT) -> list[str]:
    """One ``path:line: kind`` finding per kind per line, over the files in `paths` under ``packages/``."""
    found = []
    for rel in paths:
        path = root / rel
        if not rel.startswith(CHECKED) or not rel.endswith(SUFFIXES) or not path.is_file():
            continue
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
            found += [f"{rel}:{number}: {kind}; the spec holds it, say what the code does" for kind in kinds(line)]
    return found


def main(argv: list[str], root: Path = ROOT) -> int:
    """Check the files named in `argv`, or every tracked file under ``packages/``; return the exit status."""
    found = findings(argv or tracked_files(root), root)
    for line in found:
        print(line)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
