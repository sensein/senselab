"""Fail when a tracked file carries a user home-directory path.

Flags ``/Users/<name>/``, ``/home/<name>/`` and ``C:\\Users\\<name>\\`` (or ``C:/Users/<name>/``) in
text file contents and in file paths, except for the generic accounts ``user``, ``runner`` and
``ec2-user``; and ``~/<name>/``, except for the tool and cluster conventions in ``_GENERIC_HOME_DIRS``.
Output names the file, line and kind of match, never the value. Keep real values in the untracked
``references.local.yaml``; tracked text uses the placeholder names listed in ``references.example.yaml``.

Usage::

    uv run python scripts/check_local_paths.py            # every tracked file
    uv run python scripts/check_local_paths.py FILE ...   # just these (pre-commit)
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LOCAL_VALUES_FILE = "references.local.yaml"
_GENERIC_ACCOUNTS = r"(?:user|runner|ec2-user)(?![\w-])"
_NOT_PART_OF_WORD = r"(?<![\w~.$}-])"
_GENERIC_HOME_DIRS = r"(?:\.cache|\.local|\.config|\.ssh|\.cargo|ffmpeg|orcd/scratch|orcd/pool)(?![\w.-])"

PATTERNS: dict[str, re.Pattern[str]] = {
    "user home path": re.compile(rf"{_NOT_PART_OF_WORD}(?<![A-Za-z]:)/(?:Users|home)/(?!{_GENERIC_ACCOUNTS})\w[\w.-]*"),
    "home-relative path": re.compile(rf"{_NOT_PART_OF_WORD}~/(?!{_GENERIC_HOME_DIRS})[\w.][\w.-]*/"),
    "Windows user home path": re.compile(
        rf"(?<![\w])[A-Za-z]:[\\/]+Users[\\/]+(?!{_GENERIC_ACCOUNTS})\w[\w.-]*", re.IGNORECASE
    ),
}


def tracked_files(root: Path = ROOT) -> list[str]:
    """Return every path git tracks under ``root``."""
    out = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True).stdout
    return [p for p in out.decode("utf-8").split("\0") if p]


def kinds(text: str) -> list[str]:
    """Return the kind of every match in ``text``, in pattern order."""
    return [kind for kind, pattern in PATTERNS.items() for _ in pattern.finditer(text)]


def findings(paths: list[str], root: Path = ROOT) -> list[str]:
    """Return one ``path:line: kind`` finding per occurrence, without the matched text."""
    found: list[str] = []
    for rel in paths:
        if rel == LOCAL_VALUES_FILE:
            continue
        found.extend(f"{rel}: {kind} in path" for kind in kinds(rel))
        path = root / rel
        if not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue
        text = data.decode("utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            found.extend(f"{rel}:{lineno}: {kind}" for kind in kinds(line))
    return found


def main(argv: list[str]) -> int:
    """Check ``argv`` paths, or every tracked file when none are given."""
    hits = findings(argv or tracked_files())
    for hit in hits:
        print(hit)
    if hits:
        print(
            f"{len(hits)} local path(s) found; use a placeholder from references.example.yaml "
            f"and keep the real value in {LOCAL_VALUES_FILE}."
        )
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
