"""Fail when a tracked file carries a research-participant identifier.

Flags BIDS ``sub-``/``ses-`` entities whose label is a full UUID or an 8-hex prefix, in file
contents and in file paths. Output names the file, line and kind of match, never the value.
Synthetic labels (``sub-A``, ``sub-SUBJ01``, ``ses-01``) do not match.

Usage::

    uv run python scripts/check_participant_ids.py            # every tracked file
    uv run python scripts/check_participant_ids.py FILE ...   # just these (pre-commit)
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
PARTICIPANT_ID_RE = re.compile(
    rf"(?<![A-Za-z0-9])(?P<kind>sub|ses)-(?:(?P<uuid>{_UUID})|(?P<hex8>[0-9a-f]{{8}})(?![0-9a-z]))",
    re.IGNORECASE,
)


def tracked_files(root: Path = ROOT) -> list[str]:
    """Return every path git tracks under ``root``."""
    out = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True).stdout
    return [p for p in out.decode("utf-8").split("\0") if p]


def mask(text: str) -> str:
    """Replace every participant identifier in ``text`` with ``<kind>-<ID>``."""
    return PARTICIPANT_ID_RE.sub(lambda m: f"{m.group('kind')}-<ID>", text)


def findings(paths: list[str], root: Path = ROOT) -> list[str]:
    """Return one masked ``path:line: kind`` finding per identifier occurrence."""
    found: list[str] = []
    for rel in paths:
        if PARTICIPANT_ID_RE.search(rel):
            found.append(f"{mask(rel)}: identifier in path")
        path = root / rel
        if not path.is_file():
            continue
        text = path.read_bytes().decode("utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for m in PARTICIPANT_ID_RE.finditer(line):
                shape = "full UUID" if m.group("uuid") else "8-hex prefix"
                found.append(f"{mask(rel)}:{lineno}: {m.group('kind')}- {shape}")
    return found


def main(argv: list[str]) -> int:
    """Check ``argv`` paths, or every tracked file when none are given."""
    hits = findings(argv or tracked_files())
    for hit in hits:
        print(hit)
    if hits:
        print(f"{len(hits)} participant identifier(s) found; replace with placeholders such as sub-SUBJ01.")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
