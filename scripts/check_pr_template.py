"""Fail when the per-commit checklist of the 2.x PR template differs from the one in engineering.md.

``--write`` regenerates the template's checklist section from engineering.md.

Usage::

    uv run python scripts/check_pr_template.py [--write]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / ".github" / "PULL_REQUEST_TEMPLATE" / "2x.md"
STANDARDS = ROOT / "specs" / "20261030-140022-senselab-2x-api" / "engineering.md"
HEADING = "## Per-commit checklist"
ITEMS = re.compile(r"^- \[[ xX]\] .+(?:\n  .+)*", re.MULTILINE)


def section(text: str, heading: str = HEADING) -> str:
    """The body of `heading` in `text`, up to the next heading; empty when `heading` is missing."""
    if heading not in text:
        return ""
    return re.split(r"\n#{2,3} ", text.split(heading, 1)[1], maxsplit=1)[0]


def items(text: str) -> list[str]:
    """The checklist items of `text`, each with its continuation lines joined, without the tick."""
    return [" ".join(found[6:].split()) for found in ITEMS.findall(text)]


def checklist(standards: Path = STANDARDS) -> list[str]:
    """The per-commit checklist items of engineering.md."""
    return items(section(standards.read_text()))


def problems(template: str, standards: str) -> list[str]:
    """Why the template's checklist differs from the standards': each missing, extra or reordered item."""
    want, have = items(section(standards)), items(section(template))
    if not want:
        return [f"engineering.md has no checklist under {HEADING!r}"]
    if have == want:
        return []
    out = [f"missing: {item}" for item in want if item not in have]
    out += [f"not in engineering.md: {item}" for item in have if item not in want]
    return out or ["the items are in a different order; run scripts/check_pr_template.py --write"]


def regenerate(template: str, standards: str) -> str:
    """`template` with its checklist items replaced by the standards' items, verbatim."""
    body = section(template)
    block = "\n".join(ITEMS.findall(section(standards)))
    old = "\n".join(ITEMS.findall(body))
    return template.replace(body, body.replace(old, block, 1), 1) if old else template


def main(argv: list[str]) -> int:
    """Check the template, or with ``--write`` regenerate it; print the problems; return 1 if any remain."""
    if argv == ["--write"]:
        TEMPLATE.write_text(regenerate(TEMPLATE.read_text(), STANDARDS.read_text()))
    found = problems(TEMPLATE.read_text(), STANDARDS.read_text())
    for line in found:
        print(line)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
