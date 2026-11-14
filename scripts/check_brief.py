"""Refuse an agent brief that lacks a required heading or whose checklist differs from engineering.md's.

A brief has ``## Principles``, ``## Checklist`` and ``## Files``; its checklist items are engineering.md's per-commit
checklist, verbatim. ``--render`` fills a template under ``scripts/data/briefs/`` and prints the brief, refusing a
placeholder left without a value.

Usage::

    uv run python scripts/check_brief.py BRIEF ...
    uv run python scripts/check_brief.py --render scripts/data/briefs/implementer.md TASK_ID=enf SPEC_DIR=... > brief.md
"""

from __future__ import annotations

import string
import sys
from pathlib import Path

from check_pr_template import checklist, items, section

HEADINGS = ("## Principles", "## Checklist", "## Files")


def problems(text: str, required: list[str]) -> list[str]:
    """Why the brief `text` is refused: each missing heading, and each checklist item missing or extra."""
    lines = text.splitlines()
    out = [f"missing heading {h!r}" for h in HEADINGS if h not in lines]
    if "## Checklist" in lines:
        have = items(section(text, "## Checklist"))
        out += [f"checklist lacks: {item}" for item in required if item not in have]
        out += [f"checklist item not in engineering.md: {item}" for item in have if item not in required]
    return out


def render(template: str, values: dict[str, str]) -> str:
    """`template` with each ``${NAME}`` replaced by its value.

    Raises:
        KeyError: a placeholder has no value.
    """
    return string.Template(template).substitute(values)


def main(argv: list[str]) -> int:
    """Check each brief in `argv`, or render one template; print the problems; return 1 if any."""
    required = checklist()
    if argv[:1] == ["--render"]:
        text = render(Path(argv[1]).read_text(), dict(pair.split("=", 1) for pair in argv[2:]))
        found = problems(text, required)
        if not found:
            print(text, end="")
    else:
        found = [f"{path}: {p}" for path in argv for p in problems(Path(path).read_text(), required)]
    for line in found:
        print(line, file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
