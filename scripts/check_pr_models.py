"""Fail when a pull request body lacks its models or a per-commit checklist line, or pairs a model with itself.

The 2.x template has the lines ``- Implementer (model, effort):``, ``- Reviewer (model, effort; ...):`` and
``- Kernel item (...):``; a model is the text before the first comma, compared without case. The reviewer's model
may equal the implementer's only when both are the top tier (``TOP_TIER``) and the kernel line says yes. Every item
of engineering.md's per-commit checklist must appear in the body, ticked or not.

Usage::

    PR_BODY="$body" uv run python scripts/check_pr_models.py
"""

from __future__ import annotations

import os
import re
import sys

from check_pr_template import checklist, items

LINE = r"^- {role}\b[^:\n]*:[ \t]*([^\n]*)$"
TOP_TIER = ("opus", "fable")


def _field(text: str, role: str) -> str:
    found = re.search(LINE.format(role=role), text, re.MULTILINE)
    return found.group(1).strip() if found else ""


def models(body: str) -> tuple[str, str]:
    """The implementer and reviewer models named in `body`, each empty when its line is missing or blank."""
    text = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL)
    implementer, reviewer = (_field(text, role).split(",")[0].strip().lower() for role in ("Implementer", "Reviewer"))
    return implementer, reviewer


def kernel(body: str) -> bool:
    """Whether `body` declares a kernel item."""
    return _field(re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL), "Kernel item").lower().startswith("yes")


def problems(body: str, required: list[str]) -> list[str]:
    """Why `body` fails: a missing model, a model reviewing itself outside the rule, or a missing checklist line."""
    implementer, reviewer = models(body)
    out = []
    if not implementer or not reviewer:
        out.append("name the implementer and the reviewer model in the Models section")
    elif implementer == reviewer and not (implementer.startswith(TOP_TIER) and kernel(body)):
        out.append(
            f"the implementer and the reviewer are both {implementer!r}; the reviewer must be a different model, "
            f"or both {' or '.join(TOP_TIER)} on a kernel item that the body declares"
        )
    present = set(items(body))
    out += [f"missing checklist line: {item}" for item in required if item not in present]
    return out


def main() -> int:
    """Check the body in ``PR_BODY`` against engineering.md's checklist; print the problems; return 1 if any."""
    found = problems(os.environ.get("PR_BODY", ""), checklist())
    for line in found:
        print(line)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
