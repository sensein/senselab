"""Fail when a pull request body does not name an implementer model and a different reviewer model.

The 2.x template has the two lines ``- Implementer (model, effort):`` and ``- Reviewer (model, effort; ...):``; the
model is the text before the first comma, compared without case.

Usage::

    PR_BODY="$body" uv run python scripts/check_pr_models.py
"""

from __future__ import annotations

import os
import re
import sys

LINE = r"^- {role}\b[^:\n]*:[ \t]*([^\n]*)$"


def models(body: str) -> tuple[str, str]:
    """The implementer and reviewer models named in `body`, each empty when its line is missing or blank."""
    text = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL)
    out = []
    for role in ("Implementer", "Reviewer"):
        found = re.search(LINE.format(role=role), text, re.MULTILINE)
        out.append(found.group(1).split(",")[0].strip().lower() if found else "")
    return out[0], out[1]


def problems(body: str) -> list[str]:
    """Why `body` fails: a missing model, or both roles naming the same one."""
    implementer, reviewer = models(body)
    if not implementer or not reviewer:
        return ["name the implementer and the reviewer model in the Models section"]
    if implementer == reviewer:
        return [f"the implementer and the reviewer are both {implementer!r}; the reviewer must be a different model"]
    return []


def main() -> int:
    """Check the body in ``PR_BODY``; print the problems; return 1 if any."""
    found = problems(os.environ.get("PR_BODY", ""))
    for line in found:
        print(line)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
