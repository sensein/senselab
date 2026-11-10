"""Fail when a pull request body lacks its models or a per-commit checklist line, or breaks the P-40 pairing.

The 2.x template has the lines ``- Implementer (model, effort):``, ``- Reviewer (model, effort; ...):`` and
``- High-stakes class (...):``; a model is the text before the first comma, compared without case. The pairing is
engineering.md's P-40 as amended 2026-11-08 and 2026-11-09:

- Implementers are sonnet. Opus implements only a task escalated after a second failure or a design task, and the
  implementer line says ``escalated`` or ``design``. Haiku never implements.
- Reviewers are opus, in a fresh context: an opus reviewer of an opus implementer says ``fresh`` on its line.
  Sonnet and haiku never review.
- Fable appears, as implementer or reviewer, only when the ``High-stakes class`` line names one of ``HIGH_STAKES``.

Every item of engineering.md's per-commit checklist must appear in the body, ticked or not.

Usage::

    PR_BODY="$body" uv run python scripts/check_pr_models.py
"""

from __future__ import annotations

import os
import re
import sys

from check_pr_template import checklist, items

LINE = r"^- {role}\b[^:\n]*:[ \t]*([^\n]*)$"
TIERS = ("sonnet", "opus", "fable", "haiku")
HIGH_STAKES = ("identity-privacy-security", "protocol-adr", "second-round-design")
ESCALATED = re.compile(r"\b(escalated|design)\b", re.IGNORECASE)
FRESH = re.compile(r"\bfresh\b", re.IGNORECASE)


def _field(text: str, role: str) -> str:
    found = re.search(LINE.format(role=role), text, re.MULTILINE)
    return found.group(1).strip() if found else ""


def _text(body: str) -> str:
    return re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL)


def tier(model: str) -> str:
    """The tier a model name belongs to (``TIERS``), or empty when it names none."""
    return next((name for name in TIERS if model.startswith(name)), "")


def models(body: str) -> tuple[str, str]:
    """The implementer and reviewer models named in `body`, each empty when its line is missing or blank."""
    text = _text(body)
    implementer, reviewer = (_field(text, role).split(",")[0].strip().lower() for role in ("Implementer", "Reviewer"))
    return implementer, reviewer


def high_stakes(body: str) -> str:
    """The high-stakes class `body` declares, lowercased; empty for ``none`` or a missing line."""
    value = _field(_text(body), "High-stakes class").split(",")[0].strip().lower()
    return "" if value in ("", "none") else value


def _roles(body: str, implementer: str, reviewer: str) -> list[str]:
    """What breaks P-40's roles for two named models."""
    text = _text(body)
    out = []
    impl_tier, review_tier = tier(implementer), tier(reviewer)
    if impl_tier not in ("sonnet", "opus", "fable"):
        out.append(f"the implementer is {implementer!r}; implementation runs on sonnet (opus once escalated)")
    elif impl_tier == "opus" and not ESCALATED.search(_field(text, "Implementer")):
        out.append("an opus implementer must say `escalated` (after a second failure) or `design` on its line")
    if review_tier not in ("opus", "fable"):
        out.append(f"the reviewer is {reviewer!r}; reviews run on opus (fable only for the high-stakes classes)")
    elif review_tier == impl_tier == "opus" and not FRESH.search(_field(text, "Reviewer")):
        out.append("an opus reviewer of an opus implementer must say `fresh` (a fresh context) on its line")
    return out


def problems(body: str, required: list[str]) -> list[str]:
    """Why `body` fails: a missing model, a pairing outside P-40, an unknown class or a missing checklist line."""
    implementer, reviewer = models(body)
    out = []
    stakes = high_stakes(body)
    if stakes and stakes not in HIGH_STAKES:
        out.append(f"the high-stakes class is {stakes!r}; use none or one of {', '.join(HIGH_STAKES)}")
    if not implementer or not reviewer:
        out.append("name the implementer and the reviewer model in the Models section")
    else:
        out += _roles(body, implementer, reviewer)
        if "fable" in (tier(implementer), tier(reviewer)) and stakes not in HIGH_STAKES:
            out.append(f"fable is for the high-stakes classes only; name one of {', '.join(HIGH_STAKES)} or use opus")
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
