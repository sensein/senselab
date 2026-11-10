"""Fail when a pull request body lacks its models or a per-commit checklist line, or breaks the P-40 pairing.

The 2.x template has the lines ``- Implementer (model, effort):``, ``- Reviewer (model, effort; ...):`` and
``- High-stakes class (...):``; a model is the text before the first comma, compared without case. The pairing is
engineering.md's P-40 as amended 2026-11-08 and 2026-11-09:

- Implementers are sonnet. Opus implements only a task escalated after a second failure or a design task, and the
  implementer line says ``escalated`` or ``design`` (not negated). Fable and haiku never implement.
- Reviewers are opus or fable. The only same-model pairing is opus reviewing opus, and the reviewer line says
  ``fresh`` (not negated). Sonnet and haiku never review.
- Fable reviews only when the ``High-stakes class`` line names one of ``HIGH_STAKES``; a model name may appear
  anywhere in its field (``Claude Sonnet 5.5``), and the class may be followed by a note.

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
NEGATION = re.compile(r"(?:\b(?:not|never|no|isn't|wasn't|without)|n't)\s+(?:\w+\s+)?$", re.IGNORECASE)
TIER = re.compile(r"\b(" + "|".join(TIERS) + r")\b")


def _field(text: str, role: str) -> str:
    found = re.search(LINE.format(role=role), text, re.MULTILINE)
    return found.group(1).strip() if found else ""


def _text(body: str) -> str:
    return re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL)


def tier(model: str) -> str:
    """The first tier name (``TIERS``) that appears as a word in `model`, or empty when it names none."""
    found = TIER.search(model.lower())
    return found.group(1) if found else ""


def _says(pattern: re.Pattern[str], text: str) -> bool:
    """Whether `text` matches `pattern` at least once without a negation just before it."""
    return any(not NEGATION.search(text[: found.start()]) for found in pattern.finditer(text))


def models(body: str) -> tuple[str, str]:
    """The implementer and reviewer models named in `body`, each empty when its line is missing or blank."""
    text = _text(body)
    implementer, reviewer = (_field(text, role).split(",")[0].strip().lower() for role in ("Implementer", "Reviewer"))
    return implementer, reviewer


def high_stakes(body: str) -> str:
    """The high-stakes class `body` declares, lowercased; empty for ``none`` or a missing line."""
    found = re.match(r"[a-z][a-z-]*", _field(_text(body), "High-stakes class").lower())
    return "" if not found or found.group(0) == "none" else found.group(0)


def _roles(body: str, implementer: str, reviewer: str) -> list[str]:
    """What breaks P-40's roles for two named models."""
    text = _text(body)
    out = []
    impl_tier, review_tier = tier(implementer), tier(reviewer)
    if impl_tier not in ("sonnet", "opus"):
        out.append(f"the implementer is {implementer!r}; implementation runs on sonnet (opus once escalated), never fable")
    elif impl_tier == "opus" and not _says(ESCALATED, _field(text, "Implementer")):
        out.append("an opus implementer must say `escalated` (after a second failure) or `design` on its line")
    if review_tier not in ("opus", "fable"):
        out.append(f"the reviewer is {reviewer!r}; reviews run on opus (fable only for the high-stakes classes)")
    elif review_tier == impl_tier and not (review_tier == "opus" and _says(FRESH, _field(text, "Reviewer"))):
        out.append("the reviewer's model equals the implementer's; only opus may review opus, and says `fresh` on its line")
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
        if tier(reviewer) == "fable" and stakes not in HIGH_STAKES:
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
