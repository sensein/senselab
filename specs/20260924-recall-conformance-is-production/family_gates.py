r"""Emit the per-family conformance gate names on both sides of the recall change.

Read-only. Writes one JSON mapping family -> the branch that owns it, its task group, its
anti-pattern, the conformance gate names the **pre-change** rule returned, and the names the
**post-change** rule returns. The corpus scan consumes it, so the A/B is one table built from the
live expectation rows rather than two hand-written lists.

Usage::

    uv run python specs/20260924-recall-conformance-is-production/family_gates.py <out.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from senselab.audio.workflows.triage.nodes.branches import EXPECTATIONS
from senselab.audio.workflows.triage.nodes.gates import CONFORMANCE_GATES, Pattern, conformance_gate_names

RECALL_ANTI_PATTERN = "verbatim_source"
"""The anti-pattern the withdrawn rule keyed its reassignment on."""


def pre_change_names(pattern: Pattern, anti_pattern: str | None) -> list[str]:
    """The conformance gate names the rule withdrawn on 2026-09-24 returned.

    Args:
        pattern: The group the expectation row declares.
        anti_pattern: The row's anti-pattern, when it declares one.

    Returns:
        The gate names.
    """
    if pattern is Pattern.FREE_RESPONSE and anti_pattern == RECALL_ANTI_PATTERN:
        return ["coverage_min"]
    return list(CONFORMANCE_GATES[pattern])


def main() -> None:
    """Write the table named by the one argument."""
    table = {}
    for branch, rows in EXPECTATIONS.items():
        for family, row in rows.items():
            table[family] = {
                "branch": branch,
                "pattern": row.pattern.name,
                "anti_pattern": row.anti_pattern,
                "old": pre_change_names(row.pattern, row.anti_pattern),
                "new": list(conformance_gate_names(row.pattern)),
            }
    Path(sys.argv[1]).write_text(json.dumps(table, indent=1, sort_keys=True))
    changed = sorted(family for family, entry in table.items() if entry["old"] != entry["new"])
    print(f"{len(table)} families, {len(changed)} change: {changed}")


if __name__ == "__main__":
    main()
