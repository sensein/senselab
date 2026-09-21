"""Join the two replay sides and report conformance per family, with denominators.

Reads the JSONL each side of :mod:`corpus-replay` wrote and prints, per declared family: how many
recordings were replayed, what the pre-change branch answered, what the post-change gates answer,
and every recording on which the two disagree.

    REPLAY_OUT=<dir holding old/ and new/> python corpus-replay-report.py
"""

from __future__ import annotations

import collections
import json
import os
from pathlib import Path
from typing import Any


def load(side: Path) -> dict[str, dict[str, Any]]:
    """Every replayed recording on one side, keyed by stem.

    Args:
        side: The directory that side's array wrote into.

    Returns:
        Stem to its row.
    """
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(side.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["stem"]] = row
    return rows


def shown(value: Any) -> str:  # noqa: ANN401 — a Conformance
    """One conformance as a table cell.

    Args:
        value: True, False, ``"UNDETERMINED"`` or None.

    Returns:
        Its label.
    """
    return {True: "True", False: "False", None: "not_replayed"}.get(value, str(value))


def main() -> None:
    """Print the before-and-after table and every disagreement."""
    base = Path(os.environ["REPLAY_OUT"])
    old, new = load(base / "old"), load(base / "new")
    stems = sorted(set(old) & set(new))
    per_family: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    disagreements: list[dict[str, Any]] = []
    errors: collections.Counter[str] = collections.Counter()
    recorded_mismatch: list[dict[str, Any]] = []

    for stem in stems:
        before, after = old[stem], new[stem]
        family = before["family"] or "(no row)"
        if before.get("error") or after.get("error"):
            errors[family] += 1
            continue
        if before["branch"] is None:
            continue
        counter = per_family[family]
        counter["n"] += 1
        counter[f"before:{shown(before['conformance'])}"] += 1
        counter[f"after:{shown(after['conformance'])}"] += 1
        if before["conformance"] != after["conformance"]:
            counter["moved"] += 1
            disagreements.append(
                {
                    "stem": stem,
                    "family": family,
                    "before": shown(before["conformance"]),
                    "after": shown(after["conformance"]),
                    "readings": after.get("readings", []),
                }
            )
        branch = before["branch"]
        as_run = (before.get("recorded") or {}).get(branch)
        if branch in (before.get("recorded") or {}) and as_run != before["conformance"]:
            recorded_mismatch.append(
                {"stem": stem, "family": family, "as_run": shown(as_run), "replayed": shown(before["conformance"])}
            )

    print("| family | n | before True/False/UND | after True/False/UND | moved |")
    print("| --- | --- | --- | --- | --- |")
    totals: collections.Counter[str] = collections.Counter()
    for family in sorted(per_family):
        counter = per_family[family]
        totals.update(counter)
        shown_before = "/".join(str(counter[f"before:{value}"]) for value in ("True", "False", "UNDETERMINED"))
        shown_after = "/".join(str(counter[f"after:{value}"]) for value in ("True", "False", "UNDETERMINED"))
        print(f"| `{family}` | {counter['n']} | {shown_before} | {shown_after} | {counter['moved']} |")
    all_before = "/".join(str(totals[f"before:{value}"]) for value in ("True", "False", "UNDETERMINED"))
    all_after = "/".join(str(totals[f"after:{value}"]) for value in ("True", "False", "UNDETERMINED"))
    print(f"| **all** | **{totals['n']}** | **{all_before}** | **{all_after}** | **{totals['moved']}** |")

    print(f"\nreplayed on both sides: {len(stems)}")
    print(f"errored on one side or the other: {sum(errors.values())} {dict(errors)}")
    print(f"the replayed `before` differs from the run's own record on: {len(recorded_mismatch)}")
    for row in recorded_mismatch[:20]:
        print(f"  {row}")

    print(f"\ndisagreements: {len(disagreements)}")
    by_move: collections.Counter[str] = collections.Counter()
    for row in disagreements:
        by_move[f"{row['family']}: {row['before']} -> {row['after']}"] += 1
    for label, count in by_move.most_common():
        print(f"  {count:>6}  {label}")
    for row in disagreements[:10]:
        print(f"  example {row}")


if __name__ == "__main__":
    main()
