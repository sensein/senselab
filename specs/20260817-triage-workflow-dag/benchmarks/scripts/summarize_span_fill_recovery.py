"""Turn ``span_fill_recovery.py``'s raw JSON into the recovery tables and false-label tallies.

Usage:
    uv run python summarize_span_fill_recovery.py results.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TRIMS = (0.06, 0.10, 0.15, 0.20, 0.30, 0.40, 0.48, 0.70)
STRATA = ("stationary", "speech", "transient", "other")
ARMS = ("filled", "zero")


def load(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read back ``(reference, records)`` from ``span_fill_recovery.py``'s output JSON."""
    doc = json.loads(path.read_text())
    return doc["reference"], doc["records"]


def recovery_table(records: list[dict[str, Any]], position: str) -> str:
    """One markdown table: rows = trim, columns = stratum x arm, cells = top1 / top4 recovery."""
    cells: dict[tuple[float, str, str], list[bool]] = defaultdict(list)
    cells4: dict[tuple[float, str, str], list[bool]] = defaultdict(list)
    counts: dict[tuple[float, str, str], int] = defaultdict(int)
    for r in records:
        if r["position"] != position:
            continue
        key = (r["trim"], r["stratum"], r["arm"])
        top1_hit = r["top1_label"] == r["native_label"]
        ranked_labels = [lbl for lbl, _ in r["labels_ranked"][:4]]
        top4_hit = r["native_label"] in ranked_labels
        cells[key].append(top1_hit)
        cells4[key].append(top4_hit)
        counts[key] += 1

    header = ["trim (s)"]
    for stratum in STRATA:
        for arm in ARMS:
            header.append(f"{stratum}/{arm}")
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for trim in TRIMS:
        row = [f"{trim:.2f}"]
        for stratum in STRATA:
            for arm in ARMS:
                key = (trim, stratum, arm)
                n = counts.get(key, 0)
                if n == 0:
                    row.append("-")
                    continue
                top1_frac = sum(cells[key]) / n
                top4_frac = sum(cells4[key]) / n
                row.append(f"{top1_frac:.2f}/{top4_frac:.2f} (n={n})")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def false_labels(records: list[dict[str, Any]]) -> Counter:
    """Tally of top-1 labels produced when the fill loses the reference label (filled arm only)."""
    c: Counter = Counter()
    for r in records:
        if r["arm"] != "filled":
            continue
        if r["top1_label"] is not None and r["top1_label"] != r["native_label"]:
            c[r["top1_label"]] += 1
    return c


def main() -> None:
    """Print the recovery tables and false-label tallies for one results JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()

    reference, records = load(args.results)

    print(f"# reference set: {len(reference)} spans\n")
    by_stratum_source = Counter((r["stratum"], r["source"]) for r in reference)
    for stratum in STRATA:
        for source in ("store_span", "native_window"):
            n = by_stratum_source.get((stratum, source), 0)
            if n:
                print(f"- {stratum} / {source}: {n}")
    print()

    print("## Recovery, trimmed from the START\n")
    print(recovery_table(records, "start"))
    print()
    print("## Recovery, trimmed from the CENTRE\n")
    print(recovery_table(records, "centre"))
    print()

    print("## False top-1 labels when the FILLED arm loses the reference (all trims/positions)\n")
    for label, n in false_labels(records).most_common(25):
        print(f"- {label}: {n}")

    print("\n## False top-1 labels by stratum (filled arm)\n")
    for stratum in STRATA:
        c = false_labels([r for r in records if r["stratum"] == stratum])
        print(f"### {stratum}")
        for label, n in c.most_common(10):
            print(f"- {label}: {n}")


if __name__ == "__main__":
    main()
