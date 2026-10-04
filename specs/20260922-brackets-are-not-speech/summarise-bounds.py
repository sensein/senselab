"""Second pass: the cuts that decide whether a duration bound earns its place."""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
rows = [r for r in rows if "error" not in r]
findings = [(r, f) for r in rows if r["findings"] for f in r["findings"]]

print("== the long tail: what makes a finding long ==")
long_ones = [(r, f) for r, f in findings if f["duration_s"] > 10]
print(f"   findings > 10 s: {len(long_ones)}")
whole = [(r, f) for r, f in long_ones if f["n_covered"] >= r["n_words"] > 1]
print(f"     of which cover EVERY consensus word (the unlocated widening): {len(whole)}")
print(
    f"     of which cover 1 word only                                  : {sum(1 for _, f in long_ones if f['n_covered'] == 1)}"
)
ratios = [f["duration_s"] / f["covered_hull_s"] for _, f in long_ones if f["covered_hull_s"]]
print(f"     duration/hull among them: med={statistics.median(ratios):.2f} max={max(ratios):.1f}")
allwhole = [(r, f) for r, f in findings if f["n_covered"] >= r["n_words"] > 1]
print(f"   findings covering every word, any length: {len(allwhole)} ({100 * len(allwhole) / len(findings):.2f}%)")
print(
    f"     their durations: med={statistics.median([f['duration_s'] for _, f in allwhole]):.2f} "
    f"max={max(f['duration_s'] for _, f in allwhole):.1f}"
)

print()
print("== would a duration bound have caught the defect? ==")
brkonly = [f for _, f in findings if f["all_covered_bracketed"]]
clean = [f for _, f in findings if not f["any_covered_bracketed"]]
for bound in (2.0, 3.0, 5.0, 6.0, 8.0, 10.0):
    caught = sum(1 for f in brkonly if f["duration_s"] > bound)
    cost = sum(1 for f in clean if f["duration_s"] > bound)
    print(
        f"   refuse > {bound:4.1f} s : catches {caught:5d} / {len(brkonly)} bracket-only "
        f"({100 * caught / len(brkonly):5.1f}%)  and refuses {cost:6d} bracket-free findings"
    )
print()
for bound in (1.5, 2.0, 3.0, 5.0):
    caught = sum(1 for f in brkonly if f["covered_hull_s"] and f["duration_s"] / f["covered_hull_s"] > bound)
    cost = sum(1 for f in clean if f["covered_hull_s"] and f["duration_s"] / f["covered_hull_s"] > bound)
    print(
        f"   refuse ratio > {bound:4.1f} : catches {caught:5d} / {len(brkonly)} bracket-only "
        f"({100 * caught / len(brkonly):5.1f}%)  and refuses {cost:6d} bracket-free findings"
    )

print()
print("== what (a) alone recovers ==")
recovered = [r for r in rows if any(f["all_covered_bracketed"] for f in r["findings"])]
te_recovered = [r for r in rows if any(f["all_covered_bracketed"] and f["hits_task_extent"] for f in r["findings"])]
te_only_bracket = [
    r
    for r in rows
    if any(f["hits_task_extent"] for f in r["findings"])
    and all(f["all_covered_bracketed"] for f in r["findings"] if f["hits_task_extent"])
]
print(
    f"   recordings with >=1 finding raised ONLY on bracketed words: {len(recovered)}"
    f" ({100 * len(recovered) / len(rows):.2f}%)"
)
print(f"   ... where such a finding reaches the task_extent          : {len(te_recovered)}")
print(f"   ... and EVERY task_extent-reaching finding is bracket-only: {len(te_only_bracket)}")
print("   the last group by family:", Counter(r["family"] for r in te_only_bracket).most_common(12))
print()
mixed = [f for _, f in findings if f["any_covered_bracketed"] and not f["all_covered_bracketed"]]
print(f"   findings with SOME but not all coverage bracketed: {len(mixed)} — outcome after (a) not predictable")
print()
print("== seconds of audio ==")
print(f"   total seconds inside bracket-only findings: {sum(f['duration_s'] for f in brkonly):.0f}")
print(
    f"   median / p90 / max of those              : "
    f"{statistics.median([f['duration_s'] for f in brkonly]):.2f} / "
    f"{sorted(f['duration_s'] for f in brkonly)[int(0.9 * len(brkonly))]:.2f} / "
    f"{max(f['duration_s'] for f in brkonly):.2f}"
)
