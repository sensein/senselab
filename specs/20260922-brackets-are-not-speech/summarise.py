"""Summarise the bracket-token PII measurement. Counts, categories, durations and stems only."""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path


def quantiles(values: list[float]) -> str:
    if not values:
        return "n=0"
    ordered = sorted(values)

    def q(p: float) -> float:
        return ordered[min(len(ordered) - 1, int(p * len(ordered)))]

    return (
        f"n={len(ordered)} min={ordered[0]:.2f} p25={q(0.25):.2f} med={statistics.median(ordered):.2f} "
        f"p75={q(0.75):.2f} p90={q(0.90):.2f} p95={q(0.95):.2f} p99={q(0.99):.2f} max={ordered[-1]:.2f}"
    )


def main() -> None:
    rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
    errors = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    print(f"recordings where REDACT ran: {len(rows)}   read errors: {len(errors)}")
    if errors:
        print("  error kinds:", Counter(r["error"].split(":")[0] for r in errors).most_common(5))

    with_findings = [r for r in rows if r["findings"]]
    all_findings = [(r, f) for r in rows for f in r["findings"]]
    print(f"recordings with >=1 live pii finding: {len(with_findings)}  findings: {len(all_findings)}")
    print()

    # --- 1. bracket overlap ---
    bracket_rows = [r for r in rows if any(f["any_covered_bracketed"] for f in r["findings"])]
    print("== 1. recordings with a finding overlapping a bracketed consensus word ==")
    print(f"   {len(bracket_rows)} / {len(rows)} = {100 * len(bracket_rows) / max(1, len(rows)):.2f}% of REDACT-ran")
    print(
        f"   {len(bracket_rows)} / {len(with_findings)} = "
        f"{100 * len(bracket_rows) / max(1, len(with_findings)):.2f}% of recordings that have any finding"
    )
    fam_all = Counter(r["family"] for r in rows)
    fam_hit = Counter(r["family"] for r in bracket_rows)
    print("   by declared family (hit / REDACT-ran / share):")
    for family, hit in fam_hit.most_common():
        print(f"     {str(family):46s} {hit:6d} / {fam_all[family]:6d}  {100 * hit / fam_all[family]:6.2f}%")
    print("   families with no bracket hit:", sorted({str(f) for f in fam_all} - {str(f) for f in fam_hit}))
    print()

    # --- 4. categories ---
    print("== 4. findings by category ==")
    cat_all = Counter(f["category"] for _, f in all_findings)
    cat_br = Counter(f["category"] for _, f in all_findings if f["any_covered_bracketed"])
    cat_allbr = Counter(f["category"] for _, f in all_findings if f["all_covered_bracketed"])
    print(f"   {'category':22s} {'all':>8s} {'any-brk':>8s} {'all-brk':>8s} {'%all-brk':>9s}")
    for category, total in cat_all.most_common():
        print(
            f"   {category:22s} {total:8d} {cat_br[category]:8d} {cat_allbr[category]:8d} "
            f"{100 * cat_allbr[category] / total:8.2f}%"
        )
    print("   by haystack:", Counter(f["haystack"] for _, f in all_findings if f["any_covered_bracketed"]))
    print("   by detector source (bracket-only findings):")
    for source, n in Counter(f["source"] for _, f in all_findings if f["all_covered_bracketed"]).most_common(8):
        print(f"     {source:34s} {n}")
    print()

    # --- 2. durations ---
    print("== 2. finding duration vs the words it overlaps (seconds) ==")
    print("   all findings           :", quantiles([f["duration_s"] for _, f in all_findings]))
    clean = [f for _, f in all_findings if not f["any_covered_bracketed"]]
    dirty = [f for _, f in all_findings if f["all_covered_bracketed"]]
    print("   no bracketed word      :", quantiles([f["duration_s"] for f in clean]))
    print("   only bracketed words   :", quantiles([f["duration_s"] for f in dirty]))
    print(
        "   covered-word hull, all :", quantiles([f["covered_hull_s"] for _, f in all_findings if f["covered_hull_s"]])
    )
    ratios = [
        (f["duration_s"] / f["covered_hull_s"], f["any_covered_bracketed"])
        for _, f in all_findings
        if f["covered_hull_s"]
    ]
    print("   duration / hull, all   :", quantiles([r for r, _ in ratios]))
    print("   duration / hull, clean :", quantiles([r for r, b in ratios if not b]))
    print("   duration / hull, brkt  :", quantiles([r for r, b in ratios if b]))
    print("   words covered, all     :", Counter(f["n_covered"] for _, f in all_findings).most_common(8))
    for bound in (2.0, 3.0, 5.0, 8.0, 10.0):
        n = sum(1 for _, f in all_findings if f["duration_s"] > bound)
        nc = sum(1 for f in clean if f["duration_s"] > bound)
        print(
            f"   > {bound:4.1f} s : {n:6d} findings ({100 * n / max(1, len(all_findings)):5.2f}%)   "
            f"of which bracket-free: {nc}"
        )
    print()

    # --- 3. task extent ---
    print("== 3. recordings whose finding overlaps their own task_extent ==")
    te_rows = [r for r in rows if any(f["hits_task_extent"] for f in r["findings"])]
    te_br = [r for r in rows if any(f["hits_task_extent"] and f["any_covered_bracketed"] for f in r["findings"])]
    te_clean = [r for r in rows if any(f["hits_task_extent"] and not f["any_covered_bracketed"] for f in r["findings"])]
    have_te = [r for r in rows if r["n_task_extents"]]
    print(f"   recordings with a task_extent at all: {len(have_te)}")
    print(
        f"   finding hits the task_extent       : {len(te_rows)}  ({100 * len(te_rows) / max(1, len(rows)):.2f}% of REDACT-ran)"
    )
    print(f"     of those, bracket-involved       : {len(te_br)}")
    print(f"     of those, bracket-free           : {len(te_clean)}")
    print("   bracket-free task_extent hits by family:", Counter(r["family"] for r in te_clean).most_common(10))
    print()

    # --- what (a) leaves behind ---
    print("== after fix (a): findings whose overlap contains no bracketed word ==")
    print(f"   findings remaining: {len(clean)} / {len(all_findings)}")
    rest = [r for r in rows if any(not f["any_covered_bracketed"] for f in r["findings"])]
    print(f"   recordings remaining with a finding: {len(rest)} / {len(with_findings)}")
    print("   remaining by category:", Counter(f["category"] for f in clean).most_common(10))
    print(
        "   remaining, ratio > 3 :",
        sum(1 for f in clean if f["covered_hull_s"] and f["duration_s"] / f["covered_hull_s"] > 3),
    )
    print("   remaining, > 5 s     :", sum(1 for f in clean if f["duration_s"] > 5))
    print("   unlocated (0 words)  :", sum(1 for _, f in all_findings if f["n_covered"] == 0))


if __name__ == "__main__":
    main()
