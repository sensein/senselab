"""The same 300 recordings before and after the fix, per family.

Takes two census files and prints what moved: conformance, the found-no-subject rate, and the
flag records the fold wrote. Paired per recording, so a family's rows are the same recordings on
both sides.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

IN_FAMILY = (
    "maximum-phonation-time",
    "maximum-phonation-time-v2",
    "prolonged-vowel",
    "glides-low-to-high",
    "glides-high-to-low",
    "high-to-low",
)


def load(path: Path) -> dict[str, dict]:
    """Census rows that parsed, keyed by recording."""
    return {r["stem_id"]: r for r in (json.loads(line) for line in path.read_text().splitlines()) if r.get("ok")}


def conformance(row: dict) -> str:
    """The branch's conformance, as a printable token."""
    value = row.get("voice_conformance")
    return "none" if value is None else str(value)


def main() -> int:
    """Print the paired before/after tables."""
    before, after = load(Path(sys.argv[1])), load(Path(sys.argv[2]))
    paired = sorted(set(before) & set(after))
    print(f"paired recordings: {len(paired)}\n")

    print("== conformance[VOICE], per family ==")
    print(f"{'family':<30}{'n':>4}   {'True':>12}   {'False':>12}   {'UNDETERMINED':>14}")
    print(f"{'':<30}{'':>4}   {'before after':>12}   {'before after':>12}   {'before   after':>14}")
    for fam in sorted({before[s]["family"] for s in paired}, key=lambda f: (f not in IN_FAMILY, f)):
        rows = [s for s in paired if before[s]["family"] == fam]
        b = collections.Counter(conformance(before[s]) for s in rows)
        a = collections.Counter(conformance(after[s]) for s in rows)
        print(
            f"{fam:<30}{len(rows):>4}   {b['True']:>6} {a['True']:>5}   "
            f"{b['False']:>6} {a['False']:>5}   {b['UNDETERMINED']:>6} {a['UNDETERMINED']:>7}"
        )

    print("\n== routed and found no subject, per family ==")
    print(f"{'family':<30}{'routed':>8}{'before':>8}{'after':>7}{'rate before':>13}{'rate after':>12}")
    for fam in sorted({before[s]["family"] for s in paired}, key=lambda f: (f not in IN_FAMILY, f)):
        rows = [s for s in paired if before[s]["family"] == fam]
        routed = [s for s in rows if (before[s].get("routes") or {}).get("VOICE") == "routed"]
        b = sum(1 for s in routed if (before[s].get("findings") or {}).get("VOICE") == "absent")
        a = sum(1 for s in routed if (after[s].get("findings") or {}).get("VOICE") == "absent")
        if routed:
            print(f"{fam:<30}{len(routed):>8}{b:>8}{a:>7}{b / len(routed):>13.2f}{a / len(routed):>12.2f}")

    print("\n== VOICE flag records the fold wrote ==")
    for label, source in (("before", before), ("after", after)):
        grounds: collections.Counter = collections.Counter()
        per = collections.Counter()
        for s in paired:
            reasons = [w for n, o, w in (source[s].get("reasons") or []) if n == "VOICE" and o == "flag"]
            per[len(reasons)] += 1
            for why in reasons:
                key = (
                    "conformance: the instruction was not met"
                    if "did not happen" in why
                    else "agreement: routing expected it, none found"
                    if why.startswith("mismatch: routing routed")
                    else "agreement: routing declined it, one found"
                    if why.startswith("mismatch: routing declined")
                    else "hints: the declaration expected it, none found"
                    if why.startswith("hint mismatch")
                    else "other"
                )
                grounds[key] += 1
        spread = dict(sorted(per.items()))
        print(f"  {label}: {sum(grounds.values())} records over {len(paired)} recordings; per recording {spread}")
        for key, n in grounds.most_common():
            print(f"      {n:>4}  {key}")

    print("\n== carriers the branch recorded discarding (the fix that made this readable) ==")
    for label, source in (("before", before), ("after", after)):
        recorded = sum(len(source[s].get("rejections_in_store") or []) for s in paired)
        carrying = sum(1 for s in paired if source[s].get("rejections_in_store"))
        print(f"  {label}: {recorded} rejections recorded, on {carrying} of {len(paired)} recordings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
