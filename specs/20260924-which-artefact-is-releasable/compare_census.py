"""Compare two axes censuses — r3 against r4 — and separate the two effects.

    python3 compare.py BEFORE_DIR AFTER_DIR

Both directories hold the per-slice JSON that ``axes_census.py`` writes. The release axis and the
task conformance are reported apart, because two independent changes reach the same replay.
"""

import collections
import json
import sys
from pathlib import Path


def load(directory):
    """Sum one census's slices.

    Args:
        directory: The directory of per-slice JSON.

    Returns:
        Total rows, and one Counter per section.
    """
    total = 0
    sections = {k: collections.Counter() for k in ("release", "triage", "task_conformance")}
    for path in sorted(Path(directory).glob("*.json")):
        record = json.loads(path.read_text())
        total += record["n"]
        for name, counter in sections.items():
            for key, value in record[name]:
                counter[tuple(key)] += value
    return total, sections


before_n, before = load(sys.argv[1])
after_n, after = load(sys.argv[2])
print(f"before {before_n} recordings   after {after_n} recordings")

print("\n== release ==")
by = lambda c: collections.Counter({k[0]: v for k, v in c.items()})  # noqa: E731
b, a = by(before["release"]), by(after["release"])
for name in sorted(set(b) | set(a)):
    print(f"  {name:28s} {b.get(name, 0):7d} -> {a.get(name, 0):7d}")
print("\n== release x ground ==")
for key in sorted(set(before["release"]) | set(after["release"])):
    print(f"  {key[0]:28s} {key[1][:52]:54s} {before['release'].get(key, 0):7d} -> {after['release'].get(key, 0):7d}")

print("\n== task conformance, families that moved ==")
fams = collections.defaultdict(lambda: [collections.Counter(), collections.Counter()])
for (family, node, answer), n in before["task_conformance"].items():
    fams[family][0][answer] += n
for (family, node, answer), n in after["task_conformance"].items():
    fams[family][1][answer] += n
moved = 0
for family in sorted(fams):
    b2, a2 = fams[family]
    if b2 == a2:
        continue
    moved += 1
    keys = sorted(set(b2) | set(a2))
    print(f"  {family:34s} " + "  ".join(f"{k}: {b2.get(k, 0)} -> {a2.get(k, 0)}" for k in keys))
print(f"  ({moved} of {len(fams)} declared families moved)")

print("\n== triage ==")
tb = collections.Counter({k[1]: v for k, v in before["triage"].items()})
ta = collections.Counter({k[1]: v for k, v in after["triage"].items()})
for name in sorted(set(tb) | set(ta)):
    print(f"  {name:12s} {tb.get(name, 0):7d} -> {ta.get(name, 0):7d}")
print("\n== triage, families that moved ==")
fam_t = collections.defaultdict(lambda: [collections.Counter(), collections.Counter()])
for (family, answer), n in before["triage"].items():
    fam_t[family][0][answer] += n
for (family, answer), n in after["triage"].items():
    fam_t[family][1][answer] += n
for family in sorted(fam_t):
    b2, a2 = fam_t[family]
    if b2 == a2:
        continue
    keys = sorted(set(b2) | set(a2))
    print(f"  {family:34s} " + "  ".join(f"{k}: {b2.get(k, 0)} -> {a2.get(k, 0)}" for k in keys))
