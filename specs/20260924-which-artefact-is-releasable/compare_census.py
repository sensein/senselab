"""Compare two axes censuses — r3 against r4 — and separate the two effects.

    python3 compare_census.py BEFORE_DIR AFTER_DIR

Both directories hold the per-slice JSON that ``axes_census.py`` writes. The release axis and the
task conformance are reported apart, because two independent changes reach the same replay.

Runs under the cluster's bare ``python3``: no senselab import, no third-party dependency.
"""

import collections
import json
import sys
from pathlib import Path
from typing import Any


def load(directory: str) -> tuple[int, dict[str, collections.Counter]]:
    """Sum one census's slices.

    Args:
        directory: The directory of per-slice JSON.

    Returns:
        Total recordings read, and one Counter per section.
    """
    total = 0
    sections: dict[str, collections.Counter] = {
        name: collections.Counter() for name in ("release", "triage", "task_conformance")
    }
    for path in sorted(Path(directory).glob("*.json")):
        record = json.loads(path.read_text())
        total += record["n"]
        for name, counter in sections.items():
            for key, value in record[name]:
                counter[tuple(key)] += value
    return total, sections


def roll(counter: collections.Counter, field: int) -> collections.Counter:
    """Sum a tuple-keyed counter down to one field of its key.

    Args:
        counter: The counter, keyed by tuple.
        field: Which element of the key to keep.

    Returns:
        The summed counter, built by addition rather than by a dict comprehension, which would let
        a repeated key overwrite its siblings instead of joining them.
    """
    out: collections.Counter = collections.Counter()
    for key, value in counter.items():
        out[key[field]] += value
    return out


def pairs(section: collections.Counter, other: collections.Counter, field: int) -> dict[str, list]:
    """Both sides of one section, rolled to one field and joined on it.

    Args:
        section: The before side.
        other: The after side.
        field: Which element of the key to keep.

    Returns:
        Key to ``[before, after]``.
    """
    before, after = roll(section, field), roll(other, field)
    return {name: [before.get(name, 0), after.get(name, 0)] for name in sorted(set(before) | set(after))}


def by_family(section: collections.Counter, other: collections.Counter, answer: int) -> dict[str, list]:
    """Both sides of one section, grouped by the declared family in key position 0.

    Args:
        section: The before side.
        other: The after side.
        answer: Which element of the key holds the answer being counted.

    Returns:
        Family to ``[before Counter, after Counter]``.
    """
    grouped: dict[str, list] = collections.defaultdict(lambda: [collections.Counter(), collections.Counter()])
    for side, counter in ((0, section), (1, other)):
        for key, value in counter.items():
            grouped[str(key[0])][side][key[answer]] += value
    return dict(grouped)


def main(argv: list[str]) -> int:
    """Print the comparison.

    Args:
        argv: The two census directories.

    Returns:
        0.
    """
    before_n, before = load(argv[0])
    after_n, after = load(argv[1])
    print(f"before {before_n} recordings   after {after_n} recordings")

    print("\n== release ==")
    for name, (was, now) in pairs(before["release"], after["release"], 0).items():
        print(f"  {name:28s} {was:7d} -> {now:7d}")

    print("\n== release x ground ==")
    for key in sorted(set(before["release"]) | set(after["release"])):
        was, now = before["release"].get(key, 0), after["release"].get(key, 0)
        print(f"  {key[0]:28s} {key[1][:52]:54s} {was:7d} -> {now:7d}")

    print("\n== task conformance, families that moved ==")
    families: dict[str, Any] = by_family(before["task_conformance"], after["task_conformance"], 2)
    moved = 0
    for family in sorted(families):
        was, now = families[family]
        if was == now:
            continue
        moved += 1
        keys = sorted(set(was) | set(now))
        print(f"  {family:34s} " + "  ".join(f"{k}: {was.get(k, 0)} -> {now.get(k, 0)}" for k in keys))
    print(f"  ({moved} of {len(families)} declared families moved)")

    print("\n== triage ==")
    for name, (was_n, now_n) in pairs(before["triage"], after["triage"], 1).items():
        print(f"  {name:12s} {was_n:7d} -> {now_n:7d}")

    print("\n== triage, families that moved ==")
    for family, (was, now) in sorted(by_family(before["triage"], after["triage"], 1).items()):
        if was == now:
            continue
        keys = sorted(set(was) | set(now))
        print(f"  {family:34s} " + "  ".join(f"{k}: {was.get(k, 0)} -> {now.get(k, 0)}" for k in keys))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
