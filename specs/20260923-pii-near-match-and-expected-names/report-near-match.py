"""Turn the reduction's JSON tables into the tables the derivation quotes. Read-only, offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _rate(numerator: int, denominator: int) -> str:
    """A count as a percentage of a denominator.

    Args:
        numerator: The count.
        denominator: The denominator.

    Returns:
        The rate, or ``"-"`` when the denominator is zero.
    """
    return "-" if not denominator else f"{100.0 * numerator / denominator:.2f}%"


def main() -> None:
    """Print the near-match fit, the withhold census and the expected-names census."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tables", type=Path)
    args = parser.parse_args()
    data: dict[str, Any] = json.loads(args.tables.read_text())

    totals = data["finding_class_totals"]
    admitted = data["admitted_by_class_rule"]
    donor = data["donor_admitted_by_rule"]
    on, off, mixed = totals.get("on_stimulus", 0), totals.get("off_stimulus", 0), totals.get("mixed", 0)
    eligible = totals.get("donor_eligible", 0)

    print(f"runs                     {data['n_runs']}")
    print(f"findings on stimulus     {on}")
    print(f"findings off stimulus    {off}")
    print(f"findings mixed           {mixed}")
    print(f"findings donor-eligible  {eligible}")
    print()
    print("rule            admit(on)  yield   admit(off)  false   admit(donor)  false")
    for rule in data["rules"]:
        a_on = admitted.get(f"on_stimulus|{rule}", 0)
        a_off = admitted.get(f"off_stimulus|{rule}", 0)
        a_donor = donor.get(rule, 0)
        print(
            f"{rule:<15} {a_on:>8}  {_rate(a_on, on):>7}  "
            f"{a_off:>9}  {_rate(a_off, off):>6}  {a_donor:>11}  {_rate(a_donor, eligible):>6}"
        )

    print()
    print("substitution distance by expected-token length (the ASR's own spelling variation)")
    rows: dict[int, dict[int, int]] = {}
    for key, count in data["substitution_len_distance"].items():
        length, distance = (int(part) for part in key.split("|"))
        rows.setdefault(length, {})[distance] = count
    print("len    n     d=0    d=1    d=2    d=3    d>=4   cum(d<=1)  cum(d<=2)")
    for length in sorted(rows):
        row = rows[length]
        total = sum(row.values())
        far = sum(count for distance, count in row.items() if distance >= 4)
        le1 = sum(count for distance, count in row.items() if distance <= 1)
        le2 = sum(count for distance, count in row.items() if distance <= 2)
        print(
            f"{length:>3} {total:>6} {row.get(0, 0):>6} {row.get(1, 0):>6} {row.get(2, 0):>6} "
            f"{row.get(3, 0):>6} {far:>6}   {_rate(le1, total):>8}  {_rate(le2, total):>8}"
        )

    print()
    print("REDACT ground, corpus-wide")
    grounds: dict[str, int] = {}
    for counter in data["redact_ground_by_family"].values():
        for ground, count in counter.items():
            grounds[ground] = grounds.get(ground, 0) + count
    for ground, count in sorted(grounds.items(), key=lambda item: -item[1]):
        print(f"  {ground:<24} {count:>6}")

    print()
    print("findings and in_stimulus by family (top 20 by findings)")
    families = sorted(data["per_family"].items(), key=lambda item: -item[1].get("findings", 0))[:20]
    print(f"{'family':<34} {'runs':>5} {'redact':>6} {'find':>6} {'true':>6} {'false':>6} {'null':>6} {'exempt':>6}")
    for family, counter in families:
        tri = data["in_stimulus_by_family"].get(family, {})
        print(
            f"{family:<34} {counter.get('runs', 0):>5} {counter.get('redact_runs', 0):>6} "
            f"{counter.get('findings', 0):>6} {tri.get('True', 0):>6} {tri.get('False', 0):>6} "
            f"{tri.get('null', 0):>6} {counter.get('exempt', 0):>6}"
        )

    print()
    print("cinderella-story findings by distance to the declared cast")
    cast: dict[int, int] = {}
    by_category: dict[str, dict[int, int]] = {}
    for key, count in data["cinderella_cast_distance"].items():
        category, n_words, distance = key.split("|")
        cast[int(distance)] = cast.get(int(distance), 0) + count
        by_category.setdefault(category, {})[int(distance)] = (
            by_category.setdefault(category, {}).get(int(distance), 0) + count
        )
    total = sum(cast.values())
    print(f"  total {total}")
    for distance in sorted(cast):
        print(f"  d={distance:<3} {cast[distance]:>6}  {_rate(cast[distance], total)}")
    for category, row in sorted(by_category.items(), key=lambda item: -sum(item[1].values())):
        near = sum(count for distance, count in row.items() if distance <= 1)
        print(f"  {category:<22} n={sum(row.values()):>5}  d<=1 {near:>5}  {_rate(near, sum(row.values()))}")

    print()
    print("withhold census: REDACT outcomes corpus-wide")
    outcomes: dict[str, int] = {}
    for counter in data["per_family"].values():
        for key, count in counter.items():
            if key.startswith("redact_") and key != "redact_runs":
                outcomes[key] = outcomes.get(key, 0) + count
            if key.startswith("llm_"):
                outcomes[key] = outcomes.get(key, 0) + count
    for key, count in sorted(outcomes.items(), key=lambda item: -item[1]):
        print(f"  {key:<24} {count:>6}")


if __name__ == "__main__":
    main()
