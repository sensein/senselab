"""Measure what the viewer's axes are worth, over a built recording_vectors.parquet.

    uv run python specs/20260922-compact-recording-vectors/axis-discrimination.py PARQUET

Three questions, one pass over the file:

1. how short a key can label the subject and session axes without two ids sharing one;
2. how much each of the assignable columns discriminates, defined and derived in views.md;
3. what a default axis set costs and buys, as distinct polylines over the drawn corpus.

The parquet carries transcript text and marked PII. Nothing this script prints is read from a
recording: it prints counts, ranks and column names only.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
AXES_JS = REPO / "src/senselab/audio/workflows/triage/viewer/axes.js"

BANDS = 64
"""How many bands one value band resolves. At 62vh the band is about 446 px, so 64 puts them 7 px
apart -- comfortably above a 1 px line, and short of pretending a 1,527-category axis resolves
every category."""

REDUNDANT = 0.70
"""Two columns at or above this symmetric NMI are the same fact, and only one may take a slot."""

FIXED = ("participant", "task", "verdict")
"""Owner-directed, in this order."""

BOOKKEEPING = (
    "schema_version",
    "malformed_store_lines",
    "spans_unrowed_n",
)
"""Facts about the artefact rather than the recording. So are every ``_n`` count of how many
readings the store held, every ``_width``, and every ``gate_*_bound``; the counts that are the
fold's own decision are named in :data:`DECISION_COUNTS` and stay eligible."""

DECISION_COUNTS = (
    "flags_n",
    "pii_findings_n",
    "gate_applied_n",
    "gate_failed_n",
    "gate_flagging_n",
    "gate_undetermined_n",
    "separated_n",
    "secondary_extent_n",
)

EXCLUDED = ("release_ground",)
"""Ranked, and then excluded by hand with its reasons written in views.md."""


def catalogue() -> tuple[list[dict], dict]:
    """The viewer's own axis catalogue and category orderings, read out of axes.js by node.

    Returns:
        The assignable catalogue entries, and the declared orderings by column name.
    """
    cols = json.loads(
        subprocess.check_output(
            [
                "node",
                "-e",
                "var S=require(process.argv[1]);console.log(JSON.stringify(S.CATALOGUE.map("
                "function(c){return {name:c.name,kind:c.kind,group:c.group,"
                "assignable:c.assignable,sizeOf:c.sizeOf};})));",
                str(AXES_JS),
            ]
        )
    )
    orderings = json.loads(
        subprocess.check_output(
            ["node", "-e", "console.log(JSON.stringify(require(process.argv[1]).ORDERINGS));", str(AXES_JS)]
        )
    )
    return [c for c in cols if c["assignable"]], orderings


def eligible(name: str) -> bool:
    """Whether a column may hold a default slot: a fact about the recording or the decision."""
    if name in EXCLUDED:
        return False
    if name in DECISION_COUNTS:
        return True
    return not (name in BOOKKEEPING or name.endswith(("_n", "_width", "_bound")))


def simpson(counts: collections.Counter) -> float:
    """One minus the sum of squared shares: the chance two draws land in different bands."""
    tot = sum(counts.values())
    return 0.0 if tot == 0 else 1.0 - sum((n / tot) ** 2 for n in counts.values())


def entropy(counts: collections.Counter) -> float:
    """Shannon entropy in nats."""
    tot = sum(counts.values())
    return 0.0 if tot == 0 else -sum((n / tot) * math.log(n / tot) for n in counts.values() if n)


def mutual_information(x: list, y: list) -> float:
    """Mutual information of two labellings, in nats."""
    cx, cy, j = collections.Counter(x), collections.Counter(y), collections.Counter(zip(x, y))
    tot = len(x)
    return sum((n / tot) * math.log((n / tot) / ((cx[p] / tot) * (cy[q] / tot))) for (p, q), n in j.items())


def band_of(col: dict, values: list, orderings: dict, n_rows: int) -> tuple[list, dict]:
    """The band each row falls in on this axis, None where absent, plus the column's shape.

    Args:
        col: The catalogue entry.
        values: The per-row value the axis reads.
        orderings: The declared category orders.
        n_rows: How many rows the file holds.

    Returns:
        One band index or None per row, and a dict of coverage, distinct values and bands used.
    """
    present = [v for v in values if v is not None]
    info = {"present": len(present), "coverage": len(present) / n_rows, "distinct": 0, "bands": 0}
    if not present:
        return [None] * n_rows, info
    if col["kind"] in ("numeric", "count"):
        lo, hi = min(present), max(present)
        info["distinct"] = len(set(present))
        if hi == lo:
            return [None if v is None else 0 for v in values], info | {"bands": 1}
        idx = [None if v is None else min(BANDS - 1, int((v - lo) / (hi - lo) * BANDS)) for v in values]
        return idx, info | {"bands": len({i for i in idx if i is not None})}
    counts = collections.Counter(present)
    order = orderings.get(col["name"])
    if order:
        cats = [t for t in order if t in counts] + sorted(
            (t for t in counts if t not in order), key=lambda t: (-counts[t], t)
        )
    else:
        cats = sorted(counts, key=lambda t: (-counts[t], t))
    info["distinct"] = len(cats)
    per = max(1, math.ceil(len(cats) / BANDS))
    pos = {t: i // per for i, t in enumerate(cats)}
    idx = [None if v is None else pos[v] for v in values]
    return idx, info | {"bands": len(set(pos.values()))}


def identity_keys(table: pq.pa.Table) -> None:
    """Print how short a prefix labels every subject and session without a collision."""
    for column, prefix in (("participant", "sub-"), ("session", "ses-")):
        ids = sorted({v for v in table.column(column).to_pylist() if v is not None})
        print(
            f"\n{column}: {len(ids)} distinct, lengths {sorted({len(i) for i in ids})}, all on {prefix!r}: "
            f"{all(i.startswith(prefix) for i in ids)}"
        )
        for n in range(2, 13):
            keys = collections.Counter(i[len(prefix) : len(prefix) + n] for i in ids)
            dup = {k: c for k, c in keys.items() if c > 1}
            expected = len(ids) * (len(ids) - 1) / 2 / (16.0**n)
            print(
                f"   n={n:2d}  colliding keys {len(dup):3d}  ids involved {sum(dup.values()):3d}"
                f"   birthday expectation {expected:8.4f}"
            )


def main(argv: list[str] | None = None) -> int:
    """Run all three measurements against one parquet.

    Args:
        argv: The command line, or None to read ``sys.argv``.

    Returns:
        0.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("parquet", type=Path)
    ap.add_argument("--top", type=int, default=40, help="How many rows of the ranking to print.")
    args = ap.parse_args(argv)

    cols, orderings = catalogue()
    schema = pq.read_schema(args.parquet)
    written = {f.name for f in schema}
    need = sorted(
        {(c["sizeOf"] if c["sizeOf"] and c["name"].endswith(".size") else c["name"]) for c in cols}
        | {"verdict", "release"} & written
    )
    need = [n for n in need if n in written]
    table = pq.read_table(args.parquet, columns=need)
    n_rows = table.num_rows
    print(f"{args.parquet}: {n_rows} rows, {len(schema)} columns, {len(cols)} of them assignable to an axis")

    print("\n--- 1. how short a key can name a subject ---")
    identity_keys(table)

    raw = {n: table.column(n).to_pylist() for n in need}

    def values_of(col: dict) -> list:
        """The per-row value the axis reads, as SchemaAxes.readValue would."""
        if col["sizeOf"] and col["name"].endswith(".size"):
            return [None if v is None else len(v) for v in raw[col["sizeOf"]]]
        return raw[col["name"]]

    decision = [f"{v}|{r}" for v, r in zip(raw["verdict"], raw["release"])]
    h_decision = entropy(collections.Counter(decision))

    banding: dict[str, list] = {}
    ranked = []
    for col in cols:
        idx, info = band_of(col, values_of(col), orderings, n_rows)
        banding[col["name"]] = idx
        gini = simpson(collections.Counter(i for i in idx if i is not None))
        labelled = ["absent" if i is None else i for i in idx]
        sep = info["coverage"] ** 2 * gini
        rel = mutual_information(labelled, decision) / h_decision
        ranked.append(
            dict(
                info,
                name=col["name"],
                group=col["group"],
                sep=sep,
                rel=rel,
                disc=sep * rel,
                eligible=eligible(col["name"]),
            )
        )
    ranked.sort(key=lambda r: -r["disc"])

    print(
        f"\n--- 2. discrimination over all {len(ranked)} assignable columns "
        f"(H(verdict, release) = {h_decision:.4f} nats) ---"
    )
    print(f"{'#':>3} {'column':<46} {'group':<16} {'cov%':>6} {'bands':>6} {'sep':>8} {'rel':>7} {'disc':>8}")
    for i, r in enumerate(ranked[: args.top], 1):
        mark = "" if r["eligible"] else "   (not eligible for a default)"
        print(
            f"{i:3d} {r['name']:<46} {r['group']:<16} {100 * r['coverage']:6.1f} {r['bands']:6d} "
            f"{r['sep']:8.4f} {r['rel']:7.3f} {r['disc']:8.4f}{mark}"
        )
    print(f"    ... {len(ranked) - args.top} further columns, every one below {ranked[args.top]['disc']:.4f}")

    def nmi(a: str, b: str) -> float:
        """Symmetric normalised mutual information between two banded columns."""
        x = ["absent" if i is None else i for i in banding[a]]
        y = ["absent" if i is None else i for i in banding[b]]
        hx, hy = entropy(collections.Counter(x)), entropy(collections.Counter(y))
        return 1.0 if hx + hy == 0 else 2 * mutual_information(x, y) / (hx + hy)

    def polylines(axes: list[str]) -> tuple[int, int]:
        """Distinct banded polylines the set draws, and the largest bundle sharing one."""
        k = collections.Counter(zip(*[banding[a] for a in axes]))
        return len(k), max(k.values())

    print("\n--- 3. the seven free slots, walked down the ranking ---")
    by = {r["name"]: r for r in ranked}
    chosen = list(FIXED)
    for r in ranked:
        if len(chosen) >= 10:
            break
        if r["name"] in chosen or not r["eligible"]:
            continue
        worst, against = max((nmi(r["name"], c), c) for c in chosen)
        if worst >= REDUNDANT:
            print(f"    {r['name']:<40} disc {r['disc']:.4f}  skipped: NMI {worst:.3f} with {against}")
            continue
        chosen.append(r["name"])
        print(f"    {r['name']:<40} disc {r['disc']:.4f}  CHOSEN (worst NMI {worst:.3f}, with {against})")

    current = list(FIXED) + [
        "duration_s",
        "conformance_airway",
        "conformance_speech",
        "gate_failed_n",
        "flags_n",
        "pii_findings_n",
        "release",
    ]
    floor, floor_bundle = polylines([c for c in current if c in banding])
    print(f"\n  resolution floor, from the schema-3 defaults: {floor} polylines, largest bundle {floor_bundle}")
    got, bundle = polylines(chosen)
    print(f"  the greedy set draws {got} polylines, largest bundle {bundle}")
    if got < floor or bundle > floor_bundle:
        swaps = []
        for out in chosen[3:]:
            keep = [c for c in chosen if c != out]
            for r in ranked:
                if r["name"] in keep or not r["eligible"] or max(nmi(r["name"], c) for c in keep) >= REDUNDANT:
                    continue
                pp, bb = polylines(keep + [r["name"]])
                if pp >= floor and bb <= floor_bundle:
                    swaps.append(
                        (sum(by[c]["disc"] for c in keep[3:] + [r["name"]]), pp, bb, out, r["name"], keep + [r["name"]])
                    )
        swaps.sort(key=lambda s: (-s[0], -s[1]))
        print("  swaps that clear the floor, best total discrimination first:")
        for d, pp, bb, out, name, _ in swaps[:6]:
            print(f"    -{out:<20} +{name:<38} disc {d:.4f}  {pp:6d} polylines, bundle {bb:3d}")
        chosen = swaps[0][5]

    print(f"\n  chosen: {chosen}")
    for label, axes in (("schema 3", current), ("proposed", chosen)):
        axes = [a for a in axes if a in banding]
        pp, bb = polylines(axes)
        whole = sum(1 for i in range(n_rows) if all(banding[a][i] is not None for a in axes))
        print(
            f"    {label:<9} {pp:6d} polylines, largest bundle {bb:4d}, sum disc "
            f"{sum(by[a]['disc'] for a in axes[3:]):.4f}, sum sep {sum(by[a]['sep'] for a in axes[3:]):.4f}, "
            f"{whole} of {n_rows} lines whole ({100 * whole / n_rows:.1f}%)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
