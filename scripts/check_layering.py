"""Does a real run's artifact tree obey the layered plan? Structure only, not numbers."""

import json
import pathlib
import sys

import pandas as pd

R = pathlib.Path(sys.argv[1])
AXES = ("speech_presence", "speaker", "asr", "background_mask")
ok: list[str] = []
bad: list[str] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    """Record one invariant. ``detail`` is printed on failure, so it should name the offending value."""
    (ok if passed else bad).append(f"{name}{': ' + detail if detail else ''}")


# ── L1 measures: no axis, no fold, no cross-pass evaluation ────────────
l1 = R / "L1"
# Whole-stem match, not substring: `pyannote_speaker_diarization_community_1.parquet` contains the
# axis name "speaker" and is correctly named after its producer.
axis_named = [p for p in l1.rglob("*") if p.is_file() and p.stem in AXES]
check("L1 has no axis-named artifact", not axis_named, ", ".join(p.name for p in axis_named[:3]))
check("L1 has no stability/ (cross-pass eval is L2's)", not (l1 / "stability").exists())

sig = sorted((l1 / "signals").glob("*.parquet"))
check("L1/signals/ is populated", bool(sig), f"{len(sig)} signals")
if sig:
    d = pd.read_parquet(sig[0])
    check(
        "signal rows carry perturbation as a column",
        "perturbation" in d.columns,
        f"{sig[0].name}: {sorted(c for c in d.columns if c in ('perturbation', 'pass'))}",
    )
check("L1 declares its transforms", (l1 / "perturbations.json").exists() or (l1 / "routes.json").exists())

# ── L2 decides: one round tree, axes keyed by axis alone ───────────────
trees = [p.name for p in (R / "L2").iterdir() if p.is_dir()] if (R / "L2").exists() else []
check("L2 has one round tree", "rounds" not in trees, f"dirs={sorted(trees)}")

rounds = sorted((R / "L2" / "round").glob("*")) if (R / "L2" / "round").exists() else []
check("L2/round/<n>/ exists", bool(rounds), f"{len(rounds)} rounds")
if rounds:
    est = sorted((rounds[-1] / "estimates").glob("*.parquet"))
    check("last round has estimates", bool(est), ", ".join(p.stem for p in est))
    for p in est:
        d = pd.read_parquet(p)
        leaked = [c for c in d.columns if c in ("pass", "perturbation")]
        check(f"estimates/{p.stem} not keyed by pass", not leaked, str(leaked))

# ── round bookkeeping: the two invariants my first suite missed ────────
for rd in rounds:
    idx = int(rd.name)
    est_r = sorted((rd / "estimates").glob("*.parquet"))
    check(f"round {idx} has every axis", {p.stem for p in est_r} == set(AXES), str(sorted(p.stem for p in est_r)))
    for p in est_r:
        d = pd.read_parquet(p)
        if "round" not in d.columns:
            continue
        stamped = sorted(int(v) for v in d["round"].dropna().unique())
        check(f"round {idx}/{p.stem} rows say round {idx}", stamped in ([idx], []), str(stamped))

# ── final extracts the last round ─────────────────────────────────────
fin = sorted((R / "final" / "estimates").glob("*.parquet")) if (R / "final" / "estimates").exists() else []
check("final/estimates/ present", bool(fin), ", ".join(p.stem for p in fin))
check("all four axes reach final/", {p.stem for p in fin} == set(AXES), str(sorted(p.stem for p in fin)))
if rounds and fin:
    # Content, not bytes: `extract_final_estimates` re-serializes via pq.write_table(pq.read_table(..)),
    # so row-group layout and metadata may differ while the rows are identical. A byte check reports a
    # faithful extraction as a violation.
    src = rounds[-1] / "estimates"
    same = []
    for p in fin:
        o = src / p.name
        same.append(o.exists() and pd.read_parquet(o).equals(pd.read_parquet(p)))
    check("final/ extracts the last round (content)", all(same), f"{sum(same)}/{len(fin)} match")

# ── cross-artifact: the index ranks what was written ──────────────────
dj = R / "L2" / "disagreements.json"
if dj.exists():
    d = json.loads(dj.read_text())
    listed = set(d["totals"]["rows_by_axis"])
    check("background_mask reaches the index", "background_mask" in listed, str(sorted(listed)))
    entries = d["entries"]
    if entries:
        # Follow the entry's OWN pointer. The index is pre-adaptive by design (the adaptive stage
        # consumes it), so an entry describes the round it names — comparing it against final/ reported
        # a self-consistent artifact as violating, three times.
        mism = []
        cache: dict = {}
        for e in entries[:40]:
            ptr = R / e["parquet"]
            if not ptr.is_file():
                mism.append(f"{e['axis']}@{e['start']}: pointer {e['parquet']} missing")
                continue
            df = cache.setdefault(e["parquet"], pd.read_parquet(ptr))
            row = df[(df["start"] == e["start"]) & (df["end"] == e["end"])]
            if row.empty:
                mism.append(f"{e['axis']}@{e['start']} absent from parquet")
                continue
            pq_t, ix_t = row.iloc[0]["triage_score"], e["triage_score"]
            if (pq_t is None) != (ix_t is None) or (
                pq_t is not None and ix_t is not None and abs(float(pq_t) - float(ix_t)) > 1e-6
            ):
                mism.append(f"{e['axis']}@{e['start']}: parquet={pq_t} index={ix_t}")
        check("index triage matches the parquet it points at", not mism, "; ".join(mism[:3]))
    print(f"\nhigh_uncertainty_rate: {d['totals']['high_uncertainty_rate']:.4f} (was 0.9941)")

print(f"\n=== {len(ok)} held, {len(bad)} violated ===")
for line in ok:
    print(f"  ok    {line}")
for line in bad:
    print(f"  FAIL  {line}")
