"""Score the hinted-diarizer statistic per arm: what does forcing the count to two produce?

The incumbent honours ``num_speakers``. On the known case that hint recovers the true turn
boundary exactly, so the open question is what the same hint does to recordings that really
hold one voice. Three statistics are reported per arm, each one directly a candidate
operating point:

* ``minor_s`` — seconds the hinted run gives its smaller speaker.
* ``minor_share`` — that as a fraction of the hinted run's total attributed speech.
* ``sil_k2`` — cosine silhouette of the hinted labelling over the 2.0 s window embeddings.

Also reported: the same three for the *free* run, which is what the store already holds.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np


def arm_of(name: str) -> str:
    """Which calibration arm a row belongs to, from its staged filename."""
    for a in ("multi", "speech_single", "single"):
        if name.startswith(a + "__"):
            return a
    return "target"


def stats(run: dict[str, Any]) -> tuple[float, float]:
    """Smaller speaker's seconds and share for one diarizer run."""
    per = run.get("per_speaker_s") or {}
    vals = sorted(per.values())
    if len(vals) < 2:
        return 0.0, 0.0
    total = sum(vals)
    return vals[0], (vals[0] / total if total else 0.0)


def main() -> int:
    """Print per-arm distributions and a two-direction call table."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("rows", nargs="+")
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    for p in args.rows:
        with open(p) as fh:
            rows += [json.loads(line) for line in fh if line.strip()]

    by: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(arm_of(r.get("name", "")), []).append(r)

    print("== free run (what the store holds) ==")
    for arm in sorted(by):
        rs = by[arm]
        n2 = sum(1 for r in rs if (r.get("pyannote_free") or {}).get("n_speakers", 0) >= 2)
        ms = [
            stats(r.get("pyannote_free") or {})[0]
            for r in rs
            if (r.get("pyannote_free") or {}).get("n_speakers", 0) >= 2
        ]
        print(
            f"  {arm:14s} n={len(rs):3d}  free_n>=2: {n2:3d}"
            + (f"  minor_s median={np.median(ms):.2f} max={max(ms):.2f}" if ms else "")
        )

    print("\n== hinted run, num_speakers=2 ==")
    print(
        f"  {'arm':14s} {'n':>4s} {'minor_s p10':>12s} {'median':>9s} "
        f"{'p90':>9s} {'share med':>10s} {'sil_k2 med':>11s}"
    )
    for arm in sorted(by):
        rs = [r for r in by[arm] if (r.get("pyannote_k2") or {}).get("n_speakers", 0) >= 2]
        if not rs:
            print(f"  {arm:14s} {0:>4d}   (no hinted two-speaker result)")
            continue
        ms = np.array([stats(r["pyannote_k2"])[0] for r in rs])
        sh = np.array([stats(r["pyannote_k2"])[1] for r in rs])
        sil = [
            (r.get("candidates") or {}).get("pyannote_k2_labels", {}).get("silhouette")
            for r in rs
            if (r.get("candidates") or {}).get("pyannote_k2_labels", {}).get("valid")
        ]
        smed = f"{np.median(sil):.3f}" if sil else "-"
        print(
            f"  {arm:14s} {len(rs):>4d} {np.percentile(ms, 10):>12.2f} {np.median(ms):>9.2f} "
            f"{np.percentile(ms, 90):>9.2f} {np.median(sh):>10.3f} {smed:>11s}"
        )

    print("\n== calls: hinted minority seconds >= T ==")
    for t in (1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 12.0):
        cells = []
        for arm in sorted(by):
            rs = [r for r in by[arm] if (r.get("pyannote_k2") or {}).get("n_speakers", 0) >= 2]
            hit = sum(1 for r in rs if stats(r["pyannote_k2"])[0] >= t)
            cells.append(f"{arm}={hit}/{len(rs)}")
        print(f"  minor_s>={t:5.1f}  " + "  ".join(cells))

    print("\n== calls: hinted minority share >= T ==")
    for t in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        cells = []
        for arm in sorted(by):
            rs = [r for r in by[arm] if (r.get("pyannote_k2") or {}).get("n_speakers", 0) >= 2]
            hit = sum(1 for r in rs if stats(r["pyannote_k2"])[1] >= t)
            cells.append(f"{arm}={hit}/{len(rs)}")
        print(f"  share>={t:.2f}   " + "  ".join(cells))

    print("\n== the known case ==")
    for r in by.get("target", []):
        print(f"  free:   {r.get('pyannote_free', {}).get('per_speaker_s')}")
        print(f"  hinted: {r.get('pyannote_k2', {}).get('per_speaker_s')}")
        c = r.get("candidates") or {}
        print("  silhouettes: " + json.dumps({k: v.get("silhouette") for k, v in c.items() if v.get("valid")}))
        print(f"  best={r.get('best_candidate')} minority_runs={r.get('best_minority_runs_s')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
