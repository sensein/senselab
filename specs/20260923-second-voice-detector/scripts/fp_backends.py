"""Per-arm false-positive and recall table for a backend run over the calibration sample.

A backend "calls two" is not one rule but a family: report two or more labels, *and* give
the smaller one at least T seconds. T = 0 is the bare count. The table walks T so the
operating point is visible rather than chosen here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ARMS = ("multi", "speech_single", "single")
THRESHOLDS = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0)


def arm_of(name: str) -> str:
    """Which calibration arm a row belongs to, from its staged filename."""
    for a in ARMS:
        if name.startswith(a + "__"):
            return a
    return "target"


def minority_s(row: dict[str, Any]) -> float:
    """Seconds held by the backend's smallest label, or 0 when it reported one speaker."""
    per: dict[str, float] = {}
    for s, e, lab in row.get("segments", []):
        per[lab] = per.get(lab, 0.0) + (e - s)
    vals = sorted(per.values())
    return vals[0] if len(vals) >= 2 else 0.0


def main() -> int:
    """Print, per backend, the per-arm call rate at each minority-seconds threshold."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("rows", nargs="+")
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    for p in args.rows:
        with open(p) as fh:
            rows += [json.loads(line) for line in fh if line.strip()]

    by_backend: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for r in rows:
        name = Path(str(r.get("audio", ""))).name
        by_backend.setdefault(str(r.get("backend")), {}).setdefault(arm_of(name), []).append(r)

    for backend, arms in sorted(by_backend.items()):
        print(f"\n== {backend} ==")
        failed = sum(1 for rs in arms.values() for r in rs if r.get("status") != "ok")
        walls = [r["wall_s"] for rs in arms.values() for r in rs if r.get("status") == "ok" and r.get("wall_s")]
        durs = [r["duration_s"] for rs in arms.values() for r in rs if r.get("status") == "ok" and r.get("duration_s")]
        if walls:
            rt = np.array(durs) / np.array(walls)
            print(
                f"  n={sum(len(v) for v in arms.values())} failed={failed} "
                f"median wall {np.median(walls):.1f}s for median {np.median(durs):.1f}s audio "
                f"({np.median(rt):.2f}x realtime)"
            )
        for arm in sorted(arms):
            ok = [r for r in arms[arm] if r.get("status") == "ok"]
            if not ok:
                continue
            ms = np.array([minority_s(r) for r in ok])
            cells = " ".join(
                f"T>={t:<4.1f}:{int((ms >= t).sum() if t > 0 else (ms > 0).sum())}/{len(ok)}" for t in THRESHOLDS
            )
            print(f"  {arm:14s} {cells}")
    print("\nT is the minority label's seconds; T>=0.0 is the bare 'reported two or more'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
