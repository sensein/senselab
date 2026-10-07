"""Reduce ``detect.py`` rows to the two-direction table the recommendation rests on.

Rows are grouped by the arm encoded in the filename prefix (``multi__``, ``single__``) with
everything else treated as the known case. For each candidate partition method this prints
the silhouette distribution per arm and, for every threshold on the grid, how many of each
arm would be called two-speaker.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def arm_of(name: str) -> str:
    """Which calibration arm a row belongs to, from its staged filename."""
    if name.startswith("multi__"):
        return "multi"
    if name.startswith("single__"):
        return "single"
    return "target"


def main() -> int:
    """Print per-arm silhouette distributions and a threshold table."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("rows", nargs="+")
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.30])
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    for p in args.rows:
        with open(p) as fh:
            rows += [json.loads(line) for line in fh if line.strip()]

    by_arm: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_arm.setdefault(arm_of(r.get("name", "")), []).append(r)

    print("== rows per arm ==")
    for arm, rs in sorted(by_arm.items()):
        usable = [r for r in rs if r.get("candidates")]
        dur = [r.get("duration_s", 0) for r in rs]
        print(
            f"  {arm:7s} n={len(rs):3d} usable={len(usable):3d} "
            f"median_duration_s={np.median(dur) if dur else 0:.1f} "
            f"pyannote_free_n>=2: {sum(1 for r in rs if (r.get('pyannote_free') or {}).get('n_speakers', 0) >= 2)}"
        )

    methods = sorted({m for r in rows for m in (r.get("candidates") or {})})
    print("\n== silhouette by arm (median [p10, p90]) ==")
    header = f"{'method':22s}" + "".join(f"{a:>26s}" for a in sorted(by_arm))
    print(header)
    for m in methods:
        line = f"{m:22s}"
        for arm in sorted(by_arm):
            vals = [
                r["candidates"][m]["silhouette"]
                for r in by_arm[arm]
                if (r.get("candidates") or {}).get(m, {}).get("valid")
            ]
            if not vals:
                line += f"{'-':>26s}"
            else:
                a = np.asarray(vals)
                line += (
                    f"{f'{np.median(a):.3f} [{np.percentile(a, 10):.3f},{np.percentile(a, 90):.3f}] n={len(a)}':>26s}"
                )
        print(line)

    print("\n== calls at each silhouette threshold: multi caught / single false-positive / target ==")
    for m in methods:
        print(f"  {m}")
        for th in args.thresholds:
            cells = {}
            for arm in sorted(by_arm):
                vals = [
                    r["candidates"][m]["silhouette"]
                    for r in by_arm[arm]
                    if (r.get("candidates") or {}).get(m, {}).get("valid")
                ]
                cells[arm] = (sum(1 for v in vals if v >= th), len(vals))
            parts = " ".join(f"{a}={c}/{n}" for a, (c, n) in sorted(cells.items()))
            print(f"    sil>={th:.2f}  {parts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
