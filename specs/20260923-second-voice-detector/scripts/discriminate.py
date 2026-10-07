"""For every candidate statistic, how well does it separate the two arms?

Positives are the ``multi`` arm — recordings the incumbent already calls two-speaker with at
least 20 s of speech. Negatives are ``speech_single`` — connected-speech recordings of the
same length that it calls one-speaker. AUC is Mann-Whitney, so 0.5 is chance and a statistic
below about 0.7 cannot carry an operating point on 43,000 recordings whatever threshold is
chosen. The known case's own value is printed beside each statistic, with the fraction of
negatives it would have to drag along to catch it.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Optional

import numpy as np

ARMS = ("multi", "speech_single", "single")


def arm_of(name: str) -> str:
    """Which calibration arm a row belongs to, from its staged filename."""
    for a in ARMS:
        if name.startswith(a + "__"):
            return a
    return "target"


def _k2_minor(run: dict[str, Any]) -> Optional[float]:
    """Seconds the hinted run gives its smaller speaker."""
    per = run.get("per_speaker_s") or {}
    vals = sorted(per.values())
    return vals[0] if len(vals) >= 2 else None


def _k2_share(run: dict[str, Any]) -> Optional[float]:
    """The smaller speaker's share of the hinted run's attributed speech."""
    per = run.get("per_speaker_s") or {}
    vals = sorted(per.values())
    return (vals[0] / sum(vals)) if len(vals) >= 2 and sum(vals) else None


def extract(row: dict[str, Any]) -> dict[str, Optional[float]]:
    """Every candidate statistic for one recording."""
    out: dict[str, Optional[float]] = {}
    k2 = row.get("pyannote_k2") or {}
    out["k2_minor_s"] = _k2_minor(k2)
    out["k2_minor_share"] = _k2_share(k2)
    for name, sc in (row.get("candidates") or {}).items():
        if not sc.get("valid"):
            continue
        out[f"{name}.silhouette"] = sc.get("silhouette")
        out[f"{name}.minority_share"] = sc.get("minority_share")
        out[f"{name}.cos_between"] = sc.get("cos_between_centroids")
        out[f"{name}.sep_margin"] = sc.get("separation_margin")
    return out


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC of ``pos`` over ``neg``, ties counted as half."""
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (pos.size * neg.size))


def main() -> int:
    """Print an AUC table over every candidate statistic, with the known case beside it."""
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
    print("rows per arm: " + ", ".join(f"{k}={len(v)}" for k, v in sorted(by.items())))

    vals: dict[str, dict[str, list[float]]] = {}
    target: dict[str, float] = {}
    for arm, rs in by.items():
        for r in rs:
            for k, v in extract(r).items():
                if v is None:
                    continue
                if arm == "target":
                    target[k] = v
                else:
                    vals.setdefault(k, {}).setdefault(arm, []).append(v)

    print(
        f"\n{'statistic':34s}{'AUC':>7s}{'n+':>5s}{'n-':>5s}"
        f"{'pos median':>12s}{'neg median':>12s}{'target':>9s}{'neg>=target':>13s}"
    )
    scored = []
    for k in sorted(vals):
        pos = np.asarray(vals[k].get("multi", []), dtype=float)
        neg = np.asarray(vals[k].get("speech_single", []), dtype=float)
        if pos.size < 5 or neg.size < 5:
            continue
        a = auc(pos, neg)
        t = target.get(k)
        # Half these statistics point the other way — a lower centroid cosine means *more*
        # separation — so the tail that would have to be admitted to catch the known case is
        # whichever side the AUC says the positives sit on.
        if t is None:
            drag = "-"
        elif a >= 0.5:
            drag = f"{float((neg >= t).mean()):.2f}"
        else:
            drag = f"{float((neg <= t).mean()):.2f}"
        scored.append((abs(a - 0.5), k, a, pos, neg, t, drag))
    for _, k, a, pos, neg, t, drag in sorted(scored, reverse=True):
        tv = f"{t:.3f}" if t is not None else "-"
        print(
            f"{k:34s}{a:>7.3f}{pos.size:>5d}{neg.size:>5d}"
            f"{np.median(pos):>12.3f}{np.median(neg):>12.3f}{tv:>9s}{drag:>13s}"
        )
    print("\nAUC is of the positive arm over the negative arm; 0.5 is chance.")
    print("'neg>=target' is the fraction of true single-speaker recordings that score at")
    print("least as high as the known case — the false-positive rate of any rule that catches it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
