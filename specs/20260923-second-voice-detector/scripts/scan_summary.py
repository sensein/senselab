"""Corpus-wide summary of what the incumbent diarizer currently counts. Counts only."""

from __future__ import annotations

import argparse
import json

import numpy as np


def main() -> int:
    """Print the corpus distribution of speaker counts and second-speaker durations."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("rows")
    ap.add_argument("--min-speech-s", type=float, default=20.0)
    args = ap.parse_args()

    rows = []
    with open(args.rows) as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    ok = [r for r in rows if "error" not in r]
    print(f"rows={len(rows)}  readable={len(ok)}  errors={len(rows) - len(ok)}")

    n = np.array([r["n_speakers"] for r in ok])
    sp = np.array([r["speech_s"] for r in ok])
    ov = np.array([r["overlap_s"] for r in ok])
    print("speaker counts: " + ", ".join(f"{k}:{int((n == k).sum())}" for k in sorted(set(n.tolist()))))
    print(f"  >=2: {int((n >= 2).sum())} = {100 * float((n >= 2).mean()):.2f}%")
    print(f"overlap_s > 0: {int((ov > 0).sum())} = {100 * float((ov > 0).mean()):.2f}%")
    long = sp >= args.min_speech_s
    print(f"\nrecordings with >= {args.min_speech_s}s of speech: {int(long.sum())}")
    print(f"  of those, >=2 speakers: {int((n[long] >= 2).sum())} = {100 * float((n[long] >= 2).mean()):.2f}%")

    by_task: dict[str, list[int]] = {}
    for r in ok:
        by_task.setdefault(str(r.get("task")), []).append(r["n_speakers"])
    print("\ntop tasks by two-speaker rate (>=200 recordings):")
    ranked = []
    for t, v in by_task.items():
        a = np.asarray(v)
        if a.size >= 200:
            ranked.append((float((a >= 2).mean()), t, a.size, int((a >= 2).sum())))
    for rate, t, size, hits in sorted(ranked, reverse=True)[:12]:
        print(f"  {t:44s} {hits:5d}/{size:6d} = {100 * rate:5.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
