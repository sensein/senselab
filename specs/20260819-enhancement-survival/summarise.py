"""Render bench.py's results.json as survival tables, one per reader."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

EVENT_ORDER = ["mouth", "breath_1", "breath_2", "cough_1", "cough_2", "speech", "verified_empty"]


def cell(rows: List[Dict[str, Any]]) -> str:
    """Return the peak score across a cell's labels, or a failure marker.

    Args:
        rows: Rows sharing (snr, enhancer, scope, reader).

    Returns:
        A formatted score, ``err`` when the stage failed, or ``-``.
    """
    if not rows:
        return "-"
    bad = [r for r in rows if r["score"] is None]
    if bad and len(bad) == len(rows):
        return "err"
    scores = [r["score"] for r in rows if r["score"] is not None]
    return f"{max(scores):.3f}" if scores else "-"


def main() -> int:
    """Print one table per reader: enhancer x event, at each SNR.

    Returns:
        Process exit status.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    args = ap.parse_args()
    data = json.loads(args.results.read_text())

    grouped: Dict[Tuple[Optional[float], str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    enhancers: List[str] = []
    for r in data:
        grouped[(r["snr"], r["enhancer"], r["reader"], r["scope"])].append(r)
        if r["enhancer"] not in enhancers:
            enhancers.append(r["enhancer"])

    snrs = sorted({r["snr"] for r in data}, key=lambda s: (s is not None, -(s or 0)))
    for reader in ("yamnet", "hear"):
        present = [s for s in EVENT_ORDER if any(k[3] == s and k[2] == reader for k in grouped)]
        if not present:
            continue
        for snr in snrs:
            label = "as captured" if snr is None else f"{snr:+.0f} dB SNR"
            print(f"\n### {reader} — {label}\n")
            print("| enhancer | " + " | ".join(present) + " |")
            print("|" + "---|" * (len(present) + 1))
            for enh in enhancers:
                cells = [cell(grouped.get((snr, enh, reader, s), [])) for s in present]
                print(f"| {enh} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
