"""Print the diarization sidecars of one run directory. Timings and labels only."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def main() -> int:
    """Print every ``*_diarization.npz`` under one run directory."""
    d = Path(sys.argv[1]) / "run" / "derivatives"
    for p in sorted(d.glob("*_diarization.npz")):
        with np.load(p, allow_pickle=False) as z:
            starts = np.asarray(z["starts"], dtype=float)
            ends = np.asarray(z["ends"], dtype=float)
            spk = np.asarray(z["speakers"]).astype(str)
        labs = sorted(set(spk.tolist()))
        print(f"{p.name}: n_segments={len(starts)} n_speakers={len(labs)} labels={labs}")
        for s, e, k in zip(starts, ends, spk, strict=False):
            print(f"    [{s:7.3f}, {e:7.3f}] {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
