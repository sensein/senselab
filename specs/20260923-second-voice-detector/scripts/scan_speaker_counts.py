"""Read every ``<stream>_diarization.npz`` in a triage run tree and emit one row per recording.

Emits counts and timings only: stem, task token, speaker count, segment count, speech
seconds, overlap seconds. No transcript text and no audio leaves this script.

The npz sidecar is what PREPROCESS wrote (``starts``/``ends``/``speakers``/``duration_s``),
so this recovers the diarizer's own verdict without re-running it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Optional

import numpy as np

_TASK = re.compile(r"_(task-[^_]+)")
_TIMESTAMP = re.compile(r"_\d{8}-\d{6}(?:-\d+)?$")


def stem_of(run_root: Path) -> str:
    """Directory name with the trailing run timestamp removed."""
    return _TIMESTAMP.sub("", run_root.name)


def task_of(stem: str) -> Optional[str]:
    """The ``task-...`` token of a stem, or None."""
    m = _TASK.search(stem)
    return m.group(1) if m else None


def _overlap_seconds(starts: np.ndarray, ends: np.ndarray, speakers: np.ndarray) -> float:
    """Seconds where two distinct speaker labels are simultaneously active."""
    if len(starts) < 2:
        return 0.0
    edges = np.unique(np.concatenate([starts, ends]))
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        if hi <= lo:
            continue
        mid = 0.5 * (lo + hi)
        active = {str(s) for s, a, b in zip(speakers, starts, ends, strict=False) if a <= mid < b}
        if len(active) >= 2:
            total += float(hi - lo)
    return total


def row_for(npz_path: Path) -> Optional[dict[str, Any]]:
    """One row for one diarization sidecar, or None if it cannot be read."""
    run_root = npz_path.parent.parent.parent
    try:
        with np.load(npz_path, allow_pickle=False) as z:
            starts = np.asarray(z["starts"], dtype=float)
            ends = np.asarray(z["ends"], dtype=float)
            speakers = np.asarray(z["speakers"]).astype(str)
            duration_s = float(z["duration_s"]) if "duration_s" in z else float("nan")
    except Exception as exc:  # noqa: BLE001 — an unreadable sidecar is a row, not a crash
        return {"run_root": str(run_root), "error": repr(exc)[:200]}
    stem = stem_of(run_root)
    labels = sorted(set(speakers.tolist()))
    return {
        "run_root": str(run_root),
        "stem": stem,
        "task": task_of(stem),
        "stream": npz_path.name.rsplit("_diarization", 1)[0],
        "duration_s": duration_s,
        "n_segments": int(len(starts)),
        "n_speakers": len(labels),
        "speech_s": float(np.sum(ends - starts)) if len(starts) else 0.0,
        "overlap_s": _overlap_seconds(starts, ends, speakers),
    }


def main() -> int:
    """Scan a run tree and write one JSON row per diarization sidecar."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", help="campaign out/ directory")
    ap.add_argument("--stream", default="enhanced")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    root = Path(args.root)
    pattern = f"*/*/*/run/derivatives/{args.stream}_diarization.npz"
    paths = sorted(root.glob(pattern))
    print(f"found {len(paths)} sidecars under {root}", file=sys.stderr, flush=True)

    written = 0
    with open(args.out, "w") as fh, ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(row_for, paths, chunksize=64):
            if row is None:
                continue
            fh.write(json.dumps(row) + "\n")
            written += 1
            if written % 5000 == 0:
                print(f"  {written} rows", file=sys.stderr, flush=True)
    print(f"wrote {written} rows to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
