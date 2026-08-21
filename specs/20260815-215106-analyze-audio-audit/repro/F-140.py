"""Reproduction for F-140 (raised-by B-2).

fuse.py:831 `fuse_axes(..., unsettled_above: float = 0.6)` is a bare, undecided default that
decides whether a bucket is offered to the D-10 `remeasure` hook and counted toward C4
convergence, via the internal `_pending` helper. No caller passes `unsettled_above=` and it is
not present in `default.yaml`.

This reproduces the internal `_pending` gate directly (it is a closure inside `fuse_axes`, so we
drive it through the smallest possible `fuse_axes` call and inspect the returned per-round log,
which records `remeasure_offered` region counts) at 0.61 vs 0.59 doubt either side of the default.

Run: uv run --no-sync python specs/20260815-215106-analyze-audio-audit/repro/F-140.py
(from the repository root)
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from senselab.audio.workflows.audio_analysis.fuse import fuse_axes

BUCKET = ("speech_presence", 0.0, 1.0)


def run(uncertainty: float) -> dict:
    """A single-axis, single-bucket fuse_axes call whose only signal reads `uncertainty`."""
    row = {
        "start": 0.0,
        "end": 1.0,
        "votes": {"frame_segmentation": {"value": uncertainty}},
    }
    remeasure_calls: list[list[dict]] = []

    def remeasure(axis, regions, rows_by_axis):  # noqa: ANN001, ANN202
        remeasure_calls.append(list(regions))
        return None

    rows, log = fuse_axes(
        {"speech_presence": {"identity": [row]}},
        weights_by_axis={"speech_presence": {"frame_segmentation": 1.0}},
        remeasure=remeasure,
        couple_axes=False,
        derive=None,
        max_rounds=2,  # round 0 is the initial fold; round 1 is where `_pending`/remeasure fire
        snr_gate=None,
    )
    return {"rows": rows, "log": log, "remeasure_calls": remeasure_calls}


# `uncertainty` on a fused row is entropy_uncertainty of the Bernoulli(value) split, not the raw
# `value` a signal reports (see fuse.py:436-439's `epistemic_uncertainty([{"unsettled": v, ...}])`).
# H(0.14) ~= 0.584 (below the 0.6 default) and H(0.15) ~= 0.610 (above it) — the two raw signal
# values that straddle the `unsettled_above` cliff once folded.
below = run(0.14)
above = run(0.15)

below_offered = below["remeasure_calls"][0] if below["remeasure_calls"] else []
above_offered = above["remeasure_calls"][0] if above["remeasure_calls"] else []

print(f"raw signal value=0.14 (folded uncertainty ~0.584) -> regions offered: {below_offered!r}")
print(f"raw signal value=0.15 (folded uncertainty ~0.610) -> regions offered: {above_offered!r}")

repo_root = Path(__file__).resolve().parents[3]
overridden = False
for path in (repo_root / "src/senselab/audio/workflows/audio_analysis").rglob("*.py"):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and any(getattr(kw, "arg", None) == "unsettled_above" for kw in node.keywords):
            overridden = True
print(f"unsettled_above= passed anywhere in the package: {overridden}")

flips = len(below_offered) == 0 and len(above_offered) == 1 and not overridden

if flips:
    print(
        "DEFECT REPRODUCED: unsettled_above=0.6 (bare default, no derivation, no override path) "
        "decides whether a bucket is offered to remeasure/counted toward C4 convergence. "
        "folded uncertainty~0.584 -> 0 regions offered (wrong side of the cliff); "
        "folded uncertainty~0.610 -> 1 region offered, for two signal readings 0.01 apart. "
        f"below={below_offered!r} above={above_offered!r}"
    )
    sys.exit(0)
else:
    print("NOT REPRODUCED")
    sys.exit(1)
