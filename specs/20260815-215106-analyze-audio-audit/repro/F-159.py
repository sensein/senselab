"""Reproduction for F-159 (raised-by B-21).

adaptive/convergence.py:75-78 (`apply_convergence_marks`) computes the raw two-point delta
`improvement = float(prev_u) - float(last_u)` and gates `stalled = improvement < epsilon` on it,
with nothing distinguishing "uncertainty fell because independent evidence arrived" from
"uncertainty fell because the loop re-scored its own prior overwrite". The dead
`adaptive/provenance.py` module exists specifically to make that distinction
(`classify_resolution`) but is never called from `convergence.py` or anywhere in the live loop
(only from its own unit test) -- confirmed live by an import-site grep, and reproduced here by
running BOTH the real `classify_resolution` (which correctly tags a self-confirming re-score as
"revision", not a confidence gain) and the real `apply_convergence_marks` (which credits the
identical drop as ordinary progress, with no such distinction available).

Run: uv run --no-sync python specs/20260815-215106-analyze-audio-audit/repro/F-159.py
(from the repository root)
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from senselab.audio.workflows.audio_analysis.adaptive.belief import AXES, BeliefState
from senselab.audio.workflows.audio_analysis.adaptive.convergence import apply_convergence_marks
from senselab.audio.workflows.audio_analysis.adaptive.provenance import classify_resolution

# 1. classify_resolution correctly identifies a drop that followed a revision, with no
# independent corroboration, as "revision" -- NOT eligible to be reported as a confidence gain.
kind = classify_resolution(
    before_uncertainty=0.50, after_uncertainty=0.30, was_revised=True, independent_evidence=False
)
print(f"classify_resolution(before=0.50, after=0.30, was_revised=True, independent_evidence=False) "
      f"-> {kind!r}")

# 2. Confirm classify_resolution/RevisionRecord are dead in the live loop: grepped for real call
# sites in every adaptive/*.py file except provenance.py and its own test.
repo_root = Path(__file__).resolve().parents[3]
adaptive_dir = repo_root / "src/senselab/audio/workflows/audio_analysis/adaptive"
call_sites: list[str] = []
for path in sorted(adaptive_dir.glob("*.py")):
    if path.name == "provenance.py":
        continue
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in ("classify_resolution", "RevisionRecord"):
            call_sites.append(f"{path.name}:{node.lineno}")
print(f"call sites of classify_resolution/RevisionRecord outside provenance.py: {call_sites!r}")

# 3. apply_convergence_marks (the LIVE convergence gate) treats the identical before/after drop
# as ordinary, undifferentiated progress -- it has no `was_revised`/`independent_evidence` inputs
# at all, so it cannot apply the distinction classify_resolution just made.
policy = {
    "thresholds": {"theta_low": 0.33, "epsilon": 0.03},
    "regions": {"max_region_rounds": 2},
}
row = {
    "start": 0.0,
    "end": 1.0,
    "status": "open",
    "confidence": 1.0 - 0.30,  # doubt = 0.30, matches "after_uncertainty" above
    "history": [
        {"round": 0, "doubt": 0.50},  # matches "before_uncertainty" above
        {"round": 1, "doubt": 0.30},  # matches "after_uncertainty" above -- same drop
    ],
}
state = BeliefState("mean")
for axis in AXES:
    state.rows[axis] = [dict(row)] if axis == "speaker" else []

transitions = apply_convergence_marks(state, policy=policy, touch_counts={}, budget_left=True)
print(f"apply_convergence_marks(...) transitions = {transitions!r}")
print(f"row status after the fold = {state.rows['speaker'][0]['status']!r}")

dead_module = call_sites == []
# The SAME 0.50->0.30 drop classify_resolution just tagged "revision" (self-confirmation, not a
# confidence gain) crosses theta_low in apply_convergence_marks and is marked "converged" outright
# -- the strongest possible form of "genuine progress" -- because that function has no
# `resolution_kind`/`was_revised`/`independent_evidence` input to consult at all.
row_marked_converged = state.rows["speaker"][0]["status"] == "converged"

if kind == "revision" and dead_module and row_marked_converged:
    print(
        f"DEFECT REPRODUCED: classify_resolution correctly tags the 0.50->0.30 drop as "
        f"{kind!r} (self-confirmation, not genuine evidence -- must NOT count as a confidence "
        f"gain per RevisionRecord.improves_confidence()), but that function has ZERO call sites "
        f"anywhere in the live loop ({call_sites!r} outside its own dead module). "
        "apply_convergence_marks -- the function that actually gates the loop -- takes the "
        "identical raw delta and marks the bucket status='converged' outright, with no "
        "mechanism available to classify the drop as a self-confirming revision rather than "
        "genuine progress. The distinction the dead module exists to make is simply never "
        "consulted by the code that needs it."
    )
    sys.exit(0)
else:
    print("NOT REPRODUCED")
    sys.exit(1)
