"""Reproduction for F-154 (raised-by B-16) -- one of the three most valuable reproductions.

adaptive/loop.py:315-317 sets `run_state = "converged" if not not_admitted else
"no_runnable_interventions"` whenever no intervention `fired` this round. But a bucket whose
doubt sits strictly between `theta_low` (0.33) and `theta_high` (0.66) never seeds a region at
all (`regions.propose_regions` only seeds where doubt >= theta_high), so it can never appear in
either `admitted` or `not_admitted` -- both stay empty even though the bucket is nowhere near
`converged` (which requires doubt <= theta_low). This builds that exact belief state and drives
it through the real `build_convergence_report`, showing the top-level `converged` flag reads
`true` while the same document's own `per_axis` detail lists the bucket `"open"` with nonzero
residual mass.

Run: uv run --no-sync python specs/20260815-215106-analyze-audio-audit/repro/F-154.py
(from the repository root)
"""

from __future__ import annotations

import sys

from senselab.audio.workflows.audio_analysis.adaptive.belief import AXES, BeliefState
from senselab.audio.workflows.audio_analysis.adaptive.convergence import build_convergence_report
from senselab.audio.workflows.audio_analysis.adaptive.policy import BudgetLedger
from senselab.audio.workflows.audio_analysis.adaptive.regions import propose_regions

THETA_LOW, THETA_HIGH = 0.33, 0.66  # data/run_config/default.yaml:379-380

POLICY = {
    "thresholds": {"theta_low": THETA_LOW, "theta_high": THETA_HIGH, "epsilon": 0.03},
    "regions": {"gap_merge_s": 0.5, "pad_s": 1.0, "top_n_per_round": 8, "max_region_rounds": 2},
    "budget": {"medium_per_run": 24, "heavy_per_run": 4},
}

# A row whose doubt (1 - confidence) is 0.40: above theta_low (converged threshold) but below
# theta_high (region-seed threshold) -- the dead zone the finding describes.
STUCK_ROW = {
    "start": 0.0,
    "end": 1.0,
    "status": "open",
    "confidence": 0.60,  # control_doubt = 1 - 0.60 = 0.40
    "uncertainty": 0.9,
    "history": [{"round": 0, "uncertainty": 0.9, "doubt": 0.40}],
}

# 1. Confirm no region is ever proposed for this bucket -- nothing to trigger against.
regions = propose_regions([STUCK_ROW], axis="speaker", policy=POLICY, round_idx=0, duration_s=1.0)
print(f"propose_regions on a doubt=0.40 bucket (theta_low={THETA_LOW}, theta_high={THETA_HIGH}): {regions!r}")

# 2. Build the belief state the real loop would hold: this stuck bucket on "speaker", nothing on
# any other axis.
state = BeliefState("mean")
for axis in AXES:
    state.rows[axis] = [dict(STUCK_ROW)] if axis == "speaker" else []

# 3. Exactly loop.py:315's logic: no region was ever proposed, so nothing could fire and nothing
# could be not_admitted -- both candidate lists are empty by construction.
fired: list = []
not_admitted: list = []
run_state = "converged" if not not_admitted else "no_runnable_interventions"
print(f"fired={fired!r}, not_admitted={not_admitted!r} -> run_state={run_state!r}")

ledger = BudgetLedger(POLICY)
report = build_convergence_report(
    state=state,
    policy=POLICY,
    rounds=[],
    ledger=ledger,
    iterations=[],
    run_state=run_state,
    provenance={},
    round_states=None,
)

top_level_converged = report["converged"]
speaker_detail = report["per_axis"]["speaker"]

print(f"report['converged'] (top-level headline) = {top_level_converged}")
print(f"report['per_axis']['speaker'] (same document's own detail) = {speaker_detail!r}")

self_contradicting = (
    len(regions) == 0
    and top_level_converged is True
    and speaker_detail.get("open") == 1
    and speaker_detail.get("residual_mass", 0.0) > 0.0
)

if self_contradicting:
    print(
        "DEFECT REPRODUCED: final/decisions.json would report converged=True at the top level "
        f"while its own per_axis['speaker'] detail in the SAME document says "
        f"{speaker_detail!r} -- one open bucket with nonzero residual_mass that was never even "
        "proposed as a region, let alone resolved. run_state='converged' should only mean "
        "'every bucket settled below theta_low', not 'nothing happened to fire this round'."
    )
    sys.exit(0)
else:
    print("NOT REPRODUCED")
    sys.exit(1)
