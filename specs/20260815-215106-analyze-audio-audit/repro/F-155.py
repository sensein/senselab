"""Reproduction for F-155 (raised-by B-17).

I1/I2's added votes (adaptive/identity_repair.py's change-point/re-cluster output, consumed via
adaptive/interventions.py) carry keys like `change_point_times`/`change_point_confidence`/
`speaker_label`/`cluster_id`/`speaker_changed_from_prev` that `fuse.per_signal_uncertainty`
does not recognize (it only reads `_SCORED_FIELDS` = same_label_uncertainty /
change_inconsistency_uncertainty / value / avg_logprob / native_confidence / p_speech / p_voice /
argmax_confidence, and direction-only claims via a "speaks" key). So `fuse_axis` never scores
these votes -- but `belief.py:818`'s `contributing_sources` is built from every vote-source key
present in `active_votes`, regardless of whether `fuse_axis` scored it, so it lists the I1/I2
votes as contributors even though they moved nothing.

Run: uv run --no-sync python specs/20260815-215106-analyze-audio-audit/repro/F-155.py
(from the repository root)
"""

from __future__ import annotations

import sys

from senselab.audio.workflows.audio_analysis.fuse import per_signal_uncertainty

BUCKET_START, BUCKET_END = 0.0, 1.0

# One real diarizer vote (scoreable) plus one I1-style repair vote (unscoreable schema).
votes = {
    "diar_1": {"same_label_uncertainty": 0.2},
    "identity_repair_i1": {
        "change_point_times": [0.5],
        "change_point_confidence": [0.8],
    },
}
bucket = {"start": BUCKET_START, "end": BUCKET_END, "votes": votes}

scored = per_signal_uncertainty(bucket)
print(f"votes present in the bucket: {sorted(votes)!r}")
print(f"per_signal_uncertainty (what fuse_axis actually scores): {scored!r}")

# This is exactly belief.py:818's contributing_sources computation: every source key present in
# the active votes for every stream, regardless of whether fuse_axis scored it.
votes_by_pass = {"identity": votes}
contributing_sources = sorted({src for v in votes_by_pass.values() for src in v})
print(f"contributing_sources (belief.py:818's logic): {contributing_sources!r}")

i1_scored = "identity_repair_i1" in scored
i1_listed_as_contributor = "identity_repair_i1" in contributing_sources

if (not i1_scored) and i1_listed_as_contributor:
    print(
        "DEFECT REPRODUCED: 'identity_repair_i1' is ABSENT from per_signal_uncertainty's output "
        f"({sorted(scored)!r}) -- fuse_axis never scores it, so it can never move a fused "
        f"value -- yet it IS listed in contributing_sources ({contributing_sources!r}), belief.py's "
        "record of who spoke toward the value. A repaired region can show mean_after==mean_before "
        "every round while contributing_sources, L2/round/<n>/estimates/speaker.parquet, and the "
        "LabelStudio view all falsely list the I1/I2 votes as contributing -- and after "
        "max_region_rounds of this no-op touching, the region is marked "
        "'irreducible: no_reduction_under_available_interventions', a false verdict whose true "
        "cause is this schema mismatch, not an aleatoric limit."
    )
    sys.exit(0)
else:
    print("NOT REPRODUCED")
    sys.exit(1)
