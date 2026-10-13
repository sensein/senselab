"""Reproduction for F-158 (raised-by B-20).

adaptive/policy.py:14,196: `priority = round(gain / _COST_WEIGHT[rule["cost"]], 9)` is sorted
across every rule in one candidate list, but `gain` is not one quantity across rules:
`interventions.py:1134`'s `_mass_gain` is bounded doubt-seconds, `:1138`'s `_n_candidates_gain` is
a raw unbounded count, and `:502-503`'s `_u2_gain` is an arbitrary x10-scaled product. This calls
all three real gain functions directly and shows a P3-style raw-count gain dwarfs a genuinely
contested speaker region's bounded mass gain, purely because of which gain formula a rule happens
to be assigned.

Run: uv run --no-sync python specs/20260815-215106-analyze-audio-audit/repro/F-158.py
(from the repository root)
"""

from __future__ import annotations

import sys

from senselab.audio.workflows.audio_analysis.adaptive.interventions import (
    _mass_gain,
    _n_candidates_gain,
    _u2_gain,
)
from senselab.audio.workflows.audio_analysis.adaptive.policy import _COST_WEIGHT

print(f"_COST_WEIGHT = {_COST_WEIGHT!r}")

# P3 (missed-speech candidates): light cost, gain = raw unbounded candidate count. A recording
# with 50 uncorroborated-speech buckets.
p3_region, p3_ctx, p3_trigger = None, {}, {"n_candidates": 50}
p3_gain = _n_candidates_gain(p3_region, p3_ctx, p3_trigger)
p3_priority = round(p3_gain / _COST_WEIGHT["light"], 9)

# A genuinely contested speaker region: medium cost, gain = bounded doubt-seconds mass. A
# realistic, non-trivial regional mass (well above the median contested region).
speaker_region = {"uncertainty_mass": 0.8}
speaker_gain = _mass_gain(speaker_region, {}, {})
speaker_priority = round(speaker_gain / _COST_WEIGHT["medium"], 9)

# U2 (reserve ASR escalation): medium cost, gain = mass * epistemic * 10 (arbitrary multiplier).
u2_region = {"uncertainty_mass": 0.8}
u2_trigger = {"epistemic": 0.5}
u2_gain = _u2_gain(u2_region, {}, u2_trigger)
u2_priority = round(u2_gain / _COST_WEIGHT["medium"], 9)

print(f"P3 (_n_candidates_gain, light):   gain={p3_gain}, priority={p3_priority}")
print(f"speaker (_mass_gain, medium):     gain={speaker_gain}, priority={speaker_priority}")
print(f"U2 (_u2_gain, medium):            gain={u2_gain}, priority={u2_priority}")

p3_dwarfs_speaker = p3_priority > 50 * speaker_priority
u2_inflated_vs_equal_mass_speaker = u2_priority > speaker_priority

if p3_dwarfs_speaker and u2_inflated_vs_equal_mass_speaker:
    print(
        f"DEFECT REPRODUCED: P3's priority ({p3_priority}) dwarfs the genuinely contested "
        f"speaker region's priority ({speaker_priority}) by {p3_priority / speaker_priority:.0f}x "
        "purely because it is assigned _n_candidates_gain (a raw, unbounded count) rather than "
        f"_mass_gain (bounded doubt-seconds); and U2's arbitrary x10 multiplier inflates its "
        f"priority ({u2_priority}) above an EQUAL-mass speaker candidate's ({speaker_priority}) "
        "with no shared normalization across gain families. 'priority' reflects which gain "
        "formula a rule happened to be assigned, not relative value across rule types."
    )
    sys.exit(0)
else:
    print("NOT REPRODUCED")
    sys.exit(1)
