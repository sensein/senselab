"""Reproduction for F-145 (raised-by B-7).

speaker_identity.py:60 `_SUPPORTED_THRESHOLD = 0.5` is an unexplained midpoint gating
`SpeakerHypothesis.has_supported_evidence`, unlike `signal_support`'s derived floor
(`MIN_EVIDENCE_WEIGHT = 0.05`). `source_support=0.49` reports `has_supported_evidence: false`;
`0.50` reports `true`, in `final/speakers.json`.

Run: uv run --no-sync python specs/20260815-215106-analyze-audio-audit/repro/F-145.py
(from the repository root)
"""

from __future__ import annotations

import sys

from senselab.audio.workflows.audio_analysis.speaker_identity import SpeakerHypothesis

below = SpeakerHypothesis(
    speaker_id="S0",
    existence_uncertainty=0.1,
    supporting_sources=["diarizer_a"],
    source_support={"diarizer_a": 0.49},
)
above = SpeakerHypothesis(
    speaker_id="S0",
    existence_uncertainty=0.1,
    supporting_sources=["diarizer_a"],
    source_support={"diarizer_a": 0.50},
)

print(f"source_support=0.49 -> has_supported_evidence={below.has_supported_evidence}")
print(f"source_support=0.50 -> has_supported_evidence={above.has_supported_evidence}")

flips = below.has_supported_evidence is False and above.has_supported_evidence is True

if flips:
    print(
        "DEFECT REPRODUCED: _SUPPORTED_THRESHOLD=0.5 (unexplained midpoint, no derivation "
        "comment, unlike floors.MIN_EVIDENCE_WEIGHT=0.05's cited derivation) is a hard cliff: "
        "source_support=0.49 -> has_supported_evidence=False; 0.50 -> True, in "
        "final/speakers.json, for a 0.01 difference in a measured-but-unfitted quantity."
    )
    sys.exit(0)
else:
    print("NOT REPRODUCED")
    sys.exit(1)
