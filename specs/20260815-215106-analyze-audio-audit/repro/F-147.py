"""Reproduction for F-147 (raised-by B-9) -- one of the three most valuable reproductions.

speaker.py:636-640 (`per_speaker_tracks`, via `_bucket_clusters`) computes
`n_models = len(clusters)` where `clusters` is built only from diarizer keys actually PRESENT in
`bucket["votes"]` for that bucket. A diarizer that crashed or never ran is simply absent from that
dict -- it does not lower `speech_presence_confidence`, it shrinks the denominator, so its
disappearance is invisible. This constructs the vote structures directly (no models loaded) and
shows 3-of-4 diarizers crashing, leaving one survivor, produces the SAME
`speech_presence_confidence`/`speech_presence_uncertainty` as all 4 diarizers unanimously
agreeing.

Run: uv run --no-sync python specs/20260815-215106-analyze-audio-audit/repro/F-147.py
(from the repository root)
"""

from __future__ import annotations

import sys

from senselab.audio.workflows.audio_analysis.speaker import per_speaker_tracks

# Scenario 1: all 4 diarizers ran and unanimously placed a speaker "C0" in this bucket.
unanimous_bucket = {
    "start": 0.0,
    "end": 1.0,
    "votes": {
        "diar_1": {"cluster_id": "C0"},
        "diar_2": {"cluster_id": "C0"},
        "diar_3": {"cluster_id": "C0"},
        "diar_4": {"cluster_id": "C0"},
    },
}

# Scenario 2: 3 of 4 diarizers CRASHED (never produced a vote entry at all -- not "reported
# silence", simply absent, exactly like a process that failed before writing anything). Only the
# lone survivor's vote is present.
crashed_bucket = {
    "start": 0.0,
    "end": 1.0,
    "votes": {
        "diar_1": {"cluster_id": "C0"},
    },
}

rows_unanimous = per_speaker_tracks([unanimous_bucket])
rows_crashed = per_speaker_tracks([crashed_bucket])

row_u = rows_unanimous[0]
row_c = rows_crashed[0]

print(f"4-of-4 unanimous:   contributing_sources={row_u['contributing_sources']!r}, "
      f"confidence={row_u['speech_presence_confidence']}, uncertainty={row_u['speech_presence_uncertainty']}")
print(f"1-of-4 (3 crashed): contributing_sources={row_c['contributing_sources']!r}, "
      f"confidence={row_c['speech_presence_confidence']}, uncertainty={row_c['speech_presence_uncertainty']}")

byte_identical_confidence = (
    row_u["speech_presence_confidence"] == row_c["speech_presence_confidence"] == 1.0
    and row_u["speech_presence_uncertainty"] == row_c["speech_presence_uncertainty"] == 0.0
)
different_source_counts = len(row_u["contributing_sources"]) == 4 and len(row_c["contributing_sources"]) == 1

if byte_identical_confidence and different_source_counts:
    print(
        "DEFECT REPRODUCED: speech_presence_confidence=1.0 / speech_presence_uncertainty=0.0 "
        "are BYTE-IDENTICAL whether all 4 diarizers agree or 3 of 4 crashed and only one "
        "survivor reported a speaker -- final/per_speaker_presence.parquet cannot distinguish "
        "'unanimous 4-way agreement' from '3 models silently failed'. Correct behavior (already "
        "implemented for the sibling speaker_assignment voter): carry n_sources/source_outcomes "
        "alongside the value, as speaker.py:556 already does, so a shrunken denominator is "
        "visible rather than read as confidence."
    )
    sys.exit(0)
else:
    print("NOT REPRODUCED")
    sys.exit(1)
