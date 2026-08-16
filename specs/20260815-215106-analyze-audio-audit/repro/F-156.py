"""Reproduction for F-156 (raised-by B-18).

adaptive/identity_repair.py:227-231 (`repair_identity`) builds
`cp_conf = {round(c["time"], 4): c["confidence"] for c in cps}` from the real detected
change-points, then sets `seg["boundary_confidence"] = {"start": cp_conf.get(round(seg["start"],
4), 0.5), "end": cp_conf.get(round(seg["end"], 4), 0.5)}` -- any segment edge with no matching
detected change-point (a diarizer-only cut, or a voiced-span boundary) falls back to the literal
`0.5`, indistinguishable in the output from a genuine measured prominence of 0.5.
`adaptive/fusion.py:296-297,324` writes this into `final/diarization.json` labeled "real boundary
confidences from change-point prominence".

This calls the real `detect_change_points` on a trajectory with one strong and one genuinely
WEAK change-point (confidence 0.0333, well below 0.5), then applies identity_repair.py:227-231's
own `cp_conf.get(..., 0.5)` expression verbatim to a segment edge that has no matching detected
change-point at all -- showing the fabricated fallback (0.5) outranks the real, weak, measured
detection (0.0333).

Run: uv run --no-sync python specs/20260815-215106-analyze-audio-audit/repro/F-156.py
(from the repository root)
"""

from __future__ import annotations

import sys

from senselab.audio.workflows.audio_analysis.adaptive.identity_repair import detect_change_points

# A 40-point trajectory: one strong change-point (distance 1.8) and one genuinely real but WEAK
# one (distance 0.06) -- both clear the (mean, floor) threshold and are real local maxima.
times = [0.5 * i for i in range(1, 41)]
dist = [0.0] * 40
dist[9] = 1.8  # a strong, obvious speaker change
dist[24] = 0.06  # a real but weak change -- barely above the noise floor

cps = detect_change_points(times, dist, cp_k=0.0, cp_floor=0.001)
print(f"real detected change-points: {cps!r}")

weak_cp = next(c for c in cps if c["time"] == 12.5)
print(f"the weak, genuinely-measured detection: confidence={weak_cp['confidence']}")

# identity_repair.py:227-231, verbatim:
cp_conf = {round(c["time"], 4): c["confidence"] for c in cps}
fabricated_edge_confidence = cp_conf.get(round(0.0, 4), 0.5)  # t=0.0 has no detected change-point

print(f"segment edge at t=0.0 (no real change-point there) -> boundary_confidence = "
      f"{fabricated_edge_confidence}")

misordered = fabricated_edge_confidence == 0.5 and weak_cp["confidence"] < 0.5

if misordered:
    print(
        f"DEFECT REPRODUCED: a segment edge with NO genuine change-point gets the fabricated "
        f"fallback boundary_confidence={fabricated_edge_confidence}, which is HIGHER than the "
        f"real, measured, weak detection's confidence={weak_cp['confidence']:.4f}. "
        "final/diarization.json labels both 'real boundary confidences from change-point "
        "prominence' -- the fabricated 0.5 reads as more confident than a genuine, weak "
        "detection, misordering true vs. fabricated confidence."
    )
    sys.exit(0)
else:
    print("NOT REPRODUCED")
    sys.exit(1)
