"""Reproduction for F-150 (raised-by B-12).

disagreements.py:152 `"high_uncertainty_rate": (high_count / total_rows) if total_rows else 0.0`
reports `0.0` -- "nothing was uncertain" -- whenever `total_rows == 0` (every axis's harvest/fuse
failed and produced no rows at all), collapsing "we could not measure anything" into "we measured,
and it was all clean". `scripts/check_layering.py:121` prints this value directly against a
stored baseline of 0.9941, so a total-harvest-failure run reads as a dramatic improvement.

Run: uv run --no-sync python specs/20260815-215106-analyze-audio-audit/repro/F-150.py
(from the repository root)
"""

from __future__ import annotations

import sys
from pathlib import Path

from senselab.audio.workflows.audio_analysis.disagreements import build_disagreements_index
from senselab.audio.workflows.audio_analysis.types import FusedAxis

# Scenario A: a healthy run where 2 of 3 rows are genuinely high-uncertainty.
healthy_axes = {
    "asr": FusedAxis(axis="asr", rows=[{"start": 0.0, "end": 1.0, "triage_score": 0.95}]),
    "speaker": FusedAxis(axis="speaker", rows=[{"start": 0.0, "end": 1.0, "triage_score": 0.90}]),
    "speech_presence": FusedAxis(axis="speech_presence", rows=[{"start": 0.0, "end": 1.0, "triage_score": 0.10}]),
}

# Scenario B: total harvest/fuse failure -- every axis produced zero rows. Nothing was measured.
failed_axes = {
    "asr": FusedAxis(axis="asr", rows=[]),
    "speaker": FusedAxis(axis="speaker", rows=[]),
    "speech_presence": FusedAxis(axis="speech_presence", rows=[]),
}

common_kwargs = dict(top_n=10, run_dir=Path("/tmp/f150-repro"), config={}, incomparable_reasons={})

healthy = build_disagreements_index(fused_axes=healthy_axes, **common_kwargs)
failed = build_disagreements_index(fused_axes=failed_axes, **common_kwargs)

print(f"healthy run:  total_rows={healthy['totals']['total_rows']}, "
      f"high_uncertainty_rate={healthy['totals']['high_uncertainty_rate']}")
print(f"failed run:   total_rows={failed['totals']['total_rows']}, "
      f"high_uncertainty_rate={failed['totals']['high_uncertainty_rate']}")

reads_as_clean = failed["totals"]["total_rows"] == 0 and failed["totals"]["high_uncertainty_rate"] == 0.0
healthy_shows_real_rate = healthy["totals"]["high_uncertainty_rate"] > 0.5

if reads_as_clean and healthy_shows_real_rate:
    baseline = 0.9941  # scripts/check_layering.py's stored comparison baseline, per the finding
    print(
        f"DEFECT REPRODUCED: a total-harvest-failure run (0 rows on every axis) reports "
        f"high_uncertainty_rate=0.0, printed by scripts/check_layering.py directly against a "
        f"stored baseline of {baseline} -- reading as a dramatic improvement from {baseline} to "
        f"0.0, rather than the run having measured nothing at all. Contrast: a healthy run with "
        f"real high-uncertainty rows reports {healthy['totals']['high_uncertainty_rate']}, "
        "a value this 0.0 is indistinguishable from on its own."
    )
    sys.exit(0)
else:
    print("NOT REPRODUCED")
    sys.exit(1)
