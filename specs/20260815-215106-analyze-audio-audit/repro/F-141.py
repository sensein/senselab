"""Reproduction for F-141 (raised-by B-3).

aggregators.py's `"disagreement_weighted"` branch computes `(1 - mean_conf) * max_u`, which
algebraically reduces to `mean(uncertainty) * max(uncertainty)` — a function of the *level* of
uncertainty, not of disagreement/spread, despite its name and inline comment ("surfaces buckets
where many signals are slightly off rather than one wildly off").

Run: uv run --no-sync python specs/20260815-215106-analyze-audio-audit/repro/F-141.py
(from the repository root)
"""

from __future__ import annotations

import sys

from senselab.audio.workflows.audio_analysis.aggregators import apply_aggregator

# Five signals unanimously agreeing at 0.9 doubt each: no disagreement at all.
unanimous = apply_aggregator([0.9, 0.9, 0.9, 0.9, 0.9], "disagreement_weighted")

# Four signals confident (0.0 doubt) and one signal maximally doubtful (1.0): textbook
# disagreement — the case the docstring says this aggregator should surface.
disagreement = apply_aggregator([0.0, 0.0, 0.0, 0.0, 1.0], "disagreement_weighted")

print(f"5x unanimous 0.9 doubt -> disagreement_weighted = {unanimous}")
print(f"4x0.0 + 1x1.0 doubt (textbook disagreement) -> disagreement_weighted = {disagreement}")

assert unanimous is not None and disagreement is not None
ranks_unanimous_higher = unanimous > disagreement

if ranks_unanimous_higher:
    ratio = unanimous / disagreement
    print(
        f"DEFECT REPRODUCED: unanimous agreement (0.9, 0.9, 0.9, 0.9, 0.9) scores "
        f"{unanimous!r}, {ratio:.2f}x HIGHER than genuine 4-vs-1 disagreement ({disagreement!r}), "
        "under an aggregator named 'disagreement_weighted'. Correct behavior: a spread/variance "
        "statistic (e.g. statistics.variability) should score the disagreement case higher, not "
        "the unanimous one."
    )
    sys.exit(0)
else:
    print("NOT REPRODUCED")
    sys.exit(1)
