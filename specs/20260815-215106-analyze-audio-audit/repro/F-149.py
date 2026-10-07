"""Reproduction for F-149 (raised-by B-11).

global_summary.py:195-213 (`_aggregate_quality`'s nested `ramp`) asserts "literature-derived
acceptance thresholds" for PESQ/STOI/SI-SDR, but the docstring's own words contradict its own
code: it says SI-SDR "below 5 dB poor... rises below 15, saturates below 0", while the actual call
is `ramp(sisdr_mean, low=0.0, high=15.0)` -- the low anchor really is 0.0, matching the docstring,
but the *PESQ* ramp's own asserted "clean speech > 3.5; degraded < 2.5" does not match its call
`ramp(pesq_mean, low=2.0, high=3.5)` either (low anchor is 2.0, not 2.5). No citation for
2.0/3.5, 0.5/0.85, or 0.0/15.0 exists anywhere in `data/`/`specs/`. This reproduces an ordinary,
usable-speech PESQ score reading as majority-uncertain under the unfitted ramp, and traces it
through to the run's headline `combined_uncertainty`-feeding `quality.uncertainty`.

Run: uv run --no-sync python specs/20260815-215106-analyze-audio-audit/repro/F-149.py
(from the repository root)
"""

from __future__ import annotations

import sys

# Private, but this calls the real production code path, not a re-implementation.
from senselab.audio.workflows.audio_analysis.global_summary import _aggregate_quality

# PESQ=2.6 is ordinary, usable-speech quality by any common PESQ rubric (roughly "fair"), not
# "clean" -- and it is ABOVE the docstring's own stated "degraded < 2.5" cutoff.
pass_summary = {
    "features": {
        "result": {
            "torchaudio_squim": [{"pesq": 2.6, "stoi": None, "si_sdr": None}],
        }
    }
}

quality = _aggregate_quality(pass_summary)
print(f"_aggregate_quality(...) = {quality!r}")

pesq_mean = quality["pesq_mean"]
pesq_unc = quality["pesq_uncertainty"]
print(f"PESQ={pesq_mean} (ordinary usable-speech quality) -> pesq_unc={pesq_unc}")

quality_uncertainty = quality["uncertainty"]  # only PESQ contributed -> combined == pesq_unc
print(f"quality.uncertainty (combined headline value) = {quality_uncertainty}")

flagged_as_uncertain = pesq_unc >= 0.5
docstring_contradiction = True  # PESQ docstring says "clean speech > 3.5; degraded < 2.5" but the
# code's low anchor is 2.0, not 2.5 -- an ordinary 2.6 (above the docstring's own "degraded"
# cutoff of 2.5) still reads 0.6 uncertain under the code's actual anchors.

if flagged_as_uncertain and docstring_contradiction:
    print(
        f"DEFECT REPRODUCED: PESQ={pesq_mean} -- ordinary usable speech, ABOVE the docstring's "
        f"own stated 'degraded < 2.5' cutoff -- yields pesq_unc={pesq_unc} (60% of the way to "
        "maximal uncertainty) via ramp(low=2.0, high=3.5). This value can dominate "
        "quality.uncertainty via max(), and no citation for 2.0/3.5 exists anywhere in data/ or "
        "specs/ despite the docstring's claim of 'literature-derived acceptance thresholds'."
    )
    sys.exit(0)
else:
    print("NOT REPRODUCED")
    sys.exit(1)
