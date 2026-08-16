"""Reproduction for F-144 (raised-by B-6).

speaker_identity.py:121 `multimodal_threshold: float = 0.15` (probability cutoff for
"multimodal" speaker-count posterior) has no stated derivation. A 2-point probability shift
across it flips `is_multimodal`, which gates `SpeakerCountPosterior.converged` (via
`speaker_identity.py:585`'s `converged = not posterior.is_multimodal and doubt < 0.5`) — the
adaptive loop's stopping decision.

Run: uv run --no-sync python specs/20260815-215106-analyze-audio-audit/repro/F-144.py
(from the repository root)
"""

from __future__ import annotations

import sys

from senselab.audio.workflows.audio_analysis.speaker_identity import (
    SourceCountClaim,
    speaker_count_posterior,
)

# Two sources: one claims 2 speakers (uncertainty 0.0), one claims 3. Sweeping the second
# source's own uncertainty by 0.02 crosses the bare 0.15 probability cliff: weight = 1 - u, so
# P(3) = (1-u)/(2-u), which straddles 0.15 between u=0.83 (P(3)=0.145) and u=0.81 (P(3)=0.160).
claims_unimodal = [
    SourceCountClaim(source="diarizer_a", count=2, uncertainty=0.0, support=1.0),
    SourceCountClaim(source="diarizer_b", count=3, uncertainty=0.83, support=1.0),
]
claims_multimodal = [
    SourceCountClaim(source="diarizer_a", count=2, uncertainty=0.0, support=1.0),
    SourceCountClaim(source="diarizer_b", count=3, uncertainty=0.81, support=1.0),
]

posterior_unimodal = speaker_count_posterior(claims_unimodal, multimodal_threshold=0.15)
posterior_multimodal = speaker_count_posterior(claims_multimodal, multimodal_threshold=0.15)

print(f"probabilities={posterior_unimodal.probabilities} -> is_multimodal={posterior_unimodal.is_multimodal}")
print(f"probabilities={posterior_multimodal.probabilities} -> is_multimodal={posterior_multimodal.is_multimodal}")

p2, p3_a = posterior_unimodal.probabilities.get(2), posterior_unimodal.probabilities.get(3)
p2b, p3_b = posterior_multimodal.probabilities.get(2), posterior_multimodal.probabilities.get(3)

flips = (
    not posterior_unimodal.is_multimodal
    and posterior_multimodal.is_multimodal
    and abs((p3_b or 0) - (p3_a or 0)) < 0.03
)

if flips:
    print(
        f"DEFECT REPRODUCED: multimodal_threshold=0.15 (bare default, no derivation) flips "
        f"is_multimodal on a ~2-point probability shift. P(3)={p3_a:.4f} -> unimodal/converged; "
        f"P(3)={p3_b:.4f} -> multimodal/not-converged, a difference of "
        f"{abs(p3_b - p3_a):.4f}. This directly changes `converged` in final/speakers.json and "
        "the adaptive loop's stopping decision, with no measurement behind 0.15 vs 0.10/0.20."
    )
    sys.exit(0)
else:
    print("NOT REPRODUCED")
    sys.exit(1)
