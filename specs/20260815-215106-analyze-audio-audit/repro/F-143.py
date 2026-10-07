"""Reproduction for F-143 (raised-by B-5).

support.py:276 `MIN_LOW_FRACTION = 0.02` gates evidence-pool admission for speech-presence
corroboration (`informative_evidence`), but the docstring's own numbers were measured under a
reading bug it explicitly disowns ("the per-voter verdicts above must be re-measured before they
are cited again"). This reproduces the threshold acting as a hard cliff: one signal, identical
range/spread, differing only in whether exactly one of 50 buckets crosses the "no speech" line —
flipping inclusion in the evidence pool.

Run: uv run --no-sync python specs/20260815-215106-analyze-audio-audit/repro/F-143.py
(from the repository root)
"""

from __future__ import annotations

import sys

from senselab.audio.workflows.audio_analysis.support import MIN_LOW_FRACTION, informative_evidence

print(f"MIN_LOW_FRACTION = {MIN_LOW_FRACTION}")
assert MIN_LOW_FRACTION == 0.02

N = 50  # 1/50 = 0.02, exactly the bare cutoff


def buckets(n_low: int) -> list[dict]:
    """N buckets; `n_low` of them below EVIDENCE_LOW_THRESHOLD (0.20), the rest at 0.95.

    Spread (0.95 - 0.10 = 0.85) always clears MIN_EVIDENCE_SPREAD (0.15), isolating the
    low-fraction gate as the only thing that can change the verdict.
    """
    out = []
    for i in range(n):
        speaks = True
        conf = 0.10 if i < n_low else 0.95
        out.append({"votes": {"acoustic_hnr": {"speaks": speaks, "native_confidence": conf}}})
    return out


n = N
zero_low = informative_evidence(buckets(0), ["acoustic_hnr"])
one_low = informative_evidence(buckets(1), ["acoustic_hnr"])  # fraction = 1/50 = 0.02, >= cutoff

print(f"0/{N} buckets read 'no speech' (fraction=0.0)    -> kept as evidence: {zero_low!r}")
print(f"1/{N} buckets read 'no speech' (fraction=0.02)   -> kept as evidence: {one_low!r}")

flips = zero_low == set() and one_low == {"acoustic_hnr"}

if flips:
    print(
        "DEFECT REPRODUCED: MIN_LOW_FRACTION=0.02 (numbers behind it self-disowned in the "
        "docstring as measured under a since-fixed bug) is a hard cliff: a signal reporting "
        "'no speech' in 0/50 buckets is EXCLUDED from the evidence pool; the identical signal "
        "reporting it in exactly 1/50 buckets (fraction=0.02) is INCLUDED — one bucket flips "
        "whether this signal can corroborate/withhold support for speaker_count_posterior and "
        "reliability.measured_weights, with no valid measurement behind the boundary."
    )
    sys.exit(0)
else:
    print("NOT REPRODUCED")
    sys.exit(1)
