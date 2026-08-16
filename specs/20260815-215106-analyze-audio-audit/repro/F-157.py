"""Reproduction for F-157 (raised-by B-19).

adaptive/interventions.py:939 (`_p2_trigger`) has `fires = coarse_share >= threshold or
mean_instability > 0.0`, comparing a continuous variance-derived `frame_dispersion`-mean against
exactly `0.0`, even though the function's own docstring says "a high value means the bucket
straddles an onset" -- i.e. `mean_instability` is meant to be read as graded, not boolean. Since
real-valued frame posteriors are essentially never exactly 0.0, `P2_fine_posteriors` fires on
almost every region regardless of `coarse_share`.

Run: uv run --no-sync python specs/20260815-215106-analyze-audio-audit/repro/F-157.py
(from the repository root)
"""

from __future__ import annotations

import sys

from senselab.audio.workflows.audio_analysis.adaptive.interventions import _p2_trigger


class FakeState:
    """Duck-typed stand-in for adaptive.belief.BeliefState -- only `axis_rows` is read."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def axis_rows(self, axis: str) -> list[dict]:
        return self._rows


class FakeStore:
    """Duck-typed stand-in for VoteStore -- no active votes at all in this bucket, so
    `coarse_share` (computed from active votes) is 0.0, well below the 0.5 threshold."""

    def active_votes(self, stream: str, axis: str, bucket: tuple[float, float]) -> dict:
        return {}


region = {"axis": "speech_presence", "core_start": 0.0, "core_end": 1.0}
# A near-zero, but not exactly zero, frame_dispersion -- the ordinary case for any real-valued
# frame posterior, nowhere near "straddling an onset" as the docstring's own reading requires.
row = {"start": 0.0, "end": 1.0, "meta": {"frame_dispersion": 1e-9}}
ctx = {
    "state": FakeState([row]),
    "store": FakeStore(),
    "policy": {"speech_presence": {"coarse_share_threshold": 0.5}},
}

fires, detail = _p2_trigger(region, ctx)
print(f"coarse_share=0.0 (no coarse voters at all, well below threshold=0.5)")
print(f"mean_instability={detail['mean_frame_dispersion']} (near-zero, not 'straddling an onset')")
print(f"_p2_trigger(...) -> fires={fires}, detail={detail!r}")

fires_on_negligible_instability = fires is True and detail["coarse_share"] == 0.0 and detail["reason"] == "frame_dispersion"

if fires_on_negligible_instability:
    print(
        "DEFECT REPRODUCED: P2_fine_posteriors fires (fires=True) purely because "
        f"mean_instability={detail['mean_frame_dispersion']} > 0.0, even though coarse_share=0.0 "
        "is nowhere near the 0.5 threshold and the docstring itself says a HIGH value (not any "
        "nonzero value) is what should mean 'straddles an onset'. Since real-valued frame "
        "posteriors are essentially never exactly 0.0, this rule fires on almost every "
        "speech_presence region regardless of actual coarse dominance, consuming the medium-cost "
        "budget (capped 24/run) that U1/U2 then lose to deferred_budget."
    )
    sys.exit(0)
else:
    print("NOT REPRODUCED")
    sys.exit(1)
