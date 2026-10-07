"""Reproduction for F-146 (raised-by B-8).

identity_binding.py:145 `binding_agreement = (len(bound)/eligible) if eligible else 0.0` reports
the identical value `0.0` for two semantically opposite states: "every diarizer, still able to
label more speakers, explicitly rejected this speaker" (real disagreement) vs "every diarizer had
already reached its speaker-count capacity before this speaker was even considered" (nothing was
checked at all). Both collapse `eligible` to 0 (censored tools are excluded from `eligible` the
same way a tool with no spans is), producing bit-for-bit identical `binding_agreement=0.0`.

Run: uv run --no-sync python specs/20260815-215106-analyze-audio-audit/repro/F-146.py
(from the repository root)
"""

from __future__ import annotations

import sys

from senselab.audio.workflows.audio_analysis.identity_binding import per_speaker_presence
from senselab.audio.workflows.audio_analysis.shapes import Span, Spans

# Two fused speakers: S0 is the speaker under test; S1 is a different speaker every tool actually
# labels (so each tool DID produce spans and DID bind something -- it is not simply silent).
speaker_spans = {"S0": [(0.0, 1.0)], "S1": [(2.0, 3.0)]}


def tool_spans(capacity) -> Spans:  # noqa: ANN001
    """One tool that only ever labels S1's time window -- S0 gets no overlapping label."""
    return Spans(spans=(Span(start=2.0, end=3.0, label="L1"),), capacity=capacity)


# Scenario A: both tools are at their speaker-count CAPACITY (1) once L1 is assigned -- S0 was
# never actually checked, it's simply beyond what these tools could represent.
censored = per_speaker_presence(speaker_spans, spans_by_tool={"tool_a": tool_spans(1), "tool_b": tool_spans(1)})

# Scenario B: identical spans, but capacity is unbounded -- both tools genuinely had room to
# label S0 and explicitly did not. This is real, checked disagreement.
rejected = per_speaker_presence(
    speaker_spans, spans_by_tool={"tool_a": tool_spans("unbounded"), "tool_b": tool_spans("unbounded")}
)

s0_censored = censored["S0"]
s0_rejected = rejected["S0"]

print(f"all tools AT CAPACITY (nothing checked)   -> {s0_censored!r}")
print(f"all tools UNBOUNDED, explicit reject       -> {s0_rejected!r}")

identical = (
    s0_censored["binding_agreement"] == 0.0
    and s0_rejected["binding_agreement"] == 0.0
    and s0_censored["bound_in"] == ()
    and s0_rejected["bound_in"] == ()
    and s0_censored["censored_in"] == ("tool_a", "tool_b")
    and s0_rejected["unbound_in"] == ("tool_a", "tool_b")
)

if identical:
    print(
        "DEFECT REPRODUCED: binding_agreement=0.0 in BOTH scenarios -- 'every diarizer was "
        "already at capacity, S0 was never actually evaluated' (censored_in=('tool_a','tool_b'), "
        "eligible=0) reads bit-for-bit identical to 'every diarizer had room and explicitly "
        "rejected S0' (unbound_in=('tool_a','tool_b'), eligible=2, bound=0). A consumer of "
        "binding_agreement cannot distinguish 'unmeasured' from 'measured and unanimously "
        "rejected' -- both are 0.0."
    )
    sys.exit(0)
else:
    print("NOT REPRODUCED")
    sys.exit(1)
