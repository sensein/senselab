"""Another axis's value bounds where the speaker question applies; it never answers it.

The speaker axis asks **who** is speaking. ``speech_presence``, ``asr`` and ``background_mask``
answer whether there is anything here to attribute — a different question, whose answer is not
evidence about identity. Measured on a clean two-speaker conversation: round 0 read 0.0487 from the
diarizers' own agreement, round 1 read 0.1601 once the three ``axis::*`` inputs were injected, round
2 read 0.3601. The whole rise is those three, and none of them had looked at a speaker.
"""

from __future__ import annotations

from typing import Any

from senselab.audio.workflows.audio_analysis.axes import AXIS_NAMES, COUPLING_IS_A_GATE
from senselab.audio.workflows.audio_analysis.fuse import cross_axis_inputs, per_signal_uncertainty

KEYS = {(0.0, 0.1), (0.1, 0.2)}


def _rows(uncertainty: float) -> list[dict[str, Any]]:
    return [{"start": s, "end": e, "uncertainty": uncertainty} for s, e in sorted(KEYS)]


def _previous() -> dict[str, list[dict[str, Any]]]:
    return {
        "speech_presence": _rows(0.9),
        "asr": _rows(0.9),
        "background_mask": _rows(0.9),
        "speaker": _rows(0.0),
    }


def test_the_speaker_axis_takes_no_cross_axis_vote() -> None:
    """The regression: three axes that never looked at a speaker cannot raise speaker doubt."""
    buckets, contributors = cross_axis_inputs("speaker", _previous(), own_keys=KEYS)
    assert buckets == [], "a coupled value must not enter the speaker fold as a vote"
    assert contributors == [], "and must not be recorded as having contributed one"


def test_the_other_axes_still_couple() -> None:
    """Narrow by construction: where the coupled quantity and the question match, coupling stands."""
    for axis in ("speech_presence", "asr", "background_mask"):
        buckets, contributors = cross_axis_inputs(axis, _previous(), own_keys=KEYS)
        assert buckets, f"{axis} lost its coupling"
        assert contributors, f"{axis} lost its contributor record"


def test_a_coupled_vote_would_have_been_scored() -> None:
    """Shows the fold *does* read these, so returning nothing is what keeps them out.

    Without this, the test above passes just as well if ``per_signal_uncertainty`` had never scored
    ``axis::*`` in the first place — and then it would be asserting nothing.
    """
    buckets, _ = cross_axis_inputs("asr", _previous(), own_keys=KEYS)
    read = per_signal_uncertainty(buckets[0])
    assert {name for name in read if name.startswith("axis::")}, (
        "cross-axis inputs are not scored at all, so this whole mechanism is misdescribed"
    )


def test_every_gated_axis_is_a_real_axis() -> None:
    """A typo in the declaration would silently gate nothing."""
    assert COUPLING_IS_A_GATE <= set(AXIS_NAMES), (
        f"unknown axis in COUPLING_IS_A_GATE: {COUPLING_IS_A_GATE - set(AXIS_NAMES)}"
    )
    assert "speaker" in COUPLING_IS_A_GATE
