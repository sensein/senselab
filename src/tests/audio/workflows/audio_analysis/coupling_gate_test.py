"""Another axis's value bounds where the speaker question applies; it never answers it.

The speaker axis asks **who** is speaking. ``speech_presence``, ``asr`` and ``background_mask``
answer whether there is anything here to attribute — a different question, whose answer is not
evidence about identity. Measured on a clean two-speaker conversation: round 0 read 0.0487 from the
diarizers' own agreement, round 1 read 0.1601 once the three ``axis::*`` inputs were injected, round
2 read 0.3601. The whole rise is those three, and none of them had looked at a speaker.
"""

from __future__ import annotations

from typing import Any

import pytest

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


def test_only_speech_presence_still_receives_coupled_votes() -> None:
    """The gate covers three axes; presence is the one whose question a coupled axis can answer.

    ``asr`` asks what words were said — presence bounds where that is live, speaker answers neither.
    ``background_mask`` asks whether the target was active, and its own voters are folds of the same
    diarizers, VAD and recognizers the other axes read, so coupling returns its evidence under
    another name.

    ``speech_presence`` asks whether there was a speaker, and the mask's verdict bears on it directly:
    target-free implies no target speech. Its other two inputs are zeroed by measured overlap rather
    than by this gate, which is the right division — redundancy is measurable, relevance is not.
    """
    for axis in ("asr", "background_mask", "speaker"):
        buckets, contributors = cross_axis_inputs(axis, _previous(), own_keys=KEYS)
        assert buckets == [] and contributors == [], f"{axis} is receiving coupled votes"
    buckets, contributors = cross_axis_inputs("speech_presence", _previous(), own_keys=KEYS)
    assert buckets and contributors, "presence lost the one coupling that answers its question"


def test_relevance_and_redundancy_are_different_gates() -> None:
    """Why the measured overlap cannot replace this gate, stated as the case that separates them.

    ``asr <- axis::speaker`` reads overlap **0.00** and is right to: asr's sources are three
    recognizers, speaker's are four diarizers, and the evidence really is independent. Overlap asks
    *is this already counted*; it cannot ask *does this answer my question*. Independent-and-irrelevant
    is the failure these axes had, and a signal answering a different question must not enter at any
    weight however novel its evidence.
    """
    from senselab.audio.workflows.audio_analysis.fuse import fuse_axis, measure_axis_overlap

    def fused(votes: dict[str, Any]) -> list[dict[str, Any]]:
        return fuse_axis({"raw": [{"start": 0.0, "end": 0.1, "votes": votes}]}, weights={}, snr_gate=None)

    speaker = fused(
        {"speaker_assignment": {"value": 0.5, "n_sources": 2, "source_outcomes": {"diar_a": "C0", "diar_b": "C1"}}}
    )
    asr = fused({"rec_a": {"value": 0.1}, "rec_b": {"value": 0.2}})
    assert measure_axis_overlap(asr, speaker) == pytest.approx(0.0), (
        "the evidence is genuinely independent, so overlap cannot be the thing that keeps it out"
    )
    assert "asr" in COUPLING_IS_A_GATE, "so relevance has to"


def test_a_coupled_vote_would_have_been_scored() -> None:
    """Shows the fold *does* read these, so returning nothing is what keeps them out.

    Without this, the test above passes just as well if ``per_signal_uncertainty`` had never scored
    ``axis::*`` in the first place — and then it would be asserting nothing. Asked of
    ``speech_presence``, the one axis still receiving coupled votes.
    """
    buckets, _ = cross_axis_inputs("speech_presence", _previous(), own_keys=KEYS)
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


# ── an axis about the recording as read folds only the identity ──────────────


def test_the_mask_axis_folds_the_identity_pass_only() -> None:
    """The mask's own stage refuses the enhanced variant; its axis used to accept it.

    ``stages.py`` builds the mask on the unmodified variant alone, with the measurement behind it
    written down: the enhanced pass masked 50% of a real recording against the unmodified pass's
    17.9%, "because speech enhancement removes the non-speech evidence the mask reads target activity
    from". The axis harvested ``speakers`` / ``speech`` / ``words`` from every perturbation, and on the
    48 kHz clip its enhanced ``words`` voter read mean 0.0510 against raw's 0.0102 — 5x higher, in
    exactly the direction that note predicts.
    """
    from senselab.audio.workflows.audio_analysis.axes import passes_for_axis

    assert passes_for_axis("background_mask", ["raw", "enhanced"]) == ["raw"]


def test_the_other_axes_fold_every_perturbation() -> None:
    """Narrow by construction: a transform may legitimately change what those axes read.

    That is what makes the perturbation a *sample* rather than a contaminant, and it is what
    ``reliability.signal_stability`` measures.
    """
    from senselab.audio.workflows.audio_analysis.axes import passes_for_axis

    for axis in ("speech_presence", "speaker", "asr"):
        assert passes_for_axis(axis, ["raw", "enhanced"]) == ["raw", "enhanced"], axis


def test_an_identity_only_axis_still_folds_when_the_identity_is_absent() -> None:
    """An axis with no defensible pass is worse than one measured on the only pass there is."""
    from senselab.audio.workflows.audio_analysis.axes import passes_for_axis

    assert passes_for_axis("background_mask", ["enhanced"]) == ["enhanced"]


def test_the_identity_only_declaration_names_real_axes() -> None:
    """A typo would silently filter nothing."""
    from senselab.audio.workflows.audio_analysis.axes import AXIS_NAMES, IDENTITY_ONLY_AXES

    assert IDENTITY_ONLY_AXES <= set(AXIS_NAMES), f"unknown axis: {IDENTITY_ONLY_AXES - set(AXIS_NAMES)}"


def test_the_snr_gate_and_the_identity_filter_are_different_questions() -> None:
    """They must not be conflated: one is about degradation, the other about entitlement.

    The gate asks "is there anything here for a repair to repair" — per bucket, on SNR. This asks "is
    this perturbation entitled to answer this question at all", and for the mask the answer is no at
    any SNR. Conflating them would make the mask's exclusion depend on how noisy the recording is.
    """
    from senselab.audio.workflows.audio_analysis.axes import IDENTITY_ONLY_AXES, passes_for_axis

    assert "background_mask" in IDENTITY_ONLY_AXES
    # No SNR anywhere in the signature or the result: the filter cannot vary with degradation.
    assert passes_for_axis("background_mask", ["raw", "enhanced"]) == ["raw"]


def test_the_loops_ingest_applies_the_identity_filter_too() -> None:
    """The third reader. It was missed, and only the 48 kHz clip showed it.

    ``buckets_for_axis``'s docstring names three readers — ``link_pass``, ``fuse``, and the loop's
    ingest. ``IDENTITY_ONLY_AXES`` reached the first two, and ``final/`` is an extraction of a *loop*
    round, so the reader that decides what ships was the one still folding the enhanced pass. Invisible
    on a clean recording (``SnrGate`` excludes the enhanced pass everywhere at high SNR) and visible on
    the 48 kHz clip, where 9 buckets dip below the floor and `final/background_mask` reported
    ``contributing_passes: ['enhanced', 'raw']``.

    Asserted on the store's ingest rather than through a full run, but *over both streams*: the filter
    falls back to "fold whatever is available" when the identity is absent, so a per-stream question
    always answers yes and a filter written that way does nothing. That was the first attempt.
    """
    from senselab.audio.workflows.audio_analysis.adaptive.belief import VoteStore
    from senselab.audio.workflows.audio_analysis.votes import PassHarvest

    def _harvest(label: str) -> PassHarvest:
        return PassHarvest(
            perturbation=label,
            background_mask_evidence=[{"start": 0.0, "end": 0.1, "votes": {"words": {"value": 0.4}}}],
        )

    store = VoteStore.from_harvests({"raw": _harvest("raw"), "enhanced": _harvest("enhanced")})
    streams = {v.stream for v in store._votes.values() if v.axis == "background_mask"}
    assert streams == {"raw"}, f"the mask axis ingested {streams}; the enhanced pass is not entitled"


# ── evidence overlap must see through a fold ─────────────────────────────────


def _fused(votes: dict[str, Any]) -> list[dict[str, Any]]:
    from senselab.audio.workflows.audio_analysis.fuse import fuse_axis

    return fuse_axis({"raw": [{"start": 0.0, "end": 0.1, "votes": votes}]}, weights={}, snr_gate=None)


def test_overlap_sees_through_a_folding_signal() -> None:
    """A fold renames its sources out of visibility, and the measure was comparing names.

    ``measure_axis_overlap`` discounts a cross-axis input by how much of its evidence the receiver
    already holds — the *measured* successor to a deleted constant. Compared by signal name it worked
    wherever names collided (``speech_presence <- axis::asr`` read overlap 1.00 and was correctly
    zeroed, because asr's signals are recognizers that also vote on presence) and inverted wherever a
    fold intervened.

    Measured on the 4.9 s validation clip, ``speaker``'s one signal ``speaker_assignment`` is an
    entropy over four diarizers — **all four of which are ``speech_presence`` signals**. By name it
    collided with nothing, so overlap read 0.00 and the axis whose evidence presence already held
    *entirely* entered at full weight 1.00. Expanding to sources reads 1.00, and the weight goes to
    zero.
    """
    from senselab.audio.workflows.audio_analysis.fuse import measure_axis_overlap

    speaker = _fused(
        {
            "speaker_assignment": {
                "value": 0.5,
                "n_sources": 2,
                "source_outcomes": {"pyannote": "C0", "sortformer": "C1"},
            }
        }
    )
    presence = _fused({"pyannote": {"value": 0.1}, "sortformer": {"value": 0.2}, "lufs": {"value": 0.3}})
    assert measure_axis_overlap(presence, speaker) == pytest.approx(1.0), (
        "the receiver already holds every source behind this fold; overlap must say so"
    )


def test_an_atomic_signal_still_stands_for_itself() -> None:
    """Signals that fold nothing are unchanged — the expansion must not invent sources."""
    from senselab.audio.workflows.audio_analysis.fuse import measure_axis_overlap

    a = _fused({"x": {"value": 0.2}, "y": {"value": 0.3}})
    b = _fused({"y": {"value": 0.4}, "z": {"value": 0.5}})
    # b's sources are {y, z}; a holds y. Half of b's evidence is already present.
    assert measure_axis_overlap(a, b) == pytest.approx(0.5)


def test_a_disjoint_fold_keeps_its_weight() -> None:
    """Narrow by construction: seeing through a fold must not zero every coupled input.

    A fold whose sources the receiver does *not* hold is genuinely new information and keeps full
    weight — otherwise this would be the hand-set discount the measure exists to replace.
    """
    from senselab.audio.workflows.audio_analysis.fuse import measure_axis_overlap

    folded = _fused({"f": {"value": 0.5, "n_sources": 2, "source_outcomes": {"m1": "A", "m2": "B"}}})
    other = _fused({"unrelated": {"value": 0.1}})
    assert measure_axis_overlap(other, folded) == pytest.approx(0.0)
