"""The mask marks regions of non-task interest, and does not commit while unsure.

On a 21.5 s two-speaker conversation the mask produced exactly one region: the whole file,
``target_active``, at uncertainty 0.9997 — no time resolution and a committed verdict reached
while maximally unsure. Both are fixed here.

The useful question for a speech task is not "where is speech absent" (nearly nowhere in a
conversation) but "where is something other than the task target happening" — the regions
worth introspecting for background content.
"""

from __future__ import annotations

import pytest

from senselab.audio.workflows.audio_analysis.background_mask import build_mask
from senselab.audio.workflows.audio_analysis.calibration import load_detection_margin_profile

PROFILE = load_detection_margin_profile()


def _b(start: float, conf: float, unc: float = 0.0, nontarget: float | None = None) -> dict:
    row = {"start": start, "end": start + 0.5, "target_confidence": conf, "uncertainty": unc}
    if nontarget is not None:
        row["nontarget_confidence"] = nontarget
    return row


# ── symmetry: a committed verdict needs trustworthy evidence ────────────


def test_target_active_is_not_claimed_while_maximally_uncertain() -> None:
    """The asymmetry that produced the useless mask.

    ``target_free`` already demanded low uncertainty; ``target_active`` demanded none, so a
    bucket at confidence 0.99 and uncertainty 0.99 committed to "target active" — a verdict
    the evidence did not support in either direction.
    """
    mask = build_mask([_b(i * 0.5, 0.99, unc=0.99) for i in range(8)], "speech", profile=PROFILE)
    assert {r.state for r in mask.regions} == {"indeterminate"}


def test_target_active_still_holds_when_the_evidence_agrees() -> None:
    """The guard must not erase confident detections."""
    mask = build_mask([_b(i * 0.5, 0.95, unc=0.05) for i in range(8)], "speech", profile=PROFILE)
    assert {r.state for r in mask.regions} == {"target_active"}


# ── regions of non-task interest ───────────────────────────────────────


def test_a_gap_carrying_other_content_is_marked_as_non_task_interest() -> None:
    """What the mask is for: somewhere to look for background content.

    A pause with room tone and a pause with digital silence are both target-free, but only
    the first is worth introspecting. Conflating them is why the mask "does nothing useful".
    """
    buckets = [_b(i * 0.5, 0.95, unc=0.05) for i in range(4)]
    buckets += [_b(2.0 + i * 0.5, 0.02, unc=0.05, nontarget=0.8) for i in range(6)]
    mask = build_mask(buckets, "speech", profile=PROFILE)
    assert any(r.state == "nontarget_active" for r in mask.regions)


def test_a_silent_gap_is_target_free_not_of_interest() -> None:
    """Nothing to find is a different finding from something else being present."""
    buckets = [_b(i * 0.5, 0.95, unc=0.05) for i in range(4)]
    buckets += [_b(2.0 + i * 0.5, 0.02, unc=0.05, nontarget=0.01) for i in range(6)]
    mask = build_mask(buckets, "speech", profile=PROFILE)
    states = {r.state for r in mask.regions}
    assert "nontarget_active" not in states
    assert "target_free" in states


def test_non_task_interest_requires_the_target_to_be_absent() -> None:
    """Interest requires the target to be absent.

    Background content under active target speech is not a clean region to introspect — that
    is the leakage problem the suppression-depth measurement exists for.
    """
    mask = build_mask([_b(i * 0.5, 0.95, unc=0.05, nontarget=0.9) for i in range(8)], "speech", profile=PROFILE)
    assert all(r.state != "nontarget_active" for r in mask.regions)


def test_absent_nontarget_evidence_leaves_the_old_verdict() -> None:
    """A run without scene evidence must behave as before rather than losing its mask."""
    mask = build_mask([_b(i * 0.5, 0.02, unc=0.05) for i in range(8)], "speech", profile=PROFILE)
    assert {r.state for r in mask.regions} == {"target_free"}


def test_regions_of_interest_are_reported_separately_from_masked_time() -> None:
    """A consumer asks two different questions: what is clean, and what is worth a look."""
    buckets = [_b(i * 0.5, 0.95, unc=0.05) for i in range(4)]
    buckets += [_b(2.0 + i * 0.5, 0.02, unc=0.05, nontarget=0.8) for i in range(6)]
    doc = build_mask(buckets, "speech", profile=PROFILE).to_json()
    assert doc["nontarget_interest_s"] > 0.0
    assert "regions_of_interest" in doc


def test_a_conversation_no_longer_collapses_to_one_region() -> None:
    """The observed symptom, as a property: alternating evidence must yield time resolution."""
    buckets = []
    for i in range(20):
        speaking = (i // 4) % 2 == 0
        buckets.append(_b(i * 0.5, 0.95 if speaking else 0.02, unc=0.05, nontarget=0.7))
    mask = build_mask(buckets, "speech", profile=PROFILE)
    assert len(mask.regions) > 1


# ── heterogeneous evidence is not disagreement ─────────────────────────


def test_a_silent_breath_detector_during_speech_is_not_disagreement() -> None:
    """The cause of the useless mask, one level up from the state machine.

    For a speech task the target types are speech, breath and mouth noise, and each has its
    own detector. During ordinary speech the breath detector correctly scores ~0 — it is
    answering a different question, not contradicting the speech detector. Treating the gap
    between them as uncertainty put 0.9997 on every bucket of a 21.5 s conversation, which
    left the whole file unusable.

    Because the sources combine by maximum — any one of them establishes target activity —
    confidence far above the threshold means the verdict is not in doubt, whatever the others
    say about their own event types.
    """
    from senselab.audio.workflows.audio_analysis.background_mask import combine_target_evidence

    row = combine_target_evidence(0.0, 0.5, [0.99, 0.0], active_at=0.6, free_at=0.2)
    assert row["target_confidence"] == pytest.approx(0.99)
    assert row["uncertainty"] < 0.2


def test_evidence_near_the_threshold_is_genuinely_uncertain() -> None:
    """The honest case the change must not flatten: a decision that could go either way."""
    from senselab.audio.workflows.audio_analysis.background_mask import combine_target_evidence

    row = combine_target_evidence(0.0, 0.5, [0.55, 0.5], active_at=0.6, free_at=0.2)
    assert row["uncertainty"] > 0.5


def test_confidently_absent_evidence_is_also_certain() -> None:
    """Both ends of the decision can be confident; only the middle is doubtful."""
    from senselab.audio.workflows.audio_analysis.background_mask import combine_target_evidence

    row = combine_target_evidence(0.0, 0.5, [0.01, 0.0], active_at=0.6, free_at=0.2)
    assert row["target_confidence"] == pytest.approx(0.01)
    assert row["uncertainty"] < 0.2


def test_no_evidence_at_all_is_maximally_uncertain() -> None:
    """Absent evidence must not read as a confident "nothing here"."""
    from senselab.audio.workflows.audio_analysis.background_mask import combine_target_evidence

    row = combine_target_evidence(0.0, 0.5, [], active_at=0.6, free_at=0.2)
    assert row["uncertainty"] == pytest.approx(1.0)
