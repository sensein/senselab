"""L2 iteration: later rounds reassess signals, and the mask can withdraw trust regionally.

Round 0 fuses the L1 signals as harvested. Later rounds do two things round 0 cannot:

- **Regional trust.** A signal's weight is global in round 0, but a signal may be reliable in
  one region and not another. Once the mask says a region is target-free, a diarizer still
  claiming a speaker there has made a claim *that region* does not support, and its vote should
  be discounted **there** without being discounted everywhere.
- **Reassessment.** A better-trusted signal can prompt re-measurement inside a window, which is
  what makes the loop able to improve rather than only re-weight.

The guard throughout is that a value must not become more confident merely because it was
overwritten — uncertainty falling because of a revision is not a confidence gain.
"""

from __future__ import annotations

import pytest

from senselab.audio.workflows.audio_analysis.rounds import (
    regional_weights,
    round_converged,
)


def _region(start: float, end: float, state: str, confidence: float = 1.0) -> dict:
    return {"start": start, "end": end, "state": state, "confidence": confidence}


# ── regional trust ─────────────────────────────────────────────────────


def test_a_signal_claiming_a_speaker_in_a_target_free_region_is_discounted_there() -> None:
    """The user's criterion made regional: trust is withdrawn where it was violated."""
    weights = regional_weights(
        base_weights={"diar": 1.0},
        regions=[_region(0.0, 1.0, "target_free"), _region(1.0, 2.0, "target_active")],
        claims={"diar": [(0.0, 1.0)]},
    )
    assert weights[(0.0, 1.0)]["diar"] < 1.0
    assert weights[(1.0, 2.0)]["diar"] == pytest.approx(1.0)


def test_trust_is_not_withdrawn_globally_for_a_local_violation() -> None:
    """A signal wrong in one region may be the best evidence in another.

    Global down-weighting for a local failure is what suppressed the source that turned out to
    be right about the five speakers.
    """
    weights = regional_weights(
        base_weights={"diar": 1.0},
        regions=[_region(0.0, 1.0, "target_free"), _region(1.0, 2.0, "target_active")],
        claims={"diar": [(0.0, 1.0), (1.0, 2.0)]},
    )
    assert weights[(1.0, 2.0)]["diar"] == pytest.approx(1.0)


def test_an_uncertain_mask_region_withdraws_less_trust() -> None:
    """The mask's own confidence gates how far it may act.

    A mask that is unsure a region is target-free has not earned the right to discount a
    signal for speaking there — otherwise a guess about the mask becomes a verdict about a
    model.
    """
    confident = regional_weights(
        base_weights={"diar": 1.0},
        regions=[_region(0.0, 1.0, "target_free", confidence=1.0)],
        claims={"diar": [(0.0, 1.0)]},
    )
    unsure = regional_weights(
        base_weights={"diar": 1.0},
        regions=[_region(0.0, 1.0, "target_free", confidence=0.1)],
        claims={"diar": [(0.0, 1.0)]},
    )
    assert unsure[(0.0, 1.0)]["diar"] > confident[(0.0, 1.0)]["diar"]


def test_an_indeterminate_region_withdraws_no_trust() -> None:
    """An unresolved region is not grounds to disbelieve anyone."""
    weights = regional_weights(
        base_weights={"diar": 1.0},
        regions=[_region(0.0, 1.0, "indeterminate")],
        claims={"diar": [(0.0, 1.0)]},
    )
    assert weights[(0.0, 1.0)]["diar"] == pytest.approx(1.0)


def test_regional_trust_is_floored() -> None:
    """As everywhere else: attenuate a signal, never erase its claim."""
    weights = regional_weights(
        base_weights={"diar": 1.0},
        regions=[_region(0.0, 1.0, "target_free")],
        claims={"diar": [(0.0, 1.0)]},
    )
    assert weights[(0.0, 1.0)]["diar"] > 0.0


def test_a_signal_that_made_no_claim_in_a_region_keeps_its_weight() -> None:
    """Silence about a region is not a violation in it."""
    weights = regional_weights(
        base_weights={"diar": 1.0, "quiet": 1.0},
        regions=[_region(0.0, 1.0, "target_free")],
        claims={"diar": [(0.0, 1.0)]},
    )
    assert weights[(0.0, 1.0)]["quiet"] == pytest.approx(1.0)


# ── convergence ────────────────────────────────────────────────────────


def test_a_round_that_changes_nothing_has_converged() -> None:
    """The loop must stop when further rounds cannot move the answer."""
    previous = [{"start": 0.0, "end": 0.5, "uncertainty": 0.4}]
    assert round_converged(previous, previous, tolerance=1e-9) is True


def test_a_round_that_moves_the_answer_has_not_converged() -> None:
    """A change above tolerance means another round may still help."""
    assert (
        round_converged(
            [{"start": 0.0, "end": 0.5, "uncertainty": 0.4}],
            [{"start": 0.0, "end": 0.5, "uncertainty": 0.1}],
            tolerance=0.01,
        )
        is False
    )


def test_a_round_that_adds_a_bucket_has_not_converged() -> None:
    """New coverage is a change even when the shared buckets agree."""
    assert (
        round_converged(
            [{"start": 0.0, "end": 0.5, "uncertainty": 0.4}],
            [
                {"start": 0.0, "end": 0.5, "uncertainty": 0.4},
                {"start": 0.5, "end": 1.0, "uncertainty": 0.2},
            ],
            tolerance=0.01,
        )
        is False
    )


def test_unmeasured_buckets_compare_equal_to_each_other() -> None:
    """``None`` on both sides is agreement that nothing was measured, not a change."""
    rows = [{"start": 0.0, "end": 0.5, "uncertainty": None}]
    assert round_converged(rows, rows, tolerance=1e-9) is True


def test_a_bucket_becoming_measured_is_a_change() -> None:
    """Going from "not measured" to a value is progress the loop must not mistake for none."""
    assert (
        round_converged(
            [{"start": 0.0, "end": 0.5, "uncertainty": None}],
            [{"start": 0.0, "end": 0.5, "uncertainty": 0.3}],
            tolerance=0.01,
        )
        is False
    )


# ── the round driver ───────────────────────────────────────────────────


def _b(start: float, values: dict[str, float]) -> dict:
    return {
        "start": start,
        "end": start + 0.5,
        "votes": {k: {"same_label_uncertainty": v} for k, v in values.items()},
    }


def test_without_a_mask_the_driver_stops_at_round_zero() -> None:
    """Nothing to localise trust against.

    Later rounds would recompute identical numbers, so the reason is recorded rather than
    reporting a convergence that was never tested.
    """
    from senselab.audio.workflows.audio_analysis.fuse import fuse_rounds

    rows, log = fuse_rounds({"raw": [_b(0.0, {"a": 0.3})]}, weights={"a": 1.0}, max_rounds=5, snr_gate=None)
    assert len(rows) == 1
    assert len(log) == 1 and log[0]["converged"] is True
    assert "reason" in log[0]


def test_regional_trust_changes_the_answer_in_a_contradicted_region() -> None:
    """The point of iterating: a claim the mask contradicts carries less weight afterwards."""
    from senselab.audio.workflows.audio_analysis.fuse import fuse_rounds

    buckets = {"raw": [_b(0.0, {"honest": 0.0, "overclaimer": 1.0})]}
    round0, _ = fuse_rounds(buckets, weights={"honest": 1.0, "overclaimer": 1.0}, aggregator="min", snr_gate=None)
    later, log = fuse_rounds(
        buckets,
        weights={"honest": 1.0, "overclaimer": 1.0},
        aggregator="min",
        mask_regions=[_region(0.0, 0.5, "target_free", confidence=1.0)],
        speaker_claims={"overclaimer": [(0.0, 0.5)]},
        max_rounds=3,
        snr_gate=None,
    )
    assert later[0]["triage_score"] < round0[0]["triage_score"]
    assert any(entry["regional_trust_applied"] for entry in log)


def test_the_log_distinguishes_converged_from_out_of_rounds() -> None:
    """A bare result cannot say which happened, and they call for different follow-up."""
    from senselab.audio.workflows.audio_analysis.fuse import fuse_rounds

    _rows, log = fuse_rounds(
        {"raw": [_b(0.0, {"a": 0.3})]},
        weights={"a": 1.0},
        mask_regions=[_region(0.0, 0.5, "target_free")],
        speaker_claims={"a": [(0.0, 0.5)]},
        max_rounds=3,
        snr_gate=None,
    )
    assert {"round", "converged", "regional_trust_applied"} <= set(log[-1])


def test_c4_counts_this_loop_s_own_actions_rather_than_going_unmeasured() -> None:
    """C4 asks whether a region still has an action nobody tried, and this loop has exactly one.

    Its action set is what it can do *without new measurement*: withdraw regional trust where the
    mask contradicts a claim. Once the tightened weights apply every such region, none remains
    untried — a measured zero, not a defaulted one, so C4 stops blocking on the honest grounds
    that the inventory was actually taken.
    """
    from senselab.audio.workflows.audio_analysis.fuse import fuse_rounds

    _rows, log = fuse_rounds(
        {"raw": [_b(0.0, {"a": 0.3})]},
        weights={"a": 1.0},
        mask_regions=[_region(0.0, 0.5, "target_free", confidence=1.0)],
        speaker_claims={"a": [(0.0, 0.5)]},
        max_rounds=3,
        snr_gate=None,
    )
    assert "c4" not in log[-1]["blocking"], "the one available action was applied, and it was counted"
    assert log[-1]["action_scope"] == "regional_trust"


def test_the_action_scope_is_named_so_convergence_is_not_read_too_widely() -> None:
    """A loop running out of moves is a narrower claim than no measurement helping.

    The adaptive catalogue can still re-run models over the same region. Recording which inventory
    was counted keeps a fusion-round convergence from being read as the wider statement.
    """
    from senselab.audio.workflows.audio_analysis.fuse import fuse_rounds

    _rows, log = fuse_rounds(
        {"raw": [_b(0.0, {"a": 0.3})]},
        weights={"a": 1.0},
        mask_regions=[_region(0.0, 0.5, "target_free", confidence=1.0)],
        speaker_claims={"a": [(0.0, 0.5)]},
        max_rounds=3,
        snr_gate=None,
    )
    assert log[-1]["action_scope"] == "regional_trust"


def test_a_caller_with_a_wider_inventory_overrides_the_loop_s_own_count() -> None:
    """The adaptive loop knows about interventions this one cannot see.

    When it says actions remain, C4 must block even though the fusion loop has exhausted its own —
    otherwise the narrower inventory silently answers the wider question.
    """
    from senselab.audio.workflows.audio_analysis.fuse import fuse_rounds

    _rows, log = fuse_rounds(
        {"raw": [_b(0.0, {"a": 0.3})]},
        weights={"a": 1.0},
        mask_regions=[_region(0.0, 0.5, "target_free", confidence=1.0)],
        speaker_claims={"a": [(0.0, 0.5)]},
        max_rounds=3,
        untried_actions=2,
        snr_gate=None,
    )
    assert "c4" in log[-1]["blocking"]
    assert log[-1]["action_scope"] == "caller_supplied"


def test_the_round_zero_shortcut_does_not_claim_the_criteria_were_checked() -> None:
    """With no mask there is no round 1 to run, which is not the same as having converged.

    The loop stops because it cannot iterate, so the reason must say so rather than letting a
    reader take `converged` here for the four-criteria verdict it is everywhere else.
    """
    from senselab.audio.workflows.audio_analysis.fuse import fuse_rounds

    _rows, log = fuse_rounds({"raw": [_b(0.0, {"a": 0.3})]}, weights={"a": 1.0}, max_rounds=5, snr_gate=None)
    assert log[0]["criteria_evaluated"] is False


def test_every_round_records_its_index() -> None:
    """Rows carry the round that produced them, so successive maps are comparable."""
    from senselab.audio.workflows.audio_analysis.fuse import fuse_rounds

    rows, _log = fuse_rounds(
        {"raw": [_b(0.0, {"a": 0.3})]},
        weights={"a": 1.0},
        mask_regions=[_region(0.0, 0.5, "target_free")],
        speaker_claims={"a": [(0.0, 0.5)]},
        max_rounds=3,
        snr_gate=None,
    )
    assert rows[0]["round"] >= 1
