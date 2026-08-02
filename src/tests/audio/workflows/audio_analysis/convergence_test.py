"""Convergence for L2 rounds: C1-C4, cycle detection, and the two guards.

Stability alone is not convergence. Each criterion exists because something can settle while
something else is still moving, and the two guards exist because the obvious definitions of
"improving" are both wrong: a drop caused by overwriting a value is not a confidence gain, and a
rise caused by a refutation is not a failure.
"""

from __future__ import annotations

import pytest

from senselab.audio.workflows.audio_analysis.rounds import (
    RoundRecord,
    assess_convergence,
    detect_non_convergence,
)


def _r(idx: int, **kw: object) -> RoundRecord:
    # Distinct signature per round by default: the signature digests the round's state, so two
    # rounds that differ in any other field must not silently claim to be the same state. The
    # cycle test repeats one deliberately.
    base = dict(
        round_index=idx,
        epistemic=0.5,
        assignment={"S0": "speaker#1"},
        measured_buckets=10,
        untried_actions=0,
        overwrote_values=False,
        signature=f"s{idx}",
    )
    base.update(kw)
    return RoundRecord(**base)  # type: ignore[arg-type]


def test_all_four_criteria_must_hold() -> None:
    """Any one criterion alone would stop the loop while something was still moving."""
    history = [_r(0), _r(1)]
    assert assess_convergence(history)["converged"] is True

    # C1 alone fails: epistemic uncertainty still falling.
    assert assess_convergence([_r(0, epistemic=0.9), _r(1, epistemic=0.5)])["criteria"]["c1"] is False
    # C2: the speaker-to-channel assignment flipped, even though the numbers settled.
    flipped = assess_convergence([_r(0), _r(1, assignment={"S0": "speaker#2"})])
    assert flipped["criteria"]["c2"] is False and flipped["converged"] is False
    # C3: a bucket went unmeasured -> measured. New coverage is progress, not stability.
    grew = assess_convergence([_r(0, measured_buckets=10), _r(1, measured_buckets=14)])
    assert grew["criteria"]["c3"] is False and grew["converged"] is False
    # C4: an action nobody has tried yet means "converged" would mean "ran out of ideas".
    idle = assess_convergence([_r(0), _r(1, untried_actions=3)])
    assert idle["criteria"]["c4"] is False and idle["converged"] is False


def test_a_drop_from_overwriting_a_value_is_not_progress() -> None:
    """The self-confirmation guard: uncertainty falling because a value was replaced proves nothing.

    Without this, a round that overwrites a signal with its own preferred answer reports a
    confidence gain and buys itself more rounds on the strength of it.
    """
    honest = assess_convergence([_r(0, epistemic=0.9), _r(1, epistemic=0.4, overwrote_values=False)])
    self_confirmed = assess_convergence([_r(0, epistemic=0.9), _r(1, epistemic=0.4, overwrote_values=True)])
    assert honest["criteria"]["c1"] is False, "a genuine fall means the loop is still learning"
    assert self_confirmed["criteria"]["c1"] is True, "an overwritten fall earns no credit"
    assert self_confirmed["credited_epistemic_change"] == pytest.approx(0.0)


def test_a_rise_from_a_refutation_is_a_legitimate_outcome() -> None:
    """Convergence is not monotone decrease.

    Defining it that way would bias the loop toward ratifying whatever round 0 said and leave the
    confirmation half of the design unable to change anything. A refuting round moved things, so
    the loop has not converged — but it is not an error either.
    """
    result = assess_convergence([_r(0, epistemic=0.3), _r(1, epistemic=0.8)])
    assert result["criteria"]["c1"] is False
    assert result["converged"] is False
    assert result["diverged"] is True
    assert result["stop_reason"] is None, "divergence continues the loop rather than aborting it"


def test_an_oscillation_is_caught_rather_than_burning_the_budget() -> None:
    """D-12: with mutual influence an A->B->A->B cycle is plausible and moves every round.

    Slow-convergence detection cannot catch it — every round reports change — so the cycle has to
    be recognised by state repeating, not by movement stopping.
    """
    history = [_r(0, signature="a"), _r(1, signature="b", epistemic=0.7), _r(2, signature="a", epistemic=0.5)]
    result = assess_convergence(history)
    assert result["stop"] is True
    assert result["stop_reason"] == "oscillation"
    assert result["converged"] is False, "a cycle is a reason to stop, not evidence of agreement"


def test_a_frozen_state_that_still_blocks_a_criterion_is_stagnation_not_oscillation() -> None:
    """Flip-flopping and standing still are different failures, so they get different names.

    A loop trading two interpretations needs the disagreement resolved; a loop repeating one state
    while a criterion still blocks has nothing left to contribute and needs a new action. Reporting
    both as "cycle" told an operator to look for a conflict that was not there.
    """
    frozen = [_r(0, signature="x", untried_actions=None), _r(1, signature="x", untried_actions=None)]
    result = assess_convergence(frozen)
    assert result["converged"] is False, "C4 was never measured, so it cannot pass"
    assert result["stop_reason"] == "no_improvement"


def test_convergence_uses_the_shared_detector_rather_than_its_own_cycle_check() -> None:
    """One implementation, so the two loops cannot disagree about the same history.

    ``adaptive/convergence.py`` already detected oscillation and stagnation; a second, cruder check
    living here meant the fusion rounds and the adaptive loop could reach opposite verdicts on
    identical round states.
    """
    from senselab.audio.workflows.audio_analysis.adaptive import convergence as adaptive_convergence

    assert adaptive_convergence.detect_non_convergence is detect_non_convergence


# ── the shared detector itself (moved down from adaptive/, FR-011e) ──────────


def test_alternating_states_detected_as_oscillation() -> None:
    """Two interpretations trading places is the signature."""
    reason, states = detect_non_convergence([{"count": 1}, {"count": 4}, {"count": 1}], window=3)
    assert reason == "oscillation"
    assert len(states) == 2


def test_steady_progress_is_not_oscillation() -> None:
    """Monotone improvement is the healthy case."""
    reason, _ = detect_non_convergence([{"u": 0.9}, {"u": 0.6}, {"u": 0.3}], window=3)
    assert reason is None


def test_repeating_the_same_state_is_stagnation_not_oscillation() -> None:
    """Standing still and flip-flopping are different failures with different remedies."""
    reason, _ = detect_non_convergence([{"u": 0.5}, {"u": 0.5}, {"u": 0.5}], window=3)
    assert reason == "no_improvement"


def test_window_shorter_than_two_cannot_detect_alternation() -> None:
    """A one-round window cannot observe a flip-flop."""
    with pytest.raises(ValueError, match="window"):
        detect_non_convergence([{"a": 1}], window=1)


def test_too_few_rounds_yields_no_verdict() -> None:
    """One round is not evidence of a pattern."""
    reason, _ = detect_non_convergence([{"a": 1}], window=3)
    assert reason is None


def test_three_way_cycle_is_still_oscillation() -> None:
    """Alternation is not limited to period two."""
    reason, _ = detect_non_convergence([{"c": 1}, {"c": 2}, {"c": 3}, {"c": 1}, {"c": 2}, {"c": 3}], window=6)
    assert reason == "oscillation"


def test_oscillation_states_are_the_repeating_ones() -> None:
    """The report names which interpretations are trading places."""
    _reason, states = detect_non_convergence([{"count": 1}, {"count": 4}, {"count": 1}, {"count": 4}], window=4)
    assert {frozenset(s.items()) for s in states} == {frozenset({"count": 1}.items()), frozenset({"count": 4}.items())}


def test_a_repeat_that_fell_out_of_the_window_is_no_longer_cycling() -> None:
    """The window bounds recency, which is the point of having one.

    A state that recurred early and has not since is not *currently* oscillating, and stopping the
    loop for it would end a run that had started making progress.
    """
    history = [{"c": 1}, {"c": 2}, {"c": 1}, {"c": 3}, {"c": 4}, {"c": 5}]
    assert detect_non_convergence(history, window=3)[0] is None


def test_the_round_cap_stops_the_loop_and_says_so() -> None:
    """Running out of rounds must be distinguishable from agreeing."""
    history = [_r(i, epistemic=0.9 - 0.01 * i, signature=f"s{i}") for i in range(10)]
    result = assess_convergence(history, max_rounds=10)
    assert result["stop"] is True
    assert result["stop_reason"] == "max_rounds"
    assert result["converged"] is False


def test_a_single_round_is_not_yet_a_convergence_claim() -> None:
    """One observation cannot show stability, the same rule reliability already follows."""
    result = assess_convergence([_r(0)])
    assert result["converged"] is False
    assert result["stop"] is False
    assert result["criteria"]["c1"] is False


def test_convergence_reports_which_criterion_blocked_it() -> None:
    """A bare boolean cannot tell an operator what to change."""
    result = assess_convergence([_r(0), _r(1, untried_actions=2, measured_buckets=11)])
    assert set(result["blocking"]) == {"c3", "c4"}


def test_an_unmeasured_criterion_blocks_rather_than_passes() -> None:
    """A criterion nobody measured must not read as one that passed.

    This is the difference between "we checked four things" and "we checked two and defaulted the
    rest". ``None`` means unmeasured for both the assignment and the action inventory, and an
    absent measurement cannot license a convergence claim — the same rule support and reliability
    already follow for a factor that was never gathered.
    """
    unmeasured = [_r(0, assignment=None, untried_actions=None), _r(1, assignment=None, untried_actions=None)]
    result = assess_convergence(unmeasured)
    assert result["criteria"]["c2"] is False
    assert result["criteria"]["c4"] is False
    assert result["converged"] is False
    assert set(result["blocking"]) == {"c2", "c4"}


def test_zero_untried_actions_is_a_measurement_and_does_pass() -> None:
    """Having *checked* that no action remains is different from never having looked."""
    result = assess_convergence([_r(0, untried_actions=0), _r(1, untried_actions=0)])
    assert result["criteria"]["c4"] is True


def test_the_round_record_reads_fields_that_exist_on_fused_rows() -> None:
    """`_round_record` read `within_pass_uncertainty` — an L1-only column — off L2 rows.

    `fuse_axis` emits `epistemic_uncertainty`. So every RoundRecord built from a fused round carried
    `epistemic=None` and `measured_buckets=0`, and its signature digested a column of `None`s. C1
    had no value to compare, C3 compared 0 to 0, and every round produced an identical signature —
    which the shared detector then correctly reported as a repeating state. A real run stopped with
    `oscillation` on all four axes for exactly this reason: not four dynamics agreeing, one field
    name resolving to nothing on all of them.

    A name that resolves to nothing is indistinguishable from a value that means nothing, which is
    why this survived being looked at twice.
    """
    from senselab.audio.workflows.audio_analysis.fuse import _round_record, fuse_axis

    rows = fuse_axis(
        {
            "p": [
                {"start": 0.0, "end": 0.5, "votes": {"a": {"same_label_uncertainty": 0.4}}},
                {"start": 0.5, "end": 1.0, "votes": {"a": {"same_label_uncertainty": 0.9}}},
            ]
        },
        weights={"a": 1.0},
    )
    record = _round_record(0, rows, untried_actions=0)
    assert record.epistemic is not None, "C1 cannot judge a value the record never read"
    assert record.measured_buckets == 2, "both buckets were measured"


def test_two_rounds_with_different_values_have_different_signatures() -> None:
    """Cycle detection is meaningless if every round digests to the same string.

    With the field name wrong, every signature was the digest of `None;None;...` — so the detector
    saw a repeating state on the second round of every run, and reported oscillation.
    """
    from senselab.audio.workflows.audio_analysis.fuse import _round_record, fuse_axis

    def _rows(value: float) -> list:
        return fuse_axis(
            {"p": [{"start": 0.0, "end": 0.5, "votes": {"a": {"same_label_uncertainty": value}}}]},
            weights={"a": 1.0},
        )

    # 0.5 is maximum entropy, 0.02 near-minimum. Not 0.2 vs 0.8: `uncertainty` is the entropy of
    # {settled, unsettled}, which is symmetric — those two are genuinely the same uncertainty, and
    # a signature that distinguished them would be reporting a difference that is not there.
    a = _round_record(0, _rows(0.5), untried_actions=0)
    b = _round_record(1, _rows(0.02), untried_actions=0)
    assert a.signature != b.signature
