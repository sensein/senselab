"""Non-convergence detection (T076/T079, FR-011e / FR-011h).

A mutually-influencing loop can fail to settle in two ways a one-directional pipeline
cannot: it can **oscillate** between interpretations that each imply the other is wrong, or
it can grind without improving. Either way, emitting whichever state the last round happened
to produce would present an unsettled value as settled — so the loop must terminate and say
which condition it hit (FR-011e), and any quantity that never converged must be reported as
such rather than sitting silently beside ones that did (FR-011h).
"""

from __future__ import annotations

import pytest

from senselab.audio.workflows.audio_analysis.adaptive.convergence import (
    detect_non_convergence,
    unresolved_quantities,
)

# ── oscillation (T076, FR-011e / SC-028) ──────────────────────────────


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


# ── unresolved quantities (T079, FR-011h) ─────────────────────────────


def test_unresolved_quantities_are_listed() -> None:
    """A value that never converged must not sit silently beside ones that did."""
    per_quantity = {"a": "new_evidence", "b": "unresolved", "c": "revision"}
    assert unresolved_quantities(per_quantity) == ["b"]


def test_revision_only_resolution_is_not_counted_as_unresolved() -> None:
    """`revision` is resolved-but-not-improved, which is a different report."""
    assert unresolved_quantities({"a": "revision"}) == []


def test_unresolved_list_is_sorted_for_determinism() -> None:
    """Stable order, so output stays byte-identical."""
    assert unresolved_quantities({"z": "unresolved", "a": "unresolved"}) == ["a", "z"]


def test_empty_input_yields_no_unresolved() -> None:
    """Nothing tracked means nothing unresolved."""
    assert unresolved_quantities({}) == []
