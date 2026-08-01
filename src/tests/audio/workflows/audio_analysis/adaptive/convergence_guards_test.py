"""What the convergence report says when the loop did *not* settle (T079/T082, FR-011e/h).

A mutually-influencing loop can fail to settle in two ways a one-directional pipeline cannot:
it can **oscillate** between interpretations that each imply the other is wrong, or it can
grind without improving. Either way, emitting whichever state the last round happened to
produce would present an unsettled value as settled — so the report must say which condition
it hit (FR-011e), and any quantity that never converged must be named rather than sitting
silently beside ones that did (FR-011h).

The detector itself is tested at ``audio_analysis/convergence_test.py``, where it lives; these
tests cover what the adaptive report does with its verdict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from senselab.audio.workflows.audio_analysis.adaptive.convergence import (
    build_convergence_report,
    unresolved_quantities,
)


class _State:
    """Minimal stand-in for BeliefState: no open buckets, so nothing distracts from the verdict."""

    def axis_rows(self, _stream: str, _axis: str) -> list[dict[str, Any]]:
        return []

    def uncertainty_mass(self, _stream: str, _axis: str, _theta: float) -> float:
        return 0.0


class _Ledger:
    def as_dict(self) -> dict[str, Any]:
        return {}


def _report(**kw: Any) -> dict[str, Any]:  # noqa: ANN401 — the callee is Any-typed by design
    base: dict[str, Any] = dict(
        state=_State(),
        passes=["raw"],
        policy={"thresholds": {"theta_low": 0.2}},
        rounds=[{"round": 2}, {"round": 3}],
        ledger=_Ledger(),
        iterations=[],
        run_state="converged",
        provenance={},
    )
    base.update(kw)
    return build_convergence_report(**base)


# ── termination reason (FR-011e, SC-028) ──────────────────────────────


def test_an_oscillating_run_is_not_reported_as_converged() -> None:
    """The invariant from the contract: oscillation implies not converged, with the states named.

    Without this the loop stopped because nothing more would fire and reported that as agreement,
    which is exactly the "unsettled value presented as settled" the requirement exists to prevent.
    """
    report = _report(run_state="converged", round_states=[{"count": 1}, {"count": 4}, {"count": 1}])
    assert report["termination_reason"] == "oscillation"
    assert report["converged"] is False
    assert len(report["oscillation_states"]) == 2


def test_a_stagnant_run_says_so_rather_than_claiming_budget() -> None:
    """Standing still and running out of rounds are different diagnoses."""
    report = _report(run_state="max_rounds", round_states=[{"u": 0.5}, {"u": 0.5}, {"u": 0.5}])
    assert report["termination_reason"] == "no_improvement"
    assert report["converged"] is False


def test_running_out_of_rounds_reports_budget() -> None:
    """`max_rounds` is a budget outcome, not an agreement."""
    report = _report(run_state="max_rounds", round_states=[{"u": 0.9}, {"u": 0.6}, {"u": 0.3}])
    assert report["termination_reason"] == "budget"
    assert report["converged"] is False
    assert report["oscillation_states"] == []


def test_a_settled_run_reports_convergence() -> None:
    """Healthy progress and a clean stop."""
    report = _report(run_state="converged", round_states=[{"u": 0.9}, {"u": 0.6}, {"u": 0.3}])
    assert report["termination_reason"] == "converged"
    assert report["converged"] is True


def test_a_settled_run_holding_still_is_not_stagnation() -> None:
    """Holding still is what settling looks like, so it cannot be evidence against settling.

    The loop reports `converged` on its own grounds — no region above θ_low and nothing rejected.
    Reading the accompanying flat state as `no_improvement` would make every clean convergence
    report as a failure to converge.
    """
    report = _report(run_state="converged", round_states=[{"u": 0.0}, {"u": 0.0}, {"u": 0.0}])
    assert report["termination_reason"] == "converged"
    assert report["converged"] is True


def test_a_settled_run_that_was_still_trading_places_is_not_converged() -> None:
    """Oscillation is the one verdict that outranks the loop's own stop.

    "Nothing left to fire" says the loop ran out of moves, not that the two interpretations it was
    alternating between resolved — those values are unsettled whatever the loop concluded.
    """
    report = _report(run_state="converged", round_states=[{"count": 1}, {"count": 4}, {"count": 1}])
    assert report["termination_reason"] == "oscillation"
    assert report["converged"] is False


def test_an_unrecognised_run_state_passes_through_rather_than_being_relabelled() -> None:
    """A new run state must not be silently absorbed into "budget".

    Mapping the unknown onto a known outcome is how a state nobody has thought about starts
    reading as one that was.
    """
    report = _report(run_state="no_speech", round_states=[{"u": 0.1}, {"u": 0.1, "x": 1}])
    assert report["termination_reason"] == "no_speech"
    assert report["converged"] is False


def test_rounds_run_counts_the_initial_round_too() -> None:
    """Round 1 established the beliefs the later rounds acted on; it is a round."""
    assert _report(rounds=[{"round": 2}, {"round": 3}])["rounds_run"] == 3


# ── unresolved quantities (T079, FR-011h) ─────────────────────────────


def test_untracked_resolutions_are_reported_as_unmeasured_not_as_none_unresolved() -> None:
    """An empty list would read as "we checked and everything settled".

    Never having taken the inventory is a different statement from having taken it and found
    nothing outstanding, and only one of them licenses trusting the other artifacts.
    """
    assert _report()["unresolved_quantities"] is None


def test_tracked_resolutions_are_summarised_in_the_report() -> None:
    """When the caller does know, the unresolved names travel with the verdict."""
    report = _report(per_quantity={"count_posterior": "unresolved", "speakers.S0.span": "new_evidence"})
    assert report["unresolved_quantities"] == ["count_posterior"]
    assert report["per_quantity"] == {"count_posterior": "unresolved", "speakers.S0.span": "new_evidence"}


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


# ── layout drift must be loud, not silent (found on a real run) ──────────


def test_a_missing_outcomes_directory_warns_rather_than_returning_empty_silently(
    tmp_path: Path,
    capsys: "pytest.CaptureFixture[str]",
) -> None:
    """An absent directory and a stage that produced nothing are different facts.

    They were indistinguishable: both returned ``{}``. When pass outputs moved under ``L1/`` and
    the loader kept rebuilding ``run_dir / stream / task``, every lookup returned empty, the ASR
    fusion path received nothing, and the run emitted a transcript with no words and no error
    anywhere. The layout change was the bug; the silence is what let it reach the output.
    """
    from senselab.audio.workflows.audio_analysis.adaptive.interventions import load_outcomes_dir

    assert load_outcomes_dir(tmp_path, "raw_16k", "asr") == {}
    assert "no 'asr' outcomes directory" in capsys.readouterr().err
