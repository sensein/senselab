"""Convergence marking, round summaries, and the final report (FR-017/018/019)."""

from __future__ import annotations

from typing import Any

from senselab.audio.workflows.audio_analysis.adaptive.belief import AXES, bucket_key
from senselab.audio.workflows.audio_analysis.adaptive.types import PlannedIntervention
from senselab.audio.workflows.audio_analysis.rounds import detect_non_convergence

__all__ = [
    "apply_convergence_marks",
    "build_convergence_report",
    "round_summary",
    "unresolved_quantities",
]


def apply_convergence_marks(
    state: Any,  # noqa: ANN401 — BeliefState
    *,
    policy: dict[str, Any],
    touch_counts: dict[tuple[str, tuple[float, float]], int],
    budget_left: bool,
) -> dict[str, int]:
    """Update per-bucket status per FR-017; returns counts of status transitions.

    - ``converged``: uncertainty ≤ θ_low.
    - ``irreducible``: touched ≥ max_region_rounds with < ε improvement AND a *measured* aleatoric
      floor explains the residual (reason ``snr_floor``) — or, without floor cover, marked
      ``irreducible: no_reduction_under_available_interventions``.
    - ``budget_exhausted``: interventions still wanted but the ledger is empty.

    A bucket has one status because it has one belief. While the state held one row per (pass,
    axis) this loop could mark the same span converged on raw and budget-exhausted on enhanced,
    and the report then counted both.

    An **unmeasured** floor cannot explain anything: ``aleatoric_floor`` is ``None`` where nothing
    about the scene was measured, and that case falls to
    ``no_reduction_under_available_interventions`` rather than being read as a floor of zero.
    """
    th = policy["thresholds"]
    theta_low, epsilon = float(th["theta_low"]), float(th["epsilon"])
    max_touch = int(policy["regions"]["max_region_rounds"])
    transitions = {"converged": 0, "irreducible": 0, "budget_exhausted": 0}
    for axis in AXES:
        for row in state.axis_rows(axis):
            if row.get("status") != "open":
                continue
            u = row.get("uncertainty")
            if u is None:
                continue
            if u <= theta_low:
                row["status"] = "converged"
                transitions["converged"] += 1
                continue
            touches = touch_counts.get((axis, bucket_key(row["start"], row["end"])), 0)
            hist = row.get("history") or []
            improvement = None
            if len(hist) >= 2:
                improvement = hist[-2]["uncertainty"] - hist[-1]["uncertainty"]
            stalled = improvement is not None and improvement < epsilon
            if touches >= max_touch and stalled:
                floor = row.get("aleatoric_floor")
                row["status"] = "irreducible"
                row["irreducible_reason"] = (
                    "snr_floor"
                    if floor is not None and u <= float(floor) + epsilon
                    else "no_reduction_under_available_interventions"
                )
                transitions["irreducible"] += 1
            elif touches >= 1 and not budget_left:
                row["status"] = "budget_exhausted"
                transitions["budget_exhausted"] += 1
    return transitions


def round_summary(
    *,
    round_idx: int,
    state: Any,  # noqa: ANN401
    policy: dict[str, Any],
    fired: list[PlannedIntervention],
    not_admitted: list[PlannedIntervention],
    mass_before: dict[str, float],
    ledger: Any,  # noqa: ANN401
) -> dict[str, Any]:
    """One ``rounds/<k>/summary.json`` payload."""
    theta_low = float(policy["thresholds"]["theta_low"])
    mass_after = {a: round(state.uncertainty_mass(a, theta_low), 6) for a in AXES}
    statuses: dict[str, int] = {}
    for a in AXES:
        for row in state.axis_rows(a):
            statuses[row.get("status", "open")] = statuses.get(row.get("status", "open"), 0) + 1
    return {
        "round": round_idx,
        "interventions": {
            "fired": [c["intervention_id"] for c in fired],
            "deferred_budget": [c["rule"] for c in not_admitted if c["status"] == "deferred_budget"],
            "blocked_guard": [
                {"rule": c["rule"], "reason": c.get("error")} for c in not_admitted if c["status"] == "blocked_guard"
            ],
            "failed": [c["intervention_id"] for c in fired if c.get("exec_status") == "failed"],
        },
        "budget": ledger.as_dict(),
        "uncertainty_mass": {"before": mass_before, "after": mass_after},
        "bucket_statuses": statuses,
    }


_TERMINATION_BY_RUN_STATE = {
    "converged": "converged",
    "max_rounds": "budget",
    "budget_exhausted": "budget",
    "no_runnable_interventions": "budget",
}
"""Run states whose termination meaning is settled. Anything absent passes through unchanged
rather than being folded into ``budget`` — mapping an unrecognised state onto a known outcome is
how a state nobody has thought about starts reading as one that was."""


def build_convergence_report(
    *,
    state: Any,  # noqa: ANN401
    policy: dict[str, Any],
    rounds: list[dict[str, Any]],
    ledger: Any,  # noqa: ANN401
    iterations: list[dict[str, Any]],
    run_state: str,
    provenance: dict[str, Any],
    round_states: list[dict[str, Any]] | None = None,
    per_quantity: dict[str, str] | None = None,
    window: int = 3,
) -> dict[str, Any]:
    """``final/convergence.json`` per data-model.md ConvergenceReport.

    Args:
        state: Belief state to summarise.
        policy: Active policy.
        rounds: Per-round summaries for rounds 2..K.
        ledger: Budget ledger.
        iterations: Intervention entries across the run.
        run_state: Why the loop stopped, in the loop's own vocabulary.
        provenance: Fields merged into the report verbatim.
        round_states: Per-round state snapshots, oldest first, for non-convergence detection
            (FR-011e). Omit when the loop did not track them.
        per_quantity: ``{quantity → resolution kind}``. ``None`` means the inventory was never
            taken, which is reported as such: an empty list would read as "we checked and
            everything settled".
        window: Rounds inspected for oscillation or stagnation. Three, not the fusion rounds'
            four: this loop's rounds are expensive, so a run rarely gets deep enough for a
            longer window to see anything a shorter one missed.

    Returns:
        The report dict. ``termination_reason`` overrides ``run_state`` when the detector fires —
        a loop that stopped because nothing more would fire has still not settled if its state was
        trading places, and reporting that as agreement is the failure FR-011e exists to prevent.
    """
    per_axis: dict[str, Any] = {}
    irreducible_regions: list[dict[str, Any]] = []
    for axis in AXES:
        rows = state.axis_rows(axis)
        counts: dict[str, int] = {}
        for row in rows:
            counts[row.get("status", "open")] = counts.get(row.get("status", "open"), 0) + 1
            if row.get("status") == "irreducible":
                irreducible_regions.append(
                    {
                        "axis": axis,
                        "start": row["start"],
                        "end": row["end"],
                        "reason": row.get("irreducible_reason"),
                        "residual": row.get("uncertainty"),
                        "floor": row.get("aleatoric_floor"),
                        "floor_policy": row.get("aleatoric_floor_policy"),
                    }
                )
        per_axis[axis] = {
            "buckets": len(rows),
            **counts,
            "residual_mass": round(state.uncertainty_mass(axis, float(policy["thresholds"]["theta_low"])), 6),
        }
    next_actions = [
        {
            "rule": e["rule"],
            "region_id": e.get("region_id"),
            "priority": e.get("priority"),
            "status": e["status"],
            "reason": e.get("error"),
        }
        for e in iterations
        if e["status"] in ("deferred_budget", "blocked_guard")
    ]
    repeat_reason, repeating = detect_non_convergence(round_states or [], window=window)
    # A settled loop *holds still* — that is what settling looks like — so a frozen state cannot
    # be evidence against a convergence the loop reported on its own grounds (no region above
    # θ_low, nothing rejected). Trading places is different: those values are unsettled whatever
    # the loop concluded, which is the case FR-011e exists for.
    if repeat_reason == "no_improvement" and run_state == "converged":
        repeat_reason, repeating = None, []
    termination_reason = repeat_reason or _TERMINATION_BY_RUN_STATE.get(run_state, run_state)
    return {
        "run_state": run_state,
        "converged": termination_reason == "converged",
        "rounds_run": len(rounds) + 1,
        "termination_reason": termination_reason,
        "oscillation_states": repeating,
        "per_quantity": per_quantity,
        "unresolved_quantities": None if per_quantity is None else unresolved_quantities(per_quantity),
        "rounds": rounds,
        "per_axis": per_axis,
        "irreducible_regions": irreducible_regions,
        "budget": ledger.as_dict(),
        "next_actions": next_actions,
        **provenance,
    }


# ── non-convergence reporting (T082, FR-011e / FR-011h) ────────────────
#
# The detector itself lives in ``rounds.detect_non_convergence``: both this loop and the L2
# fusion rounds ask the same question of a round history, and two implementations of it could
# disagree about identical states. The dependency runs adaptive → workflow, so the shared piece
# belongs at the lower level.


def unresolved_quantities(per_quantity: dict[str, str]) -> list[str]:
    """Names of quantities that never converged, sorted (FR-011h).

    ``revision`` is deliberately *not* counted: it is resolved-but-not-improved, which is a
    separate report from unresolved.
    """
    return sorted(name for name, kind in per_quantity.items() if kind == "unresolved")
