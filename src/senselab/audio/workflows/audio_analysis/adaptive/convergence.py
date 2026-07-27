"""Convergence marking, round summaries, and the final report (FR-017/018/019)."""

from __future__ import annotations

from typing import Any

from senselab.audio.workflows.audio_analysis.adaptive.belief import AXES, bucket_key
from senselab.audio.workflows.audio_analysis.adaptive.types import PlannedIntervention


def apply_convergence_marks(
    state: Any,  # noqa: ANN401 — BeliefState
    *,
    passes: list[str],
    policy: dict[str, Any],
    touch_counts: dict[tuple[str, str, tuple[float, float]], int],
    budget_left: bool,
) -> dict[str, int]:
    """Update per-bucket status per FR-017; returns counts of status transitions.

    - ``converged``: uncertainty ≤ θ_low.
    - ``irreducible``: touched ≥ max_region_rounds with < ε improvement AND the
      aleatoric floor explains the residual (prototype floor = quality only;
      reason ``snr_floor``) — or, without floor cover, marked
      ``irreducible: no_reduction_under_available_interventions``.
    - ``budget_exhausted``: interventions still wanted but the ledger is empty.
    """
    th = policy["thresholds"]
    theta_low, epsilon = float(th["theta_low"]), float(th["epsilon"])
    max_touch = int(policy["regions"]["max_region_rounds"])
    transitions = {"converged": 0, "irreducible": 0, "budget_exhausted": 0}
    for stream in passes:
        for axis in AXES:
            for row in state.axis_rows(stream, axis):
                if row.get("status") != "open":
                    continue
                u = row.get("aggregated_uncertainty")
                if u is None:
                    continue
                if u <= theta_low:
                    row["status"] = "converged"
                    transitions["converged"] += 1
                    continue
                touches = touch_counts.get((stream, axis, bucket_key(row["start"], row["end"])), 0)
                hist = row.get("history") or []
                improvement = None
                if len(hist) >= 2:
                    improvement = hist[-2]["aggregated_uncertainty"] - hist[-1]["aggregated_uncertainty"]
                stalled = improvement is not None and improvement < epsilon
                if touches >= max_touch and stalled:
                    floor = float(row.get("aleatoric_floor") or 0.0)
                    if u <= floor + epsilon:
                        row["status"] = "irreducible"
                        row["irreducible_reason"] = "snr_floor"
                    else:
                        row["status"] = "irreducible"
                        row["irreducible_reason"] = "no_reduction_under_available_interventions"
                    transitions["irreducible"] += 1
                elif touches >= 1 and not budget_left:
                    row["status"] = "budget_exhausted"
                    transitions["budget_exhausted"] += 1
    return transitions


def round_summary(
    *,
    round_idx: int,
    state: Any,  # noqa: ANN401
    passes: list[str],
    policy: dict[str, Any],
    fired: list[PlannedIntervention],
    not_admitted: list[PlannedIntervention],
    mass_before: dict[str, float],
    ledger: Any,  # noqa: ANN401
) -> dict[str, Any]:
    """One ``rounds/<k>/summary.json`` payload."""
    theta_low = float(policy["thresholds"]["theta_low"])
    mass_after = {f"{s}/{a}": round(state.uncertainty_mass(s, a, theta_low), 6) for s in passes for a in AXES}
    statuses: dict[str, int] = {}
    for s in passes:
        for a in AXES:
            for row in state.axis_rows(s, a):
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


def build_convergence_report(
    *,
    state: Any,  # noqa: ANN401
    passes: list[str],
    policy: dict[str, Any],
    rounds: list[dict[str, Any]],
    ledger: Any,  # noqa: ANN401
    iterations: list[dict[str, Any]],
    run_state: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """``final/convergence.json`` per data-model.md ConvergenceReport."""
    per_axis: dict[str, Any] = {}
    irreducible_regions: list[dict[str, Any]] = []
    for stream in passes:
        for axis in AXES:
            rows = state.axis_rows(stream, axis)
            counts: dict[str, int] = {}
            for row in rows:
                counts[row.get("status", "open")] = counts.get(row.get("status", "open"), 0) + 1
                if row.get("status") == "irreducible":
                    irreducible_regions.append(
                        {
                            "axis": axis,
                            "stream": stream,
                            "start": row["start"],
                            "end": row["end"],
                            "reason": row.get("irreducible_reason"),
                            "residual": row.get("aggregated_uncertainty"),
                            "floor": row.get("aleatoric_floor"),
                        }
                    )
            per_axis[f"{stream}/{axis}"] = {
                "buckets": len(rows),
                **counts,
                "residual_mass": round(
                    state.uncertainty_mass(stream, axis, float(policy["thresholds"]["theta_low"])), 6
                ),
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
    return {
        "run_state": run_state,
        "rounds": rounds,
        "per_axis": per_axis,
        "irreducible_regions": irreducible_regions,
        "budget": ledger.as_dict(),
        "next_actions": next_actions,
        **provenance,
    }
