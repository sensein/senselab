"""``final/summary.md`` — what a person needs to know about a run, without opening a parquet.

``summary.json`` is the machine record and is already large. Someone opening a run wants four
things quickly: how many speakers, how uncertain each axis was and how much of that is
reducible, where the worst regions are, and whether the loop converged or ran out of rounds.
Those live across L2 parquets and JSON, so answering "how did this run go" currently requires
knowing the layout.

Two reporting choices carry weight:

**Unmeasured buckets are counted, never averaged in.** Treating "not measured" as zero would
report a run as more certain than it was, which is the failure mode a summary is most likely to
introduce.

**The worst regions are named with their times.** "Uncertainty was 0.4" is not actionable;
"0.9 at 0.5–1.0 s" is. A mean alone hides a single bad region, which is usually the thing worth
looking at.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

__all__ = [
    "build_run_summary",
    "render_run_summary",
]


def _axis_summary(rows: Sequence[Mapping[str, Any]], *, top_n: int) -> dict[str, Any]:
    measured = [r for r in rows if r.get("uncertainty") is not None]
    epistemic = [float(r["epistemic_uncertainty"]) for r in rows if r.get("epistemic_uncertainty") is not None]
    values = [float(r["uncertainty"]) for r in measured]
    worst = sorted(measured, key=lambda r: -float(r["uncertainty"]))[:top_n]
    return {
        "buckets": len(rows),
        "measured_buckets": len(measured),
        "unmeasured_buckets": len(rows) - len(measured),
        "mean_uncertainty": (sum(values) / len(values)) if values else None,
        "max_uncertainty": max(values) if values else None,
        "mean_epistemic_uncertainty": (sum(epistemic) / len(epistemic)) if epistemic else None,
        "worst_regions": [
            {
                "start": float(r["start"]),
                "end": float(r["end"]),
                "uncertainty": float(r["uncertainty"]),
            }
            for r in worst
        ],
    }


def build_run_summary(
    *,
    axis_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    speakers: Mapping[str, Any],
    rounds: Mapping[str, Sequence[Mapping[str, Any]]],
    top_n: int = 5,
) -> dict[str, Any]:
    """Assemble the run headline from the L2 maps, the posterior, and the round log.

    Args:
        axis_rows: ``{axis → fused rows}`` from the final round.
        speakers: The ``speakers.json`` document, or empty when identity did not run.
        rounds: ``{axis → round log}``.
        top_n: How many worst regions to name per axis.

    Returns:
        A JSON-serialisable summary. Absent measurements are reported as ``None`` rather than
        zero, so a run where nothing was measured cannot read as a confident one.
    """
    posterior = dict(speakers.get("count_posterior") or {})
    convergence = {
        axis: ("converged" if (log and log[-1].get("converged")) else "rounds_exhausted")
        for axis, log in rounds.items()
    }
    return {
        "axes": {axis: _axis_summary(rows, top_n=top_n) for axis, rows in sorted(axis_rows.items())},
        "speakers": {
            "modal_count": posterior.get("modal_count"),
            "is_multimodal": posterior.get("is_multimodal"),
            "probabilities": posterior.get("probabilities") or {},
            "hypotheses": len(speakers.get("speakers") or []),
        },
        "convergence": convergence,
    }


def _fmt(value: Any) -> str:  # noqa: ANN401
    return "n/a" if value is None else (f"{value:.3f}" if isinstance(value, float) else str(value))


def render_run_summary(doc: Mapping[str, Any]) -> str:
    """Render the summary as Markdown, so a run is legible without a parquet reader."""
    lines = ["# Run summary", ""]

    speakers = doc.get("speakers") or {}
    lines += ["## Speakers", ""]
    modal = speakers.get("modal_count")
    lines.append(f"- Modal count: **{_fmt(modal)}**" + ("  (multi-modal)" if speakers.get("is_multimodal") else ""))
    probabilities = speakers.get("probabilities") or {}
    if probabilities:
        spread = ", ".join(f"{k}: {float(v):.2f}" for k, v in sorted(probabilities.items()))
        lines.append(f"- Posterior: {spread}")
    lines.append("")

    lines += [
        "## Uncertainty by axis",
        "",
        "| axis | mean | max | reducible | measured | unmeasured |",
        "|---|---|---|---|---|---|",
    ]
    for axis, block in (doc.get("axes") or {}).items():
        lines.append(
            f"| {axis} | {_fmt(block.get('mean_uncertainty'))} | {_fmt(block.get('max_uncertainty'))} "
            f"| {_fmt(block.get('mean_epistemic_uncertainty'))} | {block.get('measured_buckets')} "
            f"| {block.get('unmeasured_buckets')} |"
        )
    lines.append("")

    for axis, block in (doc.get("axes") or {}).items():
        worst = block.get("worst_regions") or []
        if not worst:
            continue
        lines += [f"### Worst {axis} regions", ""]
        lines += [f"- {r['start']:.2f}–{r['end']:.2f} s: {r['uncertainty']:.3f}" for r in worst]
        lines.append("")

    convergence = doc.get("convergence") or {}
    if convergence:
        lines += ["## Convergence", ""]
        lines += [f"- {axis}: {state}" for axis, state in sorted(convergence.items())]
        lines.append("")
    return "\n".join(lines)
