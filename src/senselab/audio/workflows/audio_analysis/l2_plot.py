"""``L2/round/<n>/timeline.png`` — one figure per round, drawn after that round's fusion.

A single end-state figure cannot show what the iteration did. Per round, a reader can see
whether a round moved anything and where, which is what says whether the loop is earning its
cost — the same reason the maps themselves are written per round.

This replaces the chunked ``timeline_001.png`` / ``timeline_002.png`` output, whose panels were
mostly empty: a fixed time window rarely lines up with where anything actually happened, so
most chunks showed nothing and the interesting moment was split across two files.

Only fused quantities appear here. The evidence rows live in ``L1/signals.png``; keeping the
two apart is what stops a conclusion being read as another observation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = ["build_round_timeline"]

_QUANTITIES = (
    ("uncertainty", "uncertainty\n(entropy)", "#b2182b"),
    ("epistemic_uncertainty", "reducible", "#ef8a62"),
    ("confidence", "confidence", "#2166ac"),
    ("variability", "variability", "#5aae61"),
)


def build_round_timeline(
    out_dir: Path | str,
    *,
    round_index: int,
    axis_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    duration_s: float,
    title: str | None = None,
) -> Path | None:
    """Draw one row per axis per fused quantity for a single round.

    Args:
        out_dir: Run directory.
        round_index: Which round these rows came from.
        axis_rows: ``{axis → fused rows}``.
        duration_s: Recording length, so a sparse axis still spans the figure.
        title: Figure title.

    Returns:
        The written path, or ``None`` when the round produced no rows at all — an empty figure
        would suggest the round ran and found nothing, rather than not having run.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from senselab.audio.workflows.audio_analysis.layout import round_dir

    axes_present = [axis for axis in sorted(axis_rows) if axis_rows[axis]]
    if not axes_present:
        return None

    n_rows = len(axes_present) * len(_QUANTITIES)
    fig, axs = plt.subplots(n_rows, 1, figsize=(14, max(2.0, 0.5 * n_rows + 1.0)), sharex=True, squeeze=False)
    flat = [ax for (ax,) in axs]

    row = 0
    for axis in axes_present:
        rows = sorted(axis_rows[axis], key=lambda r: float(r.get("start", 0.0)))
        centres = np.array([(float(r["start"]) + float(r["end"])) / 2.0 for r in rows])
        for field, label, colour in _QUANTITIES:
            ax = flat[row]
            row += 1
            values = np.array([np.nan if r.get(field) is None else float(r[field]) for r in rows])
            ax.set_ylabel(f"{axis}\n{label}", rotation=0, ha="right", va="center", fontsize=7)
            ax.set_xlim(0, max(duration_s, 1e-6))
            ax.set_ylim(0, 1.02)
            ax.set_yticks([0, 1])
            ax.tick_params(labelsize=6)
            if centres.size:
                # Gaps are left as gaps: matplotlib breaks the line at NaN, so a stretch nobody
                # measured reads as absent rather than as a value interpolated across it.
                ax.plot(centres, values, linewidth=0.8, color=colour)
            ax.grid(True, axis="y", alpha=0.2)

    flat[-1].set_xlabel("time (s)")
    fig.suptitle(title or f"L2 round {round_index} — fused belief", fontsize=10)
    fig.tight_layout()

    dest = round_dir(out_dir, round_index)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "timeline.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
