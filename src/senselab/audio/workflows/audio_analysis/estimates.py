"""``L2/round/<n>/estimates/<axis>.parquet`` — one artifact name, one shape (D-17).

Two producers write this file. ``fuse.write_final_uncertainty`` writes the rounds it folds, and
the adaptive loop's belief store writes the rounds it iterates; the loop *adopts* fusion's last
round as its baseline, so a run's rounds 0..k come from one and k+1..n from the other, in one
directory, under one declared pattern.

They emitted different columns. Fusion's rows carried ``axis``, ``signal_weights``,
``weight_basis`` and the scene coupling; the loop's carried ``status``, ``p_voice``,
``aleatoric_floor`` and the attenuation block, and no ``axis`` at all. Both are genuinely keyed by
``(round, axis, bucket)``, so every key rule passed on both — what differed was below the key, and
the consequence is that a reader plotting one axis across the trajectory got different columns on
either side of a boundary the path does not mention.

**They are not different artifacts.** A round's estimate of an axis is one quantity, and rounds 2
and 3 are consecutive iterations of one loop; declaring them separately would say the trajectory
has a seam in it, which is precisely the thing that is wrong. So the schema is the union, declared
here, and a producer with nothing to say for a column writes null rather than omitting it —
*absent is not zero*, and "this producer does not compute a convergence status" is exactly what a
null in ``status`` means.

A column no producer may invent is the other half: :func:`estimate_frame` raises on one, so adding
a column costs an edit here, where both producers see it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, Mapping, Sequence

if TYPE_CHECKING:  # pragma: no cover — import cost stays out of the runtime path
    import pandas as pd

__all__ = ["ESTIMATE_COLUMNS", "estimate_frame"]


ESTIMATE_COLUMNS: Final[tuple[str, ...]] = (
    # What every row is, whoever wrote it.
    "start",
    "end",
    "axis",
    "round",
    # The estimate itself, in the four numbers an axis actually carries — which is why the
    # directory is named ``estimates/`` and not after any one of them.
    "uncertainty",
    "epistemic_uncertainty",
    "confidence",
    "variability",
    "triage_score",
    # What fed the fold. Provenance about the inputs, never an index on the output: a row lists
    # its contributing perturbations, it is not keyed by one.
    "contributing_signals",
    "contributing_passes",
    # Fusion's account of *how* it weighted, and what moved the value afterwards.
    "signal_weights",
    "weight_basis",
    "coupled_from",
    "scene_quality_coupling",
    "triage_score_pre_coupling",
    # The loop's account of where the value stands and what stops it going lower.
    "status",
    "irreducible_reason",
    # Calibrated P(speech) and the overlapped-speech posterior. Here because ``final/`` extracts a
    # round rather than recomputing one: these two used to exist only on a separately-built
    # ``L2/speech_presence.parquet``, so the deliverable carried a number no round did and the
    # evaluator had to score that file instead of the belief the run published.
    "speech_presence_confidence",
    "overlap_posterior",
    "p_voice",
    "aleatoric_floor",
    "aleatoric_floor_terms",
    "n_sources",
    "n_attenuated_sources",
    "attenuated_sources",
    "attenuation",
)
"""Every column an estimates row may carry, in the order it is written.

The order is part of the schema rather than incidental: a reader diffing two rounds of the same
axis should not have to sort columns to see that they agree."""


def estimate_frame(axis: str, rows: Sequence[Mapping[str, Any]]) -> "pd.DataFrame":
    """One round's estimate of one axis, in the declared shape.

    Args:
        axis: The axis these rows estimate. Stamped on every row rather than trusted from the
            caller's dicts, because the file is named for it and the two must not be able to
            disagree.
        rows: Prepared row dicts. Any declared column a row omits is written null — the producer
            had nothing to say about it, which is a fact worth recording and not a reason to
            change shape.

    Returns:
        A frame with exactly :data:`ESTIMATE_COLUMNS`, in that order. Empty rows still produce the
        full column set, so "this round believes nothing here" stays distinguishable from "this
        round was never asked".

    Raises:
        ValueError: When a row carries a column the schema does not declare. A producer that
            grows a column has to grow the schema, which is the only thing keeping one artifact
            name to one shape.
    """
    import pandas as pd

    declared = set(ESTIMATE_COLUMNS)
    undeclared = sorted({key for row in rows for key in row} - declared)
    if undeclared:
        raise ValueError(
            f"estimates/{axis}.parquet: {undeclared} is written by no declaration — add it to "
            "ESTIMATE_COLUMNS so the other producer writes it too, or two rounds of one axis will "
            "have different columns and nothing in the path will say which producer wrote which"
        )
    prepared = [{**dict.fromkeys(ESTIMATE_COLUMNS), **dict(row), "axis": axis} for row in rows]
    return pd.DataFrame(prepared, columns=list(ESTIMATE_COLUMNS))
