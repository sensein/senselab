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

**The round comes from the directory, for the same reason the axis comes from the filename.** The
declaration keys this artifact ``(axis, bucket, round)`` with *both* axis and round fixed by the
path, so a row that spells either is repeating what its location already said and the two must not
be able to disagree. They did: fusion stamped the round it folded, and the adaptive loop stamped
the round in which each bucket was last re-folded — so ``L2/round/4/estimates/speech_presence.parquet``
held rows claiming rounds 1, 3 and 4 at once, and ``final/estimates/asr.parquet`` — the verbatim
extraction of round 4 — claimed round 1. Anything deriving a path from the column (the
disagreements index does) then pointed a reader at another round's fold.

The fact the loop was overloading onto ``round`` is real and survives under its own name:
``last_refolded_round`` says which round last recomputed this value, which is *earlier* than the
round wherever an axis converged or a bucket went untouched and its estimate was carried forward.
Provenance about how the value came to be, never an index on the row — the same standing
``contributing_passes`` has, and the reason neither is a second spelling of its dimension.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, Mapping, Sequence

if TYPE_CHECKING:  # pragma: no cover — import cost stays out of the runtime path
    import pandas as pd

__all__ = ["ESTIMATE_COLUMNS", "control_doubt", "estimate_frame"]


def control_doubt(row: Mapping[str, Any]) -> float | None:
    """Doubt in this bucket's answer, on the scale ``theta_low`` / ``theta_high`` are written on.

    **The loop's gates need a probability, and only ``confidence`` is one.** Region seeding
    (``regions.propose_regions``), convergence marking (``convergence.apply_convergence_marks``) and
    the residual metric (``belief.BeliefState.uncertainty_mass``) all compared ``uncertainty``
    against those thresholds — but ``uncertainty`` is normalised *binary entropy* of the mean
    per-signal doubt, and entropy climbs steeply away from zero: ``H(0.10) = 0.469``,
    ``H(0.20) = 0.722``. The thresholds are doubt-scaled (they are the Label Studio high/low bins),
    so the comparison silently meant "flag anything above 17% doubt, converge only below 6%" —
    solve ``H(p) = 0.66`` and ``H(p) = 0.33``. Nobody chose those numbers.

    Measured cost, on a clean two-speaker conversation whose speaker-count posterior is 0.978
    unimodal and whose per-speaker ``existence_uncertainty`` is 0.0: the speaker axis read 0.666 and
    seeded **114 of 214** buckets, letting 23 converge. On doubt it seeds 13 and converges 152. Of
    the 0.666, aleatoric was 0.391 and epistemic 0.275 — so 59% of what drove region proposal was
    doubt no further measurement can remove, which is the waste ``statistics.py`` says the
    decomposition exists to prevent.

    **Why not ``epistemic_uncertainty``**, which is the reducible part and looks like the principled
    choice: it is inter-signal disagreement, so it is structurally ``0.0`` for a single-voter axis.
    ``asr`` has exactly one voter (``consensus_words``), and gating on epistemic would make it
    permanently un-investigatable while its doubt is real — measured mean 0.215, max 0.918. A lone
    confident-but-doubtful voter is a reason to add a second, not a reason to stop looking. Each rule
    keeps its own reducibility test where that question belongs: ``U1``/``U2`` gate on
    ``epistemic_uncertainty`` themselves.

    Args:
        row: An estimates row.

    Returns:
        ``1 - confidence`` clipped to ``[0, 1]``, or ``None`` when the bucket carries no confidence —
        which is "nothing was measured here", and distinct from confident agreement at zero doubt.
        ``NaN`` counts as absent: it is what a parquet null deserialises to, and comparing it against
        a threshold silently answers ``False`` in both directions.
    """
    value = row.get("confidence")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    if value != value:  # NaN — a parquet null round-tripped through pandas
        return None
    return max(0.0, min(1.0, 1.0 - float(value)))


ESTIMATE_COLUMNS: Final[tuple[str, ...]] = (
    # What every row is, whoever wrote it.
    "start",
    "end",
    "axis",
    "round",
    # Which round last recomputed the value above. Equal to ``round`` where this round re-folded
    # the bucket, earlier where the estimate was carried forward — an axis that converged, or a
    # bucket no intervention touched. Without it, carrying a value forward and re-folding it to the
    # same number are the same row.
    "last_refolded_round",
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
    # Which perturbations ran and were *not* admitted here, and so are absent from
    # ``contributing_passes`` by decision rather than by never having existed. A repair transform
    # is evidence only where the recording is degraded (``fuse.SnrGate``), and on a clean recording
    # that means it is held out of nearly every bucket — a reader seeing only ``['raw']`` above
    # cannot otherwise tell that from a run that never enhanced anything.
    "snr_gated_passes",
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


def estimate_frame(axis: str, rows: Sequence[Mapping[str, Any]], *, round_index: int) -> "pd.DataFrame":
    """One round's estimate of one axis, in the declared shape.

    Args:
        axis: The axis these rows estimate. Stamped on every row rather than trusted from the
            caller's dicts, because the file is named for it and the two must not be able to
            disagree.
        rows: Prepared row dicts. Any declared column a row omits is written null — the producer
            had nothing to say about it, which is a fact worth recording and not a reason to
            change shape.
        round_index: The round whose directory this frame is written into. Stamped for exactly the
            reason ``axis`` is: the path fixes it, so a caller cannot be the one who decides what
            it says. Required rather than defaulted — a default is what the loop's hardcoded ``1``
            was, and it outlived the round it was true of.

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
    prepared = [
        {**dict.fromkeys(ESTIMATE_COLUMNS), **dict(row), "axis": axis, "round": int(round_index)} for row in rows
    ]
    return pd.DataFrame(prepared, columns=list(ESTIMATE_COLUMNS))
