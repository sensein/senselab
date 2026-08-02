"""Canonical run-directory layout, named after the level that produces each artifact.

The directory names carry the architecture, so a reader can tell what a file *is* from where it
sits rather than having to know which module wrote it:

``L1/`` — **evidence.** Per-perturbation model outputs under ``L1/raw/`` and
``L1/perturbation/<k>/``, an index of what those perturbations are in ``L1/perturbations.json``,
and each signal's own measurement in the tool's own units under ``L1/signals/``. Nothing here is
an answer, and nothing here is named for an **axis**: an axis is a fold across signals *and*
across perturbations, so it can be neither produced by one perturbation nor stored under one.

``L1/signals/<signal>.parquet`` **accumulates across raw and every perturbation** — one file per
signal, each row carrying the perturbation it was measured under — and is the only thing L2 reads
from L1. Per-perturbation signal files were the earlier form; they made the perturbation an index
on the *location* rather than a dimension of the data, which is what let consumers reach into
``L1/<pass>/`` for one perturbation's evidence and quietly get a different answer than a fold
over all of them.

``L2/round<N>/`` — **belief.** The fused uncertainty maps, one directory per iteration. Per
round rather than only final because a single map cannot distinguish "settled immediately" from
"moved a long way and then settled", and that difference is what tells an operator whether the
loop is earning its cost.

``final/`` — **deliverables.** The summary, the timeline, and the consensus artifacts a
consumer actually acts on. Kept separate from ``L2/`` because "what do we believe" and "what do
we hand over" are different questions: the belief is per bucket and per round, the deliverable
is one transcript and one figure.

Cross-perturbation disagreement is **not** here. Comparing two perturbations is a fold over an
input dimension, by exactly the argument that makes an axis L2's, so per-signal stability is a
round derivative. Its run-level summary is not written at all: it is already on every fused row
as ``weight_basis[signal]["stability"]``, and one quantity in two places is one quantity that can
disagree with itself.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "BELIEF_DIR",
    "EVIDENCE_DIR",
    "FINAL_DIR",
    "belief_dir",
    "evidence_dir",
    "final_dir",
    "perturbation_dir",
    "signals_dir",
]

EVIDENCE_DIR = "L1"
BELIEF_DIR = "L2"
FINAL_DIR = "final"


def evidence_dir(run_dir: Path | str) -> Path:
    """``<run>/L1`` — everything measured, nothing concluded."""
    return Path(run_dir) / EVIDENCE_DIR


def perturbation_dir(run_dir: Path | str, name: str) -> Path:
    """``<run>/L1/raw`` for the identity, ``<run>/L1/perturbation/<name>`` for any other.

    Two locations rather than one because the identity is not one transform among many: it is
    the recording, the thing every other perturbation is a transform *of*, and the layout says
    so. The set below ``perturbation/`` is open — a third needs a register entry, not a code
    edit here.
    """
    from senselab.audio.workflows.audio_analysis.perturbations import IDENTITY_NAME

    base = evidence_dir(run_dir)
    return base / IDENTITY_NAME if name == IDENTITY_NAME else base / "perturbation" / name


def signals_dir(run_dir: Path | str) -> Path:
    """``<run>/L1/signals`` — per-signal measurements across raw and every perturbation.

    L2's only input from L1. One file per signal, every row carrying its perturbation, so a
    consumer that wants one perturbation's evidence has to say so on the data rather than by
    picking a directory.
    """
    return evidence_dir(run_dir) / "signals"


def belief_dir(run_dir: Path | str, round_index: int | None = None) -> Path:
    """``<run>/L2`` or ``<run>/L2/round<N>`` — the fused belief for one iteration."""
    base = Path(run_dir) / BELIEF_DIR
    return base if round_index is None else base / f"round{int(round_index)}"


def final_dir(run_dir: Path | str) -> Path:
    """``<run>/final`` — the summary, the timeline, and the consensus deliverables."""
    return Path(run_dir) / FINAL_DIR
