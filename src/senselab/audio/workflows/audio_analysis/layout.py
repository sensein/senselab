"""Canonical run-directory layout, named after the level that produces each artifact.

The directory names carry the architecture, so a reader can tell what a file *is* from where it
sits rather than having to know which module wrote it:

``L1/`` — **evidence.** Per-pass model outputs and each signal's own measurement, in the tool's
own units, plus the cross-pass stability measurements. Nothing here is an answer, and nothing here
is named for an **axis**: an axis is a fold across signals *and* across passes, so it can be
neither produced by one pass nor stored under one.

``L2/round<N>/`` — **belief.** The fused uncertainty maps, one directory per iteration. Per
round rather than only final because a single map cannot distinguish "settled immediately" from
"moved a long way and then settled", and that difference is what tells an operator whether the
loop is earning its cost.

``final/`` — **deliverables.** The summary, the timeline, and the consensus artifacts a
consumer actually acts on. Kept separate from ``L2/`` because "what do we believe" and "what do
we hand over" are different questions: the belief is per bucket and per round, the deliverable
is one transcript and one figure.

``L1/stability/<signal>.parquet`` holds the cross-pass disagreement **per signal**, and
``L1/stability/signals.json`` the run-level mean that sets each signal's fusion weight. Keyed by
signal because that is what stability is a property of: the two passes are the same recording
under a transform, so a signal that answers differently between them has not earned its weight.
The previous form — one file per *axis* under a ``raw_vs_enhanced`` pseudo-pass, obtained by
subtracting two per-pass axis folds — was wrong three times over, and had no reader anywhere.
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
    "pass_dir",
    "stability_dir",
]

EVIDENCE_DIR = "L1"
BELIEF_DIR = "L2"
FINAL_DIR = "final"


def evidence_dir(run_dir: Path | str) -> Path:
    """``<run>/L1`` — everything measured, nothing concluded."""
    return Path(run_dir) / EVIDENCE_DIR


def pass_dir(run_dir: Path | str, pass_label: str) -> Path:
    """``<run>/L1/<pass>`` — one pass's model outputs and per-signal measurements."""
    return evidence_dir(run_dir) / pass_label


def stability_dir(run_dir: Path | str) -> Path:
    """``<run>/L1/stability`` — per-**signal** cross-pass disagreement.

    ``<signal>.parquet`` per bucket, ``signals.json`` for the run-level mean that becomes each
    signal's fusion weight. A signal, not an axis: stability is a property of the thing that
    answered twice.
    """
    return evidence_dir(run_dir) / "stability"


def belief_dir(run_dir: Path | str, round_index: int | None = None) -> Path:
    """``<run>/L2`` or ``<run>/L2/round<N>`` — the fused belief for one iteration."""
    base = Path(run_dir) / BELIEF_DIR
    return base if round_index is None else base / f"round{int(round_index)}"


def final_dir(run_dir: Path | str) -> Path:
    """``<run>/final`` — the summary, the timeline, and the consensus deliverables."""
    return Path(run_dir) / FINAL_DIR
