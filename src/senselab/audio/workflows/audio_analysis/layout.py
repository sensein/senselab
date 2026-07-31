"""Canonical run-directory layout, named after the level that produces each artifact.

The directory names carry the architecture, so a reader can tell what a file *is* from where it
sits rather than having to know which module wrote it:

``L1/`` — **evidence.** Per-pass model outputs and the per-signal uncertainties harvested from
them, plus the cross-pass stability measurements. Nothing here is an answer; the per-pass
uncertainty parquets record what one pass alone would have concluded, before anything was
measured about the signals' reliability.

``L2/round<N>/`` — **belief.** The fused uncertainty maps, one directory per iteration. Per
round rather than only final because a single map cannot distinguish "settled immediately" from
"moved a long way and then settled", and that difference is what tells an operator whether the
loop is earning its cost.

``final/`` — **deliverables.** The summary, the timeline, and the consensus artifacts a
consumer actually acts on. Kept separate from ``L2/`` because "what do we believe" and "what do
we hand over" are different questions: the belief is per bucket and per round, the deliverable
is one transcript and one figure.

The cross-pass deltas live under ``L1/stability/`` rather than beside the L2 maps, because they
are not a third answer — they *are* the perturbation-stability measurement that feeds L2's
weights. Filing them as evidence is what makes that role legible.
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
    """``<run>/L1/<pass>`` — one pass's model outputs and per-signal uncertainties."""
    return evidence_dir(run_dir) / pass_label


def stability_dir(run_dir: Path | str) -> Path:
    """``<run>/L1/stability`` — cross-pass deltas, i.e. the perturbation-stability evidence."""
    return evidence_dir(run_dir) / "stability"


def belief_dir(run_dir: Path | str, round_index: int | None = None) -> Path:
    """``<run>/L2`` or ``<run>/L2/round<N>`` — the fused belief for one iteration."""
    base = Path(run_dir) / BELIEF_DIR
    return base if round_index is None else base / f"round{int(round_index)}"


def final_dir(run_dir: Path | str) -> Path:
    """``<run>/final`` — the summary, the timeline, and the consensus deliverables."""
    return Path(run_dir) / FINAL_DIR
