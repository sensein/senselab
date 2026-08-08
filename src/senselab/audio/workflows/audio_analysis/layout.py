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

``L2/round/<n>/`` — **belief.** *One* round tree, holding ``estimates/`` (the axes, one file
each), ``derivatives/`` (the mask, the votes, the stability, the regions a round proposed),
``timeline.png`` and ``summary.json``. Per round rather than only final because a single map
cannot distinguish "settled immediately" from "moved a long way and then settled", and that
difference is what tells an operator whether the loop is earning its cost.

Three of those a round **owes**: its belief, its account and its view — ``estimates/`` for every
active axis, ``summary.json``, ``timeline.png``. Two producers write into this tree (``fuse``
folds the early rounds, the adaptive loop iterates the later ones, adopting fusion's last as its
baseline), and both write all three; a round summary carries one block per producer. The
derivatives are *not* owed by every round: ``votes/`` and ``stability/`` are the ingest round's,
computed once from L1, and ``regions.json``/``votes_added.parquet`` belong to the rounds that ran
interventions. Writing an empty one elsewhere would claim "we looked and found none" of a round
that does no region proposal at all.

There were two trees: ``L2/round<N>/`` from fusion (0-based) and ``L2/rounds/<N>/`` from the
adaptive loop (1-based), so the fusion loop's round 0 and the adaptive loop's round 1 were the
same iteration under two names — and "round 1" meant different things depending on which
directory you were reading. Under one tree they are one node, and the numbering is the run's.

``estimates/`` rather than ``uncertainty/``: a row carries uncertainty, epistemic uncertainty,
confidence and variability, so naming the directory after one of the four names a column rather
than the thing itself.

``final/`` — **deliverables, by extraction.** The summary, the timeline, the consensus artifacts
a consumer acts on, and ``final/estimates/<axis>.parquet`` — the last round's estimates, copied
verbatim, one file per active axis. Kept separate from ``L2/`` because "what do we believe" and
"what do we hand over" are different questions: the belief is per bucket and per round, the
deliverable is one answer.

*Extraction*, and the word is load-bearing. A number in ``final/`` that is not in the last round
was computed at the wrong stage, and there is then nowhere to look for when it was decided. The
presence track was built that way — rebuilt from the belief state into its own file with columns
(``speech_presence_confidence``, ``overlap_posterior``) no round carried — so those columns are on
the estimate row now and this directory only moves bytes.

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
    "derivatives_dir",
    "estimates_dir",
    "evidence_dir",
    "final_dir",
    "last_round",
    "perturbation_dir",
    "round_dir",
    "rounds_present",
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


def belief_dir(run_dir: Path | str) -> Path:
    """``<run>/L2`` — the root of the round tree, and nothing else.

    It takes no round index any more. ``belief_dir(run)`` and ``belief_dir(run, n)`` returning a
    directory and its child made the root a place things could be written, which is how nine
    per-round quantities came to sit flattened at ``L2/`` with no round to belong to and the last
    writer winning.
    """
    return Path(run_dir) / BELIEF_DIR


def round_dir(run_dir: Path | str, round_index: int) -> Path:
    """``<run>/L2/round/<n>`` — one iteration of the belief loop."""
    return belief_dir(run_dir) / "round" / str(int(round_index))


def estimates_dir(run_dir: Path | str, round_index: int) -> Path:
    """``<run>/L2/round/<n>/estimates`` — the axes, one file each, one row per bucket."""
    return round_dir(run_dir, round_index) / "estimates"


def derivatives_dir(run_dir: Path | str, round_index: int) -> Path:
    """``<run>/L2/round/<n>/derivatives`` — what the round built on the way to its estimates.

    The mask, the linked votes, the cross-perturbation stability, the regions it proposed. All
    per round, because each is a thing *that round* decided and a later round may decide
    differently.
    """
    return round_dir(run_dir, round_index) / "derivatives"


def rounds_present(run_dir: Path | str) -> tuple[int, ...]:
    """Which rounds a run has written, in numeric order.

    Numeric, not lexicographic: the previous "last round" was ``sorted(glob("round*"))[-1]``,
    which puts ``round10`` before ``round2`` and so read round 9's map on any run past ten
    rounds.
    """
    base = belief_dir(run_dir) / "round"
    return tuple(sorted(int(p.name) for p in base.glob("*") if p.is_dir() and p.name.isdigit()))


def last_round(run_dir: Path | str) -> int | None:
    """The highest round index this run wrote, or ``None`` when it wrote none."""
    rounds = rounds_present(run_dir)
    return rounds[-1] if rounds else None


def final_dir(run_dir: Path | str) -> Path:
    """``<run>/final`` — the summary, the timeline, and the consensus deliverables."""
    return Path(run_dir) / FINAL_DIR
