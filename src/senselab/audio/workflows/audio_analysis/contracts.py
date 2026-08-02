"""D-17 — the pipeline is a DAG of workflows, each declaring its inputs and its outputs.

Three rounds of guards were written against the violation last found, and each missed the next
instance of the same class: a name list that omitted the fourth axis, a regex an alias slipped
past, a glob that saw the workflow package but not ``adaptive/``, three artifact rules that all
pass on a genuine per-pass axis table. **Enumerating what is forbidden cannot terminate.
Declaring what is permitted does.**

This module is that declaration, and it is the only place it exists. Nothing may restate it:
the DAG's edges are *derived* by matching one stage's declared reads against another's declared
writes, and both guards read the same tuples the DAG does.

Four things live here, in this order:

1. :data:`STAGE_CONTRACTS` — for each node (``L1``, an ``L2`` round, ``final``, ``eval``) the
   run-relative path patterns it may read and the artifacts it may write, each artifact carrying
   the **key** its rows are indexed by. The key is what makes the content rules derivable rather
   than enumerated: "an ``L1`` artifact is keyed by one perturbation" yields both *no
   perturbation* and *two perturbations* as violations without either being listed.
2. :data:`MODULE_STAGE` — which stage each pipeline module speaks for. Unlisted modules are
   ``PURE``: they may touch no run-relative path at all. The permission defaults to *none*.
3. :func:`static_violations` — walks the AST of every pipeline module (the whole package,
   ``adaptive/`` included, plus both CLI drivers), resolves local aliases to a fixpoint, and
   flags any read or write of a run-relative path outside the declaring stage's contract.
4. :func:`artifact_violations` — walks a real run's artifact tree and flags any file that is in
   no stage's declared outputs, plus any table whose key contradicts the artifact it was written
   as. This catches what static analysis cannot: a writer reached through a helper, or a file
   nobody meant to emit.

:data:`KNOWN_DEVIATIONS` records where the tree does not yet conform. Every entry names the
D-17 clause it breaks and what closes it, and a live-ness check fails when an entry stops
matching — so a fixed violation must be deleted from the register rather than left to rot into a
permanent exemption.

**What the static guard cannot see**, stated so its silence is not read as absence: a path
handed to a helper as a parameter (``_write_round_belief(rounds_dir, ...)``) is opaque to it,
because the pattern is decided at the call site. That is precisely the gap
:func:`artifact_violations` exists to close, and it is why both guards are required rather than
either being a cheaper version of the other.
"""

from __future__ import annotations

import ast
import graphlib
from dataclasses import dataclass, replace
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Final, Iterable, Iterator, Literal, Mapping, Sequence

__all__ = [
    "KNOWN_DEVIATIONS",
    "MODULE_STAGE",
    "STAGE_CONTRACTS",
    "Artifact",
    "Deviation",
    "Finding",
    "StageContract",
    "artifact_violations",
    "dag_edges",
    "matches",
    "overlap",
    "pipeline_sources",
    "static_violations",
    "topological_order",
    "unrolled_contracts",
]


# ── the pattern language ─────────────────────────────────────────────────────
#
# Run-relative, ``/``-separated, with two wildcards: ``*`` matches one segment, ``**`` matches
# any number including none. Deliberately small — a pattern language a reader has to look up is
# a declaration nobody checks.


def _segments(pattern: str) -> tuple[str, ...]:
    """Split a pattern or path into non-empty segments."""
    return tuple(part for part in pattern.split("/") if part)


def matches(path: str, pattern: str) -> bool:
    """Does one concrete run-relative path fall under ``pattern``?"""
    return _match(_segments(path), _segments(pattern))


def _match(path: Sequence[str], pattern: Sequence[str]) -> bool:
    if not pattern:
        return not path
    head, rest = pattern[0], pattern[1:]
    if head == "**":
        return _match(path, rest) or (bool(path) and _match(path[1:], pattern))
    if not path:
        return False
    return fnmatchcase(path[0], head) and _match(path[1:], rest)


def overlap(left: str, right: str) -> bool:
    """Could any concrete path fall under **both** patterns?

    This is what turns declared reads and writes into DAG edges: a consumer reads a producer's
    output exactly when one of its read patterns could name one of the producer's writes. Two
    *patterns* rather than a path against a pattern, because neither side is concrete until a
    run exists, and the DAG has to be checkable without one.
    """
    return _overlap(_segments(left), _segments(right))


def _overlap(left: Sequence[str], right: Sequence[str]) -> bool:
    if not left and not right:
        return True
    if not left:
        return all(segment == "**" for segment in right)
    if not right:
        return all(segment == "**" for segment in left)
    if left[0] == "**":
        return _overlap(left[1:], right) or _overlap(left, right[1:])
    if right[0] == "**":
        return _overlap(left, right[1:]) or _overlap(left[1:], right)
    if fnmatchcase(left[0], right[0]) or fnmatchcase(right[0], left[0]):
        return _overlap(left[1:], right[1:])
    return False


# ── how a row's key is spelled on disk ───────────────────────────────────────
#
# The key names *dimensions*; these say how each dimension appears as a parquet column. Both
# sets are vocabulary, not policy: adding a spelling teaches the guard a synonym, it does not
# add a rule. The rules come from the key.

DIMENSION_COLUMNS: Final[Mapping[str, frozenset[str]]] = {
    "perturbation": frozenset({"perturbation", "stream", "pass", "pass_label", "pass_a", "pass_b", "elected_stream"}),
    "axis": frozenset({"axis"}),
    "signal": frozenset({"signal", "source"}),
    "bucket": frozenset({"start", "end"}),
    "speaker": frozenset({"speaker", "speaker_id"}),
    "round": frozenset({"round"}),
}
"""How each key dimension is spelled as a parquet column. Vocabulary, not policy: adding a
spelling teaches the guard a synonym, it does not add a rule. Every rule comes from the key."""

INTERVAL_DIMENSIONS: Final[frozenset[str]] = frozenset({"bucket"})
"""Dimensions whose spellings co-occur by nature — a bucket is one value written as ``start``
and ``end``. For every other dimension, two spellings on one row means two *values*, which is a
fold over that dimension: ``pass_a`` beside ``pass_b`` is a comparison between perturbations."""

FOLD_COLUMNS: Final[frozenset[str]] = frozenset(
    {"uncertainty", "epistemic_uncertainty", "triage_score", "contributing_signals", "contributing_passes"}
)
"""Columns whose value is an aggregate across signals — an answer, not a measurement. Permitted
only where ``axis`` is part of the key, because an axis *is* that aggregate."""


@dataclass(frozen=True)
class Artifact:
    """One declared output: where it may be written, and what a row of it is indexed by.

    ``key`` is ``None`` for anything that is not a table (JSON, PNG, RTTM); the content rules
    then do not apply and only the path does.

    For a table the key is load-bearing, and three rules fall out of it rather than being
    listed — which is what makes them correct for the fifth axis and the third perturbation
    before either is written:

    - every key dimension the *path* does not supply must appear as a column, so an artifact
      that accumulates across perturbations has to say which one each row came from;
    - a dimension outside the key must not appear at all, so a fold cannot be indexed by a
      perturbation and a measurement cannot carry an axis;
    - a non-interval dimension may be spelled once, so ``pass_a`` beside ``pass_b`` is a
      cross-perturbation fold whatever the file is called.

    ``keyed_in_path`` names the dimensions the location already fixes: ``L1/raw/`` says which
    perturbation, ``estimates/<axis>.parquet`` says which axis.

    ``required`` is the subset that must appear as a column, defaulting to the key minus what
    the path fixes. It exists because a *tree* of tool outputs is not one table: ``L1/raw/**``
    covers a per-bucket feature frame, a per-band noise floor and a pile of JSON, and no single
    key describes all three. What does hold across them is which dimensions they may be indexed
    by — and, far more importantly, which they may not. Declaring ``required=()`` there says "any
    of these, none of them mandatory"; the prohibitions still bind, which is the half that
    matters, because the rule those files must never break is that an axis is L2's.
    """

    pattern: str
    what: str
    key: tuple[str, ...] | None = None
    keyed_in_path: tuple[str, ...] = ()
    required: tuple[str, ...] | None = None

    def must_carry(self) -> frozenset[str]:
        """Dimensions a row has to spell out. Derived unless the artifact overrides it."""
        if self.required is not None:
            return frozenset(self.required)
        return frozenset(self.key or ()) - frozenset(self.keyed_in_path)


@dataclass(frozen=True)
class StageContract:
    """One node of the DAG: what it may read, and what it may write."""

    stage: str
    why: str
    reads: tuple[str, ...] = ()
    writes: tuple[Artifact, ...] = ()

    @property
    def write_patterns(self) -> tuple[str, ...]:
        """The patterns of everything this stage may write."""
        return tuple(artifact.pattern for artifact in self.writes)

    def instantiate(self, round_index: int | None = None, last_round: int | None = None) -> StageContract:
        """Substitute the round placeholders.

        ``round_index=None`` yields the *generic* contract every static check uses, in which
        ``{n}``/``{prev}``/``{last}`` all become ``*`` — a module cannot know which round it is
        running in, so the static guard can only check the directory it writes into. Ordering
        between rounds is enforced by the unrolled DAG instead, where the placeholders become
        numbers and round ``n`` may read only round ``n-1``.
        """
        if round_index is None:
            subs = {"{n}": "*", "{prev}": "*", "{last}": "*" if last_round is None else str(last_round)}
            reads = tuple(_substitute(p, subs) for p in self.reads)
            writes = tuple(replace(a, pattern=_substitute(a.pattern, subs)) for a in self.writes)
            return replace(self, reads=reads, writes=writes)
        subs = {
            "{n}": str(round_index),
            "{prev}": str(round_index - 1),
            "{last}": "*" if last_round is None else str(last_round),
        }
        # Round 0 has no predecessor, so a pattern naming one is *dropped* rather than pointed at
        # ``round-1``: "reads the previous round" and "reads nothing" are different contracts and
        # a negative index would quietly make them the same one.
        reads = tuple(_substitute(p, subs) for p in self.reads if not (round_index == 0 and "{prev}" in p))
        writes = tuple(replace(a, pattern=_substitute(a.pattern, subs)) for a in self.writes)
        return replace(self, stage=f"{self.stage}[{round_index}]", reads=reads, writes=writes)


def _substitute(pattern: str, subs: Mapping[str, str]) -> str:
    for token, value in subs.items():
        pattern = pattern.replace(token, value)
    return pattern


# ── the declaration ──────────────────────────────────────────────────────────

L1 = StageContract(
    stage="L1",
    why=(
        "Measure. In: the recording, the task type, the perturbation definitions, model ids, "
        "device, cache location. Absent: any prior result — L1 never reads L2, never reads "
        "final/, and never reads its own earlier output, which is what makes an L1 value "
        "reproducible from its provenance alone. A perturbation is a transform of the "
        "recording; raw is the identity and the set is open, so L1 is re-enterable. "
        "Limit worth stating: the evidence views are declared here because a figure of the "
        "signals belongs beside them, but neither guard can see what a figure was *drawn from* "
        "— an L1 picture rendered from an L2 belief conforms on paths and is still wrong."
    ),
    reads=(),
    writes=(
        Artifact("L1/perturbations.json", "the open perturbation register: name, transform, parameters"),
        # A tree of tool outputs in the tools' own shapes, not one table: whichever input
        # dimensions a given model reports by, it reports by. What binds is the prohibition —
        # no axis, no fold, one perturbation — which is what makes these measurements rather
        # than answers.
        Artifact(
            "L1/raw/**",
            "the identity perturbation's model outputs",
            key=("perturbation", "signal", "bucket", "speaker"),
            keyed_in_path=("perturbation",),
            required=(),
        ),
        Artifact(
            "L1/perturbation/*/**",
            "each further transform's model outputs",
            key=("perturbation", "signal", "bucket", "speaker"),
            keyed_in_path=("perturbation",),
            required=(),
        ),
        Artifact(
            "L1/signals/**",
            "per-signal measurements accumulating across raw and every perturbation — L2's only input",
            key=("perturbation", "signal", "bucket"),
        ),
        Artifact("L1/signals.png", "evidence view: what each signal measured, in its own units"),
        Artifact("L1/timeline*.png", "evidence view: the signals against the recording, optionally chunked"),
    ),
)

L2_ROUND = StageContract(
    stage="L2_ROUND",
    why=(
        "Fuse and iterate. derive runs before estimate in every round including round 0, and "
        "both read only round n-1 and L1's signals/. An axis is an aggregator across signals "
        "AND across perturbations, so no estimate may be keyed by a perturbation; a round may "
        "not read a sibling updated within the same round, or the fixed point depends on visit "
        "order."
    ),
    reads=("L1/signals/**", "L2/round/{prev}/**"),
    writes=(
        Artifact("L2/round/{n}/derivatives/**", "mask, speaker allocation, ASR consensus, scene components"),
        Artifact(
            "L2/round/{n}/estimates/*.parquet",
            "the axes: one row per (round, axis, bucket)",
            # The round belongs in the key: an estimate is what *this* round believes, and a
            # later round may believe otherwise. The path fixes it, as it fixes the axis, so
            # neither has to appear as a column — but carrying either as provenance is not a
            # violation, and treating the round column as one was the guard reading redundancy
            # as contradiction.
            key=("axis", "bucket", "round"),
            keyed_in_path=("axis", "round"),
        ),
        Artifact("L2/round/{n}/timeline.png", "the same figure the final timeline draws"),
        Artifact("L2/round/{n}/summary.json", "what this round did, and what it now estimates"),
    ),
)

FINAL = StageContract(
    stage="FINAL",
    why=(
        "Extract. final/ is the last round's estimates plus the summaries a human reads. It "
        "computes nothing and is read by nothing: a deliverable something reads is an "
        "intermediate wearing the wrong name. If a number in final/ is not present in the last "
        "round, it was computed at the wrong stage."
    ),
    reads=("L2/round/{last}/**", "L1/signals/**"),
    writes=(
        Artifact("final/transcript.json", "fused words with confidence and alternates"),
        Artifact("final/diarization.json", "fused speaker turns"),
        Artifact("final/diarization.rttm", "the same turns in RTTM"),
        Artifact("final/speakers.json", "count posterior plus per-speaker hypotheses"),
        Artifact("final/per_speaker_presence.parquet", "one track per hypothesised speaker", key=("bucket",)),
        Artifact("final/speech_presence.parquet", "the converged presence track", key=("bucket",)),
        Artifact("final/asr.parquet", "the converged asr axis", key=("bucket",)),
        Artifact("final/background_mask.parquet", "the converged mask axis", key=("bucket",)),
        Artifact("final/speaker/*.parquet", "per_speaker, count, assignment", key=("bucket",)),
        Artifact("final/decisions.json", "trajectory, reversals, stopping reason"),
        Artifact("final/disagreements_resolved.json", "which flagged regions the rounds resolved"),
        Artifact("final/timeline.png", "the human-facing view"),
        Artifact("final/summary.md", "the human-facing summary"),
        Artifact("final/summary.json", "run provenance: policy hash, model revisions, versions, budget"),
        Artifact("final/run_summary.json", "the headline numbers of the last round"),
        Artifact("final/labelstudio_tasks.json", "the review bundle"),
        Artifact("final/labelstudio_config.xml", "the review bundle's config"),
    ),
)

EVAL = StageContract(
    stage="EVAL",
    why=(
        "Score the deliverable against ground truth. A consumer of the answer, not a stage that "
        "builds it — which is why it is the one place final/ may be read, and why it writes one "
        "file nothing else claims."
    ),
    reads=("final/transcript.json", "final/diarization.json", "final/speech_presence.parquet"),
    writes=(Artifact("final/eval.json", "the score"),),
)

LAYOUT = StageContract(
    stage="LAYOUT",
    why=(
        "Names the tree, and answers the one question about it that no path alone can: which "
        "rounds exist. Declared as a node rather than left PURE because every stage needs that "
        "answer, and each deriving it independently is how sorted(glob('round*'))[-1] came to "
        "read round 9 on a run with eleven rounds. It reads the round tree's directory names and "
        "nothing inside them, and writes nothing at all."
    ),
    reads=("L2/round/*",),
)

PURE = StageContract(
    stage="PURE",
    why=(
        "A library: it computes, and the caller decides where the result goes. Declares nothing, "
        "so any run-relative path it opens is a violation. This is the default for every module "
        "not named in MODULE_STAGE — permission defaults to none, which is the whole point of "
        "declaring the permitted rather than the forbidden."
    ),
)

ORCHESTRATOR = StageContract(
    stage="ORCHESTRATOR",
    why=(
        "Runs the stages; it is not one. A driver that opens a run artifact has inlined a stage "
        "body, and the DAG cannot see inside it. Declares nothing for the same reason PURE does."
    ),
)

STAGE_CONTRACTS: Final[Mapping[str, StageContract]] = {
    contract.stage: contract for contract in (L1, L2_ROUND, FINAL, EVAL, LAYOUT, PURE, ORCHESTRATOR)
}

DAG_STAGES: Final[tuple[str, ...]] = ("L1", "L2_ROUND", "FINAL", "EVAL")
"""The stages that are nodes of the pipeline DAG. ``PURE`` and ``ORCHESTRATOR`` are not stages."""


# ── which module speaks for which stage ──────────────────────────────────────

MODULE_STAGE: Final[Mapping[str, str]] = {
    # L1 — measurement.
    "src/senselab/audio/workflows/audio_analysis/perturbations.py": "L1",
    "src/senselab/audio/workflows/audio_analysis/stages.py": "L1",
    "src/senselab/audio/workflows/audio_analysis/stage_context.py": "L1",
    "src/senselab/audio/workflows/audio_analysis/l1_plot.py": "L1",
    "src/senselab/audio/workflows/audio_analysis/plot.py": "L1",
    # L2 — one belief state, fused and iterated.
    "src/senselab/audio/workflows/audio_analysis/fuse.py": "L2_ROUND",
    "src/senselab/audio/workflows/audio_analysis/l2_plot.py": "L2_ROUND",
    "src/senselab/audio/workflows/audio_analysis/adaptive/loop.py": "L2_ROUND",
    "src/senselab/audio/workflows/audio_analysis/adaptive/belief.py": "L2_ROUND",
    "src/senselab/audio/workflows/audio_analysis/adaptive/interventions.py": "L2_ROUND",
    "src/senselab/audio/workflows/audio_analysis/adaptive/regions.py": "L2_ROUND",
    "src/senselab/audio/workflows/audio_analysis/adaptive/policy.py": "L2_ROUND",
    "src/senselab/audio/workflows/audio_analysis/adaptive/audio_io.py": "L2_ROUND",
    # final — extraction.
    "src/senselab/audio/workflows/audio_analysis/adaptive/fusion.py": "FINAL",
    "src/senselab/audio/workflows/audio_analysis/adaptive/ls_final.py": "FINAL",
    "src/senselab/audio/workflows/audio_analysis/adaptive/plot.py": "FINAL",
    "src/senselab/audio/workflows/audio_analysis/summary.py": "FINAL",
    # the consumer.
    "src/senselab/audio/workflows/audio_analysis/adaptive/evaluate.py": "EVAL",
    # the tree's own vocabulary.
    "src/senselab/audio/workflows/audio_analysis/layout.py": "LAYOUT",
    # the drivers.
    "scripts/analyze_audio.py": "ORCHESTRATOR",
    "scripts/adaptive_loop.py": "ORCHESTRATOR",
}
"""Module (repo-relative) → the stage it speaks for. Anything unlisted is :data:`PURE`."""


# ── where the tree does not yet conform ──────────────────────────────────────


@dataclass(frozen=True)
class Deviation:
    """One known non-conformance, with the clause it breaks and what closes it.

    ``module`` is repo-relative for a static deviation and ``""`` for an artifact one.
    ``pattern`` must equal the finding's resolved pattern exactly — a glob here would waive
    violations nobody has looked at, which is how an exemption becomes permanent.
    """

    module: str
    op: Literal["read", "write", "artifact", "key"]
    pattern: str
    why: str


def _mod(name: str) -> str:
    """A workflow module's repo-relative path. Keeps the register readable at a glance."""
    return f"src/senselab/audio/workflows/audio_analysis/{name}"


_DRIVER = "scripts/analyze_audio.py"
_ADAPTIVE_DRIVER = "scripts/adaptive_loop.py"

_INLINED = (
    "scripts/analyze_audio.py runs the stages inline instead of invoking them, so the driver "
    "itself opens run artifacts and the DAG cannot see inside it. Closes when each stage owns "
    "its own writes."
)
_AT_L2_ROOT = (
    "A per-round quantity flattened to the L2 root, so it has no round to belong to and the "
    "last writer wins. D-17: it is a round artifact under L2/round/<n>/."
)
_PAST_SIGNALS = (
    "D-17: L1/signals/ is the only thing L2 reads from L1. This reaches past it into a per-perturbation directory."
)
_EARLIER_ROUND = (
    "final/ extracts the *last* round. Reading an earlier one makes the deliverable a "
    "re-computation over the trajectory rather than an extraction of the answer."
)
_TWO_PASS_TREE = (
    "The per-perturbation tree is still L1/<pass>/ with the two-pass vocabulary baked in. "
    "D-17: L1/raw/ and L1/perturbation/<k>/, with the set open."
)

KNOWN_DEVIATIONS: Final[tuple[Deviation, ...]] = (
    # ── the driver performs all three stages itself ─────────────────────────
    Deviation(_DRIVER, "write", "L1/signals/*.parquet", _INLINED),
    Deviation(_DRIVER, "write", "L1/raw/pii.json", _INLINED),
    Deviation(_DRIVER, "write", "L1/perturbation/*/pii.json", _INLINED),
    Deviation(_DRIVER, "write", "L1/raw/embeddings/*.json", _INLINED),
    Deviation(_DRIVER, "write", "L1/perturbation/*/embeddings/*.json", _INLINED),
    Deviation(_DRIVER, "write", "L2/round/*/derivatives/votes/*.parquet", _INLINED),
    Deviation(_DRIVER, "write", "L2/round/*/derivatives/votes/background_mask.parquet", _INLINED),
    Deviation(_DRIVER, "write", "L2/round/*/derivatives/stability/*.parquet", _INLINED),
    Deviation(_DRIVER, "write", "L2/disagreements.json", _INLINED),
    Deviation(_DRIVER, "write", "L2/labelstudio_tasks.json", _INLINED),
    Deviation(_DRIVER, "write", "L2/labelstudio_config.xml", _INLINED),
    Deviation(_DRIVER, "write", "final/summary.json", _INLINED),
    Deviation(_DRIVER, "write", "final/run_summary.json", _INLINED),
    Deviation(_DRIVER, "write", "final/summary.md", _INLINED),
    Deviation(_DRIVER, "read", "L2/rounds.json", _INLINED),
    Deviation(_DRIVER, "read", "L2/speakers.json", _INLINED),
    Deviation(_DRIVER, "read", "L2/per_speaker_presence.parquet", _INLINED),
    Deviation(_ADAPTIVE_DRIVER, "read", "L1/perturbations.json", _INLINED),
    Deviation(
        _DRIVER,
        "write",
        "triage.json",
        "Written at the run root, outside L1/, L2/ and final/. Its content — speech_present, "
        "needs_enhancement — selects which perturbations the run creates, so it is an L2 "
        "decision taken before L1 has run, stored where no stage claims it.",
    ),
    # ── an L1 stage writes into L2, and round 0 reads it back ───────────────
    Deviation(
        _mod("stages.py"),
        "write",
        "L2",
        "stage_background_mask runs inside run_pass (an L1 stage) and writes "
        "L2/background_mask.{parquet,json}; the driver then reads it back as round 0's only "
        "evidence for the mask axis. An L1 node producing an L2 artifact is the cycle edge D-17 "
        "forbids. The mask is a round *derivative*.",
    ),
    Deviation(_DRIVER, "read", "L2/background_mask.parquet", _AT_L2_ROOT),
    Deviation(
        _mod("plot.py"),
        "read",
        "L2/background_mask.parquet",
        "An L1 module reading an L2 artifact to draw an L1 figure — the evidence view is rendered from belief.",
    ),
    # ── round artifacts still flattened to the L2 root ──────────────────────
    Deviation(_mod("fuse.py"), "write", "L2/rounds.json", _AT_L2_ROOT),
    Deviation(_mod("adaptive/loop.py"), "write", "L2/convergence.json", _AT_L2_ROOT),
    Deviation(_mod("adaptive/loop.py"), "write", "L2/iterations.json", _AT_L2_ROOT),
    Deviation(_mod("adaptive/loop.py"), "read", "L2/disagreements.json", _AT_L2_ROOT),
    # ── L2 reads L1 outside signals/ ────────────────────────────────────────
    Deviation(
        _mod("adaptive/loop.py"),
        "read",
        "L1/perturbations.json",
        _PAST_SIGNALS + " What it actually wants are the run's *inputs* — the source recording, "
        "the duration, the perturbation set — which are not L1 evidence at all; they are what L1 "
        "was given. Closes when the loop is handed them rather than reading L1's index back. "
        "Its siblings (read_register/read_measurements, called two lines apart) reach the same "
        "file through a helper and are invisible to the static guard for the reason its docstring "
        "gives; this one is inline, so it is the one that shows.",
    ),
    # Two entries per read, one per branch of ``perturbation_dir``: the identity's directory and
    # any other perturbation's are different places, and reaching into either is the same
    # violation committed twice rather than one violation described twice.
    Deviation(_mod("adaptive/interventions.py"), "read", "L1/raw/*", _PAST_SIGNALS),
    Deviation(_mod("adaptive/interventions.py"), "read", "L1/perturbation/*/*", _PAST_SIGNALS),
    Deviation(_mod("adaptive/interventions.py"), "read", "L1/raw/*/*.json", _PAST_SIGNALS),
    Deviation(_mod("adaptive/interventions.py"), "read", "L1/perturbation/*/*/*.json", _PAST_SIGNALS),
    Deviation(_mod("adaptive/interventions.py"), "read", "L1/raw/alignment", _PAST_SIGNALS),
    Deviation(_mod("adaptive/interventions.py"), "read", "L1/perturbation/*/alignment", _PAST_SIGNALS),
    Deviation(_mod("adaptive/interventions.py"), "read", "L1/raw/alignment/*.json", _PAST_SIGNALS),
    Deviation(_mod("adaptive/interventions.py"), "read", "L1/perturbation/*/alignment/*.json", _PAST_SIGNALS),
    Deviation(_mod("adaptive/interventions.py"), "read", "L1/raw/embeddings", _PAST_SIGNALS),
    Deviation(_mod("adaptive/interventions.py"), "read", "L1/perturbation/*/embeddings", _PAST_SIGNALS),
    Deviation(_mod("adaptive/interventions.py"), "read", "L1/raw/embeddings/*.json", _PAST_SIGNALS),
    Deviation(_mod("adaptive/interventions.py"), "read", "L1/perturbation/*/embeddings/*.json", _PAST_SIGNALS),
    # ── final/ computes rather than extracts, and writes into L2 ────────────
    Deviation(
        _mod("adaptive/fusion.py"),
        "write",
        "L2/speech_presence.parquet",
        "The fusion stage writes a belief artifact the evaluator then reads. A number in final/ "
        "that is not in the last round was computed at the wrong stage; this one is in no round "
        "at all.",
    ),
    Deviation(_mod("adaptive/fusion.py"), "write", "L2/speakers.json", _AT_L2_ROOT),
    Deviation(_mod("adaptive/fusion.py"), "write", "L2/per_speaker_presence.parquet", _AT_L2_ROOT),
    Deviation(
        _mod("adaptive/fusion.py"),
        "write",
        "L2",
        "A dead mkdir: the fusion stage creates the belief directory and writes nothing there. "
        "Creating another node's tree is how five such calls came to sit in modules that "
        "produce none of its artifacts.",
    ),
    Deviation(_mod("adaptive/plot.py"), "write", "L2", "The same dead mkdir, in the timeline renderer."),
    Deviation(
        _mod("adaptive/loop.py"),
        "write",
        "final",
        "The reverse: an L2 stage creating final/. It writes nothing there — the convergence "
        "and iteration documents go to L2 — so the directory is created by a stage that does "
        "not own it.",
    ),
    Deviation(_mod("adaptive/plot.py"), "read", "L2/iterations.json", _AT_L2_ROOT),
    Deviation(_mod("adaptive/plot.py"), "read", "L2/convergence.json", _AT_L2_ROOT),
    Deviation(_mod("adaptive/plot.py"), "read", "L2/background_mask.parquet", _AT_L2_ROOT),
    Deviation(_mod("adaptive/ls_final.py"), "read", "L2/labelstudio_tasks.json", _AT_L2_ROOT),
    Deviation(_mod("adaptive/ls_final.py"), "read", "L2/labelstudio_config.xml", _AT_L2_ROOT),
    Deviation(_mod("adaptive/ls_final.py"), "read", "L2/disagreements.json", _AT_L2_ROOT),
    # ── the evaluator scores things that are not in final/ ──────────────────
    Deviation(
        _mod("adaptive/evaluate.py"),
        "read",
        "L2/speech_presence.parquet",
        "The evaluator reaches into L2 for a track that should be a deliverable. Closes with "
        "the fusion-stage write above.",
    ),
    Deviation(
        _mod("adaptive/evaluate.py"),
        "read",
        "L2/round/*/summary.json",
        "The evaluator reads the baseline round's uncertainty mass out of L2. EVAL consumes "
        "final/ and nothing else — a scorer reaching into the belief tree is scoring an "
        "intermediate. Closes when the trajectory it wants is in final/decisions.json.",
    ),
    Deviation(
        _mod("adaptive/evaluate.py"),
        "read",
        "L2/round/*/estimates/speaker.parquet",
        "Same: the speaker axis it localises against is the last round's estimate rather than "
        "the deliverable. Closes when final/ carries the converged speaker axis.",
    ),
    # ── a writer with no stage at all ───────────────────────────────────────
    Deviation(
        _mod("level.py"),
        "write",
        "level.json",
        "write_level_json writes <run>/level.json, which no stage declares and no run produces: "
        "the module has no caller in src/ or scripts/. Dead, and the guard says so — an "
        "undeclared output is the same finding whether or not anything reaches it.",
    ),
    # ══ the artifact tree ═══════════════════════════════════════════════════
    # What a completed run leaves on disk that no stage declares. Matched as patterns, so one
    # entry covers a directory the restructure moves as a unit.
    Deviation("", "artifact", "L2/rounds.json", _AT_L2_ROOT),
    Deviation("", "artifact", "L2/convergence.json", _AT_L2_ROOT),
    Deviation("", "artifact", "L2/iterations.json", _AT_L2_ROOT),
    Deviation("", "artifact", "L2/disagreements.json", _AT_L2_ROOT),
    Deviation("", "artifact", "L2/background_mask.*", _AT_L2_ROOT),
    Deviation("", "artifact", "L2/labelstudio_*", _AT_L2_ROOT),
    Deviation("", "artifact", "L2/speakers.json", _AT_L2_ROOT),
    Deviation("", "artifact", "L2/per_speaker_presence.parquet", _AT_L2_ROOT),
    Deviation("", "artifact", "L2/speech_presence.parquet", _AT_L2_ROOT),
    Deviation("", "artifact", "triage.json", "An L2-shaped decision at the run root, taken before L1 has run."),
    Deviation(
        "",
        "artifact",
        "L1/perturbation/*/**",
        "Not a violation of the layout — this is where a non-identity perturbation belongs — but "
        "of the *key*: L1 still writes each model's raw outcome JSON there, and those files are "
        "the tool's own product rather than a measurement L2 can read. They stay until every "
        "consumer reads L1/signals/ instead, which is what the interventions.py entries above "
        "track.",
    ),
)
"""Where the tree does not yet conform to D-17, one entry per distinct violation.

This is the restructure's worklist, not an exemption list: :func:`dead_static_deviations` fails
when an entry stops matching, so closing a violation *requires* deleting its entry. The reasons
are the point — an entry without one is a silenced test.

Not represented here, because neither guard can see them: an L1 figure rendered from an L2
belief (a path-conformant write of wrongly-sourced content), a round reading a sibling mutated
earlier in the same round (in-memory state, not an artifact), and **two writers producing one
declared artifact with different columns** — ``L2/round/0/estimates/<axis>.parquet`` comes from
``fuse`` and carries ``signal_weights``/``weight_basis``; later rounds come from the belief store
and carry ``p_voice``/``aleatoric_floor`` instead, so a reader comparing round 0 with round 2
gets different columns for the same quantity. The key rules pass on both, because both are
genuinely keyed by ``(axis, bucket)``; what differs is below the key. All three are named in
``specs/20260728-221507-per-speaker-identity-scene/layered-architecture.md``.

The ``artifact`` entries are matched as **patterns**, unlike the static ones, because a
restructure moves a directory as a unit and one entry per file would be a file listing rather
than a register. They carry no liveness check: run trees legitimately differ (``eval.json``
appears only with ground truth), so an unmatched artifact entry is evidence of nothing.
"""


# ── the DAG ──────────────────────────────────────────────────────────────────


def unrolled_contracts(n_rounds: int) -> tuple[StageContract, ...]:
    """The DAG's nodes for a run of ``n_rounds`` L2 rounds, with the placeholders resolved.

    Unrolling is what makes the round ordering checkable at all: as a single node ``L2_ROUND``
    reads and writes the same directory and is trivially its own predecessor. As ``n`` nodes,
    round ``n`` reads round ``n-1`` and nothing else, and a round that reached sideways into its
    own outputs shows up as the cycle it is.
    """
    if n_rounds < 1:
        raise ValueError("a run has at least one L2 round")
    rounds = [L2_ROUND.instantiate(index) for index in range(n_rounds)]
    return (L1, *rounds, FINAL.instantiate(last_round=n_rounds - 1), EVAL)


def dag_edges(contracts: Sequence[StageContract]) -> tuple[tuple[str, str], ...]:
    """Producer → consumer edges, **derived** from the declarations rather than restated.

    An edge exists when a consumer's read pattern could name a producer's write. Nothing else
    defines the graph, so a contract change moves the graph with it and the two cannot drift.
    """
    edges: list[tuple[str, str]] = []
    for producer in contracts:
        for consumer in contracts:
            if any(overlap(read, write) for read in consumer.reads for write in producer.write_patterns):
                edges.append((producer.stage, consumer.stage))
    return tuple(edges)


def topological_order(contracts: Sequence[StageContract]) -> tuple[str, ...]:
    """A run order for the stages. Raises :class:`graphlib.CycleError` when the graph has one."""
    graph: dict[str, set[str]] = {contract.stage: set() for contract in contracts}
    for producer, consumer in dag_edges(contracts):
        graph[consumer].add(producer)
    return tuple(graphlib.TopologicalSorter(graph).static_order())


# ── the static guard ─────────────────────────────────────────────────────────

READ_METHODS: Final[frozenset[str]] = frozenset(
    {
        "read_text",
        "read_bytes",
        "open",
        "glob",
        "rglob",
        "iterdir",
        "exists",
        "stat",
        "is_file",
        "is_dir",
        "samefile",
    }
)
"""Path methods that read. ``exists`` and ``stat`` are in deliberately: an existence probe that
decides what a stage does next is a read, and leaving it out is how ``analyze_audio.py`` came to
branch on ``final/labelstudio_tasks.json`` under a guard that reported the rule held."""

WRITE_METHODS: Final[frozenset[str]] = frozenset({"write_text", "write_bytes", "mkdir", "touch", "unlink", "rename"})
"""Path methods that write. ``mkdir`` counts: creating a stage's directory from another stage is
how five dead ``final.mkdir(...)`` calls came to sit in modules that write nothing there."""

READ_FUNCTIONS: Final[frozenset[str]] = frozenset(
    {"read_parquet", "read_json", "read_csv", "read_table", "read_schema", "imread", "load"}
)
"""Calls that read a path given as an argument."""

WRITE_FUNCTIONS: Final[frozenset[str]] = frozenset(
    {"to_parquet", "to_json", "to_csv", "write_table", "savefig", "imsave", "dump"}
)
"""Calls that write to a path given as an argument."""

WRITE_HELPERS: Final[frozenset[str]] = frozenset(
    {
        "write_json",
        "write_signal_parquet",
        "write_linked_votes",
        "write_signal_stability",
        "write_background_mask",
        "write_noise_floor",
        "write_background_sources",
        "write_suppression_json",
        "write_sidecar",
    }
)
"""In-repo writers that take their destination as an argument. Named here because the module
that *decides the path* is the one under contract; ``io.py`` merely writes where it is told."""

_LAYOUT_ROOTS: Final[Mapping[str, tuple[str, ...]]] = {
    "evidence_dir": ("L1",),
    "signals_dir": ("L1", "signals"),
    "final_dir": ("final",),
}
"""Layout helpers whose result is a fixed run-relative directory."""

_PERTURBATION_DIRS: Final[tuple[tuple[str, ...], ...]] = (("L1", "raw"), ("L1", "perturbation", "*"))
"""What ``perturbation_dir(run, name)`` can denote: the identity's directory, or any other's.

One call site, two possible paths, because the identity is not one transform among many. That
makes conformance a question about a **disjunction**, and the answer is the same subsumption rule
applied to each branch: an access conforms when *every* path it could name is permitted, so both
branches are recorded and both are checked. An L1 stage writing there passes on both; an L2 stage
reading there fails on both, and says so twice — once per directory it reached into.
"""

_RUN_ROOT_NAMES: Final[frozenset[str]] = frozenset({"run_dir", "out_dir", "_run_dir", "_out_dir"})
"""Names that hold the run directory itself. A path built off one is run-relative only once a
*literal* segment names it, so ``run_dir / label`` stays unknown while ``run_dir / "L2"`` does
not — the guard reports what it can resolve and stays quiet about what it cannot."""

_RUN_ROOT_ATTRS: Final[frozenset[str]] = frozenset({"run_dir"})
"""Attributes holding the run directory (``ctx.run_dir``)."""

_PASS_DIR_ATTRS: Final[frozenset[str]] = frozenset({"out_dir"})
"""Attributes holding one perturbation's L1 directory (``StageContext.out_dir``)."""

_RUN_ROOT: Final[tuple[tuple[str, ...], ...]] = ((),)
"""The run directory itself. A one-element disjunction: exactly one path, known.

Distinct from ``None`` (unknown) and from ``()`` (an empty disjunction, which would mean *no*
path can be named and would vacuously conform)."""


_Alternatives = tuple[tuple[str, ...], ...]
"""Every run-relative path one expression could denote.

Usually one. ``perturbation_dir(run, name)`` is two, because the identity's directory and any
other perturbation's are different places. An access conforms when every branch is permitted:
"could be inside the declaration" is not proof that it is, which is the same reason conformance
is subsumption rather than intersection.
"""


@dataclass(frozen=True)
class Finding:
    """One resolved read or write of a run-relative path, and where it was written."""

    module: str
    lineno: int
    op: Literal["read", "write"]
    pattern: str
    stage: str
    source_line: str
    via: str = ""

    def __str__(self) -> str:
        """One line, in the shape an editor can jump to."""
        return f"{self.module}:{self.lineno}: [{self.stage}] {self.op} {self.pattern!r}  --  {self.source_line}"


class _PathResolver(ast.NodeVisitor):
    """Resolves run-relative path expressions in one scope, aliases included.

    Aliases are resolved to a **fixpoint** because they chain: ``final = final_dir(d)`` then
    ``tasks = final / "t.json"`` then ``tasks.read_text()``. A single pass sees the first
    binding and not the second, which is how the previous regex-shaped guard walked past every
    real caller — none of them writes the read on the line that names the directory.
    """

    def __init__(self, module: str, stage: str, lines: Sequence[str]) -> None:
        self.module = module
        self.stage = stage
        self.lines = lines
        self.findings: list[Finding] = []

    def run(self, tree: ast.AST) -> list[Finding]:
        """Walk every scope, inheriting each scope's resolved aliases into the ones inside it."""
        self._scope(tree, {})
        return self.findings

    def _scope(self, scope: ast.AST, inherited: Mapping[str, _Alternatives]) -> None:
        env = dict(inherited)
        assignments = [n for n in _own_nodes(scope) if isinstance(n, (ast.Assign, ast.AnnAssign, ast.NamedExpr))]
        for _ in range(len(assignments) + 1):
            before = dict(env)
            for node in assignments:
                value = node.value
                if value is None:
                    continue
                resolved = self._eval(value, env)
                if resolved is None:
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        env[target.id] = resolved
            if env == before:
                break

        for call in (n for n in _own_nodes(scope) if isinstance(n, ast.Call)):
            self._check_call(call, env)

        for child in _nested_scopes(scope):
            self._scope(child, env)

    def _check_call(self, call: ast.Call, env: Mapping[str, _Alternatives]) -> None:
        func = call.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name is None:
            return

        if isinstance(func, ast.Attribute) and name in READ_METHODS | WRITE_METHODS:
            resolved = self._eval(func.value, env)
            if resolved is not None:
                op: Literal["read", "write"] = "read" if name in READ_METHODS else "write"
                if name == "open":
                    op = "write" if _opens_for_writing(call) else "read"
                if name in {"glob", "rglob"} and call.args:
                    # The glob pattern is part of the path: ``belief_dir(d).glob("round*/x.parquet")``
                    # names ``L2/round*/x.parquet``, and charging it to ``L2`` would report the
                    # directory rather than the artifact.
                    extra = _literal_segments(call.args[0])
                    resolved = tuple(branch + extra for branch in resolved)
                self._record(call, op, resolved, name)
                return

        if name in READ_FUNCTIONS | WRITE_FUNCTIONS | WRITE_HELPERS or (isinstance(func, ast.Name) and name == "open"):
            for argument in [*call.args, *(kw.value for kw in call.keywords)]:
                resolved = self._eval(argument, env)
                if resolved is None:
                    continue
                if name == "open":
                    self._record(call, "write" if _opens_for_writing(call) else "read", resolved, str(name))
                elif name in READ_FUNCTIONS:
                    self._record(call, "read", resolved, str(name))
                else:
                    self._record(call, "write", resolved, str(name))
                return

    def _record(self, node: ast.AST, op: Literal["read", "write"], alternatives: _Alternatives, via: str) -> None:
        for segments in alternatives:
            if not segments:
                # The run directory itself is not an artifact: every stage may create it and any
                # stage may be handed it. Only what a stage does *inside* it is under contract.
                continue
            pattern = "/".join(segments)
            lineno = getattr(node, "lineno", 0)
            line = self.lines[lineno - 1].strip() if 0 < lineno <= len(self.lines) else ""
            self.findings.append(Finding(self.module, lineno, op, pattern, self.stage, line, via))

    def _eval(self, node: ast.AST, env: Mapping[str, _Alternatives]) -> _Alternatives | None:
        """Every run-relative path this expression could denote, or ``None`` when unresolvable."""
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            return _RUN_ROOT if node.id in _RUN_ROOT_NAMES else None
        if isinstance(node, ast.Attribute):
            if node.attr in _RUN_ROOT_ATTRS:
                return _RUN_ROOT
            if node.attr in _PASS_DIR_ATTRS:
                return _PERTURBATION_DIRS
            return None
        if isinstance(node, ast.Call):
            return self._eval_call(node, env)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            left = self._eval(node.left, env)
            if left is None:
                return None
            extra = _literal_segments(node.right)
            return tuple(branch + extra for branch in left)
        return None

    def _eval_call(self, node: ast.Call, env: Mapping[str, _Alternatives]) -> _Alternatives | None:
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name in _LAYOUT_ROOTS:
            return (_LAYOUT_ROOTS[str(name)],)
        if name == "perturbation_dir":
            return _PERTURBATION_DIRS
        if name in _ROUND_DIRS:
            # The round index is a runtime value, so it resolves to ``*``: which round a module
            # writes into is the unrolled DAG's question; the static guard's is only which
            # *directory* it wrote to.
            return (_ROUND_DIRS[str(name)],)
        if name == "Path" and node.args:
            return self._eval(node.args[0], env)
        if isinstance(func, ast.Attribute) and name == "joinpath":
            base = self._eval(func.value, env)
            if base is None:
                return None
            extra: tuple[str, ...] = ()
            for argument in node.args:
                extra += _literal_segments(argument)
            return tuple(branch + extra for branch in base)
        return None


_ROUND_DIRS: Final[Mapping[str, tuple[str, ...]]] = {
    "belief_dir": ("L2",),
    "round_dir": ("L2", "round", "*"),
    "estimates_dir": ("L2", "round", "*", "estimates"),
    "derivatives_dir": ("L2", "round", "*", "derivatives"),
}
"""Layout helpers naming a place in the round tree.

``belief_dir`` used to take an optional round index and return ``L2`` or ``L2/round<n>``. One
helper returning a directory *and* its child made the root a place things could be written, which
is how nine per-round quantities came to sit flattened at ``L2/`` with no round to belong to.
"""

_NAMED_SEGMENTS: Final[Mapping[str, str]] = {
    "EVIDENCE_DIR": "L1",
    "BELIEF_DIR": "L2",
    "FINAL_DIR": "final",
    "REGISTER_FILENAME": "perturbations.json",
}
"""Module constants that stand for a fixed path segment.

Vocabulary, not policy — the same role :data:`DIMENSION_COLUMNS` plays for columns. A stage that
spells a filename through a constant is doing the right thing; a guard that could only read
string literals would punish it, and the workaround (inlining the literal beside the constant)
is a second spelling of one location.
"""


def _literal_segments(node: ast.AST) -> tuple[str, ...]:
    """Path segments contributed by the right-hand side of a ``/``.

    A non-literal contributes ``*``: the guard knows a segment is there without knowing its name,
    which is enough to place the path in a directory and is all a contract is written against.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return _segments(node.value)
    if isinstance(node, (ast.Attribute, ast.Name)):
        name = node.attr if isinstance(node, ast.Attribute) else node.id
        if name in _NAMED_SEGMENTS:
            return (_NAMED_SEGMENTS[name],)
    if isinstance(node, ast.JoinedStr):
        # An f-string resolves as far as its literal prefix: ``f"round{n}"`` is ``round*``.
        rendered = ""
        for value in node.values:
            rendered += value.value if isinstance(value, ast.Constant) and isinstance(value.value, str) else "*"
        return _segments(rendered)
    return ("*",)


def _opens_for_writing(call: ast.Call) -> bool:
    """Does this ``open(...)`` write? Read is the default, as it is in Python."""
    modes = [a for a in call.args[1:] if isinstance(a, ast.Constant)]
    modes += [kw.value for kw in call.keywords if kw.arg == "mode" and isinstance(kw.value, ast.Constant)]
    return any(isinstance(m.value, str) and set(m.value) & set("wax+") for m in modes)


def _own_nodes(scope: ast.AST) -> Iterator[ast.AST]:
    """Every node belonging to this scope, without descending into a nested one."""
    for child in ast.iter_child_nodes(scope):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        yield child
        yield from _own_nodes(child)


def _nested_scopes(scope: ast.AST) -> Iterator[ast.AST]:
    """The scopes defined directly inside this one."""
    for child in ast.iter_child_nodes(scope):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            yield child
        else:
            yield from _nested_scopes(child)


def pipeline_sources(repo_root: Path) -> tuple[Path, ...]:
    """Every module that participates in producing a run: the whole package, plus both drivers.

    The whole package including ``adaptive/`` — a previous guard globbed one directory and so
    could not see the subsystem that holds half the round logic.
    """
    package = repo_root / "src" / "senselab" / "audio" / "workflows" / "audio_analysis"
    files = sorted(package.rglob("*.py"))
    files += [repo_root / "scripts" / "analyze_audio.py", repo_root / "scripts" / "adaptive_loop.py"]
    return tuple(path for path in files if path.is_file())


def stage_for(module: str) -> StageContract:
    """The contract a module speaks under. Unlisted modules are :data:`PURE`."""
    return STAGE_CONTRACTS[MODULE_STAGE.get(module, "PURE")]


def _is_ancestor(created: str, declared: str) -> bool:
    """Is ``created`` a directory on the way to something ``declared``?

    Making a directory is not writing an artifact, so a stage may create its own tree; what it
    may not do is create another stage's. Positional, and shorter-or-equal: ``L2/round*`` leads
    to ``L2/round/*/timeline.png`` and ``final`` does not.
    """
    made, target = _segments(created), _segments(declared)
    if len(made) > len(target):
        return False
    return all(fnmatchcase(a, b) or fnmatchcase(b, a) for a, b in zip(made, target))


def check_source(module: str, source: str, contract: StageContract | None = None) -> list[Finding]:
    """Every run-relative read or write in one module that its contract does not permit.

    Conformance is **subsumption**, not intersection: every path the access could name must fall
    inside the contract. A segment the resolver could not name resolves to ``*``, and a ``*``
    that *might* land inside the declaration is not proof that it does — an access the guard
    cannot prove conformant is not a permitted one. Taking intersection instead was tried and
    silently unenforced the rule that matters most here: ``pass_dir(run_dir, stream) / "asr"``
    resolves to ``L1/*/asr``, whose ``*`` intersects the ``signals`` in ``L1/signals/**``, so
    every read of ``L1/<perturbation>/asr/`` from ``adaptive/`` read as permitted.
    """
    contract = (contract or stage_for(module)).instantiate()
    permitted_writes = contract.write_patterns
    # A stage reads its own outputs as a matter of course — they are its state. What it may not
    # do is reach into another node's tree.
    permitted_reads = tuple(contract.reads) + permitted_writes
    resolver = _PathResolver(module, contract.stage, source.splitlines())
    offenders: list[Finding] = []
    for finding in resolver.run(ast.parse(source)):
        allowed = permitted_reads if finding.op == "read" else permitted_writes
        if finding.via == "mkdir":
            if any(_is_ancestor(finding.pattern, pattern) for pattern in allowed):
                continue
        elif any(matches(finding.pattern, pattern) for pattern in allowed):
            continue
        offenders.append(finding)
    return offenders


#: The guard itself walks artifact trees by construction, so scanning it would report the
#: mechanism as a violation of the thing it enforces.
GUARD_MODULE: Final[str] = "src/senselab/audio/workflows/audio_analysis/contracts.py"


def static_violations(repo_root: Path) -> list[Finding]:
    """Conformance of every pipeline module against its declared contract."""
    offenders: list[Finding] = []
    for path in pipeline_sources(repo_root):
        module = path.relative_to(repo_root).as_posix()
        if module == GUARD_MODULE:
            continue
        offenders += check_source(module, path.read_text())
    return offenders


# ── the dynamic guard ────────────────────────────────────────────────────────


def _declared_artifacts() -> tuple[Artifact, ...]:
    """Every artifact any stage declares, with the round placeholders made generic."""
    return tuple(artifact for stage in DAG_STAGES for artifact in STAGE_CONTRACTS[stage].instantiate().writes)


def _table_columns(path: Path) -> frozenset[str] | None:
    """A parquet's column names, or ``None`` when the file is not a table."""
    if path.suffix != ".parquet":
        return None
    import pyarrow.parquet as pq

    return frozenset(pq.read_schema(path).names)


def _key_violations(relative: str, artifact: Artifact, columns: frozenset[str]) -> list[str]:
    """Where a table's columns contradict the key the artifact was declared with.

    Three rules, all derived from the key rather than listed — see :class:`Artifact`.
    """
    if artifact.key is None:
        return []
    problems: list[str] = []
    for dimension, spellings in DIMENSION_COLUMNS.items():
        present = sorted(columns & spellings)
        if dimension not in artifact.key:
            if present:
                problems.append(
                    f"{relative}: keyed {artifact.key}, so it is not indexed by {dimension} — but it carries {present}"
                )
            continue
        if not present and dimension in artifact.must_carry():
            problems.append(
                f"{relative}: keyed {artifact.key} and the path does not fix {dimension}, "
                f"so a row cannot say which one it came from"
            )
        if len(present) > 1 and dimension not in INTERVAL_DIMENSIONS:
            problems.append(
                f"{relative}: one {dimension} per row, but {present} names more than one — "
                f"relating two values of an input dimension is a fold, which belongs to L2"
            )
    folds = sorted(columns & FOLD_COLUMNS)
    if folds and "axis" not in artifact.key:
        problems.append(f"{relative}: keyed {artifact.key} but carries fold column(s) {folds}")
    return problems


def artifact_violations(run_dir: Path) -> list[str]:
    """Every file in a real run that no stage declared, and every table whose key contradicts it.

    Static analysis cannot see a path handed to a helper as a parameter, nor a file emitted by a
    library the caller pointed somewhere unexpected. Walking what was actually written can.
    """
    root = Path(run_dir)
    declared = _declared_artifacts()
    problems: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        owners = [artifact for artifact in declared if matches(relative, artifact.pattern)]
        if not owners:
            problems.append(f"{relative}: written by no declared stage output")
            continue
        columns = _table_columns(path)
        if columns is None:
            continue
        for artifact in owners:
            problems += _key_violations(relative, artifact, columns)
    return problems


# ── the register's own arithmetic ────────────────────────────────────────────


def unwaived(findings: Iterable[Finding]) -> list[Finding]:
    """Static findings no entry in :data:`KNOWN_DEVIATIONS` accounts for."""
    waived = {(d.module, d.op, d.pattern) for d in KNOWN_DEVIATIONS if d.op in {"read", "write"}}
    return [f for f in findings if (f.module, f.op, f.pattern) not in waived]


def dead_static_deviations(findings: Iterable[Finding]) -> list[Deviation]:
    """Register entries that no longer match anything — a fixed violation must be deleted.

    Without this the register decays into a permanent exemption list, which is the failure mode
    of every waiver mechanism: the entry outlives the defect and then silently covers the next
    one that happens to land on the same line.
    """
    seen = {(f.module, f.op, f.pattern) for f in findings}
    return [d for d in KNOWN_DEVIATIONS if d.op in {"read", "write"} and (d.module, d.op, d.pattern) not in seen]


def unwaived_artifacts(problems: Iterable[str]) -> list[str]:
    """Artifact-tree problems no entry in :data:`KNOWN_DEVIATIONS` accounts for."""
    waived = tuple(d.pattern for d in KNOWN_DEVIATIONS if d.op in {"artifact", "key"})
    return [p for p in problems if not any(matches(p.split(":", 1)[0], pattern) for pattern in waived)]
