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
   no stage's declared outputs, any file whose *kind* the declaring pattern does not permit, any
   file the guard could not read, and any table whose key contradicts the artifact it was written
   as. Its mirror :func:`unproduced_declarations` flags the opposite: a declared output a complete
   run produces nothing for. Together they catch what static analysis cannot: a writer reached
   through a helper, a file nobody meant to emit, and a declaration nobody satisfies.

**A guard is defeated by the case it does not consider, and every one of those was found by
constructing it rather than by reading the code.** Four were, and closing them is what the shape
of this module is now for:

- a declaration broad enough to permit anything. ``**`` used to be free; it now costs a pinned
  set of ``suffixes``, a ``key`` that prohibits at least one dimension, and conformance to
  :func:`structural_vocabulary` — and :meth:`Artifact.__post_init__` refuses the declaration
  outright rather than leaving the guard to go quiet beneath it.
- a content rule that falls to a file extension. The key rules read every format in
  :data:`TABULAR_SUFFIXES`, the declaration pins which of them may appear where, and a file that
  cannot be read is :class:`UnreadableArtifact` — a finding, never a pass.
- a path bound in a way the resolver did not watch for. Assignment is one of seven binding forms
  (:data:`_BINDING_NODES`); tuple targets, starred targets, ``/=``, walrus and ``for`` were the
  six that were not.
- a real-run fixture that passed on a fragment. Completeness is judged against the declaration,
  by the same :func:`unproduced_declarations` that reports a declaration nothing satisfies.

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
import csv
import graphlib
import json
import math
from dataclasses import dataclass, replace
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Final, Iterable, Iterator, Literal, Mapping, Sequence

__all__ = [
    "ENUMERABLE_DIMENSIONS",
    "KNOWN_DEVIATIONS",
    "MODULE_STAGE",
    "STAGE_CONTRACTS",
    "TABULAR_SUFFIXES",
    "Artifact",
    "Deviation",
    "Finding",
    "StageContract",
    "UnreadableArtifact",
    "artifact_violations",
    "dag_edges",
    "dead_artifact_deviations",
    "declared_artifacts",
    "enumerated_members",
    "folding_stages",
    "matches",
    "overlap",
    "pipeline_sources",
    "static_violations",
    "structural_vocabulary",
    "topological_order",
    "unproduced_declarations",
    "unrolled_contracts",
    "unwaived_unproduced",
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
    "bucket": frozenset({"start", "end", "bucket_start", "bucket_end"}),
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


TABULAR_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".parquet", ".feather", ".arrow", ".csv", ".tsv", ".json", ".jsonl", ".ndjson"}
)
"""Every suffix this repo can write a table as, and therefore every suffix the key rules must be
able to read. The guard used to read ``.parquet`` and return "not a table" for everything else,
which made the whole content half of the declaration optional: the same rows written as
``speaker.csv`` conformed. A format the repo can produce and the guard cannot read is a hole the
size of every rule below."""


ENUMERABLE_DIMENSIONS: Final[frozenset[str]] = frozenset({"round", "axis"})
"""The dimensions whose members are knowable without opening a file, and therefore the only ones
a declaration may require *every* member of.

A wildcard is otherwise satisfied by a single match, which is the hole this closes: on a five-round
run where two rounds wrote a ``summary.json``, ``L2/round/*/summary.json`` matched something and
read as produced; on the same run the fourth axis stopped after round 2 and
``L2/round/*/estimates/*.parquet`` matched three axes in five rounds and read as produced. Both are
"a declaration nothing satisfies" wearing the one match that hides it.

A round is enumerable because the tree names its rounds; an axis is enumerable because
:data:`senselab.audio.workflows.audio_analysis.axes.AXIS_NAMES` declares them. A *perturbation* is
deliberately not here: its set is open and lives in ``L1/perturbations.json``, so requiring every
member would mean reading a run artifact to decide what the declaration says."""


def _enumeration_slot(pattern: str, dimension: str) -> int | None:
    """Which segment of ``pattern`` a member of ``dimension`` is substituted into.

    Positional rather than token-based, because ``instantiate`` rewrites ``{n}`` to ``*`` before
    any guard sees the pattern — a token check would pass on the declaration and fail on the
    thing actually walked. A round is the segment after the literal ``round``; an axis is the
    filename stem. ``None`` when the pattern has no such slot, which
    :meth:`Artifact.__post_init__` refuses.
    """
    segments = _segments(pattern)
    if dimension == "round":
        for index, segment in enumerate(segments[:-1]):
            if segment == "round" and {"*", "{"} & set(segments[index + 1]):
                return index + 1
        return None
    if dimension == "axis":
        return len(segments) - 1 if segments and segments[-1].startswith("*.") else None
    return None


def _substitute_segment(pattern: str, index: int, value: str) -> str:
    """``pattern`` with segment ``index`` replaced — the stem only, where it has a suffix."""
    segments = list(_segments(pattern))
    suffix = _pattern_suffix(pattern)
    segments[index] = f"{value}{suffix}" if index == len(segments) - 1 and suffix else value
    return "/".join(segments)


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

    ``folded`` names dimensions this artifact **relates** rather than indexes, so a row may spell
    one of them more than once: cross-perturbation stability carries ``pass_a`` beside ``pass_b``
    because comparing two perturbations is the whole content of the file. It is a licence only a
    deciding stage may take — an L1 artifact that declared it would be measuring and folding in
    the same breath — and :func:`folding_stages` is where that is enforced, because an artifact
    does not know which stage declares it.

    ``suffixes`` pins the file kinds permitted at this pattern. It defaults to the extension the
    pattern itself names (``final/transcript.json`` permits ``.json`` and nothing else), and must
    be given explicitly wherever the pattern ends in ``**`` and therefore names none.

    ``enumerated`` names the dimensions whose **every member** must appear. A wildcard is
    otherwise satisfied by one match, which is how ``L2/round/{n}/summary.json`` read as produced
    on a run where three of five rounds wrote none, and how ``L2/round/{n}/estimates/*.parquet``
    read as produced on a run where the fourth axis stopped after round 2. Only two dimensions
    are enumerable without opening a file — the rounds a run wrote and the axes the design
    declares — so those are the two :data:`ENUMERATED_TOKENS` knows how to expand.
    """

    pattern: str
    what: str
    key: tuple[str, ...] | None = None
    keyed_in_path: tuple[str, ...] = ()
    required: tuple[str, ...] | None = None
    folded: tuple[str, ...] = ()
    suffixes: tuple[str, ...] | None = None
    enumerated: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject a declaration too broad to constrain anything, at the moment it is written.

        **Breadth is refused rather than discouraged**, and refused *here* rather than in a test,
        because of how a broad declaration fails: it does not raise, it goes quiet. A ``**`` with
        no content rule makes every file beneath it conform, so the guard's report is
        indistinguishable from a clean tree — and a rule enforced only by a test can be violated
        by any caller that imports this module without running that test, which includes the guard
        itself when it is called from a script. Refusing at construction converts a silent
        weakening of the guard into a loud failure on the line that wrote it. The test suite still
        proves the refusal fires; it cannot be the only thing that does.

        Three things a ``**`` must pay for its breadth with:

        - ``suffixes``, because ``**`` names no extension and the key rules can only check a file
          they can read — an unpinned suffix is how ``L1/signals/speaker.csv`` conformed;
        - ``key``, because ``key=None`` means "not a table" and beneath a ``**`` that is a claim
          about a whole subtree rather than about one file;
        - a proper subset of the dimensions, because a key naming all of them prohibits nothing.
          ``L1/raw/**`` may not be indexed by an axis or a round; that prohibition is the reason
          the pattern is allowed to be broad at all.

        The fourth payment is not per-artifact and so is not checked here: a path matched only by
        a ``**`` is also held to :func:`structural_vocabulary`, which is derived from every
        declaration at once.
        """
        unknown = (set(self.key or ()) | set(self.keyed_in_path) | set(self.required or ()) | set(self.folded)) - set(
            DIMENSION_COLUMNS
        )
        if unknown:
            raise ValueError(f"{self.pattern}: {sorted(unknown)} name no key dimension in DIMENSION_COLUMNS")
        unexpandable = set(self.enumerated) - set(ENUMERABLE_DIMENSIONS)
        if unexpandable:
            raise ValueError(
                f"{self.pattern}: enumerated={sorted(unexpandable)} names a dimension nothing can expand — "
                f"only {sorted(ENUMERABLE_DIMENSIONS)} have a member set knowable without opening a file"
            )
        for dimension in self.enumerated:
            if _enumeration_slot(self.pattern, dimension) is None:
                raise ValueError(
                    f"{self.pattern}: enumerated={self.enumerated} names {dimension!r}, but the pattern has no "
                    f"wildcard in the place a {dimension} goes, so no member could be substituted into it"
                )
        if not set(self.folded) <= set(self.key or ()):
            raise ValueError(f"{self.pattern}: folded={self.folded} names a dimension outside key={self.key}")
        if "**" not in _segments(self.pattern):
            if not self.permitted_suffixes():
                raise ValueError(
                    f"{self.pattern}: names no file extension and declares no suffixes, so any file kind conforms"
                )
            return
        if not self.suffixes:
            raise ValueError(
                f"{self.pattern}: a '**' names no extension, so it must declare suffixes — otherwise the "
                "same rows written as .csv or .feather fall outside every content rule"
            )
        if self.key is None:
            raise ValueError(
                f"{self.pattern}: a '**' with key=None applies no content rule to anything beneath it, "
                "so every file under it conforms and the guard reports a clean tree"
            )
        if set(self.key) >= set(DIMENSION_COLUMNS):
            raise ValueError(
                f"{self.pattern}: key={self.key} names every dimension, so it prohibits none — "
                "breadth of location has to be paid for with narrowness of content"
            )

    def must_carry(self) -> frozenset[str]:
        """Dimensions a row has to spell out. Derived unless the artifact overrides it."""
        if self.required is not None:
            return frozenset(self.required)
        return frozenset(self.key or ()) - frozenset(self.keyed_in_path)

    def permitted_suffixes(self) -> frozenset[str]:
        """File kinds permitted here: what the pattern names, or what the artifact declares."""
        if self.suffixes is not None:
            return frozenset(self.suffixes)
        named = _pattern_suffix(self.pattern)
        return frozenset({named}) if named else frozenset()

    def slices_of_one_table(self) -> bool:
        """Is every file matching this pattern the same table, sliced by its key?

        **Derived, not declared**, from what the pattern's wildcards are: when every one of them
        enumerates a dimension, the files beneath it differ only in the value of a key dimension
        — which is the definition of a slice. ``L2/round/*/estimates/*.parquet`` varies in round
        and axis and in nothing else, so a round-3 file with different columns than a round-0 one
        is two artifacts sharing a name, and a reader cannot tell which producer wrote a round.
        ``L1/signals/**`` varies in *which tool measured*, so its files are different tables by
        construction and no such rule applies.

        The consequence is one schema per artifact name. Where two producers genuinely emit
        different things, the declaration has to say so by declaring two artifacts.
        """
        wildcards = sum(1 for segment in _segments(self.pattern) if {"*", "?", "{"} & set(segment))
        return self.key is not None and wildcards > 0 and len(self.enumerated) == wildcards

    def instances(self, members: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
        """Every concrete pattern this artifact owes, one per combination of enumerated members.

        Empty when the artifact enumerates nothing — the pattern-level rule then stands alone,
        which is all a non-enumerable wildcard can support.
        """
        if not self.enumerated:
            return ()
        patterns = [self.pattern]
        for dimension in self.enumerated:
            slot = _enumeration_slot(self.pattern, dimension)
            assert slot is not None  # noqa: S101 — __post_init__ refuses the declaration otherwise
            patterns = [_substitute_segment(p, slot, member) for p in patterns for member in members.get(dimension, ())]
        return tuple(patterns)


def _pattern_suffix(pattern: str) -> str | None:
    """The extension a pattern names, or ``None`` when its last segment names none."""
    segments = _segments(pattern)
    if not segments:
        return None
    last = segments[-1]
    if last == "**" or "." not in last:
        return None
    suffix = last[last.rindex(".") :]
    return None if {"*", "?"} & set(suffix) else suffix


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
        # A concrete index leaves no slot for a round member to be substituted into, and the claim
        # was never about this node anyway: "every round writes a summary" is a statement about the
        # generic contract, and the unrolled node is one round.
        writes = tuple(
            replace(
                artifact,
                pattern=_substitute(artifact.pattern, subs),
                enumerated=tuple(d for d in artifact.enumerated if d != "round"),
            )
            for artifact in self.writes
        )
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
            suffixes=(".json", ".parquet"),
        ),
        Artifact(
            "L1/perturbation/*/**",
            "each further transform's model outputs",
            key=("perturbation", "signal", "bucket", "speaker"),
            keyed_in_path=("perturbation",),
            required=(),
            suffixes=(".json", ".parquet"),
        ),
        Artifact(
            "L1/signals/**",
            "per-signal measurements accumulating across raw and every perturbation — L2's only input",
            key=("perturbation", "signal", "bucket"),
            suffixes=(".parquet",),
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
    # What a round owes, and what it merely may leave. Three artifacts are owed by *every* round —
    # its belief, its account of how it got there, and its view — because every round has all
    # three; a round missing one is a round whose trajectory cannot be read. The four derivative
    # families are not: ``votes/`` and ``stability/`` are the ingest round's, computed once from
    # L1 and unchanged afterwards, and ``regions.json``/``votes_added.parquet`` belong to the
    # rounds that ran interventions. Writing an empty one from a round that does no region
    # proposal would be the claim "we looked and found none", which is not what happened —
    # absent is not zero, and that distinction is exactly what ``enumerated`` is for.
    writes=(
        # The derivatives are named one family at a time rather than swept up by a ``**``. The
        # single broad pattern that used to stand here carried ``key=None``, so nothing beneath it
        # was checked at all and ``derivatives/estimates/speaker.parquet`` — a per-perturbation
        # axis table, the exact thing D-16 says cannot exist — sat inside the declaration without
        # a finding. There are three families and they have three different keys; one pattern
        # could only describe them by describing none of them.
        Artifact(
            "L2/round/{n}/derivatives/votes/*.parquet",
            "one source's statement about one bucket of one perturbation, per axis",
            key=("axis", "perturbation", "signal", "bucket"),
            keyed_in_path=("axis",),
        ),
        Artifact(
            "L2/round/{n}/derivatives/votes_added.parquet",
            "the votes this round added, beside the estimates they moved",
            key=("axis", "perturbation", "signal", "bucket", "round"),
            keyed_in_path=("round",),
        ),
        Artifact(
            "L2/round/{n}/derivatives/stability/*.parquet",
            "one signal's cross-perturbation disagreement per bucket",
            key=("perturbation", "signal", "bucket"),
            # The fold this file *is*: a row relates two perturbations, which is why it is a round
            # derivative and not evidence. Only a deciding stage may say this.
            folded=("perturbation",),
        ),
        Artifact(
            "L2/round/{n}/derivatives/regions.json",
            "the high-uncertainty regions this round proposed",
            key=("axis", "bucket"),
            required=("axis",),
        ),
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
            # Every active axis, every round. The fourth axis had estimates in rounds 0-2 and none
            # in 3-4, and under a rule satisfied by one match that read as produced — while the
            # convergence report, asked about an axis the loop carried no belief through, answered
            # "0 buckets, residual mass 0.0", which is *settled* rather than *never asked*.
            enumerated=("round", "axis"),
        ),
        Artifact("L2/round/{n}/timeline.png", "the same figure the final timeline draws", enumerated=("round",)),
        Artifact(
            "L2/round/{n}/summary.json",
            "what this round did, and what it now estimates",
            enumerated=("round",),
        ),
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
        Artifact(
            "final/per_speaker_presence.parquet",
            "one track per hypothesised speaker",
            key=("bucket", "speaker"),
        ),
        # One declaration for the axes, enumerated over the axis set rather than written out. The
        # four it replaces named speech_presence, asr, background_mask and a "final/speaker/"
        # directory — so the deliverable set was a list of three axes with the *speaker* axis
        # missing altogether, which is precisely the failure ``axes.AXES`` exists to make
        # impossible. Same key and same schema as a round's estimate, because that is what these
        # files are: the last round's, copied.
        Artifact(
            "final/estimates/*.parquet",
            "the last round's estimates, extracted verbatim — one file per active axis",
            key=("axis", "bucket", "round"),
            keyed_in_path=("axis",),
            enumerated=("axis",),
        ),
        Artifact("final/decisions.json", "trajectory, reversals, stopping reason, every intervention"),
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
    reads=(
        "final/transcript.json",
        "final/diarization.json",
        "final/decisions.json",
        "final/estimates/*.parquet",
    ),
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
    op: Literal["read", "write", "artifact", "key", "unproduced"]
    pattern: str
    why: str


_FOREIGN_FILES: Final[frozenset[str]] = frozenset({".DS_Store", "Thumbs.db", ".gitkeep"})
"""Files no stage wrote and none claims — skipped rather than reported as undeclared outputs."""

_FOREIGN_SUFFIXES: Final[frozenset[str]] = frozenset({".pyc", ".swp", ".tmp", ".part"})
"""Suffixes belonging to tooling rather than to the run."""


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
_FINAL_AT_L2_ROOT = (
    "Declared as a deliverable and written to the L2 root instead, so final/ carries no such "
    "file at all. The mirror of the L2/ artifact entry for the same name: one defect, seen from "
    "the side that declares and from the side that writes. Closes with that entry."
)
_LOOP_INLINES_FINAL = (
    "The loop calls the FINAL stage inline rather than the DAG invoking it, so an L2 node writes "
    "final/decisions.json and drives the extraction into final/estimates/. The content is right — "
    "final/ now carries the run's account and the last round's axes — and the *caller* is not: "
    "these are FINAL's artifacts written from an L2 node. Closes with the same restructure that "
    "un-inlines the driver."
)
_AXIS_NEVER_EXTRACTED = (
    "final/ is an extraction of the last round's estimates, and this axis is never extracted: "
    "every consumer reads L2/round/<n>/estimates/ directly. A deliverable nothing produces is a "
    "declaration nothing satisfies — as wrong as an artifact nothing declares, and quieter."
)

KNOWN_DEVIATIONS: Final[tuple[Deviation, ...]] = (
    # ── the driver performs all three stages itself ─────────────────────────
    Deviation(_DRIVER, "write", "L1/signals/*.parquet", _INLINED),
    Deviation(_DRIVER, "write", "L1/raw/pii.json", _INLINED),
    Deviation(_DRIVER, "write", "L1/perturbation/*/pii.json", _INLINED),
    Deviation(_DRIVER, "write", "L1/raw/embeddings/*.json", _INLINED),
    Deviation(_DRIVER, "write", "L1/perturbation/*/embeddings/*.json", _INLINED),
    # No separate entry for background_mask.parquet: the mask's votes are written by the same loop
    # over ``HARVESTED_AXES`` as every other axis's, from the same per-bucket harvest. The second,
    # per-region write that needed its own entry is gone.
    Deviation(_DRIVER, "write", "L2/round/*/derivatives/votes/*.parquet", _INLINED),
    Deviation(_DRIVER, "write", "L2/round/*/derivatives/stability/*.parquet", _INLINED),
    Deviation(_DRIVER, "write", "L2/disagreements.json", _INLINED),
    Deviation(_DRIVER, "write", "L2/labelstudio_tasks.json", _INLINED),
    Deviation(_DRIVER, "write", "L2/labelstudio_config.xml", _INLINED),
    Deviation(_DRIVER, "write", "final/summary.json", _INLINED),
    Deviation(_DRIVER, "write", "final/run_summary.json", _INLINED),
    Deviation(_DRIVER, "write", "final/summary.md", _INLINED),
    Deviation(_DRIVER, "read", "final/speakers.json", _INLINED),
    Deviation(_DRIVER, "read", "final/per_speaker_presence.parquet", _INLINED),
    Deviation(_ADAPTIVE_DRIVER, "read", "L1/perturbations.json", _INLINED),
    Deviation(
        _mod("stage_io.py"),
        "read",
        "*",
        "StageIO.locate is the capability itself, so it is the one place a path legitimately "
        "exists — the static guard has no way to express 'this module is the authority' and can "
        "only see a read of '*'. Not an exemption: the existence check was moved here out of "
        "measurements.py precisely because the guard flagged a helper holding a raw path, which "
        "is the shape of every defeat the earlier guards suffered. Closes when the static guard "
        "is deleted (removal-ledger Step 2), since D-18 replaces path inspection with capability "
        "passing and this finding is an artifact of the model being replaced.",
    ),
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
    Deviation(_mod("adaptive/loop.py"), "write", "final", _LOOP_INLINES_FINAL),
    Deviation(_mod("adaptive/loop.py"), "write", "final/decisions.json", _LOOP_INLINES_FINAL),
    Deviation(_mod("adaptive/plot.py"), "read", "L2/background_mask.parquet", _AT_L2_ROOT),
    Deviation(_mod("adaptive/ls_final.py"), "read", "L2/labelstudio_tasks.json", _AT_L2_ROOT),
    Deviation(_mod("adaptive/ls_final.py"), "read", "L2/labelstudio_config.xml", _AT_L2_ROOT),
    Deviation(_mod("adaptive/ls_final.py"), "read", "L2/disagreements.json", _AT_L2_ROOT),
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
    Deviation("", "artifact", "L2/disagreements.json", _AT_L2_ROOT),
    Deviation("", "artifact", "L2/background_mask.*", _AT_L2_ROOT),
    Deviation("", "artifact", "L2/labelstudio_*", _AT_L2_ROOT),
    Deviation("", "artifact", "triage.json", "An L2-shaped decision at the run root, taken before L1 has run."),
    # ══ declarations a complete run satisfies with nothing ═══════════════════
    # The other half of the artifact question, and the one nothing used to ask. A declaration
    # nothing produces is as wrong as an artifact nothing declares: both make "which stage
    # produces this" unanswerable, and this half is the more dangerous, because every content
    # rule passes on a file that is not there — which is how a 26-file fragment was accepted as a
    # completed run.
    Deviation(
        "",
        "unproduced",
        "final/eval.json",
        "EVAL scores the deliverable against ground truth, and a run without ground truth has "
        "nothing to score. The one declared output whose absence is a property of the *input* "
        "rather than of the pipeline — recorded here so that distinction is stated rather than "
        "assumed by a guard that quietly tolerates every absence.",
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
than a register. The ``unproduced`` entries are matched by **string equality** instead: their
subject is a declared pattern rather than a path, so waiving ``final/speaker/*.parquet`` by glob
would waive whatever else that glob happened to reach.

Both are now live-checked, against the *recorded* complete tree rather than against whatever run
is on the machine. The exemption they used to carry — run trees legitimately differ, so an
unmatched entry is evidence of nothing — was true of an arbitrary run and false of a fixed one,
and under it ``L1/perturbation/*/**`` outlived its defect by two steps of the restructure.
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
        "merge_json",
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
        bindings = [n for n in _own_nodes(scope) if isinstance(n, _BINDING_NODES)]
        for _ in range(len(bindings) + 1):
            before = dict(env)
            for node in bindings:
                self._bind_node(node, env)
            if env == before:
                break

        for call in (n for n in _own_nodes(scope) if isinstance(n, ast.Call)):
            self._check_call(call, env)

        for child in _nested_scopes(scope):
            self._scope(child, env)

    # ── binding ──────────────────────────────────────────────────────────────
    #
    # A path is under contract wherever it is named, and Python has more ways to name one than
    # ``x = expr``. Recording only ``ast.Name`` targets meant that
    # ``speakers_path, presence_path = belief / "speakers.json", belief / "..."`` bound two run
    # artifacts the guard could not see, and every use of either was silent afterwards. The
    # binding form is not the rule; the path is.

    def _bind_node(self, node: ast.AST, env: dict[str, _Alternatives]) -> None:
        """Record whatever run-relative paths one binding statement puts into scope."""
        if isinstance(node, ast.AugAssign):
            # ``p /= "speakers.json"`` extends the path in place. Read as a plain rebinding it
            # reported the *directory*, which is a finding naming the wrong artifact.
            if not isinstance(node.op, ast.Div):
                return
            base = self._eval(node.target, env)
            if base is None:
                return
            extra = _literal_segments(node.value)
            self._bind(node.target, tuple(branch + extra for branch in base), env)
            return
        if isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            self._bind_iteration(node.target, node.iter, env)
            return
        if isinstance(node, ast.Assign):
            targets: Sequence[ast.expr] = node.targets
        else:
            assert isinstance(node, (ast.AnnAssign, ast.NamedExpr))
            targets = [node.target]
        for target in targets:
            self._bind_value(target, getattr(node, "value", None), env)

    def _bind_value(self, target: ast.expr, value: ast.AST | None, env: dict[str, _Alternatives]) -> None:
        """Bind one assignment target, unpacking element-wise when both sides are sequences."""
        if value is None:
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            elements = self._eval_elements(value, env)
            if elements is not None:
                self._bind_sequence(target.elts, elements, env)
                return
            # ``a, b = f()`` — the call is one value the resolver cannot take apart, so each name
            # gets every path it could have been. Over-broad by construction, and deliberately:
            # an access the guard cannot prove conformant is not a permitted one.
            whole = self._eval(value, env)
            if whole is not None:
                self._bind(target, whole, env)
            return
        resolved = self._eval(value, env)
        if resolved is not None:
            self._bind(target, resolved, env)

    def _bind(self, target: ast.expr, resolved: _Alternatives, env: dict[str, _Alternatives]) -> None:
        """Bind one target to one disjunction, descending through starred and nested targets."""
        if isinstance(target, ast.Starred):
            self._bind(target.value, resolved, env)
        elif isinstance(target, ast.Name):
            env[target.id] = resolved
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._bind(element, resolved, env)

    def _bind_sequence(
        self, targets: Sequence[ast.expr], elements: Sequence[_Alternatives | None], env: dict[str, _Alternatives]
    ) -> None:
        """Element-wise unpacking, with a starred target absorbing the middle as a disjunction."""
        starred = [index for index, target in enumerate(targets) if isinstance(target, ast.Starred)]
        if not starred:
            if len(targets) != len(elements):
                return
            for target, resolved in zip(targets, elements):
                if resolved is not None:
                    self._bind(target, resolved, env)
            return
        at = starred[0]
        after = len(targets) - at - 1
        if len(elements) < len(targets) - 1:
            return
        for target, resolved in zip(targets[:at], elements[:at]):
            if resolved is not None:
                self._bind(target, resolved, env)
        absorbed = _union(elements[at : len(elements) - after])
        if absorbed:
            self._bind(targets[at], absorbed, env)
        if after:
            for target, resolved in zip(targets[at + 1 :], elements[len(elements) - after :]):
                if resolved is not None:
                    self._bind(target, resolved, env)

    def _bind_iteration(self, target: ast.expr, iterable: ast.AST, env: dict[str, _Alternatives]) -> None:
        """Bind a ``for`` (or comprehension) target over a sequence the resolver can take apart."""
        elements = self._eval_elements(iterable, env)
        if elements is None:
            # ``for path in signals_dir(run).glob("*.parquet")`` — the iterable is one expression
            # naming every path it yields, so the loop variable is that same disjunction.
            resolved = self._eval(iterable, env)
            if resolved is not None:
                self._bind(target, resolved, env)
            return
        if isinstance(target, (ast.Tuple, ast.List)) and isinstance(iterable, (ast.Tuple, ast.List, ast.Set)):
            columns: list[list[_Alternatives]] = [[] for _ in target.elts]
            for item in iterable.elts:
                per_item = self._eval_elements(item, env)
                if per_item is None or len(per_item) != len(target.elts):
                    return
                for index, resolved in enumerate(per_item):
                    if resolved is not None:
                        columns[index].append(resolved)
            for element, branches in zip(target.elts, columns):
                absorbed = _union(branches)
                if absorbed:
                    self._bind(element, absorbed, env)
            return
        absorbed = _union(elements)
        if absorbed:
            self._bind(target, absorbed, env)

    def _eval_elements(self, node: ast.AST, env: Mapping[str, _Alternatives]) -> list[_Alternatives | None] | None:
        """Per-element resolution of a literal sequence, or ``None`` when it is not one."""
        if not isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return None
        return [self._eval(element, env) for element in node.elts]

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
        if isinstance(node, ast.NamedExpr):
            # ``(p := belief / "x.json").exists()`` names the path in the same expression it
            # reads it in, so the walrus has to resolve as a *value* and not only as a binding.
            return self._eval(node.value, env)
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
        if isinstance(func, ast.Attribute) and name in {"glob", "rglob", "iterdir"}:
            # What the call *yields*, so a ``for`` over it binds the paths rather than nothing.
            base = self._eval(func.value, env)
            if base is None:
                return None
            if name == "iterdir":
                return tuple(branch + ("*",) for branch in base)
            below: tuple[str, ...] = ("**",) if name == "rglob" else ()
            below += _literal_segments(node.args[0]) if node.args else ("*",)
            return tuple(branch + below for branch in base)
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


_BINDING_NODES: Final[tuple[type[ast.AST], ...]] = (
    ast.Assign,
    ast.AnnAssign,
    ast.AugAssign,
    ast.NamedExpr,
    ast.For,
    ast.AsyncFor,
    ast.comprehension,
)
"""Every statement form that can put a run-relative path into scope.

Enumerated because Python's binding forms are a closed set, unlike the violations they can hide.
The list that preceded it held three of the seven, and the four it omitted were not exotic: a
tuple assignment, a ``for``, an in-place ``/=`` and a comprehension are ordinary lines that
happened to be invisible."""


def _union(alternatives: Iterable[_Alternatives | None]) -> _Alternatives:
    """One disjunction covering every branch of several, in first-seen order."""
    seen: dict[tuple[str, ...], None] = {}
    for resolved in alternatives:
        for branch in resolved or ():
            seen.setdefault(branch, None)
    return tuple(seen)


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
    return tuple(artifact for stage, artifact in declared_artifacts())


def declared_artifacts() -> tuple[tuple[str, Artifact], ...]:
    """Every declared artifact paired with the stage that declares it."""
    return tuple((stage, artifact) for stage in DAG_STAGES for artifact in STAGE_CONTRACTS[stage].instantiate().writes)


def folding_stages() -> Mapping[str, tuple[str, ...]]:
    """Which stage declares each artifact that takes the ``folded`` licence.

    Separate from :class:`Artifact` because an artifact does not know who declares it, and the
    rule is about the declarer: relating two values of an input dimension is a decision, so a
    measuring stage may not take it.
    """
    taken: dict[str, list[str]] = {}
    for stage, artifact in declared_artifacts():
        if artifact.folded:
            taken.setdefault(stage, []).append(artifact.pattern)
    return {stage: tuple(patterns) for stage, patterns in taken.items()}


def structural_vocabulary(artifacts: Sequence[Artifact] | None = None) -> Mapping[str, frozenset[int]]:
    """Every literal *directory* segment the declaration uses, and the depths it uses it at.

    This is the fourth thing a ``**`` is held to, and the only one that is not per-artifact: it is
    derived from every declaration at once, so it grows and shrinks with the tree rather than
    being a list of reserved words somebody has to remember to extend.

    The rule it supports is short. A ``**`` may admit a segment the declaration never mentions —
    that is what makes ``L1/raw/asr/whisper.json`` a legitimate tool output nobody had to
    predict. What it may not admit is a segment the declaration *does* mention, at a depth the
    declaration does not put it: ``L1/raw/final/transcript.json`` and
    ``L1/raw/estimates/asr.parquet`` are another stage's shape smuggled inside L1's open tree,
    and both used to conform. Filenames are excluded because a filename is not structure —
    ``derivatives/votes/asr.parquet`` and ``final/asr.parquet`` are two files, not a collision.
    """
    places: dict[str, set[int]] = {}
    for artifact in _declared_artifacts() if artifacts is None else artifacts:
        segments = _segments(artifact.pattern)
        for index, segment in enumerate(segments[:-1]):
            if {"*", "?"} & set(segment):
                continue
            places.setdefault(segment, set()).add(index)
    return {segment: frozenset(indices) for segment, indices in places.items()}


class UnreadableArtifact(Exception):
    """A file the guard could not read, and therefore could not check.

    Raised rather than swallowed because the two used to be the same outcome: ``_table_columns``
    returned ``None`` for anything that was not a parquet, and ``None`` meant "not a table, no
    rules apply". A file the guard cannot open has not passed anything.
    """


def _table_columns(path: Path) -> frozenset[str] | None:
    """Column names of a file written as a table, or ``None`` when it is not one.

    Every format the repo can write, not just parquet — see :data:`TABULAR_SUFFIXES`. A JSON
    document is a table only when it is a non-empty list of objects: an object is a document, and
    an empty list has no row that could contradict a key.
    """
    suffix = path.suffix.lower()
    if suffix not in TABULAR_SUFFIXES:
        return None
    try:
        if suffix == ".parquet":
            import pyarrow.parquet as pq

            return frozenset(pq.read_schema(path).names)
        if suffix in {".feather", ".arrow"}:
            import pyarrow.feather as feather

            return frozenset(feather.read_table(path).schema.names)
        if suffix in {".csv", ".tsv"}:
            with path.open(newline="", encoding="utf-8") as handle:
                header = next(csv.reader(handle, delimiter="\t" if suffix == ".tsv" else ","), None)
            return frozenset(header) if header else None
        return _json_record_columns(path, suffix)
    except Exception as exc:  # noqa: BLE001 — every failure to read is the same finding
        raise UnreadableArtifact(f"{type(exc).__name__}: {exc}") from exc


def _json_records(path: Path, suffix: str) -> list[dict[str, Any]]:
    """The rows of a JSON document that is a list of records, or empty when it is not one.

    An object is a *document* and an empty list has no row that could contradict anything, so both
    read as "not a table" — the same judgement, made once, for the name rules and the value rules.
    """
    text = path.read_text(encoding="utf-8")
    if suffix in {".jsonl", ".ndjson"}:
        records: list[object] = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            return []
        records = list(parsed)
    rows = [record for record in records if isinstance(record, dict)]
    if not rows or len(rows) != len(records):
        return []
    return rows


def _json_record_columns(path: Path, suffix: str) -> frozenset[str] | None:
    """Column names of a JSON document that is a list of records, else ``None``."""
    rows = _json_records(path, suffix)
    if not rows:
        return None
    return frozenset().union(*(frozenset(row) for row in rows))


def _suffix_violation(relative: str, suffix: str, owners: Sequence[Artifact]) -> str | None:
    """Where a file's extension is not one the pattern that admitted it permits."""
    if any(suffix in artifact.permitted_suffixes() for artifact in owners):
        return None
    permitted = sorted({s for artifact in owners for s in artifact.permitted_suffixes()})
    return (
        f"{relative}: {suffix or 'no extension'} is not a permitted file kind here — "
        f"{', '.join(artifact.pattern for artifact in owners)} permits {permitted}"
    )


def _vocabulary_violations(
    relative: str, owners: Sequence[Artifact], vocabulary: Mapping[str, frozenset[int]]
) -> list[str]:
    """Where a ``**`` admitted a segment the declaration places somewhere else.

    Only for a path that *every* matching pattern reached through a ``**``: a bounded pattern
    fixed each of its segments itself, so there is nothing it could have swallowed.
    """
    if any("**" not in _segments(artifact.pattern) for artifact in owners):
        return []
    segments = _segments(relative)
    problems: list[str] = []
    for index, segment in enumerate(segments[:-1]):
        places = vocabulary.get(segment)
        if places is None or index in places:
            continue
        problems.append(
            f"{relative}: '{segment}' is the tree's own name for depth {sorted(places)}, and here it is at "
            f"depth {index} under {', '.join(artifact.pattern for artifact in owners)} — "
            f"a '**' does not license another stage's shape"
        )
    return problems


def _path_fixed_dimensions(relative: str, artifact: Artifact) -> dict[str, str]:
    """What the *location* of ``relative`` says each of its path-fixed dimensions is.

    Only the dimensions :func:`_enumeration_slot` can place — ``round`` and ``axis``. A
    ``perturbation`` is also fixed by its path, but the set of names a perturbation segment may
    take is open, so there is no slot to read it out of and this returns nothing for it rather
    than guessing which segment it was.
    """
    segments = _segments(relative)
    fixed: dict[str, str] = {}
    for dimension in artifact.keyed_in_path:
        slot = _enumeration_slot(artifact.pattern, dimension)
        if slot is None or slot >= len(segments):
            continue
        segment = segments[slot]
        if slot == len(segments) - 1 and "." in segment:
            segment = segment[: segment.rindex(".")]
        fixed[dimension] = segment
    return fixed


def _dimension_token(value: Any) -> str | None:  # noqa: ANN401 — any cell a table can hold
    """How a path segment would spell ``value``, or ``None`` where it says nothing.

    A null is skipped rather than reported: "this row does not repeat what the path said" is
    exactly what ``keyed_in_path`` permits, and the column-presence rules are :func:`_key_violations`'
    business. What is checked here is only a row that *does* speak and contradicts its location.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return None if math.isnan(value) else (str(int(value)) if value.is_integer() else str(value))
    if isinstance(value, int):
        return str(value)
    text = str(value).strip()
    return text or None


def _location_violations(relative: str, artifact: Artifact, columns: frozenset[str], path: Path) -> list[str]:
    """Where a row's own value for a path-fixed dimension disagrees with its location.

    The rules above are about column *names*; this one is about values, and it is the only rule
    that can catch the defect it exists for. ``L2/round/4/estimates/speech_presence.parquet`` held
    rows whose ``round`` said 1, 3 and 4 — every name-level rule passed, because the column is
    declared and there is only one spelling of it. What was wrong was that three of those numbers
    named a directory the file is not in, and anything deriving a path from the column (the
    disagreements index does) then sent a reader to a different fold's numbers.

    ``keyed_in_path`` is what licenses the check: the declaration says the location already fixes
    this dimension, so a column repeating it is provenance and provenance that contradicts its own
    subject is worse than no provenance at all.
    """
    fixed = _path_fixed_dimensions(relative, artifact)
    if not fixed:
        return []
    wanted = {
        column: dimension
        for dimension in fixed
        for column in sorted(DIMENSION_COLUMNS.get(dimension, frozenset()) & columns)
    }
    if not wanted:
        return []
    values = _table_column_values(path, tuple(wanted))
    problems: list[str] = []
    for column, dimension in sorted(wanted.items()):
        disagreeing = sorted(
            {
                t
                for value in values.get(column, ())
                if (t := _dimension_token(value)) != fixed[dimension] and t is not None
            }
        )
        if disagreeing:
            problems.append(
                f"{relative}: its location fixes {dimension}={fixed[dimension]!r}, but its {column!r} column "
                f"carries {disagreeing} — a row that contradicts the directory it is in points every "
                f"consumer that reads the column at another {dimension}'s numbers"
            )
    return problems


def _table_column_values(path: Path, names: Sequence[str]) -> dict[str, list[Any]]:
    """The values ``names`` take in a table, for the same file kinds :func:`_table_columns` reads.

    Read column-wise where the format allows it, so checking a value costs the two columns the
    declaration names rather than the whole frame.
    """
    suffix = path.suffix.lower()
    try:
        if suffix == ".parquet":
            import pyarrow.parquet as pq

            table = pq.read_table(path, columns=list(names))
            return {name: table.column(name).to_pylist() for name in names}
        if suffix in {".feather", ".arrow"}:
            import pyarrow.feather as feather

            table = feather.read_table(path, columns=list(names))
            return {name: table.column(name).to_pylist() for name in names}
        if suffix in {".csv", ".tsv"}:
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t" if suffix == ".tsv" else ","))
            return {name: [row.get(name) for row in rows] for name in names}
        records = _json_records(path, suffix)
        return {name: [record.get(name) for record in records] for name in names}
    except Exception as exc:  # noqa: BLE001 — every failure to read is the same finding
        raise UnreadableArtifact(f"{type(exc).__name__}: {exc}") from exc


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
        if len(present) > 1 and dimension not in INTERVAL_DIMENSIONS and dimension not in artifact.folded:
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

    Six questions per file, in the order a defeat gets past them: is it declared at all; did a
    ``**`` admit it by admitting another stage's shape; is its *kind* one the declaring pattern
    permits; do its columns contradict the key; do its column *values* contradict the location the
    declaration says fixes them; and — across files rather than per file — do two slices of one
    artifact carry different columns. The third and fourth used to be one question that only
    parquet could fail; the fifth is the only one that looks at a value, and it is what a row in
    ``L2/round/4/`` claiming ``round: 1`` gets past when every name-level rule is satisfied; the
    sixth is a question about a *set* of files and so could not be asked at all by a loop that
    looked at one at a time.
    """
    root = Path(run_dir)
    declared = _declared_artifacts()
    vocabulary = structural_vocabulary(declared)
    problems: list[str] = []
    shapes: dict[str, dict[frozenset[str], list[str]]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        # Not written by the pipeline and not a claim about anything: a desktop file manager's index,
        # a Python cache, an editor's swap. Reporting them as undeclared outputs buries the findings
        # that are about *this* code among findings about whoever opened the folder.
        if path.name in _FOREIGN_FILES or path.suffix in _FOREIGN_SUFFIXES:
            continue
        relative = path.relative_to(root).as_posix()
        owners = [artifact for artifact in declared if matches(relative, artifact.pattern)]
        if not owners:
            problems.append(f"{relative}: written by no declared stage output")
            continue
        problems += _vocabulary_violations(relative, owners, vocabulary)
        wrong_kind = _suffix_violation(relative, path.suffix.lower(), owners)
        if wrong_kind is not None:
            problems.append(wrong_kind)
            continue
        try:
            columns = _table_columns(path)
        except UnreadableArtifact as unreadable:
            problems.append(f"{relative}: could not be read ({unreadable}), so nothing about it has been checked")
            continue
        if columns is None:
            continue
        for artifact in owners:
            problems += _key_violations(relative, artifact, columns)
            try:
                problems += _location_violations(relative, artifact, columns, path)
            except UnreadableArtifact as unreadable:
                problems.append(f"{relative}: could not be read ({unreadable}), so its key values are unchecked")
            if artifact.slices_of_one_table():
                shapes.setdefault(artifact.pattern, {}).setdefault(columns, []).append(relative)
    return problems + _shape_violations(shapes)


def _shape_violations(shapes: Mapping[str, Mapping[frozenset[str], Sequence[str]]]) -> list[str]:
    """Where one artifact name covers two schemas.

    Reported against the **majority** shape rather than against an arbitrary first file, so the
    message names the odd producer instead of whichever slice the walk happened to reach first.
    Columns present in one group and absent in the other are listed both ways: a reader of
    ``L2/round/3/estimates/asr.parquet`` needs to know both what it gained and what it lost
    relative to round 2's, because it is the same declared quantity and neither difference is
    visible from the path.
    """
    problems: list[str] = []
    for pattern, by_columns in sorted(shapes.items()):
        if len(by_columns) < 2:
            continue
        groups = sorted(by_columns.items(), key=lambda item: (-len(item[1]), sorted(item[1])[0]))
        majority, majority_files = groups[0]
        for columns, files in groups[1:]:
            problems.append(
                f"{sorted(files)[0]}: two shapes under one artifact name — {pattern} is one table sliced by its "
                f"key, but this slice carries {sorted(columns - majority)} which {sorted(majority_files)[0]} "
                f"does not, and lacks {sorted(majority - columns)} which it does"
            )
    return problems


def enumerated_members(run_dir: Path) -> Mapping[str, tuple[str, ...]]:
    """The member set of each enumerable dimension for one run.

    Rounds come from the tree, because how many a run took is a property of that run. Axes come
    from :mod:`~senselab.audio.workflows.audio_analysis.axes`, because which axes exist is a
    property of the design — reading them off the tree instead would make "the fourth axis
    stopped after round 2" self-justifying, since a tree that stopped producing it would also
    stop declaring it.
    """
    from senselab.audio.workflows.audio_analysis.axes import AXIS_NAMES

    rounds = Path(run_dir) / "L2" / "round"
    present = (
        sorted((p.name for p in rounds.iterdir() if p.is_dir() and p.name.isdigit()), key=int)
        if (rounds.is_dir())
        else []
    )
    return {"round": tuple(present), "axis": tuple(AXIS_NAMES)}


def unproduced_declarations(run_dir: Path, declared: Sequence[tuple[str, Artifact]] | None = None) -> list[str]:
    """Declared outputs a **complete** run produced nothing for.

    The guard nobody had written, and the mirror of the one above: a declaration nothing
    satisfies is as wrong as an artifact nothing declares. Both make "which stage produces this"
    unanswerable, and the unproduced half is the more dangerous of the two, because it is what
    lets a fixture judge a fragment complete — every rule passes on a file that is not there.

    Asked twice, because one match is not production. The pattern-level question ("did anything
    match at all") is what a non-enumerable wildcard can support; where the wildcard enumerates a
    dimension the members are known, and each is asked for separately. On the run that prompted
    this, ``L2/round/*/summary.json`` matched three files out of five rounds and
    ``L2/round/*/estimates/*.parquet`` matched three axes out of four — both reported produced,
    both a declaration two thirds of the tree does not satisfy.

    Read against a run known to be complete. On a partial tree every declaration is unproduced
    and the answer says nothing, which is exactly why :func:`unproduced_declarations` is also
    what completeness is judged by: the two are the same question asked for different reasons.
    """
    root = Path(run_dir)
    artifacts = declared_artifacts() if declared is None else tuple(declared)
    produced = [path.relative_to(root).as_posix() for path in sorted(root.rglob("*")) if path.is_file()]
    members = enumerated_members(root)
    problems: list[str] = []
    for stage, artifact in artifacts:
        if not any(matches(relative, artifact.pattern) for relative in produced):
            problems.append(
                f"{artifact.pattern}: declared by {stage} ({artifact.what}) and the run produced nothing matching it"
            )
            continue
        problems += [
            f"{instance}: declared by {stage} ({artifact.pattern}) and the run produced nothing matching it"
            for instance in artifact.instances(members)
            if not any(matches(relative, instance) for relative in produced)
        ]
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


def unwaived_unproduced(problems: Iterable[str]) -> list[str]:
    """Unproduced declarations no ``unproduced`` register entry accounts for.

    Matched by string equality on the declared pattern rather than by :func:`matches`, because
    the subject here is a *pattern* and not a path: waiving ``final/speaker/*.parquet`` by glob
    would waive whatever else the glob happens to cover.
    """
    waived = {d.pattern for d in KNOWN_DEVIATIONS if d.op == "unproduced"}
    return [p for p in problems if p.split(":", 1)[0] not in waived]


def dead_artifact_deviations(problems: Iterable[str]) -> list[Deviation]:
    """Artifact-side register entries that waive nothing in ``problems``.

    The static register has had this check since it was written; the artifact register was
    exempted from it on the grounds that run trees legitimately differ. They do — which is why
    this is asked of the *recorded* complete tree rather than of whatever run happens to be on
    the machine. Against a fixed tree the argument for the exemption disappears, and what is left
    is the same rot: an entry that outlives its defect and then silently covers the next one.
    """
    subjects = [problem.split(":", 1)[0] for problem in problems]
    return [
        deviation
        for deviation in KNOWN_DEVIATIONS
        if deviation.op in {"artifact", "key", "unproduced"}
        and not any(subject == deviation.pattern or matches(subject, deviation.pattern) for subject in subjects)
    ]
