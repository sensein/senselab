"""The family taxonomy ruleset: which branches a recording's own content routes it to.

Every branch's gates are evaluated on every recording. The task instruction routes nothing: it is
read through the family sets in :mod:`~senselab.audio.workflows.triage.routing_analysis.families`
into :attr:`RouteEvaluation.declared`, which is the reference standard the routed set is scored
against and never a filter on which gates run.

Every gate is one feature path, one comparison and one threshold, all of them read from
``taxonomy.ruleset`` in ``data/config/default.yaml``. A gate either routes a branch or flags it:
``branch_gates`` decides entry, ``branch_flags`` annotates a branch already entered. Ahead of both,
one precondition asks whether the recording carried anything at all. The operating points, which of
them are provisional, and the open questions are in
``specs/20260817-triage-workflow-dag/family-taxonomy-ruleset.md``.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.routing_analysis.detectors import Detector, detector_value
from senselab.audio.workflows.triage.routing_analysis.families import DECLARED_KIND, SYLLABLE_REPETITION
from senselab.audio.workflows.triage.routing_analysis.features import RecordingFeatures
from senselab.audio.workflows.triage.routing_analysis.report import Confusion
from senselab.audio.workflows.triage.vocabulary import BRANCHES

RULESET_PATH = "taxonomy.ruleset"
"""Where the ruleset's data lives in the triage configuration."""

TRANSCRIPT_REPEAT = "transcript_repeat"
"""The feature source this module reads itself, because no extracted feature carries it."""

BRACKETED_SET = "bracketed_set"
"""The feature source whose arguments are a named bracket-token set resolved out of configuration."""

BRACKET_SET_PATHS: Mapping[str, str] = {"airway": "taxonomy.airway_bracket_tokens"}
"""Each named bracket-token set, and the configuration key that lists its types."""

AT_LEAST = "at_least"
AT_MOST = "at_most"

FAMILY_SETS: Mapping[str, frozenset[str]] = {**DECLARED_KIND, "syllable_repetition": SYLLABLE_REPETITION}
"""Every named family set a branch's reference standard may be taken from."""

_PUNCTUATION = re.compile(r"[^\w\s]+")


class GateOutcome(Enum):
    """What a gate concluded about one recording."""

    FIRED = "fired"
    SILENT = "silent"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class Gate:
    """One threshold rule over one number.

    Attributes:
        name: The gate's id, unique across the ruleset.
        feature: What to read, as a :func:`~senselab.audio.workflows.triage.routing_analysis.
            detectors.detector_value` reader, or ``("transcript_repeat",)``.
        op: ``at_least`` or ``at_most``.
        threshold: The cut the read number is compared against.
    """

    name: str
    feature: tuple[Any, ...]
    op: str
    threshold: float


@dataclass(frozen=True)
class Emptiness:
    """The precondition that decides a recording carried nothing to route.

    Attributes:
        peak_streams: The ``<stream>|<classifier>`` summaries whose highest tracked-label score
            must all fall below the floor.
        peak_floor: The score every named stream must stay under.
    """

    peak_streams: tuple[str, ...]
    peak_floor: float


@dataclass(frozen=True)
class Ruleset:
    """The whole declarative rule set, as loaded from the configuration.

    Attributes:
        gates: Every gate, by name.
        branch_gates: Branch to its gates, any one of which routes it on any recording.
        branch_flags: Branch to the gates that annotate it. A flag gate is evaluated and reported
            and never routes: it is read after a branch is entered, not to enter it.
        reference_family_set: Branch to the :data:`FAMILY_SETS` entry it is scored against. This
            mapping is a reference standard, not a router: no gate is skipped because of it.
        emptiness: The precondition evaluated ahead of every branch gate.
    """

    gates: Mapping[str, Gate]
    branch_gates: Mapping[str, tuple[str, ...]]
    branch_flags: Mapping[str, tuple[str, ...]]
    reference_family_set: Mapping[str, str]
    emptiness: Emptiness

    def reference_branches(self, family: str) -> tuple[str, ...]:
        """Which branches a family is a reference positive for.

        Args:
            family: A task family.

        Returns:
            The branches whose reference family set contains it, in :data:`~senselab.audio.
            workflows.triage.vocabulary.BRANCHES` order; empty when no set does.
        """
        return tuple(
            branch
            for branch in BRANCHES
            if branch in self.reference_family_set and family in FAMILY_SETS[self.reference_family_set[branch]]
        )


@dataclass(frozen=True)
class RouteEvaluation:
    """What the ruleset made of one recording.

    Attributes:
        stem: The recording's BIDS stem.
        family: The task family its stem collapses into.
        routed: The branches a gate fired for, from content alone, in branch order.
        declared: The branches the family is a reference positive for, for comparison only.
        agreed: ``routed`` and ``declared`` both.
        missed: Declared and not routed.
        extra: Routed and not declared.
        unavailable: Per branch, the gates whose feature could not be read, whether or not another
            gate routed the branch anyway. Only branches with such a gate are keyed.
        flags: Per branch, the flag gates that fired. A flag annotates a branch and never routes
            it, so it is absent from every other field here. Only branches with a fired flag are
            keyed.
        empty: Whether the emptiness precondition fired, in which case no branch gate ran.
        fell_through: Whether the recording carried content and still routed nowhere. An empty
            recording is not a fall-through.
        gate_outcomes: Every gate's outcome, by gate name; empty when ``empty``.
    """

    stem: str
    family: str
    routed: tuple[str, ...]
    declared: tuple[str, ...]
    agreed: tuple[str, ...]
    missed: tuple[str, ...]
    extra: tuple[str, ...]
    unavailable: Mapping[str, tuple[str, ...]]
    flags: Mapping[str, tuple[str, ...]]
    empty: bool
    fell_through: bool
    gate_outcomes: Mapping[str, GateOutcome]


@dataclass(frozen=True)
class FamilyTally:
    """One family's counts over a stream of evaluations.

    Attributes:
        family: The task family.
        recordings: How many evaluations carried it.
        routed: Per branch, how many routed there on content.
        declared: Per branch, how many the family is a reference positive for.
        agreed: Per branch, how many were routed and declared.
        missed: Per branch, how many were declared and not routed.
        extra: Per branch, how many were routed and not declared.
        unavailable: Per branch, how many carried an unreadable gate for it.
        flagged: Per branch, how many carried a fired flag gate for it.
        empty: How many the emptiness precondition fired on.
        fell_through: How many carried content and still routed to no branch at all. The empty
            ones are counted in ``empty`` and never here.
    """

    family: str
    recordings: int
    routed: Mapping[str, int]
    declared: Mapping[str, int]
    agreed: Mapping[str, int]
    missed: Mapping[str, int]
    extra: Mapping[str, int]
    unavailable: Mapping[str, int]
    flagged: Mapping[str, int]
    empty: int
    fell_through: int


def max_token_repeat(transcript: str) -> int:
    """How many times the most-repeated normalised token appears in a transcript.

    Args:
        transcript: The consensus transcript, possibly truncated at
            :data:`~senselab.audio.workflows.triage.routing_analysis.features.TRANSCRIPT_CAP`.

    Returns:
        The largest per-token count, or 0 when the transcript holds no token. Tokens are lowercased
        and split on whitespace and punctuation alike, so a token the cap cut in half is a token of
        its own rather than another instance of the one it came from.
    """
    tokens = _PUNCTUATION.sub(" ", transcript.lower()).split()
    return max(Counter(tokens).values(), default=0)


def _resolve_feature(config: TriageConfig, name: str, feature: tuple[Any, ...]) -> tuple[Any, ...]:
    """Expand a gate's feature path, substituting a named bracket-token set for its members.

    Args:
        config: The resolved configuration.
        name: The gate's name, for the error message.
        feature: The feature path as configured.

    Returns:
        The path a reader can be built from: unchanged, unless it names a bracket-token set, in
        which case the set name is replaced by that set's types in sorted order.

    Raises:
        ValueError: When the path names a bracket-token set :data:`BRACKET_SET_PATHS` does not
            have.
    """
    if feature[0] != BRACKETED_SET:
        return feature
    set_name = str(feature[1])
    if set_name not in BRACKET_SET_PATHS:
        raise ValueError(f"{RULESET_PATH}.gates[{name}].feature names no bracket-token set: {set_name!r}")
    return (BRACKETED_SET, *sorted(str(token) for token in config.require(BRACKET_SET_PATHS[set_name])))


def load_ruleset(config: TriageConfig) -> Ruleset:
    """Read the ruleset out of a resolved triage configuration.

    Args:
        config: The resolved configuration.

    Returns:
        The ruleset.

    Raises:
        ValueError: When a branch names a family set that does not exist, when a branch names a gate
            the ``gates`` mapping does not define, or when a gate names an unsupported comparison.
    """
    reference = dict(config.require(f"{RULESET_PATH}.reference_family_set"))
    branch_gates = {branch: tuple(names) for branch, names in config.require(f"{RULESET_PATH}.branch_gates").items()}
    branch_flags = {branch: tuple(names) for branch, names in config.require(f"{RULESET_PATH}.branch_flags").items()}
    gates = {
        name: Gate(
            name=name,
            feature=_resolve_feature(config, name, tuple(row["feature"])),
            op=str(row["op"]),
            threshold=float(row["threshold"]),
        )
        for name, row in config.require(f"{RULESET_PATH}.gates").items()
    }
    emptiness = Emptiness(
        peak_streams=tuple(str(stream) for stream in config.require(f"{RULESET_PATH}.emptiness.peak_streams")),
        peak_floor=float(config.require(f"{RULESET_PATH}.emptiness.peak_floor")),
    )

    for branch, set_name in reference.items():
        if set_name not in FAMILY_SETS:
            raise ValueError(f"{RULESET_PATH}.reference_family_set[{branch}] names no family set: {set_name!r}")
    for key, assignment in (("branch_gates", branch_gates), ("branch_flags", branch_flags)):
        for branch, names in assignment.items():
            for name in names:
                if name not in gates:
                    raise ValueError(f"{RULESET_PATH}.{key}[{branch}] names no gate: {name!r}")
    for branch, names in branch_gates.items():
        overlap = set(names) & set(branch_flags.get(branch, ()))
        if overlap:
            raise ValueError(f"{RULESET_PATH}.branch_flags[{branch}] also gates it: {sorted(overlap)}")
    for gate in gates.values():
        if gate.op not in (AT_LEAST, AT_MOST):
            raise ValueError(f"{RULESET_PATH}.gates[{gate.name}].op is not a comparison: {gate.op!r}")

    return Ruleset(
        gates=gates,
        branch_gates=branch_gates,
        branch_flags=branch_flags,
        reference_family_set=reference,
        emptiness=emptiness,
    )


def evaluate_emptiness(features: RecordingFeatures, emptiness: Emptiness) -> GateOutcome:
    """Whether a recording carried nothing at all, before any branch gate is asked anything.

    Args:
        features: The recording's extracted evidence.
        emptiness: The precondition.

    Returns:
        ``FIRED`` when every named stream's highest tracked-label score is under the floor,
        ``SILENT`` when at least one is at or over it, and ``UNAVAILABLE`` when a named stream's
        summary is not in the store at all, which is not a stream that scored zero.
    """
    for stream in emptiness.peak_streams:
        name, _, classifier = stream.partition("|")
        value = detector_value(
            features,
            Detector(name="emptiness", kind="", reader=("stream_peak_max", name, classifier), unit="", thresholds=()),
        )
        if value is None:
            return GateOutcome.UNAVAILABLE
        if value >= emptiness.peak_floor:
            return GateOutcome.SILENT
    return GateOutcome.FIRED


def gate_value(features: RecordingFeatures, gate: Gate) -> float | None:
    """The number a gate reads out of one recording.

    Args:
        features: The recording's extracted evidence.
        gate: The gate.

    Returns:
        The number, or None when the evidence it reads is not in the store.
    """
    if gate.feature[0] == TRANSCRIPT_REPEAT:
        return float(max_token_repeat(features.transcript)) if features.consensus_present else None
    return detector_value(features, Detector(name=gate.name, kind="", reader=gate.feature, unit="", thresholds=()))


def evaluate_gate(features: RecordingFeatures, gate: Gate) -> GateOutcome:
    """Whether a gate fired, stayed silent, or could not be read at all.

    Args:
        features: The recording's extracted evidence.
        gate: The gate.

    Returns:
        The outcome. ``UNAVAILABLE`` is not a negative: the measurement the gate needs was never
        written.
    """
    value = gate_value(features, gate)
    if value is None:
        return GateOutcome.UNAVAILABLE
    fired = value >= gate.threshold if gate.op == AT_LEAST else value <= gate.threshold
    return GateOutcome.FIRED if fired else GateOutcome.SILENT


def evaluate_routes(features: RecordingFeatures, ruleset: Ruleset) -> RouteEvaluation:
    """Route one recording from its content, then compare that against what its family declares.

    The emptiness precondition runs first. When it fires no branch gate is asked anything and the
    recording routes nowhere, which is distinct from carrying content that matched no gate. When it
    does not, every branch's gates are evaluated, whatever the task asked for, and every branch's
    flag gates are evaluated beside them without contributing to ``routed``.

    Args:
        features: The recording's extracted evidence.
        ruleset: The loaded ruleset.

    Returns:
        The evaluation.
    """
    declared = ruleset.reference_branches(features.family)
    if evaluate_emptiness(features, ruleset.emptiness) is GateOutcome.FIRED:
        return RouteEvaluation(
            stem=features.stem,
            family=features.family,
            routed=(),
            declared=declared,
            agreed=(),
            missed=declared,
            extra=(),
            unavailable={},
            flags={},
            empty=True,
            fell_through=False,
            gate_outcomes={},
        )

    outcomes: dict[str, GateOutcome] = {}
    routed: list[str] = []
    unavailable: dict[str, tuple[str, ...]] = {}
    flags: dict[str, tuple[str, ...]] = {}
    for branch in BRANCHES:
        names = ruleset.branch_gates.get(branch, ())
        states = {name: _outcome(features, ruleset, outcomes, name) for name in names}
        if GateOutcome.FIRED in states.values():
            routed.append(branch)
        unread = tuple(name for name, state in states.items() if state is GateOutcome.UNAVAILABLE)
        if unread:
            unavailable[branch] = unread
        raised = tuple(
            name
            for name in ruleset.branch_flags.get(branch, ())
            if _outcome(features, ruleset, outcomes, name) is GateOutcome.FIRED
        )
        if raised:
            flags[branch] = raised

    return RouteEvaluation(
        stem=features.stem,
        family=features.family,
        routed=tuple(routed),
        declared=declared,
        agreed=tuple(branch for branch in routed if branch in declared),
        missed=tuple(branch for branch in declared if branch not in routed),
        extra=tuple(branch for branch in routed if branch not in declared),
        unavailable=unavailable,
        flags=flags,
        empty=False,
        fell_through=not routed,
        gate_outcomes=outcomes,
    )


def tally_families(evaluations: Iterable[RouteEvaluation]) -> dict[str, FamilyTally]:
    """Aggregate a stream of evaluations into per-family counts.

    Args:
        evaluations: The evaluations, in any order.

    Returns:
        One tally per family seen, keyed by family name.
    """
    counters: dict[str, dict[str, Counter[str]]] = {}
    recordings: Counter[str] = Counter()
    fell_through: Counter[str] = Counter()
    empty: Counter[str] = Counter()
    for evaluation in evaluations:
        family = evaluation.family
        rows = counters.setdefault(family, {axis: Counter() for axis in AXES})
        recordings[family] += 1
        fell_through[family] += int(evaluation.fell_through)
        empty[family] += int(evaluation.empty)
        for axis in AXES:
            rows[axis].update(branches_on(evaluation, axis))
    return {
        family: FamilyTally(
            family=family,
            recordings=recordings[family],
            routed=_per_branch(rows["routed"]),
            declared=_per_branch(rows["declared"]),
            agreed=_per_branch(rows["agreed"]),
            missed=_per_branch(rows["missed"]),
            extra=_per_branch(rows["extra"]),
            unavailable=_per_branch(rows["unavailable"]),
            flagged=_per_branch(rows["flagged"]),
            empty=empty[family],
            fell_through=fell_through[family],
        )
        for family, rows in sorted(counters.items())
    }


def score_branches(evaluations: Iterable[RouteEvaluation]) -> dict[str, Confusion]:
    """Score content-only routing against the declared family sets, one 2x2 per branch.

    Args:
        evaluations: The evaluations, in any order.

    Returns:
        One :class:`~senselab.audio.workflows.triage.routing_analysis.report.Confusion` per branch,
        in branch order, carrying the raw counts and the sensitivity and specificity derived from
        them. A recording is a reference positive for a branch when its family is in that branch's
        reference family set, and a prediction positive when the branch is in ``routed``.
    """
    counts = {branch: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for branch in BRANCHES}
    for evaluation in evaluations:
        for branch in BRANCHES:
            predicted = branch in evaluation.routed
            positive = branch in evaluation.declared
            counts[branch]["tp" if predicted else "fn"] += int(positive)
            counts[branch]["fp" if predicted else "tn"] += int(not positive)
    return {branch: Confusion(**row) for branch, row in counts.items()}


AXES: Sequence[str] = ("routed", "declared", "agreed", "missed", "extra", "unavailable", "flagged")
"""The per-branch axes an evaluation carries and a tally counts."""

_KEYED_AXES: Mapping[str, str] = {"unavailable": "unavailable", "flagged": "flags"}
"""The axes an evaluation keys by branch rather than listing, and the field each is keyed in."""


def branches_on(evaluation: RouteEvaluation, axis: str) -> tuple[str, ...]:
    """The branches one evaluation carries on one axis.

    Args:
        evaluation: The evaluation.
        axis: One of :data:`AXES`.

    Returns:
        The branches, in branch order. ``unavailable`` and ``flagged`` are keyed by branch rather
        than listed, so each contributes every branch its mapping names.
    """
    field = _KEYED_AXES.get(axis)
    if field is not None:
        keyed: Mapping[str, tuple[str, ...]] = getattr(evaluation, field)
        return tuple(branch for branch in BRANCHES if branch in keyed)
    branches: tuple[str, ...] = getattr(evaluation, axis)
    return branches


def _per_branch(counter: Counter[str]) -> dict[str, int]:
    """One axis's counts, carrying every branch so an absent one reads zero rather than missing.

    Args:
        counter: The accumulated counts.

    Returns:
        The count per branch, in branch order.
    """
    return {branch: counter.get(branch, 0) for branch in BRANCHES}


def _outcome(features: RecordingFeatures, ruleset: Ruleset, outcomes: dict[str, GateOutcome], name: str) -> GateOutcome:
    """Evaluate one gate once per recording, recording what it concluded.

    Args:
        features: The recording's extracted evidence.
        ruleset: The loaded ruleset.
        outcomes: The per-recording record each evaluation is added to.
        name: The gate's name.

    Returns:
        The gate's outcome.
    """
    if name not in outcomes:
        outcomes[name] = evaluate_gate(features, ruleset.gates[name])
    return outcomes[name]
