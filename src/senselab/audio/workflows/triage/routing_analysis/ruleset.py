"""The family taxonomy ruleset: which branches a recording's own content routes it to.

Every branch's gates are evaluated on every recording. The task instruction routes nothing: it is
read through the family sets in :mod:`~senselab.audio.workflows.triage.routing_analysis.families`
into :attr:`RouteEvaluation.declared`, which is the reference standard the routed set is scored
against and never a filter on which gates run.

Every gate is one feature path, one comparison and one threshold, all of them read from
``taxonomy.ruleset`` in ``data/config/default.yaml``. The operating points, which of them are
provisional, and the open questions are in
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
class Ruleset:
    """The whole declarative rule set, as loaded from the configuration.

    Attributes:
        gates: Every gate, by name.
        branch_gates: Branch to its gates, any one of which routes it on any recording.
        reference_family_set: Branch to the :data:`FAMILY_SETS` entry it is scored against. This
            mapping is a reference standard, not a router: no gate is skipped because of it.
    """

    gates: Mapping[str, Gate]
    branch_gates: Mapping[str, tuple[str, ...]]
    reference_family_set: Mapping[str, str]

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
        fell_through: Whether ``routed`` is empty.
        gate_outcomes: Every gate's outcome, by gate name.
    """

    stem: str
    family: str
    routed: tuple[str, ...]
    declared: tuple[str, ...]
    agreed: tuple[str, ...]
    missed: tuple[str, ...]
    extra: tuple[str, ...]
    unavailable: Mapping[str, tuple[str, ...]]
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
        fell_through: How many routed to no branch at all.
    """

    family: str
    recordings: int
    routed: Mapping[str, int]
    declared: Mapping[str, int]
    agreed: Mapping[str, int]
    missed: Mapping[str, int]
    extra: Mapping[str, int]
    unavailable: Mapping[str, int]
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
    gates = {
        name: Gate(name=name, feature=tuple(row["feature"]), op=str(row["op"]), threshold=float(row["threshold"]))
        for name, row in config.require(f"{RULESET_PATH}.gates").items()
    }

    for branch, set_name in reference.items():
        if set_name not in FAMILY_SETS:
            raise ValueError(f"{RULESET_PATH}.reference_family_set[{branch}] names no family set: {set_name!r}")
    for branch, names in branch_gates.items():
        for name in names:
            if name not in gates:
                raise ValueError(f"{RULESET_PATH}.branch_gates[{branch}] names no gate: {name!r}")
    for gate in gates.values():
        if gate.op not in (AT_LEAST, AT_MOST):
            raise ValueError(f"{RULESET_PATH}.gates[{gate.name}].op is not a comparison: {gate.op!r}")

    return Ruleset(gates=gates, branch_gates=branch_gates, reference_family_set=reference)


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

    Every branch's gates are evaluated, whatever the task asked for.

    Args:
        features: The recording's extracted evidence.
        ruleset: The loaded ruleset.

    Returns:
        The evaluation.
    """
    outcomes: dict[str, GateOutcome] = {}
    routed: list[str] = []
    unavailable: dict[str, tuple[str, ...]] = {}
    for branch in BRANCHES:
        names = ruleset.branch_gates.get(branch, ())
        states = {name: _outcome(features, ruleset, outcomes, name) for name in names}
        if GateOutcome.FIRED in states.values():
            routed.append(branch)
        unread = tuple(name for name, state in states.items() if state is GateOutcome.UNAVAILABLE)
        if unread:
            unavailable[branch] = unread

    declared = ruleset.reference_branches(features.family)
    return RouteEvaluation(
        stem=features.stem,
        family=features.family,
        routed=tuple(routed),
        declared=declared,
        agreed=tuple(branch for branch in routed if branch in declared),
        missed=tuple(branch for branch in declared if branch not in routed),
        extra=tuple(branch for branch in routed if branch not in declared),
        unavailable=unavailable,
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
    for evaluation in evaluations:
        family = evaluation.family
        rows = counters.setdefault(family, {axis: Counter() for axis in AXES})
        recordings[family] += 1
        fell_through[family] += int(evaluation.fell_through)
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


AXES: Sequence[str] = ("routed", "declared", "agreed", "missed", "extra", "unavailable")
"""The per-branch axes an evaluation carries and a tally counts."""


def branches_on(evaluation: RouteEvaluation, axis: str) -> tuple[str, ...]:
    """The branches one evaluation carries on one axis.

    Args:
        evaluation: The evaluation.
        axis: One of :data:`AXES`.

    Returns:
        The branches, in branch order. ``unavailable`` is keyed by branch rather than listed, so it
        contributes each branch that carried at least one unreadable gate.
    """
    if axis == "unavailable":
        return tuple(branch for branch in BRANCHES if branch in evaluation.unavailable)
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
