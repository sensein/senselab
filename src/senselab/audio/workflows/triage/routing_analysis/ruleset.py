"""The family taxonomy ruleset: which branches a recording's own content routes it to.

Every branch's gates are evaluated on every recording. The task instruction routes nothing: it is
read through the family sets in :mod:`~senselab.audio.workflows.triage.routing_analysis.families`
into :attr:`RouteEvaluation.declared`, which is the reference standard the routed set is scored
against and never a filter on which gates run.

Every gate is one feature path, one comparison and one threshold, all of them read from
``taxonomy.ruleset`` in ``data/config/default.yaml``. A gate either routes a branch or flags it:
``branch_gates`` decides entry, ``branch_flags`` annotates a branch already entered. Behind both,
one bypass asks whether the recording carried anything at all, and it is consulted only where no
gate fired. The operating points, which of them are provisional, and the open questions are in
``specs/20260817-triage-workflow-dag/family-taxonomy-ruleset.md``.

A branch's reference family set may leave out families whose content is exactly what the branch is
for — ``lexical_speech`` excludes the syllable-repetition families by construction — so
``excluded_by_construction`` names them per branch and :func:`score_branches` reports the 2x2 both
with them in the negatives and with them held out of the population. :func:`branch_recall_curves`
scores every routing gate the way a router is scored, by recall at a fixed over-routing budget;
``specs/20260911-recall-first-thresholds/design.md`` says why that criterion and not Youden's J.
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
from senselab.audio.workflows.triage.routing_analysis.report import (
    OVER_ROUTING_BUDGETS,
    Confusion,
    RecallCurve,
    recall_at_budgets,
)
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

_POLARITY: Mapping[str, str] = {AT_LEAST: "above", AT_MOST: "below"}
"""Each comparison's firing side, in the polarity vocabulary the recall curve reads."""

FAMILY_SETS: Mapping[str, frozenset[str]] = {**DECLARED_KIND, "syllable_repetition": SYLLABLE_REPETITION}
"""Every named family set a branch's reference standard may be taken from."""

_PUNCTUATION = re.compile(r"[^\w\s]+")


class GateOutcome(Enum):
    """What a gate concluded about one recording."""

    FIRED = "fired"
    SILENT = "silent"
    UNAVAILABLE = "unavailable"


class RouteState(Enum):
    """What the ruleset made of one recording. Exactly one member holds for every evaluation.

    Only :attr:`UNEXPLAINED` is a charge against the ruleset: it is content no gate could account
    for. :attr:`EMPTY` is a charge against the recording, and :attr:`ROUTED` is neither.
    """

    ROUTED = "routed"
    EMPTY = "empty"
    UNEXPLAINED = "unexplained"


ROUTE_STATES: tuple[str, ...] = tuple(state.value for state in RouteState)
"""Every state's name, in declaration order, so a tally can carry all three whether or not seen."""


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
    """The bypass that decides a recording nothing routed carried nothing to route.

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
        excluded_by_construction: Branch to the :data:`FAMILY_SETS` entry its reference set leaves
            out although the content is what the branch is for. Those families are neither
            positives nor negatives: scoring holds them out of the population rather than charging
            the branch for routing them. A branch with nothing to hold out is absent from the
            mapping.
        emptiness: The bypass evaluated after every branch gate, and only where none fired.
    """

    gates: Mapping[str, Gate]
    branch_gates: Mapping[str, tuple[str, ...]]
    branch_flags: Mapping[str, tuple[str, ...]]
    reference_family_set: Mapping[str, str]
    excluded_by_construction: Mapping[str, str]
    emptiness: Emptiness

    def held_out(self, branch: str) -> frozenset[str]:
        """The families held out of one branch's scored population.

        Args:
            branch: The branch.

        Returns:
            That branch's construction exclusions, empty when it declares none.
        """
        set_name = self.excluded_by_construction.get(branch)
        return FAMILY_SETS[set_name] if set_name else frozenset()

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
        state: Which of the three outcomes this recording had.
        gate_outcomes: Every gate's outcome, by gate name. Every gate is evaluated on every
            recording, so this is never empty for a ruleset that declares one.
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
    state: RouteState
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
        states: How many landed in each :class:`RouteState`, keyed by its value. All three keys are
            present whether or not the family carried one, and they sum to ``recordings``.
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
    states: Mapping[str, int]


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
        ValueError: When a branch names a family set that does not exist, when a branch holds out
            families its own reference set calls positive, when a branch names a gate the ``gates``
            mapping does not define, or when a gate names an unsupported comparison.
    """
    reference = dict(config.require(f"{RULESET_PATH}.reference_family_set"))
    excluded = {
        branch: str(set_name)
        for branch, set_name in config.require(f"{RULESET_PATH}.excluded_by_construction").items()
        if set_name
    }
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
    for branch, set_name in excluded.items():
        if set_name not in FAMILY_SETS:
            raise ValueError(f"{RULESET_PATH}.excluded_by_construction[{branch}] names no family set: {set_name!r}")
        contradiction = FAMILY_SETS[set_name] & FAMILY_SETS.get(reference.get(branch, ""), frozenset())
        if contradiction:
            raise ValueError(
                f"{RULESET_PATH}.excluded_by_construction[{branch}] holds out families the reference set "
                f"calls positive: {sorted(contradiction)}"
            )
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
        excluded_by_construction=excluded,
        emptiness=emptiness,
    )


def evaluate_emptiness(features: RecordingFeatures, emptiness: Emptiness) -> GateOutcome:
    """Whether a recording carried nothing at all, asked only where no branch gate fired.

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

    Every branch's gates are evaluated, whatever the task asked for, and every branch's flag gates
    are evaluated beside them without contributing to ``routed``. Emptiness is a bypass rather than
    a precondition: it is consulted only where no gate fired, so a gate firing on a recording the
    emptiness rule would have called empty routes it normally. That disagreement is a reading of the
    emptiness rule, not something the evaluation suppresses.

    Args:
        features: The recording's extracted evidence.
        ruleset: The loaded ruleset.

    Returns:
        The evaluation.
    """
    declared = ruleset.reference_branches(features.family)
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

    if routed:
        state = RouteState.ROUTED
    elif evaluate_emptiness(features, ruleset.emptiness) is GateOutcome.FIRED:
        state = RouteState.EMPTY
    else:
        state = RouteState.UNEXPLAINED

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
        state=state,
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
    states: dict[str, Counter[str]] = {}
    for evaluation in evaluations:
        family = evaluation.family
        rows = counters.setdefault(family, {axis: Counter() for axis in AXES})
        recordings[family] += 1
        states.setdefault(family, Counter())[evaluation.state.value] += 1
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
            states={name: states[family].get(name, 0) for name in ROUTE_STATES},
        )
        for family, rows in sorted(counters.items())
    }


@dataclass(frozen=True)
class BranchScore:
    """One branch's content-only routing, scored against its reference family set two ways.

    Attributes:
        branch: The branch.
        reference_family_set: The :data:`FAMILY_SETS` entry positives were taken from.
        against_reference: The 2x2 over every recording.
        excluding_construction: The same, with the branch's construction exclusions dropped from
            the population. Identical to ``against_reference`` when the branch declares none.
        held_out_family_set: The :data:`FAMILY_SETS` entry that was held out, or None.
        n_held_out: How many recordings the exclusion removed.
    """

    branch: str
    reference_family_set: str
    against_reference: Confusion
    excluding_construction: Confusion
    held_out_family_set: str | None
    n_held_out: int

    def as_json(self) -> dict[str, Any]:
        """This branch's score, for the machine-readable output.

        Returns:
            Both 2x2 tables with their derived rates, and what was held out of the second.
        """
        return {
            "branch": self.branch,
            "reference_family_set": self.reference_family_set,
            "held_out_family_set": self.held_out_family_set,
            "n_held_out": self.n_held_out,
            "against_reference": self.against_reference.as_json(),
            "excluding_construction": self.excluding_construction.as_json(),
        }


def score_branches(evaluations: Iterable[RouteEvaluation], ruleset: Ruleset) -> dict[str, BranchScore]:
    """Score content-only routing against the reference family sets, one branch at a time.

    Each branch gets two 2x2 tables over the same evaluations. The first counts every recording.
    The second holds out the branch's construction exclusions — families the reference set leaves
    out although their content is what the branch is for — because a recording routed there is not
    an error, and counting it as one charges the branch for behaving correctly.

    Args:
        evaluations: The evaluations, in any order.
        ruleset: The loaded ruleset, read for each branch's reference and exclusion sets.

    Returns:
        One :class:`BranchScore` per branch, in branch order. A recording is a reference positive
        for a branch when its family is in that branch's reference family set, and a prediction
        positive when the branch is in ``routed``.
    """
    counts = {branch: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for branch in BRANCHES}
    kept = {branch: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for branch in BRANCHES}
    held_out = dict.fromkeys(BRANCHES, 0)
    for evaluation in evaluations:
        for branch in BRANCHES:
            predicted = branch in evaluation.routed
            positive = branch in evaluation.declared
            cell = ("tp" if positive else "fp") if predicted else ("fn" if positive else "tn")
            counts[branch][cell] += 1
            if evaluation.family in ruleset.held_out(branch):
                held_out[branch] += 1
            else:
                kept[branch][cell] += 1
    return {
        branch: BranchScore(
            branch=branch,
            reference_family_set=ruleset.reference_family_set.get(branch, ""),
            against_reference=Confusion(**counts[branch]),
            excluding_construction=Confusion(**kept[branch]),
            held_out_family_set=ruleset.excluded_by_construction.get(branch),
            n_held_out=held_out[branch],
        )
        for branch in BRANCHES
    }


@dataclass(frozen=True)
class GateRecall:
    """One routing gate's configured operating point, read against its recall-at-budget curve.

    Attributes:
        branch: The branch the gate routes.
        gate: The gate's name.
        threshold: The configured cut.
        configured: The 2x2 the configured cut produces on the curve's population.
        curve: What the same gate reaches at each over-routing budget.
    """

    branch: str
    gate: str
    threshold: float
    configured: Confusion
    curve: RecallCurve

    def as_json(self) -> dict[str, Any]:
        """This gate's recall report, for the machine-readable output.

        Returns:
            The configured point with its rates, and the curve.
        """
        return {
            "branch": self.branch,
            "gate": self.gate,
            "threshold": self.threshold,
            "configured": self.configured.as_json(),
            "curve": self.curve.as_json(),
        }


def gate_recall(
    records: Sequence[RecordingFeatures],
    ruleset: Ruleset,
    branch: str,
    gate_name: str,
    *,
    budgets: Sequence[float] = OVER_ROUTING_BUDGETS,
) -> GateRecall:
    """One gate's recall at each over-routing budget, beside the operating point it is configured at.

    Args:
        records: The recordings to score over.
        ruleset: The loaded ruleset.
        branch: The branch whose reference and exclusion sets define positives and the population.
        gate_name: The gate.
        budgets: The false-positive rates over the negatives that each point may not exceed.

    Returns:
        The report. A recording whose evidence the gate cannot read counts as a non-firing, both at
        the configured point and along the curve: a router that cannot read a recording does not
        route it.
    """
    gate = ruleset.gates[gate_name]
    positives = FAMILY_SETS[ruleset.reference_family_set[branch]]
    excluded = ruleset.held_out(branch)
    observations: list[tuple[float | None, bool]] = []
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for record in records:
        if record.family in excluded:
            continue
        value = gate_value(record, gate)
        positive = record.family in positives
        observations.append((value, positive))
        fired = evaluate_gate(record, gate) is GateOutcome.FIRED
        counts[("tp" if positive else "fp") if fired else ("fn" if positive else "tn")] += 1
    population = f"{ruleset.reference_family_set[branch]} as positives"
    if ruleset.excluded_by_construction.get(branch):
        population += f", {ruleset.excluded_by_construction[branch]} held out"
    return GateRecall(
        branch=branch,
        gate=gate_name,
        threshold=gate.threshold,
        configured=Confusion(**counts),
        curve=recall_at_budgets(
            observations,
            name=gate_name,
            reference=ruleset.reference_family_set[branch],
            polarity=_POLARITY[gate.op],
            budgets=budgets,
            population=population,
        ),
    )


def branch_recall_curves(
    records: Sequence[RecordingFeatures],
    ruleset: Ruleset,
    *,
    budgets: Sequence[float] = OVER_ROUTING_BUDGETS,
) -> list[GateRecall]:
    """Every routing gate's recall-at-budget report, in branch and declaration order.

    Args:
        records: The recordings to score over.
        ruleset: The loaded ruleset.
        budgets: The false-positive rates over the negatives that each point may not exceed.

    Returns:
        One :class:`GateRecall` per gate in ``branch_gates``. Flag gates route nothing and are not
        reported here.
    """
    return [
        gate_recall(records, ruleset, branch, name, budgets=budgets)
        for branch in BRANCHES
        if branch in ruleset.reference_family_set
        for name in ruleset.branch_gates.get(branch, ())
    ]


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
