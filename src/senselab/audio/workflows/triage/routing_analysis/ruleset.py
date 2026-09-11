"""The family taxonomy ruleset: which branch a recording routes to, and what confirms the route.

A route is either *declared* — read off the ``task-`` entity through the family sets in
:mod:`~senselab.audio.workflows.triage.routing_analysis.families` — or *discovered*, meaning content
the recording carries whatever its instruction asked for. A discovered route is added to the
declared ones and never replaces them.

Every gate is one feature path, one comparison and one threshold, all of them read from
``taxonomy.ruleset`` in ``data/config/default.yaml``. The operating points, the firing spread each
came from, and the two open questions are in
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
from senselab.audio.workflows.triage.vocabulary import BRANCHES

RULESET_PATH = "taxonomy.ruleset"
"""Where the ruleset's data lives in the triage configuration."""

TRANSCRIPT_REPEAT = "transcript_repeat"
"""The feature source this module reads itself, because no extracted feature carries it."""

AT_LEAST = "at_least"
AT_MOST = "at_most"

FAMILY_SETS: Mapping[str, frozenset[str]] = {**DECLARED_KIND, "syllable_repetition": SYLLABLE_REPETITION}
"""Every named family set a branch's declared route may be taken from."""

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
        declared_family_set: Branch to the :data:`FAMILY_SETS` entry whose families declare it.
        confirming_gates: Branch to the gates, any one of which confirms its declared route.
        discovery_gates: Branch to the gate that adds it whatever the task declared.
    """

    gates: Mapping[str, Gate]
    declared_family_set: Mapping[str, str]
    confirming_gates: Mapping[str, tuple[str, ...]]
    discovery_gates: Mapping[str, str]

    def declared_branches(self, family: str) -> tuple[str, ...]:
        """Which branches a family's instruction declares.

        Args:
            family: A task family.

        Returns:
            The branches whose family set contains it, in :data:`~senselab.audio.workflows.triage.
            vocabulary.BRANCHES` order; empty when no set does.
        """
        return tuple(
            branch
            for branch in BRANCHES
            if branch in self.declared_family_set and family in FAMILY_SETS[self.declared_family_set[branch]]
        )


@dataclass(frozen=True)
class RouteEvaluation:
    """What the ruleset made of one recording.

    Attributes:
        stem: The recording's BIDS stem.
        family: The task family its stem collapses into.
        declared: The branches the task declared.
        confirmed: The declared branches a confirming gate fired for.
        unconfirmed: The declared branches whose gates were all readable and none fired.
        unavailable: The declared branches whose gates could not all be read and none fired.
        discovered: The branches a discovery gate added.
        routed: The union actually routed to, in branch order.
        fell_through: Whether ``routed`` is empty.
        gate_outcomes: Every evaluated gate's outcome, by gate name.
    """

    stem: str
    family: str
    declared: tuple[str, ...]
    confirmed: tuple[str, ...]
    unconfirmed: tuple[str, ...]
    unavailable: tuple[str, ...]
    discovered: tuple[str, ...]
    routed: tuple[str, ...]
    fell_through: bool
    gate_outcomes: Mapping[str, GateOutcome]


@dataclass(frozen=True)
class FamilyTally:
    """One family's counts over a stream of evaluations.

    Attributes:
        family: The task family.
        recordings: How many evaluations carried it.
        declared: Per branch, how many declared that branch.
        confirmed: Per branch, how many had it confirmed.
        discovered: Per branch, how many had it discovered.
        unavailable: Per branch, how many declared it with an unreadable gate.
        fell_through: How many routed to no branch at all.
    """

    family: str
    recordings: int
    declared: Mapping[str, int]
    confirmed: Mapping[str, int]
    discovered: Mapping[str, int]
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
    declared_family_set = dict(config.require(f"{RULESET_PATH}.declared_family_set"))
    confirming = {branch: tuple(names) for branch, names in config.require(f"{RULESET_PATH}.confirming_gates").items()}
    discovery = dict(config.require(f"{RULESET_PATH}.discovery_gates"))
    gates = {
        name: Gate(name=name, feature=tuple(row["feature"]), op=str(row["op"]), threshold=float(row["threshold"]))
        for name, row in config.require(f"{RULESET_PATH}.gates").items()
    }

    for branch, set_name in declared_family_set.items():
        if set_name not in FAMILY_SETS:
            raise ValueError(f"{RULESET_PATH}.declared_family_set[{branch}] names no family set: {set_name!r}")
    for branch, names in confirming.items():
        for name in names:
            if name not in gates:
                raise ValueError(f"{RULESET_PATH}.confirming_gates[{branch}] names no gate: {name!r}")
    for branch, name in discovery.items():
        if name not in gates:
            raise ValueError(f"{RULESET_PATH}.discovery_gates[{branch}] names no gate: {name!r}")
    for gate in gates.values():
        if gate.op not in (AT_LEAST, AT_MOST):
            raise ValueError(f"{RULESET_PATH}.gates[{gate.name}].op is not a comparison: {gate.op!r}")

    return Ruleset(
        gates=gates,
        declared_family_set=declared_family_set,
        confirming_gates=confirming,
        discovery_gates=discovery,
    )


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
    """Route one recording: what its task declared, what confirmed it, and what was found anyway.

    Args:
        features: The recording's extracted evidence.
        ruleset: The loaded ruleset.

    Returns:
        The evaluation.
    """
    outcomes: dict[str, GateOutcome] = {}
    declared = ruleset.declared_branches(features.family)

    confirmed: list[str] = []
    unconfirmed: list[str] = []
    unavailable: list[str] = []
    for branch in declared:
        states = [_outcome(features, ruleset, outcomes, name) for name in ruleset.confirming_gates.get(branch, ())]
        if GateOutcome.FIRED in states:
            confirmed.append(branch)
        elif GateOutcome.UNAVAILABLE in states or not states:
            unavailable.append(branch)
        else:
            unconfirmed.append(branch)

    discovered = [
        branch
        for branch in BRANCHES
        if branch in ruleset.discovery_gates
        and branch not in confirmed
        and _outcome(features, ruleset, outcomes, ruleset.discovery_gates[branch]) is GateOutcome.FIRED
    ]
    routed = tuple(branch for branch in BRANCHES if branch in confirmed or branch in discovered)

    return RouteEvaluation(
        stem=features.stem,
        family=features.family,
        declared=declared,
        confirmed=tuple(confirmed),
        unconfirmed=tuple(unconfirmed),
        unavailable=tuple(unavailable),
        discovered=tuple(discovered),
        routed=routed,
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
        rows = counters.setdefault(family, {axis: Counter() for axis in _AXES})
        recordings[family] += 1
        fell_through[family] += int(evaluation.fell_through)
        rows["declared"].update(evaluation.declared)
        rows["confirmed"].update(evaluation.confirmed)
        rows["discovered"].update(evaluation.discovered)
        rows["unavailable"].update(evaluation.unavailable)
    return {
        family: FamilyTally(
            family=family,
            recordings=recordings[family],
            declared=_per_branch(rows["declared"]),
            confirmed=_per_branch(rows["confirmed"]),
            discovered=_per_branch(rows["discovered"]),
            unavailable=_per_branch(rows["unavailable"]),
            fell_through=fell_through[family],
        )
        for family, rows in sorted(counters.items())
    }


_AXES: Sequence[str] = ("declared", "confirmed", "discovered", "unavailable")


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
