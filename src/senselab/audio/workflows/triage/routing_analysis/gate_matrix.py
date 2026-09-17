"""Where each task family routes, the family x gate matrix behind it, and the disagreements left.

Every configured gate is re-evaluated on every recording of a finished features shard and reduced
three ways. :func:`family_routing` is one cell per (task family, branch): how many recordings of
the family that branch routed, which gates fired to send them, which gate each routing hinged on
alone, and how far past its cut the routing was. :func:`gate_matrix` is one cell per (task family,
gate), the per-gate layer underneath: how many recordings the gate fired on, stayed silent on and
could not be read on, plus the distribution of the number it read. :func:`qualify_disagreements`
takes the branches a recording's declared family assigns it that routing did not select, and the
branches routing selected that the declaration does not assign, and names the deciding gate and its
reading for each.

**Routing is additive and the reference is multi-label.** A recording routes to every branch one of
whose gates fires, so several branches on one recording is intended rather than an error, and a
family declares the *set* of branches that legitimately apply to it — a diadochokinesis family
declares ``SPEECH``. A branch routed beyond that set is therefore reported as
:data:`BEYOND_DECLARATION` and never as a false positive of a precision.

``unavailable`` is a category of its own throughout. A gate whose evidence was never written did
not decline to fire, so :attr:`GateCell.fired_rate_evaluable` is None on a cell no recording could
evaluate and is never 0.0 there.

The declared task family is a reference standard and not ground truth, so every rate here is an
agreement rate with the declaration. ``specs/20260915-gate-family-matrix/design.md`` carries the
framing, the reporting band's status and the measurements.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from senselab.audio.workflows.triage.routing_analysis.features import RecordingFeatures, value_stats
from senselab.audio.workflows.triage.routing_analysis.ruleset import (
    AT_LEAST,
    ROUTE_STATES,
    Gate,
    GateOutcome,
    Ruleset,
    evaluate_routes,
    gate_outcome_of,
    gate_value,
)
from senselab.audio.workflows.triage.vocabulary import BRANCHES

PROFILE_DIR = Path(__file__).parent.parent / "data" / "disagreement_profile"
"""Where the bundled disagreement profiles live, one dated YAML per revision."""

PROFILE_VERSION = "1"
"""Schema version of the profile file. A reader refuses any other value."""

MISSED = "missed"
"""A branch the declared family assigns that routing did not select."""

EXTRA = "extra"
"""A branch routing selected that the declared family does not assign."""

DISAGREEMENT_KINDS: tuple[str, ...] = (MISSED, EXTRA)
"""Both directions of disagreement, in report order."""

DECLARED = "declared"
"""A (family, branch) cell the family's reference family sets name."""

BEYOND_DECLARATION = "beyond_declaration"
"""A (family, branch) cell the reference sets do not name. Not an error: routing is additive."""


class Deciding(Enum):
    """Which of three findings one disagreement is, read off its deciding gate.

    The three are different questions about different things and are never summed together.
    """

    NEAR_THRESHOLD = "near_threshold"
    FAR = "far"
    UNAVAILABLE = "unavailable"


DECIDING_CLASSES: tuple[str, ...] = tuple(member.value for member in Deciding)
"""Every finding's name, in declaration order, so a group can carry all three whether or not seen."""


def _bundled_profile_path() -> Path:
    """The newest bundled disagreement profile.

    Returns:
        The last dated profile in the bundled directory.

    Raises:
        FileNotFoundError: If the package ships no profile, which leaves the qualification
            unavailable rather than silently banded at a value nobody wrote down.
    """
    bundled = sorted(PROFILE_DIR.glob("*.yaml"))
    if not bundled:
        raise FileNotFoundError(f"no disagreement profile in {PROFILE_DIR}")
    return bundled[-1]


@lru_cache(maxsize=None)
def load_disagreement_profile(path: str | None = None) -> dict[str, Any]:
    """Load and validate a disagreement profile.

    Args:
        path: Profile path, or None for the newest bundled one.

    Returns:
        The validated profile, carrying ``near_threshold_band`` and its ``derivation``.

    Raises:
        FileNotFoundError: If ``path`` names a file that does not exist.
        ValueError: If the schema version is unknown, or if the band is not a positive number.
    """
    resolved = _bundled_profile_path() if path is None else Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"disagreement profile not found: {resolved}")
    profile: dict[str, Any] = yaml.safe_load(resolved.read_text())
    if str(profile.get("schema_version")) != PROFILE_VERSION:
        raise ValueError(f"{resolved}: schema_version is not {PROFILE_VERSION}: {profile.get('schema_version')!r}")
    band = profile.get("near_threshold_band")
    if not isinstance(band, (int, float)) or isinstance(band, bool) or band <= 0:
        raise ValueError(f"{resolved}: near_threshold_band is not a positive number: {band!r}")
    profile["near_threshold_band"] = float(band)
    profile["source"] = str(resolved)
    return profile


def gate_order(ruleset: Ruleset) -> tuple[str, ...]:
    """Every gate the ruleset declares, routing gates first, in branch and declaration order.

    Args:
        ruleset: The loaded ruleset.

    Returns:
        Each routing gate once, branch by branch in :data:`~senselab.audio.workflows.triage.
        vocabulary.BRANCHES` order, then each flag gate the same way, then any gate the ``gates``
        mapping defines that no branch names. A gate two branches both name appears at its first.
    """
    ordered: list[str] = []
    for assignment in (ruleset.branch_gates, ruleset.branch_flags):
        for branch in BRANCHES:
            for name in assignment.get(branch, ()):
                if name not in ordered:
                    ordered.append(name)
    ordered.extend(name for name in sorted(ruleset.gates) if name not in ordered)
    return tuple(ordered)


def routing_branch_of(ruleset: Ruleset, gate: str) -> str | None:
    """Which branch a gate routes, or None when it routes nothing.

    Args:
        ruleset: The loaded ruleset.
        gate: The gate's name.

    Returns:
        The first branch in :data:`~senselab.audio.workflows.triage.vocabulary.BRANCHES` order
        whose ``branch_gates`` names it, or None for a flag gate and for a gate no branch names.
    """
    for branch in BRANCHES:
        if gate in ruleset.branch_gates.get(branch, ()):
            return branch
    return None


def relative_margin(value: float, gate: Gate) -> float:
    """How far a reading sits past its gate's threshold, as a signed fraction of that threshold.

    Args:
        value: The number the gate read.
        gate: The gate.

    Returns:
        Positive when the gate fired and negative when it did not, however the comparison points,
        and scaled by the threshold so gates in seconds, dB, counts and probabilities are
        comparable. A gate whose threshold is 0 is scaled by 1 instead, which leaves the margin in
        the gate's own units rather than dividing by zero.
    """
    scale = abs(gate.threshold) or 1.0
    signed = value - gate.threshold if gate.op == AT_LEAST else gate.threshold - value
    return signed / scale


@dataclass(frozen=True)
class GateCell:
    """One gate's behaviour over one task family's recordings.

    Attributes:
        family: The task family.
        gate: The gate's name.
        n: How many recordings of the family were read at all.
        fired: How many the gate fired on.
        silent: How many it was readable on and did not fire on.
        unavailable: How many its evidence was never written on. Not a non-firing.
        values: :func:`~senselab.audio.workflows.triage.routing_analysis.features.value_stats`
            over the readings of the evaluable recordings, and empty when there were none.
    """

    family: str
    gate: str
    n: int
    fired: int
    silent: int
    unavailable: int
    values: Mapping[str, float]

    @property
    def n_evaluable(self) -> int:
        """How many recordings the gate could read at all."""
        return self.fired + self.silent

    @property
    def fired_rate(self) -> float | None:
        """Fraction of every recording of the family the gate fired on, or None when there are none.

        An unreadable recording is in this denominator, so this rate falls when evidence goes
        missing.
        """
        return self.fired / self.n if self.n else None

    @property
    def fired_rate_evaluable(self) -> float | None:
        """Fraction of the readable recordings the gate fired on, or None when none were readable.

        None means no recording of this family could evaluate this gate, which is not the gate
        never firing.
        """
        return self.fired / self.n_evaluable if self.n_evaluable else None

    @property
    def unavailable_rate(self) -> float | None:
        """Fraction of the family's recordings whose evidence for the gate was never written."""
        return self.unavailable / self.n if self.n else None

    @property
    def median(self) -> float | None:
        """The median reading over the readable recordings, or None when none were readable."""
        return self.values.get("median")

    @property
    def p90(self) -> float | None:
        """The 90th-percentile reading over the readable recordings, or None when none were."""
        return self.values.get("p90")

    def as_json(self) -> dict[str, Any]:
        """This cell as one flat row.

        Returns:
            The counts, the three rates and the reading distribution. A rate that is None stays
            None rather than becoming 0.0.
        """
        row: dict[str, Any] = {
            "family": self.family,
            "gate": self.gate,
            "n": self.n,
            "fired": self.fired,
            "silent": self.silent,
            "unavailable": self.unavailable,
            "n_evaluable": self.n_evaluable,
            "fired_rate": self.fired_rate,
            "fired_rate_evaluable": self.fired_rate_evaluable,
            "unavailable_rate": self.unavailable_rate,
        }
        row.update({f"value.{key}": number for key, number in self.values.items()})
        return row


@dataclass(frozen=True)
class GateMatrix:
    """Every (task family, gate) cell over one corpus, with the axes it was built on.

    Attributes:
        families: Every family seen, in name order. The matrix's rows.
        gates: Every gate the ruleset declares, in :func:`gate_order`. The matrix's columns.
        cells: The cell for each ``(family, gate)``. Every pair of the two axes is keyed, so a
            family that carried no readable recording for a gate is present and says so rather
            than being absent.
        recordings: How many recordings each family carried.
        n_recordings: How many recordings were reduced.
    """

    families: tuple[str, ...]
    gates: tuple[str, ...]
    cells: Mapping[tuple[str, str], GateCell]
    recordings: Mapping[str, int]
    n_recordings: int

    def cell(self, family: str, gate: str) -> GateCell:
        """One cell.

        Args:
            family: The family.
            gate: The gate.

        Returns:
            The cell.

        Raises:
            KeyError: When either name is off the matrix's axes.
        """
        return self.cells[(family, gate)]

    def rows(self) -> list[dict[str, Any]]:
        """Every cell as one flat row, in family then gate order.

        Returns:
            The rows.
        """
        return [self.cell(family, gate).as_json() for family in self.families for gate in self.gates]


def gate_matrix(records: Sequence[RecordingFeatures], ruleset: Ruleset) -> GateMatrix:
    """Reduce a corpus to one cell per task family and configured gate.

    Every gate is evaluated on every recording, whatever the task asked for, which is what the
    ruleset itself does. The reading is recomputed rather than read back, because the measurement
    that records a route keeps each gate's outcome and not the number behind it.

    Args:
        records: The recordings, from a features shard.
        ruleset: The loaded ruleset.

    Returns:
        The matrix. Its columns are every gate the ruleset declares, flag gates included, so a
        gate that routes nothing is still measured.
    """
    gates = gate_order(ruleset)
    families = tuple(sorted({record.family for record in records}))
    counts: dict[tuple[str, str], Counter[str]] = {(family, gate): Counter() for family in families for gate in gates}
    readings: dict[tuple[str, str], list[float]] = {(family, gate): [] for family in families for gate in gates}
    seen: Counter[str] = Counter()
    for record in records:
        seen[record.family] += 1
        for name in gates:
            gate = ruleset.gates[name]
            key = (record.family, name)
            value = gate_value(record, gate)
            counts[key][gate_outcome_of(value, gate).value] += 1
            if value is not None:
                readings[key].append(value)
    return GateMatrix(
        families=families,
        gates=gates,
        cells={
            key: GateCell(
                family=key[0],
                gate=key[1],
                n=seen[key[0]],
                fired=counts[key][GateOutcome.FIRED.value],
                silent=counts[key][GateOutcome.SILENT.value],
                unavailable=counts[key][GateOutcome.UNAVAILABLE.value],
                values=value_stats(readings[key]),
            )
            for key in counts
        },
        recordings={family: seen[family] for family in families},
        n_recordings=len(records),
    )


def unassigned_families(records: Sequence[RecordingFeatures], ruleset: Ruleset) -> dict[str, int]:
    """Which task families no reference family set assigns to any branch, and how many recordings.

    A family here is in no branch's denominator on either side: it cannot be missed and its
    routings cannot be extra, so it is reported on its own rather than folded into a negative.

    Args:
        records: The recordings.
        ruleset: The loaded ruleset, read for each branch's reference family set.

    Returns:
        Family to its recording count, in name order. Empty when every family seen is assigned.
    """
    counts: Counter[str] = Counter()
    for record in records:
        if not ruleset.reference_branches(record.family):
            counts[record.family] += 1
    return {family: counts[family] for family in sorted(counts)}


@dataclass(frozen=True)
class Disagreement:
    """One branch on one recording that routing and the declared family do not agree about.

    Neither side adjudicates: a declared family is a statement about what the participant was
    asked for, not about what the recording holds.

    Attributes:
        stem: The recording's BIDS stem.
        family: Its task family.
        branch: The branch disagreed about.
        kind: :data:`MISSED` or :data:`EXTRA`.
        deciding_gate: The gate the disagreement turns on: for a miss the branch's gate that came
            closest to firing, for an extra the firing gate that cleared its threshold by least.
            None only when every gate of the branch was unavailable.
        value: What the deciding gate read, or None when there was none to read.
        threshold: The deciding gate's configured cut, or None when there is no deciding gate.
        margin: The deciding gate's :func:`relative_margin`, negative on a miss and positive on an
            extra, or None when there is no deciding gate.
        finding: Which of the three :class:`Deciding` questions this disagreement raises.
        n_unavailable_gates: How many of the branch's gates could not be read at all. On an extra
            this is how much of the branch's evidence was missing while it was routed anyway, and
            it is reported beside the finding rather than folded into it.
        n_gates: How many gates the branch declares.
    """

    stem: str
    family: str
    branch: str
    kind: str
    deciding_gate: str | None
    value: float | None
    threshold: float | None
    margin: float | None
    finding: Deciding
    n_unavailable_gates: int
    n_gates: int

    def as_json(self) -> dict[str, Any]:
        """This disagreement as one flat row.

        Returns:
            Every field, with ``finding`` as its name.
        """
        return {
            "stem": self.stem,
            "family": self.family,
            "branch": self.branch,
            "kind": self.kind,
            "deciding_gate": self.deciding_gate,
            "value": self.value,
            "threshold": self.threshold,
            "margin": self.margin,
            "finding": self.finding.value,
            "n_unavailable_gates": self.n_unavailable_gates,
            "n_gates": self.n_gates,
        }


@dataclass(frozen=True)
class _Deciding:
    """What one branch's gates came to on one recording.

    Attributes:
        gate: The deciding gate's name, or None when every gate was unavailable.
        value: What it read, or None when there is no deciding gate.
        threshold: Its configured cut, or None when there is no deciding gate.
        margin: Its :func:`relative_margin`, or None when there is no deciding gate.
        finding: Which of the three questions the disagreement raises.
        unread: How many of the branch's gates could not be read at all.
    """

    gate: str | None
    value: float | None
    threshold: float | None
    margin: float | None
    finding: Deciding
    unread: int


def _deciding(record: RecordingFeatures, ruleset: Ruleset, branch: str, kind: str, band: float) -> _Deciding:
    """Which gate one disagreement turns on, and which of the three findings it raises.

    For a miss the deciding gate is the one that came closest to firing, so a miss reads as a
    threshold question only when nothing else about the branch was closer. For an extra it is the
    firing gate that cleared its cut by least, so an extra reads as a threshold question only when
    no other gate carried it further.

    Args:
        record: The recording's extracted evidence.
        ruleset: The loaded ruleset.
        branch: The branch disagreed about.
        kind: :data:`MISSED` or :data:`EXTRA`.
        band: The relative margin inside which a reading is a threshold question.

    Returns:
        The deciding gate and its finding.
    """
    readable: list[tuple[float, str, float]] = []
    unread = 0
    for name in ruleset.branch_gates.get(branch, ()):
        gate = ruleset.gates[name]
        value = gate_value(record, gate)
        if value is None:
            unread += 1
            continue
        readable.append((relative_margin(value, gate), name, value))
    if not readable:
        return _Deciding(None, None, None, None, Deciding.UNAVAILABLE, unread)
    fired = [entry for entry in readable if entry[0] >= 0.0]
    margin, name, value = (
        min(fired, key=lambda entry: entry[0]) if kind == EXTRA and fired else max(readable, key=lambda entry: entry[0])
    )
    return _Deciding(
        gate=name,
        value=value,
        threshold=ruleset.gates[name].threshold,
        margin=margin,
        finding=Deciding.NEAR_THRESHOLD if abs(margin) <= band else Deciding.FAR,
        unread=unread,
    )


def qualify_disagreements(
    records: Sequence[RecordingFeatures], ruleset: Ruleset, *, near_threshold_band: float
) -> list[Disagreement]:
    """Name the deciding gate and its reading for every branch routing and the declaration differ on.

    Args:
        records: The recordings.
        ruleset: The loaded ruleset.
        near_threshold_band: The relative margin inside which a deciding gate's reading is reported
            as a threshold question rather than an evidence question. A declared reporting
            convention from the disagreement profile, not a fitted cut.

    Returns:
        One entry per (recording, branch) pair the two sides disagree about, misses before extras
        within each recording and in shard order across them.
    """
    out: list[Disagreement] = []
    for record in records:
        evaluation = evaluate_routes(record, ruleset)
        for kind, branches in ((MISSED, evaluation.missed), (EXTRA, evaluation.extra)):
            for branch in branches:
                decided = _deciding(record, ruleset, branch, kind, near_threshold_band)
                out.append(
                    Disagreement(
                        stem=record.stem,
                        family=record.family,
                        branch=branch,
                        kind=kind,
                        deciding_gate=decided.gate,
                        value=decided.value,
                        threshold=decided.threshold,
                        margin=decided.margin,
                        finding=decided.finding,
                        n_unavailable_gates=decided.unread,
                        n_gates=len(ruleset.branch_gates.get(branch, ())),
                    )
                )
    return out


@dataclass(frozen=True)
class DisagreementGroup:
    """Every disagreement of one kind about one branch, grouped by the task family it came from.

    Attributes:
        family: The task family.
        branch: The branch.
        kind: :data:`MISSED` or :data:`EXTRA`.
        n: How many recordings of the family disagree this way.
        findings: How many raise each :class:`Deciding` question, keyed by its name. All three keys
            are present whether or not the group carried one, and they sum to ``n``.
        gates: How many turn on each deciding gate, keyed by gate name and in descending count.
        margins: :func:`~senselab.audio.workflows.triage.routing_analysis.features.value_stats`
            over the deciding gates' relative margins, and empty when no gate was readable.
        n_with_unavailable_gate: How many carried at least one unreadable gate for the branch,
            whatever their finding. On an extra this counts routings made on partial evidence.
    """

    family: str
    branch: str
    kind: str
    n: int
    findings: Mapping[str, int]
    gates: Mapping[str, int]
    margins: Mapping[str, float]
    n_with_unavailable_gate: int

    def gates_summary(self) -> str:
        """Which gates this group's disagreements turn on, and how many each.

        Returns:
            ``<gate>:<count>`` per deciding gate, most common first, or ``-`` when no gate of the
            branch was readable on any of them.
        """
        return ", ".join(f"{gate}:{count}" for gate, count in self.gates.items()) or "-"

    def as_json(self) -> dict[str, Any]:
        """This group as one flat row.

        Returns:
            The counts, one column per finding, the deciding gates joined into one field, and the
            margin distribution.
        """
        row: dict[str, Any] = {
            "family": self.family,
            "branch": self.branch,
            "kind": self.kind,
            "n": self.n,
            "n_with_unavailable_gate": self.n_with_unavailable_gate,
            "deciding_gates": self.gates_summary(),
        }
        row.update({f"finding.{name}": self.findings[name] for name in DECIDING_CLASSES})
        row.update({f"margin.{key}": number for key, number in self.margins.items()})
        return row


def group_disagreements(disagreements: Iterable[Disagreement]) -> list[DisagreementGroup]:
    """Group disagreements by task family, branch and direction.

    Args:
        disagreements: What :func:`qualify_disagreements` returned, in any order.

    Returns:
        One group per ``(family, branch, kind)`` seen, misses before extras, then by descending
        count and by family name.
    """
    findings: dict[tuple[str, str, str], Counter[str]] = {}
    gates: dict[tuple[str, str, str], Counter[str]] = {}
    margins: dict[tuple[str, str, str], list[float]] = {}
    unread: Counter[tuple[str, str, str]] = Counter()
    totals: Counter[tuple[str, str, str]] = Counter()
    for entry in disagreements:
        key = (entry.family, entry.branch, entry.kind)
        totals[key] += 1
        findings.setdefault(key, Counter())[entry.finding.value] += 1
        if entry.deciding_gate is not None:
            gates.setdefault(key, Counter())[entry.deciding_gate] += 1
        if entry.margin is not None:
            margins.setdefault(key, []).append(entry.margin)
        if entry.n_unavailable_gates:
            unread[key] += 1
    ordered = sorted(totals, key=lambda key: (DISAGREEMENT_KINDS.index(key[2]), -totals[key], key[0], key[1]))
    return [
        DisagreementGroup(
            family=family,
            branch=branch,
            kind=kind,
            n=totals[(family, branch, kind)],
            findings={name: findings[(family, branch, kind)].get(name, 0) for name in DECIDING_CLASSES},
            gates=dict(gates.get((family, branch, kind), Counter()).most_common()),
            margins=value_stats(sorted(margins.get((family, branch, kind), []))),
            n_with_unavailable_gate=unread[(family, branch, kind)],
        )
        for family, branch, kind in ordered
    ]


@dataclass(frozen=True)
class FamilyRouting:
    """Where one task family's recordings routed on one branch, and which gate sent them there.

    One row of the answer to "for each task family, how many are routed where and what the
    decision criteria are". A branch routes when *any* of its gates fires, so ``fired_gates`` sums
    to at least ``routed`` and ``sole_gates`` to at most it: a routing several gates agreed on is
    counted once in ``routed``, once per gate in ``fired_gates``, and in no entry of ``sole_gates``.

    Attributes:
        family: The task family.
        branch: The branch.
        n: How many recordings of the family were read.
        declared: Whether the family's reference family sets name this branch. The reference is
            multi-label, so a family may declare several branches: a diadochokinesis family
            declares ``SPEECH``.
        routed: How many recordings of the family this branch routed on content.
        fired_gates: Gate to how many of those routings it fired on, in descending count. A routing
            two gates both fired on appears under each.
        sole_gates: Gate to how many routings it was the only firing gate on, in descending count.
            A gate's entry here is the routing that disappears if the gate is removed.
        not_routed: How many recordings of the family this branch did not route.
        not_routed_all_unavailable: Of those, how many had *every* gate of the branch unreadable,
            which is a missing-evidence non-routing and not a silent one.
        not_routed_some_unavailable: Of those, how many had at least one gate unreadable.
        routed_with_unavailable: Of the routings, how many were made while at least one gate of the
            branch could not be read, so the branch was entered on partial evidence.
        margins: :func:`~senselab.audio.workflows.triage.routing_analysis.features.value_stats`
            over the relative margin of the least-clearing firing gate on each routing, and empty
            when the branch routed nothing. How far past its cut the routing actually was.
    """

    family: str
    branch: str
    n: int
    declared: bool
    routed: int
    fired_gates: Mapping[str, int]
    sole_gates: Mapping[str, int]
    not_routed: int
    not_routed_all_unavailable: int
    not_routed_some_unavailable: int
    routed_with_unavailable: int
    margins: Mapping[str, float]

    @property
    def routed_rate(self) -> float | None:
        """Fraction of the family's recordings this branch routed, or None when there are none."""
        return self.routed / self.n if self.n else None

    @property
    def agreement(self) -> str:
        """How this cell stands against the declaration, as one word.

        Returns:
            ``declared`` when the family names the branch, ``beyond_declaration`` when it does not.
            Neither is a verdict: the declaration says what the participant was asked for, not what
            the recording holds, and additive routing is intended.
        """
        return DECLARED if self.declared else BEYOND_DECLARATION

    def gates_summary(self) -> str:
        """Which gates routed this cell, and how many each.

        Returns:
            ``<gate>:<count>`` per firing gate, most common first, or ``-`` when the branch routed
            nothing here.
        """
        return ", ".join(f"{gate}:{count}" for gate, count in self.fired_gates.items()) or "-"

    def sole_summary(self) -> str:
        """Which gates were the only one firing, and how many each.

        Returns:
            ``<gate>:<count>`` per gate, most common first, or ``-`` when no routing here hinged on
            a single gate.
        """
        return ", ".join(f"{gate}:{count}" for gate, count in self.sole_gates.items()) or "-"

    def as_json(self) -> dict[str, Any]:
        """This cell as one flat row.

        Returns:
            The counts, the routed rate, how the cell stands against the declaration, the firing
            and sole-firing gates joined into one field each, and the margin distribution.
        """
        row: dict[str, Any] = {
            "family": self.family,
            "branch": self.branch,
            "n": self.n,
            "declared": self.declared,
            "agreement": self.agreement,
            "routed": self.routed,
            "routed_rate": self.routed_rate,
            "not_routed": self.not_routed,
            "not_routed_all_unavailable": self.not_routed_all_unavailable,
            "not_routed_some_unavailable": self.not_routed_some_unavailable,
            "routed_with_unavailable": self.routed_with_unavailable,
            "firing_gates": self.gates_summary(),
            "sole_firing_gates": self.sole_summary(),
        }
        row.update({f"margin.{key}": number for key, number in self.margins.items()})
        return row


@dataclass(frozen=True)
class FamilyStates:
    """What became of one task family's recordings overall, across every branch at once.

    Attributes:
        family: The task family.
        n: How many recordings of the family were read.
        declared: The branches the family's reference family sets name, in branch order.
        states: How many recordings landed in each
            :class:`~senselab.audio.workflows.triage.routing_analysis.ruleset.RouteState`, keyed by
            its value. Every state is present whether or not the family carried one, and they sum
            to ``n``.
        branch_counts: How many recordings routed to each number of branches, keyed by the count as
            a string. ``"0"`` is every recording no gate fired on, and an entry above ``"1"`` is
            additive routing, which is intended.
    """

    family: str
    n: int
    declared: tuple[str, ...]
    states: Mapping[str, int]
    branch_counts: Mapping[str, int]

    def as_json(self) -> dict[str, Any]:
        """This family as one flat row.

        Returns:
            The counts, the declared branches joined by ``+``, one column per route state and one
            per branch-count bucket.
        """
        row: dict[str, Any] = {
            "family": self.family,
            "n": self.n,
            "declared": "+".join(self.declared) or "-",
            "n_declared": len(self.declared),
        }
        row.update({f"state.{name}": count for name, count in self.states.items()})
        row.update({f"branches_routed.{key}": count for key, count in self.branch_counts.items()})
        return row


def _routing_gates(record: RecordingFeatures, ruleset: Ruleset, branch: str) -> tuple[list[tuple[float, str]], int]:
    """Which of one branch's gates fired on one recording, with their margins, and how many were unread.

    Args:
        record: The recording's extracted evidence.
        ruleset: The loaded ruleset.
        branch: The branch.

    Returns:
        The ``(relative margin, gate name)`` of every gate that fired, and how many gates of the
        branch could not be read at all. An unread gate is in neither the firing list nor a silence.
    """
    fired: list[tuple[float, str]] = []
    unread = 0
    for name in ruleset.branch_gates.get(branch, ()):
        gate = ruleset.gates[name]
        value = gate_value(record, gate)
        if value is None:
            unread += 1
        elif gate_outcome_of(value, gate) is GateOutcome.FIRED:
            fired.append((relative_margin(value, gate), name))
    return fired, unread


def family_routing(records: Sequence[RecordingFeatures], ruleset: Ruleset) -> list[FamilyRouting]:
    """How many recordings of each task family routed to each branch, and which gate sent them.

    Every (family, branch) pair is returned, so a branch a family never routed to is present and
    reads zero rather than being absent. Routing is additive by design: a recording routes to every
    branch one of whose gates fires, and several branches on one recording is not an error.

    Args:
        records: The recordings, from a features shard.
        ruleset: The loaded ruleset.

    Returns:
        One cell per ``(family, branch)``, in family name then
        :data:`~senselab.audio.workflows.triage.vocabulary.BRANCHES` order.
    """
    families = sorted({record.family for record in records})
    seen: Counter[str] = Counter()
    routed: Counter[tuple[str, str]] = Counter()
    fired_gates: dict[tuple[str, str], Counter[str]] = {}
    sole_gates: dict[tuple[str, str], Counter[str]] = {}
    margins: dict[tuple[str, str], list[float]] = {}
    routed_partial: Counter[tuple[str, str]] = Counter()
    missing_all: Counter[tuple[str, str]] = Counter()
    missing_some: Counter[tuple[str, str]] = Counter()
    for record in records:
        seen[record.family] += 1
        for branch in BRANCHES:
            key = (record.family, branch)
            fired, unread = _routing_gates(record, ruleset, branch)
            if fired:
                routed[key] += 1
                fired_gates.setdefault(key, Counter()).update(name for _, name in fired)
                if len(fired) == 1:
                    sole_gates.setdefault(key, Counter())[fired[0][1]] += 1
                margins.setdefault(key, []).append(min(margin for margin, _ in fired))
                if unread:
                    routed_partial[key] += 1
                continue
            if unread:
                missing_some[key] += 1
                if unread == len(ruleset.branch_gates.get(branch, ())):
                    missing_all[key] += 1
    return [
        FamilyRouting(
            family=family,
            branch=branch,
            n=seen[family],
            declared=branch in ruleset.reference_branches(family),
            routed=routed[(family, branch)],
            fired_gates=dict(fired_gates.get((family, branch), Counter()).most_common()),
            sole_gates=dict(sole_gates.get((family, branch), Counter()).most_common()),
            not_routed=seen[family] - routed[(family, branch)],
            not_routed_all_unavailable=missing_all[(family, branch)],
            not_routed_some_unavailable=missing_some[(family, branch)],
            routed_with_unavailable=routed_partial[(family, branch)],
            margins=value_stats(sorted(margins.get((family, branch), []))),
        )
        for family in families
        for branch in BRANCHES
    ]


def family_states(records: Sequence[RecordingFeatures], ruleset: Ruleset) -> list[FamilyStates]:
    """What became of each task family's recordings overall: the route state and how many branches.

    Args:
        records: The recordings.
        ruleset: The loaded ruleset.

    Returns:
        One entry per family seen, in name order.
    """
    seen: Counter[str] = Counter()
    states: dict[str, Counter[str]] = {}
    widths: dict[str, Counter[str]] = {}
    for record in records:
        evaluation = evaluate_routes(record, ruleset)
        seen[record.family] += 1
        states.setdefault(record.family, Counter())[evaluation.state.value] += 1
        widths.setdefault(record.family, Counter())[str(len(evaluation.routed))] += 1
    return [
        FamilyStates(
            family=family,
            n=seen[family],
            declared=ruleset.reference_branches(family),
            states={name: states[family].get(name, 0) for name in ROUTE_STATES},
            branch_counts={str(width): widths[family].get(str(width), 0) for width in range(len(BRANCHES) + 1)},
        )
        for family in sorted(seen)
    ]
