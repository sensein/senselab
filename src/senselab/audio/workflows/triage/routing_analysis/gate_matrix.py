"""The family x gate matrix over the shipped ruleset, and the disagreements it leaves behind.

Every configured gate is re-evaluated on every recording of a finished features shard and reduced
two ways. :func:`gate_matrix` is one cell per (task family, gate): how many recordings the gate
fired on, stayed silent on and could not be read on, plus the distribution of the number it read.
:func:`qualify_disagreements` takes the branches a recording's declared family assigns it that
routing did not select, and the branches routing selected that the declaration does not assign, and
names the deciding gate and its reading for each.

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
