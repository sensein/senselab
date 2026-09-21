"""The VERDICT node: the store's contents read into the vocabulary's fold, and the result recorded.

This is the graph's only decision about the recording. The branches and QUALITY write
``branch_report`` entities — typed deviations, no outcome and no conformance — and propose spans;
the deciding nodes write ``verdict`` entities; this node reads all three, adds the declared task and
the routing decisions, and hands them to ``vocabulary.fold_file_verdict``. REDACT's optional LLM
re-read is read here too, as an annotation: its ``flagged`` reaches the triage axis under
``verdict.llm_redaction_flags`` and the release axis on no path.

**The declared task's conformance is decided here.** :func:`gate_conformance` reads the declared
family's task group's gates from ``verdict.gates``, reads their inputs off the branch's own
``measure`` findings, and substitutes the answer onto the report of the branch that owns the family
and reported ``in_family``. An absent reading and an unmeasured bound both answer
:data:`UNDETERMINED`; every gate applied is recorded on the verdict.

The two axes this node keeps apart — triage and release — and the tables it implements are in
``specs/20260817-triage-workflow-dag/verdict.md``; the gates are in
``specs/20260921-gates-in-verdict/`` and the re-read in
``specs/20260817-triage-workflow-dag/llm-check.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.live_evidence import declared_task, recording_stem
from senselab.audio.workflows.triage.nodes.branches import BRANCH_FAMILY, EXPECTATIONS, Expectation
from senselab.audio.workflows.triage.nodes.common import (
    NodeResult,
    find_measurement,
    software_agent,
    write_verdict,
)
from senselab.audio.workflows.triage.nodes.gates import (
    DEFAULT_LAYER,
    GATE_SECTION,
    GATE_SPECS,
    AppliedGate,
    GateBounds,
    apply_gates,
    conformance_gate_names,
    load_gate_bounds,
)
from senselab.audio.workflows.triage.vocabulary import (
    GRAPH_ORDER,
    REDACTION_LLM_ANNOTATION,
    RULESET_ROUTING,
    TASK,
    UNDETERMINED,
    BranchDecision,
    BranchReport,
    Conformance,
    FileVerdict,
    FoldPolicy,
    NodeVerdict,
    Outcome,
    RunState,
    fold_file_verdict,
)
from senselab.utils.prov_store import PROV_TYPE, Entity, ProvStore

NODE = "VERDICT"

_GRAPH_ORDER = GRAPH_ORDER[:-1]


@dataclass(frozen=True)
class VerdictResult(NodeResult):
    """What VERDICT returns.

    Attributes:
        file_verdict: The graph's conclusion about the recording, on both axes.
    """

    file_verdict: FileVerdict


def _node_verdict_from_entity(entity: Entity) -> NodeVerdict:
    """The vocabulary verdict a ``write_verdict`` entity carries.

    Args:
        entity: A ``verdict`` entity.

    Returns:
        Its vocabulary verdict. An ``outcome`` outside :class:`Outcome` is returned as a ``flag``
        naming the unreadable value rather than folded as a conclusion.
    """
    attributes = entity.attributes
    node = str(attributes["node"])
    raw = attributes["outcome"]
    try:
        outcome = Outcome(raw)
    except ValueError:
        return NodeVerdict(
            node=node,
            outcome=Outcome.FLAG,
            kind=None,
            why=f"{node} wrote outcome {raw!r}, which is not a node outcome; its verdict was not folded",
        )
    return NodeVerdict(node=node, outcome=outcome, kind=attributes.get("kind"), why=attributes["why"])


def _conformance_of_entity(entity: Entity) -> Conformance:
    """The conformance a ``branch_report`` entity carries.

    Args:
        entity: A ``branch_report`` entity.

    Returns:
        True, False, or :data:`UNDETERMINED`. A value that is neither a bool nor the
        :data:`UNDETERMINED` token reads as :data:`UNDETERMINED`.
    """
    raw = entity.attributes.get("conformance")
    return raw if isinstance(raw, bool) else UNDETERMINED


def _branch_report_from_entity(entity: Entity) -> BranchReport:
    """The vocabulary report a ``write_report`` entity carries.

    Args:
        entity: A ``branch_report`` entity.

    Returns:
        Its vocabulary report. ``conformance_of`` is read as written, an unknown referent included.
    """
    attributes = entity.attributes
    return BranchReport(
        node=str(attributes["node"]),
        kind=attributes.get("kind"),
        conformance=_conformance_of_entity(entity),
        conformance_of=str(attributes.get("conformance_of")),
        deviations=tuple(str(name) for name in attributes.get("deviations") or ()),
        unmeasured=tuple(str(name) for name in attributes.get("unmeasured") or ()),
        in_family=bool(attributes.get("in_family")),
    )


def _live_latest(store: ProvStore, prov_type: PROV_TYPE, key: Callable[[Entity], str]) -> list[Entity]:
    """Entities of one type under the store's shared rule, one per key.

    An invalidated entity is never read, and of the survivors sharing a key the latest write wins.

    Args:
        store: The provenance store.
        prov_type: The entity type to read.
        key: What makes two entities the same assertion — a node name, a kind name, a branch name.

    Returns:
        One entity per key, the latest live one, in order of each key's first appearance.
    """
    latest: dict[str, Entity] = {}
    for entity in store.entities(prov_type):
        if store.is_invalidated(entity.id):
            continue
        latest[key(entity)] = entity
    return list(latest.values())


def _node_verdicts_in_graph_order(store: ProvStore) -> list[tuple[Entity, NodeVerdict]]:
    """Node verdict entities, ordered by the graph, with nodes outside it last.

    Args:
        store: The provenance store.

    Returns:
        One ``(entity, verdict)`` pair per node, the node's latest live verdict. The file verdict
        itself is excluded by its ``node`` attribute.
    """
    pairs = [
        (entity, _node_verdict_from_entity(entity))
        for entity in _live_latest(store, "verdict", lambda e: str(e.attributes.get("node")))
        if entity.attributes.get("node") != NODE
    ]
    return sorted(
        pairs,
        key=lambda pair: _GRAPH_ORDER.index(pair[1].node) if pair[1].node in _GRAPH_ORDER else len(_GRAPH_ORDER),
    )


def _branch_reports(store: ProvStore) -> list[tuple[Entity, BranchReport]]:
    """The reporting nodes' reports, one per node, under the store's shared rule.

    Args:
        store: The provenance store.

    Returns:
        One ``(entity, report)`` pair per node that reported, in order of first appearance.
    """
    return [
        (entity, _branch_report_from_entity(entity))
        for entity in _live_latest(store, "branch_report", lambda e: str(e.attributes.get("node")))
    ]


def _spans_by_node(store: ProvStore) -> dict[str, int]:
    """How many spans each reporting node proposed into its own family.

    A span is attributed to a node by the activity that generated it, and counts only in that
    node's own family.

    Args:
        store: The provenance store.

    Returns:
        Node name to live span count. A node that proposed none is absent.
    """
    counts: dict[str, int] = {}
    for span in store.entities("span"):
        if store.is_invalidated(span.id):
            continue
        activity_id = store.generated_by(span.id)
        if activity_id is None:
            continue
        node = store.get_activity(activity_id).node
        if span.attributes.get("family") != BRANCH_FAMILY.get(node):
            continue
        counts[node] = counts.get(node, 0) + 1
    return counts


def _route_state(store: ProvStore) -> tuple[str | None, list[str]]:
    """What the ruleset made of the whole recording, as ROUTING recorded it.

    Args:
        store: The provenance store.

    Returns:
        The file-level route state and the id it came from, or ``(None, [])`` when ROUTING wrote no
        evaluation.
    """
    measurement = find_measurement(store, RULESET_ROUTING)
    if measurement is None or measurement.attributes.get("state") is None:
        return None, []
    return str(measurement.attributes["state"]), [measurement.id]


def _critical_absences(store: ProvStore) -> dict[str, dict[str, str]]:
    """Which branches the ruleset could form no opinion about at all, and why, as ROUTING recorded it.

    Read off the same ``ruleset_routing`` measurement :func:`_route_state` reads.

    Args:
        store: The provenance store.

    Returns:
        Per branch not one of whose gates could be read, each gate and the absence the node that
        failed to write its evidence recorded. Empty on every non-critical path.
    """
    measurement = find_measurement(store, RULESET_ROUTING)
    if measurement is None:
        return {}
    unavailable = measurement.attributes.get("unavailable") or {}
    return {
        str(branch): {str(gate): str(why) for gate, why in (unavailable.get(str(branch)) or {}).items()}
        for branch in measurement.attributes.get("unreadable") or ()
    }


def _llm_redaction(store: ProvStore) -> tuple[dict[str, object] | None, list[str]]:
    """REDACT's LLM re-read annotation, as the detector that wrote it recorded it.

    Args:
        store: The provenance store.

    Returns:
        The annotation and the id it came from, or ``(None, [])`` when REDACT wrote none. ``name``
        and ``signal`` are dropped; every other attribute is carried through unread.
    """
    measurement = find_measurement(store, REDACTION_LLM_ANNOTATION)
    if measurement is None:
        return None, []
    return {key: value for key, value in measurement.attributes.items() if key not in ("name", "signal")}, [
        measurement.id
    ]


def _branch_decisions(store: ProvStore) -> tuple[dict[str, BranchDecision], list[str]]:
    """ROUTING's decision per branch.

    Args:
        store: The provenance store.

    Returns:
        The decision per branch name, one per branch under the store's shared rule, and the ids of
        the entities they came from. Empty when ROUTING never ran.
    """
    decisions: dict[str, BranchDecision] = {}
    ids: list[str] = []
    for entity in _live_latest(store, "branch_decision", lambda e: str(e.attributes["branch"])):
        branch = str(entity.attributes["branch"])
        decisions[branch] = BranchDecision(
            branch=branch,
            will_run=bool(entity.attributes["will_run"]),
            route_state=str(entity.attributes["route_state"]),
            forced_by_declaration=bool(entity.attributes["forced_by_declaration"]),
            declared=bool(entity.attributes["declared"]),
            hint_tags=tuple(str(tag) for tag in entity.attributes.get("hint_tags") or ()),
            bad_map_values={
                str(tag): str(value) for tag, value in (entity.attributes.get("bad_map_values") or {}).items()
            },
            withheld_critical=bool(entity.attributes.get("withheld_critical")),
        )
        ids.append(entity.id)
    return decisions, ids


def _hint_claims(
    decisions: Mapping[str, BranchDecision], hint: AudioHints | None, *, declared_family: str
) -> dict[str, bool] | None:
    """Which branches the recording's declaration claimed, read off ROUTING's own record of reading it.

    The declaration is read back from each branch's decision, never re-resolved here.

    Args:
        decisions: ROUTING's decisions, as read from the store.
        hint: What the recording was declared to contain, if anything.
        declared_family: The task family the recording's own stem declares, empty when it declares
            none. Tested for existence only.

    Returns:
        True per claimed branch; a branch the declaration did not name is simply absent. None when a
        declaration existed and no decision survived to say what ROUTING made of it.
    """
    if (hint is not None or declared_family) and not decisions:
        return None
    return {decision.branch: True for decision in decisions.values() if decision.declared}


def declared_expectation(declared_family: str | None) -> tuple[str, Expectation] | None:
    """The branch that owns the declared task, and the row it holds for it.

    Args:
        declared_family: The task family the recording declares, or None.

    Returns:
        ``(branch, expectation)``, or None when nothing declares a family this graph has a row for.
    """
    if declared_family is None:
        return None
    for branch, table in EXPECTATIONS.items():
        if declared_family in table:
            return branch, table[declared_family]
    return None


def gate_readings(store: ProvStore, names: Sequence[str]) -> dict[str, Any]:
    """What the reporting node read for each gate, off the measurements it wrote.

    Args:
        store: The provenance store.
        names: The gates whose readings to look for.

    Returns:
        Reading name to its value. A reading nothing live carries, and one written as null, is
        absent — the two are the same thing to a gate, which is that nobody measured it.
    """
    readings: dict[str, Any] = {}
    for name in names:
        reading = GATE_SPECS[name].reading
        if reading is None:
            continue
        measurement = find_measurement(store, reading)
        if measurement is None:
            continue
        value = measurement.attributes.get("value")
        if value is not None:
            readings[reading] = value
    return readings


def _gate_path(gate: AppliedGate) -> str:
    """Where a gate's bound was configured, as a dotted config path.

    Args:
        gate: One applied gate.

    Returns:
        The path, naming the layer that supplied the bound and the key it was keyed under.
    """
    if gate.layer == DEFAULT_LAYER:
        return f"{GATE_SECTION}.{gate.layer}.{gate.name}"
    return f"{GATE_SECTION}.{gate.layer}.{gate.keyed_under}.{gate.name}"


@dataclass(frozen=True)
class GateOutcome:
    """What VERDICT's gates made of the declared task.

    Attributes:
        node: The branch whose report the conformance belongs to, or None when nothing was gated.
        conformance: What the gates decided.
        bounds: The group's configured gates, or None when no group was resolved.
        applied: One record per gate applied, in the order the group declares them.
    """

    node: str | None
    conformance: Conformance
    bounds: GateBounds | None
    applied: tuple[AppliedGate, ...]

    @property
    def unmeasured(self) -> tuple[str, ...]:
        """The gates this fold wanted and nobody has measured, by their full config paths.

        Returns:
            One path per applied gate whose bound is null, named by the layer that supplied it, in
            the order they were applied. They join the reporting node's own unmeasured asks, so
            ``verdict.unmeasured_points_flag`` reaches a gate the same way it reaches an
            instrument setting.
        """
        return tuple(_gate_path(gate) for gate in self.applied if gate.bound is None)

    def record(self) -> dict[str, Any]:
        """This application, as the verdict records it.

        Returns:
            The node, the group and its bounds, and every gate applied. Empty when no group was
            resolved, which is every recording that declares no task this graph holds a row for.
        """
        if self.bounds is None:
            return {}
        return {
            "node": self.node,
            **self.bounds.record(),
            "applied": [gate.record() for gate in self.applied],
        }


def gate_conformance(
    store: ProvStore, config: TriageConfig, reports: Sequence[BranchReport], declared_family: str | None
) -> GateOutcome:
    """Decide the declared task's conformance from the readings the branch reported.

    The gates are the declared family's own task group's, and only the branch that owns that
    family and evaluated it in family is gated: every other report evaluated no task.

    Args:
        store: The provenance store, read for the gates' readings.
        config: The resolved triage configuration, read for ``verdict.gates``.
        reports: Every reporting node's report.
        declared_family: The task family the recording declares, or None.

    Returns:
        The outcome. Its conformance is :data:`UNDETERMINED` whenever no group was resolved, the
        owning branch left no in-family report, or any applied gate could not be answered.
    """
    owner = declared_expectation(declared_family)
    if owner is None:
        return GateOutcome(None, UNDETERMINED, None, ())
    branch, expectation = owner
    bounds = load_gate_bounds(config, expectation.pattern, declared_family)
    reported = next((report for report in reports if report.node == branch and report.in_family), None)
    if reported is None:
        return GateOutcome(branch, UNDETERMINED, bounds, ())
    names = conformance_gate_names(expectation.pattern, anti_pattern=expectation.anti_pattern)
    conformance, applied = apply_gates(names, bounds, gate_readings(store, names))
    return GateOutcome(branch, conformance, bounds, tuple(applied))


def _derived_ran(
    store: ProvStore, verdicts: Sequence[NodeVerdict], reports: Sequence[BranchReport]
) -> dict[str, RunState]:
    """Whether each graph node ran, as far as the store can say.

    A node that reported counts as having concluded exactly as one that decided does.

    Args:
        store: The provenance store, read for which nodes have an activity.
        verdicts: Every node verdict read from the store.
        reports: Every branch report read from the store.

    Returns:
        ``COMPLETED`` for a node carrying a verdict or a report, ``ERRORED`` for one carrying an
        activity but neither, and ``SKIPPED`` for one carrying none of the three.
    """
    concluded = {v.node for v in verdicts} | {r.node for r in reports}
    attempted = {activity.node for activity in store.activities()}
    return {
        node: RunState.COMPLETED if node in concluded else RunState.ERRORED if node in attempted else RunState.SKIPPED
        for node in _GRAPH_ORDER
    }


def verdict(
    store: ProvStore,
    source: None,
    config: TriageConfig,
    hint: AudioHints | None = None,
    *,
    run_dir: Path,
    ran: Mapping[str, RunState] | None = None,
) -> VerdictResult:
    """Decide the file, from the reports, the verdicts, the spans, the routes and the declared task.

    Args:
        store: The provenance store, holding every node's ``verdict`` entity, ROUTING's
            ``branch_decision`` entities, its ``ruleset_routing`` measurement, and ADMIT's
            ``recording`` stream. Nothing else is read.
        source: Accepted for the shared node shape; not read.
        config: The triage configuration, named in the activity by its hash and read for the
            ``verdict.*`` section, which holds every threshold that turns a reading into a judgement.
        hint: What the recording was declared to contain. Read for branch mismatch only: a hint never
            resolves a finding and never turns a flag into a pass.
        run_dir: Accepted for the shared node shape; VERDICT writes no sidecars.
        ran: Whether each node ran, from the runner, merged over :func:`_derived_ran` so that a
            partial mapping overrides per node without erasing the rest.

    Returns:
        The file verdict on both axes, the verdict entity it was written to, and a view leading with
        that entity followed by every id the fold consumed.
    """
    pairs = _node_verdicts_in_graph_order(store)
    node_verdicts = [node_verdict for _, node_verdict in pairs]
    report_pairs = _branch_reports(store)
    declared_family = declared_task(recording_stem(store))[1]
    reported = [report for _, report in report_pairs]
    outcome = gate_conformance(store, config, reported, declared_family or None)
    reports = [
        replace(
            report,
            conformance=outcome.conformance,
            unmeasured=tuple(dict.fromkeys((*report.unmeasured, *outcome.unmeasured))),
        )
        if report.node == outcome.node and report.in_family and report.conformance_of == TASK
        else report
        for report in reported
    ]
    route_state, route_ids = _route_state(store)
    decisions, decision_ids = _branch_decisions(store)
    annotation, annotation_ids = _llm_redaction(store)
    resolved_ran = {**_derived_ran(store, node_verdicts, reports), **(ran or {})}
    file_verdict = fold_file_verdict(
        node_verdicts,
        branch_reports=reports,
        spans_by_node=_spans_by_node(store),
        branch_decisions=decisions,
        ran=resolved_ran,
        hint_claims=_hint_claims(decisions, hint, declared_family=declared_family),
        route_state=route_state,
        declared_family=declared_family or None,
        llm_redaction=annotation,
        critical_absences=_critical_absences(store),
        gates=outcome.record(),
        policy=FoldPolicy.from_config(config),
    )

    software = software_agent(store)
    activity = store.activity(node=NODE, step=None, parameters={"config_hash": config.config_hash})
    store.was_associated_with(activity, software)
    folded_ids = (
        [entity.id for entity, _ in pairs]
        + [entity.id for entity, _ in report_pairs]
        + route_ids
        + decision_ids
        + annotation_ids
    )
    for folded_id in folded_ids:
        store.used(activity, folded_id)

    verdict_id, node_verdict = write_verdict(
        store,
        activity,
        software,
        node=NODE,
        outcome=file_verdict.triage,
        kind=None,
        why=(
            f"folded {len(node_verdicts)} node verdict(s) and {len(reports)} branch report(s) over "
            f"{len(file_verdict.routes)} routed branches"
        ),
        detail=file_verdict.record(),
    )
    return VerdictResult(
        verdict=node_verdict,
        view=(verdict_id, *folded_ids),
        verdict_entity_id=verdict_id,
        file_verdict=file_verdict,
    )
