"""The VERDICT node: the store's contents read into the vocabulary's fold, and the result recorded.

This is the graph's only decision about the recording. The branches and QUALITY write
``branch_report`` entities — task conformance, typed deviations, no outcome — and propose spans; the
deciding nodes write ``verdict`` entities; this node reads all three, adds the declared task and the
routing decisions, and hands them to ``vocabulary.fold_file_verdict``. The two axes it keeps apart —
triage and release — and the tables it implements are in
``specs/20260817-triage-workflow-dag/verdict.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.live_evidence import declared_task, recording_stem
from senselab.audio.workflows.triage.nodes.branches import BRANCH_FAMILY
from senselab.audio.workflows.triage.nodes.common import (
    NodeResult,
    find_measurement,
    software_agent,
    write_verdict,
)
from senselab.audio.workflows.triage.vocabulary import (
    GRAPH_ORDER,
    RULESET_ROUTING,
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
        Its vocabulary verdict. An ``outcome`` outside :class:`Outcome` is not folded as a
        conclusion: the node is reported as having written something no reader can act on, which
        flags the file, where raising here would lose the whole file verdict along with it.
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
        :data:`UNDETERMINED` token reads as :data:`UNDETERMINED`: a report nobody can interpret
        answered no conformance question, and reading it as a False would flag on a value the fold
        does not understand.
    """
    raw = entity.attributes.get("conformance")
    return raw if isinstance(raw, bool) else UNDETERMINED


def _branch_report_from_entity(entity: Entity) -> BranchReport:
    """The vocabulary report a ``write_report`` entity carries.

    Args:
        entity: A ``branch_report`` entity.

    Returns:
        Its vocabulary report. ``conformance_of`` is read as written; an unknown referent is carried
        through rather than corrected, and the fold treats anything but ``task`` as not being a
        claim about a task.
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

    The rule is the one ``common.find_measurement`` and ``common.resolve_stream`` apply: an
    invalidated entity is never read, and of the survivors sharing a key the latest write wins.

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
        One ``(entity, verdict)`` pair per node, the node's latest live verdict; the file verdict
        itself is excluded by its ``node`` attribute, which is the only discriminator the entity
        carries. A withdrawn verdict does not vote and a superseded one is replaced, not added.
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
        One ``(entity, report)`` pair per node that reported, in order of first appearance. A
        withdrawn report does not vote and a superseded one is replaced, not added.
    """
    return [
        (entity, _branch_report_from_entity(entity))
        for entity in _live_latest(store, "branch_report", lambda e: str(e.attributes.get("node")))
    ]


def _spans_by_node(store: ProvStore) -> dict[str, int]:
    """How many spans each reporting node proposed into its own family.

    The count is taken from the store rather than from the report, because the spans *are* the
    record: a count copied into the report would be a second one able to disagree with it. A span is
    attributed to a node by the activity that generated it and is counted only in that node's own
    family, so a node reaching into another's — which ``dispatch`` already refuses — could not
    inflate its own found/not-found reading here either.

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
        evaluation — which is a reading never made, not a recording nothing routed.
    """
    measurement = find_measurement(store, RULESET_ROUTING)
    if measurement is None or measurement.attributes.get("state") is None:
        return None, []
    return str(measurement.attributes["state"]), [measurement.id]


def _branch_decisions(store: ProvStore) -> tuple[dict[str, BranchDecision], list[str]]:
    """ROUTING's decision per branch.

    Args:
        store: The provenance store.

    Returns:
        The decision per branch name, one per branch under the store's shared rule, and the ids of
        the entities they came from. Empty when ROUTING never ran, which is a graph in which no
        branch was ever asked.
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
        )
        ids.append(entity.id)
    return decisions, ids


def _hint_claims(
    decisions: Mapping[str, BranchDecision], hint: AudioHints | None, *, declared_family: str
) -> dict[str, bool] | None:
    """Which branches the recording's declaration claimed, read off ROUTING's own record of reading it.

    ROUTING resolved the declaration — the stem's own task family through the ruleset's reference
    family sets, and any hint tag ``routing.hint_branch_map`` maps — and wrote the result onto each
    branch's decision. Reading it back is what makes the declaration that added a route the same
    declaration that names a mismatch: a second resolution here could disagree with the first
    whenever the config or the hint handed to the two nodes differ.

    Args:
        decisions: ROUTING's decisions, as read from the store.
        hint: What the recording was declared to contain, if anything.
        declared_family: The task family the recording's own stem declares, empty when it declares
            none. Tested for existence only; what it claims is ROUTING's to say.

    Returns:
        True per claimed branch; a branch the declaration did not name is simply absent. None when a
        declaration existed and no decision survived to say what ROUTING made of it — the claims
        are then unknown, which is not the same as no claim.
    """
    if (hint is not None or declared_family) and not decisions:
        return None
    return {decision.branch: True for decision in decisions.values() if decision.declared}


def _derived_ran(
    store: ProvStore, verdicts: Sequence[NodeVerdict], reports: Sequence[BranchReport]
) -> dict[str, RunState]:
    """Whether each graph node ran, as far as the store can say (N26).

    Operational fact only, and unchanged by the report/decide split: the three states are what the
    runner records and what this derivation falls back to. A node that *reported* counts as having
    concluded exactly as one that decided does, which is what keeps a branch under the new contract
    out of the ``errored`` column.

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
            ``recording`` stream, read only for whether the recording declares a task at all. This
            node reads nothing else.
        source: Accepted for the shared node shape; not read.
        config: The triage configuration, named in the activity by its hash and read for the
            ``verdict.*`` section — which is where every threshold that turns a reading into a
            judgement now lives, because a branch reports and this node decides. The hint was
            already resolved by ROUTING and is read back rather than re-resolved.
        hint: What the recording was declared to contain. Read for branch mismatch only: a hint never
            resolves a finding and never turns a flag into a pass.
        run_dir: Accepted for the shared node shape; VERDICT writes no sidecars.
        ran: Whether each node ran, from the runner, merged over what the store derives so that a
            partial mapping overrides per node without erasing the rest. The derivation reads a
            written verdict as ``completed``, an activity without one as ``errored`` and neither as
            ``skipped`` (N26); the runner's mapping still wins where it speaks, since it knows why a
            node it never called was left out.

    Returns:
        The file verdict on both axes, the verdict entity it was written to, and a view leading with
        that entity followed by every id the fold consumed.
    """
    pairs = _node_verdicts_in_graph_order(store)
    node_verdicts = [node_verdict for _, node_verdict in pairs]
    report_pairs = _branch_reports(store)
    reports = [report for _, report in report_pairs]
    route_state, route_ids = _route_state(store)
    decisions, decision_ids = _branch_decisions(store)
    resolved_ran = {**_derived_ran(store, node_verdicts, reports), **(ran or {})}
    declared_family = declared_task(recording_stem(store))[1]
    file_verdict = fold_file_verdict(
        node_verdicts,
        branch_reports=reports,
        spans_by_node=_spans_by_node(store),
        branch_decisions=decisions,
        ran=resolved_ran,
        hint_claims=_hint_claims(decisions, hint, declared_family=declared_family),
        route_state=route_state,
        declared_family=declared_family or None,
        policy=FoldPolicy.from_config(config),
    )

    software = software_agent(store)
    activity = store.activity(node=NODE, step=None, parameters={"config_hash": config.config_hash})
    store.was_associated_with(activity, software)
    folded_ids = (
        [entity.id for entity, _ in pairs] + [entity.id for entity, _ in report_pairs] + route_ids + decision_ids
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
        detail={
            "triage": file_verdict.triage.value,
            "release": file_verdict.release.value,
            "discard_ground": file_verdict.discard_ground,
            "declared_family": file_verdict.declared_family,
            "findings": dict(file_verdict.findings),
            "conformance": dict(file_verdict.conformance),
            "conformance_of": dict(file_verdict.conformance_of),
            "deviations": {node: list(names) for node, names in file_verdict.deviations.items()},
            "unmeasured": {node: list(names) for node, names in file_verdict.unmeasured.items()},
            "detector_covariates": dict(file_verdict.detector_covariates),
            "routes": dict(file_verdict.routes),
            "route_state": file_verdict.route_state,
            "agreement": dict(file_verdict.agreement),
            "hints": dict(file_verdict.hints),
            "branches": dict(file_verdict.branches),
            "bad_map_values": dict(file_verdict.bad_map_values),
            "ran": {node: state.value for node, state in file_verdict.ran.items()},
            "reasons": [
                {"node": r.node, "outcome": r.outcome.value, "kind": r.kind, "why": r.why} for r in file_verdict.reasons
            ],
        },
    )
    return VerdictResult(
        verdict=node_verdict,
        view=(verdict_id, *folded_ids),
        verdict_entity_id=verdict_id,
        file_verdict=file_verdict,
    )
