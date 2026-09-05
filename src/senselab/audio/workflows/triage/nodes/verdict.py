"""The VERDICT node: the store's contents read into the vocabulary's fold, and the result recorded.

The fold itself is ``vocabulary.fold_file_verdict``; this node maps store facts onto its inputs and
writes its result back. The two axes it keeps apart — triage and release — and the tables it
implements are in ``specs/20260817-triage-workflow-dag/verdict.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.common import NodeResult, software_agent, write_verdict
from senselab.audio.workflows.triage.vocabulary import (
    GRAPH_ORDER,
    BranchDecision,
    FileVerdict,
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


def _screened(store: ProvStore) -> tuple[dict[str, str], list[str]]:
    """TAXONOMY's classification per kind, verbatim.

    Args:
        store: The provenance store.

    Returns:
        The state per kind as the string TAXONOMY wrote, one per kind under the store's shared rule,
        and the ids of the entities they came from.
    """
    screened: dict[str, str] = {}
    ids: list[str] = []
    for entity in _live_latest(store, "kind", lambda e: str(e.attributes["kind"])):
        screened[str(entity.attributes["kind"])] = str(entity.attributes["state"])
        ids.append(entity.id)
    return screened, ids


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
            kind=str(entity.attributes["kind"]),
            will_run=bool(entity.attributes["will_run"]),
            kind_state=str(entity.attributes["kind_state"]),
            forced_by_hint=bool(entity.attributes["forced_by_hint"]),
            hint_tags=tuple(str(tag) for tag in entity.attributes.get("hint_tags") or ()),
            bad_map_values={
                str(tag): str(value) for tag, value in (entity.attributes.get("bad_map_values") or {}).items()
            },
        )
        ids.append(entity.id)
    return decisions, ids


def _hint_claims(decisions: Mapping[str, BranchDecision], hint: AudioHints | None) -> dict[str, bool] | None:
    """Which kinds the caller's declaration claimed, read off ROUTING's own record of reading it.

    ROUTING resolved the declaration against ``routing.hint_kind_map`` and wrote the tags naming each
    branch's kind onto that branch's decision. Reading them back is what makes the tag that forced a
    branch the same tag that names a mismatch: a second resolution here could disagree with the first
    whenever the config or the hint handed to the two nodes differ.

    Args:
        decisions: ROUTING's decisions, as read from the store.
        hint: What the recording was declared to contain, if anything.

    Returns:
        True per claimed kind; a kind no declared tag reached is simply absent. None when a
        declaration was supplied and no decision survived to say what ROUTING made of it — the claims
        are then unknown, which is not the same as no claim.
    """
    if hint is not None and not decisions:
        return None
    return {decision.kind: True for decision in decisions.values() if decision.hint_tags}


def _derived_ran(store: ProvStore, verdicts: Sequence[NodeVerdict]) -> dict[str, RunState]:
    """Whether each graph node ran, as far as the store can say (N26).

    Args:
        store: The provenance store, read for which nodes have an activity.
        verdicts: Every node verdict read from the store.

    Returns:
        ``COMPLETED`` for a node carrying a verdict, ``ERRORED`` for one carrying an activity but no
        live verdict, and ``SKIPPED`` for one carrying neither.
    """
    concluded = {v.node for v in verdicts}
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
    """Fold every node's verdict, ROUTING's decisions and TAXONOMY's classification into one file verdict.

    Args:
        store: The provenance store, holding every node's ``verdict`` entity, ROUTING's
            ``branch_decision`` entities and TAXONOMY's ``kind`` entities. This node reads nothing
            else.
        source: Accepted for the shared node shape; not read.
        config: The triage configuration, named in the activity by its hash. VERDICT has no
            thresholds and reads no key: the hint was already resolved by ROUTING, and this node
            reads that resolution rather than repeating it.
        hint: What the recording was declared to contain. Read for branch mismatch only: a hint never
            resolves a kind and never turns a flag into a pass.
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
    screened, kind_ids = _screened(store)
    decisions, decision_ids = _branch_decisions(store)
    resolved_ran = {**_derived_ran(store, node_verdicts), **(ran or {})}
    file_verdict = fold_file_verdict(
        node_verdicts,
        screened=screened,
        branch_decisions=decisions,
        ran=resolved_ran,
        hint_claims=_hint_claims(decisions, hint),
    )

    software = software_agent(store)
    activity = store.activity(node=NODE, step=None, parameters={"config_hash": config.config_hash})
    store.was_associated_with(activity, software)
    folded_ids = [entity.id for entity, _ in pairs] + kind_ids + decision_ids
    for folded_id in folded_ids:
        store.used(activity, folded_id)

    verdict_id, node_verdict = write_verdict(
        store,
        activity,
        software,
        node=NODE,
        outcome=file_verdict.triage,
        kind=None,
        why=f"folded {len(node_verdicts)} node verdicts over {len(screened)} screened kinds",
        detail={
            "triage": file_verdict.triage.value,
            "release": file_verdict.release.value,
            "discard_ground": file_verdict.discard_ground,
            "kinds": dict(file_verdict.kinds),
            "screened": dict(file_verdict.screened),
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
