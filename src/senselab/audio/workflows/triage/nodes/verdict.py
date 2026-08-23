"""The VERDICT node: the store's contents read into the vocabulary's fold, and the result recorded.

The fold itself is ``vocabulary.fold_file_verdict``; this node maps store facts onto its inputs and
writes its result back. The two axes it keeps apart — triage and release — and the mapping table it
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
    FileVerdict,
    KindState,
    NodeVerdict,
    Outcome,
    Release,
    RunState,
    fold_file_verdict,
)
from senselab.utils.prov_store import PROV_TYPE, Entity, ProvStore

NODE = "VERDICT"

_GRAPH_ORDER = ("ADMIT", "PREPROCESS", "TAXONOMY", "AIRWAY", "SPEECH", "VOICE", "REDACT")
_NOT_SCREENED = "not_screened"


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
        Its vocabulary verdict.
    """
    attributes = entity.attributes
    return NodeVerdict(
        node=attributes["node"],
        outcome=Outcome(attributes["outcome"]),
        kind=attributes.get("kind"),
        why=attributes["why"],
    )


def _live_latest(store: ProvStore, prov_type: PROV_TYPE, key: Callable[[Entity], str]) -> list[Entity]:
    """Entities of one type under the store's shared rule, one per key.

    The rule is the one ``common.find_measurement`` and ``common.resolve_stream`` apply: an
    invalidated entity is never read, and of the survivors sharing a key the latest write wins.

    Args:
        store: The provenance store.
        prov_type: The entity type to read.
        key: What makes two entities the same assertion — a node name, a kind name.

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


def _kind_predictions(store: ProvStore) -> tuple[dict[str, KindState], list[str]]:
    """TAXONOMY's kind entities as ``KindState``s; ``not_screened`` is ``UNDECIDED`` (N27).

    Args:
        store: The provenance store.

    Returns:
        The predictions per kind, one per kind under the store's shared rule, and the ids of the
        entities they came from.

    Raises:
        ValueError: If a kind entity carries a state outside the vocabulary. The message names the
            kind and the entity so the writer can be found; a state nobody can read must not be
            folded into a verdict.
    """
    predictions: dict[str, KindState] = {}
    ids: list[str] = []
    for entity in _live_latest(store, "kind", lambda e: str(e.attributes["kind"])):
        kind = entity.attributes["kind"]
        state = entity.attributes["state"]
        if state == _NOT_SCREENED:
            predictions[kind] = KindState.UNDECIDED
        else:
            try:
                predictions[kind] = KindState(state)
            except ValueError as error:
                raise ValueError(
                    f"kind entity {entity.id} for kind {kind!r} carries state {state!r}, which is not a KindState; "
                    "the node that wrote it must be repaired before this screen can be folded"
                ) from error
        ids.append(entity.id)
    return predictions, ids


def _release_from(verdicts: Sequence[NodeVerdict]) -> Release:
    """REDACT's outcome as a release state; an absent verdict means unexamined, never releasable.

    Args:
        verdicts: Every node verdict read from the store.

    Returns:
        The release state for REDACT's artifacts only — never for anything in the store. Only
        ``pass`` clears an artifact, so the mapping is total over ``Outcome``: a flag, or any member
        added later, withholds rather than defaulting to cleared.
    """
    redact = next((v for v in verdicts if v.node == "REDACT"), None)
    if redact is None:
        return Release.NOT_ASSESSED
    return Release.RELEASABLE if redact.outcome is Outcome.PASS else Release.WITHHELD


def _derived_ran(verdicts: Sequence[NodeVerdict]) -> dict[str, RunState]:
    """Whether each graph node ran, as far as the store can say (N26).

    Args:
        verdicts: Every node verdict read from the store.

    Returns:
        ``COMPLETED`` for every node carrying a verdict and ``SKIPPED`` for every other graph node.
        ``ERRORED`` never appears: a node that raised wrote no verdict and is indistinguishable here
        from one that was never asked to run.
    """
    concluded = {v.node for v in verdicts}
    return {node: RunState.COMPLETED if node in concluded else RunState.SKIPPED for node in _GRAPH_ORDER}


def _is_gated(predictions: Mapping[str, KindState], verdicts: Sequence[NodeVerdict]) -> bool:
    """Whether any absent-predicted kind went without a branch verdict.

    Args:
        predictions: TAXONOMY's prediction per kind.
        verdicts: Every node verdict read from the store.

    Returns:
        True when at least one kind predicted absent has no branch verdict to be read against, so
        the contradiction check never happened for it.
    """
    screened = {v.kind for v in verdicts if v.kind}
    return any(state is KindState.ABSENT and kind not in screened for kind, state in predictions.items())


def verdict(
    store: ProvStore,
    source: None,
    config: TriageConfig,
    hint: AudioHints | None = None,
    *,
    run_dir: Path,
    ran: Mapping[str, RunState] | None = None,
) -> VerdictResult:
    """Fold every node's verdict, TAXONOMY's kinds and REDACT's release into one file verdict.

    Args:
        store: The provenance store, holding every node's ``verdict`` entity and TAXONOMY's ``kind``
            entities. This node reads nothing else.
        source: Accepted for the shared node shape; not read.
        config: The triage configuration, named in the activity by its hash. VERDICT has no
            thresholds.
        hint: Accepted for the shared node shape; not read.
        run_dir: Accepted for the shared node shape; VERDICT writes no sidecars.
        ran: Whether each node ran, from the runner, merged over what the store derives so that a
            partial mapping overrides per node without erasing the rest. The derivation reads a
            written verdict as ``completed`` and every other graph node as ``skipped``, and cannot
            see ``errored``: a node that raised wrote no verdict and is indistinguishable there from
            one never asked to run (N26), which is why the runner's mapping wins where it speaks.

    Returns:
        The file verdict on both axes, the verdict entity it was written to, and a view leading with
        that entity followed by every id the fold consumed.

    Raises:
        ValueError: If a ``kind`` entity carries a state outside the vocabulary (see
            :func:`_kind_predictions`).
    """
    pairs = _node_verdicts_in_graph_order(store)
    node_verdicts = [node_verdict for _, node_verdict in pairs]
    predictions, kind_ids = _kind_predictions(store)
    resolved_ran = {**_derived_ran(node_verdicts), **(ran or {})}
    file_verdict = fold_file_verdict(node_verdicts, predictions, resolved_ran, release=_release_from(node_verdicts))
    gated = _is_gated(predictions, node_verdicts)

    software = software_agent(store)
    activity = store.activity(node=NODE, step=None, parameters={"config_hash": config.config_hash})
    store.was_associated_with(activity, software)
    folded_ids = [entity.id for entity, _ in pairs] + kind_ids
    for folded_id in folded_ids:
        store.used(activity, folded_id)

    verdict_id, node_verdict = write_verdict(
        store,
        activity,
        software,
        node=NODE,
        outcome=file_verdict.triage,
        kind=None,
        why=f"folded {len(node_verdicts)} node verdicts over {len(predictions)} screened kinds",
        detail={
            "triage": file_verdict.triage.value,
            "release": file_verdict.release.value,
            "kinds": {kind: state.value for kind, state in file_verdict.kinds.items()},
            "ran": {node: state.value for node, state in file_verdict.ran.items()},
            "screened": bool(predictions),
            "gated": gated,
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
