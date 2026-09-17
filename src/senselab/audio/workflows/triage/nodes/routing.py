"""ROUTING — which branches run, written down before any of them does.

It measures nothing and classifies nothing. It evaluates the family taxonomy ruleset over the store
as TAXONOMY left it, records that reading as the ``ruleset_routing`` measurement, and turns it into
one ``branch_decision`` per branch. A declared task always adds a route to its own branch; the
declaration never rewrites the reading, never removes a branch and never relaxes a threshold. Its
verdict is always a ``pass``: this node reaches no conclusion about the recording, and an empty
execution set is recorded on the decisions for VERDICT to read rather than flagged here.

A failure to evaluate the ruleset fails the node, because the ruleset is what decides execution.
``specs/20260912-ruleset-in-pipeline/design.md`` holds the staging that brought it here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.live_evidence import evaluate_live_routes, route_attributes
from senselab.audio.workflows.triage.nodes.common import (
    NodeResult,
    software_agent,
    write_measurement,
    write_verdict,
)
from senselab.audio.workflows.triage.routing_analysis.ruleset import load_ruleset
from senselab.audio.workflows.triage.vocabulary import (
    BRANCHES,
    DECLINED,
    ROUTED,
    RULESET_ROUTING,
    UNAVAILABLE,
    UNGATED,
    Outcome,
)
from senselab.utils.prov_store import ProvStore

NODE = "routing"

_SPEECH_TYPE = "speech_type"
_STREAM = "plain"


@dataclass(frozen=True)
class RoutingResult(NodeResult):
    """What ROUTING decided.

    Attributes:
        runs: The branches that will run, in :data:`~senselab.audio.workflows.triage.vocabulary.BRANCHES`
            order. A branch no node implements can be in it; the runner records that rather than
            raising.
        skipped: The branches that will not.
        forced: The branches that run only because the declaration named them, in branch order.
        declared: Every branch the declaration named, whether or not content routed it too.
        empty_set: Whether no branch runs at all.
        route_state: What the ruleset made of the whole recording — ``routed``, ``empty`` or
            ``unexplained``.
    """

    runs: tuple[str, ...]
    skipped: tuple[str, ...]
    forced: tuple[str, ...]
    declared: tuple[str, ...]
    empty_set: bool
    route_state: str


def _declared_tags(hint: AudioHints | None) -> list[str]:
    """Every tag the caller declared, from ``may_contain`` and the task's ``speech_type``.

    Args:
        hint: What the recording was declared to contain, if anything.

    Returns:
        The tags in declaration order, each once.
    """
    if hint is None:
        return []
    declared = [str(tag) for tag in hint.may_contain]
    speech_type = hint.metadata.get(_SPEECH_TYPE)
    if speech_type is not None:
        declared.append(str(speech_type))
    seen: dict[str, None] = {}
    for tag in declared:
        seen.setdefault(tag, None)
    return list(seen)


def _map_tags(tags: list[str], branch_map: dict[str, Any]) -> tuple[dict[str, list[str]], list[str], dict[str, str]]:
    """Sort the declared tags into the branches they name, the ones that name nothing, and the typos.

    Args:
        tags: The declared tags.
        branch_map: ``routing.hint_branch_map`` — tag or ``speech_type`` value to branch. Matched
            ``casefold()``ed on both sides.

    Returns:
        The tags per branch; the tags that reached no branch this graph routes to, whether because
        the map has no entry for them or because the entry names a branch that does not exist; and
        the map entries whose value is not a branch, as ``{tag: value}``.
    """
    folded = {str(tag).casefold(): str(branch) for tag, branch in branch_map.items()}
    by_branch: dict[str, list[str]] = {}
    unmapped: list[str] = []
    bad_values: dict[str, str] = {}
    for tag in tags:
        branch = folded.get(tag.casefold())
        if branch in BRANCHES:
            by_branch.setdefault(str(branch), []).append(tag)
            continue
        unmapped.append(tag)
        if branch is not None:
            bad_values[tag] = branch
    return by_branch, unmapped, bad_values


def _route_states(attributes: dict[str, Any]) -> dict[str, str]:
    """Each branch's route state, from the recorded evaluation.

    Args:
        attributes: The ``ruleset_routing`` measurement's attributes.

    Returns:
        One of :data:`~senselab.audio.workflows.triage.vocabulary.BRANCH_ROUTE_STATES` per branch.
        A branch a gate fired for is ``routed``; one with no fired gate and at least one gate whose
        feature could not be read is ``unavailable``, which is a branch that was never judged rather
        than one that declined; one that names no gate at all is ``ungated``, which is a branch the
        ruleset never looked at; and one whose gates were all silent is ``declined``.
    """
    routed = {str(branch) for branch in attributes.get("routed") or ()}
    unreadable = attributes.get("unavailable") or {}
    ungated = {str(branch) for branch in attributes.get("ungated") or ()}
    states: dict[str, str] = {}
    for branch in BRANCHES:
        if branch in routed:
            states[branch] = ROUTED
        elif unreadable.get(branch):
            states[branch] = UNAVAILABLE
        elif branch in ungated:
            states[branch] = UNGATED
        else:
            states[branch] = DECLINED
    return states


def _why(state: str, forced_by_declaration: bool) -> str:
    """One decision's reason, in controlled vocabulary.

    Args:
        state: The branch's route state, one of :data:`BRANCH_ROUTE_STATES`.
        forced_by_declaration: Whether the branch runs only because the declaration named it.

    Returns:
        The reason.
    """
    return f"route_{state}_forced_by_declaration" if forced_by_declaration else f"route_{state}"


def routing(
    store: ProvStore,
    source: str | None,
    config: TriageConfig,
    hint: AudioHints | None = None,
    *,
    run_dir: Path,
) -> RoutingResult:
    """Evaluate the ruleset over the store and turn its reading, with the declaration, into an execution set.

    A branch the ruleset routed runs. A branch the recording's own declaration names **also** runs,
    whatever the gates made of it, and the decision records the disagreement rather than resolving
    it. The two sources are additive in one direction only: a declaration adds a route and removes
    none.

    ``route_state`` is a closed vocabulary: every value written is in
    :data:`~senselab.audio.workflows.triage.vocabulary.BRANCH_ROUTE_STATES`, and it describes the
    content reading alone — a declared route never rewrites it.

    Args:
        store: The provenance store, holding PREPROCESS's derivatives and TAXONOMY's summaries. Every
            gate reads one of those, so this node runs after TAXONOMY and not before it. It also
            holds ADMIT's ``recording`` stream, whose path carries the declared task.
        source: The stream the pass is running over; ``None`` means the conditioned stream. Recorded
            on every decision so a second pass over another stream stays tellable apart.
        config: The triage configuration, read for ``taxonomy.ruleset`` and
            ``routing.hint_branch_map``.
        hint: What the recording was declared to contain, if anything.
        run_dir: The run directory the store's sidecar paths are relative to. ROUTING writes no
            sidecars of its own; the reader resolves the evidence's against it.

    Returns:
        The branches that run, those that do not, those the declaration added, every branch the
        declaration named, whether the set is empty, and what the ruleset made of the recording as a
        whole.

    Raises:
        ValueError: When the ruleset or the membership rule cannot be loaded from the configuration.
    """
    stream = source or _STREAM
    tags_by_branch, unmapped, bad_values = _map_tags(_declared_tags(hint), config.get("routing.hint_branch_map") or {})

    software = software_agent(store)
    evaluation = store.activity(
        node=NODE, step="ruleset_routing", parameters={"config_hash": config.config_hash, "stream": stream}
    )
    store.was_associated_with(evaluation, software)
    attributes = route_attributes(evaluate_live_routes(store, config, run_dir=run_dir), load_ruleset(config))
    measurement_id = write_measurement(
        store, evaluation, software, name=RULESET_ROUTING, signal=stream, attributes=attributes, extent=None
    )

    states = _route_states(attributes)
    route_state = str(attributes["state"])
    declared_family = str(attributes["family"])
    by_family = {str(branch) for branch in attributes["declared"]}

    activity = store.activity(node=NODE, step=None, parameters={"config_hash": config.config_hash, "stream": stream})
    store.was_associated_with(activity, software)
    store.used(activity, measurement_id)

    runs: list[str] = []
    skipped: list[str] = []
    forced: list[str] = []
    declared: list[str] = []
    declined: list[str] = []
    view: list[str] = [measurement_id]

    for branch in BRANCHES:
        state = states[branch]
        hint_tags = tags_by_branch.get(branch, [])
        by_ruleset = state == ROUTED
        by_declaration = branch in by_family or bool(hint_tags)
        forced_by_declaration = by_declaration and not by_ruleset
        will_run = by_ruleset or forced_by_declaration

        decision_id = store.entity(
            prov_type="branch_decision",
            extent=None,
            attributes={
                "branch": branch,
                "will_run": will_run,
                "route_state": state,
                "unavailable_gates": list((attributes.get("unavailable") or {}).get(branch) or ()),
                "flag_gates": list((attributes.get("flags") or {}).get(branch) or ()),
                "declared": by_declaration,
                "forced_by_declaration": forced_by_declaration,
                "declared_family": declared_family,
                "declared_by_family": branch in by_family,
                "hint_tags": hint_tags,
                "unmapped_tags": unmapped,
                "bad_map_values": bad_values,
                "why": _why(state, forced_by_declaration),
                "stream": stream,
            },
        )
        store.was_generated_by(decision_id, activity)
        store.was_attributed_to(decision_id, software)
        store.was_derived_from(decision_id, measurement_id)
        view.append(decision_id)

        if will_run:
            runs.append(branch)
        else:
            skipped.append(branch)
            declined.append(f"{branch} {state}")
        if by_declaration:
            declared.append(branch)
        if forced_by_declaration:
            forced.append(branch)

    empty_set = not runs
    if empty_set:
        why = f"no branch runs ({route_state}); " + ", ".join(declined)
    else:
        why = "runs: " + ", ".join(runs)

    verdict_id, verdict = write_verdict(
        store,
        activity,
        software,
        node=NODE,
        outcome=Outcome.PASS,
        kind=None,
        why=why,
        detail={
            "runs": list(runs),
            "skipped": list(skipped),
            "forced": list(forced),
            "declared": list(declared),
            "declared_family": declared_family,
            "empty_set": empty_set,
            "route_state": route_state,
            "routes": dict(states),
        },
    )
    view.append(verdict_id)
    return RoutingResult(
        verdict=verdict,
        view=tuple(view),
        verdict_entity_id=verdict_id,
        runs=tuple(runs),
        skipped=tuple(skipped),
        forced=tuple(forced),
        declared=tuple(declared),
        empty_set=empty_set,
        route_state=route_state,
    )
