"""ROUTING — which branches run, written down before any of them does.

It measures nothing and classifies nothing: it reads TAXONOMY's ``kind`` elements and the caller's
hints, and writes one ``branch_decision`` per branch. A hint forces a branch to run; it never
rewrites the classification, never removes a branch and never relaxes a threshold. Its verdict is
always a ``pass``: this node reaches no conclusion about the recording, and an empty execution set is
recorded on the decisions for VERDICT to read rather than flagged here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.common import NodeResult, live_entities, software_agent, write_verdict
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.prov_store import Entity, ProvStore

NODE = "routing"

BRANCH_FOR_KIND = {"airway": "AIRWAY", "speech": "SPEECH", "voice": "VOICE"}

PRESENT = "present"
ABSENT = "absent"
UNCERTAIN = "uncertain"
UNREADABLE = "unreadable"

KIND_STATES = (PRESENT, ABSENT, UNCERTAIN)

_SPEECH_TYPE = "speech_type"
_STREAM = "plain"


@dataclass(frozen=True)
class RoutingResult(NodeResult):
    """What ROUTING decided.

    Attributes:
        runs: The branches that will run, in ``BRANCH_FOR_KIND`` order.
        skipped: The branches that will not.
        forced: The branches that run only because a hint named their kind.
        empty_set: Whether no branch runs at all.
    """

    runs: tuple[str, ...]
    skipped: tuple[str, ...]
    forced: tuple[str, ...]
    empty_set: bool


def _classifications(store: ProvStore) -> dict[str, Entity]:
    """TAXONOMY's live ``kind`` elements, the latest one per kind.

    Args:
        store: The provenance store.

    Returns:
        The kind element per kind name. A kind nothing wrote is simply absent from the mapping.
    """
    latest: dict[str, Entity] = {}
    for entity in live_entities(store, "kind"):
        latest[str(entity.attributes["kind"])] = entity
    return latest


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


def _map_tags(tags: list[str], kind_map: dict[str, Any]) -> tuple[dict[str, list[str]], list[str], dict[str, str]]:
    """Sort the declared tags into the kinds they name, the ones that name nothing, and the typos.

    Args:
        tags: The declared tags.
        kind_map: ``routing.hint_kind_map`` — tag or ``speech_type`` value to kind. Matched
            ``casefold()``ed on both sides.

    Returns:
        The tags per kind; the tags that reached no kind this graph screens, whether because the map
        has no entry for them or because the entry names a kind that does not exist; and the map
        entries whose value is not a kind, as ``{tag: value}``.
    """
    folded = {str(tag).casefold(): str(kind) for tag, kind in kind_map.items()}
    by_kind: dict[str, list[str]] = {}
    unmapped: list[str] = []
    bad_values: dict[str, str] = {}
    for tag in tags:
        kind = folded.get(tag.casefold())
        if kind in BRANCH_FOR_KIND:
            by_kind.setdefault(str(kind), []).append(tag)
            continue
        unmapped.append(tag)
        if kind is not None:
            bad_values[tag] = kind
    return by_kind, unmapped, bad_values


def _why(state: str, forced_by_hint: bool) -> str:
    """One decision's reason, in controlled vocabulary.

    Args:
        state: The kind's state, already folded into the closed vocabulary.
        forced_by_hint: Whether the branch runs only because a hint named its kind.

    Returns:
        The reason.
    """
    return f"kind_{state}_forced_by_hint" if forced_by_hint else f"kind_{state}"


def routing(
    store: ProvStore,
    source: str | None,
    config: TriageConfig,
    hint: AudioHints | None = None,
    *,
    run_dir: Path,
) -> RoutingResult:
    """Turn TAXONOMY's classification and the caller's hints into an execution set.

    A kind classified ``absent`` withholds its branch; anything else runs it, since a state this node
    cannot read is not evidence of absence. A hint naming an absent kind forces that branch to run and
    the decision records the disagreement rather than resolving it.

    ``kind_state`` and ``why`` are closed vocabularies: a state outside :data:`KIND_STATES` reads
    ``unreadable``, and the string TAXONOMY actually wrote is kept verbatim in ``raw_state``.

    Args:
        store: The provenance store, holding TAXONOMY's ``kind`` elements.
        source: The stream the pass is running over; ``None`` means the conditioned stream. Recorded
            on every decision so a second pass over another stream stays tellable apart.
        config: The triage configuration, read for ``routing.hint_kind_map``.
        hint: What the recording was declared to contain, if anything.
        run_dir: Accepted for the shared node shape; ROUTING writes no sidecars.

    Returns:
        The branches that run, those that do not, those a hint forced, and whether the set is empty.
    """
    stream = source or _STREAM
    tags_by_kind, unmapped, bad_values = _map_tags(_declared_tags(hint), config.get("routing.hint_kind_map") or {})
    classified = _classifications(store)

    software = software_agent(store)
    activity = store.activity(node=NODE, step=None, parameters={"config_hash": config.config_hash, "stream": stream})
    store.was_associated_with(activity, software)
    for kind in BRANCH_FOR_KIND:
        classification = classified.get(kind)
        if classification is not None:
            store.used(activity, classification.id)

    runs: list[str] = []
    skipped: list[str] = []
    forced: list[str] = []
    declined: list[str] = []
    view: list[str] = []

    for kind, branch in BRANCH_FOR_KIND.items():
        classification = classified.get(kind)
        raw_state = str(classification.attributes["state"]) if classification is not None else None
        if raw_state is None:
            state = UNCERTAIN
        else:
            state = raw_state if raw_state in KIND_STATES else UNREADABLE
        hint_tags = tags_by_kind.get(kind, [])
        by_classification = state != ABSENT
        forced_by_hint = bool(hint_tags) and not by_classification
        will_run = by_classification or forced_by_hint

        decision_id = store.entity(
            prov_type="branch_decision",
            extent=None,
            attributes={
                "branch": branch,
                "kind": kind,
                "will_run": will_run,
                "kind_state": state,
                "raw_state": raw_state,
                "forced_by_hint": forced_by_hint,
                "hint_tags": hint_tags,
                "unmapped_tags": unmapped,
                "bad_map_values": bad_values,
                "why": _why(state, forced_by_hint),
                "stream": stream,
            },
        )
        store.was_generated_by(decision_id, activity)
        store.was_attributed_to(decision_id, software)
        if classification is not None:
            store.was_derived_from(decision_id, classification.id)
        view.append(decision_id)

        if will_run:
            runs.append(branch)
        else:
            skipped.append(branch)
            declined.append(f"{kind} {state}")
        if forced_by_hint:
            forced.append(branch)

    empty_set = not runs
    if empty_set:
        why = "no branch runs; " + ", ".join(declined)
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
        detail={"runs": list(runs), "skipped": list(skipped), "forced": list(forced), "empty_set": empty_set},
    )
    view.append(verdict_id)
    return RoutingResult(
        verdict=verdict,
        view=tuple(view),
        verdict_entity_id=verdict_id,
        runs=tuple(runs),
        skipped=tuple(skipped),
        forced=tuple(forced),
        empty_set=empty_set,
    )
