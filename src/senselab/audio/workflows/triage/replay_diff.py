"""The differential over a replayed store: what re-deciding a finished run changed.

A replay retires rather than deletes, so one replayed store carries both decisions and the
invalidation edges between them. The entities a replay retired are exactly those that were live
when it started; everything a replayed node generated and the replay did not retire is what it
wrote. :func:`split_generations` recovers those two sets from the edges alone, :func:`generation`
folds each into a comparable summary, and :func:`diff_store` reports what moved between them.

Every value this module emits is categorical or a count: outcome and route names, declared
deviation types, controlled-vocabulary grounds, config paths, PII detector categories and
per-recording stems. No transcript text and no detected string is read out of a store here.

See ``specs/20260922-replay-decisions-over-a-finished-corpus/differential.md``.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from senselab.audio.workflows.triage.corpus_report import reason_ground
from senselab.audio.workflows.triage.extend import (
    DECISION_SUPERSEDED,
    REPLAY_MARKER_STEP,
    REPLAY_NODE,
    REPLAYED_NODES,
    RUN_SUBDIR,
    STORE_FILE,
)
from senselab.utils.prov_store import Entity, ProvStore

VERDICT_NODE = "VERDICT"
"""The node whose verdict entity carries the whole file-level decision."""

SPEECH_NODE = "SPEECH"
REDACT_NODE = "REDACT"
PREPROCESS_NODE = "PREPROCESS"

RECORDING_STREAM = "recording"
"""The stream ADMIT writes only when it admits the recording, and QUALITY reads."""

COMPLETED = "completed"
"""The run state a node that ran to the end carries."""

PII_SCAN = "pii_scan"
"""The measurement SPEECH writes for its PII scan, whether or not the scan ran."""

SCANNED_KEY = "scanned"
"""The key a ``pii_scan`` measurement carries only when the scan was skipped."""

DEVIATE_VERB = "deviate"
"""The verb on the assertion a branch writes for one typed deviation."""

DEVIATION_TYPE_KEY = "deviation_type"

BEFORE = "before"
AFTER = "after"

OK = "ok"
"""A store carrying a replay marker and both generations of decision."""

NOT_REPLAYED = "not_replayed"
"""A store carrying no replay marker: the replay has not reached this recording."""

NO_STORE = "no_store"
"""A run root holding no store at all."""

UNREADABLE = "unreadable"
"""A store that would not parse."""

NOTHING_RETIRED = "nothing_retired"
"""A replayed store whose replay found no live decision to retire."""

AMBIGUOUS = "ambiguous"
"""A store replayed more than once, whose retirements cannot be attributed to one pass."""

NOT_REPLAYABLE = "not_replayable"
"""A run the replay could not re-decide, because the output it re-enters over was never written.

The replay starts at TAXONOMY and does not re-run ADMIT or PREPROCESS, and neither node is a gate
it can observe. So a run those two did not carry has its replayed nodes *called* where the original
run skipped them, and they error. The moves that follow are the re-entry point's, not decisions,
and are excluded from the comparison rather than counted.
"""

STATUSES = (OK, NOT_REPLAYED, NO_STORE, UNREADABLE, NOTHING_RETIRED, AMBIGUOUS, NOT_REPLAYABLE)
"""Every status a row may carry."""

ADMIT_DID_NOT_ADMIT = "ADMIT did not admit the recording; the store holds no source stream"
"""Why a run is not replayable when ADMIT rejected it."""

PREPROCESS_DID_NOT_COMPLETE = "PREPROCESS did not complete; there is no conditioned output to read"
"""Why a run is not replayable when PREPROCESS errored or was skipped."""

UNREAD = "UNREAD"
"""The stand-in for a generation that left no file-level decision."""

ABSENT = "absent"
"""The stand-in for a per-node value one generation carries and the other does not."""

_TRANSITION_KEYS = (
    "triage",
    "release",
    "route_state",
    "discard_ground",
    "declared_family",
)
"""The scalar decision fields a transition matrix is taken over."""

_MAPPING_KEYS = (
    "routes",
    "findings",
    "conformance",
    "conformance_of",
    "agreement",
    "hints",
    "ran",
)
"""The node- or branch-keyed decision fields a transition matrix is taken over, key by key."""

_SET_KEYS = ("deviations", "unmeasured")
"""The node-keyed decision fields whose values are sets of names, compared as gained and lost."""


@dataclass(frozen=True)
class Generation:
    """One pass's decision over one recording, as the differential compares it.

    Attributes:
        decision: The ``VERDICT`` verdict entity's attributes, or None when the pass left none.
        grounds: ``node|ground`` for every contributing verdict that did not pass.
        report_deviations: Branch node to the deviation types its ``branch_report`` carried.
        deviation_assertions: Branch node to deviation type to how many assertions it wrote.
        pii_scanned: Whether SPEECH's PII scan ran, or None when it left no ``pii_scan``.
        pii_categories: PII detector category to how many findings the pass recorded.
        entities: How many entities of the replayed nodes this pass accounts for.
    """

    decision: dict[str, Any] | None = None
    grounds: tuple[str, ...] = ()
    report_deviations: dict[str, list[str]] = field(default_factory=dict)
    deviation_assertions: dict[str, dict[str, int]] = field(default_factory=dict)
    pii_scanned: bool | None = None
    pii_categories: dict[str, int] = field(default_factory=dict)
    entities: int = 0

    def value(self, key: str) -> Any:  # noqa: ANN401 — whatever the decision field holds
        """One decision field, or the unread stand-in when the pass decided nothing.

        Args:
            key: The field's name, as :meth:`FileVerdict.record` keys it.

        Returns:
            The field's value, or :data:`UNREAD` when there is no decision, or None when the
            decision carries no such field.
        """
        if self.decision is None:
            return UNREAD
        return self.decision.get(key)

    def mapping(self, key: str) -> dict[str, Any]:
        """One node- or branch-keyed decision field, as a mapping.

        Args:
            key: The field's name.

        Returns:
            The mapping, empty when the pass decided nothing or carries no such field.
        """
        value = self.value(key)
        return dict(value) if isinstance(value, dict) else {}

    def pii_findings(self) -> int:
        """How many PII findings the pass recorded.

        Returns:
            The total across every category.
        """
        return sum(self.pii_categories.values())


def replay_markers(store: ProvStore) -> list[dict[str, Any]]:
    """Every replay marker a store carries, in write order.

    Args:
        store: The run's store.

    Returns:
        Each marker's parameters — its config hash, the commit it ran under and how many decisions
        it retired.
    """
    return [
        dict(activity.parameters) for activity in store.activities(REPLAY_NODE) if activity.step == REPLAY_MARKER_STEP
    ]


def _retirement_activities(store: ProvStore) -> set[str]:
    """The ids of the activities a replay used to retire the decisions it was about to remake.

    Args:
        store: The run's store.

    Returns:
        The activity ids.
    """
    return {activity.id for activity in store.activities(REPLAY_NODE) if activity.step == DECISION_SUPERSEDED}


def _generating_node(store: ProvStore, entity_id: str) -> str | None:
    """The node whose activity generated one entity.

    Args:
        store: The run's store.
        entity_id: The entity's id.

    Returns:
        The node's name, or None when the entity names no generating activity the store holds.
    """
    activity_id = store.generated_by(entity_id)
    if activity_id is None:
        return None
    try:
        return store.get_activity(activity_id).node
    except KeyError:
        return None


def split_generations(store: ProvStore) -> tuple[list[Entity], list[Entity]]:
    """The two passes a replayed store holds, recovered from its invalidation edges alone.

    Args:
        store: The run's store.

    Returns:
        ``(before, after)`` — the entities the replay retired, and the entities a replayed node
        generated that it did not. Both in the store's own order.
    """
    retiring = _retirement_activities(store)
    before: list[Entity] = []
    after: list[Entity] = []
    for entity in store.entities():
        if any(activity in retiring for activity in store.invalidated_by(entity.id)):
            before.append(entity)
        elif _generating_node(store, entity.id) in REPLAYED_NODES:
            after.append(entity)
    return before, after


def generation(store: ProvStore, entities: Sequence[Entity]) -> Generation:
    """Fold one pass's entities into the summary the differential compares.

    Args:
        store: The run's store, read for each entity's generating node.
        entities: The pass's entities, as :func:`split_generations` separated them.

    Returns:
        The pass's summary.
    """
    decision: dict[str, Any] | None = None
    report_deviations: dict[str, list[str]] = {}
    assertions: dict[str, Counter[str]] = {}
    categories: Counter[str] = Counter()
    scans: list[dict[str, Any]] = []
    for entity in entities:
        attributes = entity.attributes
        if entity.prov_type == "verdict" and attributes.get("node") == VERDICT_NODE:
            decision = dict(attributes)
        elif entity.prov_type == "branch_report":
            report_deviations[str(attributes.get("node"))] = sorted(
                str(name) for name in attributes.get("deviations") or []
            )
        elif entity.prov_type == "assertion" and attributes.get("verb") == DEVIATE_VERB:
            node = _generating_node(store, entity.id) or "unattributed"
            assertions.setdefault(node, Counter())[str(attributes.get(DEVIATION_TYPE_KEY))] += 1
        elif entity.prov_type == "pii":
            categories[str(attributes.get("category"))] += 1
        elif entity.prov_type == "measurement" and attributes.get("name") == PII_SCAN:
            scans.append(dict(attributes))
    return Generation(
        decision=decision,
        grounds=_grounds(decision),
        report_deviations=report_deviations,
        deviation_assertions={node: dict(counts.most_common()) for node, counts in sorted(assertions.items())},
        pii_scanned=_scanned(scans),
        pii_categories=dict(categories.most_common()),
        entities=len(entities),
    )


def _scanned(scans: Sequence[Mapping[str, Any]]) -> bool | None:
    """Whether SPEECH's PII scan ran, from the ``pii_scan`` measurements one pass wrote.

    Args:
        scans: Every ``pii_scan`` measurement's attributes, in write order.

    Returns:
        False when any of them records the scan as skipped, True when the pass wrote one and none
        does, and None when the pass wrote none at all.
    """
    if not scans:
        return None
    return not any(scan.get(SCANNED_KEY) is False for scan in scans)


def _grounds(decision: Mapping[str, Any] | None) -> tuple[str, ...]:
    """Every ground a decision's contributing verdicts gave for not passing.

    Args:
        decision: The file-level decision, or None.

    Returns:
        ``node|ground`` for each, sorted and deduplicated.
    """
    if decision is None:
        return ()
    found: set[str] = set()
    for reason in decision.get("reasons") or []:
        if not isinstance(reason, dict) or str(reason.get("outcome")) == "pass":
            continue
        if reason.get("why"):
            found.add(f"{reason.get('node')}|{reason_ground(str(reason['why']))}")
    return tuple(sorted(found))


def _transition(before: Any, after: Any) -> str:  # noqa: ANN401 — whatever the field holds
    """One field's move, as the key a transition matrix counts under.

    Args:
        before: The value the first pass held.
        after: The value the second pass holds.

    Returns:
        ``before->after``.
    """
    return f"{before}->{after}"


def _set_change(before: Iterable[Any], after: Iterable[Any]) -> dict[str, list[str]]:
    """What one pass added to and removed from a set of names.

    Args:
        before: The first pass's names.
        after: The second pass's names.

    Returns:
        ``{gained, lost}``, each sorted, with either key omitted when it is empty.
    """
    first = {str(name) for name in before}
    second = {str(name) for name in after}
    change: dict[str, list[str]] = {}
    if second - first:
        change["gained"] = sorted(second - first)
    if first - second:
        change["lost"] = sorted(first - second)
    return change


def _mapping_transitions(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, str]:
    """Every key's move across two node- or branch-keyed fields.

    Args:
        before: The first pass's mapping.
        after: The second pass's mapping.

    Returns:
        Key to ``before->after``, over the keys whose value moved, with a key only one of the two
        mappings carries reading against :data:`ABSENT`.
    """
    return {
        key: _transition(before.get(key, ABSENT), after.get(key, ABSENT))
        for key in sorted(set(before) | set(after))
        if before.get(key, ABSENT) != after.get(key, ABSENT)
    }


def _counter_change(
    before: Mapping[str, Mapping[str, int]], after: Mapping[str, Mapping[str, int]]
) -> dict[str, dict[str, list[int]]]:
    """Per node and type, how many typed findings each pass wrote, where the two differ.

    Args:
        before: The first pass's node to type to count.
        after: The second pass's.

    Returns:
        Node to type to ``[before, after]``, carrying only the entries that moved.
    """
    out: dict[str, dict[str, list[int]]] = {}
    for node in sorted(set(before) | set(after)):
        first, second = before.get(node, {}), after.get(node, {})
        moved = {
            name: [int(first.get(name, 0)), int(second.get(name, 0))]
            for name in sorted(set(first) | set(second))
            if first.get(name, 0) != second.get(name, 0)
        }
        if moved:
            out[node] = moved
    return out


def _pii_axis(before: Generation, after: Generation) -> dict[str, Any]:
    """What the PII scan and the redaction that depends on it did across the two passes.

    Args:
        before: The retired pass.
        after: The replayed pass.

    Returns:
        The scan's state, the findings recorded and the run state of SPEECH and REDACT, each as
        ``[before, after]``, plus the categories whose count moved.
    """
    axis: dict[str, Any] = {
        SCANNED_KEY: [before.pii_scanned, after.pii_scanned],
        "findings_n": [before.pii_findings(), after.pii_findings()],
        "speech_ran": [before.mapping("ran").get(SPEECH_NODE, ABSENT), after.mapping("ran").get(SPEECH_NODE, ABSENT)],
        "redact_ran": [before.mapping("ran").get(REDACT_NODE, ABSENT), after.mapping("ran").get(REDACT_NODE, ABSENT)],
    }
    categories = {
        name: [before.pii_categories.get(name, 0), after.pii_categories.get(name, 0)]
        for name in sorted(set(before.pii_categories) | set(after.pii_categories))
        if before.pii_categories.get(name, 0) != after.pii_categories.get(name, 0)
    }
    if categories:
        axis["categories"] = categories
    return axis


def diff_generations(before: Generation, after: Generation) -> dict[str, Any]:
    """What moved between two passes over one recording.

    Args:
        before: The retired pass.
        after: The replayed pass.

    Returns:
        The row body: the scalar transitions, the fields whose value moved, the grounds and
        deviation types gained and lost, the PII axis, and whether anything moved at all.
    """
    first = before.decision or {}
    second = after.decision or {}
    comparable = sorted(set(first) & set(second))
    changed = [key for key in comparable if first[key] != second[key]]
    body: dict[str, Any] = {
        "transitions": {key: _transition(before.value(key), after.value(key)) for key in _TRANSITION_KEYS},
        "changed_keys": changed,
        "new_keys": sorted(set(second) - set(first)),
        "dropped_keys": sorted(set(first) - set(second)),
        "entities": [before.entities, after.entities],
        "pii": _pii_axis(before, after),
    }
    for key in _MAPPING_KEYS:
        moved = _mapping_transitions(before.mapping(key), after.mapping(key))
        if moved:
            body.setdefault("moved", {})[key] = moved
    for key in _SET_KEYS:
        per_node = {
            node: change
            for node in sorted(set(before.mapping(key)) | set(after.mapping(key)))
            if (change := _set_change(before.mapping(key).get(node) or [], after.mapping(key).get(node) or []))
        }
        if per_node:
            body.setdefault("moved", {})[key] = per_node
    grounds = _set_change(before.grounds, after.grounds)
    if grounds:
        body["grounds"] = grounds
    reports = {
        node: change
        for node in sorted(set(before.report_deviations) | set(after.report_deviations))
        if (change := _set_change(before.report_deviations.get(node) or [], after.report_deviations.get(node) or []))
    }
    if reports:
        body["report_deviations"] = reports
    assertions = _counter_change(before.deviation_assertions, after.deviation_assertions)
    if assertions:
        body["deviation_assertions"] = assertions
    decided_alike = (before.decision is None) == (after.decision is None)
    body["identical"] = decided_alike and not (changed or grounds or reports or assertions or _pii_moved(body["pii"]))
    return body


def _pii_moved(axis: Mapping[str, Any]) -> bool:
    """Whether anything on the PII axis differs between the two passes.

    Args:
        axis: What :func:`_pii_axis` built.

    Returns:
        True when any of its paired readings differ.
    """
    return any(
        isinstance(value, list) and len(value) == 2 and value[0] != value[1]
        for key, value in axis.items()
        if key != "categories"
    ) or bool(axis.get("categories"))


def replay_blocker(store: ProvStore, before: Generation) -> str | None:
    """Why this run's replayed nodes had nothing to read, or None when they did.

    Args:
        store: The run's store.
        before: The retired pass, read for what the original run recorded as having run.

    Returns:
        :data:`ADMIT_DID_NOT_ADMIT`, :data:`PREPROCESS_DID_NOT_COMPLETE`, or None.
    """
    names = {
        str(entity.attributes.get("name")) for entity in store.entities("stream") if not store.is_invalidated(entity.id)
    }
    if RECORDING_STREAM not in names:
        return ADMIT_DID_NOT_ADMIT
    ran = before.mapping("ran")
    if PREPROCESS_NODE in ran and str(ran[PREPROCESS_NODE]) != COMPLETED:
        return PREPROCESS_DID_NOT_COMPLETE
    return None


def diff_store(store: ProvStore) -> dict[str, Any]:
    """The differential over one replayed store.

    Args:
        store: The run's store, read under any run id.

    Returns:
        ``{status, ...}`` — :data:`OK` with the row body when the store carries a replay and both
        generations, :data:`NOT_REPLAYED` when it carries no marker, :data:`NOT_REPLAYABLE` with
        the blocker when the replayed nodes had nothing to read, :data:`NOTHING_RETIRED` when the
        replay found no decision to retire, and :data:`AMBIGUOUS` when more than one configuration
        has replayed it.
    """
    markers = replay_markers(store)
    if not markers:
        return {"status": NOT_REPLAYED}
    hashes = sorted({str(marker.get("config_hash")) for marker in markers})
    before_entities, after_entities = split_generations(store)
    before, after = generation(store, before_entities), generation(store, after_entities)
    body = diff_generations(before, after)
    body["config_hash"] = hashes[0] if len(hashes) == 1 else hashes
    body["commit"] = markers[-1].get("commit")
    blocker = replay_blocker(store, before)
    if blocker is not None:
        return {"status": NOT_REPLAYABLE, "blocker": blocker, **body}
    if len(hashes) > 1:
        return {"status": AMBIGUOUS, **body}
    if not before_entities:
        return {"status": NOTHING_RETIRED, **body}
    return {"status": OK, **body}


def diff_run(run_root: Path, *, stem: str) -> dict[str, Any]:
    """The differential over one replayed run root.

    Args:
        run_root: The replayed run root, holding ``run/store.jsonl``.
        stem: The recording's file stem, which is how a row is identified.

    Returns:
        The row, always carrying ``stem``, ``run_root`` and ``status``.
    """
    store_path = run_root / RUN_SUBDIR / STORE_FILE
    row: dict[str, Any] = {"stem": stem, "run_root": str(run_root)}
    if not store_path.is_file():
        return {**row, "status": NO_STORE}
    try:
        store = ProvStore.read_jsonl(store_path, run_id=run_root.name)
    except (OSError, ValueError) as error:
        return {**row, "status": UNREADABLE, "detail": f"{type(error).__name__}: {error}"}
    return {**row, **diff_store(store)}


@dataclass(frozen=True)
class ReplayDiff:
    """What a corpus of replayed stores moved, counted.

    Attributes:
        rows: How many rows were read.
        statuses: Row status to count.
        compared: How many rows carried both generations and were compared.
        identical: How many compared decisions did not move at all.
        not_replayable: The runs the replayed nodes had nothing to read, by blocker, with their
            stems. Excluded from every count below, because what moved on them is the re-entry
            point's doing and not a decision.
        new_keys: A decision field the replay added to how many recordings.
        dropped_keys: A decision field the replay dropped from how many recordings.
        triage: Triage transition to count, over every compared recording.
        release: Release transition to count.
        route_state: File route state transition to count.
        discard_ground: Discard ground transition to count.
        declared_family: Declared family transition to count.
        moved_stems: Field to transition to the stems that took it, for the transitions that moved.
        grounds: ``node|ground`` to how many recordings gained and lost it.
        routes: Branch to route transition to count, over the transitions that moved.
        findings: Branch to finding transition to count, over the transitions that moved.
        conformance: Node to conformance transition to count, over the transitions that moved.
        conformance_of: Node to referent transition to count, over the transitions that moved.
        agreement: Branch to agreement transition to count, over the transitions that moved.
        hints: Branch to hint-reading transition to count, over the transitions that moved.
        ran: Node to run-state transition to count, over the transitions that moved.
        deviations: Node to deviation type to how many recordings gained and lost it in the
            file-level decision.
        report_deviations: Node to deviation type to how many recordings gained and lost it in the
            branch's own report.
        deviation_assertions: Node to deviation type to the assertions written before and after and
            how many recordings gained and lost them.
        unmeasured: Node to config path to how many recordings gained and lost it.
        pii: The PII axis, counted: the scan, the findings and the two nodes' run states.
        pii_scanned_then_not: The recordings scanned before and not scanned now, and their stems.
        pii_not_scanned_then_now: The recordings not scanned before and scanned now, and their stems.
        pii_redact_dropped: The recordings REDACT ran on before and not now, and their stems.
        pii_redact_added: The recordings REDACT did not run on before and does now, and their stems.
    """

    rows: int = 0
    statuses: dict[str, int] = field(default_factory=dict)
    compared: int = 0
    identical: int = 0
    not_replayable: dict[str, Any] = field(default_factory=dict)
    new_keys: dict[str, int] = field(default_factory=dict)
    dropped_keys: dict[str, int] = field(default_factory=dict)
    triage: dict[str, int] = field(default_factory=dict)
    release: dict[str, int] = field(default_factory=dict)
    route_state: dict[str, int] = field(default_factory=dict)
    discard_ground: dict[str, int] = field(default_factory=dict)
    declared_family: dict[str, int] = field(default_factory=dict)
    moved_stems: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    grounds: dict[str, dict[str, int]] = field(default_factory=dict)
    routes: dict[str, dict[str, int]] = field(default_factory=dict)
    findings: dict[str, dict[str, int]] = field(default_factory=dict)
    conformance: dict[str, dict[str, int]] = field(default_factory=dict)
    conformance_of: dict[str, dict[str, int]] = field(default_factory=dict)
    agreement: dict[str, dict[str, int]] = field(default_factory=dict)
    hints: dict[str, dict[str, int]] = field(default_factory=dict)
    ran: dict[str, dict[str, int]] = field(default_factory=dict)
    deviations: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)
    report_deviations: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)
    deviation_assertions: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)
    unmeasured: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)
    pii: dict[str, dict[str, int]] = field(default_factory=dict)
    pii_scanned_then_not: dict[str, Any] = field(default_factory=dict)
    pii_not_scanned_then_now: dict[str, Any] = field(default_factory=dict)
    pii_redact_dropped: dict[str, Any] = field(default_factory=dict)
    pii_redact_added: dict[str, Any] = field(default_factory=dict)


def read_rows(root: Path) -> Iterator[dict[str, Any]]:
    """Read every differential row under a tree of slice logs.

    Args:
        root: A directory holding ``*.jsonl`` slice logs at any depth.

    Yields:
        Each row, in file then line order. A line that will not parse is skipped.
    """
    for path in sorted(root.rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                yield row


class _Tally:
    """The mutable counters :func:`aggregate` fills, kept apart from the report it returns."""

    def __init__(self) -> None:
        """Start every counter empty."""
        self.statuses: Counter[str] = Counter()
        self.new_keys: Counter[str] = Counter()
        self.dropped_keys: Counter[str] = Counter()
        self.scalars: dict[str, Counter[str]] = {key: Counter() for key in _TRANSITION_KEYS}
        self.moved_stems: dict[str, dict[str, list[str]]] = {}
        self.grounds: dict[str, Counter[str]] = {}
        self.mappings: dict[str, dict[str, Counter[str]]] = {key: {} for key in _MAPPING_KEYS}
        self.sets: dict[str, dict[str, dict[str, Counter[str]]]] = {key: {} for key in _SET_KEYS}
        self.reports: dict[str, dict[str, Counter[str]]] = {}
        self.assertions: dict[str, dict[str, Counter[str]]] = {}
        self.pii: dict[str, Counter[str]] = {}
        self.pii_stems: dict[str, list[str]] = {
            "scanned_then_not": [],
            "not_scanned_then_now": [],
            "redact_dropped": [],
            "redact_added": [],
        }

    def scalar(self, row: Mapping[str, Any], stem: str) -> None:
        """Count one row's scalar transitions and remember the stems that moved.

        Args:
            row: The row.
            stem: The recording's stem.
        """
        for key, move in (row.get("transitions") or {}).items():
            if key not in self.scalars:
                continue
            self.scalars[key][str(move)] += 1
            head, _, tail = str(move).partition("->")
            if head != tail:
                self.moved_stems.setdefault(key, {}).setdefault(str(move), []).append(stem)

    def mapping(self, row: Mapping[str, Any]) -> None:
        """Count one row's node- and branch-keyed transitions.

        Args:
            row: The row.
        """
        moved = row.get("moved") or {}
        for key in _MAPPING_KEYS:
            for name, move in (moved.get(key) or {}).items():
                self.mappings[key].setdefault(str(name), Counter())[str(move)] += 1
        for key in _SET_KEYS:
            for name, change in (moved.get(key) or {}).items():
                _fold_change(self.sets[key].setdefault(str(name), {}), change)

    def rest(self, row: Mapping[str, Any], stem: str) -> None:
        """Count one row's grounds, branch reports, typed findings and PII axis.

        Args:
            row: The row.
            stem: The recording's stem.
        """
        for direction, names in (row.get("grounds") or {}).items():
            for name in names:
                self.grounds.setdefault(str(name), Counter())[str(direction)] += 1
        for node, change in (row.get("report_deviations") or {}).items():
            _fold_change(self.reports.setdefault(str(node), {}), change)
        for node, types in (row.get("deviation_assertions") or {}).items():
            for name, pair in types.items():
                counts = self.assertions.setdefault(str(node), {}).setdefault(str(name), Counter())
                counts[BEFORE] += int(pair[0])
                counts[AFTER] += int(pair[1])
                counts["gained" if int(pair[1]) > int(pair[0]) else "lost"] += 1
        self.pii_axis(row.get("pii") or {}, stem)

    def pii_axis(self, axis: Mapping[str, Any], stem: str) -> None:
        """Count one row's PII axis, naming the recordings whose scan or redaction moved.

        Args:
            axis: What :func:`_pii_axis` built.
            stem: The recording's stem.
        """
        for key, value in axis.items():
            if key == "categories" or not (isinstance(value, list) and len(value) == 2):
                continue
            self.pii.setdefault(str(key), Counter())[_transition(value[0], value[1])] += 1
        scanned = axis.get(SCANNED_KEY) or [None, None]
        if scanned[0] is True and scanned[1] is not True:
            self.pii_stems["scanned_then_not"].append(stem)
        if scanned[0] is not True and scanned[1] is True:
            self.pii_stems["not_scanned_then_now"].append(stem)
        redact = axis.get("redact_ran") or [ABSENT, ABSENT]
        if redact[0] == "completed" and redact[1] != "completed":
            self.pii_stems["redact_dropped"].append(stem)
        if redact[0] != "completed" and redact[1] == "completed":
            self.pii_stems["redact_added"].append(stem)


def _fold_change(into: dict[str, Counter[str]], change: Mapping[str, Any]) -> None:
    """Fold one row's gained and lost names into a name-keyed tally.

    Args:
        into: Name to its ``gained``/``lost`` counter.
        change: What :func:`_set_change` produced.
    """
    for direction, names in change.items():
        for name in names:
            into.setdefault(str(name), Counter())[str(direction)] += 1


def _ordered(counters: Mapping[str, Counter[str]]) -> dict[str, dict[str, int]]:
    """Order a two-level count table, outer key then descending count.

    Args:
        counters: Outer key to its counter.

    Returns:
        The same table as plain dicts, each inner one most-frequent first.
    """
    return {key: dict(counters[key].most_common()) for key in sorted(counters)}


def _ordered_nested(table: Mapping[str, Mapping[str, Counter[str]]]) -> dict[str, dict[str, dict[str, int]]]:
    """Order a three-level count table.

    Args:
        table: Outer key to inner key to its counter.

    Returns:
        The same table as plain dicts.
    """
    return {key: _ordered(table[key]) for key in sorted(table)}


def aggregate(rows: Iterable[Mapping[str, Any]]) -> ReplayDiff:
    """Count a corpus of differential rows.

    Args:
        rows: What :func:`read_rows` yields.

    Returns:
        The corpus differential.
    """
    tally = _Tally()
    total = compared = identical = 0
    blockers: dict[str, list[str]] = {}
    for row in rows:
        total += 1
        status = str(row.get("status"))
        tally.statuses[status] += 1
        stem = str(row.get("stem") or row.get("run_root") or "")
        if status == NOT_REPLAYABLE:
            blockers.setdefault(str(row.get("blocker")), []).append(stem)
            continue
        if status not in (OK, NOTHING_RETIRED, AMBIGUOUS):
            continue
        compared += 1
        if row.get("identical"):
            identical += 1
        tally.new_keys.update(str(key) for key in row.get("new_keys") or [])
        tally.dropped_keys.update(str(key) for key in row.get("dropped_keys") or [])
        tally.scalar(row, stem)
        tally.mapping(row)
        tally.rest(row, stem)
    return ReplayDiff(
        rows=total,
        statuses=dict(tally.statuses.most_common()),
        compared=compared,
        identical=identical,
        not_replayable={
            "count": sum(len(stems) for stems in blockers.values()),
            "by_blocker": {blocker: len(stems) for blocker, stems in sorted(blockers.items())},
            "stems": sorted(stem for stems in blockers.values() for stem in stems),
        },
        new_keys=dict(tally.new_keys.most_common()),
        dropped_keys=dict(tally.dropped_keys.most_common()),
        triage=dict(tally.scalars["triage"].most_common()),
        release=dict(tally.scalars["release"].most_common()),
        route_state=dict(tally.scalars["route_state"].most_common()),
        discard_ground=dict(tally.scalars["discard_ground"].most_common()),
        declared_family=dict(tally.scalars["declared_family"].most_common()),
        moved_stems={
            key: {move: sorted(stems) for move, stems in sorted(moves.items())}
            for key, moves in sorted(tally.moved_stems.items())
        },
        grounds=_ordered(tally.grounds),
        routes=_ordered(tally.mappings["routes"]),
        findings=_ordered(tally.mappings["findings"]),
        conformance=_ordered(tally.mappings["conformance"]),
        conformance_of=_ordered(tally.mappings["conformance_of"]),
        agreement=_ordered(tally.mappings["agreement"]),
        hints=_ordered(tally.mappings["hints"]),
        ran=_ordered(tally.mappings["ran"]),
        deviations=_ordered_nested(tally.sets["deviations"]),
        report_deviations=_ordered_nested(tally.reports),
        deviation_assertions=_ordered_nested(tally.assertions),
        unmeasured=_ordered_nested(tally.sets["unmeasured"]),
        pii=_ordered(tally.pii),
        pii_scanned_then_not=_stems("scanned before, not scanned now", tally.pii_stems["scanned_then_not"]),
        pii_not_scanned_then_now=_stems("not scanned before, scanned now", tally.pii_stems["not_scanned_then_now"]),
        pii_redact_dropped=_stems("REDACT ran before, not now", tally.pii_stems["redact_dropped"]),
        pii_redact_added=_stems("REDACT did not run before, runs now", tally.pii_stems["redact_added"]),
    )


def _stems(what: str, stems: Sequence[str]) -> dict[str, Any]:
    """One named set of recordings, counted and listed.

    Args:
        what: What the set is.
        stems: The recordings' stems.

    Returns:
        ``{what, count, stems}``, the stems sorted.
    """
    return {"what": what, "count": len(stems), "stems": sorted(stems)}


STEM_CAP = 25
"""How many stems one Markdown row names before it says how many more there are."""


def _share(count: int, total: int) -> str:
    """One count as a share of a total.

    Args:
        count: The count.
        total: The denominator.

    Returns:
        The share, or ``--`` when there is no denominator.
    """
    return f"{100.0 * count / total:.2f}%" if total else "--"


def _matrix(title: str, counts: Mapping[str, int], total: int, moved: Mapping[str, Sequence[str]]) -> list[str]:
    """Render one transition matrix, marking the transitions that moved.

    Args:
        title: The table's heading.
        counts: Transition to count.
        total: The denominator every share is taken against.
        moved: Transition to the stems that took it.

    Returns:
        The table's lines, empty when there is nothing to count.
    """
    if not counts:
        return []
    lines = [f"### {title}", "", "| before | after | files | share | moved |", "|---|---|---:|---:|:--:|"]
    for move, count in counts.items():
        before, _, after = move.partition("->")
        mark = "yes" if move in moved else ""
        lines.append(f"| `{before}` | `{after}` | {count} | {_share(count, total)} | {mark} |")
    return [*lines, ""]


def _gained_lost(title: str, table: Mapping[str, Mapping[str, int]], total: int) -> list[str]:
    """Render one gained-and-lost table.

    Args:
        title: The section's heading.
        table: Name to its ``gained``/``lost`` counts.
        total: The denominator every share is taken against.

    Returns:
        The section's lines, empty when there is nothing to count.
    """
    if not table:
        return []
    lines = [f"### {title}", "", "| name | gained | lost | net | share gained |", "|---|---:|---:|---:|---:|"]
    for name, counts in sorted(table.items(), key=lambda item: -(item[1].get("gained", 0) + item[1].get("lost", 0))):
        gained, lost = counts.get("gained", 0), counts.get("lost", 0)
        lines.append(f"| `{name}` | {gained} | {lost} | {gained - lost:+d} | {_share(gained, total)} |")
    return [*lines, ""]


def _nested_transitions(title: str, table: Mapping[str, Mapping[str, int]], total: int) -> list[str]:
    """Render one key-to-transition table.

    Args:
        title: The section's heading.
        table: Key to transition to count.
        total: The denominator every share is taken against.

    Returns:
        The section's lines, empty when there is nothing to count.
    """
    if not table:
        return []
    lines = [f"### {title}", "", "| key | before | after | files | share |", "|---|---|---|---:|---:|"]
    for key, counts in table.items():
        for move, count in counts.items():
            before, _, after = move.partition("->")
            lines.append(f"| `{key}` | `{before}` | `{after}` | {count} | {_share(count, total)} |")
    return [*lines, ""]


def _nested_recordings(title: str, table: Mapping[str, Mapping[str, Mapping[str, int]]], total: int) -> list[str]:
    """Render one node-keyed table of how many recordings gained and lost a name.

    Args:
        title: The section's heading.
        table: Node to name to its ``gained``/``lost`` counts.
        total: The denominator every share is taken against.

    Returns:
        The section's lines, empty when there is nothing to count.
    """
    if not table:
        return []
    lines = [f"### {title}", "", "| node | name | gained | lost | net |", "|---|---|---:|---:|---:|"]
    for node, names in table.items():
        for name, counts in names.items():
            gained, lost = counts.get("gained", 0), counts.get("lost", 0)
            lines.append(f"| `{node}` | `{name}` | {gained} | {lost} | {gained - lost:+d} |")
    return [*lines, ""]


def _nested_totals(title: str, table: Mapping[str, Mapping[str, Mapping[str, int]]], total: int) -> list[str]:
    """Render one node-keyed table of what each pass wrote and how many recordings moved.

    Args:
        title: The section's heading.
        table: Node to name to its counts.
        total: The denominator every share is taken against.

    Returns:
        The section's lines, empty when there is nothing to count.
    """
    if not table:
        return []
    header = "| node | name | before | after | rose on | fell on |"
    lines = [f"### {title}", "", header, "|---|---|---:|---:|---:|---:|"]
    for node, names in table.items():
        for name, counts in names.items():
            lines.append(
                f"| `{node}` | `{name}` | {counts.get(BEFORE, 0)} | {counts.get(AFTER, 0)} | "
                f"{counts.get('gained', 0)} | {counts.get('lost', 0)} |"
            )
    return [*lines, ""]


def _not_replayable_lines(block: Mapping[str, Any], rows: int) -> list[str]:
    """Render the runs excluded from the comparison, and why each was.

    Args:
        block: What :func:`aggregate` counted under ``not_replayable``.
        rows: How many rows were read, the denominator the share is taken against.

    Returns:
        The section's lines, empty when every run was replayable.
    """
    count = int(block.get("count", 0))
    if not count:
        return []
    lines = [
        "### Not replayable, and excluded from every count below",
        "",
        f"**{count} runs** ({_share(count, rows)}) had nothing for the replayed nodes to read, so "
        "the nodes the original run skipped were called and errored. What moved on them is the "
        "re-entry point's doing and is not a decision; none of it is counted anywhere else in this "
        "report.",
        "",
        "| blocker | runs |",
        "|---|---:|",
    ]
    for blocker, number in (block.get("by_blocker") or {}).items():
        lines.append(f"| {blocker} | {number} |")
    stems = [str(stem) for stem in block.get("stems") or []]
    shown = ", ".join(f"`{stem}`" for stem in stems[:STEM_CAP])
    more = f" and {len(stems) - STEM_CAP} more" if len(stems) > STEM_CAP else ""
    return [*lines, "", f"They are {shown}{more}.", ""]


def _stem_lines(block: Mapping[str, Any], total: int) -> list[str]:
    """Render one named set of recordings, capped.

    Args:
        block: What :func:`_stems` produced.
        total: The denominator the share is taken against.

    Returns:
        The lines.
    """
    count = int(block.get("count", 0))
    stems = [str(stem) for stem in block.get("stems") or []]
    shown = ", ".join(f"`{stem}`" for stem in stems[:STEM_CAP])
    more = f" and {len(stems) - STEM_CAP} more" if len(stems) > STEM_CAP else ""
    tail = f" — {shown}{more}" if stems else ""
    return [f"- **{block.get('what')}: {count}** ({_share(count, total)}){tail}", ""]


def render_markdown(report: ReplayDiff, source: Path | str) -> str:
    """Render the corpus differential as a report.

    Args:
        report: What :func:`aggregate` counted.
        source: The tree the rows were read from, named in the heading.

    Returns:
        The report.
    """
    total = report.compared
    moved = report.moved_stems
    excluded = int(report.not_replayable.get("count", 0))
    lines = [
        f"# What the replay changed — {source}",
        "",
        f"**{report.rows} rows read, {total} compared"
        + (f", {excluded} not replayable and excluded" if excluded else "")
        + ".** "
        f"**{report.identical} decisions did not move at all** ({_share(report.identical, total)}); "
        f"{total - report.identical} moved on at least one axis.",
        "",
        "Every value below is categorical or a count, as the decisions it is read from are. A stem "
        "identifies a recording; nothing here carries transcript text or a detected string.",
        "",
        "---",
        "",
        "## Was a decision read at all",
        "",
    ]
    lines += _table("Row status", report.statuses, report.rows)
    lines += _not_replayable_lines(report.not_replayable, report.rows)
    lines += _table("Decision fields the replay added", report.new_keys, total)
    lines += _table("Decision fields the replay dropped", report.dropped_keys, total)
    lines += ["## Did the file-level decision move", ""]
    lines += _matrix("Triage", report.triage, total, moved.get("triage", {}))
    lines += _matrix("Release", report.release, total, moved.get("release", {}))
    lines += _matrix("File route state", report.route_state, total, moved.get("route_state", {}))
    lines += _matrix("Discard ground", report.discard_ground, total, moved.get("discard_ground", {}))
    lines += _matrix("Declared family", report.declared_family, total, moved.get("declared_family", {}))
    lines += ["## Which flag grounds appeared and disappeared", ""]
    lines += _gained_lost("Grounds, by `node|ground`", report.grounds, total)
    lines += ["## Per node, what moved", ""]
    lines += _nested_transitions("Branch route", report.routes, total)
    lines += _nested_transitions("Branch finding", report.findings, total)
    lines += _nested_transitions("Conformance", report.conformance, total)
    lines += _nested_transitions("Conformance is about", report.conformance_of, total)
    lines += _nested_transitions("Agreement", report.agreement, total)
    lines += _nested_transitions("Hint reading", report.hints, total)
    lines += _nested_transitions("Node run state", report.ran, total)
    lines += ["## Typed findings", ""]
    lines += _nested_recordings("Deviation types in the file decision", report.deviations, total)
    lines += _nested_recordings("Deviation types in the branch's own report", report.report_deviations, total)
    lines += _nested_totals("Deviation assertions written", report.deviation_assertions, total)
    lines += _nested_recordings("Unmeasured config paths", report.unmeasured, total)
    lines += ["## What the PII change did", ""]
    lines += _stem_lines(report.pii_scanned_then_not, total)
    lines += [
        "  A recording that was scanned and is not scanned now is the direction in which a mistake "
        "leaks, so it is counted first and named.",
        "",
    ]
    lines += _stem_lines(report.pii_not_scanned_then_now, total)
    lines += _stem_lines(report.pii_redact_dropped, total)
    lines += _stem_lines(report.pii_redact_added, total)
    lines += _nested_transitions("PII axis", report.pii, total)
    return "\n".join(lines).rstrip() + "\n"


def _table(title: str, counts: Mapping[str, int], total: int) -> list[str]:
    """Render one count table with its share.

    Args:
        title: The table's heading.
        counts: Value to count.
        total: The denominator every share is taken against.

    Returns:
        The table's lines, empty when there is nothing to count.
    """
    if not counts:
        return []
    lines = [f"### {title}", "", "| value | files | share |", "|---|---:|---:|"]
    for value, count in counts.items():
        lines.append(f"| `{value}` | {count} | {_share(count, total)} |")
    return [*lines, ""]


def write_report(report: ReplayDiff, out: Path, source: Path | str) -> tuple[Path, Path]:
    """Write the differential as machine-readable JSON and a rendered report.

    Args:
        report: What :func:`aggregate` counted.
        out: The directory both products go in, created if absent.
        source: The tree the rows were read from, named in the heading.

    Returns:
        The JSON's path and the Markdown's.
    """
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "replay_diff.json"
    markdown_path = out / "replay_diff.md"
    json_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report, source), encoding="utf-8")
    return json_path, markdown_path
