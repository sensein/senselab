"""The VERDICT node: the store's contents read into the vocabulary's fold, and the result recorded.

This is the graph's only decision about the recording. The branches and QUALITY write
``branch_report`` entities — typed deviations, no outcome and no conformance — and propose spans;
the deciding nodes write ``verdict`` entities; this node reads all three, adds the declared task and
the routing decisions, and hands them to ``vocabulary.fold_file_verdict``. REDACT's optional LLM
re-read is read here too, as an annotation: its ``flagged`` reaches the triage axis under
``verdict.llm_redaction_flags`` and the release axis on no path.

**The release axis is decided here too, from the evidence rather than from whether REDACT ran.**
REDACT runs only where SPEECH's scan found something, so its silence is the ordinary case;
:func:`_redaction_evidence` gathers SPEECH's lexical count, the ``pii_scan`` tri-state and the live
findings, and the fold's table turns them into a state and a ground.

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

import functools
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage.background_model import BACKGROUND_MODEL
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.ddk_task import CYCLE, DDK_READING, SYLLABLE
from senselab.audio.workflows.triage.live_evidence import declared_task, recording_stem
from senselab.audio.workflows.triage.nodes.airway import EVENTS_FOUND as AIRWAY_EVENTS_FOUND
from senselab.audio.workflows.triage.nodes.airway import INSTRUMENT_ABSENT as AIRWAY_INSTRUMENT_ABSENT
from senselab.audio.workflows.triage.nodes.airway_task import (
    BREATH_READING,
    COUGH_READING,
    required_event,
)
from senselab.audio.workflows.triage.nodes.background import BACKGROUND_NODE, SESSION_FLOOR, SESSION_NODE
from senselab.audio.workflows.triage.nodes.branches import (
    BRANCH_FAMILY,
    EXPECTATIONS,
    Expectation,
    Pattern,
    declared_task_family,
)
from senselab.audio.workflows.triage.nodes.common import (
    REMINT,
    NodeResult,
    consensus_words,
    find_measurement,
    find_measurements,
    find_verdict,
    lexical_words,
    mint_live,
    software_agent,
    write_verdict,
)
from senselab.audio.workflows.triage.nodes.gates import (
    AT_LEAST,
    DEFAULT_LAYER,
    FAIL,
    FLAG_GATES,
    GATE_SECTION,
    GATE_SPECS,
    INAPPLICABLE,
    INSTRUCTED_COUNT_FRACTION,
    INSTRUMENT_ABSENT,
    NO_CARRIER,
    NO_INSTRUCTED_COUNT,
    NO_OWNER_REPORT,
    NO_SPEECH,
    NOT_COMPUTED,
    NULL_NO_OVERLAP,
    NULL_VALUE,
    UNCOMPUTED_REASONS,
    UNDECIDED,
    AppliedGate,
    GateBounds,
    apply_flag_gates,
    apply_gates,
    conformance_gate_names,
    load_gate_bounds,
)
from senselab.audio.workflows.triage.nodes.redact import (
    NEW,
    UNMASKED_BY_REVIEWER,
    MaskPlan,
    mask_plan,
    name_approvals,
    padding_ms,
    task_texts,
)
from senselab.audio.workflows.triage.nodes.voice import PHONATION_READING
from senselab.audio.workflows.triage.quality_join import QUALITY_JOIN
from senselab.audio.workflows.triage.routing_analysis.families import SYLLABLE_REPETITION
from senselab.audio.workflows.triage.task_lexicon import task_lexicon
from senselab.audio.workflows.triage.vocabulary import (
    BREATH_COUNTED,
    BREATH_SUSTAINED,
    COUGH_COUNTED,
    COUGH_PERFORMED,
    GRAPH_ORDER,
    KEY_NODE_OUTCOME_UNREADABLE,
    PII_SCAN,
    REDACTION_LLM_ANNOTATION,
    RULESET_ROUTING,
    SCANNED,
    SECOND_OPINION_ANSWERS,
    SUPERSEDES,
    TASK,
    TASK_EXTENT_SPAN_ROLE,
    UNDETERMINED,
    BranchDecision,
    BranchReport,
    Conformance,
    FileVerdict,
    FoldPolicy,
    NodeVerdict,
    Outcome,
    RedactionEvidence,
    RunState,
    TaskEvidence,
    fold_file_verdict,
    release_value,
    reviewer_may_unmask,
)
from senselab.utils.prov_store import PROV_TYPE, Entity, ProvStore

NODE = "VERDICT"

SPEECH = "SPEECH"
"""The branch whose transcript is the only thing a redaction can read."""

_GRAPH_ORDER = GRAPH_ORDER[:-1]

_REVIEW = "REVIEW"
_REDACT_NODE = "REDACT"


@dataclass(frozen=True)
class VerdictResult(NodeResult):
    """What VERDICT returns.

    Attributes:
        file_verdict: The graph's conclusion about the recording, on both axes.
        ledger_entity_id: The ``pii_ledger`` measurement written beside the verdict.
    """

    file_verdict: FileVerdict
    ledger_entity_id: str


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
            key=KEY_NODE_OUTCOME_UNREADABLE,
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


TASK_MINIMUM_DURATION_PATH = Path(__file__).parents[1] / "data" / "task_minimum_duration.yaml"
AIRWAY_EVENT_TOKENS_PATH = Path(__file__).parents[1] / "data" / "airway_event_tokens.yaml"
DISCARD_CONTESTED_PATH = Path(__file__).parents[1] / "data" / "discard_contested.yaml"


@functools.cache
def _task_minimum_durations() -> tuple[float, dict[str, float]]:
    """``data/task_minimum_duration.yaml``: the default minimum and the per-family ones, in seconds."""
    document = yaml.safe_load(TASK_MINIMUM_DURATION_PATH.read_text()) or {}
    families = {str(name): float(value) for name, value in (document.get("families") or {}).items()}
    return float(document["default_s"]), families


@functools.cache
def _discard_contested() -> dict[str, Any]:
    """``data/discard_contested.yaml``, as written."""
    return yaml.safe_load(DISCARD_CONTESTED_PATH.read_text()) or {}


def contest_events_min(instructed: int | None, event: str = "cough") -> int:
    """The detector events that contest a measure's no-event discard of a family.

    Args:
        instructed: The family's instructed count, or None where its instruction names none.
        event: The family's required event, ``breath`` or ``cough``.

    Returns:
        The instructed count times ``instructed_fraction``, rounded up; where no count is
        instructed, the breath section's ``uncounted_events_min`` for a breath family and the
        top-level one otherwise.
    """
    document = _discard_contested()
    if instructed is not None:
        return math.ceil(instructed * float(document["instructed_fraction"]))
    section = (document.get("breath") or {}) if event == "breath" else {}
    return int(section.get("uncounted_events_min", document["uncounted_events_min"]))


def contest_rise_db_min(event: str) -> float | None:
    """The breath-train rise a breath family's contested discard needs, or None for any other event.

    Args:
        event: The family's required event.

    Returns:
        ``breath.train_rise_db_min`` for a breath family; None otherwise.
    """
    if event != "breath":
        return None
    return float((_discard_contested().get("breath") or {})["train_rise_db_min"])


def minimum_duration_s(declared_family: str | None) -> float | None:
    """The shortest recording a declared family can occupy, or None where nothing is declared.

    Args:
        declared_family: The task family the recording declares.

    Returns:
        The family's own minimum, else the profile's default; None where no family is declared.
    """
    if not declared_family:
        return None
    default, families = _task_minimum_durations()
    return families.get(declared_family, default)


@functools.cache
def _airway_event_tokens() -> dict[str, frozenset[str]]:
    """``data/airway_event_tokens.yaml``: per declared airway family, the event tokens naming its event."""
    document = yaml.safe_load(AIRWAY_EVENT_TOKENS_PATH.read_text()) or {}
    return {
        str(family): frozenset(str(token).casefold() for token in document.get(kind) or ())
        for family, kind in (document.get("families") or {}).items()
    }


def event_tokens_n(store: ProvStore, declared_family: str | None) -> int:
    """How many bracketed consensus tokens name the declared airway family's own event.

    Args:
        store: The provenance store, read for its consensus words.
        declared_family: The task family the recording declares.

    Returns:
        The count of ``[cough]``-like tokens for a cough family, ``[breath]``-like for a breath
        family, and 0 for every other family or where no consensus exists.
    """
    tokens = _airway_event_tokens().get(declared_family or "")
    if not tokens:
        return 0
    return sum(
        1
        for word in consensus_words(store)
        if word.attributes.get("bracketed")
        and str(word.attributes.get("text") or "").strip().strip("[]()<>").strip().casefold() in tokens
    )


def _airway_events_found(store: ProvStore) -> int | None:
    """How many events of its own kind the owning AIRWAY branch detected, or None where it reported no count.

    Args:
        store: The provenance store, read for AIRWAY's events-found measurement.

    Returns:
        The count, or None.
    """
    measurement = find_measurement(store, AIRWAY_EVENTS_FOUND)
    if measurement is None:
        return None
    value = measurement.attributes.get("value")
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _owner_absent_inputs(
    store: ProvStore, owners: Sequence[str], gate_record: Mapping[str, Any] | None
) -> tuple[str, ...]:
    """What an owning branch needed to look for its task and did not have.

    Args:
        store: The provenance store, read for AIRWAY's instrument-absent measurement and ROUTING's
            critical absences.
        owners: The branches owning the declared family.
        gate_record: The gate outcome record, read for the owning node's uncomputed readings.

    Returns:
        The absent inputs, each named as its branch, gate or derivative, in a stable order.
    """
    absent: list[str] = []
    if "AIRWAY" in owners:
        measurement = find_measurement(store, AIRWAY_INSTRUMENT_ABSENT)
        if measurement is not None:
            absent.extend(f"AIRWAY:{name}" for name in measurement.attributes.get("absent") or ())
    absences = _critical_absences(store)
    for branch in owners:
        absent.extend(f"{branch}:{gate}" for gate in sorted(absences.get(branch, {})))
    record = dict(gate_record or {})
    if record.get("node") in owners:
        absent.extend(
            f"{record['node']}:{gate.get('gate')}"
            for gate in record.get("applied") or ()
            if isinstance(gate, Mapping)
            and gate.get("passed") == UNDETERMINED
            and gate.get("reason") in UNCOMPUTED_REASONS
        )
    return tuple(dict.fromkeys(absent))


DECISION_INPUTS: dict[str, tuple[str, ...]] = {
    BREATH_READING: ("decision",),
    COUGH_READING: ("onsets_n", "review"),
    PHONATION_READING: ("found",),
    DDK_READING: ("decision",),
}
"""Per task reading, the attributes the fold decides on; a reading lacking one is not measured."""


def _undecidable(name: str, node: str, attributes: Mapping[str, Any]) -> tuple[str, ...]:
    """The decision inputs a stored task reading lacks, named for the owner's absent inputs.

    Args:
        name: The reading's measurement name.
        node: The node that wrote it.
        attributes: The measurement's attributes.

    Returns:
        ``<node>:<name>.<field>`` for each field of :data:`DECISION_INPUTS` that is missing or None.
    """
    return tuple(f"{node}:{name}.{field}" for field in DECISION_INPUTS.get(name, ()) if attributes.get(field) is None)


def _airway_reading(store: ProvStore, name: str) -> tuple[dict[str, Any] | None, tuple[str, ...]]:
    """AIRWAY's task reading and the inputs it lacked, as AIRWAY wrote them.

    Args:
        store: The provenance store.
        name: The reading's measurement name.

    Returns:
        The measurement's attributes and ``()``; ``None`` and the absent inputs where AIRWAY lacked
        them or the reading lacks a :data:`DECISION_INPUTS` field; ``None`` and the measurement's own
        name where AIRWAY reported and wrote no reading;
        ``None`` and ``()`` where AIRWAY did not report at all.
    """
    measurement = find_measurement(store, name)
    if measurement is None:
        reported = any(report.node == "AIRWAY" for _, report in _branch_reports(store))
        return None, ((f"AIRWAY:{name}",) if reported else ())
    absent = tuple(str(each) for each in measurement.attributes.get("absent") or ())
    absent = absent or _undecidable(name, "AIRWAY", measurement.attributes)
    return (None, absent) if absent else (dict(measurement.attributes), ())


def _train_rise_db(reading: dict[str, Any] | None) -> float | None:
    """The breath train's median burst rise from AIRWAY's breath reading, or None where none was read."""
    train = ((reading or {}).get("reading") or {}).get("train") or {}
    rise = train.get("rise_db")
    return float(rise) if rise is not None else None


def _voice_reading(store: ProvStore) -> tuple[dict[str, Any] | None, tuple[str, ...]]:
    """VOICE's phonation reading and the inputs it lacked, as VOICE wrote them.

    Args:
        store: The provenance store.

    Returns:
        The measurement's attributes and ``()``; ``None`` and the absent inputs where VOICE lacked
        them; ``None`` and the measurement's own name where VOICE reported and wrote no reading;
        ``None`` and ``()`` where VOICE did not report at all.
    """
    measurement = find_measurement(store, PHONATION_READING)
    if measurement is None:
        reported = any(report.node == "VOICE" for _, report in _branch_reports(store))
        return None, ((f"VOICE:{PHONATION_READING}",) if reported else ())
    absent = tuple(f"VOICE:{each}" for each in measurement.attributes.get("absent") or ())
    absent = absent or _undecidable(PHONATION_READING, "VOICE", measurement.attributes)
    return (None, absent) if absent else (dict(measurement.attributes), ())


def _speech_ddk_reading(store: ProvStore) -> tuple[dict[str, Any] | None, tuple[str, ...]]:
    """SPEECH's syllable-task reading and the inputs it lacked, as SPEECH wrote them.

    Args:
        store: The provenance store.

    Returns:
        The measurement's attributes and ``()``; ``None`` and the absent inputs where SPEECH lacked
        them; ``None`` and the measurement's own name where SPEECH reported and wrote no reading;
        ``None`` and ``()`` where SPEECH did not report at all.
    """
    measurement = find_measurement(store, DDK_READING)
    if measurement is None:
        reported = any(report.node == "SPEECH" for _, report in _branch_reports(store))
        return None, ((f"SPEECH:{DDK_READING}",) if reported else ())
    absent = tuple(f"SPEECH:{each}" for each in measurement.attributes.get("absent") or ())
    absent = absent or _undecidable(DDK_READING, "SPEECH", measurement.attributes)
    return (None, absent) if absent else (dict(measurement.attributes), ())


def _task_evidence(
    store: ProvStore,
    declared_family: str | None,
    gate_record: Mapping[str, Any] | None = None,
) -> TaskEvidence:
    """Whether the declared task was performed at all: its owner, the duration and its event tokens.

    Args:
        store: The provenance store, read for ADMIT's ``recording`` stream, the consensus words and
            AIRWAY's task readings.
        declared_family: The task family the recording declares.
        gate_record: The gate outcome record, read for the owning node's uncomputed readings.

    Returns:
        The evidence :func:`~senselab.audio.workflows.triage.vocabulary.fold_file_verdict` reads. A
        breath family of ``data/airway_event_requirements.yaml`` is decided on AIRWAY's breathing-
        pattern reading and a cough family on its cough-onset reading; the reading's absent inputs
        then stand for the owner's, and a reading AIRWAY never wrote is itself an absent input.
    """
    owners = tuple(branch for branch, rows in EXPECTATIONS.items() if declared_family and declared_family in rows)
    recording = next(
        (
            stream
            for stream in store.entities("stream")
            if stream.attributes.get("name") == "recording" and not store.is_invalidated(stream.id)
        ),
        None,
    )
    duration = recording.extent[1] - recording.extent[0] if recording is not None and recording.extent else None
    airway = EXPECTATIONS["AIRWAY"].get(declared_family or "") if "AIRWAY" in owners else None
    needed = required_event(declared_family) if "AIRWAY" in owners else None
    absent = _owner_absent_inputs(store, owners, gate_record)
    breath_mode: str | None = None
    reading: dict[str, Any] | None = None
    if needed == "breath" and airway is not None:
        breath_mode = BREATH_SUSTAINED if airway.pattern == Pattern.SOUND_COVERAGE else BREATH_COUNTED
        reading, missing = _airway_reading(store, BREATH_READING)
        absent = missing if reading is not None or missing else absent
    cough_mode: str | None = None
    cough: dict[str, Any] | None = None
    if needed == "cough" and airway is not None:
        cough_mode = COUGH_PERFORMED if airway.required_count is None else COUGH_COUNTED
        cough, missing = _airway_reading(store, COUGH_READING)
        absent = missing if cough is not None or missing else absent
    voice = EXPECTATIONS["VOICE"].get(declared_family or "") if "VOICE" in owners else None
    voice_mode: str | None = None
    phonation: dict[str, Any] | None = None
    if voice is not None:
        voice_mode = "glide" if voice.pattern == Pattern.GLIDE else "sustained"
        phonation, missing = _voice_reading(store)
        absent = missing if phonation is not None or missing else absent
    speech = EXPECTATIONS["SPEECH"].get(declared_family or "") if "SPEECH" in owners else None
    ddk_mode: str | None = None
    ddk: dict[str, Any] | None = None
    if speech is not None and declared_family in SYLLABLE_REPETITION:
        ddk_mode = CYCLE if speech.pattern == Pattern.SYLLABLE_SEQUENCE else SYLLABLE
        ddk, missing = _speech_ddk_reading(store)
        absent = missing if ddk is not None or missing else absent
    instructed = airway.required_count.value if airway is not None and airway.required_count is not None else None
    joined = find_measurement(store, QUALITY_JOIN)
    quality = (
        {k: v for k, v in joined.attributes.items() if k not in ("name", "signal")}
        if joined is not None
        else {"missing": [QUALITY_JOIN]}
        if find_measurement(store, BACKGROUND_MODEL) is not None
        else {}
    )
    return TaskEvidence(
        owning_branches=owners,
        duration_s=duration,
        minimum_duration_s=minimum_duration_s(declared_family) if owners else None,
        event_tokens_n=event_tokens_n(store, declared_family),
        owner_absent_inputs=absent,
        required_event=(needed if needed != "cough" or cough_mode is not None else None)
        or ("phonation" if voice_mode is not None else None)
        or ("syllable" if ddk_mode is not None else None),
        events_found_n=_airway_events_found(store) if "AIRWAY" in owners else None,
        event_kind=airway.label_set if airway is not None else None,
        instructed_count=airway.required_count.value
        if airway is not None and airway.required_count is not None
        else None,
        contest_events_min=contest_events_min(instructed, needed) if needed in ("breath", "cough") else None,
        contest_rise_db_min=contest_rise_db_min(needed) if needed in ("breath", "cough") else None,
        breath_train_rise_db=_train_rise_db(reading),
        breath_mode=breath_mode,
        breath_pattern=reading["pattern"] if reading is not None else None,
        breath_events_n=reading["events_n"] if reading is not None else None,
        breath_vetoed_by=reading.get("vetoed_by") if reading is not None else None,
        breath_train_breaths=reading.get("breaths") if reading is not None else None,
        breath_decision=reading.get("decision") if reading is not None else None,
        breath_review=bool(reading.get("review")) if reading is not None else False,
        breath_reading={"mode": breath_mode, **reading["reading"]} if reading is not None else {},
        cough_mode=cough_mode,
        cough_onsets_n=cough["onsets_n"] if cough is not None else None,
        cough_review=bool(cough.get("review")) if cough is not None else False,
        cough_reading={"mode": cough_mode, **cough["reading"]} if cough is not None else {},
        quality=quality,
        voice_mode=voice_mode,
        voice_found=bool(phonation["found"]) if phonation is not None else None,
        voice_mismatch=phonation.get("mismatch") if phonation is not None else None,
        voice_review=tuple(phonation.get("review") or ()) if phonation is not None else (),
        voice_outside_speech=tuple((phonation or {}).get("outside_speech") or ()),
        voice_reading={
            "mode": voice_mode,
            **{k: v for k, v in phonation.items() if k not in ("name", "signal", "absent")},
        }
        if phonation is not None
        else {},
        ddk_mode=ddk_mode,
        ddk_decision=ddk.get("decision") if ddk is not None else None,
        ddk_reading={
            "mode": ddk_mode,
            **{k: v for k, v in ddk.items() if k not in ("name", "signal", "absent", "value", "reading")},
        }
        if ddk is not None
        else {},
    )


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
        The annotation and the id it came from, or ``(None, [])`` when REDACT wrote none. ``name``,
        ``signal`` and ``remint`` are dropped; every other attribute is carried through unread.
    """
    measurement = find_measurement(store, REDACTION_LLM_ANNOTATION)
    if measurement is None:
        return None, []
    return {key: value for key, value in measurement.attributes.items() if key not in ("name", "signal", REMINT)}, [
        measurement.id
    ]


def _second_opinion(store: ProvStore) -> tuple[dict[str, object] | None, list[str]]:
    """SECOND_OPINION's measurement, as it recorded it.

    Args:
        store: The provenance store.

    Returns:
        The attributes and the id they came from, or ``(None, [])`` where none was written.
    """
    measurement = find_measurement(store, SECOND_OPINION_ANSWERS)
    if measurement is None:
        return None, []
    return {key: value for key, value in measurement.attributes.items() if key not in ("name", "signal", REMINT)}, [
        measurement.id
    ]


def _redaction_evidence(
    store: ProvStore, reports: Sequence[tuple[Entity, BranchReport]], plan: MaskPlan
) -> RedactionEvidence:
    """What the store says about whether this recording carried anything a redaction could remove.

    Args:
        store: The provenance store, read for its ``pii_scan`` measurements and its live findings.
        reports: The reporting nodes' reports, paired with the entities they were read from, so
            SPEECH's own lexical count can be read off its report.
        plan: Which masks stand once the reviewer's unmasks and the content-word trim are applied.

    Returns:
        SPEECH's lexical count, its scan record as a tri-state, how many live ``pii`` findings the
        store holds, what REDACT's re-scan still read after its re-plan, and the masks that stand.
    """
    speech = next((entity for entity, report in reports if report.node == SPEECH), None)
    words = None if speech is None else speech.attributes.get("words_n")
    scans = [measurement.attributes for measurement in find_measurements(store, PII_SCAN)]
    redact = find_verdict(store, _REDACT_NODE)
    survivors = () if redact is None else tuple(str(c) for c in redact.attributes.get("unremediable") or ())
    return RedactionEvidence(
        lexical_words_n=None if words is None else int(words),
        scanned=None if not scans else not any(scan.get(SCANNED) is False for scan in scans),
        findings_n=len([finding for finding in store.entities("pii") if not store.is_invalidated(finding.id)]),
        rescan_survivors=survivors,
        masks_n=len(plan.masks),
        masks_final_n=len(plan.final),
        masks_changed=plan.changed,
        reviewer_unmasked_n=plan.count(UNMASKED_BY_REVIEWER),
        policy_masks_n=plan.policy_masks_n,
        person_names_masked_n=plan.person_names_masked,
        name_release_proposed=plan.name_release_proposed,
        reviewer_requested_n=sum(1 for span in plan.proposals if span.agreement == NEW),
    )


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


def flag_gate_readings(store: ProvStore, names: Sequence[str]) -> dict[str, Any]:
    """The worst reading each flag gate can be answered with, over every extent that carries one.

    A branch mints one task extent per component, so a reading can be written more than once. A
    flag gate asks whether the recording carries the circumstance anywhere, so the answer is the
    reading that is hardest on it: the smallest for an ``at_least`` gate, the largest for an
    ``at_most`` one.

    Args:
        store: The provenance store.
        names: The gates whose readings to look for.

    Returns:
        Reading name to its worst value. A reading nothing live carries, and one written only as
        null, is absent.
    """
    readings: dict[str, Any] = {}
    for name in names:
        spec = GATE_SPECS[name]
        if spec.reading is None:
            continue
        values = [
            float(measurement.attributes["value"])
            for measurement in find_measurements(store, spec.reading)
            if measurement.attributes.get("value") is not None
        ]
        if values:
            readings[spec.reading] = min(values) if spec.op == AT_LEAST else max(values)
    return readings


CARRIER_READING_PREFIXES = ("carrier_", "sweep_", "glide_extent_", "production_declared_")
"""The readings VOICE takes off a qualifying carrier; their absence means no carrier qualified."""

TRACKS_ABSENT_MEASUREMENTS = ("phonation_extent", "sweep_extent")
"""The measurements VOICE writes, valued ``NOT_SEPARABLE_BY_THIS_DESIGN``, when its tracks never arrived."""

CARRIER_PRESENCE_READING = "carrier_duration_s"
"""The reading VOICE writes for every carrier it measured over: its absence means none qualified."""

SPEAKER_SHARE_READING = "extent_dominant_speaker_share"
"""The reading the speaker gate is answered with."""


def reading_absences(store: ProvStore, names: Sequence[str], *, flag: bool = False) -> dict[str, tuple[str, str]]:
    """Why each gate reading nothing live carries is absent, and what the gate should answer.

    Args:
        store: The provenance store.
        names: The gates whose readings to explain.
        flag: Whether these are flag gates, for which a production that does not exist has no
            quality to be asked about, so an absent carrier reading is not applicable rather than a
            failure.

    Returns:
        Reading name to ``(disposition, reason)``, only for readings with no live value.
    """
    out: dict[str, tuple[str, str]] = {}
    lexical_n: int | None = None
    for name in names:
        reading = GATE_SPECS[name].reading
        if reading is None or reading in out:
            continue
        values = [m.attributes.get("value") for m in find_measurements(store, reading)]
        if any(value is not None for value in values):
            continue
        if values:
            out[reading] = (UNDECIDED, NULL_NO_OVERLAP if reading == SPEAKER_SHARE_READING else NULL_VALUE)
            continue
        if reading.startswith(CARRIER_READING_PREFIXES):
            if any(find_measurement(store, absent) is not None for absent in TRACKS_ABSENT_MEASUREMENTS):
                out[reading] = (UNDECIDED, INSTRUMENT_ABSENT)
            elif find_measurement(store, CARRIER_PRESENCE_READING) is not None:
                # A carrier qualified and this one reading of it is missing: a defect, not an absence.
                out[reading] = (UNDECIDED, NOT_COMPUTED)
            else:
                out[reading] = (INAPPLICABLE if flag else FAIL, NO_CARRIER)
        elif reading == INSTRUCTED_COUNT_FRACTION:
            out[reading] = (INAPPLICABLE, NO_INSTRUCTED_COUNT)
        elif reading == SPEAKER_SHARE_READING:
            if lexical_n is None:
                lexical_n = len(lexical_words(store))
            out[reading] = (INAPPLICABLE, NO_SPEECH) if lexical_n == 0 else (UNDECIDED, NOT_COMPUTED)
        elif find_measurement(store, AIRWAY_INSTRUMENT_ABSENT) is not None and reading.startswith("airway_"):
            out[reading] = (UNDECIDED, INSTRUMENT_ABSENT)
        else:
            out[reading] = (UNDECIDED, NOT_COMPUTED)
    return out


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
        applied: One record per conformance gate applied, in the order the group declares them.
        flagging: One record per :data:`FLAG_GATES` gate applied. These decide no conformance;
            each one that did not pass is a flag ground of its own.
        reason: Why no conformance gate was applied, where none was: :data:`NO_OWNER_REPORT`.
        exempt: The flag gates the declared family's instruction exempts, which were not applied.
    """

    node: str | None
    conformance: Conformance
    bounds: GateBounds | None
    applied: tuple[AppliedGate, ...]
    flagging: tuple[AppliedGate, ...] = ()
    reason: str | None = None
    exempt: tuple[str, ...] = ()

    @property
    def unmeasured(self) -> tuple[str, ...]:
        """The gates this fold wanted and nobody has measured, by their full config paths.

        Returns:
            One path per applied gate whose bound is null, named by the layer that supplied it, in
            the order they were applied. They join the reporting node's own unmeasured asks, so
            ``verdict.unmeasured_points_flag`` reaches a gate the same way it reaches an
            instrument setting.
        """
        return tuple(_gate_path(gate) for gate in (*self.applied, *self.flagging) if gate.bound is None)

    def record(self) -> dict[str, Any]:
        """This application, as the verdict records it.

        Returns:
            The node, the group and its bounds, every conformance gate applied and every flagging
            gate applied. Empty when no group was resolved, which is every recording that declares
            no task this graph holds a row for.
        """
        if self.bounds is None:
            return {}
        return {
            "node": self.node,
            **self.bounds.record(),
            "applied": [gate.record() for gate in self.applied],
            "flagging": [gate.record() for gate in self.flagging],
            **({"reason": self.reason} if self.reason is not None else {}),
            **({"exempt": list(self.exempt)} if self.exempt else {}),
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
    exempt = flag_gate_exemptions(config, declared_family)
    # The flag gates ask about the recording's circumstances, not about the instruction, so they
    # are applied whether or not the owning branch evaluated the task in family.
    flag_names = tuple(name for name in FLAG_GATES if name not in exempt)
    flagging = apply_flag_gates(
        bounds,
        flag_gate_readings(store, flag_names),
        reading_absences(store, flag_names, flag=True),
        exempt=exempt,
    )
    reported = next((report for report in reports if report.node == branch and report.in_family), None)
    if reported is None:
        return GateOutcome(branch, UNDETERMINED, bounds, (), tuple(flagging), NO_OWNER_REPORT, exempt)
    names = conformance_gate_names(expectation.pattern)
    conformance, applied = apply_gates(names, bounds, gate_readings(store, names), reading_absences(store, names))
    return GateOutcome(branch, conformance, bounds, tuple(applied), tuple(flagging), None, exempt)


FLAG_GATE_EXEMPTIONS = "verdict.flag_gate_exemptions"
"""Declared family to the flag gates its own instruction makes inapplicable."""


def flag_gate_exemptions(config: TriageConfig, declared_family: str | None) -> tuple[str, ...]:
    """The flag gates the declared family's instruction exempts.

    Args:
        config: The resolved triage configuration.
        declared_family: The declared family, or None.

    Returns:
        The exempt gate names, sorted; empty for a family the table does not name.

    Raises:
        ValueError: If the table names a gate that is not a flag gate.
    """
    table = config.get(FLAG_GATE_EXEMPTIONS) or {}
    names = tuple(sorted(str(name) for name in (table.get(declared_family) or ()))) if declared_family else ()
    unknown = sorted(set(names) - set(FLAG_GATES))
    if unknown:
        raise ValueError(f"{FLAG_GATE_EXEMPTIONS}.{declared_family} names {unknown}, which are not flag gates")
    return names


def _derived_ran(
    store: ProvStore, verdicts: Sequence[NodeVerdict], reports: Sequence[BranchReport]
) -> dict[str, RunState]:
    """Whether each graph node ran, as far as the store can say.

    A node that reported counts as having concluded exactly as one that decided does, and REVIEW,
    which writes neither, concludes with its live annotation. An activity every output of which a
    later pass retired was superseded, not attempted.

    Args:
        store: The provenance store, read for which nodes have an activity.
        verdicts: Every node verdict read from the store.
        reports: Every branch report read from the store.

    Returns:
        ``COMPLETED`` for a node carrying a verdict, a report, (REVIEW) an annotation or (SESSION,
        BACKGROUND) its measurement, ``ERRORED``
        for one carrying a live activity but none of those, and ``SKIPPED`` for one carrying none.
    """
    concluded = {v.node for v in verdicts} | {r.node for r in reports}
    if find_measurement(store, REDACTION_LLM_ANNOTATION) is not None:
        concluded.add(_REVIEW)
    for node, name in ((SESSION_NODE, SESSION_FLOOR), (BACKGROUND_NODE, BACKGROUND_MODEL)):
        if find_measurement(store, name) is not None:
            concluded.add(node)
    outputs: dict[str, list[str]] = {}
    for entity in store.entities():
        activity_id = store.generated_by(entity.id)
        if activity_id is not None:
            outputs.setdefault(activity_id, []).append(entity.id)
    attempted = {
        activity.node
        for activity in store.activities()
        if not (outputs.get(activity.id) and all(store.is_invalidated(entity_id) for entity_id in outputs[activity.id]))
    }
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
    opinion, opinion_ids = _second_opinion(store)
    resolved_ran = {**_derived_ran(store, node_verdicts, reports), **(ran or {})}
    policy = FoldPolicy.from_config(config)
    plan = mask_plan(
        store,
        reviewer_applies=policy.llm_reset_redactions and reviewer_may_unmask(annotation),
        padding_ms=padding_ms(config),
        condition_categories=policy.condition_categories,
        protected_categories=policy.trim_protected_categories,
        cohort_conditions=policy.cohort_conditions,
        lexicon=task_lexicon(config, declared_task_family(store), hint),
        language=None if hint is None else str(hint.metadata.get("language") or "") or None,
        name_approvals=name_approvals(config, recording_stem(store)),
        task_text=task_texts(hint),
    )
    task_evidence = _task_evidence(
        store,
        declared_family or None,
        outcome.record(),
    )
    file_verdict = fold_file_verdict(
        node_verdicts,
        branch_reports=reports,
        spans_by_node=_spans_by_node(store),
        branch_decisions=decisions,
        ran=resolved_ran,
        hint_claims=_hint_claims(decisions, hint, declared_family=declared_family),
        route_state=route_state,
        declared_family=declared_family or None,
        redaction=_redaction_evidence(store, report_pairs, plan),
        llm_redaction=annotation,
        critical_absences=_critical_absences(store),
        gates=outcome.record(),
        flag_gates=[gate.record() for gate in outcome.flagging],
        policy=policy,
        agreed_redactions=plan.agreed,
        unplaced=[(finding.family, finding.state) for finding in plan.unplaced],
        second_opinion=opinion,
        task=task_evidence,
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
        + opinion_ids
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
    ledger_id = mint_live(
        store,
        prov_type="measurement",
        extent=None,
        attributes=plan.record(release=release_value(file_verdict.release), release_ground=file_verdict.release_ground),
    )
    store.was_generated_by(ledger_id, activity)
    store.was_attributed_to(ledger_id, software)
    store.was_derived_from(ledger_id, verdict_id)
    return VerdictResult(
        verdict=node_verdict,
        view=(verdict_id, *folded_ids),
        verdict_entity_id=verdict_id,
        file_verdict=file_verdict,
        ledger_entity_id=ledger_id,
    )
