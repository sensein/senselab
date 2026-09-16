"""The foundation the four triage branches sit on: expectations, proposals, and the two modes.

A branch has exactly two entry points. ``align_<branch>`` evaluates a task of the branch's own kind
against what its instruction asked for; ``detect_<branch>`` finds the branch's own speciality on a
task of any other kind and evaluates nothing. :func:`dispatch` picks between them from the declared
task family and enforces what each mode may return.

Both modes write by ``propose`` only: a branch mints spans in its own family and never edits a span
another node proposed, so ``wasDerivedFrom`` is the whole record of where an extent came from and
:func:`propose_span` refuses a proposal that carries none.

The design, its measurements and what each operating point owes are in
``specs/20260817-triage-workflow-dag/expected-patterns.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal, NamedTuple, Protocol, Sequence

import numpy as np

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage.config import TriageConfig, UnmeasuredConfigKey
from senselab.audio.workflows.triage.consensus import vocabulary_key
from senselab.audio.workflows.triage.nodes.common import find_measurement, live_entities
from senselab.audio.workflows.triage.routing_analysis.families import (
    AIRWAY_ELICITING,
    SPEECH_ELICITING,
    SYLLABLE_REPETITION,
    VOICE_ELICITING,
    task_family,
    task_id_of,
)
from senselab.utils.prov_store import Entity, ProvStore

UNDETERMINED: Literal["UNDETERMINED"] = "UNDETERMINED"
"""What a mode returns when it evaluated no task, or when its only instrument is absent."""

NOT_SEPARABLE_BY_THIS_DESIGN = "NOT_SEPARABLE_BY_THIS_DESIGN"
"""The value of a measurement the design states no viable approach for."""

UNKNOWN_TASK = "unknown"
"""What :func:`~...routing_analysis.families.task_id_of` returns for a stem carrying no task."""

RECORDING_STREAM = "recording"
"""The stream entity ADMIT writes, whose ``path`` is the only in-store carrier of the declaration."""

BRANCHES = ("AIRWAY", "SPEECH", "VOICE", "DDK")
"""The four branches this module serves, in the vocabulary's own spelling."""

BRANCH_FAMILY = {"AIRWAY": "airway", "SPEECH": "speech", "VOICE": "voice", "DDK": "ddk"}
"""Branch name to the lowercase span family it, and only it, may propose into."""

Done = bool | Literal["UNDETERMINED"]
"""Whether the expected patterns were found. Read off the recording, never off the declaration."""

RESERVED_SPAN_ATTRIBUTES = frozenset({"family", "role"})
"""Attribute names a proposal may not carry: the proposer stamps both from its own arguments."""


# --------------------------------------------------------------------- what a branch returns


class Proposal(NamedTuple):
    """A span a branch mints in its own family. It never edits a span another node proposed.

    Attributes:
        family: The branch's own family, lowercase; fixed by :func:`proposer`, never by a caller.
        role: What this span is inside the task, e.g. ``"task_extent"`` or ``"count_in"``.
        start: Extent start, in seconds.
        end: Extent end, in seconds.
        derived_from: The entity ids this extent came from. Required: under propose-only it is the
            whole record of the relationship to the region the span was derived from.
        attributes: The span entity's remaining attributes.
    """

    family: str
    role: str
    start: float
    end: float
    derived_from: tuple[str, ...]
    attributes: dict[str, Any]


class Finding(NamedTuple):
    """Everything a branch has to say that is not a proposed span.

    Attributes:
        kind: One of ``deviation``, ``count``, ``measure`` or ``contest``.
        name: The deviation type, the count's name, the measurement's name, or the claim.
        start: Extent start, or None when the finding is per recording.
        end: Extent end, or None when the finding is per recording.
        evidence: The finding's own payload.
    """

    kind: str
    name: str
    start: float | None
    end: float | None
    evidence: dict[str, Any]


FINDING_KINDS = ("deviation", "count", "measure", "contest")
"""Every ``Finding.kind`` the store knows how to write."""


class Result(NamedTuple):
    """What every branch entry point returns: the three things a branch reports and nothing else.

    There is no outcome here and there is no outcome downstream of it. ``done`` becomes the report's
    task conformance, ``components`` become the spans, ``deviations`` become the findings, and
    ``vocabulary.fold_file_verdict`` is the only place any of it is turned into a decision.

    Attributes:
        done: Whether the expected patterns were found — the branch's task conformance.
            ``detect_*`` always returns :data:`UNDETERMINED`, because no pattern was expected of it,
            and a body that could not read the operating point its qualifier needed returns it too.
        components: The spans this branch proposes, each in its own family, each naming its
            evidence. An empty list is a result, not a failure.
        deviations: Every finding that is not a proposed span.
    """

    done: Done
    components: list[Proposal]
    deviations: list[Finding]


class Propose(Protocol):
    """One branch's minting function, with its family already bound."""

    def __call__(
        self,
        role: str,
        extent: tuple[float, float],
        /,
        *derived_from: str,
        **attributes: Any,  # noqa: ANN401
    ) -> Proposal:
        """Mint one proposal.

        Args:
            role: What this span is inside the task. Positional only, so a ``role`` keyword
                reaches the reserved-attribute check rather than colliding with this parameter.
            extent: ``(start, end)``, in seconds. Positional only, for the same reason.
            *derived_from: The entity ids the extent came from. At least one is required.
            **attributes: The span's remaining attributes.

        Returns:
            The proposal.
        """
        ...


def proposer(family: str) -> Propose:
    """One minting function per branch. The family is fixed here, never by the caller.

    Args:
        family: The branch's own lowercase family.

    Returns:
        A callable that mints :class:`Proposal` records in that family and no other.
    """

    def _propose(
        role: str,
        extent: tuple[float, float],
        /,
        *derived_from: str,
        **attributes: Any,  # noqa: ANN401
    ) -> Proposal:
        """Mint one proposal in this proposer's family.

        Args:
            role: What this span is inside the task.
            extent: ``(start, end)``, in seconds.
            *derived_from: The entity ids the extent came from.
            **attributes: The span's remaining attributes.

        Returns:
            The proposal.

        Raises:
            ValueError: If no evidence is named, if the extent has no positive duration, or if an
                attribute shadows a key the proposer itself stamps.
        """
        shadowed = attributes.keys() & RESERVED_SPAN_ATTRIBUTES
        if shadowed:
            raise ValueError(
                f"{family}/{role}: an attribute may not shadow {sorted(shadowed)}; the family is "
                "the branch's own and the role is this argument, so a second value for either "
                "would let the written span disagree with the proposal"
            )
        if not derived_from:
            raise ValueError(
                f"{family}/{role}: a proposed span names its evidence. Under propose-only the "
                "derivation is the whole record of where the extent came from, so a proposal "
                "without one loses information rather than omitting a detail."
            )
        if not extent[1] > extent[0]:
            raise ValueError(f"{family}/{role}: a proposed span has positive duration; got {extent!r}")
        return Proposal(family, role, float(extent[0]), float(extent[1]), tuple(derived_from), dict(attributes))

    return _propose


PROPOSERS: dict[str, Propose] = {branch: proposer(family) for branch, family in BRANCH_FAMILY.items()}
"""Branch name to its minting function. A branch takes its own and cannot reach another's."""

airway_span = PROPOSERS["AIRWAY"]
speech_span = PROPOSERS["SPEECH"]
voice_span = PROPOSERS["VOICE"]
ddk_span = PROPOSERS["DDK"]
quality_span = proposer("quality")
"""QUALITY is not a routed branch and has only the detect mode, so it is not in :data:`PROPOSERS`."""


def deviation(name: str, start: float | None, end: float | None, **evidence: Any) -> Finding:  # noqa: ANN401
    """A finding that the recording departs from what was expected.

    Args:
        name: The deviation type.
        start: Extent start, or None when the deviation is per recording.
        end: Extent end, or None.
        **evidence: The deviation's payload.

    Returns:
        The finding.
    """
    return Finding("deviation", name, start, end, dict(evidence))


def contest(span_id: str, extent: tuple[float, float], claim: str, reason: str) -> Finding:
    """An assertion beside an existing span that it does not carry what was proposed.

    Args:
        span_id: The span being contested.
        extent: That span's extent.
        claim: What is being contested.
        reason: Why, in controlled vocabulary.

    Returns:
        The finding.
    """
    return Finding("contest", claim, extent[0], extent[1], {"of_span": span_id, "reason": reason})


def count(name: str, found: Any, declared: Any) -> Finding:  # noqa: ANN401
    """What was found beside what the instruction declared. Asserts no discrepancy.

    Args:
        name: The count's name.
        found: What was measured.
        declared: What the instruction asked for.

    Returns:
        The finding.
    """
    return Finding("count", name, None, None, {"found": found, "declared": declared})


def measured(name: str, start: float | None, end: float | None, value: Any, **covariates: Any) -> Finding:  # noqa: ANN401
    """A branch measurement over its own extent, with the covariates that qualify it.

    Args:
        name: The measurement's name.
        start: Extent start, or None when the measurement is per recording.
        end: Extent end, or None.
        value: The value.
        **covariates: What the value must be read against.

    Returns:
        The finding.
    """
    return Finding("measure", name, start, end, {"value": value, **dict(covariates)})


def deviation_names(findings: Sequence[Finding]) -> tuple[str, ...]:
    """The deviation types among these findings, sorted and deduplicated.

    Args:
        findings: A branch's findings.

    Returns:
        The names, for the report's ``deviations``.
    """
    return tuple(sorted({finding.name for finding in findings if finding.kind == "deviation"}))


def unviable(name: str, why: str) -> Finding:
    """A measurement the design states no viable approach for, recorded rather than omitted.

    Args:
        name: The measurement that cannot be taken.
        why: The reason, as the expectation's ``unviable`` entry carries it.

    Returns:
        The finding, whose value is :data:`NOT_SEPARABLE_BY_THIS_DESIGN`.
    """
    return Finding("measure", name, None, None, {"value": NOT_SEPARABLE_BY_THIS_DESIGN, "why": why})


# --------------------------------------------------------------------- the write path


def propose_span(store: ProvStore, activity_id: str, agent_id: str, proposal: Proposal) -> str:
    """Write one proposed span, with its family, its role and its derivation.

    Args:
        store: The provenance store.
        activity_id: The activity that proposed it.
        agent_id: The agent answerable for it.
        proposal: What to mint.

    Returns:
        The span entity's id.

    Raises:
        ValueError: If the proposal names no evidence, which :func:`proposer` already refuses; the
            check is repeated here because this is the only function that writes.
    """
    if not proposal.derived_from:
        raise ValueError(
            f"{proposal.family}/{proposal.role}: a proposed span names its evidence; refusing to "
            "write a span whose relationship to the region it came from is unrecorded"
        )
    span_id = store.entity(
        prov_type="span",
        extent=(proposal.start, proposal.end),
        attributes={**proposal.attributes, "family": proposal.family, "role": proposal.role},
    )
    store.was_generated_by(span_id, activity_id)
    store.was_attributed_to(span_id, agent_id)
    for source_id in proposal.derived_from:
        store.was_derived_from(span_id, source_id)
    return span_id


def propose_spans(store: ProvStore, activity_id: str, agent_id: str, proposals: Sequence[Proposal]) -> list[str]:
    """Write every proposed span, in order.

    Args:
        store: The provenance store.
        activity_id: The activity that proposed them.
        agent_id: The agent answerable for them.
        proposals: What to mint.

    Returns:
        The span entity ids, in the order the proposals were given.
    """
    return [propose_span(store, activity_id, agent_id, proposal) for proposal in proposals]


def write_findings(
    store: ProvStore, activity_id: str, agent_id: str, findings: Sequence[Finding], *, signal: str
) -> list[str]:
    """Write a branch's findings, each in the form its kind calls for.

    A ``deviation`` and a ``contest`` become assertions beside a span rather than edits to one; a
    ``measure`` becomes its own measurement; every ``count`` folds into one ``counts`` measurement
    carrying ``found`` beside ``declared`` per entry, which asserts no discrepancy.

    Args:
        store: The provenance store.
        activity_id: The activity that found them.
        agent_id: The agent answerable for them.
        findings: What to write.
        signal: The stream the findings were taken over.

    Returns:
        The entity ids written, assertions and measurements in the order the findings were given,
        with the single ``counts`` measurement last when any count was present.

    Raises:
        ValueError: If a finding carries a kind outside :data:`FINDING_KINDS`.
    """
    unknown = sorted({finding.kind for finding in findings} - set(FINDING_KINDS))
    if unknown:
        raise ValueError(f"unknown finding kinds {unknown}; expected one of {list(FINDING_KINDS)}")
    written: list[str] = []
    counts: dict[str, Any] = {}
    for finding in findings:
        extent = None if finding.start is None or finding.end is None else (finding.start, finding.end)
        if finding.kind == "count":
            counts[finding.name] = dict(finding.evidence)
            continue
        if finding.kind == "measure":
            attributes = {"name": finding.name, "signal": signal, **finding.evidence}
            entity_id = store.entity(prov_type="measurement", extent=extent, attributes=attributes)
        else:
            verb = "deviate" if finding.kind == "deviation" else "contest"
            key = "deviation_type" if finding.kind == "deviation" else "claim"
            entity_id = store.entity(
                prov_type="assertion", extent=extent, attributes={"verb": verb, key: finding.name, **finding.evidence}
            )
        store.was_generated_by(entity_id, activity_id)
        store.was_attributed_to(entity_id, agent_id)
        written.append(entity_id)
    if counts:
        entity_id = store.entity(
            prov_type="measurement", extent=None, attributes={"name": "counts", "signal": signal, "entries": counts}
        )
        store.was_generated_by(entity_id, activity_id)
        store.was_attributed_to(entity_id, agent_id)
        written.append(entity_id)
    return written


# --------------------------------------------------------------------- the expectation, as data


class Pattern(Enum):
    """The kinds of expected pattern an instruction can ask for."""

    ORDERED_TOKENS = "ordered_tokens"
    FREE_RESPONSE = "free_response"
    ITEM_LIST = "item_list"
    NO_LEXICAL = "no_lexical"
    SUSTAINED = "sustained"
    GLIDE = "glide"
    EFFORT = "effort"
    PER_SENTENCE = "per_sentence"
    EVENT_SERIES = "event_series"
    EVENT_ALTERNATION = "event_alternation"
    SOUND_COVERAGE = "sound_coverage"
    SYLLABLE_TRAIN = "syllable_train"
    SYLLABLE_SEQUENCE = "syllable_sequence"


@dataclass(frozen=True)
class Expectation:
    """What one instruction asked for, as data. One row per in-family (branch, family) pair.

    Attributes:
        pattern: Which matcher the branch's entry point selects.
        tokens: The tokens the instruction prescribes, when it prescribes them literally.
        token_source: Where the tokens come from when they are not literal.
        expected_event_count: How many events the instruction asks for, when it counts them.
        declared_duration_s: How long the instruction runs, when it is timed rather than counted.
        label_set: Which entry of ``branch.label_sets`` names this task's own sound.
        sequence: The places of articulation a syllable sequence cycles through, in order.
        declared_direction: Which way a pitch sweep is asked to go.
        declared_route: Nose or mouth, where the instruction prescribes one.
        route_from_index: Whether the route is carried by the task's trailing index.
        relax_s: Leading interval the instruction gives to settling rather than to the task.
        repetition_allowed: Whether repeating an item is permitted.
        repetition_from_category: Whether the repetition rule follows from the recording's category.
        lexical_separator: Whether a lexical count-in precedes the production.
        forbid_lexical: Whether any lexical content is a deviation.
        emit_filler: Whether a filler token is reported as a deviation.
        expect_inhale: Whether an inhale precedes the production.
        contrast: Whether the measurement is a within-recording contrast.
        timed_intervals: Whether the intervals between events are themselves prescribed.
        anti_pattern: A pattern whose presence is the deviation.
        connected: Whether the production is connected speech, so breath groups are structure.
        unviable: ``(measurement, why)`` pairs the design states no viable approach for.
    """

    pattern: Pattern
    tokens: tuple[str, ...] | None = None
    token_source: str | None = None
    expected_event_count: int | None = None
    declared_duration_s: float | None = None
    label_set: str | None = None
    sequence: tuple[str, ...] | None = None
    declared_direction: str | None = None
    declared_route: str | None = None
    route_from_index: bool = False
    relax_s: float | None = None
    repetition_allowed: bool | None = None
    repetition_from_category: bool = False
    lexical_separator: bool = False
    forbid_lexical: bool = False
    emit_filler: bool = True
    expect_inhale: bool = False
    contrast: bool = False
    timed_intervals: bool = False
    anti_pattern: str | None = None
    connected: bool = False
    unviable: tuple[tuple[str, str], ...] = ()

    def as_mapping(self) -> dict[str, Any]:
        """This row as plain data, so a run can record which expectation it applied.

        Returns:
            Every field, with the pattern as its value and tuples as lists.
        """
        out: dict[str, Any] = {}
        for field_ in fields(self):
            value = getattr(self, field_.name)
            if isinstance(value, Pattern):
                out[field_.name] = value.value
            elif isinstance(value, tuple):
                out[field_.name] = [list(item) if isinstance(item, tuple) else item for item in value]
            else:
                out[field_.name] = value
        return out

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "Expectation":
        """Rebuild a row from :meth:`as_mapping`'s output.

        Args:
            mapping: What :meth:`as_mapping` returned.

        Returns:
            The row.
        """
        values = dict(mapping)
        values["pattern"] = Pattern(values["pattern"])
        for name in ("tokens", "sequence"):
            if values.get(name) is not None:
                values[name] = tuple(values[name])
        if values.get("unviable") is not None:
            values["unviable"] = tuple((str(pair[0]), str(pair[1])) for pair in values["unviable"])
        return cls(**values)


VOICE_EXPECTATIONS: dict[str, Expectation] = {
    "prolonged-vowel": Expectation(
        pattern=Pattern.SUSTAINED,
        tokens=("one", "two", "three"),
        token_source="instructions",
        declared_duration_s=12.0,
        lexical_separator=True,
    ),
    "maximum-phonation-time": Expectation(pattern=Pattern.SUSTAINED, forbid_lexical=True, expect_inhale=True),
    "maximum-phonation-time-v2": Expectation(pattern=Pattern.SUSTAINED, forbid_lexical=True, expect_inhale=False),
    "glides-low-to-high": Expectation(pattern=Pattern.GLIDE, declared_direction="up"),
    "glides-high-to-low": Expectation(pattern=Pattern.GLIDE, declared_direction="down"),
    "high-to-low": Expectation(pattern=Pattern.GLIDE, declared_direction="down"),
}
"""VOICE's six in-family rows: ``VOICE_ELICITING``."""

VOICE_EXPECTATIONS_PENDING_DECLARATION: dict[str, Expectation] = {
    "loudness": Expectation(
        pattern=Pattern.EFFORT,
        tokens=("hey",),
        expected_event_count=3,
        unviable=(("effort_absolute", "`level` is uncalibrated and no SPL reference exists in the graph"),),
    ),
    "loudness-v2": Expectation(pattern=Pattern.EFFORT, tokens=("hey",), expected_event_count=2, contrast=True),
    "cape-v-sentences": Expectation(pattern=Pattern.PER_SENTENCE, token_source="stimulus_text"),
    "cape-v-sentences-v2": Expectation(pattern=Pattern.PER_SENTENCE, token_source="stimulus_text"),
}
"""VOICE's own instrument families, which ``families.py`` places in ``LEXICAL_SPEECH``.

Out of family for VOICE under the reference family set, so ``align_voice`` does not read this table
and :data:`EXPECTATIONS` does not carry it. Changing that is a ``families.py`` decision.
"""

SPEECH_EXPECTATIONS: dict[str, Expectation] = {
    "harvard-sentences-list": Expectation(pattern=Pattern.ORDERED_TOKENS, token_source="stimulus_text"),
    "cape-v-sentences": Expectation(pattern=Pattern.ORDERED_TOKENS, token_source="stimulus_text"),
    "cape-v-sentences-v2": Expectation(pattern=Pattern.ORDERED_TOKENS, token_source="stimulus_text"),
    "rainbow-passage": Expectation(pattern=Pattern.ORDERED_TOKENS, token_source="stimulus_text", connected=True),
    "caterpillar-passage": Expectation(pattern=Pattern.ORDERED_TOKENS, token_source="stimulus_text", connected=True),
    "word-color-stroop": Expectation(
        pattern=Pattern.ORDERED_TOKENS, token_source="stimulus_text", declared_duration_s=75.0, emit_filler=False
    ),
    "loudness": Expectation(pattern=Pattern.ORDERED_TOKENS, tokens=("hey", "hey", "hey"), expected_event_count=3),
    "loudness-v2": Expectation(pattern=Pattern.ORDERED_TOKENS, tokens=("hey", "hey"), expected_event_count=2),
    "diadochokinesis-buttercup": Expectation(
        pattern=Pattern.ORDERED_TOKENS, tokens=("buttercup",) * 10, expected_event_count=10, emit_filler=False
    ),
    "diadochokinesis-v2-buttercup": Expectation(
        pattern=Pattern.ORDERED_TOKENS, tokens=("buttercup",), declared_duration_s=5.0, emit_filler=False
    ),
    "free-speech": Expectation(
        pattern=Pattern.FREE_RESPONSE, token_source="stimulus_text", anti_pattern="verbatim_prompt"
    ),
    "free-speech-v2": Expectation(pattern=Pattern.FREE_RESPONSE, declared_duration_s=30.0),
    "story-recall": Expectation(
        pattern=Pattern.FREE_RESPONSE, token_source="stimulus_text", anti_pattern="verbatim_source"
    ),
    "story-recall-v2": Expectation(
        pattern=Pattern.FREE_RESPONSE, token_source="stimulus_text", anti_pattern="verbatim_source"
    ),
    "cinderella-story": Expectation(
        pattern=Pattern.FREE_RESPONSE,
        unviable=(("source_overlap", "`stimulus_text` is empty on all 258; the source is a physical storybook"),),
    ),
    "productive-vocabulary": Expectation(
        pattern=Pattern.FREE_RESPONSE,
        token_source="stimulus_text",
        unviable=(("defines_its_cue", "a lexicon or a text model, branch-local, and no waveform"),),
    ),
    "picture-description": Expectation(pattern=Pattern.FREE_RESPONSE, connected=True),
    "picture-description-option1": Expectation(pattern=Pattern.FREE_RESPONSE, connected=True),
    "picture-description-option2": Expectation(pattern=Pattern.FREE_RESPONSE, connected=True),
    "open-response-questions": Expectation(
        pattern=Pattern.FREE_RESPONSE, token_source="stimulus_text", declared_duration_s=30.0, connected=True
    ),
    "animal-fluency": Expectation(
        pattern=Pattern.ITEM_LIST,
        declared_duration_s=60.0,
        repetition_allowed=False,
        unviable=(("category_membership", "a lexicon or a text embedding, one consumer, no waveform"),),
    ),
    "random-item-generation": Expectation(
        pattern=Pattern.ITEM_LIST,
        repetition_from_category=True,
        unviable=(("category_membership", "a lexicon or a text embedding, one consumer, no waveform"),),
    ),
    "random-item-generation-v2": Expectation(
        pattern=Pattern.ITEM_LIST,
        repetition_from_category=True,
        unviable=(("category_membership", "a lexicon or a text embedding, one consumer, no waveform"),),
    ),
}
"""SPEECH's 31 in-family rows: ``LEXICAL_SPEECH`` (21) plus ``SYLLABLE_REPETITION`` (10)."""

for _family in sorted(SYLLABLE_REPETITION - {"diadochokinesis-buttercup", "diadochokinesis-v2-buttercup"}):
    SPEECH_EXPECTATIONS[_family] = Expectation(pattern=Pattern.NO_LEXICAL)

AIRWAY_EXPECTATIONS: dict[str, Expectation] = {
    "respiration-and-cough-cough": Expectation(pattern=Pattern.EVENT_SERIES, label_set="cough", expected_event_count=5),
    "respiration-and-cough-v2-hardcough": Expectation(
        pattern=Pattern.EVENT_SERIES,
        label_set="cough",
        expected_event_count=None,
        unviable=(("effort_absolute", "no within-recording contrast and no SPL reference; `hard` is not measurable"),),
    ),
    "voluntary-cough": Expectation(pattern=Pattern.EVENT_ALTERNATION, label_set="cough", expected_event_count=3),
    "respiration-and-cough-fivebreaths": Expectation(
        pattern=Pattern.EVENT_SERIES,
        label_set="breath",
        expected_event_count=5,
        route_from_index=True,
        unviable=(
            (
                "route",
                "the discriminating band sits above the 8 kHz ceiling and the residual tilt is confounded, "
                "one for one, with mouth-to-microphone geometry",
            ),
        ),
    ),
    "respiration-and-cough-v2-threebreathsnose": Expectation(
        pattern=Pattern.EVENT_SERIES,
        label_set="breath",
        expected_event_count=3,
        declared_route="nose",
        unviable=(("route", "as `fivebreaths`"),),
    ),
    "respiration-and-cough-v2-threebreathsmouth": Expectation(
        pattern=Pattern.EVENT_SERIES,
        label_set="breath",
        expected_event_count=3,
        declared_route="mouth",
        unviable=(("route", "as `fivebreaths`"),),
    ),
    "respiration-and-cough-threequickbreaths": Expectation(
        pattern=Pattern.EVENT_SERIES, label_set="breath", expected_event_count=3, timed_intervals=True
    ),
    "respiration-and-cough-v2-threebreaths": Expectation(
        pattern=Pattern.EVENT_SERIES, label_set="breath", expected_event_count=3, timed_intervals=True
    ),
    "respiration-and-cough-breath": Expectation(
        pattern=Pattern.SOUND_COVERAGE, label_set="breath", declared_duration_s=30.0
    ),
    "respiration-and-cough-v2-breath": Expectation(
        pattern=Pattern.SOUND_COVERAGE,
        label_set="breath",
        declared_duration_s=20.0,
        declared_route="mouth",
        unviable=(("route", "as `fivebreaths`"),),
    ),
    "breath-sounds": Expectation(
        pattern=Pattern.EVENT_SERIES,
        label_set="breath",
        expected_event_count=3,
        declared_route="mouth",
        relax_s=60.0,
        declared_duration_s=73.0,
        unviable=(("route", "as `fivebreaths`"),),
    ),
}
"""AIRWAY's eleven in-family rows: ``AIRWAY_ELICITING``."""

DDK_EXPECTATIONS: dict[str, Expectation] = {
    "diadochokinesis-pa": Expectation(pattern=Pattern.SYLLABLE_TRAIN, expected_event_count=10),
    "diadochokinesis-ta": Expectation(pattern=Pattern.SYLLABLE_TRAIN, expected_event_count=10),
    "diadochokinesis-ka": Expectation(pattern=Pattern.SYLLABLE_TRAIN, expected_event_count=10),
    "diadochokinesis-v2-puh": Expectation(pattern=Pattern.SYLLABLE_TRAIN, declared_duration_s=5.0),
    "diadochokinesis-v2-tuh": Expectation(pattern=Pattern.SYLLABLE_TRAIN, declared_duration_s=5.0),
    "diadochokinesis-v2-kuh": Expectation(pattern=Pattern.SYLLABLE_TRAIN, declared_duration_s=5.0),
    "diadochokinesis-pataka": Expectation(
        pattern=Pattern.SYLLABLE_SEQUENCE, sequence=("labial", "alveolar", "velar"), expected_event_count=30
    ),
    "diadochokinesis-v2-puhtuhkuh": Expectation(
        pattern=Pattern.SYLLABLE_SEQUENCE, sequence=("labial", "alveolar", "velar"), declared_duration_s=5.0
    ),
    "diadochokinesis-buttercup": Expectation(
        pattern=Pattern.ORDERED_TOKENS, tokens=("buttercup",), expected_event_count=10
    ),
    "diadochokinesis-v2-buttercup": Expectation(
        pattern=Pattern.ORDERED_TOKENS, tokens=("buttercup",), declared_duration_s=5.0
    ),
}
"""DDK's ten in-family rows: ``SYLLABLE_REPETITION``."""

EXPECTATIONS: dict[str, dict[str, Expectation]] = {
    "AIRWAY": AIRWAY_EXPECTATIONS,
    "SPEECH": SPEECH_EXPECTATIONS,
    "VOICE": VOICE_EXPECTATIONS,
    "DDK": DDK_EXPECTATIONS,
}
"""58 rows over 48 declared families: the whole in-family membership test, as data.

The ten ``SYLLABLE_REPETITION`` families are in family for both SPEECH and DDK, and that is not an
overlap to resolve: SPEECH's expectation over them is *no lexical content*, DDK's is *a syllable
train*, and neither is the other's.
"""

REFERENCE_FAMILY_SET: dict[str, frozenset[str]] = {
    "AIRWAY": AIRWAY_ELICITING,
    "SPEECH": SPEECH_ELICITING,
    "VOICE": VOICE_ELICITING,
    "DDK": SYLLABLE_REPETITION,
}
"""Each branch's in-family family set, as ``families.py`` declares it.

The packaged config names the same four sets by key under ``taxonomy.ruleset.reference_family_set``.
This is the membership test :func:`dispatch` applies, and :data:`EXPECTATIONS` must cover it exactly.
"""


# --------------------------------------------------------------------- the two modes


def declared_task_family(store: ProvStore, hint: AudioHints | None = None) -> str | None:
    """The declared task family, from whichever carrier the graph has.

    Nothing passes a task family to a branch — routing hands a branch one bit, ``will_run`` — so
    the mode test is the branch's own and this is the one place it is derived. Two carriers are
    read, in order: the hint's ``task_token`` metadata, which is the clean route, and then the
    ``path`` ADMIT writes onto the ``recording`` stream entity, which is the one available today.
    A third carrier is one more entry in this function and no change in any branch.

    Args:
        store: The provenance store.
        hint: What the recording was declared to contain, when the caller supplied one.

    Returns:
        The family, or None when no carrier names one — an absent ``recording`` entity, a path that
        is not a BIDS stem, or a stem whose task id is ``"unknown"``. None takes the out-of-family
        mode, which is the safe arm: the branch annotates its speciality and concludes nothing.
    """
    for task_id in _declared_task_ids(store, hint):
        family = task_family(task_id)
        if family and family != UNKNOWN_TASK:
            return family
    return None


def _declared_task_ids(store: ProvStore, hint: AudioHints | None) -> list[str]:
    """Every task id the store and the hint carry, best carrier first.

    Args:
        store: The provenance store.
        hint: The recording's hints, or None.

    Returns:
        The candidate task ids, lowercased, in the order they should be tried.
    """
    found: list[str] = []
    token = (hint.metadata or {}).get("task_token") if hint is not None else None
    if isinstance(token, str) and token.strip():
        found.append(token.strip().lower())
    recording = [
        entity
        for entity in live_entities(store, "stream")
        if entity.attributes.get("name") == RECORDING_STREAM and entity.attributes.get("path")
    ]
    if recording:
        found.append(task_id_of(Path(str(recording[-1].attributes["path"])).stem))
    return found


class AlignMode(Protocol):
    """The in-family entry point every branch implements."""

    def __call__(self, task_family: str, store: ProvStore, hint: AudioHints | None, params: BranchParams) -> Result:
        """Evaluate a task of this branch's own kind against what its instruction asked for.

        Args:
            task_family: The declared family, which is a key of this branch's expectation table.
            store: The provenance store.
            hint: What the recording was declared to contain.
            params: The operating points.

        Returns:
            Whether the expected patterns were found, the spans proposed, and the deviations.
        """
        ...


class DetectMode(Protocol):
    """The out-of-family entry point every branch implements."""

    def __call__(self, store: ProvStore, params: BranchParams) -> Result:
        """Find this branch's own speciality wherever it occurs, and evaluate no task.

        Args:
            store: The provenance store.
            params: The operating points.

        Returns:
            A result whose ``done`` is :data:`UNDETERMINED`, the spans proposed, and the findings.
        """
        ...


def dispatch(
    branch: str,
    store: ProvStore,
    params: BranchParams,
    hint: AudioHints | None = None,
    *,
    align: AlignMode,
    detect: DetectMode,
) -> Result:
    """The whole of the mode decision, and the contract each mode is held to.

    The declaration picks the mode; it never supplies the answer. ``task_family`` absent from this
    branch's table — undeclared, unreadable, or another branch's kind — takes the out-of-family
    mode. There is no third arm and no fallback that guesses a family.

    Args:
        branch: One of :data:`BRANCHES`.
        store: The provenance store.
        params: The operating points.
        hint: What the recording was declared to contain.
        align: This branch's in-family entry point.
        detect: This branch's out-of-family entry point.

    Returns:
        What the selected mode returned.

    Raises:
        KeyError: If ``branch`` is not one of :data:`BRANCHES`.
        ValueError: If a mode proposed a span outside this branch's own family, or if the
            out-of-family mode returned anything but :data:`UNDETERMINED`.
    """
    table = EXPECTATIONS[branch]
    family = declared_task_family(store, hint)
    in_family = family is not None and family in table
    result = align(family, store, hint, params) if in_family and family is not None else detect(store, params)
    own = BRANCH_FAMILY[branch]
    intruders = sorted({proposal.family for proposal in result.components} - {own})
    if intruders:
        raise ValueError(f"{branch} proposed into {intruders}; a branch mints only into {own!r}")
    if not in_family and result.done != UNDETERMINED:
        raise ValueError(
            f"detect_{own} returned done={result.done!r}; the out-of-family mode evaluates no task, "
            f"so its only answer is {UNDETERMINED!r}"
        )
    return result


def mode_of(branch: str, store: ProvStore, hint: AudioHints | None = None) -> tuple[str, str | None]:
    """Which mode :func:`dispatch` would select, and on what family. For a verdict's record.

    Args:
        branch: One of :data:`BRANCHES`.
        store: The provenance store.
        hint: What the recording was declared to contain.

    Returns:
        ``("align", family)`` or ``("detect", family)``, the family being what was declared, which
        is None when nothing carried one and may be another branch's kind.

    Raises:
        KeyError: If ``branch`` is not one of :data:`BRANCHES`.
    """
    family = declared_task_family(store, hint)
    return ("align" if family is not None and family in EXPECTATIONS[branch] else "detect", family)


# --------------------------------------------------------------------- the operating points


PARAM_SECTION = "branch"
"""The config section every operating point in :class:`BranchParams` is read from."""


def _band(value: Any) -> tuple[float, float]:  # noqa: ANN401 — one config leaf, shape checked here
    """A ``[lo, hi]`` config leaf as a pair of floats.

    Args:
        value: The leaf.

    Returns:
        ``(lo, hi)``.
    """
    lo, hi = value
    return float(lo), float(hi)


def _bands(value: Any) -> dict[str, tuple[float, float]]:  # noqa: ANN401 — one config leaf
    """A name-to-``[lo, hi]`` config mapping as a mapping of pairs.

    Args:
        value: The leaf.

    Returns:
        Each name's band.
    """
    return {str(name): _band(band) for name, band in value.items()}


def _label_sets(value: Any) -> dict[str, tuple[str, ...]]:  # noqa: ANN401 — one config leaf
    """A label-set mapping as a mapping of tuples.

    Args:
        value: The leaf.

    Returns:
        Each set's labels.
    """
    return {str(name): tuple(str(label) for label in labels) for name, labels in value.items()}


POINT_TYPES: dict[str, Callable[[Any], Any]] = {
    "smoothing_window_s": float,
    "peak_prominence_db": float,
    "trough_return_db": float,
    "event_min_s": float,
    "score_min": float,
    "breath_coverage_min": float,
    "voiced_strength_min": float,
    "voiced_fraction_min": float,
    "f0_spread_window_s": float,
    "f0_spread_max_semitones": float,
    "continuity_min": float,
    "production_min_s": float,
    "monotone_tolerance_semitones": float,
    "dominant_segment_min_fraction": float,
    "response_min_s": float,
    "pause_min_s": float,
    "run_gap_max_s": float,
    "breath_group_min_gap_s": float,
    "repeat_overlap_min": float,
    "echo_ngram_n": int,
    "echo_overlap_max": float,
    "verbatim_overlap_max": float,
    "coverage_min": float,
    "expected_lexical_max": int,
    "interval_max_s": float,
    "modulation_band_hz": _band,
    "rate_prominence_min": float,
    "train_min_s": float,
    "repeat_min_occurrences": int,
    "burst_window_ms": float,
    "place_centroid_bands_hz": _bands,
    "place_margin_db": float,
    "effort_split_hz": float,
    "gap_off_task_min_s": float,
    "label_sets": _label_sets,
}
"""Every ``branch.*`` key, and the type its value is read as. The one declaration of both.

An entry here is an operating point a body may ask for; :data:`PARAM_KEYS` is its key order and
``config_test`` pins the pair against the packaged section.
"""

UNMEASURED_POINTS = "unmeasured_operating_points"
"""The measurement naming every ``branch.*`` key a body asked for and nobody has measured."""


@dataclass
class BranchParams:
    """Every operating point a branch body may ask for, read from the config and never refused.

    **A branch does not decide, and a refusal is a decision.** So there is one accessor,
    :meth:`point`, it returns None for a point nobody has measured, and it records what was asked
    for so the branch can report it as a fact beside its spans. Nothing here raises for an
    unmeasured value; :meth:`point` raises only for a key that is not a ``branch.*`` key at all,
    which is a typo in the calling code and not a measurement the graph is missing.

    Reading is lazy and the misses accumulate per instance, so what a report names is what *this*
    recording's bodies actually asked for rather than the whole section.

    Attributes:
        config: The resolved triage configuration.
        missing: The keys read while null, in first-read order.
    """

    config: TriageConfig
    missing: list[str] = field(default_factory=list)

    def point(self, key: str) -> Any:  # noqa: ANN401 — each key's own type; several are not floats
        """One operating point, or None when nobody has measured it.

        Args:
            key: The key's name inside the ``branch`` section.

        Returns:
            The value in the type :data:`POINT_TYPES` declares for it, or None when the packaged or
            overridden value is null. A null is recorded in :attr:`missing` on first read.

        Raises:
            KeyError: If the name is not a ``branch`` key. A typo in a body is a programming error,
                not a measurement the graph is missing, so it surfaces here rather than reading as
                one more unmeasured point.
            UnknownConfigKey: If the name is a :data:`POINT_TYPES` key the packaged file does not
                spell — a drift between the two, which the second guard catches because
                :meth:`TriageConfig.require` tells an unspelled key apart from a null one and this
                method catches only the null.
        """
        if key not in POINT_TYPES:
            raise KeyError(f"{PARAM_SECTION}.{key} is not a branch operating point; check it against PARAM_KEYS")
        try:
            value = self.config.require(f"{PARAM_SECTION}.{key}")
        except UnmeasuredConfigKey:
            if key not in self.missing:
                self.missing.append(key)
            return None
        return POINT_TYPES[key](value)

    def setting(self, path: str, coerce: Callable[[Any], Any] = str) -> Any:  # noqa: ANN401 — one config leaf
        """One config value outside the ``branch`` section, read without refusing.

        A body sometimes needs a setting another section owns — the stimulus aligner's sentence
        terminators, for instance. Reading it with ``require`` would be a refusal by the branch, so
        it is read here and a null is recorded beside the branch's own unmeasured points.

        Args:
            path: The full dotted path.
            coerce: How to read the value.

        Returns:
            The value, or None when it is null.

        Raises:
            UnknownConfigKey: If no packaged key spells the path, which is a typo in the body.
        """
        try:
            return coerce(self.config.require(path))
        except UnmeasuredConfigKey:
            if path not in self.missing:
                self.missing.append(path)
            return None

    def record(self) -> list[Finding]:
        """What was asked for and could not be read, as one finding.

        Returns:
            One :data:`UNMEASURED_POINTS` measurement, or nothing when every key read had a value.
            The list is in **read order**, not sorted: which point a body reached for first is what
            says where the evaluation stopped being able to proceed.
        """
        if not self.missing:
            return []
        return [measured(UNMEASURED_POINTS, None, None, list(self.missing), section=PARAM_SECTION)]

    @property
    def p_normalise(self) -> Callable[[str], str]:
        """How a token is normalised before it is compared. Not a config key: it is a function.

        Returns:
            ``consensus.vocabulary_key``, which is the normalisation PREPROCESS's own consensus and
            stimulus alignment already declare (``casefold; keep alphanumerics and apostrophe``).
            A second spelling here would compare tokens against a transcript normalised another way.
        """
        return vocabulary_key


PARAM_KEYS = (
    "smoothing_window_s",
    "peak_prominence_db",
    "trough_return_db",
    "event_min_s",
    "score_min",
    "breath_coverage_min",
    "voiced_strength_min",
    "voiced_fraction_min",
    "f0_spread_window_s",
    "f0_spread_max_semitones",
    "continuity_min",
    "production_min_s",
    "monotone_tolerance_semitones",
    "dominant_segment_min_fraction",
    "response_min_s",
    "pause_min_s",
    "run_gap_max_s",
    "breath_group_min_gap_s",
    "repeat_overlap_min",
    "echo_ngram_n",
    "echo_overlap_max",
    "verbatim_overlap_max",
    "coverage_min",
    "expected_lexical_max",
    "interval_max_s",
    "modulation_band_hz",
    "rate_prominence_min",
    "train_min_s",
    "repeat_min_occurrences",
    "burst_window_ms",
    "place_margin_db",
    "effort_split_hz",
    "gap_off_task_min_s",
    "place_centroid_bands_hz",
    "label_sets",
)
"""Every key the ``branch`` config section holds, in the order the section declares them.

``p_normalise`` has no key: it is a function, and a config key naming one would be a plugin hook
nobody has measured. Every other entry is a key of :data:`POINT_TYPES` and of the packaged section,
and ``config_test`` pins the three against each other.
"""


def branch_params(config: TriageConfig) -> BranchParams:
    """The operating points, over one resolved configuration.

    Args:
        config: The resolved triage configuration.

    Returns:
        The record. Nothing is read until :meth:`BranchParams.point` is called, and a fresh record
        per node call is what makes its ``missing`` list this recording's own.
    """
    return BranchParams(config=config)


# --------------------------------------------------------------------- extent arithmetic


def overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """Whether two extents share any ground.

    Args:
        a: One extent.
        b: The other.

    Returns:
        True when they intersect; touching at a boundary is not an overlap.
    """
    return a[0] < b[1] and a[1] > b[0]


def duration(extent: tuple[float, float] | None) -> float:
    """How long an extent is.

    Args:
        extent: The extent, or None.

    Returns:
        Its length in seconds, or 0.0 when it is None.
    """
    return 0.0 if extent is None else float(extent[1] - extent[0])


def hull(extents: Sequence[tuple[float, float]]) -> tuple[float, float] | None:
    """The one extent covering all of them.

    Args:
        extents: The extents.

    Returns:
        ``(earliest start, latest end)``, or None when there are none.
    """
    if not extents:
        return None
    return (min(start for start, _ in extents), max(end for _, end in extents))


def merge(extents: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Coalesce touching or overlapping extents.

    Joins only what touches. Ordinary speech has a gap between every pair of words, so this is not
    the way to group words into runs — :func:`lexical_runs` is.

    Args:
        extents: The extents, in any order.

    Returns:
        The disjoint extents, earliest first.
    """
    out: list[tuple[float, float]] = []
    for start, end in sorted(extents):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def touches_edge(extent: tuple[float, float], stream_extent: tuple[float, float]) -> bool:
    """Whether an extent reaches either end of the recording.

    Args:
        extent: The extent.
        stream_extent: The recording's own extent.

    Returns:
        True when it starts at or before the recording's start, or ends at or after its end.
    """
    return extent[0] <= stream_extent[0] or extent[1] >= stream_extent[1]


def stream_extent(store: ProvStore, name: str = RECORDING_STREAM) -> tuple[float, float] | None:
    """One stream's extent, which is the measured duration every "was it done" test needs.

    Args:
        store: The provenance store.
        name: The stream entity's name.

    Returns:
        The extent, or None when no live stream of that name carries one.
    """
    found = [
        entity
        for entity in live_entities(store, "stream")
        if entity.attributes.get("name") == name and entity.extent is not None
    ]
    return found[-1].extent if found else None


# --------------------------------------------------------------------- reading the store's spans and words


def spans_by_measure(spans: Sequence[Entity], measure: str) -> list[Entity]:
    """PREPROCESS's spans of one measure, earliest first.

    Args:
        spans: The span entities.
        measure: One of ``amplitude``, ``continuity``, ``asr`` or ``gap``.

    Returns:
        The matching spans that carry an extent, earliest first.
    """
    found = [span for span in spans if span.attributes.get("measure") == measure and span.extent is not None]
    return sorted(found, key=lambda span: span.extent or (0.0, 0.0))


def amplitude_spans(spans: Sequence[Entity]) -> list[Entity]:
    """The amplitude spans, which are the carriers a production is looked for inside.

    Args:
        spans: The span entities.

    Returns:
        The spans whose ``measure`` is ``amplitude``.
    """
    return spans_by_measure(spans, "amplitude")


def asr_spans(spans: Sequence[Entity]) -> list[Entity]:
    """The ASR spans.

    Args:
        spans: The span entities.

    Returns:
        The spans whose ``measure`` is ``asr``.
    """
    return spans_by_measure(spans, "asr")


def gaps(spans: Sequence[Entity]) -> list[Entity]:
    """The gap spans, which are where off-task material is looked for.

    Args:
        spans: The span entities.

    Returns:
        The spans whose ``measure`` is ``gap``.
    """
    return spans_by_measure(spans, "gap")


def lexical(words: Sequence[Entity]) -> list[Entity]:
    """The consensus words that are not bracketed.

    Args:
        words: The ``word`` entities, in index order.

    Returns:
        The subset whose ``bracketed`` attribute is false.
    """
    return [word for word in words if not word.attributes.get("bracketed")]


def word_text(word: Entity) -> str:
    """One word's text.

    Args:
        word: A ``word`` entity.

    Returns:
        Its ``text`` attribute, or the empty string.
    """
    return str(word.attributes.get("text", ""))


def word_extent(word: Entity) -> tuple[float, float]:
    """One word's extent.

    Args:
        word: A ``word`` entity.

    Returns:
        Its extent, or ``(0.0, 0.0)`` when it carries none.
    """
    return word.extent if word.extent is not None else (0.0, 0.0)


def off_task(components: Sequence[Proposal], spans: Sequence[Entity], p_gap_off_task_min_s: float) -> list[Finding]:
    """Gaps long enough to matter that no proposed span covers.

    Off-task material is the absence of the branch's speciality rather than an instance of it, so it
    is a deviation and never a proposed span: a branch does not mint over ground it is disclaiming.

    Args:
        components: The spans this branch proposed.
        spans: The span entities.
        p_gap_off_task_min_s: Shortest gap reported.

    Returns:
        One ``off_task_extent`` deviation per uncovered gap.
    """
    out: list[Finding] = []
    for gap in gaps(spans):
        extent = gap.extent
        if extent is None or duration(extent) < p_gap_off_task_min_s:
            continue
        if any(overlaps(extent, (component.start, component.end)) for component in components):
            continue
        out.append(deviation("off_task_extent", extent[0], extent[1], measure="gap"))
    return out


def off_task_findings(components: Sequence[Proposal], spans: Sequence[Entity], params: BranchParams) -> list[Finding]:
    """:func:`off_task`, with the gap minimum read rather than passed.

    Args:
        components: The spans this branch proposed.
        spans: The span entities.
        params: The operating points.

    Returns:
        One ``off_task_extent`` deviation per uncovered gap, and nothing at all when the gap minimum
        is unmeasured: without it no gap is long enough to matter or short enough not to be, and the
        ask is recorded in ``params.missing``.
    """
    minimum = params.point("gap_off_task_min_s")
    return [] if minimum is None else off_task(components, spans, minimum)


def declared_duration_count(store: ProvStore, declared: float | None) -> list[Finding]:
    """The recording's measured duration beside what the sidecar declared.

    A sidecar-consistency check, not a task-completion measurement: the finding is that the two
    disagree, and which of them is wrong is a separate question.

    Args:
        store: The provenance store.
        declared: The declared duration, or None when the instruction declares none.

    Returns:
        One ``declared_duration_s`` count, or nothing.
    """
    if declared is None:
        return []
    return [count("declared_duration_s", round(duration(stream_extent(store)), 2), declared)]


# --------------------------------------------------------------------- the derivatives, as arrays


@dataclass(frozen=True)
class EnvelopeTrack:
    """The energy envelope and the one global floor every rise is measured against.

    Attributes:
        envelope_dbfs: The envelope, one value per sample of the envelope's own rate.
        floor_dbfs: The recording's single global noise floor. Not a local floor.
        sampling_rate: The envelope's sampling rate, in Hz.
    """

    envelope_dbfs: np.ndarray
    floor_dbfs: float
    sampling_rate: float


@dataclass(frozen=True)
class ContinuityTrack:
    """The spectral-stationarity trace: cosine similarity between consecutive log spectra.

    Attributes:
        continuity: The trace, per sample, in ``[0, 1]``.
        sampling_rate: Its sampling rate, in Hz.
    """

    continuity: np.ndarray
    sampling_rate: float


@dataclass(frozen=True)
class PhonationTracks:
    """F0 and its strength at a 10 ms hop.

    Attributes:
        times_s: Frame centres, in seconds.
        f0_hz: F0 per frame; zero or non-finite where unvoiced.
        strength: Pitch strength per frame.
    """

    times_s: np.ndarray
    f0_hz: np.ndarray
    strength: np.ndarray


@dataclass(frozen=True)
class SpectrogramBlock:
    """One short-time power spectrogram and the transform it was taken under.

    Attributes:
        spectrogram: Power, ``(bin, frame)``.
        n_fft: The transform length, which fixes the bin centres.
        hop_length: The hop, in samples of the working rate.
        sampling_rate: The working rate the hop and the bins are in, in Hz. Not an attribute of the
            stored derivative: a reader takes it from the resample.
    """

    spectrogram: np.ndarray
    n_fft: int
    hop_length: int
    sampling_rate: float


def derivative_arrays(store: ProvStore, run_dir: Path, name: str) -> dict[str, np.ndarray] | None:
    """Every array of one persisted derivative, or None when it never reached the store.

    Args:
        store: The provenance store.
        run_dir: The run directory sidecar paths are relative to.
        name: The measurement's name, e.g. ``"energy_envelope"``.

    Returns:
        The arrays by key, or None when the measurement, its path or the file is absent.
    """
    measurement = find_measurement(store, name)
    if measurement is None:
        return None
    relative = measurement.attributes.get("path")
    if not relative:
        return None
    sidecar = run_dir / str(relative)
    if not sidecar.is_file():
        return None
    with np.load(sidecar) as loaded:
        return {key: np.asarray(loaded[key]) for key in loaded.files}


def read_envelope_track(store: ProvStore, run_dir: Path, name: str = "energy_envelope") -> EnvelopeTrack | None:
    """The energy envelope, its global floor and its rate.

    Args:
        store: The provenance store.
        run_dir: The run directory sidecar paths are relative to.
        name: The measurement's name; ``"normalized_envelope"`` is the AGC'd one.

    Returns:
        The track, or None when the derivative or its rate is absent.
    """
    arrays = derivative_arrays(store, run_dir, name)
    measurement = find_measurement(store, name)
    if arrays is None or measurement is None or "envelope_dbfs" not in arrays:
        return None
    rate = measurement.attributes.get("sampling_rate")
    if rate is None:
        return None
    floor = np.asarray(arrays.get("floor_dbfs", np.array([])), dtype=float)
    return EnvelopeTrack(
        envelope_dbfs=np.asarray(arrays["envelope_dbfs"], dtype=float),
        floor_dbfs=float(floor.reshape(-1)[0]) if floor.size else float("nan"),
        sampling_rate=float(rate),
    )


def read_continuity_track(store: ProvStore, run_dir: Path) -> ContinuityTrack | None:
    """The spectral-stationarity trace and its rate.

    Args:
        store: The provenance store.
        run_dir: The run directory sidecar paths are relative to.

    Returns:
        The track, or None when the derivative or its rate is absent.
    """
    arrays = derivative_arrays(store, run_dir, "continuity_trace")
    measurement = find_measurement(store, "continuity_trace")
    if arrays is None or measurement is None or "continuity" not in arrays:
        return None
    rate = measurement.attributes.get("sampling_rate")
    if rate is None:
        return None
    return ContinuityTrack(continuity=np.asarray(arrays["continuity"], dtype=float), sampling_rate=float(rate))


def read_phonation_tracks(store: ProvStore, run_dir: Path) -> PhonationTracks | None:
    """F0, its strength and the frame centres.

    Args:
        store: The provenance store.
        run_dir: The run directory sidecar paths are relative to.

    Returns:
        The tracks, or None when the derivative is absent or incomplete.
    """
    arrays = derivative_arrays(store, run_dir, "phonation_tracks")
    if arrays is None or not {"times_s", "f0_hz", "strength"} <= set(arrays):
        return None
    return PhonationTracks(
        times_s=np.asarray(arrays["times_s"], dtype=float),
        f0_hz=np.asarray(arrays["f0_hz"], dtype=float),
        strength=np.asarray(arrays["strength"], dtype=float),
    )


def read_spectrogram_block(store: ProvStore, run_dir: Path, name: str, sampling_rate: float) -> SpectrogramBlock | None:
    """One spectrogram and the transform it was taken under.

    Args:
        store: The provenance store.
        run_dir: The run directory sidecar paths are relative to.
        name: ``"spectrogram_wideband"`` or ``"spectrogram_narrowband"``.
        sampling_rate: The working rate, which the derivative does not record.

    Returns:
        The block, or None when the derivative or either transform attribute is absent.
    """
    arrays = derivative_arrays(store, run_dir, name)
    measurement = find_measurement(store, name)
    if arrays is None or measurement is None or "spectrogram" not in arrays:
        return None
    n_fft, hop_length = measurement.attributes.get("n_fft"), measurement.attributes.get("hop_length")
    if n_fft is None or hop_length is None:
        return None
    return SpectrogramBlock(
        spectrogram=np.asarray(arrays["spectrogram"], dtype=float),
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        sampling_rate=float(sampling_rate),
    )


# --------------------------------------------------------------------- array helpers


def envelope_slice(envelope: EnvelopeTrack, extent: tuple[float, float]) -> tuple[np.ndarray, int]:
    """The envelope over one extent, and where it starts.

    Args:
        envelope: The envelope track.
        extent: The extent, in seconds.

    Returns:
        The slice and its first sample index, so a found index can be put back into seconds. The
        slice is empty when the extent falls outside the envelope.
    """
    rate = float(envelope.sampling_rate)
    values = np.asarray(envelope.envelope_dbfs, dtype=float)
    lo = max(0, int(round(extent[0] * rate)))
    hi = min(values.size, int(round(extent[1] * rate)))
    if hi <= lo:
        return np.empty(0, dtype=float), lo
    return values[lo:hi], lo


def trace_slice(trace: ContinuityTrack, extent: tuple[float, float]) -> np.ndarray:
    """The continuity trace over one extent.

    Args:
        trace: The continuity track.
        extent: The extent, in seconds.

    Returns:
        The slice, empty when the extent falls outside the trace.
    """
    rate = float(trace.sampling_rate)
    values = np.asarray(trace.continuity, dtype=float)
    lo = max(0, int(round(extent[0] * rate)))
    hi = min(values.size, int(round(extent[1] * rate)))
    return values[lo:hi] if hi > lo else np.empty(0, dtype=float)


def boxcar(x: np.ndarray, width: int) -> np.ndarray:
    """A moving average, the same length as its input.

    Args:
        x: The values.
        width: The window, in samples. One or less is a no-op. An even width is raised to the next
            odd one, so the average stays centred on a sample.

    Returns:
        The smoothed values.
    """
    if width <= 1 or x.size == 0:
        return x
    width = min(width, x.size)
    if width % 2 == 0:
        width = min(width + 1, x.size)
    return np.convolve(x, np.ones(width, dtype=float) / float(width), mode="same")


@dataclass(frozen=True)
class TrackSlice:
    """The phonation tracks over one extent, with voicing already decided.

    Attributes:
        times_s: Frame centres inside the extent.
        f0_hz: F0 per frame.
        strength: Pitch strength per frame.
        voiced: Whether each frame cleared the strength minimum.
        hop_s: The median hop of the frames, in seconds.
    """

    times_s: np.ndarray
    f0_hz: np.ndarray
    strength: np.ndarray
    voiced: np.ndarray
    hop_s: float


def track_slice(tracks: PhonationTracks, extent: tuple[float, float], p_voiced_strength_min: float) -> TrackSlice:
    """The phonation tracks over one extent.

    Args:
        tracks: The whole-recording tracks.
        extent: The extent, in seconds.
        p_voiced_strength_min: Pitch strength at or above which a frame counts as voiced.

    Returns:
        The slice.
    """
    times = np.asarray(tracks.times_s, dtype=float)
    inside = (times >= extent[0]) & (times < extent[1])
    hop = float(np.median(np.diff(times))) if times.size > 1 else 0.01
    strength = np.asarray(tracks.strength, dtype=float)[inside]
    return TrackSlice(
        times_s=times[inside],
        f0_hz=np.asarray(tracks.f0_hz, dtype=float)[inside],
        strength=strength,
        voiced=strength >= p_voiced_strength_min,
        hop_s=hop,
    )


def voiced_extent(span_extent: tuple[float, float], track: TrackSlice) -> tuple[float, float]:
    """A production's own boundaries: the first and last voiced frame inside its carrier.

    Args:
        span_extent: The carrier span's extent, returned when nothing inside it is voiced.
        track: The tracks over that carrier.

    Returns:
        The voiced extent.
    """
    inside = track.times_s[track.voiced]
    if inside.size == 0:
        return span_extent
    return (float(inside.min()), float(inside.max()) + track.hop_s)


def semitones(f0_hz: np.ndarray, ref_hz: float | None = None) -> np.ndarray:
    """F0 as semitones about a reference, with the unvoiced frames left as NaN.

    Args:
        f0_hz: F0 per frame.
        ref_hz: The reference. None takes the median of the voiced frames.

    Returns:
        Semitones, NaN where F0 is non-finite or not positive.
    """
    f0 = np.asarray(f0_hz, dtype=float)
    voiced = np.isfinite(f0) & (f0 > 0.0)
    if not voiced.any():
        return np.full(f0.shape, np.nan)
    if ref_hz is None:
        ref_hz = float(np.median(f0[voiced]))
    out = np.full(f0.shape, np.nan)
    out[voiced] = 12.0 * np.log2(f0[voiced] / ref_hz)
    return out


def robust_spread(values: np.ndarray) -> float:
    """The 5th-to-95th percentile range, which one outlying frame cannot set.

    Args:
        values: The values; non-finite ones are dropped.

    Returns:
        The spread, or 0.0 when fewer than two values are finite.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return 0.0
    return float(np.percentile(finite, 95.0) - np.percentile(finite, 5.0))


def max_windowed_spread(values: np.ndarray, hop_s: float, window_s: float) -> float:
    """The worst local spread, so a slow drift over a long production does not read as instability.

    Args:
        values: The values, one per hop.
        hop_s: The hop, in seconds.
        window_s: The window, in seconds.

    Returns:
        The largest :func:`robust_spread` over any window, or the whole-series spread when the
        series is no longer than one window.
    """
    series = np.asarray(values, dtype=float)
    width = max(2, int(round(window_s / hop_s))) if hop_s > 0.0 else 2
    if series.size <= width:
        return robust_spread(series)
    worst = 0.0
    for first in range(0, series.size - width + 1):
        worst = max(worst, robust_spread(series[first : first + width]))
    return worst


def longest_monotone_run(values: np.ndarray, tolerance: float) -> tuple[int, int, int] | None:
    """The longest run that never reverses by more than ``tolerance``.

    Args:
        values: The values; non-finite ones are skipped rather than breaking the run.
        tolerance: How far the series may go back on itself.

    Returns:
        ``(first index, last index, sign)``, or None when fewer than two values are finite.
    """
    series = np.asarray(values, dtype=float)
    finite = np.flatnonzero(np.isfinite(series))
    if finite.size < 2:
        return None
    best: tuple[int, int, int] | None = None
    for sign in (1, -1):
        run_start = int(finite[0])
        extreme = float(series[run_start])
        previous = run_start
        candidates: list[tuple[int, int, int]] = []
        for raw in finite[1:]:
            index = int(raw)
            value = float(series[index])
            if sign * (value - extreme) >= -tolerance:
                extreme = max(extreme, value) if sign > 0 else min(extreme, value)
            else:
                candidates.append((run_start, previous, sign))
                run_start, extreme = index, value
            previous = index
        candidates.append((run_start, previous, sign))
        for candidate in candidates:
            if best is None or (candidate[1] - candidate[0]) > (best[1] - best[0]):
                best = candidate
    return best


def band_power(block: SpectrogramBlock, extent: tuple[float, float], lo_hz: float, hi_hz: float) -> float:
    """The power one band carries over one extent.

    Args:
        block: The spectrogram and its transform.
        extent: The extent, in seconds.
        lo_hz: Band start, inclusive.
        hi_hz: Band end, exclusive.

    Returns:
        The summed power, or NaN when the extent or the band selects nothing.
    """
    power = np.asarray(block.spectrogram, dtype=float)
    freqs = np.fft.rfftfreq(int(block.n_fft), d=1.0 / block.sampling_rate)
    bins = (freqs >= lo_hz) & (freqs < hi_hz)
    hop_s = float(block.hop_length) / block.sampling_rate
    first = max(0, int(extent[0] / hop_s))
    last = min(power.shape[1], int(np.ceil(extent[1] / hop_s)))
    if last <= first or not bins.any():
        return float("nan")
    return float(power[np.ix_(bins[: power.shape[0]], np.arange(first, last))].sum())


def spectral_balance_db(block: SpectrogramBlock, extent: tuple[float, float], split_hz: float) -> float:
    """High-band power over low-band power, in dB.

    Args:
        block: The spectrogram and its transform.
        extent: The extent, in seconds.
        split_hz: Where the two bands meet.

    Returns:
        The balance in dB, or NaN when either band is unreadable or the low band is empty.
    """
    low = band_power(block, extent, 0.0, split_hz)
    high = band_power(block, extent, split_hz, block.sampling_rate / 2.0)
    if not np.isfinite(low) or not np.isfinite(high) or low <= 0.0:
        return float("nan")
    return float(10.0 * np.log10((high + 1e-20) / low))


def peak_over_floor_db(envelope: EnvelopeTrack, extent: tuple[float, float]) -> float:
    """How far the loudest sample of an extent stands over the recording's global floor.

    Args:
        envelope: The envelope track.
        extent: The extent, in seconds.

    Returns:
        The rise in dB, or NaN when the extent falls outside the envelope.
    """
    values, _ = envelope_slice(envelope, extent)
    if values.size == 0:
        return float("nan")
    return float(values.max() - float(envelope.floor_dbfs))


def acquisition_covariates(store: ProvStore, extent: tuple[float, float]) -> dict[str, Any]:
    """What every acoustic measurement must be read against.

    Args:
        store: The provenance store.
        extent: The extent the measurement was taken over.

    Returns:
        The whole-file level, whether the extent overlaps a clipped span, the file's disruptions,
        and ``uncalibrated``, which is always true: no SPL reference exists anywhere in the graph.
        A level or disruption reading the store does not hold comes back None.
    """
    level = find_measurement(store, "level")
    disruptions = find_measurement(store, "disruptions_file")
    spans = live_entities(store, "span")
    return {
        "file_peak_dbfs": None if level is None else level.attributes.get("peak_dbfs"),
        "file_rms_dbfs": None if level is None else level.attributes.get("rms_dbfs"),
        "file_lufs": None if level is None else level.attributes.get("lufs"),
        "contains_clip": any(
            span.attributes.get("contains_clip") and span.extent is not None and overlaps(span.extent, extent)
            for span in spans
        ),
        "disruptions": None if disruptions is None else disruptions.attributes.get("summary"),
        "uncalibrated": True,
    }


# --------------------------------------------------------------------- the three instruments


def events_in_extent(
    envelope: EnvelopeTrack, extent: tuple[float, float], params: BranchParams
) -> list[tuple[float, float]]:
    """Every event inside one extent, by peak prominence and a trough-return walk.

    Multiple maxima inside one carrier become separate events, which is the whole point: a span
    holding three coughs is three events, not one.

    Args:
        envelope: The envelope track.
        extent: The extent to look inside, in seconds.
        params: The operating points.

    Returns:
        The events, coalesced, earliest first. Empty when one of the walk's own operating points is
        unmeasured: the walk cannot be taken without it, and a walk taken on a substituted number
        would report events nobody's setting found. The ask is recorded in ``params.missing``.
    """
    window_s = params.point("smoothing_window_s")
    prominence = params.point("peak_prominence_db")
    return_db = params.point("trough_return_db")
    minimum_s = params.point("event_min_s")
    if window_s is None or prominence is None or return_db is None or minimum_s is None:
        return []
    rate = float(envelope.sampling_rate)
    raw, offset = envelope_slice(envelope, extent)
    if raw.size < 3:
        return []
    smoothed = boxcar(raw, max(1, int(round(window_s * rate))))
    floor_dbfs = float(envelope.floor_dbfs)

    events: list[tuple[float, float]] = []
    for index in range(1, smoothed.size - 1):
        if not (smoothed[index] >= smoothed[index - 1] and smoothed[index] > smoothed[index + 1]):
            continue
        if smoothed[index] - floor_dbfs < prominence:
            continue
        left, left_min = index - 1, float(smoothed[index])
        while left >= 0 and smoothed[left] < smoothed[index]:
            left_min = min(left_min, float(smoothed[left]))
            left -= 1
        right, right_min = index + 1, float(smoothed[index])
        while right < smoothed.size and smoothed[right] < smoothed[index]:
            right_min = min(right_min, float(smoothed[right]))
            right += 1
        if float(smoothed[index]) - max(left_min, right_min) < prominence:
            continue
        target = float(smoothed[index]) - return_db
        onset = index
        while onset > 0 and smoothed[onset - 1] > target:
            onset -= 1
        tail = index
        while tail < smoothed.size - 1 and smoothed[tail + 1] > target:
            tail += 1
        start = (offset + onset) / rate
        end = (offset + tail) / rate
        if end - start >= minimum_s:
            events.append((start, end))
    return merge(events)


def events_in_span(envelope: EnvelopeTrack, span: Entity, params: BranchParams) -> list[tuple[float, float]]:
    """Every event inside one carrier span.

    Args:
        envelope: The envelope track.
        span: The carrier span.
        params: The operating points.

    Returns:
        The events, or nothing when the span carries no extent.
    """
    if span.extent is None:
        return []
    return events_in_extent(envelope, span.extent, params)


def sounds_like(span: Entity, windows: Sequence[Entity], label_set: Sequence[str], p_score_min: float) -> bool:
    """Whether any classifier window over this span scored one of these labels.

    Reads ``raw_scores``, which PREPROCESS always writes, rather than ``labels``, which is a top-K
    decision over it that the shipped config leaves unmade for two of the three classifiers.

    Args:
        span: The span.
        windows: The per-span classifier windows, e.g. ``span_hear`` and ``span_yamnet`` together.
        label_set: The labels that are this sound.
        p_score_min: Score at or above which a label is present.

    Returns:
        True when one window over this span cleared the minimum on one of the labels.
    """
    wanted = set(label_set)
    for window in windows:
        if window.attributes.get("span_id") != span.id:
            continue
        scores = window.attributes.get("raw_scores") or {}
        for label in wanted:
            if float(scores.get(label, 0.0)) >= p_score_min:
                return True
    return False


def train_rate_hz(envelope: EnvelopeTrack, extent: tuple[float, float], params: BranchParams) -> float | None:
    """The rate of a repetition train, as the peak of its envelope's modulation spectrum.

    Args:
        envelope: The envelope track.
        extent: The extent, in seconds.
        params: The operating points.

    Returns:
        The rate in Hz, or None when the extent is too short, the band selects nothing, the peak
        does not stand over its own band's mean, or one of the two operating points the search
        needs is unmeasured. The ask is recorded in ``params.missing``.
    """
    band_hz = params.point("modulation_band_hz")
    prominence_min = params.point("rate_prominence_min")
    if band_hz is None or prominence_min is None:
        return None
    rate = float(envelope.sampling_rate)
    values, _ = envelope_slice(envelope, extent)
    if values.size < 8:
        return None
    windowed = (values - values.mean()) * np.hanning(values.size)
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(values.size, d=1.0 / rate)
    lo, hi = band_hz
    band = (freqs >= lo) & (freqs <= hi)
    if not band.any():
        return None
    peak = int(np.argmax(np.where(band, spectrum, 0.0)))
    background = float(spectrum[band].mean())
    if background <= 0.0 or float(spectrum[peak]) / background < prominence_min:
        return None
    return float(freqs[peak])


# --------------------------------------------------------------------- lexical arithmetic


def ordered_run(
    expected: Sequence[str], words: Sequence[Entity], normalise: Callable[[str], str]
) -> tuple[list[tuple[str, Entity]], list[str]]:
    """A greedy left-to-right ordered match: the fallback where no stimulus alignment exists.

    Args:
        expected: The tokens the instruction prescribes, in order.
        words: The lexical consensus words, in index order.
        normalise: How a token is compared.

    Returns:
        The matched ``(token, word)`` pairs and the tokens nothing realised.
    """
    matched: list[tuple[str, Entity]] = []
    omissions: list[str] = []
    cursor = 0
    for token in expected:
        hit: int | None = None
        for index in range(cursor, len(words)):
            if normalise(word_text(words[index])) == normalise(token):
                hit = index
                break
        if hit is None:
            omissions.append(token)
        else:
            matched.append((token, words[hit]))
            cursor = hit + 1
    return matched, omissions


def ngram_echo_fraction(source_tokens: Sequence[str], produced_tokens: Sequence[str], n: int) -> float:
    """What fraction of the source's n-grams the production reproduces.

    Args:
        source_tokens: The source's tokens, normalised.
        produced_tokens: The production's tokens, normalised.
        n: The n-gram length.

    Returns:
        The fraction, or 0.0 when either side is shorter than one n-gram.
    """
    if len(source_tokens) < n or len(produced_tokens) < n:
        return 0.0
    source = {tuple(source_tokens[i : i + n]) for i in range(len(source_tokens) - n + 1)}
    produced = {tuple(produced_tokens[i : i + n]) for i in range(len(produced_tokens) - n + 1)}
    if not source:
        return 0.0
    return len(source & produced) / len(source)


def content_coverage(source_tokens: Sequence[str], produced_tokens: Sequence[str]) -> float:
    """What fraction of the source's distinct tokens appear in the production.

    Args:
        source_tokens: The source's tokens, normalised.
        produced_tokens: The production's tokens, normalised.

    Returns:
        The fraction, or 0.0 when the source has no tokens.
    """
    source = set(source_tokens)
    if not source:
        return 0.0
    return len(source & set(produced_tokens)) / len(source)


def lexical_runs(words: Sequence[Entity], max_gap_s: float) -> list[tuple[float, float]]:
    """Consecutive lexical words separated by no more than ``max_gap_s``, as one extent each.

    Not :func:`merge` over the word extents: ``merge`` joins only touching or overlapping intervals,
    and ordinary speech has a gap between every pair of words, so ``merge`` returns one extent per
    word. This is the grouping SPEECH already does over the consensus word timings.

    Args:
        words: The lexical consensus words, in index order.
        max_gap_s: Largest gap two words may straddle and stay one run.

    Returns:
        The runs, in the order the words were given.
    """
    runs: list[tuple[float, float]] = []
    for word in words:
        start, end = word_extent(word)
        if runs and start - runs[-1][1] <= max_gap_s:
            runs[-1] = (runs[-1][0], max(runs[-1][1], end))
        else:
            runs.append((start, end))
    return runs


def inter_word_gaps(words: Sequence[Entity], min_gap_s: float) -> list[tuple[float, float]]:
    """The gaps between consecutive words that are at least ``min_gap_s`` long.

    Args:
        words: The lexical consensus words, in index order.
        min_gap_s: Shortest gap reported.

    Returns:
        One extent per qualifying gap.
    """
    out: list[tuple[float, float]] = []
    for previous, current in zip(words, words[1:]):
        end = word_extent(previous)[1]
        start = word_extent(current)[0]
        if start - end >= min_gap_s:
            out.append((end, start))
    return out


def group_by_breaks(words: Sequence[Entity], breaks: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Group words into extents separated by the given breaks.

    Args:
        words: The lexical consensus words, in index order.
        breaks: The extents that end a group.

    Returns:
        One extent per group, or nothing when there are no words.
    """
    if not words:
        return []
    groups: list[tuple[float, float]] = []
    start = word_extent(words[0])[0]
    for previous, current in zip(words, words[1:]):
        previous_end = word_extent(previous)[1]
        between = (previous_end, max(word_extent(current)[0], previous_end + 1e-9))
        if any(overlaps(each, between) for each in breaks):
            groups.append((start, previous_end))
            start = word_extent(current)[0]
    groups.append((start, word_extent(words[-1])[1]))
    return groups


def token_occurrences(words: Sequence[Entity], normalise: Callable[[str], str]) -> dict[str, list[Entity]]:
    """The words each normalised token was realised by.

    Args:
        words: The lexical consensus words, in index order.
        normalise: How a token is compared.

    Returns:
        Normalised token to its words, in index order.
    """
    out: dict[str, list[Entity]] = {}
    for word in words:
        out.setdefault(normalise(word_text(word)), []).append(word)
    return out


def relax(expectation: Expectation, extent: tuple[float, float]) -> tuple[float, float]:
    """The part of an extent the instruction gives to the task rather than to settling.

    Args:
        expectation: The row, whose ``relax_s`` is the leading interval to drop.
        extent: The extent.

    Returns:
        The extent with ``relax_s`` removed from its start, or unchanged when the row declares none
        or the removal would leave nothing.
    """
    if expectation.relax_s is None:
        return extent
    start = extent[0] + float(expectation.relax_s)
    return (start, extent[1]) if extent[1] > start else extent


def without_unviable(expectation: Expectation) -> Expectation:
    """The same row with its unviable entries dropped, for a caller that has already emitted them.

    Args:
        expectation: The row.

    Returns:
        The row with ``unviable`` empty.
    """
    return replace(expectation, unviable=())


def unviable_findings(expectation: Expectation) -> list[Finding]:
    """One finding per measurement this row states no viable approach for.

    Args:
        expectation: The row.

    Returns:
        The findings, so a measurement the design cannot take is recorded rather than omitted.
    """
    return [unviable(name, why) for name, why in expectation.unviable]
