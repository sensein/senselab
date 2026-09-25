"""The REVIEW node: one instruction-tuned reader over every transcript that could be released.

Reached from the transcript, not from the scan. SPEECH declines the PII scan wherever every lexical
word sits in the task's own stimulus, so a reviewer behind that gate can check the detectors' output
but never the gate itself; REVIEW runs wherever PREPROCESS left consensus words, whatever the
detectors did or did not do.

It reads two texts -- the recording's words, and the text an applied redaction produced where one
exists -- and answers three questions independently: whether that redaction removed what identifies
the speaker, whether the words carry anything identifying at all, and whether they show more than
one person speaking in the recording. Beside them it writes a proposal: the redaction it would
apply instead, in either direction.

It writes no verdict. Its summary is a ``redaction_llm_annotation`` measurement, which VERDICT reads
under the ``verdict.llm_redaction_*`` keys. A proposal is applied, if at all, by
:func:`apply_proposal`, which writes the recording's one redacted stream from the refined span set;
the detector-derived version is not kept, and the store's edges are what say which span set the
audio came from and why each span is or is not in it.

See ``specs/20260924-reviewer-over-every-transcript/design.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.branches import declared_task_family, expected_names
from senselab.audio.workflows.triage.nodes.common import (
    find_measurement,
    find_verdict,
    software_agent,
)
from senselab.audio.workflows.triage.nodes.redact import transcript_texts
from senselab.audio.workflows.triage.vocabulary import PII_SCAN, REDACTION_LLM_ANNOTATION, SCANNED
from senselab.text.tasks.pii_detection.redaction_review import (
    REDACT,
    ReviewProposal,
    review_payload,
    review_transcript,
    shutdown_review_worker,
)
from senselab.utils.prov_store import ProvStore

NODE = "REVIEW"

_LLM_SECTION = "redaction.llm_check"
_LLM_REVIEW_MEASUREMENT = "redaction_llm_review"
_LLM_PLACEHOLDER = "[LLM_{category}]"  # what a proposed redaction is masked with inside the loop

SCANNED_STATE = "scanned"
DECLINED_STATE = "declined"
UNSCANNED_STATE = "unscanned"
DETECTOR_STATES = (SCANNED_STATE, DECLINED_STATE, UNSCANNED_STATE)
"""What the detectors did to this recording: read it, declined to read it, or never reached it."""

DISABLED = "disabled"
NOTHING_TO_READ = "nothing_to_read"
ABSENT = "absent"
CLEAN = "clean"
FLAGGED = "flagged"
REVIEW_STATES = (DISABLED, NOTHING_TO_READ, ABSENT, CLEAN, FLAGGED)
"""Switched off, nothing to read, tried and could not load, and the two readings it can reach."""


@dataclass(frozen=True)
class ReviewOutcome:
    """What REVIEW returns. It carries no verdict, because it decides nothing.

    Attributes:
        status: One of :data:`REVIEW_STATES`.
        annotation_id: The ``redaction_llm_annotation`` entity it wrote, on every path.
        view: Ids of the store entities this node wrote.
    """

    status: str
    annotation_id: str
    view: tuple[str, ...]


@dataclass(frozen=True)
class _Reading:
    """One completed pass of the bounded loop, as the annotation records it.

    Attributes:
        status: One of :data:`REVIEW_STATES`.
        iterations: How many reviews ran.
        flagged: The categories the proposal named for removal, sorted.
        redaction: The model's judgment on the applied redaction, or the empty string.
        original: Its judgment on the recording's own words, or the empty string.
        speakers: Its judgment on how many people the words show speaking, or the empty string.
        proposal: The last round's proposal, as the mappings the annotation carries.
        model_id: The repo asked, or the empty string when nothing was.
        revision: The commit the reviewer loaded, or None.
        failure: Why it did not run, when it did not.
    """

    status: str
    iterations: int
    flagged: tuple[str, ...]
    redaction: str
    original: str
    speakers: str
    proposal: tuple[dict[str, str], ...]
    model_id: str
    revision: str | None
    failure: str | None


def _stamp() -> str:
    """Now, as the ISO 8601 UTC string an activity's ``started``/``ended`` carries.

    Returns:
        The timestamp.
    """
    return datetime.now(timezone.utc).isoformat()


def _llm_settings(config: TriageConfig) -> dict[str, Any]:
    """Every ``redaction.llm_check`` key, read in one place.

    Args:
        config: The triage configuration.

    Returns:
        The settings.

    Raises:
        ValueError: If a key is unmeasured. They are read together, before any model is contacted.
    """
    names = (
        "enabled",
        "model_id",
        "ref",
        "max_iterations",
        "max_new_tokens",
        "timeout_s",
        "keep_worker_resident",
    )
    return {name: config.require(f"{_LLM_SECTION}.{name}") for name in names}


def task_context(store: ProvStore, hint: AudioHints | None) -> dict[str, Any]:
    """What this recording declares about itself, for the reviewer to reason with.

    Args:
        store: The provenance store.
        hint: The caller's declaration, or None.

    Returns:
        ``task``, ``asked_to_say`` and ``declared_names``, each omitted where the recording
        declares none. Facts about what was asked for; the prompt says outright that they settle
        nothing, because a task whose materials contain names is not a task in which every name is
        the task's.
    """
    family = declared_task_family(store, hint)
    prompts = [
        expected.text for expected in (getattr(hint, "expected_speech", None) or []) if getattr(expected, "text", None)
    ]
    context: dict[str, Any] = {}
    if family:
        context["task"] = str(family)
    if prompts:
        context["asked_to_say"] = " ".join(str(prompt) for prompt in prompts)
    names = expected_names(family)
    if names:
        context["declared_names"] = list(names)
    return context


def detector_state(store: ProvStore) -> str:
    """What the PII detectors did to this recording.

    Args:
        store: The provenance store.

    Returns:
        One of :data:`DETECTOR_STATES`. ``declined`` is the population SPEECH's stimulus gate
        excluded, which is the one a reading over the transcript exists to reach.
    """
    scan = find_measurement(store, PII_SCAN)
    if scan is None:
        return UNSCANNED_STATE
    return DECLINED_STATE if scan.attributes.get(SCANNED) is False else SCANNED_STATE


def _redact_outcome(store: ProvStore) -> str:
    """REDACT's own outcome, where it left one.

    Args:
        store: The provenance store.

    Returns:
        The outcome's value, or the empty string where REDACT wrote no verdict. Not having run is
        not a pass, so it is not reported as one.
    """
    verdict = find_verdict(store, "REDACT")
    return str(verdict.attributes.get("outcome") or "") if verdict is not None else ""


def _findings_n(store: ProvStore) -> int:
    """How many live ``pii`` findings the detectors left.

    Args:
        store: The provenance store.

    Returns:
        The count. Zero beside a ``flagged`` reading is a contradicted clean scan.
    """
    return len([entity for entity in store.entities("pii") if not store.is_invalidated(entity.id)])


def _mask(text: str, proposal: Sequence[ReviewProposal]) -> str:
    """Replace each proposed removal with its placeholder, for the next round only.

    Args:
        text: The text the round read.
        proposal: That round's proposal.

    Returns:
        The text the next round reads. A ``release`` entry masks nothing -- it asks for less to be
        removed, and re-reading with it applied would hide the very question it raises.
    """
    masked = text
    for entry in proposal:
        if entry.action == REDACT and entry.text and entry.text in masked:
            masked = masked.replace(entry.text, _LLM_PLACEHOLDER.format(category=entry.category))
    return masked


def _entries(proposal: Sequence[ReviewProposal]) -> tuple[dict[str, str], ...]:
    """One proposal as the mappings the annotation carries.

    Args:
        proposal: The model's proposal.

    Returns:
        The mappings, in order.
    """
    return tuple(
        {"text": entry.text, "action": entry.action, "category": entry.category, "why": entry.why} for entry in proposal
    )


def _is_flag(result: Any) -> bool:  # noqa: ANN401 — the backend's own result type
    """Whether a reading says something identifying is still there.

    Args:
        result: One round's result.

    Returns:
        True where the redaction was judged incomplete, the words were judged to carry PII, or the
        proposal asks for a removal. A proposal of only ``release`` entries is not a flag: it asks
        for less to be removed, which is not a leak.
    """
    return (
        result.redaction == "incomplete"
        or result.original == "carries_pii"
        or any(entry.action == REDACT for entry in result.proposal)
    )


def _rounds(
    original: str, redacted: str | None, settings: Mapping[str, Any], context: Mapping[str, Any]
) -> tuple[_Reading, list[dict[str, Any]]]:
    """The bounded review / mask / re-review loop, without the worker's lifetime.

    Args:
        original: The recording's words.
        redacted: The text an applied redaction produced, or None where none was applied.
        settings: :func:`_llm_settings`' mapping.
        context: :func:`task_context`'s mapping.

    Returns:
        ``(reading, reviews)`` -- the summary and one payload per round, in order.
    """
    reviews: list[dict[str, Any]] = []
    current = redacted
    flagged: list[str] = []
    model_id = str(settings["model_id"])
    revision: str | None = None
    last: Any = None
    for iteration in range(1, int(settings["max_iterations"]) + 1):
        result = review_transcript(
            original,
            redacted=current,
            context=context,
            model_id=model_id,
            ref=str(settings["ref"]),
            max_new_tokens=int(settings["max_new_tokens"]),
            timeout_s=int(settings["timeout_s"]),
        )
        reviews.append({**review_payload(result), "iteration": iteration})
        revision = result.revision or revision
        if not result.available:
            status = FLAGGED if flagged else ABSENT
            return (
                _Reading(
                    status,
                    iteration,
                    tuple(sorted(set(flagged))),
                    last.redaction if last is not None else "",
                    last.original if last is not None else "",
                    last.speakers if last is not None else "",
                    _entries(last.proposal) if last is not None else (),
                    model_id,
                    revision,
                    result.failure,
                ),
                reviews,
            )
        last = result
        if not _is_flag(result):
            return (
                _Reading(
                    FLAGGED if flagged else CLEAN,
                    iteration,
                    tuple(sorted(set(flagged))),
                    result.redaction,
                    result.original,
                    result.speakers,
                    _entries(result.proposal),
                    model_id,
                    revision,
                    None,
                ),
                reviews,
            )
        flagged.extend(entry.category for entry in result.proposal if entry.action == REDACT)
        current = _mask(current if current is not None else original, result.proposal)
    return (
        _Reading(
            FLAGGED,
            int(settings["max_iterations"]),
            tuple(sorted(set(flagged))),
            last.redaction if last is not None else "",
            last.original if last is not None else "",
            last.speakers if last is not None else "",
            _entries(last.proposal) if last is not None else (),
            model_id,
            revision,
            None,
        ),
        reviews,
    )


def _read(
    original: str, redacted: str | None, settings: Mapping[str, Any], context: Mapping[str, Any]
) -> tuple[_Reading, list[dict[str, Any]]]:
    """The loop, with the worker released afterwards unless the run asked to keep it.

    Args:
        original: The recording's words.
        redacted: The text an applied redaction produced, or None.
        settings: :func:`_llm_settings`' mapping.
        context: :func:`task_context`'s mapping.

    Returns:
        :func:`_rounds`' pair.
    """
    try:
        return _rounds(original, redacted, settings, context)
    finally:
        if not settings["keep_worker_resident"]:
            shutdown_review_worker(forget_failure=False)


def review(store: ProvStore, config: TriageConfig, hint: AudioHints | None = None) -> ReviewOutcome:
    """Read this recording's transcript back and record what the reader made of it.

    Takes no run directory and opens no audio: the store carries both texts, which is what lets a
    review-only driver run over a finished corpus without replaying the graph.

    Args:
        store: The provenance store.
        config: The triage configuration.
        hint: The caller's declaration, which is where the task's own facts come from.

    Returns:
        The outcome. An annotation is written on every path, including the paths where no model was
        contacted, so that "no review happened" is a record rather than an absence.

    Raises:
        ValueError: If any ``redaction.llm_check`` key is unmeasured. They are read together,
            before any model is contacted.
    """
    settings = _llm_settings(config)
    context = task_context(store, hint)
    original, redacted = transcript_texts(store)
    state = detector_state(store)
    findings_n = _findings_n(store)
    outcome = _redact_outcome(store)

    started = _stamp()
    reviews: list[dict[str, Any]] = []
    if not settings["enabled"]:
        reading = _Reading(DISABLED, 0, (), "", "", "", (), "", None, None)
    elif not original.strip():
        reading = _Reading(
            NOTHING_TO_READ, 0, (), "", "", "", (), "", None, "the transcript is empty; there was no text to read"
        )
    else:
        reading, reviews = _read(original, redacted, settings, context)

    software = software_agent(store)
    view: list[str] = []
    activity = store.activity(
        node=NODE,
        step="llm_check",
        parameters={
            "model_id": reading.model_id or str(settings["model_id"]),
            "max_iterations": int(settings["max_iterations"]),
            "enabled": bool(settings["enabled"]),
            "read_redacted": redacted is not None,
            "context_keys": sorted(context),
        },
        started=started,
        ended=_stamp(),
    )
    store.was_associated_with(activity, software)
    if reading.status not in (DISABLED, NOTHING_TO_READ):
        if reading.revision is not None:
            store.was_associated_with(
                activity, store.agent(agent_type="model", model_id=reading.model_id, commit_sha=reading.revision)
            )
        else:
            store.was_associated_with(
                activity,
                store.agent(
                    agent_type="model",
                    model_id=reading.model_id,
                    unresolved_reason=reading.failure or "the reviewer did not load",
                ),
            )
        consensus = find_measurement(store, "consensus_transcript")
        if consensus is not None:
            store.used(activity, consensus.id)
        for review_payload_ in reviews:
            review_id = store.entity(
                prov_type="measurement",
                extent=None,
                attributes={"name": _LLM_REVIEW_MEASUREMENT, "signal": "consensus_transcript", **review_payload_},
            )
            store.was_generated_by(review_id, activity)
            store.was_attributed_to(review_id, software)
            view.append(review_id)

    annotation_id = store.entity(
        prov_type="measurement",
        extent=None,
        attributes={
            "name": REDACTION_LLM_ANNOTATION,
            "signal": "consensus_transcript",
            "status": reading.status,
            "iterations": reading.iterations,
            "flagged": list(reading.flagged),
            "redaction": reading.redaction,
            "original": reading.original,
            "speakers": reading.speakers,
            "proposal": [dict(entry) for entry in reading.proposal],
            "proposal_redact_n": sum(1 for entry in reading.proposal if entry["action"] == REDACT),
            "proposal_release_n": sum(1 for entry in reading.proposal if entry["action"] != REDACT),
            "model_id": reading.model_id,
            "revision": reading.revision,
            "failure": reading.failure,
            "detector_state": state,
            "detector_findings_n": findings_n,
            "detector_outcome": outcome,
            "read_redacted": redacted is not None,
            "task_context": dict(context),
        },
    )
    store.was_generated_by(annotation_id, activity)
    store.was_attributed_to(annotation_id, software)
    view.append(annotation_id)
    return ReviewOutcome(status=reading.status, annotation_id=annotation_id, view=tuple(view))
