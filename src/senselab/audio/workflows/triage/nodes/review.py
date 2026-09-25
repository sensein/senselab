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
from pathlib import Path
from typing import Any, Mapping, Sequence

from senselab.audio.data_structures import AudioHints
from senselab.audio.tasks.redaction.api import RedactionExtent, apply_redactions, plan_redactions
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.branches import declared_task_family, expected_names
from senselab.audio.workflows.triage.nodes.common import (
    consensus_words,
    find_measurement,
    find_verdict,
    path_attributes,
    resolve_stream,
    software_agent,
    word_hull,
    write_stream,
)
from senselab.audio.workflows.triage.nodes.redact import REDACTION_SPAN, planned_extents, transcript_texts
from senselab.audio.workflows.triage.nodes.redact import STREAM_NAME as REDACTED_STREAM
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
        # Another round is only worth taking if this one changed the text it will read. The loop's
        # whole action is the mask, and masking a proposal with no removal in it is a no-op: the
        # next round would read a byte-identical string and answer the same way. Measured over the
        # first 9,670 reviewed recordings -- 3,131 ran to the ceiling, 86% of them on
        # `redaction == incomplete` with no proposal at all, and 99.2% of every multi-round
        # recording gained nothing after round one. That was 39% of the pass's GPU time spent
        # re-reading unchanged text.
        before_mask = current if current is not None else original
        masked = _mask(before_mask, result.proposal)
        if masked == before_mask:
            return (
                _Reading(
                    FLAGGED,
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
        current = masked
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


@dataclass(frozen=True)
class RefinedPlan:
    """The span set the reviewer's proposal asks for, over the detectors' own.

    Attributes:
        extents: What to remove, padded and merged; the applied set.
        added_n: How many extents the reviewer added that the detectors did not have.
        released_n: How many detector extents the reviewer's proposal dropped.
        unplaced: The proposal texts that could not be located in the transcript, verbatim quotes
            of the model's own words. A removal it could not place keeps everything and a release
            it could not place changes nothing, so an unplaceable entry can only fail closed.
        reasons: Category to the reviewer's one-sentence reason, for every entry it placed.
    """

    extents: list[RedactionExtent]
    added_n: int
    released_n: int
    unplaced: tuple[str, ...]
    reasons: dict[str, str]


def _word_spans(words: Sequence[Any]) -> tuple[str, list[tuple[int, int, Any]]]:
    """The transcript as one string, with each word's character range in it.

    Args:
        words: PREPROCESS's consensus words, in stream order.

    Bracketed words are skipped, because ``transcript_texts`` drops them: the join has to be the
    string the reviewer read, or a quote spanning where a ``[UH]`` used to sit would not be found
    and the redaction it asked for would be silently dropped as unplaced.

    Returns:
        ``(text, spans)``, each span ``(start_char, end_char, word)``. The join is the one
        :func:`~senselab.audio.workflows.triage.nodes.redact.transcript_texts` produces, so an
        offset found here indexes the text the reviewer was actually shown.
    """
    parts: list[str] = []
    spans: list[tuple[int, int, Any]] = []
    cursor = 0
    for word in words:
        if word.attributes.get("bracketed"):
            continue
        surface = str(word.attributes.get("text") or "")
        if parts:
            cursor += 1
        parts.append(surface)
        spans.append((cursor, cursor + len(surface), word))
        cursor += len(surface)
    return " ".join(parts), spans


def _covered(text: str, spans: Sequence[tuple[int, int, Any]], quoted: str) -> list[Any]:
    """The words a quoted substring covers.

    Args:
        text: The transcript as one string.
        spans: Each word's character range in it.
        quoted: The substring the reviewer named.

    Returns:
        The words it overlaps, in order. Empty where the substring is not in the text, which is the
        one case that must not be guessed at: a finding nothing could place was once widened to the
        whole transcript, and a 9.1 s span over a sustained vowel is what that produced.
    """
    lowered, needle = text.lower(), quoted.strip().lower()
    if not needle:
        return []
    at = lowered.find(needle)
    if at == -1:
        return []
    end = at + len(needle)
    return [word for start, stop, word in spans if start < end and stop > at]


def refine_plan(store: ProvStore, *, padding_ms: int) -> RefinedPlan:
    """The redaction the reviewer's proposal asks for, as extents over this recording.

    The proposal is about text and an extent is about time, so the consensus words' own timings are
    the join. A proposal entry naming text the transcript does not contain is recorded and
    otherwise ignored.

    Args:
        store: The provenance store, carrying the annotation and the detectors' own spans.
        padding_ms: The redaction margin, as ``redaction.padding_ms``.

    Returns:
        The refined plan.
    """
    annotation = find_measurement(store, REDACTION_LLM_ANNOTATION)
    proposal = list(annotation.attributes.get("proposal") or ()) if annotation is not None else []
    detector = planned_extents(store)
    words = consensus_words(store)
    text, spans = _word_spans(words)

    keep = list(detector)
    added: list[RedactionExtent] = []
    unplaced: list[str] = []
    reasons: dict[str, str] = {}
    released = 0
    for entry in proposal:
        quoted = str(entry.get("text") or "")
        action = str(entry.get("action") or REDACT)
        covered = _covered(text, spans, quoted)
        if not covered:
            unplaced.append(quoted)
            continue
        hulls = [word_hull(word) for word in covered if word.extent is not None]
        if not hulls:
            unplaced.append(quoted)
            continue
        category = str(entry.get("category") or "OTHER").upper()
        reasons[category] = str(entry.get("why") or "")
        bounds = (min(hull[0] for hull in hulls), max(hull[1] for hull in hulls))
        if action == REDACT:
            added.append(RedactionExtent(start=bounds[0], end=bounds[1], category=category))
        else:
            before = len(keep)
            keep = [extent for extent in keep if not (extent.start < bounds[1] and extent.end > bounds[0])]
            released += before - len(keep)

    combined = plan_redactions([*keep, *added], padding_ms=padding_ms) if (keep or added) else []
    return RefinedPlan(
        extents=combined,
        added_n=len(added),
        released_n=released,
        unplaced=tuple(unplaced),
        reasons=reasons,
    )


def apply_proposal(store: ProvStore, config: TriageConfig, *, run_dir: Path, source: str) -> str | None:
    """Write this recording's redacted stream from the refined span set.

    One artefact: the detectors' own version is not kept beside it. What says which span set the
    audio came from is the store -- every applied span is generated by this node's ``apply``
    activity, carries the reviewer's reason where the reviewer placed it, and the spans the
    proposal released are invalidated rather than deleted, so a reader can ask why a span is no
    longer cut and get an answer.

    Args:
        store: The provenance store.
        config: The triage configuration.
        run_dir: The run directory whose ``streams/`` the audio is written into.
        source: The stream the redaction is applied to.

    Returns:
        The written stream's entity id, or None where the reviewer proposed nothing and the
        detectors' own stream therefore already is the refined one.
    """
    annotation = find_measurement(store, REDACTION_LLM_ANNOTATION)
    if annotation is None or not (annotation.attributes.get("proposal") or ()):
        return None
    padding_ms = int(config.require("redaction.padding_ms"))
    fill = str(config.require("redaction.fill"))
    refined = refine_plan(store, padding_ms=padding_ms)

    software = software_agent(store)
    activity = store.activity(
        node=NODE,
        step="apply",
        parameters={
            "redactions_n": len(refined.extents),
            "added_n": refined.added_n,
            "released_n": refined.released_n,
            "unplaced_n": len(refined.unplaced),
            "fill": fill,
            "padding_ms": padding_ms,
        },
    )
    store.was_associated_with(activity, software)
    store.used(activity, annotation.id)
    for stale in planned_extents_ids(store):
        store.was_invalidated_by(stale, activity)

    span_ids: list[str] = []
    for extent in refined.extents:
        span_id = store.entity(
            prov_type="span",
            extent=(extent.start, extent.end),
            attributes={
                "name": REDACTION_SPAN,
                "category": extent.category,
                "why": refined.reasons.get(extent.category, ""),
            },
        )
        store.was_generated_by(span_id, activity)
        store.was_attributed_to(span_id, software)
        store.was_derived_from(span_id, annotation.id)
        span_ids.append(span_id)

    stream_id, recording = resolve_stream(store, run_dir, source)
    redacted = apply_redactions(recording, refined.extents, fill=fill, bleep_hz=config.get("redaction.bleep_hz"))
    (run_dir / "streams").mkdir(parents=True, exist_ok=True)
    relative, report = write_stream(redacted, run_dir, REDACTED_STREAM)
    written = store.entity(
        prov_type="stream",
        extent=(0.0, float(redacted.waveform.shape[-1]) / float(redacted.sampling_rate)),
        attributes={
            "name": REDACTED_STREAM,
            **path_attributes(relative, run_dir),
            "sampling_rate": int(redacted.sampling_rate),
            "channels": int(redacted.waveform.shape[0]),
            "write_gain": report.gain,
            "fill": fill,
            "from_proposal": True,
        },
    )
    store.was_generated_by(written, activity)
    store.was_attributed_to(written, software)
    store.was_derived_from(written, stream_id)
    for span_id in span_ids:
        store.was_derived_from(written, span_id)
    return written


def planned_extents_ids(store: ProvStore) -> list[str]:
    """The ids of the live redaction spans, for a refinement to retire.

    Args:
        store: The provenance store.

    Returns:
        The ids.
    """
    from senselab.audio.workflows.triage.nodes.common import live_entities

    return [
        span.id
        for span in live_entities(store, "span")
        if span.attributes.get("name") == REDACTION_SPAN and span.extent is not None
    ]
