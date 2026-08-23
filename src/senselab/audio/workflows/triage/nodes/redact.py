"""The REDACT node: every PII finding padded, merged, silenced, and verified on the node's own output.

Every non-invalidated ``pii`` entity is redacted regardless of speaker. Extents are padded and merged
by ``plan_redactions``; the margin is the ``redaction.padding_ms`` config key, whose derivation is in
``data/config/default.yaml`` and which must be a non-negative whole number of milliseconds.
Verification re-runs the recognizers PREPROCESS used, at their recorded commits, plus the PII scan
over the redacted audio and transcript; the verdict records ``audio_check: "bounded"`` (see
``specs/20260817-triage-workflow-dag/redact.md``). The set it is measured against is the one
PREPROCESS declared in its own activities, with the word-derived set as the fallback and
``expected_source`` naming which was read. A store whose ``pii_scan`` measurement records a
failed or absent detector is unchecked rather than clean, and verification over a scan in which no
detector ran is not a result — such a store withholds without needing any recognizer, since nothing
has to be verified to refuse a release. Only a pass produces a released pair; a flag withholds
exactly like a fail, and the verdict's ``artifacts_withheld`` records it. Artifacts are written into
a directory disjoint from the run directory, and carry no store element id.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from senselab.audio.data_structures import Audio, AudioHints
from senselab.audio.tasks.redaction.api import RedactionExtent, apply_redactions, plan_redactions
from senselab.audio.tasks.speech_to_text.api import transcribe_audios
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.common import (
    NodeResult,
    find_measurement,
    resolve_stream,
    software_agent,
    write_verdict,
)
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.text.tasks.pii_detection.api import flatten_script_line, scan_for_pii
from senselab.utils.data_structures import HFModel
from senselab.utils.prov_store import Entity, ProvStore

NODE = "REDACT"

_PADDING_KEY = "redaction.padding_ms"
_PREPROCESS_NODE = "PREPROCESS"  # whose activities declare the recognizer set, a node name not a value
_MODEL_PARAMETER = "model"  # the activity parameter PREPROCESS's recognizer steps name their model in
_NOT_REQUIRED = "not_required"  # the expected-set source when the scan already withholds the release
_RESERVED_CATEGORY_CHAR = "+"  # plan_redactions' merge separator; a string, not a threshold
_UNPLACED_PLACEHOLDER = "[UNPLACED]"  # a word the store places nowhere; a category-less placeholder


@dataclass(frozen=True)
class RedactResult(NodeResult):
    """What REDACT returns.

    Attributes:
        artifacts: The released paths, ``{"audio": ..., "transcript": ...}``; empty on anything
            but a pass.
    """

    artifacts: dict[str, Path]


@dataclass(frozen=True)
class _Verification:
    """What re-running the recognizers and the scan over the node's own output established.

    Attributes:
        verified: Whether every scan ran and none of them found anything.
        survived: The categories found on the redacted output, sorted; never matched text.
        scan_ran: Whether a detector ran at all — an empty ``failures`` is not evidence that one did.
    """

    verified: bool
    survived: list[str]
    scan_ran: bool


def _padding_ms(config: TriageConfig) -> int:
    """The redaction margin, in whole milliseconds, as a validity check rather than a tunable.

    Args:
        config: The triage configuration.

    Returns:
        The margin.

    Raises:
        ValueError: If ``redaction.padding_ms`` has no value, is not a number, is not finite, is not
            integral, or is negative. A negative or fractional margin is refused rather than coerced.
    """
    raw = config.require(_PADDING_KEY)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{_PADDING_KEY} must be a number of milliseconds, not {type(raw).__name__}")
    if isinstance(raw, float):
        if not math.isfinite(raw):
            raise ValueError(f"{_PADDING_KEY} must be finite, got {raw!r}")
        if raw != int(raw):
            raise ValueError(f"{_PADDING_KEY} must be a whole number of milliseconds, got {raw!r}")
    value = int(raw)
    if value < 0:
        raise ValueError(f"{_PADDING_KEY} must be >= 0, got {value}; a negative margin narrows every extent")
    return value


def _scan_evidence(scan: Entity) -> tuple[list[str], list[str]]:
    """What the store's ``pii_scan`` measurement says about its own completeness.

    Args:
        scan: The measurement entity.

    Returns:
        ``(scanned_by, failed)`` — the detector names that ran and those that did not, sorted. Only
        names: a failure's message may quote the scanned input.
    """
    scanned_by = scan.attributes.get("scanned_by") or []
    failed = scan.attributes.get("failed") or []
    return sorted(str(name) for name in scanned_by), sorted(str(name) for name in failed)


def _findings(store: ProvStore) -> list[Entity]:
    """The ``pii`` entities still standing, by the store's latest-non-invalidated rule.

    Args:
        store: The provenance store.

    Returns:
        The non-invalidated findings.
    """
    return [finding for finding in store.entities("pii") if not store.is_invalidated(finding.id)]


def _extents_from_findings(findings: list[Entity]) -> list[RedactionExtent]:
    """Every finding, regardless of speaker; the membership check that secures the error path.

    Args:
        findings: The live ``pii`` entities.

    Returns:
        One extent per finding.

    Raises:
        ValueError: If a finding's category is empty or carries the reserved merge character, or if
            a finding has no extent. The error names bounds and category, never any matched text.
    """
    extents = []
    for finding in findings:
        category = finding.attributes.get("category", "")
        if not category or _RESERVED_CATEGORY_CHAR in category:
            raise ValueError(
                f"category {category!r} is empty or contains the reserved merge character; "
                "it cannot be planned without silently decomposing on re-planning"
            )
        if finding.extent is None:
            raise ValueError(f"pii finding {finding.id} has no extent; nothing locatable can be redacted")
        extents.append(RedactionExtent(start=finding.extent[0], end=finding.extent[1], category=category))
    return extents


def _verification_model(model_id: str, commit_sha: str) -> HFModel:
    """A recognizer at the commit the store's model agent recorded — never a ref (N14).

    Args:
        model_id: The recognizer's model id.
        commit_sha: The resolved 40-hex commit the store recorded for it.

    Returns:
        The model spec.
    """
    return HFModel(path_or_uri=model_id, revision=commit_sha)


def _verify(redacted: Audio, transcript_text: str, asr_models: list[tuple[str, str]]) -> _Verification:
    """Re-run the recognizers and the scan on the node's own output.

    Args:
        redacted: The redacted audio.
        transcript_text: The redacted transcript.
        asr_models: ``(model_id, commit_sha)`` pairs read from the store's model agents.

    Returns:
        What the re-run established. Any finding anywhere fails; a scan with a failed detector or
        with no detector at all did not run, which is not a clean result.
    """
    hypotheses = []
    for model_id, commit_sha in asr_models:
        (line,) = transcribe_audios([redacted], model=_verification_model(model_id, commit_sha))
        hypotheses.append(flatten_script_line(line))
    scans = scan_for_pii([*hypotheses, transcript_text])
    scans = scans if isinstance(scans, list) else [scans]
    if any(s.failures or not s.detectors_used for s in scans):
        return _Verification(verified=False, survived=[], scan_ran=False)
    survived = sorted({span.category for s in scans for span in s.spans})
    return _Verification(verified=not survived, survived=survived, scan_ran=True)


def _declared_recognizers(store: ProvStore) -> list[str]:
    """The recognizers PREPROCESS declared, read from its own activities rather than from words.

    A PREPROCESS activity naming a model in its parameters and running under that model's agent is
    a declared recognizer whether or not it went on to write a word.

    Args:
        store: The provenance store.

    Returns:
        The declared model ids, sorted; empty when the store carries no such declaration.
    """
    declared: set[str] = set()
    for activity in store.activities(_PREPROCESS_NODE):
        model_id = activity.parameters.get(_MODEL_PARAMETER)
        if model_id is None:
            continue
        for agent_id in store.associated_with(activity.id):
            agent = store.get_agent(agent_id)
            if agent.agent_type == "model" and agent.model_id == str(model_id):
                declared.add(str(model_id))
    return sorted(declared)


def _asr_models(store: ProvStore) -> tuple[list[tuple[str, str]], list[str], str]:
    """The re-runnable recognizers and the set they are expected to cover (N14).

    Args:
        store: The provenance store.

    Returns:
        ``([(model_id, commit_sha), ...], [model_id, ...], source)`` — the re-runnable pairs, every
        recognizer verification is expected to cover, and where that expected set came from:
        ``"preprocess"`` for the node's own declaration, ``"words"`` when the store carries none and
        only the recognizers that wrote a word can be named.

    Raises:
        ValueError: If no word entity leads to a model agent with a resolved commit — verification
            cannot re-run recognizers it cannot name.
    """
    pairs: dict[str, str] = {}
    wrote_words: set[str] = set()
    for word in store.entities("word"):
        recognizer = word.attributes.get("recognizer")
        activity_id = store.generated_by(word.id)
        if not recognizer or activity_id is None or store.is_invalidated(word.id):
            continue
        for agent_id in store.associated_with(activity_id):
            agent = store.get_agent(agent_id)
            if agent.agent_type == "model" and agent.model_id == recognizer:
                wrote_words.add(str(recognizer))
                if agent.commit_sha is not None:
                    pairs[str(recognizer)] = agent.commit_sha
    if not pairs:
        raise ValueError("no recognizer model agent with a resolved commit in the store; nothing can re-verify")
    declared = _declared_recognizers(store)
    if declared:
        return sorted(pairs.items()), declared, "preprocess"
    return sorted(pairs.items()), sorted(wrote_words), "words"


def _overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """Whether two extents share any temporal intersection > 0."""
    return a[0] < b[1] and a[1] > b[0]


def _consensus_words(store: ProvStore) -> list[Entity]:
    """SPEECH's word entities, non-invalidated, in time order; the unplaced sort first."""
    words = []
    for word in store.entities("word"):
        if store.is_invalidated(word.id):
            continue
        activity_id = store.generated_by(word.id)
        if activity_id is not None and store.get_activity(activity_id).node == "SPEECH":
            words.append(word)
    return sorted(words, key=lambda w: w.extent or (-1.0, -1.0))


def _transcript(words: list[Entity], planned: list[RedactionExtent]) -> tuple[str, int]:
    """The transcript with every planned extent's words rendered as one ``[CATEGORY]`` placeholder.

    A word overlapping a planned extent is replaced along with its padded-in neighbours, matching
    what the audio lost. A word the store places nowhere overlaps no extent, so it is rendered as
    the category-less placeholder rather than released verbatim. No timestamps, no ids, no matched
    text.

    Args:
        words: SPEECH's consensus words, in time order.
        planned: The padded, merged extents.

    Returns:
        The transcript text and the number of words released as unplaced.
    """
    tokens: list[str] = []
    emitted: set[int] = set()
    unplaced = 0
    for word in words:
        if word.extent is None:
            unplaced += 1
            tokens.append(_UNPLACED_PLACEHOLDER)
            continue
        index = next((i for i, p in enumerate(planned) if _overlaps(word.extent, (p.start, p.end))), None)
        if index is None:
            tokens.append(str(word.attributes.get("text") or ""))
        elif index not in emitted:
            emitted.add(index)
            tokens.append(f"[{planned[index].category}]")
    return " ".join(token for token in tokens if token), unplaced


def _write_artifacts(redacted: Audio, transcript_text: str, artifacts_dir: Path) -> dict[str, Path]:
    """Write the releasable pair. Takes no store and no element id, so it cannot embed one.

    Args:
        redacted: The verified redacted audio.
        transcript_text: The verified redacted transcript.
        artifacts_dir: The release directory.

    Returns:
        The written paths, keyed ``audio``/``transcript``.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    audio_path = artifacts_dir / "audio.wav"
    redacted.save_to_file(str(audio_path))
    transcript_path = artifacts_dir / "transcript.txt"
    transcript_path.write_text(transcript_text + "\n")
    return {"audio": audio_path, "transcript": transcript_path}


def redact(
    store: ProvStore,
    source: str,
    config: TriageConfig,
    hint: AudioHints | None = None,
    *,
    run_dir: Path,
    artifacts_dir: Path,
) -> RedactResult:
    """Redact every PII finding from the recording and verify the result before releasing it.

    Args:
        store: The provenance store, holding SPEECH's ``pii`` entities, words and scan measurement.
        source: The store-held stream name, ``"recording"`` (N17).
        config: The triage configuration.
        hint: Accepted for the shared node shape; not read.
        run_dir: The run directory sidecar paths are relative to.
        artifacts_dir: The release directory; must not contain or be contained by ``run_dir``.

    Returns:
        The verdict, the view over what this node wrote, and the released artifacts — empty unless
        the outcome is a pass.

    Raises:
        ValueError: If ``redaction.padding_ms`` has no usable value (see :func:`_padding_ms`), if
            ``artifacts_dir`` and ``run_dir`` contain one another, if the store carries no PII scan
            measurement (N15), if a finding's category is unusable (see
            :func:`_extents_from_findings`), or if a scan claiming completeness has no re-runnable
            recognizer (see :func:`_asr_models`) — an incoherent store, as distinct from a complete
            store with nothing to scan, which concludes.
        LookupError: If no live stream carries ``source``.
    """
    padding_ms = _padding_ms(config)
    run_resolved, release_resolved = run_dir.resolve(), artifacts_dir.resolve()
    if run_resolved.is_relative_to(release_resolved) or release_resolved.is_relative_to(run_resolved):
        raise ValueError(
            f"artifacts_dir {artifacts_dir} and run_dir {run_dir} must not contain one another; "
            "the store and the release directory must not be sweepable by one publish step"
        )
    scan_measurement = find_measurement(store, "pii_scan")
    if scan_measurement is None:
        raise ValueError("no PII scan measurement in the store (N15); an unscanned recording is unchecked, not clean")
    scanned_by, scan_failed = _scan_evidence(scan_measurement)
    scan_incomplete = bool(scan_failed) or not scanned_by
    findings = _findings(store)
    extents = _extents_from_findings(findings)
    planned = plan_redactions(extents, padding_ms=padding_ms)
    if scan_incomplete:
        asr_models: list[tuple[str, str]] = []
        expected_recognizers: list[str] = []
        expected_source = _NOT_REQUIRED
    else:
        asr_models, expected_recognizers, expected_source = _asr_models(store)
    stream_id, recording = resolve_stream(store, run_dir, source)

    software = software_agent(store)
    view: list[str] = []

    plan_act = store.activity(node=NODE, step="plan", parameters={"padding_ms": padding_ms})
    store.was_associated_with(plan_act, software)
    store.used(plan_act, scan_measurement.id)
    for finding in findings:
        store.used(plan_act, finding.id)
    span_ids: list[str] = []
    for extent in planned:
        span_id = store.entity(
            prov_type="span",
            extent=(extent.start, extent.end),
            attributes={"name": "redaction", "category": extent.category},
        )
        store.was_generated_by(span_id, plan_act)
        store.was_attributed_to(span_id, software)
        for finding in findings:
            if finding.extent is not None and _overlaps(finding.extent, (extent.start, extent.end)):
                store.was_derived_from(span_id, finding.id)
        span_ids.append(span_id)
        view.append(span_id)

    apply_act = store.activity(node=NODE, step="apply", parameters={"redactions_n": len(planned)})
    store.was_associated_with(apply_act, software)
    store.used(apply_act, stream_id)
    for span_id in span_ids:
        store.used(apply_act, span_id)
    redacted = apply_redactions(recording, planned)
    words = _consensus_words(store)
    for word in words:
        store.used(apply_act, word.id)
    transcript_text, unplaced_n = _transcript(words, planned)

    verify_systems = [] if scan_incomplete else [model_id for model_id, _ in asr_models]
    verify_act = store.activity(node=NODE, step="verify", parameters={"systems": verify_systems})
    store.was_associated_with(verify_act, software)
    if scan_incomplete:
        checked = _Verification(verified=False, survived=[], scan_ran=False)
    else:
        for model_id, commit_sha in asr_models:
            model_agent = store.agent(agent_type="model", model_id=model_id, commit_sha=commit_sha)
            store.was_associated_with(verify_act, model_agent)
        checked = _verify(redacted, transcript_text, asr_models)
    unverifiable = sorted(set(expected_recognizers) - set(verify_systems))

    artifacts: dict[str, Path] = {}
    if scan_incomplete:
        outcome = Outcome.FAIL
        missing = f"detectors failed: {', '.join(scan_failed)}" if scan_failed else "no detector ran"
        why = f"the store's pii scan is incomplete ({missing}); an unchecked recording is not a clean one (N15)"
    elif not checked.scan_ran:
        outcome = Outcome.FAIL
        why = "verification did not run: no pii detector ran over the output; an unverified artifact is withheld (N16)"
    elif checked.survived:
        outcome = Outcome.FAIL
        why = "verification found pii on the redacted output: " + ", ".join(checked.survived)
    elif unverifiable:
        outcome = Outcome.FLAG
        why = (
            "the redacted output re-scans clean, but verification re-ran only "
            f"{', '.join(verify_systems)}; {', '.join(unverifiable)} wrote no word at a resolved commit "
            "to re-run at"
        )
    else:
        outcome = Outcome.PASS
        why = "every finding redacted; the redacted output re-scans clean"
        artifacts = _write_artifacts(redacted, transcript_text, artifacts_dir)
    verdict_id, verdict = write_verdict(
        store,
        verify_act,
        software,
        node=NODE,
        outcome=outcome,
        kind=None,
        why=why,
        detail={
            "redactions_n": len(planned),
            "by_category": dict(Counter(extent.category for extent in planned)),
            "padding_ms": padding_ms,
            "verified": checked.verified,
            "survived": checked.survived,
            "verify_systems": verify_systems,
            "expected_systems": expected_recognizers,
            "expected_source": expected_source,
            "scan_failed": scan_failed,
            "unplaced_words_n": unplaced_n,
            "audio_check": "bounded",
            "artifacts_withheld": not artifacts,
        },
    )
    view.append(verdict_id)
    return RedactResult(verdict=verdict, view=tuple(view), verdict_entity_id=verdict_id, artifacts=artifacts)
