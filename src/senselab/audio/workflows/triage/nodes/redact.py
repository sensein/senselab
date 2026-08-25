"""The REDACT node: every PII finding padded, merged, masked with the declared fill, and verified.

REDACT is the last step of the SPEECH branch and runs only when SPEECH's PII scan found something;
the runner gates on that, and this node refuses an incoherent store — findings with no scan
measurement — rather than concluding over one. Every non-invalidated ``pii`` entity is redacted
regardless of speaker. Extents are padded and merged by ``plan_redactions``; the margin is the
``redaction.padding_ms`` config key and must be a non-negative whole number of milliseconds. What is
written into an extent is ``redaction.fill``, which ships with no default, at ``redaction.bleep_hz``
when it is a bleep.

**No recognizer runs here.** Verification is a re-scan of the redacted consensus text with the same
detectors, judged complete by ``pii.required_detectors``: a surviving finding is a fail, an
incomplete re-scan is a flag. A finding the planner placed and the verifier still sees is remediable
exactly once — the verifier's words are fed back for a single re-planning pass, and what survives
that is ``unremediable``. The verdict's ``audio_check`` is the constant ``"bounded"`` on every path:
the re-scan establishes that the redacted text no longer carries the finding and nothing about the
audio. See ``specs/20260817-triage-workflow-dag/redact.md``.

A word carries a PII marking through a live ``assertion`` entity whose ``verb`` is ``"label"``,
whose ``label`` is ``"pii"``, and which is ``wasDerivedFrom`` the word — the store's shared shape for
a label. Only a pass produces a released pair; a flag withholds exactly like a fail, and the
verdict's ``artifacts_withheld`` records it. Artifacts are written into a directory disjoint from the
run directory, and carry no store element id.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from senselab.audio.data_structures import Audio, AudioHints
from senselab.audio.tasks.redaction.api import RedactionExtent, apply_redactions, plan_redactions
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.common import (
    NodeResult,
    find_measurement,
    live_entities,
    resolve_stream,
    software_agent,
    write_verdict,
)
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.text.tasks.pii_detection.api import scan_for_pii
from senselab.utils.prov_store import Entity, ProvStore

NODE = "REDACT"

_FILL_KEY = "redaction.fill"
_BLEEP_HZ_KEY = "redaction.bleep_hz"
_PADDING_KEY = "redaction.padding_ms"
_REQUIRED_DETECTORS_KEY = "pii.required_detectors"
_LABEL_VERB = "label"  # the store's assertion verb, a vocabulary term not a value
_PII_LABEL = "pii"  # the marking SPEECH places on a word carrying a finding
_RESERVED_CATEGORY_CHAR = "+"  # plan_redactions' merge separator; a string, not a threshold
_UNPLACED_PLACEHOLDER = "[UNPLACED]"  # a word the store places nowhere; a category-less placeholder
_AUDIO_CHECK = "bounded"  # what a text re-scan can claim about the audio, on every path


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
    """What re-scanning the redacted consensus text established.

    Attributes:
        verified: Whether the re-scan ran completely and found nothing.
        survived: The categories found on the redacted text, sorted; never matched text.
        scan_ran: Whether a complete re-scan happened at all — an empty ``failures`` is not evidence
            that one did.
        failed: Detectors the **verification** re-scan attempted and that raised. Names only; a
            failure's message may quote the scanned input.
        missing: Detectors ``pii.required_detectors`` names that the verification re-scan never
            attempted. Kept apart from ``failed`` for the reason the planning scan keeps them apart:
            "it broke" and "nobody ran it" are different findings, and the second is the silent one.
            Both are reported in the verdict separately from the planning scan's own ``scan_failed``
            and ``scan_missing``, because a store whose planning scan was complete and whose
            verification was not is a different state from the reverse, and an operator reading one
            pair of keys for both could not tell which half failed.
    """

    verified: bool
    survived: list[str]
    scan_ran: bool
    failed: list[str]
    missing: list[str]


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


def _scan_evidence(scan: Entity, required: list[str]) -> tuple[list[str], list[str], list[str]]:
    """What the store's ``pii_scan`` measurement says about its own completeness.

    Args:
        scan: The measurement entity.
        required: The detector set ``pii.required_detectors`` names.

    Returns:
        ``(scanned_by, failed, missing)`` — the detector names that ran, those that were attempted
        and failed, and those required but never attempted, each sorted. Only names: a failure's
        message may quote the scanned input.
    """
    scanned_by = sorted(str(name) for name in scan.attributes.get("scanned_by") or [])
    failed = sorted(str(name) for name in scan.attributes.get("failed") or [])
    missing = sorted(set(required) - set(scanned_by) - set(failed))
    return scanned_by, failed, missing


def _findings(store: ProvStore) -> list[Entity]:
    """The ``pii`` entities still standing, by the store's latest-non-invalidated rule.

    Args:
        store: The provenance store.

    Returns:
        The non-invalidated findings.
    """
    return live_entities(store, "pii")


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


def _verify(transcript_text: str, required: list[str]) -> _Verification:
    """Re-scan the redacted consensus text with the same detectors.

    No recognizer runs. Re-transcribing would draw a second sample from the recognizers, which is a
    different measurement of a different signal rather than a check on this one, and the claim about
    the audio is bounded either way.

    Args:
        transcript_text: The redacted consensus transcript.
        required: The detector set ``pii.required_detectors`` names.

    Returns:
        What the re-scan established. A finding that survives fails; a re-scan that skipped a
        required detector did not run, which is not a clean result.
    """
    scan = scan_for_pii(transcript_text)
    scan = scan[0] if isinstance(scan, list) else scan
    missing = sorted(set(required) - set(scan.detectors_used) - set(scan.failures))
    failed = sorted(scan.failures)
    if failed or not scan.detectors_used or missing:
        return _Verification(verified=False, survived=[], scan_ran=False, failed=failed, missing=missing)
    survived = sorted({span.category for span in scan.spans})
    return _Verification(verified=not survived, survived=survived, scan_ran=True, failed=[], missing=[])


def _overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """Whether two extents share any temporal intersection > 0.

    Args:
        a: One extent.
        b: The other.

    Returns:
        True when they intersect.
    """
    return a[0] < b[1] and a[1] > b[0]


def _consensus_words(store: ProvStore) -> list[Entity]:
    """The consensus words, in time order; a word the store places nowhere sorts first.

    Args:
        store: The provenance store.

    Returns:
        The live ``word`` entities PREPROCESS authored, oldest-extent first.
    """
    return sorted(live_entities(store, "word"), key=lambda w: w.extent or (-1.0, -1.0))


def _pii_marked_words(store: ProvStore) -> dict[str, set[str]]:
    """Which PII categories the store's live label assertions place on each word.

    Args:
        store: The provenance store.

    Returns:
        A mapping from word entity id to the categories marked on it. A word nothing marks is
        absent rather than mapped to an empty set.
    """
    marked: dict[str, set[str]] = {}
    for assertion in live_entities(store, "assertion"):
        attributes = assertion.attributes
        if attributes.get("verb") != _LABEL_VERB or attributes.get("label") != _PII_LABEL:
            continue
        category = str(attributes.get("category") or "")
        if not category:
            continue
        for source_id in store.derived_from(assertion.id):
            marked.setdefault(source_id, set()).add(category)
    return marked


def _matches_surviving(word: Entity, category: str, planned: list[RedactionExtent], marked: set[str]) -> bool:
    """Whether this word carries a surviving category that no planned extent already covers.

    Args:
        word: A live consensus ``word`` entity.
        category: A category the verification re-scan still saw.
        planned: The padded, merged extents the failing pass produced.
        marked: The PII categories the store's label assertions place on this word.

    Returns:
        True when the re-plan should widen to this word. A word a planned extent already covers is
        excluded, so the re-plan widens what the first pass missed rather than re-planning it.
    """
    if category not in marked or word.extent is None:
        return False
    return not any(_overlaps(word.extent, (extent.start, extent.end)) for extent in planned)


def _transcript(words: list[Entity], planned: list[RedactionExtent]) -> tuple[str, int]:
    """The transcript with every planned extent's words rendered as one ``[CATEGORY]`` placeholder.

    A word overlapping a planned extent is replaced along with its padded-in neighbours, matching
    what the audio lost. A word the store places nowhere overlaps no extent, so it is rendered as
    the category-less placeholder rather than released verbatim. No timestamps, no ids, no matched
    text.

    Args:
        words: PREPROCESS's consensus words, in time order.
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
    """Redact every PII finding from the recording and verify the redacted text before releasing it.

    Args:
        store: The provenance store, holding SPEECH's ``pii`` entities and scan measurement and
            PREPROCESS's consensus words.
        source: The store-held stream name, ``"recording"`` (N17).
        config: The triage configuration.
        hint: Accepted for the shared node shape; not read.
        run_dir: The run directory sidecar paths are relative to.
        artifacts_dir: The release directory; must not contain or be contained by ``run_dir``.

    Returns:
        The verdict, the view over what this node wrote, and the released artifacts — empty unless
        the outcome is a pass.

    Raises:
        ValueError: If ``redaction.fill`` has no value, if ``redaction.padding_ms`` has no usable
            value (see :func:`_padding_ms`), if ``artifacts_dir`` and ``run_dir`` contain one
            another, if the store carries no PII scan measurement (N15) — an incoherent store, as
            distinct from a complete store with nothing to scan, which concludes — or if a finding's
            category is unusable (see :func:`_extents_from_findings`).
        LookupError: If no live stream carries ``source``.
    """
    fill = str(config.require(_FILL_KEY))
    bleep_hz = config.get(_BLEEP_HZ_KEY)
    padding_ms = _padding_ms(config)
    required_detectors = sorted(str(name) for name in config.require(_REQUIRED_DETECTORS_KEY))
    run_resolved, release_resolved = run_dir.resolve(), artifacts_dir.resolve()
    if run_resolved.is_relative_to(release_resolved) or release_resolved.is_relative_to(run_resolved):
        raise ValueError(
            f"artifacts_dir {artifacts_dir} and run_dir {run_dir} must not contain one another; "
            "the store and the release directory must not be sweepable by one publish step"
        )
    scan_measurement = find_measurement(store, "pii_scan")
    if scan_measurement is None:
        raise ValueError("no PII scan measurement in the store (N15); an unscanned recording is unchecked, not clean")
    scanned_by, scan_failed, scan_missing = _scan_evidence(scan_measurement, required_detectors)
    scan_incomplete = bool(scan_failed) or bool(scan_missing) or not scanned_by
    findings = _findings(store)
    extents = _extents_from_findings(findings)
    words = _consensus_words(store)
    marked = _pii_marked_words(store)

    planned = plan_redactions(extents, padding_ms=padding_ms)
    transcript_text, unplaced_n = _transcript(words, planned)
    checked = (
        _verify(transcript_text, required_detectors)
        if not scan_incomplete
        else _Verification(verified=False, survived=[], scan_ran=False, failed=[], missing=[])
    )
    replanned_n = 0
    unremediable: list[str] = []
    if checked.scan_ran and checked.survived:
        replanned_n = 1
        extents = extents + [
            RedactionExtent(start=word.extent[0], end=word.extent[1], category=category)
            for category in checked.survived
            for word in words
            if word.extent is not None and _matches_surviving(word, category, planned, marked.get(word.id, set()))
        ]
        planned = plan_redactions(extents, padding_ms=padding_ms)
        transcript_text, unplaced_n = _transcript(words, planned)
        checked = _verify(transcript_text, required_detectors)
        unremediable = list(checked.survived)

    stream_id, recording = resolve_stream(store, run_dir, source)
    redacted = apply_redactions(recording, planned, fill=fill, bleep_hz=bleep_hz)

    software = software_agent(store)
    view: list[str] = []

    plan_act = store.activity(node=NODE, step="plan", parameters={"padding_ms": padding_ms, "replanned_n": replanned_n})
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

    apply_act = store.activity(node=NODE, step="apply", parameters={"redactions_n": len(planned), "fill": fill})
    store.was_associated_with(apply_act, software)
    store.used(apply_act, stream_id)
    for span_id in span_ids:
        store.used(apply_act, span_id)
    for word in words:
        store.used(apply_act, word.id)

    verify_act = store.activity(node=NODE, step="verify", parameters={"required_detectors": required_detectors})
    store.was_associated_with(verify_act, software)

    artifacts: dict[str, Path] = {}
    if scan_incomplete:
        outcome = Outcome.FAIL
        reasons = []
        if scan_failed:
            reasons.append(f"detectors failed: {', '.join(scan_failed)}")
        if scan_missing:
            reasons.append(f"required detectors were not attempted: {', '.join(scan_missing)}")
        if not scanned_by:
            reasons.append("no detector ran")
        why = (
            f"the store's pii scan is incomplete ({'; '.join(reasons)}); "
            "an unchecked recording is not a clean one (N15)"
        )
    elif not checked.scan_ran:
        outcome = Outcome.FLAG
        parts = []
        if checked.failed:
            parts.append(f"detectors failed: {', '.join(checked.failed)}")
        if checked.missing:
            parts.append(f"required detectors were not attempted: {', '.join(checked.missing)}")
        if not parts:
            parts.append("no detector ran")
        why = (
            f"the re-scan over the redacted text is incomplete ({'; '.join(parts)}); an unverified artifact is withheld"
        )
    elif checked.survived:
        outcome = Outcome.FAIL
        why = "verification found pii on the redacted transcript: " + ", ".join(checked.survived)
    else:
        outcome = Outcome.PASS
        why = "every finding redacted; the redacted transcript re-scans clean"
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
            "fill": fill,
            "verified": checked.verified,
            "survived": checked.survived,
            "unremediable": unremediable,
            "replanned_n": replanned_n,
            "scan_failed": scan_failed,
            "scan_missing": scan_missing,
            "verify_failed": checked.failed,
            "verify_missing": checked.missing,
            "required_detectors": required_detectors,
            "unplaced_words_n": unplaced_n,
            "audio_check": _AUDIO_CHECK,
            "artifacts_withheld": not artifacts,
        },
    )
    view.append(verdict_id)
    return RedactResult(verdict=verdict, view=tuple(view), verdict_entity_id=verdict_id, artifacts=artifacts)
