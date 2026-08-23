"""The REDACT node: every PII finding padded, merged, silenced, and verified on the node's own output.

Every ``pii`` entity is redacted regardless of speaker. Extents are padded and merged by
``plan_redactions``; the margin is the ``redaction.padding_ms`` config key, whose derivation is in
``data/config/default.yaml``. Verification re-runs both recognizers PREPROCESS used, at their
recorded commits, plus the PII scan over the redacted audio and transcript; the verdict records
``audio_check: "bounded"`` (see ``specs/20260817-triage-workflow-dag/redact.md``). Artifacts are
written only on verified success, into a directory disjoint from the run directory, and carry no
store element id.
"""

from __future__ import annotations

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

_RESERVED_CATEGORY_CHAR = "+"  # plan_redactions' merge separator; a string, not a threshold


@dataclass(frozen=True)
class RedactResult(NodeResult):
    """What REDACT returns.

    Attributes:
        artifacts: The released paths, ``{"audio": ..., "transcript": ...}``; empty on fail.
    """

    artifacts: dict[str, Path]


def _extents_from_findings(store: ProvStore) -> list[RedactionExtent]:
    """Every pii entity, regardless of speaker; the membership check that secures the error path.

    Args:
        store: The provenance store.

    Returns:
        One extent per ``pii`` entity.

    Raises:
        ValueError: If a finding's category is empty or carries the reserved merge character, or if
            a finding has no extent. The error names bounds and category, never any matched text.
    """
    extents = []
    for finding in store.entities("pii"):
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


def _verify(redacted: Audio, transcript_text: str, asr_models: list[tuple[str, str]]) -> tuple[bool, list[str], bool]:
    """Re-run both recognizers and the scan on the node's own output.

    Args:
        redacted: The redacted audio.
        transcript_text: The redacted transcript.
        asr_models: ``(model_id, commit_sha)`` pairs read from the store's model agents.

    Returns:
        ``(verified, survived_categories, scan_ran)``. Any finding anywhere fails; a detector
        failure means the check did not run, which is not a clean result.
    """
    hypotheses = []
    for model_id, commit_sha in asr_models:
        (line,) = transcribe_audios([redacted], model=_verification_model(model_id, commit_sha))
        hypotheses.append(flatten_script_line(line))
    scans = scan_for_pii([*hypotheses, transcript_text])
    scans = scans if isinstance(scans, list) else [scans]
    if any(s.failures for s in scans):
        return False, [], False
    survived = sorted({span.category for s in scans for span in s.spans})
    return not survived, survived, True


def _asr_models(store: ProvStore) -> list[tuple[str, str]]:
    """``(model_id, commit_sha)`` per recognizer with words in the store, from its model agent (N14).

    Args:
        store: The provenance store.

    Returns:
        The pairs, sorted by model id.

    Raises:
        ValueError: If no word entity leads to a model agent with a resolved commit — verification
            cannot re-run recognizers it cannot name.
    """
    pairs: dict[str, str] = {}
    for word in store.entities("word"):
        recognizer = word.attributes.get("recognizer")
        activity_id = store.generated_by(word.id)
        if not recognizer or activity_id is None:
            continue
        for agent_id in store.associated_with(activity_id):
            agent = store.get_agent(agent_id)
            if agent.agent_type == "model" and agent.model_id == recognizer and agent.commit_sha is not None:
                pairs[str(recognizer)] = agent.commit_sha
    if not pairs:
        raise ValueError("no recognizer model agent with a resolved commit in the store; nothing can re-verify")
    return sorted(pairs.items())


def _overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """Whether two extents share any temporal intersection > 0."""
    return a[0] < b[1] and a[1] > b[0]


def _consensus_words(store: ProvStore) -> list[Entity]:
    """SPEECH's word entities, non-invalidated, in time order."""
    words = []
    for word in store.entities("word"):
        if store.is_invalidated(word.id):
            continue
        activity_id = store.generated_by(word.id)
        if activity_id is not None and store.get_activity(activity_id).node == "SPEECH":
            words.append(word)
    return sorted(words, key=lambda w: w.extent or (0.0, 0.0))


def _transcript(words: list[Entity], planned: list[RedactionExtent]) -> str:
    """The transcript with every planned extent's words rendered as one ``[CATEGORY]`` placeholder.

    A word overlapping a planned extent is replaced along with its padded-in neighbours, matching
    what the audio lost. No timestamps, no ids, no matched text.

    Args:
        words: SPEECH's consensus words, in time order.
        planned: The padded, merged extents.

    Returns:
        The transcript text.
    """
    tokens: list[str] = []
    emitted: set[int] = set()
    for word in words:
        extent = word.extent or (0.0, 0.0)
        index = next((i for i, p in enumerate(planned) if _overlaps(extent, (p.start, p.end))), None)
        if index is None:
            tokens.append(str(word.attributes.get("text") or ""))
        elif index not in emitted:
            emitted.add(index)
            tokens.append(f"[{planned[index].category}]")
    return " ".join(token for token in tokens if token)


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
        verification passed.

    Raises:
        ValueError: If ``redaction.padding_ms`` has no value, if ``artifacts_dir`` and ``run_dir``
            contain one another, if the store carries no PII scan measurement (N15), or if a finding's
            category is unusable (see :func:`_extents_from_findings`).
        LookupError: If no live stream carries ``source``.
    """
    padding_ms = int(config.require("redaction.padding_ms"))
    run_resolved, release_resolved = run_dir.resolve(), artifacts_dir.resolve()
    if run_resolved.is_relative_to(release_resolved) or release_resolved.is_relative_to(run_resolved):
        raise ValueError(
            f"artifacts_dir {artifacts_dir} and run_dir {run_dir} must not contain one another; "
            "the store and the release directory must not be sweepable by one publish step"
        )
    scan_measurement = find_measurement(store, "pii_scan")
    if scan_measurement is None:
        raise ValueError("no PII scan measurement in the store (N15); an unscanned recording is unchecked, not clean")
    findings = store.entities("pii")
    extents = _extents_from_findings(store)
    planned = plan_redactions(extents, padding_ms=padding_ms)
    asr_models = _asr_models(store)
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
    transcript_text = _transcript(words, planned)

    verify_act = store.activity(node=NODE, step="verify", parameters={"systems": [mid for mid, _ in asr_models]})
    store.was_associated_with(verify_act, software)
    for model_id, commit_sha in asr_models:
        model_agent = store.agent(agent_type="model", model_id=model_id, commit_sha=commit_sha)
        store.was_associated_with(verify_act, model_agent)
    verified, survived, scan_ran = _verify(redacted, transcript_text, asr_models)

    artifacts: dict[str, Path] = {}
    if verified:
        outcome, why = Outcome.PASS, "every finding redacted; the redacted output re-scans clean"
        artifacts = _write_artifacts(redacted, transcript_text, artifacts_dir)
    elif not scan_ran:
        outcome = Outcome.FAIL
        why = "verification did not run: a pii detector failed; an unverified artifact is withheld (N16)"
    else:
        outcome = Outcome.FAIL
        why = "verification found pii on the redacted output: " + ", ".join(survived)
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
            "verified": verified,
            "survived": survived,
            "audio_check": "bounded",
        },
    )
    view.append(verdict_id)
    return RedactResult(verdict=verdict, view=tuple(view), verdict_entity_id=verdict_id, artifacts=artifacts)
