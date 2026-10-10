"""The SECOND_OPINION node: a decision model's answers about the text the reviewer reads and the text that ships.

It asks :data:`~senselab.text.tasks.decision_model.second_opinion.QUESTIONS` over the same original
transcript and task context REVIEW reads, beside the masked text the final mask plan would ship (the
reviewer's releases included, as VERDICT folds them), and writes the probabilities as one
``second_opinion_answers`` measurement. It runs after REVIEW. It writes no verdict; VERDICT compares the
probabilities with the reviewer's reading under ``verdict.second_opinion_*``.

The model is reached through a caller-supplied ``ask``, so a driver can hold one pinned server for a
whole slice. See ``specs/20261003-clef-second-opinion/design.md``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.branches import declared_task_family
from senselab.audio.workflows.triage.nodes.common import find_measurement, mint_live, software_agent
from senselab.audio.workflows.triage.nodes.redact import fold_mask_plan, shipped_texts, transcript_texts
from senselab.audio.workflows.triage.nodes.review import task_context
from senselab.audio.workflows.triage.task_lexicon import task_lexicon
from senselab.audio.workflows.triage.vocabulary import (
    REDACTION_LLM_ANNOTATION,
    SECOND_OPINION_ANSWERS,
    FoldPolicy,
    reviewer_may_unmask,
)
from senselab.text.tasks.decision_model.ollama import OllamaPin
from senselab.text.tasks.decision_model.second_opinion import (
    QUESTION_SET_VERSION,
    SecondOpinion,
    ask_second_opinion,
)
from senselab.text.tasks.pii_detection.redaction_review import redaction_policy_text, task_guidance_digest
from senselab.utils.prov_store import ProvStore
from senselab.utils.tasks.cached_inference import (
    result_cache_key,
    result_lookup,
    result_store,
    transcript_signature,
)

NODE = "SECOND_OPINION"
PROCESS = "second_opinion_answers"

OK = "ok"
DISABLED = "disabled"
NOTHING_TO_READ = "nothing_to_read"
ABSENT = "absent"
STATUSES = (OK, DISABLED, NOTHING_TO_READ, ABSENT)
"""``second_opinion_answers.status``: answered, switched off, no text, or asked and not answered."""

_SECTION = "second_opinion"
Ask = Callable[[Mapping[str, Any], Mapping[str, Mapping[str, Any]]], Mapping[str, Any]]


def settings(config: TriageConfig) -> dict[str, Any]:
    """Every ``second_opinion`` key, read together.

    Args:
        config: The triage configuration.

    Returns:
        The settings.
    """
    names = (
        "enabled",
        "name",
        "tag",
        "blob_digest",
        "config_digest",
        "manifest_digest",
        "seed",
        "timeout_s",
        "load_timeout_s",
        "workers",
        "max_consecutive_errors",
    )
    return {name: config.require(f"{_SECTION}.{name}") for name in names}


def pin_of(config: TriageConfig) -> OllamaPin:
    """The pinned model the configuration names.

    Args:
        config: The triage configuration.

    Returns:
        The pin.
    """
    held = settings(config)
    return OllamaPin(
        name=str(held["name"]),
        tag=str(held["tag"]),
        blob_digest=str(held["blob_digest"]),
        config_digest=str(held["config_digest"]),
        manifest_digest=str(held["manifest_digest"]),
    )


def opinion_cache_key(original: str, masked: str, context: Mapping[str, Any], pin: OllamaPin, seed: int) -> str:
    """The result-cache key of one second opinion.

    Args:
        original: The transcript asked about.
        masked: The masked text sent beside it.
        context: The task context sent with it.
        pin: The model.
        seed: The sampling seed.

    Returns:
        The key: both texts and the context, the question set version, the task guidance and redaction
        policy digests, the seed and the pin.
    """
    signature = transcript_signature(
        original + "\x1f" + masked + "\x1f" + json.dumps(dict(context), sort_keys=True, default=str)
    )
    return result_cache_key(
        input_signature=signature,
        process=PROCESS,
        model_id=pin.model_id,
        commit_sha=pin.blob_digest,
        params={
            "question_set_version": QUESTION_SET_VERSION,
            "task_guidance": task_guidance_digest(),
            "redaction_policy": hashlib.sha256(redaction_policy_text().encode("utf-8")).hexdigest(),
            "seed": int(seed),
            "config": pin.config_digest,
            "manifest": pin.manifest_digest,
        },
    )


@dataclass(frozen=True)
class OpinionOutcome:
    """What one call wrote.

    Attributes:
        status: One of :data:`STATUSES`.
        measurement_id: The ``second_opinion_answers`` entity.
    """

    status: str
    measurement_id: str


def masked_text(store: ProvStore, config: TriageConfig, hint: AudioHints | None) -> tuple[str, str]:
    """The consensus transcript and the text the release would ship, under VERDICT's final mask plan.

    Args:
        store: The provenance store, REVIEW's annotation included where it ran.
        config: The triage configuration.
        hint: The caller's declaration.

    Returns:
        ``(original, masked)`` from :func:`~senselab.audio.workflows.triage.nodes.redact.shipped_texts`,
        the reviewer's releases applied exactly where VERDICT applies them.
    """
    measurement = find_measurement(store, REDACTION_LLM_ANNOTATION)
    annotation = dict(measurement.attributes) if measurement is not None else None
    applies = FoldPolicy.from_config(config).llm_reset_redactions and reviewer_may_unmask(annotation)
    return shipped_texts(store, fold_mask_plan(store, config, hint, reviewer_applies=applies))


def second_opinion(
    store: ProvStore,
    config: TriageConfig,
    hint: AudioHints | None,
    ask: Ask | None,
    *,
    num_parallel: int | None = None,
) -> OpinionOutcome:
    """Ask the question set over the reviewer's text and record the answers.

    Args:
        store: The provenance store.
        config: The triage configuration.
        hint: The caller's declaration, the source of the task context.
        ask: Sends a state and questions to the pinned model; None when no server is held, which
            records ``absent`` unless the answer is already cached.
        num_parallel: How many requests the server answers at once; None reads
            ``second_opinion.workers``.

    Returns:
        The outcome; a measurement is written on every path.
    """
    held = settings(config)
    pin = pin_of(config)
    seed = int(held["seed"])
    parallel = int(held["workers"] if num_parallel is None else num_parallel)
    context = task_context(store, hint, task_lexicon(config, declared_task_family(store, hint), hint))
    original, _ = transcript_texts(store)
    masked = masked_text(store, config, hint)[1] if original.strip() else ""
    annotation = find_measurement(store, REDACTION_LLM_ANNOTATION)
    opinion: SecondOpinion | None = None
    failure: str | None = None
    cache: dict[str, Any] = {"key": None, "hit": False}
    if not held["enabled"]:
        status = DISABLED
    elif not original.strip():
        status = NOTHING_TO_READ
    else:
        key = opinion_cache_key(original, masked, context, pin, seed)
        stored = result_lookup(key)
        if stored is not None:
            result = stored["result"]
            opinion = SecondOpinion(
                probabilities={str(k): float(v) for k, v in (result.get("probabilities") or {}).items()},
                choices={str(k): str(v) for k, v in (result.get("choices") or {}).items()},
                class_probabilities={
                    str(question): {str(label): float(p) for label, p in classes.items()}
                    for question, classes in (result.get("class_probabilities") or {}).items()
                },
                raw=dict(result.get("raw") or {}),
            )
            cache = {"key": key, "hit": True}
        elif ask is None:
            failure = "no decision-model server was held for this read"
        else:
            try:
                opinion = ask_second_opinion(ask, original, context, masked)
            except (OSError, ValueError, RuntimeError) as error:
                failure = f"{type(error).__name__}: {error}"
            if opinion is not None:
                wrote = result_store(
                    key,
                    {
                        "probabilities": opinion.probabilities,
                        "choices": opinion.choices,
                        "class_probabilities": opinion.class_probabilities,
                        "raw": opinion.raw,
                    },
                    process=PROCESS,
                    model_id=pin.model_id,
                    commit_sha=pin.blob_digest,
                )
                cache = {"key": key, "hit": False, "stored": wrote}
        status = OK if opinion is not None else ABSENT

    software = software_agent(store)
    activity = store.activity(
        node=NODE,
        step="decide",
        parameters={
            "model_id": pin.model_id,
            "question_set_version": QUESTION_SET_VERSION,
            "seed": seed,
            "num_parallel": parallel,
        },
    )
    store.was_associated_with(activity, software)
    if status in (OK, ABSENT):
        store.was_associated_with(
            activity,
            store.agent(
                agent_type="model",
                model_id=f"{pin.model_id}@{pin.blob_digest}",
                unresolved_reason="pinned by its Ollama blob digest, which is not a git commit",
                version=pin.blob_digest,
            ),
        )
        consensus = find_measurement(store, "consensus_transcript")
        if consensus is not None:
            store.used(activity, consensus.id)
        if annotation is not None:
            store.used(activity, annotation.id)
    measurement_id = mint_live(
        store,
        prov_type="measurement",
        extent=None,
        attributes={
            "name": SECOND_OPINION_ANSWERS,
            "signal": "consensus_transcript",
            "status": status,
            "probabilities": dict(opinion.probabilities) if opinion is not None else {},
            "choices": dict(opinion.choices) if opinion is not None else {},
            "class_probabilities": (
                {question: dict(classes) for question, classes in opinion.class_probabilities.items()}
                if opinion is not None
                else {}
            ),
            "question_set_version": QUESTION_SET_VERSION,
            "model_id": pin.model_id,
            "blob_digest": pin.blob_digest,
            "config_digest": pin.config_digest,
            "manifest_digest": pin.manifest_digest,
            "transcript_chars": len(original),
            "masked_chars": len(masked),
            "masked_differs": masked != original,
            "review_annotation_id": annotation.id if annotation is not None else None,
            "seed": seed,
            "num_parallel": parallel,
            "failure": failure,
            "context_keys": sorted(context),
            "result_cache": cache,
        },
    )
    store.was_generated_by(measurement_id, activity)
    store.was_attributed_to(measurement_id, software)
    return OpinionOutcome(status=status, measurement_id=measurement_id)
