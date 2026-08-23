"""TAXONOMY — which kinds are in the recording. Advisory: it predicts, and gates nothing.

Each detector answers presence on its own grid; families vote, not detectors. Presence needs
``min_families[kind]`` agreement — unanimity while that count is unmeasured — and absence needs
unanimity of the eligible families. Every branch runs regardless of the outcome here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from senselab.audio.data_structures import AudioHints
from senselab.audio.tasks.classification.api import classify_audios
from senselab.audio.tasks.classification.label_scores import label_scores
from senselab.audio.tasks.health_acoustics.api import detect_health_acoustic_events
from senselab.audio.tasks.health_acoustics.hear import HEAR_MODEL_ID, HEAR_REVISION
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.common import (
    NodeResult,
    find_measurement,
    resolve_stream,
    software_agent,
    write_verdict,
)
from senselab.audio.workflows.triage.nodes.preprocess import CRISPERWHISPER_ID
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.data_structures import HFModel
from senselab.utils.prov_store import ProvStore

NODE = "TAXONOMY"
AST_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"

SCREENED_KINDS = ("airway", "speech")


def _ast_model() -> HFModel:
    """The AST model spec; its commit resolves at construction."""
    return HFModel(path_or_uri=AST_ID, revision="main")


@dataclass(frozen=True)
class TaxonomyResult(NodeResult):
    """TAXONOMY's result.

    Attributes:
        kinds: Predicted state per kind — the design's verdict mapping.
    """

    kinds: dict[str, str]


def _windowed_max(windows: list[dict[str, Any]], labels: set[str]) -> tuple[float, str | None]:
    """The highest score any of these labels reaches in any window, and which label reached it."""
    best, best_label = 0.0, None
    for window in windows:
        for pair in label_scores(window):
            for label, score in pair.items():
                if label in labels and float(score) > best:
                    best, best_label = float(score), label
    return best, best_label


def _is_bracketed(text: str) -> bool:
    """Whether a recognizer token is a non-lexical annotation like ``[cough]``."""
    return text.startswith("[") and text.endswith("]")


def taxonomy(  # noqa: C901 — one member per detector, one fold per kind
    store: ProvStore,
    source: str,
    config: TriageConfig,
    hint: AudioHints | None = None,
    *,
    run_dir: Path,
) -> TaxonomyResult:
    """Predict which kinds are in the recording. Nothing downstream is gated on the answer.

    Args:
        store: The provenance store, holding PREPROCESS's derivatives.
        source: The store-held stream name to classify, ``"plain"``.
        config: The triage configuration.
        hint: Accepted for the shared node shape; not read (the design's signature has none).
        run_dir: The run directory sidecar paths are relative to.

    Returns:
        The verdict, the kind entity ids as the view, and the per-kind states.

    Raises:
        ValueError: If a ``taxonomy.min_families`` override lies outside ``[1, n_eligible]``.
    """
    software = software_agent(store)
    stream_id, plain = resolve_stream(store, run_dir, source)

    floors = {
        "yamnet": config.get("taxonomy.presence_floor.yamnet"),
        "ast": config.get("taxonomy.presence_floor.ast"),
        "hear": config.get("taxonomy.presence_floor.hear"),
    }
    audioset_labels = {
        "airway": {str(label) for label in config.require("taxonomy.audioset_airway_labels")},
        "speech": {str(label) for label in config.require("taxonomy.audioset_speech_labels")},
    }
    hear_labels = {str(label) for label in config.require("taxonomy.hear_airway_labels")}
    lexical_tokens = [str(token).lower() for token in config.require("taxonomy.lexical_airway_tokens")]
    ast_frame_s = float(config.require("taxonomy.ast_frame_s"))

    yamnet_meas = find_measurement(store, "yamnet_windows")
    yamnet_windows: list[dict[str, Any]] | None = None
    if yamnet_meas is not None:
        yamnet_windows = json.loads((run_dir / yamnet_meas.attributes["path"]).read_text())

    ast_scores: dict[str, float] | None = None
    ast_error: str | None = None
    try:
        model = _ast_model()
        ast_agent = store.agent(agent_type="model", model_id=str(model.path_or_uri), commit_sha=model.commit_sha)
        ast_activity = store.activity(
            node=NODE,
            step="classify_ast",
            parameters={"model": str(model.path_or_uri), "function_to_apply": "sigmoid", "top_k": None},
        )
        store.was_associated_with(ast_activity, ast_agent)
        store.used(ast_activity, stream_id)
        [ast_result] = classify_audios([plain], model=model, function_to_apply="sigmoid", top_k=None)
        ast_scores = {label: float(score) for label, score in zip(ast_result.labels, ast_result.scores)}
    except Exception as err:  # noqa: BLE001 — an unavailable detector abstains; it is not absence evidence
        ast_error = type(err).__name__

    hear_windows: list[dict[str, Any]] | None = None
    hear_error: str | None = None
    try:
        hear_agent = store.agent(agent_type="model", model_id=HEAR_MODEL_ID, commit_sha=HEAR_REVISION)
        hear_activity = store.activity(node=NODE, step="classify_hear", parameters={"model": HEAR_MODEL_ID})
        store.was_associated_with(hear_activity, hear_agent)
        store.used(hear_activity, stream_id)
        [hear_windows] = detect_health_acoustic_events([plain], top_k=None)
    except Exception as err:  # noqa: BLE001 — same rule as AST
        hear_error = type(err).__name__

    words = [
        w
        for w in store.entities("word")
        if w.attributes.get("recognizer") == CRISPERWHISPER_ID and not store.is_invalidated(w.id)
    ]
    crisper_available = find_measurement(store, "asr_crisperwhisper") is not None

    def _yamnet_member(kind: str) -> dict[str, Any]:
        """Family A's first member, read from the store's native windows."""
        if not yamnet_windows:
            why = "yamnet_windows absent from the store" if yamnet_windows is None else "yamnet_windows is empty"
            return {"state": "unavailable", "why": why}
        if floors["yamnet"] is None:
            return {"state": "abstained", "why": "presence floor unmeasured"}
        best, best_label = _windowed_max(yamnet_windows, audioset_labels[kind])
        state = "present" if best >= float(floors["yamnet"]) else "absent"
        return {"state": state, "max_score": best, "label": best_label}

    def _ast_member(kind: str) -> dict[str, Any]:
        """Family A's second member, file-level over the model's fixed ``taxonomy.ast_frame_s`` frame."""
        if ast_scores is None:
            return {"state": "unavailable", "why": ast_error or "no scores"}
        best, best_label = 0.0, None
        for label, score in ast_scores.items():
            if label in audioset_labels[kind] and score > best:
                best, best_label = score, label
        if floors["ast"] is None:
            return {
                "state": "abstained",
                "why": "presence floor unmeasured",
                "max_score": best,
                "label": best_label,
                "frame_s": ast_frame_s,
            }
        state = "present" if best >= float(floors["ast"]) else "absent"
        return {"state": state, "max_score": best, "label": best_label, "frame_s": ast_frame_s}

    def _lexical_member(kind: str) -> dict[str, Any]:
        """Family B: words for speech, bracketed non-lexical tokens for airway."""
        if not crisper_available:
            return {"state": "unavailable", "why": "asr_crisperwhisper absent from the store"}
        if kind == "speech":
            lexical = [w for w in words if w.attributes.get("text") and not _is_bracketed(str(w.attributes["text"]))]
            return {"state": "present" if lexical else "absent", "n_words": len(lexical)}
        matched = [
            w.id
            for w in words
            if w.attributes.get("text")
            and _is_bracketed(str(w.attributes["text"]))
            and any(token in str(w.attributes["text"]).lower() for token in lexical_tokens)
        ]
        return {"state": "present" if matched else "absent", "word_ids": matched}

    def _hear_member() -> dict[str, Any]:
        """Family C, airway only: the detector's own sliding grid."""
        if not hear_windows:
            return {"state": "unavailable", "why": hear_error or "no windows"}
        if floors["hear"] is None:
            return {"state": "abstained", "why": "presence floor unmeasured"}
        best, best_label = _windowed_max(hear_windows, hear_labels)
        state = "present" if best >= float(floors["hear"]) else "absent"
        return {"state": state, "max_score": best, "label": best_label}

    def _family_a(kind: str) -> dict[str, Any]:
        """AudioSet family: members must agree; an abstaining member leaves it to the other."""
        members = {"yamnet": _yamnet_member(kind), "ast": _ast_member(kind)}
        votes = [m["state"] for m in members.values() if m["state"] in ("present", "absent")]
        if votes and all(v == votes[0] for v in votes):
            state = votes[0]
        else:
            state = "unsure"
        return {"state": state, "members": members}

    def _single(member_name: str, member: dict[str, Any]) -> dict[str, Any]:
        """A one-member family: the member's vote, unsure when it cannot vote."""
        state = member["state"] if member["state"] in ("present", "absent") else "unsure"
        return {"state": state, "members": {member_name: member}}

    def _fold_kind(kind: str, families: dict[str, dict[str, Any]]) -> tuple[str, Any]:
        """The design's presence/absence/undecided fold, honest about the unmeasured count.

        An override is validated before any state is read, so a bad count raises whatever the
        recording contains.
        """
        states = [family["state"] for family in families.values()]
        min_families = config.get(f"taxonomy.min_families.{kind}")
        if min_families is not None:
            min_int = int(min_families)
            if not 1 <= min_int <= len(states):
                raise ValueError(
                    f"taxonomy.min_families.{kind} = {min_int} lies outside [1, {len(states)}] eligible families"
                )
            if all(state == "absent" for state in states):
                return "absent", min_int
            if sum(1 for state in states if state == "present") >= min_int:
                return "present", min_int
            return "undecided", min_int
        if states and all(state == "absent" for state in states):
            return "absent", "unmeasured"
        if states and all(state == "present" for state in states):
            return "present", "unmeasured"
        return "undecided", "unmeasured"

    airway_families = {
        "A_audioset": _family_a("airway"),
        "B_lexical": _single("crisperwhisper", _lexical_member("airway")),
        "C_health": _single("hear", _hear_member()),
    }
    # HeAR is barred from the speech kind: family C is not eligible and never enters this fold.
    speech_families = {
        "A_audioset": _family_a("speech"),
        "B_lexical": _single("crisperwhisper", _lexical_member("speech")),
    }

    fold = store.activity(node=NODE, step="fold", parameters={"kinds": list(SCREENED_KINDS)})
    store.was_associated_with(fold, software)
    store.used(fold, stream_id)
    if yamnet_meas is not None:
        store.used(fold, yamnet_meas.id)
    for word in words:
        store.used(fold, word.id)

    kinds_out: dict[str, str] = {}
    view: list[str] = []
    for kind, families in (("airway", airway_families), ("speech", speech_families)):
        state, min_recorded = _fold_kind(kind, families)
        kinds_out[kind] = state
        kind_id = store.entity(
            prov_type="kind",
            extent=None,
            attributes={"kind": kind, "state": state, "families": families, "min_families": min_recorded},
        )
        store.was_generated_by(kind_id, fold)
        store.was_attributed_to(kind_id, software)
        view.append(kind_id)

    residual_id = store.entity(
        prov_type="kind",
        extent=None,
        attributes={"kind": "voice_no_words", "state": "not_screened", "families": {}, "min_families": None},
    )
    store.was_generated_by(residual_id, fold)
    store.was_attributed_to(residual_id, software)
    view.append(residual_id)
    kinds_out["voice_no_words"] = "not_screened"

    screened = [kinds_out[kind] for kind in SCREENED_KINDS]
    if all(state == "absent" for state in screened):
        outcome, why = Outcome.FAIL, "every screened kind is absent; nothing is predicted present"
    elif any(state == "undecided" for state in screened):
        undecided = [kind for kind in SCREENED_KINDS if kinds_out[kind] == "undecided"]
        outcome, why = Outcome.FLAG, "undecided: " + ", ".join(undecided)
    else:
        outcome, why = Outcome.PASS, "every screened kind is decided, and at least one is present"

    verdict_id, verdict = write_verdict(
        store, fold, software, node=NODE, outcome=outcome, kind=None, why=why, detail={"kinds": kinds_out}
    )
    view.append(verdict_id)
    return TaxonomyResult(verdict=verdict, view=tuple(view), verdict_entity_id=verdict_id, kinds=kinds_out)
