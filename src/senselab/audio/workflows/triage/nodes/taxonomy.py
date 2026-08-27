"""TAXONOMY — which kinds are in the recording, folded from PREPROCESS's stored derivatives.

It runs no model, reads no hint and localises nothing. Each kind's rule reads named evidence lines,
each line counts stored elements against its own configured floor, and a line whose derivative never
reached the store is ``unavailable`` — which makes its kind uncertain, never absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.common import (
    NodeResult,
    find_measurement,
    live_entities,
    software_agent,
    write_verdict,
)
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.prov_store import Entity, ProvStore

NODE = "TAXONOMY"

SCREENED_KINDS = ("airway", "speech", "voice")

PRESENT = "present"
ABSENT = "absent"
UNCERTAIN = "uncertain"
UNAVAILABLE = "unavailable"

_PHONATION_FAMILY = "phonation"


@dataclass(frozen=True)
class TaxonomyResult(NodeResult):
    """TAXONOMY's result.

    Attributes:
        kinds: The classified state per kind — ``present``, ``absent`` or ``uncertain``.
    """

    kinds: dict[str, str]


def _unavailable_windows() -> dict[str, Any]:
    """The evidence a line reports when the derivative it reads is not in the store.

    Returns:
        ``{available, n_windows, element_ids}`` with nothing counted.
    """
    return {"available": False, "n_windows": 0, "element_ids": []}


def _window_evidence(store: ProvStore, classifier: str, family: set[str]) -> dict[str, Any]:
    """One acoustic line's evidence: how many of this classifier's windows carry a family member.

    Args:
        store: The provenance store.
        classifier: ``"yamnet"``, ``"ast"`` or ``"hear"``.
        family: The kind's label family for this classifier.

    Returns:
        ``{available, n_windows, element_ids}``. ``available`` is False when the classifier's pooled
        measurement is absent, which is the state a null threshold leaves.
    """
    pooled = find_measurement(store, f"{classifier}_windows")
    if pooled is None:
        return _unavailable_windows()
    windows_by_label: dict[str, list[str]] = pooled.attributes.get("windows_by_label") or {}
    matched = {window_id for label, ids in windows_by_label.items() if label in family for window_id in ids}
    return {"available": True, "n_windows": len(matched), "element_ids": sorted(matched)}


def _acoustic_line(store: ProvStore, family: set[str]) -> dict[str, Any]:
    """The AudioSet line, over YAMNet and AST together: either grid's windows are acoustic evidence.

    Args:
        store: The provenance store.
        family: The kind's AudioSet label family.

    Returns:
        ``{available, n_windows, element_ids}`` over both grids; unavailable only when neither ran.
    """
    yamnet = _window_evidence(store, "yamnet", family)
    ast = _window_evidence(store, "ast", family)
    if not yamnet["available"] and not ast["available"]:
        return _unavailable_windows()
    return {
        "available": True,
        "n_windows": yamnet["n_windows"] + ast["n_windows"],
        "element_ids": yamnet["element_ids"] + ast["element_ids"],
    }


def _lexical_line(store: ProvStore) -> dict[str, Any]:
    """The lexical line: consensus ``word`` entities. Bracketed and onomatopoeic events are not words.

    Args:
        store: The provenance store.

    Returns:
        ``{available, n_words, element_ids}``; unavailable when no consensus transcript was written.
    """
    if find_measurement(store, "consensus_transcript") is None:
        return {"available": False, "n_words": 0, "element_ids": []}
    words = live_entities(store, "word")
    return {"available": True, "n_words": len(words), "element_ids": [w.id for w in words]}


def _phonation_spans(store: ProvStore) -> list[Entity] | None:
    """PREPROCESS's phonation spans, or None when the pass left nothing in the store at all.

    Args:
        store: The provenance store.

    Returns:
        The live phonation spans, possibly empty; None when no phonation activity ran, so a reader
        can tell "the pass found nothing" from "the pass did not happen".
    """
    if not [activity for activity in store.activities("PREPROCESS") if activity.step == "phonation_spans"]:
        return None
    return [e for e in live_entities(store, "span") if e.attributes.get("family") == _PHONATION_FAMILY]


def _line_state(available: bool, evidence: int, floor: Any) -> str:  # noqa: ANN401
    """One line's state from its evidence and its floor.

    Args:
        available: Whether the derivative the line reads is in the store.
        evidence: The count the line measured.
        floor: The configured floor, or None while it is unmeasured.

    Returns:
        ``unavailable`` when the derivative is missing or the floor is unmeasured — a line that
        cannot be judged has said nothing, which is not the same as saying absent — else
        ``present`` or ``absent``.
    """
    if not available or floor is None:
        return UNAVAILABLE
    return PRESENT if evidence >= int(floor) else ABSENT


def _window_line(evidence: dict[str, Any], floor: Any) -> dict[str, Any]:  # noqa: ANN401
    """One window-counting line, as it is written onto the kind element.

    Args:
        evidence: What :func:`_window_evidence` or :func:`_acoustic_line` returned.
        floor: The line's configured floor.

    Returns:
        The line's state, its evidence, its unit, its floor and the elements it read.
    """
    return {
        "state": _line_state(evidence["available"], evidence["n_windows"], floor),
        "evidence": evidence["n_windows"],
        "unit": "windows",
        "floor": floor,
        "element_ids": evidence["element_ids"],
    }


def _fold_two_lines(lines: dict[str, dict[str, Any]]) -> str:
    """The two-line rule: present when both carry evidence, absent when neither does, else uncertain.

    Args:
        lines: The kind's evidence lines, each carrying its ``state``.

    Returns:
        ``present``, ``absent`` or ``uncertain``.
    """
    states = [line["state"] for line in lines.values()]
    if all(state == PRESENT for state in states):
        return PRESENT
    if all(state == ABSENT for state in states):
        return ABSENT
    return UNCERTAIN


def _fold_speech_lines(lines: dict[str, dict[str, Any]]) -> str:
    """Classify lexical speech from the authoritative consensus transcript.

    The consensus is the workflow's authoritative ASR product. A completed consensus with no
    words therefore rules out lexical speech, even if an AudioSet model emitted an isolated speech
    label; the acoustic line remains recorded as corroborating evidence. A missing or unfitted
    lexical line is genuinely unknown and remains uncertain.

    Args:
        lines: The speech acoustic and lexical evidence lines.

    Returns:
        ``present`` or ``absent`` when the lexical line is measured, otherwise ``uncertain``.
    """
    lexical_state = str(lines["lexical"]["state"])
    if lexical_state == UNAVAILABLE:
        return UNCERTAIN
    return lexical_state


def _voice_line(spans: list[Entity] | None, min_s: Any, uncertain_s: Any) -> tuple[dict[str, Any], str]:  # noqa: ANN401
    """The voice kind's single line, from the longest phonation span's duration alone.

    Args:
        spans: The live phonation spans, or None when the pass did not run.
        min_s: ``taxonomy.voice_min_duration_s``, or None while it is unmeasured.
        uncertain_s: ``taxonomy.voice_uncertain_duration_s``, or None while it is unmeasured.

    Returns:
        The line as it is written onto the kind element, and the kind's state.
    """
    longest_s = max((float(e.attributes["duration_s"]) for e in spans), default=0.0) if spans else 0.0
    if spans is None or min_s is None or uncertain_s is None:
        line_state, kind_state = UNAVAILABLE, UNCERTAIN
    elif longest_s >= float(min_s):
        line_state, kind_state = PRESENT, PRESENT
    elif longest_s >= float(uncertain_s):
        line_state, kind_state = PRESENT, UNCERTAIN
    else:
        line_state, kind_state = ABSENT, ABSENT
    line = {
        "state": line_state,
        "evidence": longest_s,
        "unit": "seconds",
        "floor": min_s,
        "uncertain_floor": uncertain_s,
        "element_ids": [e.id for e in spans] if spans else [],
    }
    return line, kind_state


def taxonomy(
    store: ProvStore,
    source: str,
    config: TriageConfig,
    hint: AudioHints | None = None,
    *,
    run_dir: Path,
) -> TaxonomyResult:
    """Classify which kinds are in the recording, from the store alone.

    Args:
        store: The provenance store, holding PREPROCESS's derivatives.
        source: The stream every element it writes names, ``"plain"``.
        config: The triage configuration.
        hint: Accepted for the shared node shape and **not read**. A classification that reads the
            declaration cannot disagree with it; forcing a branch is ``routing``'s job.
        run_dir: Accepted for the shared node shape; this node writes no sidecars.

    Returns:
        The verdict, the three kind element ids as the view, and the state per kind.
    """
    software = software_agent(store)
    speech_family = {str(label) for label in (config.get("taxonomy.speech_labels") or [])}
    audioset_airway = {str(label) for label in config.require("taxonomy.audioset_airway_labels")}
    hear_airway = {str(label) for label in config.require("taxonomy.hear_airway_labels")}
    floors = {
        ("speech", "acoustic"): config.get("taxonomy.presence_floor.speech.acoustic"),
        ("speech", "lexical"): config.get("taxonomy.presence_floor.speech.lexical"),
        ("airway", "health_acoustic"): config.get("taxonomy.presence_floor.airway.health_acoustic"),
        ("airway", "acoustic"): config.get("taxonomy.presence_floor.airway.acoustic"),
    }

    speech_acoustic = _acoustic_line(store, speech_family) if speech_family else _unavailable_windows()
    speech_lexical = _lexical_line(store)
    lexical_floor = floors[("speech", "lexical")]

    lines: dict[str, dict[str, dict[str, Any]]] = {
        "speech": {
            "acoustic": _window_line(speech_acoustic, floors[("speech", "acoustic")]),
            "lexical": {
                "state": _line_state(speech_lexical["available"], speech_lexical["n_words"], lexical_floor),
                "evidence": speech_lexical["n_words"],
                "unit": "words",
                "floor": lexical_floor,
                "element_ids": speech_lexical["element_ids"],
            },
        },
        "airway": {
            "health_acoustic": _window_line(
                _window_evidence(store, "hear", hear_airway), floors[("airway", "health_acoustic")]
            ),
            "acoustic": _window_line(_acoustic_line(store, audioset_airway), floors[("airway", "acoustic")]),
        },
    }
    voice_line, voice_state = _voice_line(
        _phonation_spans(store),
        config.get("taxonomy.voice_min_duration_s"),
        config.get("taxonomy.voice_uncertain_duration_s"),
    )
    lines["voice"] = {"phonation": voice_line}

    states = {
        "speech": _fold_speech_lines(lines["speech"]),
        "airway": _fold_two_lines(lines["airway"]),
        "voice": voice_state,
    }

    fold = store.activity(node=NODE, step="fold", parameters={"kinds": list(SCREENED_KINDS), "stream": source})
    store.was_associated_with(fold, software)
    read_ids = {
        element_id
        for kind_lines in lines.values()
        for line in kind_lines.values()
        for element_id in line["element_ids"]
    }
    for element_id in sorted(read_ids):
        store.used(fold, element_id)

    view: list[str] = []
    for kind in SCREENED_KINDS:
        kind_id = store.entity(
            prov_type="kind",
            extent=None,
            attributes={"kind": kind, "state": states[kind], "lines": lines[kind], "stream": source},
        )
        store.was_generated_by(kind_id, fold)
        store.was_attributed_to(kind_id, software)
        view.append(kind_id)

    if all(state == ABSENT for state in states.values()):
        outcome, why = Outcome.FAIL, "every kind is absent"
    elif any(state == UNCERTAIN for state in states.values()):
        uncertain = [kind for kind in SCREENED_KINDS if states[kind] == UNCERTAIN]
        outcome, why = Outcome.FLAG, "uncertain: " + ", ".join(uncertain)
    else:
        outcome, why = Outcome.PASS, "every kind is present or absent, and at least one is present"

    verdict_id, verdict = write_verdict(
        store, fold, software, node=NODE, outcome=outcome, kind=None, why=why, detail={"kinds": states}
    )
    view.append(verdict_id)
    return TaxonomyResult(verdict=verdict, view=tuple(view), verdict_entity_id=verdict_id, kinds=states)
