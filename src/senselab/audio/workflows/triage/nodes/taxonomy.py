"""TAXONOMY — which kinds are in the recording, folded from PREPROCESS's stored derivatives.

It runs no model and reads no hint. One exception to "localises nothing": phonation-span
*detection* moved here from PREPROCESS (owner-directed) — PREPROCESS measures F0 and formant
tracks over the whole stream and decides no boundary; this node reads those tracks back and
proposes the sustained-phonation/glide spans over them, the one place in TAXONOMY that places an
extent rather than only counting against a floor. Every other kind's rule reads named evidence
lines, each line counts stored elements against its own configured floor, and a line whose
derivative never reached the store is ``unavailable`` — which makes its kind uncertain, never
absent.

``airway``'s two lines read PREPROCESS's per-span ``span_hear``/``span_yamnet`` measurements
directly (owner-directed this session), not the whole-file pooled windows the ``speech`` acoustic
line still uses — a span already carrying a live consensus word is excluded from both, since ASR is
strictly stronger content evidence than either classifier and an ASR-explained span is not airway
evidence no matter what HeAR or YAMNet also say about it. The two lines are no longer folded by
equal-weight agreement: ``health_acoustic`` (HeAR, the domain-specific detector) is authoritative,
``acoustic`` (YAMNet alone — unlike ``speech``'s own ``acoustic`` line, which pools YAMNet+AST over
whole-file windows; AST carries no per-span measurement for AIRWAY to read) only corroborates — the
same authoritative-plus-corroboration shape ``speech`` already gave its lexical/acoustic pair,
generalised in :func:`_fold_authoritative_line`. Corroboration count is never read as evidence strength: a span's
``corroborated_by`` list says other *sources* also proposed something overlapping it, not that its
*content* is more certain — a vanilla amplitude/continuity span corroborated ten times over still
says nothing about content on its own, so it is not read here at all. A hint is never read here
either (see ``taxonomy()``'s own docstring) — evidence must be found in what the spans actually
show, since a single recording can genuinely carry more than one task's content (a counting task
and a prolonged vowel in the same file), and a hint naming one must never suppress evidence for
the other.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from senselab.audio.data_structures import AudioHints
from senselab.audio.tasks.classification.label_scores import label_scores
from senselab.audio.tasks.phonation.api import (
    FormantTrack,
    propose_phonation_spans,
    propose_word_aligned_phonation_spans,
)
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.common import (
    NodeResult,
    find_measurement,
    find_measurements,
    live_entities,
    software_agent,
    write_measurement,
    write_verdict,
)
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.prov_store import Entity, ProvStore

NODE = "TAXONOMY"

SCREENED_KINDS = ("airway", "speech", "voice")

SUMMARISED_CLASSIFIERS = ("yamnet", "ast", "hear")

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


def _transcribed_span_ids(store: ProvStore) -> set[str]:
    """Every live general span a live consensus word overlaps.

    ASR is the strongest content evidence there is, stronger than any acoustic classifier — a span
    a word already explains is lexical content, not an airway candidate, whatever HeAR or YAMNet
    also fired on it. Mirrors AIRWAY's own ``_is_transcribed`` check exactly (same overlap rule),
    kept here too so TAXONOMY's evidence and AIRWAY's branch never disagree about which spans are
    already explained by the transcript.

    Args:
        store: The provenance store.

    Returns:
        The ids of every live, family-less span overlapping at least one live ``word`` entity.
    """
    words = live_entities(store, "word")
    if not words:
        return set()
    transcribed: set[str] = set()
    for span in live_entities(store, "span"):
        if span.attributes.get("family") is not None or span.extent is None:
            continue
        start, end = span.extent
        if any(w.extent is not None and w.extent[0] < end and w.extent[1] > start for w in words):
            transcribed.add(span.id)
    return transcribed


def _span_label_evidence(
    store: ProvStore, classifier: str, family: set[str], exclude_span_ids: set[str]
) -> dict[str, Any]:
    """One per-span classifier line: how many non-transcribed spans carry a family label.

    Reads PREPROCESS's ``span_hear``/``span_yamnet`` measurements directly — the same per-span
    classification AIRWAY's own branch now reuses rather than re-deriving — instead of the
    whole-file pooled windows the speech acoustic line still reads. A span already explained by a
    live consensus word is never this line's evidence (see :func:`_transcribed_span_ids`): ASR
    outranks both classifiers, so a transcribed span carries no airway content regardless of what
    either model says about it.

    Args:
        store: The provenance store.
        classifier: ``"hear"`` or ``"yamnet"`` — reads that classifier's ``span_<classifier>``
            measurement.
        family: The kind's label family for this classifier.
        exclude_span_ids: Spans a live consensus word already explains.

    Returns:
        ``{available, n_spans, element_ids}``. ``available`` is False when PREPROCESS's own
        ``span_<classifier>`` pass never ran at all (checked by activity, not by measurement count,
        the same distinction :func:`_phonation_spans` draws), and also when it ran but labelled
        nothing because ``windows.<classifier>.default_threshold`` is unmeasured — this line counts
        labels, so windows that were never labelled leave it unable to judge. A window is taken as
        labelled unless it says otherwise, so a store written before ``labelled`` existed keeps its
        meaning. Both are distinct
        from it running over zero spans, or labelling spans and matching none of them, which are
        "ran, found nothing".
    """
    if not [a for a in store.activities("PREPROCESS") if a.step == f"span_{classifier}"]:
        return {"available": False, "n_spans": 0, "element_ids": []}
    windows = find_measurements(store, f"span_{classifier}")
    if windows and all(window.attributes.get("labelled") is False for window in windows):
        return {"available": False, "n_spans": 0, "element_ids": []}
    matched: dict[str, list[str]] = {}
    for window in windows:
        span_id = window.attributes.get("span_id")
        if span_id is None or span_id in exclude_span_ids:
            continue
        labels = {str(label) for label in window.attributes.get("labels") or []}
        if family & labels:
            matched.setdefault(str(span_id), []).append(window.id)
    element_ids = sorted(window_id for ids in matched.values() for window_id in ids)
    return {"available": True, "n_spans": len(matched), "element_ids": element_ids}


def _span_line(evidence: dict[str, Any], floor: Any) -> dict[str, Any]:  # noqa: ANN401
    """One per-span-counting line, as it is written onto the kind element.

    Args:
        evidence: What :func:`_span_label_evidence` returned.
        floor: The line's configured floor.

    Returns:
        The line's state, its evidence, its unit, its floor and the elements it read.
    """
    return {
        "state": _line_state(evidence["available"], evidence["n_spans"], floor),
        "evidence": evidence["n_spans"],
        "unit": "spans",
        "floor": floor,
        "element_ids": evidence["element_ids"],
    }


def _fold_authoritative_line(lines: dict[str, dict[str, Any]], authoritative: str) -> str:
    """The authoritative-plus-corroboration rule: the named line alone decides.

    Generalises the rule ``speech`` already gave its lexical/acoustic pair (the consensus
    transcript decides; an AudioSet label only corroborates) to any kind whose strongest evidence
    source should decide alone, rather than needing independent agreement from a weaker,
    merely-corroborating source. Two sources agreeing is not stronger evidence than one strong
    source alone — HeAR finding a clear cough does not need YAMNet's independent agreement any more
    than a consensus transcript needs an AudioSet speech label's.

    Args:
        lines: The kind's evidence lines, each carrying its own ``state``.
        authoritative: Which line's state the kind's state is copied from.

    Returns:
        ``present`` or ``absent`` when the authoritative line is measured, otherwise ``uncertain``.
    """
    state = str(lines[authoritative]["state"])
    if state == UNAVAILABLE:
        return UNCERTAIN
    return state


def _propose_phonation_spans(store: ProvStore, config: TriageConfig, run_dir: Path, software: str) -> list[str]:
    """Read PREPROCESS's F0/formant tracks and localise the phonation spans over them.

    Guarded, not raised, on absence: no ``phonation_tracks`` measurement (parselmouth missing,
    PREPROCESS itself failed) leaves this a no-op, and :func:`_phonation_spans` below reports that
    as "the pass did not run" — the same cascading-absence contract every PREPROCESS-dependent
    reader in this node already follows.

    Args:
        store: The provenance store, holding PREPROCESS's ``phonation_tracks`` measurement.
        config: The triage configuration; reads the same ``phonation_spans.*`` keys the detector
            used before it moved here — the values did not change meaning, only which node applies
            them.
        run_dir: Where PREPROCESS wrote ``derivatives/phonation_tracks.npz``.
        software: The agent answerable for the spans this writes.

    Returns:
        Every entity id written (each span plus its own ``formant_tracks`` sub-measurement), for the
        caller's view; empty when ``phonation_tracks`` is absent or no span was proposed.
    """
    tracks = find_measurement(store, "phonation_tracks")
    if tracks is None:
        return []
    hop_s = float(tracks.attributes["hop_s"])
    parameters: dict[str, Any] = {
        "hop_s": hop_s,
        "f0_stability_cents": float(config.require("phonation_spans.f0_stability_cents")),
        "formant_stability_hz": float(config.require("phonation_spans.formant_stability_hz")),
        "glide_min_excursion_cents": float(config.require("phonation_spans.glide_min_excursion_cents")),
        "hangover_ms": float(config.require("phonation_spans.hangover_ms")),
        "voicing_strength_floor": float(config.require("phonation_spans.voicing_strength_floor")),
        "mixed_voiced_fraction": float(config.require("phonation_spans.mixed_voiced_fraction")),
        "unvoiced_max_formant_bandwidth_hz": float(config.require("phonation_spans.unvoiced_max_formant_bandwidth_hz")),
        "word_aligned_min_evidence_fraction": float(
            config.require("phonation_spans.word_aligned_min_evidence_fraction")
        ),
    }
    activity = store.activity(node=NODE, step="phonation_spans", parameters=parameters)
    store.was_associated_with(activity, software)
    store.used(activity, tracks.id)

    npz = np.load(run_dir / "derivatives" / "phonation_tracks.npz")
    times, f0_hz, strength = npz["times_s"], npz["f0_hz"], npz["strength"]
    formants = FormantTrack(
        times_s=npz["formant_times_s"],
        f_hz=tuple(npz[f"f{order + 1}_hz"] for order in range(4)),  # type: ignore[arg-type]
        bandwidth_hz=tuple(npz[f"f{order + 1}_bw_hz"] for order in range(4)),  # type: ignore[arg-type]
    )
    proposals = propose_phonation_spans(
        times=times,
        f0_hz=f0_hz,
        strength=strength,
        formants=formants,
        hop_s=parameters["hop_s"],
        f0_stability_cents=parameters["f0_stability_cents"],
        formant_stability_hz=parameters["formant_stability_hz"],
        glide_min_excursion_cents=parameters["glide_min_excursion_cents"],
        hangover_ms=parameters["hangover_ms"],
        voicing_strength_floor=parameters["voicing_strength_floor"],
        mixed_voiced_fraction=parameters["mixed_voiced_fraction"],
        unvoiced_max_formant_bandwidth_hz=parameters["unvoiced_max_formant_bandwidth_hz"],
    )
    consensus = find_measurement(store, "consensus_transcript")
    word_ids = list(consensus.attributes["word_ids"]) if consensus is not None else []
    word_extents = [store.get_entity(word_id).extent for word_id in word_ids]
    word_spans = propose_word_aligned_phonation_spans(
        times=times,
        f0_hz=f0_hz,
        strength=strength,
        formants=formants,
        word_extents=[extent for extent in word_extents if extent is not None],
        voicing_strength_floor=parameters["voicing_strength_floor"],
        mixed_voiced_fraction=parameters["mixed_voiced_fraction"],
        unvoiced_max_formant_bandwidth_hz=parameters["unvoiced_max_formant_bandwidth_hz"],
        min_evidence_fraction=parameters["word_aligned_min_evidence_fraction"],
    )
    word_sources = {
        extent: word_id for word_id, extent in zip(word_ids, word_extents, strict=True) if extent is not None
    }
    proposals.extend(
        proposal
        for proposal in word_spans
        if not any(existing.start <= proposal.start and proposal.end <= existing.end for existing in proposals)
    )
    signal = str(tracks.attributes.get("signal") or "plain")
    written: list[str] = []
    for proposal in proposals:
        span_id = store.entity(
            prov_type="span",
            extent=(proposal.start, proposal.end),
            attributes={
                "family": "phonation",
                "member": proposal.member,
                "duration_s": proposal.end - proposal.start,
                "production": proposal.production,
                "voiced_fraction": proposal.voiced_fraction,
                "f0_median_hz": proposal.f0_median_hz,
                "f0_start_hz": proposal.f0_start_hz,
                "f0_end_hz": proposal.f0_end_hz,
                "glide_direction": proposal.glide_direction,
                "glide_extent_cents": proposal.glide_extent_cents,
                "offset_criterion": proposal.offset_criterion,
                "signal": signal,
                "hop_s": hop_s,
            },
        )
        store.was_generated_by(span_id, activity)
        store.was_attributed_to(span_id, software)
        store.was_derived_from(span_id, tracks.id)
        if proposal.member == "word_aligned":
            store.was_derived_from(span_id, word_sources[(proposal.start, proposal.end)])
        written.append(span_id)
        inside = (formants.times_s >= proposal.start) & (formants.times_s < proposal.end)
        track_id = write_measurement(
            store,
            activity,
            software,
            name="formant_tracks",
            signal=signal,
            extent=(proposal.start, proposal.end),
            attributes={
                "times_s": formants.times_s[inside].tolist(),
                "hop_s": hop_s,
                **{f"f{order + 1}_hz": formants.f_hz[order][inside].tolist() for order in range(4)},
                **{f"f{order + 1}_bw_hz": formants.bandwidth_hz[order][inside].tolist() for order in range(4)},
            },
            derived_from=(span_id,),
        )
        written.append(track_id)
    return written


def _phonation_spans(store: ProvStore) -> list[Entity] | None:
    """This node's own phonation spans, or None when the pass left nothing in the store at all.

    Args:
        store: The provenance store.

    Returns:
        The live phonation spans, possibly empty; None when no phonation activity ran, so a reader
        can tell "the pass found nothing" from "the pass did not happen".
    """
    if not [activity for activity in store.activities(NODE) if activity.step == "phonation_spans"]:
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


def _fold_speech_lines(lines: dict[str, dict[str, Any]]) -> str:
    """Classify lexical speech from the authoritative consensus transcript.

    The consensus is the workflow's authoritative ASR product. A completed consensus with no
    words therefore rules out lexical speech, even if an AudioSet model emitted an isolated speech
    label; the acoustic line remains recorded as corroborating evidence. A missing or unfitted
    lexical line is genuinely unknown and remains uncertain. A thin, specifically-named wrapper over
    :func:`_fold_authoritative_line` — kept separate for the docstring above, not a different rule.

    Args:
        lines: The speech acoustic and lexical evidence lines.

    Returns:
        ``present`` or ``absent`` when the lexical line is measured, otherwise ``uncertain``.
    """
    return _fold_authoritative_line(lines, "lexical")


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


def _label_score_distribution(store: ProvStore, classifier: str, run_dir: Path) -> dict[str, Any] | None:
    """Every label's score distribution over the whole recording, with no threshold applied.

    Args:
        store: The provenance store, holding PREPROCESS's verbatim score measurement.
        classifier: ``"yamnet"``, ``"ast"`` or ``"hear"``.
        run_dir: Where PREPROCESS wrote the classifier's windows.

    Returns:
        ``{n_windows, win_length_s, hop_s, labels, element_id}`` where ``labels`` maps a label to
        ``{peak, median, n_windows}``, ordered by descending peak; or None when the measurement or
        its sidecar is absent.
    """
    raw = find_measurement(store, f"{classifier}_scores")
    if raw is None:
        return None
    path = raw.attributes.get("path")
    if not path:
        return None
    sidecar = run_dir / str(path)
    if not sidecar.is_file():
        return None
    windows = json.loads(sidecar.read_text())
    per_label: dict[str, list[float]] = {}
    for window in windows:
        for pair in label_scores(window):
            for label, score in pair.items():
                per_label.setdefault(str(label), []).append(float(score))
    labels = {
        label: {"peak": float(max(scores)), "median": float(np.median(scores)), "n_windows": len(scores)}
        for label, scores in per_label.items()
    }
    return {
        "n_windows": len(windows),
        "win_length_s": raw.attributes.get("win_length_s"),
        "hop_s": raw.attributes.get("hop_s"),
        "labels": dict(sorted(labels.items(), key=lambda item: -item[1]["peak"])),
        "element_id": raw.id,
    }


def _write_label_summaries(store: ProvStore, run_dir: Path, software: str) -> list[str]:
    """Write one whole-file label-score summary per classifier that produced scores.

    Args:
        store: The provenance store.
        run_dir: Where PREPROCESS wrote the score sidecars.
        software: This node's software agent.

    Returns:
        The ids written, for the node's view. A classifier whose scores are absent contributes
        nothing rather than an empty summary, so a missing summary and an all-zero one stay
        distinguishable.
    """
    written: list[str] = []
    for classifier in SUMMARISED_CLASSIFIERS:
        distribution = _label_score_distribution(store, classifier, run_dir)
        if distribution is None:
            continue
        element_id = str(distribution.pop("element_id"))
        activity = store.activity(node=NODE, step=f"{classifier}_label_summary", parameters={"classifier": classifier})
        store.was_associated_with(activity, software)
        store.used(activity, element_id)
        written.append(
            write_measurement(
                store,
                activity,
                software,
                name=f"{classifier}_label_summary",
                signal="plain",
                attributes={"classifier": classifier, **distribution},
                derived_from=(element_id,),
                extent=None,
            )
        )
    return written


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
        run_dir: Where PREPROCESS wrote ``derivatives/phonation_tracks.npz`` and each classifier's
            verbatim ``derivatives/<classifier>_scores.json`` — the sidecars this node reads. It
            writes none of its own.

    Returns:
        The verdict, the three kind element ids plus each classifier's whole-file label summary as
        the view, and the state per kind.
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
    transcribed = _transcribed_span_ids(store)

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
            "health_acoustic": _span_line(
                _span_label_evidence(store, "hear", hear_airway, transcribed),
                floors[("airway", "health_acoustic")],
            ),
            "acoustic": _span_line(
                _span_label_evidence(store, "yamnet", audioset_airway, transcribed),
                floors[("airway", "acoustic")],
            ),
        },
    }
    summary_view = _write_label_summaries(store, run_dir, software)
    phonation_view = _propose_phonation_spans(store, config, run_dir, software)
    voice_line, voice_state = _voice_line(
        _phonation_spans(store),
        config.get("taxonomy.voice_min_duration_s"),
        config.get("taxonomy.voice_uncertain_duration_s"),
    )
    lines["voice"] = {"phonation": voice_line}

    states = {
        "speech": _fold_speech_lines(lines["speech"]),
        "airway": _fold_authoritative_line(lines["airway"], "health_acoustic"),
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

    view: list[str] = list(summary_view) + list(phonation_view)
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
