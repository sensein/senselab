"""The SPEECH branch: transcript agreement, spans from word timings, diarization, PII, quality.

Speech spans come from ASR word timings, never the envelope; the node runs no ASR — it reads and
fuses the two hypotheses PREPROCESS wrote. pyannote sees only ``[first word start, last word end]``.
Diarization is about speech, so a diarizer segment is never withdrawn for overlapping an airway
event and the speaker count is the live segment count. The PII
decision is this branch's own and is speaker-scoped, and the ``pii_scan`` measurement is written on
every path, including the one with no words, where it records that nothing was scanned. Quality is
reported, never gating; SQUIM reads ``plain`` while the disruption counts and the zero-crossing rate
read the original ``recording``, since normalising and resampling destroy the defects they look for.
Every parameter's derivation is in ``data/config/default.yaml``.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from senselab.audio.data_structures import Audio, AudioHints, TargetSpeakerEmbedding
from senselab.audio.tasks.classification.label_scores import label_scores
from senselab.audio.tasks.disruptions.api import detect_disruptions
from senselab.audio.tasks.features_extraction.torchaudio_squim import (
    extract_objective_quality_features_from_audios,
)
from senselab.audio.tasks.preprocessing.preprocessing import extract_segments
from senselab.audio.tasks.source_separation.api import separate_audios
from senselab.audio.tasks.speaker_diarization.api import diarize_audios
from senselab.audio.tasks.speaker_embeddings.api import extract_speaker_embeddings_from_audios
from senselab.audio.tasks.speech_to_text_ensemble.api import fuse_word_streams
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.common import (
    NodeResult,
    clamp_extent,
    find_measurement,
    resolve_stream,
    software_agent,
    write_verdict,
)
from senselab.audio.workflows.triage.nodes.preprocess import CRISPERWHISPER_ID, QWEN_ID
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.text.tasks.pii_detection.api import PiiScan, scan_for_pii
from senselab.utils.data_structures import HFModel, PyannoteAudioModel, ScriptLine, SpeechBrainModel
from senselab.utils.prov_store import Entity, ProvStore

NODE = "SPEECH"
EMBEDDING_ID = "speechbrain/spkrec-ecapa-voxceleb"
ORIGINAL = "recording"  # the stream disruptions are measured on: as captured, unnormalised, unresampled


def _diarization_model() -> PyannoteAudioModel:
    """The diarizer's model spec; its commit resolves at construction."""
    return PyannoteAudioModel(path_or_uri="pyannote/speaker-diarization-community-1", revision="main")


def _second_diarizer_model(model_id: str) -> HFModel:
    """The configured second diarizer's model spec; its commit resolves at construction."""
    return HFModel(path_or_uri=model_id, revision="main")


def _separation_model() -> HFModel:
    """The ClearVoice separation checkpoint; its commit resolves at construction."""
    return HFModel(path_or_uri="alibabasglab/MossFormer2_SS_16K", revision="main")


def _embedding_model() -> SpeechBrainModel:
    """The speaker-embedding model spec; its commit resolves at construction."""
    return SpeechBrainModel(path_or_uri=EMBEDDING_ID, revision="main")


def _target_refusal(target: TargetSpeakerEmbedding) -> str | None:
    """Why a target embedding cannot be compared with this node's probe, or ``None`` when it can.

    Args:
        target: The caller's target-speaker embedding.

    Returns:
        The refusal, in controlled vocabulary, or ``None`` when the target is comparable.
    """
    if target.provenance.model_commit_sha is None:
        return "target embedding carries no resolved model commit; refused rather than compared"
    if target.provenance.model_id != EMBEDDING_ID:
        return (
            f"target embedding model {target.provenance.model_id} is not the probe {EMBEDDING_ID}; "
            "embeddings from different models are not comparable"
        )
    return None


def _required(config: TriageConfig, hint: AudioHints | None) -> dict[str, Any]:
    """Resolve every ``require()`` key at entry, so an unmeasured key precedes any store write.

    Args:
        config: The triage configuration.
        hint: The caller's hint; a comparable target embedding additionally requires the match cut.

    Returns:
        The resolved values, keyed by their short names.
    """
    values = {
        "word_gap_ms": float(config.require("speech.word_gap_ms")),
        "coverage_threshold": float(config.require("yamnet.coverage_threshold")),
        "hint_tags": [str(tag) for tag in config.require("speech.hint_tags")],
        "clip_headroom": float(config.require("disruptions.clip_headroom")),
        "min_clip_run": int(config.require("disruptions.min_clip_run")),
        "min_dropout_ms": float(config.require("disruptions.min_dropout_ms")),
        "discontinuity_local_factor": float(config.require("disruptions.discontinuity_local_factor")),
        "discontinuity_window_ms": float(config.require("disruptions.discontinuity_window_ms")),
        "required_detectors": sorted(str(name) for name in config.require("pii.required_detectors")),
    }
    target = hint.target_speaker if hint is not None else None
    if target is not None and _target_refusal(target) is None:
        values["target_match_cosine"] = float(config.require("speech.target_match_cosine"))
    return values


def _hint_asserts_speech(hint: AudioHints | None, hint_tags: list[str]) -> bool:
    """Whether the caller asserted speech content (N25)."""
    if hint is None:
        return False
    if hint.expected_speech:
        return True
    return bool({tag.lower() for tag in hint.may_contain} & {tag.lower() for tag in hint_tags})


def _overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """Whether two extents share any temporal intersection > 0 (N10)."""
    return a[0] < b[1] and a[1] > b[0]


def _author_node(store: ProvStore, entity_id: str) -> str | None:
    """The node whose activity generated an entity, or None when nothing did."""
    activity_id = store.generated_by(entity_id)
    return store.get_activity(activity_id).node if activity_id else None


def _group_words_into_spans(words: list[dict[str, Any]], gap_ms: float) -> list[tuple[float, float, list[int]]]:
    """A span is the extent of a run of words; a gap over ``gap_ms`` starts a new run."""
    spans: list[tuple[float, float, list[int]]] = []
    order = sorted(range(len(words)), key=lambda i: float(words[i]["start"]))
    for i in order:
        word = words[i]
        if spans and (float(word["start"]) - spans[-1][1]) * 1000.0 <= gap_ms:
            start, end, members = spans[-1]
            spans[-1] = (start, max(end, float(word["end"])), [*members, i])
        else:
            spans.append((float(word["start"]), float(word["end"]), [i]))
    return spans


def _speech_coverage(windows: list[dict[str, Any]], extent: tuple[float, float], threshold: float) -> float:
    """The fraction of overlapping YAMNet windows whose ``Speech`` score clears the threshold (N5)."""
    overlapping = [w for w in windows if float(w["start"]) < extent[1] and float(w["end"]) > extent[0]]
    if not overlapping:
        return 0.0
    cleared = 0
    for window in overlapping:
        score = 0.0
        for pair in label_scores(window):
            if "Speech" in pair:
                score = float(pair["Speech"])
                break
        if score >= threshold:
            cleared += 1
    return cleared / len(overlapping)


def _norm_token(token: str) -> str:
    """A token normalised for subsequence matching: casefolded, edge punctuation stripped."""
    return token.casefold().strip(".,;:!?\"'()[]{}")


def _locate(finding_text: str, span_words: list[dict[str, Any]]) -> tuple[float, float] | None:
    """The extent of the finding's tokens as a contiguous subsequence of the span's words (N11)."""
    tokens = [_norm_token(t) for t in finding_text.split()]
    haystack = [_norm_token(str(w["text"])) for w in span_words]
    if not tokens or not haystack:
        return None
    for i in range(len(haystack) - len(tokens) + 1):
        if haystack[i : i + len(tokens)] == tokens:
            return float(span_words[i]["start"]), float(span_words[i + len(tokens) - 1]["end"])
    return None


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two flattened vectors."""
    return float(torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0))


def _failure_type(failure: str) -> str:
    """The exception type a detector failure leads with, projected away from its message.

    Args:
        failure: The detector's failure string, conventionally ``"<Type>: <message>"``.

    Returns:
        The leading type name, or a controlled placeholder when the string does not carry one.
    """
    head, separator, _ = failure.partition(":")
    return head if separator and head.isidentifier() else "type not recorded"


def _missing_detectors(required: list[str], scanned_by: set[str], failures: dict[str, str]) -> list[str]:
    """The required detectors that neither scanned nor recorded a failure, sorted.

    Args:
        required: The detector set ``pii.required_detectors`` names.
        scanned_by: The detectors that ran.
        failures: The detectors that were attempted and failed.

    Returns:
        The detector names that were never attempted, sorted.
    """
    return sorted(set(required) - scanned_by - set(failures))


def _decide_pii(
    findings: list[dict[str, Any]],
    failures: dict[str, str],
    missing: list[str],
    target_speaker: str | None,
) -> list[str]:
    """This branch's own rule over ``scan_for_pii``'s evidence — not ``decide_pii``'s.

    Flags when a finding overlaps the target speaker's words, when no target is known and anything
    was found, when any detector failed, or when a required detector was never attempted:
    could-not-check is not clean either way. A failure is projected to its detector and exception
    type; its message may quote the scanned input.
    """
    reasons: list[str] = []
    for detector in missing:
        reasons.append(f"required pii detector {detector} was not attempted; the scan could not check for it")
    for detector, failure in sorted(failures.items()):
        reasons.append(f"pii detector {detector} did not run ({_failure_type(failure)})")
    if findings and target_speaker is None:
        reasons.append("pii found and no target speaker is known; there is no speaker to exempt")
    if target_speaker is not None:
        for finding in findings:
            if not finding["resolved"]:
                reasons.append(
                    f"pii ({finding['category']}) whose speaker cannot be resolved is treated as the target's"
                )
            elif finding["speaker"] == target_speaker:
                reasons.append(f"pii ({finding['category']}) in the target speaker's speech")
    return reasons


def speech(  # noqa: C901 — the branch's eight steps, in design order
    store: ProvStore,
    source: str,
    config: TriageConfig,
    hint: AudioHints | None = None,
    *,
    run_dir: Path,
) -> NodeResult:
    """Run the SPEECH branch over the store PREPROCESS (and optionally AIRWAY) left behind.

    Args:
        store: The provenance store, holding PREPROCESS's words, envelope and YAMNet windows.
        source: The store-held stream name, ``"plain"``.
        config: The triage configuration.
        hint: What the recording was declared to contain; a target embedding enables the match.
        run_dir: The run directory sidecar paths are relative to.

    Returns:
        The verdict, and the view over every element this branch authored or asserted over.

    Raises:
        LookupError: If either the named stream or the original ``recording`` stream is absent.
    """
    values = _required(config, hint)
    software = software_agent(store)
    plain_id, plain = resolve_stream(store, run_dir, source)
    recording_id, recording = resolve_stream(store, run_dir, ORIGINAL)
    sr = int(plain.sampling_rate)
    view: list[str] = []
    flags: list[str] = []

    # Step 1 — transcript: fuse the two hypotheses; agreement is confidence, never correctness.
    raw_words: dict[str, list[dict[str, Any]]] = {CRISPERWHISPER_ID: [], QWEN_ID: []}
    source_word_ids: list[str] = []
    for entity in store.entities("word"):
        recognizer = entity.attributes.get("recognizer")
        if recognizer not in raw_words or store.is_invalidated(entity.id):
            continue
        start, end = entity.extent or (0.0, 0.0)
        word: dict[str, Any] = {"text": str(entity.attributes.get("text") or ""), "start": start, "end": end}
        if entity.attributes.get("score") is not None:
            word["confidence"] = float(entity.attributes["score"])
        for field in ("timestamp_source", "timestamp_model"):
            if entity.attributes.get(field):
                word[field] = str(entity.attributes[field])
        raw_words[recognizer].append(word)
        source_word_ids.append(entity.id)

    transcript = store.activity(node=NODE, step="transcript", parameters={"systems": [CRISPERWHISPER_ID, QWEN_ID]})
    store.was_associated_with(transcript, software)
    store.used(transcript, plain_id)
    for word_id in source_word_ids:
        store.used(transcript, word_id)

    hint_asserts = _hint_asserts_speech(hint, values["hint_tags"])
    if not any(raw_words.values()):
        why = "no words from either recognizer; this branch has no subject"
        outcome = Outcome.FAIL
        if hint_asserts:
            outcome = Outcome.FLAG
            why += "; a hint asserts speech not found"
            flags.append(why)
        empty_pii = store.activity(node=NODE, step="pii", parameters={"systems": [CRISPERWHISPER_ID, QWEN_ID]})
        store.was_associated_with(empty_pii, software)
        empty_scan_id = store.entity(
            prov_type="measurement",
            extent=None,
            attributes={
                "name": "pii_scan",
                "scanned_by": [],
                "failed": [],
                "missing": list(values["required_detectors"]),
            },
        )
        store.was_generated_by(empty_scan_id, empty_pii)
        store.was_attributed_to(empty_scan_id, software)
        verdict_id, verdict = write_verdict(
            store,
            empty_pii,
            software,
            node=NODE,
            outcome=outcome,
            kind="speech",
            why=why,
            detail={
                "speaker_count": 0,
                "words_n": 0,
                "speech_s": 0.0,
                "pii": {
                    "categories": [],
                    "n": 0,
                    "scanned_by": [],
                    "failed": [],
                    "missing": list(values["required_detectors"]),
                },
                "flags": flags,
                "second_diarizer": "not_consulted",
                "agreement_flag": "not_evaluated",
            },
        )
        return NodeResult(verdict=verdict, view=(empty_scan_id, verdict_id), verdict_entity_id=verdict_id)

    fused = fuse_word_streams({rid: words for rid, words in raw_words.items() if words})

    envelope = find_measurement(store, "energy_envelope")
    fabrication: list[int] = []
    if envelope is not None:
        store.used(transcript, envelope.id)
        sidecar = np.load(run_dir / envelope.attributes["path"])
        env, floor = sidecar["envelope_dbfs"], sidecar["floor_dbfs"]
        env_sr = int(envelope.attributes["sampling_rate"])
        for index, word in enumerate(fused):
            lo, hi = int(float(word["start"]) * env_sr), int(float(word["end"]) * env_sr)
            segment_env, segment_floor = env[lo:hi], floor[lo:hi]
            if segment_env.size and not np.any(segment_env > segment_floor):
                fabrication.append(index)
    if fabrication:
        flags.append(f"{len(fabrication)} fabrication candidate(s) survive the energy test")

    agreement_floor = config.get("speech.agreement_flag_floor")
    if agreement_floor is None:
        agreement_flag: Any = "not_evaluated"
    else:
        confidences = [float(w["confidence"]) for w in fused if w.get("confidence") is not None]
        agreement = sum(confidences) / len(confidences) if confidences else 0.0
        agreement_flag = {"agreement": agreement, "floor": float(agreement_floor)}
        if agreement < float(agreement_floor):
            flags.append(f"recognizer agreement {agreement:.2f} below {float(agreement_floor):.2f}")

    # Step 2 — speech spans from word timings, in memory until corroborated.
    grouped = _group_words_into_spans(fused, values["word_gap_ms"])
    speech_s = sum(end - start for start, end, _ in grouped)

    # Step 3 — corroborate: YAMNet Speech coverage from the stored windows; SQUIM as the speech test.
    prior_spans: list[Entity] = [
        e for e in store.entities("span") if not store.is_invalidated(e.id) and _author_node(store, e.id) != NODE
    ]
    yamnet = find_measurement(store, "yamnet_windows")
    windows: list[dict[str, Any]] = []
    if yamnet is not None:
        windows = json.loads((run_dir / yamnet.attributes["path"]).read_text())
    corroborate = store.activity(
        node=NODE,
        step="corroborate",
        parameters={"word_gap_ms": values["word_gap_ms"], "coverage_threshold": values["coverage_threshold"]},
    )
    store.was_associated_with(corroborate, software)
    if yamnet is not None:
        store.used(corroborate, yamnet.id)
    for prior in prior_spans:
        store.used(corroborate, prior.id)

    stoi_floor = config.get("speech.speech_test_stoi_floor")
    si_sdr_floor = config.get("speech.speech_test_si_sdr_floor")
    squim_by_span: list[dict[str, Any]] = []
    span_ids: list[str] = []
    span_extents: list[tuple[float, float]] = []
    for raw_start, raw_end, members in grouped:
        start, end = clamp_extent((raw_start, raw_end), plain)
        clip = Audio(waveform=plain.waveform[:, int(start * sr) : int(end * sr)], sampling_rate=sr)
        squim: dict[str, Any]
        try:
            [scores] = extract_objective_quality_features_from_audios([clip])
            squim = {"stoi": float(scores["stoi"]), "pesq": float(scores["pesq"]), "si_sdr": float(scores["si_sdr"])}
        except Exception as err:  # noqa: BLE001 — a span SQUIM refuses is unmeasured, not padded
            squim = {"unmeasured": type(err).__name__}
        squim_by_span.append(squim)

        coverage = _speech_coverage(windows, (start, end), values["coverage_threshold"])
        yamnet_vote = "confirm" if coverage >= values["coverage_threshold"] else "disconfirm"
        if yamnet_vote == "disconfirm":
            flags.append(f"yamnet disconfirms span {start:.2f}-{end:.2f}s (Speech coverage {coverage:.2f})")
        if stoi_floor is None or si_sdr_floor is None or "unmeasured" in squim:
            squim_vote = "not_evaluated"
        else:
            squim_ok = squim["stoi"] >= float(stoi_floor) and squim["si_sdr"] >= float(si_sdr_floor)
            squim_vote = "confirm" if squim_ok else "disconfirm"
            if squim_vote != yamnet_vote:
                flags.append(
                    f"instruments disagree on span {start:.2f}-{end:.2f}s: yamnet {yamnet_vote} "
                    f"(coverage {coverage:.2f}), squim {squim_vote} (stoi {squim['stoi']:.2f})"
                )
        span_id = store.entity(
            prov_type="span",
            extent=(start, end),
            attributes={
                "words_n": len(members),
                "yamnet_coverage": coverage,
                "yamnet_vote": yamnet_vote,
                "squim_vote": squim_vote,
            },
        )
        store.was_generated_by(span_id, corroborate)
        store.was_attributed_to(span_id, software)
        for prior in prior_spans:
            if prior.extent is not None and _overlaps((start, end), prior.extent):
                store.was_derived_from(span_id, prior.id)
        span_ids.append(span_id)
        span_extents.append((start, end))
        view.append(span_id)

    # Step 4 — diarize pyannote over [first word start, last word end] only; every segment counts.
    interval = clamp_extent((min(float(w["start"]) for w in fused), max(float(w["end"]) for w in fused)), plain)
    diarizer = _diarization_model()
    diarize_act = store.activity(
        node=NODE, step="diarize", parameters={"interval": list(interval), "model": str(diarizer.path_or_uri)}
    )
    diarizer_agent = store.agent(agent_type="model", model_id=str(diarizer.path_or_uri), commit_sha=diarizer.commit_sha)
    store.was_associated_with(diarize_act, diarizer_agent)
    store.used(diarize_act, plain_id)
    interval_id = store.entity(prov_type="interval", extent=interval, attributes={"name": "diarization_interval"})
    store.was_generated_by(interval_id, diarize_act)
    store.was_attributed_to(interval_id, software)
    view.append(interval_id)

    (cropped,) = extract_segments([(plain, [interval])])[0]
    [segments] = diarize_audios([cropped], model=diarizer)
    shifted = [
        ScriptLine(speaker=s.speaker, start=(s.start or 0.0) + interval[0], end=(s.end or 0.0) + interval[0])
        for s in segments
    ]

    speaker_segments: list[tuple[str, str, tuple[float, float]]] = []  # (entity_id, speaker, extent)
    for seg_line in shifted:
        extent = (float(seg_line.start or 0.0), float(seg_line.end or 0.0))
        speaker_id = store.entity(
            prov_type="speaker",
            extent=extent,
            attributes={"speaker": seg_line.speaker, "diarizer": str(diarizer.path_or_uri)},
        )
        store.was_generated_by(speaker_id, diarize_act)
        store.was_attributed_to(speaker_id, diarizer_agent)
        view.append(speaker_id)
        speaker_segments.append((speaker_id, str(seg_line.speaker), extent))

    count = len({speaker for _, speaker, _ in speaker_segments})
    second = config.get("speech.second_diarizer")
    second_record: Any = "not_consulted"
    if count != 1:
        flags.append(f"speaker count {count} != 1")
        if second is not None:
            second_model = _second_diarizer_model(str(second))
            second_agent = store.agent(
                agent_type="model", model_id=str(second_model.path_or_uri), commit_sha=second_model.commit_sha
            )
            second_act = store.activity(
                node=NODE, step="second_diarizer", parameters={"model": str(second_model.path_or_uri)}
            )
            store.was_associated_with(second_act, second_agent)
            store.used(second_act, interval_id)
            [second_segments] = diarize_audios([cropped], model=second_model)
            second_count = len({s.speaker for s in second_segments})
            second_record = {"model": str(second), "count": second_count, "agrees": second_count == count}
            if second_count != count:
                flags.append(f"second diarizer counts {second_count} speakers against {count}")

    # Step 5 — separate only when the count is exactly the checkpoint's 2; >= 3 is reported instead.
    stream_ids: list[str] = []
    if count >= 3:
        flags.append(f"separation cannot serve {count} speakers; the checkpoint separates exactly 2")
    elif count == 2:
        separator = _separation_model()
        separation_agent = store.agent(
            agent_type="model", model_id=str(separator.path_or_uri), commit_sha=separator.commit_sha
        )
        separate_act = store.activity(
            node=NODE, step="separate", parameters={"n_sources": 2, "interval": list(interval)}
        )
        store.was_associated_with(separate_act, separation_agent)
        store.used(separate_act, plain_id)
        store.used(separate_act, interval_id)
        [separated] = separate_audios([cropped], model=separator, n_sources=2)
        for source_audio in separated:
            meta = dict(source_audio.metadata.get("clearvoice") or {})
            index = int(meta.get("source_index", len(stream_ids)))
            path = f"streams/separated_{index}.wav"
            source_audio.save_to_file(str(run_dir / path))
            stream_id = store.entity(
                prov_type="stream",
                extent=interval,
                attributes={
                    "name": f"separated_{index}",
                    "path": path,
                    "sampling_rate": int(source_audio.sampling_rate),
                    "channels": 1,
                    "source_index": index,
                    "input_norm_scalar": meta.get("input_norm_scalar"),
                    "separation_model": meta.get("model"),
                    "separation_commit": meta.get("commit"),
                },
            )
            store.was_generated_by(stream_id, separate_act)
            store.was_attributed_to(stream_id, separation_agent)
            store.was_derived_from(stream_id, plain_id)
            stream_ids.append(stream_id)
            view.append(stream_id)

    # Step 6 — identify: words to speakers by timing; straddlers are marked, not assigned.
    identify = store.activity(node=NODE, step="identify", parameters={})
    store.was_associated_with(identify, software)
    assignments: list[tuple[str | None, str | None]] = []
    word_ids: list[str] = []
    for word in fused:
        extent = (float(word["start"]), float(word["end"]))
        overlapping = [entry for entry in speaker_segments if _overlaps(extent, entry[2])]
        speaker: str | None
        note: str | None
        if len(overlapping) > 1:
            speaker, note = None, "straddles"
        elif len(overlapping) == 1:
            speaker, note = overlapping[0][1], None
        else:
            speaker, note = None, "unassigned"
        assignments.append((speaker, note))
        attributes: dict[str, Any] = {
            "text": word["text"],
            "confidence": word.get("confidence"),
            "existence_confidence": word.get("existence_confidence"),
            "temporal_confidence": word.get("temporal_confidence"),
            "coverage": word.get("coverage"),
            "speaker": speaker,
            "stream": plain_id,
        }
        if note is not None:
            attributes["speaker_note"] = note
        word_id = store.entity(prov_type="word", extent=extent, attributes=attributes)
        store.was_generated_by(word_id, identify)
        store.was_attributed_to(word_id, software)
        word_ids.append(word_id)
        view.append(word_id)

    for index in fabrication:
        label_id = store.entity(
            prov_type="assertion",
            extent=(float(fused[index]["start"]), float(fused[index]["end"])),
            attributes={"verb": "label", "label": "fabrication_candidate", "periodicity": "not_evaluated"},
        )
        store.was_generated_by(label_id, transcript)
        store.was_attributed_to(label_id, software)
        store.was_derived_from(label_id, word_ids[index])
        view.append(label_id)

    target_speaker: str | None = None
    target = hint.target_speaker if hint is not None else None
    if target is not None:
        refusal = _target_refusal(target)
        if refusal is not None:
            flags.append(refusal)
        elif speaker_segments:
            probe = _embedding_model()
            labels = sorted({speaker for _, speaker, _ in speaker_segments})
            audios: list[Audio] = []
            for label in labels:
                slices = []
                for _, speaker, extent in speaker_segments:
                    if speaker != label:
                        continue
                    s, e = clamp_extent(extent, plain)
                    slices.append(plain.waveform[:, int(s * sr) : int(e * sr)])
                audios.append(Audio(waveform=torch.cat(slices, dim=1), sampling_rate=sr))
            embedding_agent = store.agent(
                agent_type="model", model_id=str(probe.path_or_uri), commit_sha=probe.commit_sha
            )
            store.was_associated_with(identify, embedding_agent)
            embeddings = extract_speaker_embeddings_from_audios(audios, model=probe)
            target_vector = torch.tensor(target.vector, dtype=torch.float32)
            cut = float(values["target_match_cosine"])
            best: tuple[float, str] | None = None
            for label, embedding in zip(labels, embeddings):
                similarity = _cosine(embedding, target_vector)
                match_id = store.entity(
                    prov_type="target_match",
                    extent=None,
                    attributes={
                        "speaker": label,
                        "similarity": similarity,
                        "threshold": cut,
                        "target_model": target.provenance.model_id,
                        "target_commit": target.provenance.model_commit_sha,
                        "probe_model": str(probe.path_or_uri),
                        "probe_commit": probe.commit_sha,
                        "stream": plain_id,
                    },
                )
                store.was_generated_by(match_id, identify)
                store.was_attributed_to(match_id, embedding_agent)
                view.append(match_id)
                if best is None or similarity > best[0]:
                    best = (similarity, label)
            if best is not None and best[0] >= cut:
                target_speaker = best[1]
            else:
                flags.append("a target was given and no speaker matches it")

    # Step 7 — PII: scan both hypotheses per span; the decision is speaker-scoped and this branch's own.
    pii_act = store.activity(node=NODE, step="pii", parameters={"systems": [CRISPERWHISPER_ID, QWEN_ID]})
    store.was_associated_with(pii_act, software)
    for span_id in span_ids:
        store.used(pii_act, span_id)
    lines: list[ScriptLine] = []
    line_meta: list[tuple[int, str, list[dict[str, Any]]]] = []
    for span_index, extent in enumerate(span_extents):
        for recognizer, words in raw_words.items():
            span_words = sorted(
                (w for w in words if _overlaps((float(w["start"]), float(w["end"])), extent)),
                key=lambda w: float(w["start"]),
            )
            if not span_words:
                continue
            chunks = [ScriptLine(text=str(w["text"]), start=float(w["start"]), end=float(w["end"])) for w in span_words]
            lines.append(ScriptLine(text="", start=extent[0], end=extent[1], chunks=chunks))
            line_meta.append((span_index, recognizer, span_words))

    raw_scans = scan_for_pii(lines) if lines else []
    scans: list[PiiScan] = raw_scans if isinstance(raw_scans, list) else [raw_scans]
    failures: dict[str, str] = {}
    scanned_by: set[str] = set()
    findings: list[dict[str, Any]] = []
    pii_ids: list[str] = []
    for (span_index, recognizer, span_words), scan in zip(line_meta, scans):
        failures.update(scan.failures)
        scanned_by.update(scan.detectors_used)
        for finding in scan.spans:
            located = _locate(str(finding.text or ""), span_words)
            extent = located if located is not None else span_extents[span_index]
            if located is None:
                flags.append(f"pii_unlocated ({finding.category})")
            speakers = {
                assignments[i][0]
                for i, word in enumerate(fused)
                if _overlaps((float(word["start"]), float(word["end"])), extent)
            }
            resolved = len(speakers) == 1 and None not in speakers
            speaker = next(iter(speakers)) if resolved else None
            pii_id = store.entity(
                prov_type="pii",
                extent=extent,
                attributes={
                    "category": finding.category,
                    "source": finding.source,
                    "asr_model": recognizer,
                    "detectors_used": sorted(scan.detectors_used),
                    "detectors_failed": sorted(scan.failures),
                },
            )
            store.was_generated_by(pii_id, pii_act)
            store.was_attributed_to(pii_id, software)
            store.was_derived_from(pii_id, span_ids[span_index])
            pii_ids.append(pii_id)
            view.append(pii_id)
            for i, word in enumerate(fused):
                if _overlaps((float(word["start"]), float(word["end"])), extent):
                    mark_id = store.entity(
                        prov_type="assertion",
                        extent=(float(word["start"]), float(word["end"])),
                        attributes={"verb": "label", "label": "pii", "category": finding.category},
                    )
                    store.was_generated_by(mark_id, pii_act)
                    store.was_attributed_to(mark_id, software)
                    store.was_derived_from(mark_id, word_ids[i])
                    view.append(mark_id)
            findings.append({"category": finding.category, "speaker": speaker, "resolved": resolved})
    missing = _missing_detectors(values["required_detectors"], scanned_by, failures)
    flags.extend(_decide_pii(findings, failures, missing, target_speaker))
    scan_id = store.entity(
        prov_type="measurement",
        extent=None,
        attributes={
            "name": "pii_scan",
            "scanned_by": sorted(scanned_by),
            "failed": sorted(failures),
            "missing": missing,
        },
    )
    store.was_generated_by(scan_id, pii_act)
    store.was_attributed_to(scan_id, software)
    view.append(scan_id)

    # Step 8 — quality: SQUIM on plain, disruptions on the original recording; reported, never gating.
    quality = store.activity(
        node=NODE,
        step="quality",
        parameters={
            "clip_headroom": values["clip_headroom"],
            "min_clip_run": values["min_clip_run"],
            "min_dropout_ms": values["min_dropout_ms"],
            "discontinuity_local_factor": values["discontinuity_local_factor"],
            "discontinuity_window_ms": values["discontinuity_window_ms"],
        },
    )
    store.was_associated_with(quality, software)
    store.used(quality, plain_id)
    store.used(quality, recording_id)
    for span_id, extent, squim in zip(span_ids, span_extents, squim_by_span):
        squim_id = store.entity(
            prov_type="measurement",
            extent=extent,
            attributes={"name": "squim", "stream": plain_id, **squim},
        )
        store.was_generated_by(squim_id, quality)
        store.was_derived_from(squim_id, span_id)
        view.append(squim_id)
        original_extent = clamp_extent(extent, recording)
        disruptions = detect_disruptions(
            recording,
            original_extent[0],
            original_extent[1],
            clip_headroom=values["clip_headroom"],
            min_clip_run=values["min_clip_run"],
            min_dropout_ms=values["min_dropout_ms"],
            discontinuity_local_factor=values["discontinuity_local_factor"],
            discontinuity_window_ms=values["discontinuity_window_ms"],
        )
        counts = {k: v for k, v in asdict(disruptions).items() if k not in ("start", "end")}
        disruption_id = store.entity(
            prov_type="measurement",
            extent=extent,
            attributes={"name": "disruptions", "stream": recording_id, **counts},
        )
        store.was_generated_by(disruption_id, quality)
        store.was_derived_from(disruption_id, span_id)
        view.append(disruption_id)

    # Outcome — fail only from the no-words row above; flag from the accumulated reasons; else pass.
    categories = sorted({str(f["category"]) for f in findings})
    detail: dict[str, Any] = {
        "speaker_count": count,
        "words_n": len(fused),
        "speech_s": speech_s,
        "pii": {
            "categories": categories,
            "n": len(findings),
            "scanned_by": sorted(scanned_by),
            "failed": sorted(failures),
            "missing": missing,
        },
        "flags": flags,
        "second_diarizer": second_record,
        "agreement_flag": agreement_flag,
    }
    if target_speaker is not None:
        detail["target_speaker"] = target_speaker
    if flags:
        outcome, why = Outcome.FLAG, "; ".join(flags)
    else:
        outcome = Outcome.PASS
        why = "words, spans, speakers and quality are in the store"
    verdict_id, verdict = write_verdict(
        store, quality, software, node=NODE, outcome=outcome, kind="speech", why=why, detail=detail
    )
    view.append(verdict_id)
    return NodeResult(verdict=verdict, view=tuple(view), verdict_entity_id=verdict_id)
