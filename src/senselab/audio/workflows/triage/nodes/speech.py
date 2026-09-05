"""The SPEECH branch: the consensus transcript, diarization, an enrolled target, PII, quality.

It runs no ASR and never re-transcribes: PREPROCESS produced the consensus with
``fuse_consensus_words`` and this branch reads it. Speech spans come from consensus word timings,
never the envelope, and pyannote sees only ``[first word start, last word end]``. The second
diarizer runs only when pyannote's count is not 1; separation runs only when
``speech.separation_backend`` names a backend. The target speaker is identified by a caller-supplied
enrollment, not by a per-file hint, and an enrollment is refused rather than compared unless its
model and its resolved commit are both the probe's. The PII scan reads the consensus transcript and
nothing else, once, and marks every occurrence of what it finds. This branch marks; it removes
nothing.

Every parameter's derivation is in ``data/config/default.yaml``; the design is in
``specs/20260817-triage-workflow-dag/branch-speech.md``.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from senselab.audio.data_structures import Audio, AudioHints
from senselab.audio.tasks.disruptions.api import detect_disruptions
from senselab.audio.tasks.features_extraction.torchaudio_squim import (
    extract_objective_quality_features_from_audios,
)
from senselab.audio.tasks.preprocessing.preprocessing import extract_segments
from senselab.audio.tasks.source_separation.api import separate_audios
from senselab.audio.tasks.spans.api import group_extents_into_runs
from senselab.audio.tasks.speaker_diarization.api import diarize_audios
from senselab.audio.tasks.speaker_embeddings.api import extract_speaker_embeddings_from_audios
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.enrollment import Enrollment
from senselab.audio.workflows.triage.nodes.common import (
    NodeResult,
    clamp_extent,
    find_measurement,
    resolve_stream,
    software_agent,
    write_verdict,
)
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.text.tasks.pii_detection.api import PiiScan, scan_for_pii
from senselab.utils.data_structures import HFModel, PyannoteAudioModel, SpeechBrainModel
from senselab.utils.prov_store import Entity, ProvStore

NODE = "SPEECH"
ORIGINAL = "recording"  # the stream disruptions are measured on: as captured, unnormalised, unresampled
DIARIZER_ID = "pyannote/speaker-diarization-community-1"
CLEARVOICE_ORG = "alibabasglab"
UNASDIFF_BACKEND = "unasdiff"
SEPARABLE_SOURCES = 2
NONTARGET_LEGS = ("level_db", "tilt_db_per_octave", "d_to_r_db")


def _diarization_model() -> PyannoteAudioModel:
    """The diarizer's model spec; its commit resolves at construction.

    Returns:
        The model spec.
    """
    return PyannoteAudioModel(path_or_uri=DIARIZER_ID, revision="main")


def _second_diarizer_model(model_id: str) -> HFModel:
    """The configured second diarizer's model spec; its commit resolves at construction.

    Args:
        model_id: From ``speech.second_diarizer``.

    Returns:
        The model spec.
    """
    return HFModel(path_or_uri=model_id, revision="main")


def _clearvoice_model(model_id: str) -> HFModel:
    """A ClearVoice separation checkpoint's model spec; its commit resolves at construction.

    Args:
        model_id: The fully qualified checkpoint id.

    Returns:
        The model spec.
    """
    return HFModel(path_or_uri=model_id, revision="main")


def _embedding_model(model_id: str, revision: str) -> SpeechBrainModel:
    """The probe's model spec, at the commit the enrollment was estimated with.

    Args:
        model_id: From ``speech.enrollment_model.model_id``.
        revision: The resolved 40-hex commit from ``speech.enrollment_model.revision``.

    Returns:
        The model spec.
    """
    return SpeechBrainModel(path_or_uri=model_id, revision=revision)


def _required(config: TriageConfig, enrollment: Enrollment | None) -> dict[str, Any]:
    """Resolve every ``require()`` key at entry, so an unmeasured key precedes any measurement.

    Args:
        config: The triage configuration.
        enrollment: The caller's enrollment; one additionally requires the probe and the match cut.

    Returns:
        The resolved values, keyed by their short names.
    """
    values: dict[str, Any] = {
        "coverage_threshold": float(config.require("yamnet.coverage_threshold")),
        "clip_headroom": float(config.require("disruptions.clip_headroom")),
        "min_clip_run": int(config.require("disruptions.min_clip_run")),
        "min_dropout_ms": float(config.require("disruptions.min_dropout_ms")),
        "discontinuity_local_factor": float(config.require("disruptions.discontinuity_local_factor")),
        "discontinuity_window_ms": float(config.require("disruptions.discontinuity_window_ms")),
        "required_detectors": sorted(str(name) for name in config.require("pii.required_detectors")),
    }
    if enrollment is not None:
        model = config.require("speech.enrollment_model")
        values["enrollment_model_id"] = str(model["model_id"])
        values["enrollment_revision"] = str(model["revision"])
        values["target_match_cosine"] = float(config.require("speech.target_match_cosine"))
    return values


def _overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """Whether two extents share any temporal intersection > 0 (N10).

    Args:
        a: One extent.
        b: The other.

    Returns:
        True when they intersect.
    """
    return a[0] < b[1] and a[1] > b[0]


def _author_node(store: ProvStore, entity_id: str) -> str | None:
    """The node whose activity generated an entity, or None when nothing did.

    Args:
        store: The provenance store.
        entity_id: The entity.

    Returns:
        The node's name, or None.
    """
    activity_id = store.generated_by(entity_id)
    return store.get_activity(activity_id).node if activity_id else None


def _speech_coverage(windows: list[Entity], extent: tuple[float, float], family: set[str]) -> float | None:
    """The fraction of overlapping classifier windows whose label set meets the speech family (V3).

    Args:
        windows: The live ``yamnet_window`` measurements, each carrying the label set the
            threshold fold retained.
        extent: The span.
        family: The AudioSet speech family from ``taxonomy.speech_labels``.

    Returns:
        The fraction, or None when no window overlaps the span.
    """
    overlapping = [w for w in windows if w.extent is not None and _overlaps(extent, w.extent)]
    if not overlapping:
        return None
    carried = sum(
        1 for window in overlapping if family & {str(label) for label in (window.attributes.get("labels") or [])}
    )
    return carried / len(overlapping)


def _norm_token(token: str) -> str:
    """A token normalised for subsequence matching: casefolded, edge punctuation stripped.

    Args:
        token: The raw token.

    Returns:
        The normalised token.
    """
    return token.casefold().strip(".,;:!?\"'()[]{}")


def _locate(finding_text: str, words: list[Entity]) -> list[tuple[int, int]]:
    """Every place the finding's tokens match the consensus words, as contiguous runs (N11).

    Every occurrence, not the first: the scan dedupes by ``(category, text, source)``, so a name
    said twice arrives as one finding, and locating only its first match leaves the second
    occurrence unmarked and therefore unredacted.

    Args:
        finding_text: The detector's matched text.
        words: The consensus word entities, in transcript order.

    Returns:
        ``[(first index, last index), ...]``, non-overlapping and in transcript order. Empty when
        nothing matches.
    """
    tokens = [_norm_token(token) for token in finding_text.split()]
    haystack = [_norm_token(str(word.attributes.get("text") or "")) for word in words]
    if not tokens or not haystack or len(tokens) > len(haystack):
        return []
    matches: list[tuple[int, int]] = []
    start = 0
    while start <= len(haystack) - len(tokens):
        if haystack[start : start + len(tokens)] == tokens:
            matches.append((start, start + len(tokens) - 1))
            start += len(tokens)
        else:
            start += 1
    return matches


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two flattened vectors.

    Args:
        a: One vector.
        b: The other.

    Returns:
        The similarity.
    """
    return float(torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0))


def _dbfs(value: float) -> float:
    """A linear amplitude in dBFS, floored so silence is finite.

    Args:
        value: The amplitude.

    Returns:
        The level in dBFS.
    """
    return float(20.0 * np.log10(max(float(value), 1e-12)))


def _spectral_tilt(segment: np.ndarray, sampling_rate: int) -> float:
    """The least-squares slope of the log-magnitude spectrum against ``log2(frequency)``.

    Args:
        segment: The span's samples, mono.
        sampling_rate: The stream's rate.

    Returns:
        The slope in dB per octave; 0.0 when the span carries too few bins to fit a line.
    """
    if segment.size < 4:
        return 0.0
    magnitude = np.abs(np.fft.rfft(segment))
    frequencies = np.fft.rfftfreq(segment.size, d=1.0 / sampling_rate)
    keep = frequencies > 0.0
    if int(keep.sum()) < 2:
        return 0.0
    octaves = np.log2(frequencies[keep])
    levels = 20.0 * np.log10(np.maximum(magnitude[keep], 1e-12))
    slope, _ = np.polyfit(octaves, levels, 1)
    return float(slope)


def _direct_to_reverberant(segment: np.ndarray) -> float:
    """The span's direct-to-reverberant energy ratio, over its own autocorrelation.

    Args:
        segment: The span's samples, mono.

    Returns:
        The ratio in dB, taking the peak lag as direct and the tail as reverberant.
    """
    if segment.size < 2:
        return 0.0
    autocorrelation = np.correlate(segment, segment, mode="full")[segment.size - 1 :]
    energies = np.square(autocorrelation)
    peak = float(energies.max())
    tail = float(energies.sum()) - peak
    return float(10.0 * np.log10(max(peak, 1e-20) / max(tail, 1e-20)))


def _proximity(segment: np.ndarray, sampling_rate: int, reference_rms_dbfs: float | None) -> dict[str, Any]:
    """The proximity leg's measures over one span, against the file's own reference level.

    Args:
        segment: The span's samples, mono.
        sampling_rate: The stream's rate.
        reference_rms_dbfs: The file's RMS from PREPROCESS's ``level`` measurement, or None when
            PREPROCESS wrote none.

    Returns:
        ``{rms_dbfs, peak_dbfs, level_over_reference_db, tilt_db_per_octave, d_to_r_db}``.
    """
    rms_dbfs = _dbfs(float(np.sqrt(np.mean(np.square(segment)))) if segment.size else 0.0)
    peak_dbfs = _dbfs(float(np.abs(segment).max()) if segment.size else 0.0)
    return {
        "rms_dbfs": rms_dbfs,
        "peak_dbfs": peak_dbfs,
        "level_over_reference_db": None if reference_rms_dbfs is None else rms_dbfs - reference_rms_dbfs,
        "tilt_db_per_octave": _spectral_tilt(segment, sampling_rate),
        "d_to_r_db": _direct_to_reverberant(segment),
    }


def _behind_the_target(measure: dict[str, Any], cuts: dict[str, float]) -> bool:
    """Whether all three proximity legs place a span away from the target.

    Args:
        measure: The span's proximity reading.
        cuts: The three ``speech.nontarget`` thresholds, all of them supplied.

    Returns:
        True when every leg falls on the far side of its cut.
    """
    level = measure["level_over_reference_db"]
    return (
        level is not None
        and level <= cuts["level_db"]
        and measure["tilt_db_per_octave"] <= cuts["tilt_db_per_octave"]
        and measure["d_to_r_db"] <= cuts["d_to_r_db"]
    )


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

    Args:
        findings: One record per finding, carrying its category and its resolved speaker.
        failures: The detectors that were attempted and failed.
        missing: The required detectors that were never attempted.
        target_speaker: The diarized speaker the enrollment matched, or None.

    Returns:
        The reasons this branch flags, in controlled vocabulary.
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


def _flag_before_measuring(store: ProvStore, why: str) -> NodeResult:
    """Flag on a caller input this branch cannot act on, before any measurement is taken.

    Args:
        store: The provenance store.
        why: The refusal, in controlled vocabulary.

    Returns:
        The flag verdict and a view over it alone.
    """
    software = software_agent(store)
    activity = store.activity(node=NODE, step="enrollment", parameters={})
    store.was_associated_with(activity, software)
    verdict_id, verdict = write_verdict(
        store,
        activity,
        software,
        node=NODE,
        outcome=Outcome.FLAG,
        kind="speech",
        why=why,
        detail={"flags": [why]},
    )
    return NodeResult(verdict=verdict, view=(verdict_id,), verdict_entity_id=verdict_id)


def speech(  # noqa: C901 — the branch's nine steps, in design order
    store: ProvStore,
    source: str,
    config: TriageConfig,
    hint: AudioHints | None = None,
    *,
    run_dir: Path,
    enrollment: Optional[Enrollment] = None,
) -> NodeResult:
    """Run the SPEECH branch over the store PREPROCESS left behind.

    Args:
        store: The provenance store, holding the consensus transcript, the envelope and the
            classifier windows.
        source: The store-held stream name, ``"plain"``.
        config: The triage configuration.
        hint: What the recording was declared to contain. Neither ``target_speaker`` nor
            ``targeted_speaker_count`` is read as evidence.
        run_dir: The run directory sidecar paths are relative to.
        enrollment: The target speaker's enrollment, estimated across the subject's recordings.

    Returns:
        The verdict, and the view over every element this branch authored or asserted over.

    Raises:
        LookupError: If a stream this branch needs, or the consensus transcript, is absent.
        ValueError: If a key this branch requires has no value and no enrollment was supplied.
    """
    try:
        values = _required(config, enrollment)
    except ValueError as error:
        if enrollment is None:
            raise
        return _flag_before_measuring(store, f"{error}")

    software = software_agent(store)
    plain_id, plain = resolve_stream(store, run_dir, source)
    recording_id, recording = resolve_stream(store, run_dir, ORIGINAL)
    sampling_rate = int(plain.sampling_rate)
    view: list[str] = []
    flags: list[str] = []

    # Step 1 — the consensus transcript is the transcript; this branch reads it and re-fuses nothing.
    consensus = find_measurement(store, "consensus_transcript")
    if consensus is None:
        raise LookupError("no consensus_transcript in the store; PREPROCESS has not run")
    words = [store.get_entity(word_id) for word_id in consensus.attributes["word_ids"]]
    words = [word for word in words if not store.is_invalidated(word.id)]
    transcript_text = str(consensus.attributes["text"])

    transcript = store.activity(node=NODE, step="transcript", parameters={"read": "consensus_transcript"})
    store.was_associated_with(transcript, software)
    store.used(transcript, plain_id)
    store.used(transcript, consensus.id)
    for word in words:
        store.used(transcript, word.id)

    if hint is not None and hint.target_speaker is not None:
        flags.append(
            "this branch identifies the target by enrollment, not by hint.target_speaker, "
            "which was supplied and is not read"
        )

    if not words:
        why = "no consensus word; this branch has no subject"
        outcome = Outcome.FAIL
        verdict_id, verdict = write_verdict(
            store,
            transcript,
            software,
            node=NODE,
            outcome=outcome,
            kind="speech",
            why=why,
            detail={
                "speaker_count": None,
                "diarization": "no_words",
                "words_n": 0,
                "speech_s": 0.0,
                "nontarget_speech_s": None,
                "pii": {"categories": [], "n": 0, "scanned_by": [], "failed": [], "missing": []},
                "second_diarizer": "not_consulted",
                "separation": "no_speaker_count",
                "flags": flags,
            },
        )
        return NodeResult(verdict=verdict, view=(verdict_id,), verdict_entity_id=verdict_id)

    single_source = [word.id for word in words if len(word.attributes.get("recognizers") or []) == 1]
    if single_source:
        flags.append(f"{len(single_source)} single-recognizer word(s) survive as fabrication candidates")

    # Step 2 — speech spans from consensus word timings, in memory until corroborated.
    word_extents = [word.extent or (0.0, 0.0) for word in words]
    grouped = group_extents_into_runs(word_extents)
    span_extents = [clamp_extent((start, end), plain) for start, end, _ in grouped]
    speech_s = sum(end - start for start, end in span_extents)

    # Step 3 — corroborate: the classifier's retained Speech label set, and SQUIM as the speech test.
    prior_spans: list[Entity] = [
        e for e in store.entities("span") if not store.is_invalidated(e.id) and _author_node(store, e.id) != NODE
    ]
    fold = find_measurement(store, "yamnet_windows")
    classifier_windows: list[Entity] = []
    if fold is not None:
        classifier_windows = [
            e
            for e in store.entities("measurement")
            if e.attributes.get("name") == "yamnet_window" and not store.is_invalidated(e.id)
        ]
    speech_family = {str(label) for label in (config.get("taxonomy.speech_labels") or [])}
    corroborate = store.activity(
        node=NODE,
        step="corroborate",
        parameters={
            "coverage_threshold": values["coverage_threshold"],
            "speech_labels": sorted(speech_family) or None,
        },
    )
    store.was_associated_with(corroborate, software)
    if fold is not None:
        store.used(corroborate, fold.id)
    for prior in prior_spans:
        store.used(corroborate, prior.id)

    stoi_floor = config.get("speech.speech_test_stoi_floor")
    si_sdr_floor = config.get("speech.speech_test_si_sdr_floor")
    corroboration: list[dict[str, Any]] = []
    squim_by_span: list[dict[str, Any]] = []
    for start, end in span_extents:
        clip = Audio(
            waveform=plain.waveform[:, int(start * sampling_rate) : int(end * sampling_rate)],
            sampling_rate=sampling_rate,
        )
        squim: dict[str, Any]
        try:
            [scores] = extract_objective_quality_features_from_audios([clip])
            squim = {"stoi": float(scores["stoi"]), "pesq": float(scores["pesq"]), "si_sdr": float(scores["si_sdr"])}
        except Exception as err:  # noqa: BLE001 — a span SQUIM refuses is unmeasured, not padded
            squim = {"unmeasured": type(err).__name__}
        squim_by_span.append(squim)

        coverage = None if not speech_family else _speech_coverage(classifier_windows, (start, end), speech_family)
        if not speech_family:
            yamnet_vote = "unavailable"
        elif coverage is None:
            yamnet_vote = "not_evaluated"
        else:
            yamnet_vote = "confirm" if coverage >= values["coverage_threshold"] else "disconfirm"
            if yamnet_vote == "disconfirm":
                flags.append(f"the classifier disconfirms span {start:.2f}-{end:.2f}s (speech coverage {coverage:.2f})")
        if stoi_floor is None or si_sdr_floor is None or "unmeasured" in squim:
            squim_vote = "not_evaluated"
        else:
            squim_ok = squim["stoi"] >= float(stoi_floor) and squim["si_sdr"] >= float(si_sdr_floor)
            squim_vote = "confirm" if squim_ok else "disconfirm"
        if {yamnet_vote, squim_vote} <= {"confirm", "disconfirm"} and squim_vote != yamnet_vote:
            flags.append(
                f"instruments disagree on span {start:.2f}-{end:.2f}s: classifier {yamnet_vote}, squim {squim_vote}"
            )
        corroboration.append({"yamnet_coverage": coverage, "yamnet_vote": yamnet_vote, "squim_vote": squim_vote})

    # Step 4 — diarize over [first word start, last word end] only; every segment counts.
    interval = clamp_extent(
        (min(start for start, _, _ in grouped), max(end for _, end, _ in grouped)),
        plain,
    )
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

    count: int | None
    diarization_state: str
    cropped: Audio | None = None
    speaker_segments: list[tuple[str, str, tuple[float, float]]] = []  # (entity_id, speaker, extent)
    try:
        (cropped,) = extract_segments([(plain, [interval])])[0]
    except ValueError as error:
        count = None
        diarization_state = "interval_selects_no_samples"
        flags.append(f"the diarization interval selects no samples: {error}")
    else:
        [segments] = diarize_audios([cropped], model=diarizer)
        for segment in segments:
            extent = (float(segment.start or 0.0) + interval[0], float(segment.end or 0.0) + interval[0])
            speaker_id = store.entity(
                prov_type="speaker",
                extent=extent,
                attributes={"speaker": segment.speaker, "diarizer": str(diarizer.path_or_uri)},
            )
            store.was_generated_by(speaker_id, diarize_act)
            store.was_attributed_to(speaker_id, diarizer_agent)
            view.append(speaker_id)
            speaker_segments.append((speaker_id, str(segment.speaker), extent))
        count = len({speaker for _, speaker, _ in speaker_segments})
        diarization_state = "diarized"

    second = config.get("speech.second_diarizer")
    second_record: Any = "not_consulted"
    if count is not None and count != 1:
        flags.append(f"speaker count {count} != 1")
        if second is not None and cropped is not None:
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
            second_count = len({segment.speaker for segment in second_segments})
            second_record = {"model": str(second), "count": second_count, "agrees": second_count == count}
            if second_count != count:
                flags.append(f"second diarizer counts {second_count} speakers against {count}")

    # Step 5 — separation: measurement-gated, and neither backend is selected by default.
    backend = config.get("speech.separation_backend")
    sound_class = config.get("speech.separation_sound_class")
    separation_state: Any
    separated: list[Audio] = []
    if count is None or cropped is None:
        separation_state = "no_speaker_count"
    elif count < SEPARABLE_SOURCES:
        separation_state = "not_needed"
    elif backend is None:
        separation_state = "not_selected"
    elif count > SEPARABLE_SOURCES:
        separation_state = f"count_{count}_exceeds_backend"
        flags.append(f"separation cannot serve {count} speakers; the checkpoints separate exactly 2")
    elif str(backend) == UNASDIFF_BACKEND and sound_class is None:
        separation_state = "unconditioned_sound_slot_unavailable"
        flags.append(
            "unasdiff speech_sound requires a conditioning class for its sound slot and "
            "speech.separation_sound_class is unmeasured"
        )
    elif str(backend) == UNASDIFF_BACKEND:
        separation_state = {
            "backend": UNASDIFF_BACKEND,
            "mode": "speech_sound",
            "source_classes": [str(sound_class)],
        }
        separated = separate_audios(
            [cropped],
            model=None,
            n_sources=SEPARABLE_SOURCES,
            mode="speech_sound",
            source_classes=[str(sound_class)],
        )[0]
    else:
        separator = _clearvoice_model(f"{CLEARVOICE_ORG}/{backend}")
        separation_state = {"backend": str(backend), "n_sources": SEPARABLE_SOURCES}
        separated = separate_audios([cropped], model=separator, n_sources=SEPARABLE_SOURCES)[0]

    if separated:
        separate_act = store.activity(
            node=NODE, step="separate", parameters={"backend": str(backend), "interval": list(interval)}
        )
        store.was_associated_with(separate_act, software)
        store.used(separate_act, plain_id)
        store.used(separate_act, interval_id)
        for position, stream_audio in enumerate(separated):
            meta = dict(stream_audio.metadata.get("clearvoice") or {})
            index = int(meta.get("source_index", position))
            path = f"streams/separated_{index}.wav"
            stream_audio.save_to_file(str(run_dir / path))
            stream_id = store.entity(
                prov_type="stream",
                extent=interval,
                attributes={
                    "name": f"separated_{index}",
                    "path": path,
                    "sampling_rate": int(stream_audio.sampling_rate),
                    "channels": 1,
                    "source_index": index,
                    "backend": str(backend),
                    "input_norm_scalar": meta.get("input_norm_scalar"),
                    "separation_model": meta.get("model"),
                    "separation_commit": meta.get("commit"),
                },
            )
            store.was_generated_by(stream_id, separate_act)
            store.was_attributed_to(stream_id, software)
            store.was_derived_from(stream_id, plain_id)
            view.append(stream_id)

    # Step 6 — identify: words to speakers by timing, and the target by enrollment.
    identify = store.activity(node=NODE, step="identify", parameters={})
    store.was_associated_with(identify, software)
    word_speakers: list[str | None] = []
    for word in words:
        extent = word.extent or (0.0, 0.0)
        overlapping = [entry for entry in speaker_segments if _overlaps(extent, entry[2])]
        speaker: str | None
        note: str | None
        if len(overlapping) > 1:
            speaker, note = None, "straddles"
        elif len(overlapping) == 1:
            speaker, note = overlapping[0][1], None
        else:
            speaker, note = None, "unassigned"
        word_speakers.append(speaker)
        attribution_id = store.entity(
            prov_type="assertion",
            extent=extent,
            attributes={"verb": "attribute", "speaker": speaker, "note": note, "stream": plain_id},
        )
        store.was_generated_by(attribution_id, identify)
        store.was_attributed_to(attribution_id, software)
        store.was_derived_from(attribution_id, word.id)
        view.append(attribution_id)

    enrollment_id: str | None = None
    target_speaker: str | None = None
    if enrollment is not None:
        enrollment_id = store.entity(
            prov_type="enrollment",
            extent=None,
            attributes={
                "subject_id": enrollment.subject_id,
                "model_id": enrollment.provenance.model_id,
                "model_commit_sha": enrollment.provenance.model_commit_sha,
                "unresolved_reason": enrollment.provenance.unresolved_reason,
                "task": enrollment.task,
                "method": enrollment.provenance.method,
                "sources": enrollment.sources,
                "n_windows_used": enrollment.provenance.n_windows_used,
                "n_windows_dropped": enrollment.provenance.n_windows_dropped,
                "dimension": len(enrollment.vector),
            },
        )
        store.was_generated_by(enrollment_id, identify)
        store.was_attributed_to(enrollment_id, software)
        view.append(enrollment_id)

        refusal = enrollment.refusal_against(values["enrollment_model_id"], values["enrollment_revision"])
        if refusal is not None:
            flags.append(refusal)
        elif speaker_segments:
            probe = _embedding_model(values["enrollment_model_id"], values["enrollment_revision"])
            labels = sorted({speaker for _, speaker, _ in speaker_segments})
            audios: list[Audio] = []
            for label in labels:
                slices = [
                    plain.waveform[:, int(s * sampling_rate) : int(e * sampling_rate)]
                    for _, speaker, extent in speaker_segments
                    if speaker == label
                    for s, e in [clamp_extent(extent, plain)]
                ]
                audios.append(Audio(waveform=torch.cat(slices, dim=1), sampling_rate=sampling_rate))
            embedding_agent = store.agent(
                agent_type="model", model_id=str(probe.path_or_uri), commit_sha=probe.commit_sha
            )
            store.was_associated_with(identify, embedding_agent)
            embeddings = extract_speaker_embeddings_from_audios(audios, model=probe)
            enrolled = torch.tensor(enrollment.vector, dtype=torch.float32)
            cut = float(values["target_match_cosine"])
            best: tuple[float, str] | None = None
            for label, embedding in zip(labels, embeddings):
                similarity = _cosine(embedding, enrolled)
                match_id = store.entity(
                    prov_type="target_match",
                    extent=None,
                    attributes={
                        "speaker": label,
                        "similarity": similarity,
                        "threshold": cut,
                        "enrollment_model": enrollment.provenance.model_id,
                        "enrollment_commit": enrollment.provenance.model_commit_sha,
                        "probe_model": str(probe.path_or_uri),
                        "probe_revision": str(probe.revision),
                        "probe_commit": probe.commit_sha,
                        "stream": plain_id,
                    },
                )
                store.was_generated_by(match_id, identify)
                store.was_attributed_to(match_id, embedding_agent)
                store.was_derived_from(match_id, enrollment_id)
                view.append(match_id)
                if best is None or similarity > best[0]:
                    best = (similarity, label)
            if best is not None and best[0] >= cut:
                target_speaker = best[1]
            else:
                flags.append("an enrollment was given and no speaker matches it")

    # The span elements, carrying every conclusion drawn over them. Nothing here is invalidated.
    span_ids: list[str] = []
    for position, ((start, end), (_, _, members)) in enumerate(zip(span_extents, grouped)):
        owners = {word_speakers[index] for index in members}
        attributed_to = owners.pop() if len(owners) == 1 else None
        span_id = store.entity(
            prov_type="span",
            extent=(start, end),
            attributes={
                "family": "speech",
                "words_n": len(members),
                "attributed_to": attributed_to,
                "nontarget": None
                if target_speaker is None or attributed_to is None
                else attributed_to != target_speaker,
                **corroboration[position],
            },
        )
        store.was_generated_by(span_id, corroborate)
        store.was_attributed_to(span_id, software)
        for prior in prior_spans:
            if prior.extent is not None and _overlaps((start, end), prior.extent):
                store.was_derived_from(span_id, prior.id)
        span_ids.append(span_id)
        view.append(span_id)

    # Step 7 — PII: one scan, one text, and that text is the consensus transcript.
    pii_act = store.activity(node=NODE, step="pii", parameters={"text": "consensus_transcript"})
    store.was_associated_with(pii_act, software)
    store.used(pii_act, consensus.id)
    raw_scans = scan_for_pii([transcript_text])
    scans: list[PiiScan] = raw_scans if isinstance(raw_scans, list) else [raw_scans]
    failures: dict[str, str] = {}
    scanned_by: set[str] = set()
    findings: list[dict[str, Any]] = []
    for scan in scans:
        failures.update(scan.failures)
        scanned_by.update(scan.detectors_used)
        for finding in scan.spans:
            # One occurrence, one finding: the scan dedupes by (category, text, source), so a name
            # said twice arrives here once and must still be marked at both places it was said.
            located = _locate(str(finding.text or ""), words)
            if not located:
                flags.append(f"pii_unlocated ({finding.category})")
                occurrences = [(0, len(words) - 1)]
            else:
                occurrences = located
            for first, last in occurrences:
                covered = list(range(first, last + 1))
                extent = (
                    (span_extents[0][0], span_extents[-1][1])
                    if not located
                    else (
                        float((words[first].extent or (0.0, 0.0))[0]),
                        float((words[last].extent or (0.0, 0.0))[1]),
                    )
                )
                recognizers = sorted(
                    {str(name) for index in covered for name in (words[index].attributes.get("recognizers") or [])}
                )
                speakers = {word_speakers[index] for index in covered}
                resolved = len(speakers) == 1 and None not in speakers
                pii_id = store.entity(
                    prov_type="pii",
                    extent=extent,
                    attributes={
                        "category": finding.category,
                        "source": finding.source,
                        "recognizers": recognizers,
                        "occurrence": occurrences.index((first, last)),
                        "occurrences_n": len(occurrences),
                        "detectors_used": sorted(scan.detectors_used),
                        "detectors_failed": sorted(scan.failures),
                    },
                )
                store.was_generated_by(pii_id, pii_act)
                store.was_attributed_to(pii_id, software)
                store.was_derived_from(pii_id, consensus.id)
                view.append(pii_id)
                for index in covered:
                    mark_id = store.entity(
                        prov_type="assertion",
                        extent=words[index].extent,
                        attributes={"verb": "label", "label": "pii", "category": finding.category},
                    )
                    store.was_generated_by(mark_id, pii_act)
                    store.was_attributed_to(mark_id, software)
                    store.was_derived_from(mark_id, words[index].id)
                    view.append(mark_id)
                findings.append(
                    {
                        "category": finding.category,
                        "speaker": next(iter(speakers)) if resolved else None,
                        "resolved": resolved,
                    }
                )
    missing = _missing_detectors(values["required_detectors"], scanned_by, failures)
    flags.extend(_decide_pii(findings, failures, missing, target_speaker))
    scan_id = store.entity(
        prov_type="measurement",
        extent=None,
        attributes={
            "name": "pii_scan",
            "signal": "consensus_transcript",
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

    # Step 9 — the non-target axis: measured and reported per span, compared against nothing.
    proximity_act = store.activity(node=NODE, step="proximity", parameters={})
    store.was_associated_with(proximity_act, software)
    store.used(proximity_act, plain_id)
    level = find_measurement(store, "level")
    reference_rms_dbfs = None if level is None else float(level.attributes["rms_dbfs"])
    if level is not None:
        store.used(proximity_act, level.id)
    proximity_by_span: list[dict[str, Any]] = []
    for span_id, (start, end) in zip(span_ids, span_extents):
        segment = plain.waveform[0, int(start * sampling_rate) : int(end * sampling_rate)].numpy()
        measure = _proximity(segment, sampling_rate, reference_rms_dbfs)
        proximity_by_span.append(measure)
        proximity_id = store.entity(
            prov_type="measurement",
            extent=(start, end),
            attributes={"name": "proximity", "stream": plain_id, **measure},
        )
        store.was_generated_by(proximity_id, proximity_act)
        store.was_attributed_to(proximity_id, software)
        store.was_derived_from(proximity_id, span_id)
        view.append(proximity_id)

    cuts = {leg: config.get(f"speech.nontarget.{leg}") for leg in NONTARGET_LEGS}
    nontarget_speech_s: float | None
    if any(cut is None for cut in cuts.values()):
        nontarget_speech_s = None
    else:
        supplied = {leg: float(cut) for leg, cut in cuts.items()}
        nontarget_speech_s = float(
            sum(
                end - start
                for (start, end), measure in zip(span_extents, proximity_by_span)
                if _behind_the_target(measure, supplied)
            )
        )

    # Outcome — fail only from the no-words row above; flag from the accumulated reasons; else pass.
    detail: dict[str, Any] = {
        "speaker_count": count,
        "diarization": diarization_state,
        "words_n": len(words),
        "speech_s": speech_s,
        "nontarget_speech_s": nontarget_speech_s,
        "pii": {
            "categories": sorted({str(finding["category"]) for finding in findings}),
            "n": len(findings),
            "scanned_by": sorted(scanned_by),
            "failed": sorted(failures),
            "missing": missing,
        },
        "second_diarizer": second_record,
        "separation": separation_state,
        "flags": flags,
    }
    if target_speaker is not None:
        detail["target_speaker"] = target_speaker
    if enrollment_id is not None:
        detail["enrollment_id"] = enrollment_id
    if flags:
        outcome, why = Outcome.FLAG, "; ".join(flags)
    else:
        outcome = Outcome.PASS
        why = "words, spans, speakers and quality are in the store"
    verdict_id, verdict = write_verdict(
        store, proximity_act, software, node=NODE, outcome=outcome, kind="speech", why=why, detail=detail
    )
    view.append(verdict_id)
    return NodeResult(verdict=verdict, view=tuple(view), verdict_entity_id=verdict_id)
