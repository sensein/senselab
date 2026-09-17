"""The SPEECH branch: the expectation, the consensus transcript, the speakers, PII and quality.

Two entry points and one mode decision. ``align_speech`` evaluates a declared speech task against
what its own instruction asked for; ``detect_speech`` finds lexical speech on a recording of another
branch's kind and evaluates nothing. :func:`~...nodes.branches.dispatch` picks between them from the
declared task family alone, and both write by ``propose`` only.

It runs no ASR and never re-transcribes: PREPROCESS wrote the consensus text stream with
``senselab.audio.workflows.triage.consensus.align_sources`` and this branch reads it. It runs no
diarizer either: the speakers are PREPROCESS's whole-file ``<stream>_diarization`` derivative, read
back. Speech spans come from the lexical consensus words' timings, never the envelope. The second
diarizer runs only when the read count is not 1 and ``speech.second_diarizer`` names a model;
separation runs only when ``speech.separation_backend`` names a backend. The target speaker is
identified by a caller-supplied enrollment, not by a per-file hint, and an enrollment is refused
rather than compared unless its model and its resolved commit are both the probe's. The PII scan
reads the consensus transcript and each recognizer's own transcript, once, and marks every
occurrence of what it finds on the consensus words. This branch marks; it removes nothing.

Every parameter's derivation is in ``data/config/default.yaml``; the design is in
``specs/20260817-triage-workflow-dag/branch-speech.md`` and what porting it decided is in
``branch-speech-implementation.md`` beside it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

from senselab.audio.data_structures import Audio, AudioHints
from senselab.audio.tasks.disruptions.api import detect_disruptions
from senselab.audio.tasks.features_extraction.torchaudio_squim import (
    extract_objective_quality_features_from_audios,
)
from senselab.audio.tasks.source_separation.api import separate_audios
from senselab.audio.tasks.spans.api import group_extents_into_runs
from senselab.audio.tasks.speaker_diarization.api import diarize_audios
from senselab.audio.tasks.speaker_embeddings.api import extract_speaker_embeddings_from_audios
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.enrollment import Enrollment
from senselab.audio.workflows.triage.nodes.branches import (
    BRANCH_FAMILY,
    PARAM_KEYS,
    PARAM_SECTION,
    PROPOSERS,
    SPEECH_EXPECTATIONS,
    UNDETERMINED,
    BranchParams,
    Done,
    Expectation,
    Finding,
    Pattern,
    Proposal,
    Result,
    asr_spans,
    branch_params,
    content_coverage,
    contest,
    count,
    declared_duration_count,
    derivative_arrays,
    deviation,
    deviation_names,
    dispatch,
    duration,
    group_by_breaks,
    hull,
    inter_word_gaps,
    lexical_runs,
    measured,
    merge,
    mode_of,
    ngram_echo_fraction,
    off_task,
    ordered_run,
    overlaps,
    propose_spans,
    stream_extent,
    touches_edge,
    unviable,
    unviable_findings,
    word_extent,
    word_text,
    write_findings,
)
from senselab.audio.workflows.triage.nodes.common import (
    BranchResult,
    clamp_extent,
    consensus_words,
    find_measurement,
    find_measurements,
    lexical_words,
    live_entities,
    path_attributes,
    resolve_stream,
    software_agent,
    write_report,
    write_stream,
)
from senselab.audio.workflows.triage.nodes.ddk import (
    NO_ENVELOPE,
    NO_PPG,
    DdkReads,
    align_ddk,
    read_ddk,
    syllable_detail,
)
from senselab.audio.workflows.triage.routing_analysis.families import SYLLABLE_REPETITION
from senselab.audio.workflows.triage.stimulus import LexicalWord, StimulusAlignment, align_stimulus
from senselab.audio.workflows.triage.vocabulary import TASK
from senselab.text.tasks.pii_detection.api import PiiScan, scan_for_pii
from senselab.utils.data_structures import HFModel, SpeechBrainModel
from senselab.utils.prov_store import Entity, ProvStore

NODE = "SPEECH"
ORIGINAL = "recording"  # the stream disruptions are measured on: as captured, unnormalised, unresampled
DIARIZATION_DERIVATIVE = "diarization"
"""The stem of PREPROCESS's per-stream diarization measurement, ``<stream>_diarization``."""
CLEARVOICE_ORG = "alibabasglab"
UNASDIFF_BACKEND = "unasdiff"
SEPARABLE_SOURCES = 2
NONTARGET_LEGS = ("level_db", "tilt_db_per_octave", "d_to_r_db")
REPETITION_ALLOWED_CATEGORIES = ("Letters", "Numbers")
"""The `random-item-generation` categories whose own instruction permits repeating an item."""


def diarization_measurement(stream: str) -> str:
    """The measurement name one stream's whole-file diarization is written under.

    The same spelling ``preprocess.diarization_measurement`` writes. It is repeated here rather
    than imported, because importing PREPROCESS into a branch would put the whole of PREPROCESS's
    model imports on a branch's import path; ``speech_test`` pins the two against each other.

    Args:
        stream: The stream's name.

    Returns:
        The measurement's name.
    """
    return f"{stream}_{DIARIZATION_DERIVATIVE}"


@dataclass(frozen=True)
class _Diarization:
    """One stream's whole-file diarization, as this branch reads it back.

    Attributes:
        measurement: The measurement's name.
        measurement_id: Its entity id, which every speaker entity derives from.
        signal: The stream it was measured on.
        model: The diarizer that produced it.
        exclusive: Whether the exclusive partition was taken. False keeps pyannote's overlapping
            view, under which one word can straddle two speakers' segments.
        n_speakers: The count the measurement itself records.
        segments: ``(start, end, speaker)`` per segment, in time order.
    """

    measurement: str
    measurement_id: str
    signal: str
    model: str
    exclusive: bool
    n_speakers: int
    segments: list[tuple[float, float, str]]


def _read_diarization(store: ProvStore, run_dir: Path, config: TriageConfig) -> _Diarization | None:
    """PREPROCESS's whole-file diarization of the first configured stream that has one.

    Args:
        store: The provenance store.
        run_dir: The run directory the sidecar path is relative to.
        config: The triage configuration, read for ``diarization.streams``.

    Returns:
        The reading, or None when no configured stream has a live measurement whose sidecar is
        readable.
    """
    for stream in config.get("diarization.streams") or ():
        name = diarization_measurement(str(stream))
        measurement = find_measurement(store, name)
        arrays = derivative_arrays(store, run_dir, name)
        if measurement is None or arrays is None:
            continue
        segments = sorted(
            (float(start), float(end), str(speaker))
            for start, end, speaker in zip(arrays["starts"], arrays["ends"], arrays["speakers"])
        )
        return _Diarization(
            measurement=name,
            measurement_id=measurement.id,
            signal=str(measurement.attributes.get("signal", stream)),
            model=str(measurement.attributes.get("model", "")),
            exclusive=bool(measurement.attributes.get("exclusive")),
            n_speakers=int(measurement.attributes.get("n_speakers", 0)),
            segments=segments,
        )
    return None


def _exclusive_slices(label: str, segments: list[tuple[str, str, tuple[float, float]]]) -> list[tuple[float, float]]:
    """One speaker's segments with every region another speaker also holds removed.

    Under ``diarization.exclusive: false`` two speakers' segments can overlap, so a speaker's
    concatenated audio would otherwise carry the other's voice wherever they spoke at once. The
    old in-branch pass took the exclusive partition and could not produce that; this keeps the
    property without asking for a threshold.

    Args:
        label: The speaker whose audio is wanted.
        segments: ``(entity id, speaker, extent)`` per segment.

    Returns:
        The extents, earliest first, with the overlapped regions cut out.
    """
    mine = merge([extent for _, speaker, extent in segments if speaker == label])
    theirs = merge([extent for _, speaker, extent in segments if speaker != label])
    kept: list[tuple[float, float]] = []
    for start, end in mine:
        cursor = start
        for other_start, other_end in theirs:
            if other_end <= cursor or other_start >= end:
                continue
            if other_start > cursor:
                kept.append((cursor, other_start))
            cursor = max(cursor, other_end)
        if end > cursor:
            kept.append((cursor, end))
    return kept


def _second_diarizer_model(model_id: str) -> HFModel:
    """The configured second diarizer's model spec; its commit resolves at construction.

    Args:
        model_id: From ``speech.second_diarizer``.

    Returns:
        The model spec.
    """
    return HFModel(path_or_uri=model_id, revision="main")


def _clearvoice_model(model_id: str) -> HFModel:
    """A ClearerVoice separation checkpoint's model spec; its commit resolves at construction.

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


def _required(params: BranchParams, enrollment: Enrollment | None) -> dict[str, Any]:
    """Resolve the settings this branch reads, refusing none of them.

    The first group are settings of other sections that ship values; a null in one is an override
    error and ``require`` surfaces it. The enrollment group is nullable in the packaged file, so it
    is read through :meth:`BranchParams.setting`, which records the ask and returns None rather
    than stopping the branch — a refusal would be this branch deciding.

    Args:
        params: The operating points, which also carry the configuration and the record of misses.
        enrollment: The caller's enrollment; one additionally needs the probe and the match cut.

    Returns:
        The resolved values, keyed by their short names. An enrollment value nobody measured is
        None and is recorded in ``params.missing``.
    """
    config = params.config
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
        model = params.setting("speech.enrollment_model", dict)
        values["enrollment_model_id"] = None if model is None else str(model["model_id"])
        values["enrollment_revision"] = None if model is None else str(model["revision"])
        values["target_match_cosine"] = params.setting("speech.target_match_cosine", float)
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


def _locate(finding_text: str, haystack_tokens: list[str]) -> list[tuple[int, int]]:
    """Every place the finding's tokens match the haystack, as contiguous runs (N11).

    Every occurrence, not the first: the scan dedupes by ``(category, text, source)``, so a name
    said twice arrives as one finding, and locating only its first match leaves the second
    occurrence unmarked and therefore unredacted.

    Args:
        finding_text: The detector's matched text.
        haystack_tokens: The scanned text's tokens, one per word, in the order they were scanned.

    Returns:
        ``[(first index, last index), ...]`` into ``haystack_tokens``, non-overlapping and in order.
        Empty when nothing matches.
    """
    tokens = [_norm_token(token) for token in finding_text.split()]
    haystack = [_norm_token(token) for token in haystack_tokens]
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


def _reading(word: Entity, haystack: str) -> str:
    """The token this word contributed to one scanned text.

    Args:
        word: A consensus ``word`` entity.
        haystack: ``"consensus"`` for the consensus text, else a source name.

    Returns:
        The word's ``text`` for the consensus haystack, else that source's own reading of it.
    """
    if haystack == "consensus":
        return str(word.attributes.get("text") or "")
    return str(word.attributes["readings"][haystack])


def _timings_hull(words: list[Entity], covered: list[int]) -> tuple[float, float]:
    """The hull of the covered words' per-source timings: every recognizer's placement of them.

    Args:
        words: The consensus words, in stream order.
        covered: The positions a finding covers.

    Returns:
        ``(min member start, max member end)`` over every source timing of every covered word.
    """
    spans = [span for index in covered for span in words[index].attributes["timings"].values()]
    return min(float(span[0]) for span in spans), max(float(span[1]) for span in spans)


def _hypotheses(store: ProvStore, source_names: list[str]) -> dict[str, Entity]:
    """The live ``asr_hypothesis`` measurement of each consensus source, latest write per source.

    Args:
        store: The provenance store.
        source_names: The sources the consensus was aligned over.

    Returns:
        ``{source name: measurement entity}``.

    Raises:
        LookupError: If a source has no live hypothesis measurement in the store.
    """
    found: dict[str, Entity] = {}
    for measurement in live_entities(store, "measurement"):
        if measurement.attributes.get("role") == "asr_hypothesis":
            found[str(measurement.attributes["source"])] = measurement
    missing = [name for name in source_names if name not in found]
    if missing:
        raise LookupError(f"no asr_hypothesis measurement for consensus source(s) {missing}")
    return {name: found[name] for name in source_names}


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


def _pii_notes(
    findings: list[dict[str, Any]],
    failures: dict[str, str],
    missing: list[str],
    target_speaker: str | None,
) -> list[str]:
    """What the PII scan could not establish, named. This branch decides nothing about it.

    Args:
        findings: One record per finding, carrying its category and its resolved speaker.
        failures: The detectors that were attempted and failed.
        missing: The required detectors that were never attempted.
        target_speaker: The diarized speaker the enrollment matched, or None.

    Returns:
        The observations, in controlled vocabulary, for the report's ``notes``. REDACT's gate is
        the ``pii`` entities in the store (``run._speech_found_pii``) and not this list, so nothing
        here suppresses or triggers a redaction.
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


@dataclass(frozen=True)
class _SpeakerRun:
    """One contiguous run of consensus words the diarization gives to the same speaker.

    Attributes:
        speaker: The diarizer's own label, or None where no single segment carries the run.
        note: Why the speaker is None — ``straddles`` or ``unassigned`` — or None.
        words: The ``word`` entities in the run, in stream order.
    """

    speaker: str | None
    note: str | None
    words: list[Entity]


def _speaker_runs(words: list[Entity], speakers: list[str | None], notes: list[str | None]) -> list[_SpeakerRun]:
    """Group the consensus words into the runs one speaker holds.

    Args:
        words: The consensus words, in stream order.
        speakers: The speaker each word was attributed to, in the same order.
        notes: The note each word carries, in the same order.

    Returns:
        The runs, in stream order. A word whose speaker could not be resolved joins the run of
        words beside it that could not either, so the aggregate says which case it is.
    """
    runs: list[_SpeakerRun] = []
    for word, speaker, note in zip(words, speakers, notes):
        if runs and runs[-1].speaker == speaker and runs[-1].note == note:
            runs[-1].words.append(word)
        else:
            runs.append(_SpeakerRun(speaker=speaker, note=note, words=[word]))
    return runs


# --------------------------------------------------------------------- the two modes

MINT = PROPOSERS[NODE]
"""SPEECH's own minting function. It proposes into ``family: "speech"`` and can reach no other."""

STIMULUS_MEASUREMENT = "stimulus_alignment"
"""PREPROCESS's derivative the fully-specified families are evaluated against."""

BREATH_TOKEN = "[breath]"
"""The one bracketed token that is never a filler: a breath in a passage reading is structure."""


def _consensus_id(store: ProvStore) -> str | None:
    """The live ``consensus_transcript`` measurement's id, which every proposal here derives from.

    Args:
        store: The provenance store.

    Returns:
        The id, or None when PREPROCESS wrote no consensus.
    """
    consensus = find_measurement(store, "consensus_transcript")
    return None if consensus is None else consensus.id


def _stimulus(store: ProvStore, hint: AudioHints | None, params: BranchParams) -> tuple[str, StimulusAlignment] | None:
    """PREPROCESS's stimulus alignment, as the three projections a branch reads.

    Args:
        store: The provenance store.
        hint: What the recording was declared to contain.
        params: The operating points, read for ``stimulus.sentence_terminators``.

    Returns:
        The measurement's id and the alignment, or None when the derivative or the declaration is
        absent.
    """
    measurement = find_measurement(store, STIMULUS_MEASUREMENT)
    prompts = list(hint.expected_speech) if hint is not None else []
    if measurement is None or not prompts:
        return None
    words = [
        LexicalWord(
            index=int(word.attributes["index"]),
            text=word_text(word),
            extent=word.extent,
            agreement=float(word.attributes["agreement"]),
        )
        for word in lexical_words(store)
    ]
    terminators = params.setting("stimulus.sentence_terminators")
    if terminators is None:
        return None
    return measurement.id, align_stimulus(prompts, words, terminators=terminators)


def _stimulus_agrees(store: ProvStore, alignment: StimulusAlignment) -> bool:
    """Whether the rebuilt alignment carries the counts PREPROCESS recorded for the derivative.

    Args:
        store: The provenance store.
        alignment: The rebuilt alignment.

    Returns:
        True when every count the measurement carries matches the rebuild, and when it carries none
        to compare against.
    """
    measurement = find_measurement(store, STIMULUS_MEASUREMENT)
    if measurement is None:
        return True
    keys = ("n_expected", "n_realised", "n_substituted", "n_absent", "n_unexpected")
    recorded = {key: measurement.attributes[key] for key in keys if measurement.attributes.get(key) is not None}
    return all(int(value) == int(alignment.provenance[key]) for key, value in recorded.items())


def _anchor(alignment: StimulusAlignment, index: int) -> float | None:
    """Where in the signal an omitted token should have been: the end of the last one before it.

    Args:
        alignment: The alignment.
        index: The omitted token's index in the expected stream.

    Returns:
        The time, or None when nothing before it was realised.
    """
    ends = [token.extent[1] for token in alignment.expected if token.index < index and token.extent is not None]
    return max(ends) if ends else None


def _repeat_fraction(expected: Sequence[str], produced: Sequence[str]) -> float:
    """What fraction of the expected keys the production realised more than once.

    Args:
        expected: The expected token keys, in order.
        produced: The produced lexical keys, in order.

    Returns:
        The fraction, or 0.0 when nothing was expected.
    """
    wanted = set(expected)
    if not wanted:
        return 0.0
    counts: dict[str, int] = {}
    for key in produced:
        if key in wanted:
            counts[key] = counts.get(key, 0) + 1
    return sum(1 for key in wanted if counts.get(key, 0) > 1) / len(wanted)


def _breath_groups(store: ProvStore, points: BranchParams) -> list[tuple[float, float]]:
    """The breath groups of a connected production, from the inter-word gaps and breath evidence.

    Args:
        store: The provenance store.
        points: The operating points.

    Returns:
        One extent per group, or nothing when the gap that breaks a group is unmeasured or the
        recording carries no lexical word.
    """
    words = lexical_words(store)
    min_gap = points.point("breath_group_min_gap_s")
    if not words or min_gap is None:
        return []
    breaks = list(inter_word_gaps(words, float(min_gap)))
    score_min = points.point("score_min")
    breath_labels = set((points.point("label_sets") or {}).get("breath", ())) if score_min is not None else set()
    if breath_labels:
        for window in find_measurements(store, "span_hear"):
            scores = window.attributes.get("raw_scores") or {}
            if window.extent is None:
                continue
            if any(float(scores.get(label, 0.0)) >= float(score_min) for label in breath_labels):
                breaks.append(window.extent)
    for word in consensus_words(store):
        if word.attributes["bracketed"] and word_text(word) == BREATH_TOKEN and word.extent is not None:
            breaks.append(word.extent)
    return group_by_breaks(words, merge([extent for extent in breaks if extent[1] > extent[0]]))


def _breath_group_components(store: ProvStore, points: BranchParams, evidence: Sequence[str]) -> list[Proposal]:
    """One proposed span per breath group of a connected production.

    Args:
        store: The provenance store.
        points: The operating points.
        evidence: The entity ids every group derives from.

    Returns:
        The proposals, in time order. Empty when no evidence can be named.
    """
    if not evidence:
        return []
    return [
        MINT(f"breath_group_{index}", extent, *evidence, group_index=index)
        for index, extent in enumerate(_breath_groups(store, points))
        if extent[1] > extent[0]
    ]


def _off_task(components: Sequence[Proposal], store: ProvStore, points: BranchParams) -> list[Finding]:
    """The gaps no proposed span covers, once the shortest one reported has been measured.

    Args:
        components: The spans this evaluation proposed.
        store: The provenance store.
        points: The operating points.

    Returns:
        The findings, or nothing while ``branch.gap_off_task_min_s`` is unmeasured.
    """
    cut = points.point("gap_off_task_min_s")
    if cut is None:
        return []
    return off_task(components, live_entities(store, "span"), float(cut))


def _speech_ordered(  # noqa: C901 — the two token sources and the five departures, in order
    expectation: Expectation, store: ProvStore, hint: AudioHints | None, params: BranchParams
) -> Result:
    """A task whose instruction prescribes a token sequence: was it produced, and how?

    Proposes ``task_extent``, one span per realised structure unit the alignment yields — a CAPE-V
    sentence, a Rainbow sentence — and the breath groups where the family is connected. No span per
    token: a realised token already has a ``word`` entity carrying its own extent, its agreement and
    its per-source timings.

    Args:
        expectation: The row for this family.
        store: The provenance store.
        hint: What the recording was declared to contain.
        params: The operating points.

    Returns:
        Whether the prescribed sequence was produced, the spans, and the deviations.
    """
    points = params
    consensus_id = _consensus_id(store)
    words = lexical_words(store)
    if consensus_id is None:
        return Result(UNDETERMINED, [], [unviable("expected_token_sequence", "no consensus_transcript in the store")])

    substitutions: list[tuple[str, Entity]] = []
    omissions: list[tuple[int, str, float | None]] = []
    structure: list[tuple[int, tuple[float, float], dict[str, Any]]] = []
    repeat_fraction: float | None = None
    evidence: list[str] = [consensus_id]
    alignment: StimulusAlignment | None = None
    if expectation.tokens is not None:
        matched, omitted = ordered_run(list(expectation.tokens), words, params.p_normalise)
        omissions = [(index, token, None) for index, token in enumerate(omitted)]
        matched_ids = {word.id for _, word in matched}
        insertions = [word for word in words if word.id not in matched_ids]
    else:
        read = _stimulus(store, hint, params)
        if read is None:
            # The derivative is absent and the expectation is per recording, so no extent can be
            # placed: propose nothing rather than a span whose boundaries are guessed.
            return Result(
                UNDETERMINED,
                [],
                [
                    unviable(
                        "expected_token_sequence",
                        f"{STIMULUS_MEASUREMENT} is absent; the transcript alone cannot say what was expected",
                    )
                ],
            )
        stimulus_id, alignment = read
        evidence = [stimulus_id, consensus_id]
        by_index = {int(word.attributes["index"]): word for word in words}
        matched = [
            (token.text, by_index[token.word_index])
            for token in alignment.expected
            if token.realisation == "realised" and token.word_index in by_index
        ]
        substitutions = [
            (token.text, by_index[token.word_index])
            for token in alignment.substitutions
            if token.word_index in by_index
        ]
        omissions = [(token.index, token.text, _anchor(alignment, token.index)) for token in alignment.omissions]
        insertions = [by_index[word.index] for word in alignment.unexpected if word.index in by_index]
        structure = [
            (
                unit.index,
                unit.extent,
                {
                    "expected_n": len(unit.token_indices),
                    "realised_n": unit.n_realised,
                    "substituted_n": unit.n_substituted,
                },
            )
            for unit in alignment.structure_spans()
            if unit.extent is not None and unit.extent[1] > unit.extent[0]
        ]
        repeat_fraction = _repeat_fraction(
            [token.key for token in alignment.expected], [params.p_normalise(word_text(word)) for word in words]
        )

    components: list[Proposal] = []
    findings: list[Finding] = []
    if alignment is not None and not _stimulus_agrees(store, alignment):
        findings.append(measured("stimulus_alignment_rebuild_agrees", None, None, False, evidence[0]))
    if matched:
        read_extent = (word_extent(matched[0][1])[0], word_extent(matched[-1][1])[1])
        if read_extent[1] > read_extent[0]:
            components.append(
                MINT(
                    "task_extent",
                    read_extent,
                    *evidence,
                    *(word.id for _, word in matched),
                    words_n=len(matched),
                    expected_n=len(matched) + len(substitutions) + len(omissions),
                )
            )
            recording = stream_extent(store)
            if recording is not None and touches_edge(read_extent, recording):
                findings.append(
                    deviation("truncation", read_extent[0], read_extent[1], *(word.id for _, word in matched))
                )
            if repeat_fraction is not None:
                findings.append(
                    measured("expected_sequence_repeat_fraction", None, None, round(repeat_fraction, 3), *evidence)
                )
                cut = points.point("repeat_overlap_min")
                if cut is not None and repeat_fraction >= float(cut):
                    findings.append(
                        deviation(
                            "repeat_reading",
                            read_extent[0],
                            read_extent[1],
                            *evidence,
                            *(word.id for _, word in matched),
                            overlap=round(repeat_fraction, 3),
                        )
                    )
        for index, extent, counts in structure:
            components.append(MINT(f"structure_{index}", extent, *evidence, structure_index=index, **counts))

    for expected_text, word in substitutions:
        start, end = word_extent(word)
        findings.append(
            deviation(
                "stimulus_mismatch",
                start,
                end,
                word.id,
                expected=expected_text,
                read=word_text(word),
                agreement=word.attributes.get("agreement"),
                variants=word.attributes.get("variants"),
            )
        )
    for word in insertions:
        start, end = word_extent(word)
        findings.append(
            deviation(
                "stimulus_mismatch",
                start,
                end,
                word.id,
                expected=None,
                read=word_text(word),
                agreement=word.attributes.get("agreement"),
            )
        )
    for index, token, anchor in omissions:
        # An omission has no extent of its own — a skip-arc-free aligner assigns every stimulus word
        # an interval whether or not it was spoken — so it is placed where it should have been. The
        # `acoustic_score_max` covariate this used to carry named `branch.omission_score_max`, a cut
        # on an acoustic score no derivative in the graph produces; the key and the covariate were
        # removed together rather than the key being given a default it could not be reasoned into.
        findings.append(deviation("omission", anchor, anchor, *evidence, expected=token, expected_index=index))

    if expectation.emit_filler:
        for word in consensus_words(store):
            if word.attributes["bracketed"] and word_text(word) != BREATH_TOKEN:
                start, end = word_extent(word)
                findings.append(deviation("filler", start, end, word.id, text=word_text(word)))

    if expectation.connected:
        components.extend(_breath_group_components(store, points, evidence))

    if expectation.expected_event_count is not None:
        findings.append(
            count(
                "expected_event_count",
                len(matched),
                expectation.expected_event_count,
                *(word.id for _, word in matched),
            )
        )
    findings.extend(unviable_findings(expectation))
    findings.extend(declared_duration_count(store, expectation.declared_duration_s))
    findings.extend(_off_task(components, store, points))
    findings.extend(points.record())
    return Result(bool(matched) and not omissions, components, findings)


def _speech_free_response(  # noqa: C901 — the response, the connected measures and the anti-pattern
    expectation: Expectation, store: ProvStore, hint: AudioHints | None, params: BranchParams
) -> Result:
    """A task prescribing no words: was there a response, and how was it produced?

    Proposes ``task_extent`` over the hull of PREPROCESS's ASR spans, plus one span per breath group
    where the family is connected. A family carrying no ``stimulus_text`` — every
    ``picture-description`` and every ``cinderella-story`` in the corpus — expects nothing lexical,
    so every lexical word is the response rather than a departure from one.

    Args:
        expectation: The row for this family.
        store: The provenance store.
        hint: What the recording was declared to contain.
        params: The operating points.

    Returns:
        Whether a response was produced, the spans, and the findings.
    """
    points = params
    runs = asr_spans(live_entities(store, "span"))
    words = lexical_words(store)
    response = hull([span.extent for span in runs if span.extent is not None])
    consensus_id = _consensus_id(store)
    components: list[Proposal] = []
    findings: list[Finding] = []
    if response is not None and response[1] > response[0]:
        components.append(MINT("task_extent", response, *(span.id for span in runs), words_n=len(words)))
    minimum = points.point("response_min_s")
    done: Done = UNDETERMINED if minimum is None else (response is not None and duration(response) >= float(minimum))

    if expectation.connected and response is not None and duration(response) > 0.0:
        evidence = [entity_id for entity_id in (consensus_id,) if entity_id is not None]
        groups = _breath_groups(store, points)
        components.extend(
            MINT(f"breath_group_{index}", extent, *evidence, group_index=index)
            for index, extent in enumerate(groups)
            if extent[1] > extent[0] and evidence
        )
        # Named for their measurement convention, never `rate`: the name says what was counted.
        findings.append(
            measured(
                "speech_rate_from_consensus_words_per_s",
                response[0],
                response[1],
                round(len(words) / duration(response), 3),
                *(span.id for span in runs),
                *(word.id for word in words),
                support_words=len(words),
            )
        )
        pause_min = points.point("pause_min_s")
        if pause_min is not None:
            pauses = inter_word_gaps(words, float(pause_min))
            findings.append(
                measured(
                    "pause_fraction_of_response",
                    response[0],
                    response[1],
                    round(sum(duration(pause) for pause in pauses) / duration(response), 3),
                    *(span.id for span in runs),
                    *(word.id for word in words),
                    support_pauses=len(pauses),
                )
            )
        findings.append(count("breath_groups", len(groups), None, *evidence))

    if expectation.anti_pattern is not None:
        read = _stimulus(store, hint, params)
        ngram_n = points.point("echo_ngram_n")
        cut = points.point(
            "echo_overlap_max" if expectation.anti_pattern == "verbatim_prompt" else "verbatim_overlap_max"
        )
        if read is None:
            findings.append(unviable(f"anti_pattern_{expectation.anti_pattern}", f"{STIMULUS_MEASUREMENT} is absent"))
            done = UNDETERMINED
        elif ngram_n is None:
            done = UNDETERMINED
        else:
            stimulus_id, alignment = read
            source = [token.key for token in alignment.expected]
            produced = [params.p_normalise(word_text(word)) for word in words]
            echo = ngram_echo_fraction(source, produced, int(ngram_n))
            word_ids = tuple(word.id for word in words)
            findings.append(
                measured(
                    "verbatim_overlap_fraction", None, None, round(echo, 3), stimulus_id, *word_ids, n=int(ngram_n)
                )
            )
            if cut is not None and echo > float(cut):
                findings.append(
                    deviation(
                        "stimulus_mismatch",
                        response[0] if response is not None else None,
                        response[1] if response is not None else None,
                        *(span.id for span in runs),
                        reading=expectation.anti_pattern,
                        overlap=round(echo, 3),
                    )
                )
            if expectation.anti_pattern == "verbatim_source":
                # "Recall in your own words": semantic coverage is expected and verbatim
                # reproduction is the deviation, so coverage is what `done` reads.
                covered = content_coverage(source, produced)
                findings.append(
                    measured("source_content_coverage", None, None, round(covered, 3), stimulus_id, *word_ids)
                )
                coverage_min = points.point("coverage_min")
                done = UNDETERMINED if coverage_min is None else covered >= float(coverage_min)

    findings.extend(unviable_findings(expectation))
    findings.extend(declared_duration_count(store, expectation.declared_duration_s))
    findings.extend(_off_task(components, store, points))
    findings.extend(points.record())
    return Result(done, components, findings)


def _speech_item_list(
    expectation: Expectation, store: ProvStore, hint: AudioHints | None, params: BranchParams
) -> Result:
    """A task asking for a list of items: how many, and was one repeated where none may be?

    Proposes ``task_extent`` only. An item is one ``word`` entity with its own extent, so a per-item
    span would duplicate ground and carry no measurement of its own; the repetition finding is a
    deviation over that word's extent.

    Args:
        expectation: The row for this family.
        store: The provenance store.
        hint: What the recording was declared to contain.
        params: The operating points.

    Returns:
        Whether any item was produced, the span, and the findings.
    """
    points = params
    if expectation.repetition_from_category:
        # Eight of ten categories say "Do not repeat any item"; `Letters` and `Numbers` allow it. A
        # family-scoped rule inverts the instruction on those, so an unreadable category concludes
        # nothing rather than guessing which of the two rules applies.
        category = (hint.metadata or {}).get("category") if hint is not None else None
        if category is None:
            return Result(
                UNDETERMINED,
                [],
                [
                    unviable(
                        "repetition_rule",
                        "the category lives only in `instructions`; no grain above the recording carries it",
                    )
                ],
            )
        repetition_allowed = str(category) in REPETITION_ALLOWED_CATEGORIES
    else:
        repetition_allowed = bool(expectation.repetition_allowed)

    items = lexical_words(store)
    consensus_id = _consensus_id(store)
    components: list[Proposal] = []
    findings: list[Finding] = []
    first_seen: dict[str, float] = {}
    for word in items:
        key = params.p_normalise(word_text(word))
        start, end = word_extent(word)
        if key in first_seen and not repetition_allowed:
            findings.append(
                deviation("repeated_item", start, end, word.id, first_at=first_seen[key], text=word_text(word))
            )
        first_seen.setdefault(key, start)

    extent = hull([word_extent(word) for word in items])
    if extent is not None and extent[1] > extent[0] and consensus_id is not None:
        components.append(
            MINT(
                "task_extent",
                extent,
                consensus_id,
                *(word.id for word in items),
                items_n=len(items),
                repetition_allowed=repetition_allowed,
            )
        )
    findings.append(count("items", len(items), None, *(word.id for word in items)))
    findings.append(count("repetition_allowed", repetition_allowed, None))
    findings.extend(unviable_findings(expectation))
    findings.extend(declared_duration_count(store, expectation.declared_duration_s))
    findings.extend(_off_task(components, store, points))
    findings.extend(points.record())
    return Result(len(items) > 0, components, findings)


def align_speech(
    task_family: str,
    store: ProvStore,
    hint: AudioHints | None,
    params: BranchParams,
    *,
    reads: DdkReads = DdkReads(),
) -> Result:
    """Evaluate a declared speech task against what its own instruction asked for.

    Args:
        task_family: The declared family, which is a key of ``SPEECH_EXPECTATIONS``.
        store: The provenance store.
        hint: What the recording was declared to contain.
        params: The operating points.
        reads: The derivatives the syllable body measures over, loaded by :func:`speech`.

    Returns:
        Whether the expected patterns were found, the spans proposed, and the deviations.

    Raises:
        KeyError: If the family is not a SPEECH family, which is the caller owing detect_speech.
        NotImplementedError: If the row names a pattern this branch serves no body for.
    """
    expectation = SPEECH_EXPECTATIONS.get(task_family)
    if expectation is None:
        raise KeyError(f"{task_family} is not a SPEECH family; the caller owes detect_speech")
    if task_family in SYLLABLE_REPETITION:
        return align_ddk(expectation, store, params, reads=reads)
    if expectation.pattern is Pattern.ORDERED_TOKENS:
        return _speech_ordered(expectation, store, hint, params)
    if expectation.pattern is Pattern.FREE_RESPONSE:
        return _speech_free_response(expectation, store, hint, params)
    if expectation.pattern is Pattern.ITEM_LIST:
        return _speech_item_list(expectation, store, hint, params)
    raise NotImplementedError(f"{task_family}: SPEECH serves no body for {expectation.pattern}")


def detect_speech(store: ProvStore, params: BranchParams) -> Result:
    """Find lexical speech on a recording of another branch's kind, and evaluate no task.

    Needs no alignment at all: nothing lexical is expected of a breath, a cough or a held vowel, so
    every lexical word is the finding. This is the successor to both a lexical-intrusion detector
    and a count-in detector — on a ``prolonged-vowel`` recording it proposes a span over
    ``one two three``, and ``align_voice``, for which that family is in family, is what decides
    whether the prescribed count-in happened.

    Args:
        store: The provenance store.
        params: The operating points.

    Returns:
        A result whose ``done`` is ``UNDETERMINED``, one span per run of lexical words, and the
        findings.
    """
    points = params
    words = lexical_words(store)
    consensus_id = _consensus_id(store)
    gap = points.point("run_gap_max_s")
    components: list[Proposal] = []
    findings: list[Finding] = []
    # Not `merge` over the word extents: `merge` joins only what touches and ordinary speech has a
    # gap between every pair of words, so it would propose one span per word.
    runs = [] if gap is None or consensus_id is None else lexical_runs(words, float(gap))
    for index, extent in enumerate(runs):
        if extent[1] <= extent[0] or consensus_id is None:
            continue
        inside = [word for word in words if overlaps(word_extent(word), extent)]
        agreements = [
            float(word.attributes["agreement"]) for word in inside if word.attributes.get("agreement") is not None
        ]
        components.append(
            MINT(
                f"lexical_run_{index}",
                extent,
                consensus_id,
                *(word.id for word in inside),
                words_n=len(inside),
                text=" ".join(word_text(word) for word in inside),
                agreement=min(agreements) if agreements else None,
                evaluates_no_task=True,
            )
        )
    for span in live_entities(store, "span"):
        # Only somebody else's claim of speech: contesting this pass's own proposals would have the
        # branch argue with itself, and a span it has not yet written cannot be read here anyway.
        if span.attributes.get("family") != BRANCH_FAMILY[NODE] or span.extent is None:
            continue
        if _author_node(store, span.id) == NODE:
            continue
        if not any(overlaps(span.extent, extent) for extent in runs):
            findings.append(contest(span.id, span.extent, "speech", "no_consensus_word_inside"))
    findings.append(count("lexical_words", len(words), None, *(word.id for word in words)))
    findings.extend(points.record())
    return Result(UNDETERMINED, components, findings)


# --------------------------------------------------------------------- the node


def speech(  # noqa: C901 — the branch's nine steps, in design order
    store: ProvStore,
    source: str,
    config: TriageConfig,
    hint: AudioHints | None = None,
    *,
    run_dir: Path,
    enrollment: Optional[Enrollment] = None,
) -> BranchResult:
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
    params = branch_params(config)
    values = _required(params, enrollment)

    software = software_agent(store)
    reads = read_ddk(store, run_dir, source)
    plain_id, plain = resolve_stream(store, run_dir, source)
    recording_id, recording = resolve_stream(store, run_dir, ORIGINAL)
    sampling_rate = int(plain.sampling_rate)
    view: list[str] = []
    notes: list[str] = []

    # Step 1 — the consensus transcript is the transcript; this branch reads it and re-fuses nothing.
    consensus = find_measurement(store, "consensus_transcript")
    if consensus is None:
        raise LookupError("no consensus_transcript in the store; PREPROCESS has not run")
    words = [store.get_entity(word_id) for word_id in consensus.attributes["word_ids"]]
    words = [word for word in words if not store.is_invalidated(word.id)]
    lexical_index = [position for position, word in enumerate(words) if not word.attributes["bracketed"]]
    lexical = [words[position] for position in lexical_index]
    transcript_text = str(consensus.attributes["text"])
    source_names = [str(row["name"]) for row in consensus.attributes["sources"]]
    hypotheses = _hypotheses(store, source_names)

    transcript = store.activity(node=NODE, step="transcript", parameters={"read": "consensus_transcript"})
    store.was_associated_with(transcript, software)
    store.used(transcript, plain_id)
    store.used(transcript, consensus.id)
    for word in words:
        store.used(transcript, word.id)

    if hint is not None and hint.target_speaker is not None:
        notes.append(
            "this branch identifies the target by enrollment, not by hint.target_speaker, "
            "which was supplied and is not read"
        )

    # The expectation: one mode or the other, chosen from the declared family and nothing else. It
    # runs before this branch proposes anything, so neither mode reads a span this pass authored,
    # and it runs *before* the no-lexical exit below rather than after it: a syllable-repetition
    # recording with no lexical word is one `_speech_no_lexical` reads as conforming, and preempting
    # both modes would report the opposite.
    mode, declared_family = mode_of(NODE, store, hint)
    expect = store.activity(
        node=NODE, step="expect", parameters={"mode": mode, "task_family": declared_family, "stream": source}
    )
    store.was_associated_with(expect, software)
    store.used(expect, consensus.id)
    def _align(task_family: str, store: ProvStore, hint: AudioHints | None, params: BranchParams) -> Result:
        """Bind the loaded derivatives to the in-family mode."""
        return align_speech(task_family, store, hint, params, reads=reads)

    result = dispatch(NODE, store, params, hint, align=_align, detect=detect_speech)
    expectation_findings = [*result.deviations, *params.record()]
    syllable = syllable_detail(result) if declared_family in SYLLABLE_REPETITION and mode == "align" else {}
    if syllable:
        if reads.envelope is None:
            notes.append(NO_ENVELOPE)
        if reads.ppg is None:
            notes.append(NO_PPG)
    view.extend(propose_spans(store, expect, software, result.components))
    view.extend(write_findings(store, expect, software, expectation_findings, signal=source))

    if not lexical:
        # No lexical word is a reading, not a refusal: the mode above already said whether the
        # instruction's own pattern was found, and this branch reports that beside the spans it
        # proposed. VERDICT decides what it means for the file.
        notes.append("no consensus word; this branch measured no lexical subject")
        report_id, report = write_report(
            store,
            transcript,
            software,
            node=NODE,
            kind="speech",
            conformance=result.done,
            conformance_of=TASK,
            deviations=deviation_names(expectation_findings),
            unmeasured=tuple(params.missing),
            in_family=mode == "align",
            detail={
                "speaker_count": None,
                "diarization": "no_words",
                "expectation": {
                    "mode": mode,
                    "task_family": declared_family,
                    "spans_n": len(result.components),
                    "findings_n": len(expectation_findings),
                },
                **syllable,
                "words_n": 0,
                "speech_s": 0.0,
                "nontarget_speech_s": None,
                "pii": {"categories": [], "n": 0, "scanned_by": [], "failed": [], "missing": []},
                "second_diarizer": "not_consulted",
                "separation": "no_speaker_count",
                "notes": notes,
            },
        )
        view.append(report_id)
        return BranchResult(report=report, view=tuple(view), report_entity_id=report_id)

    single_source = [word.id for word in lexical if word.attributes["outcome"] == "insertion"]
    if single_source:
        notes.append(f"{len(single_source)} single-recognizer word(s) survive as fabrication candidates")

    # Step 2 — speech spans from the lexical words' timings, in memory until corroborated. A run
    # the consensus places at one instant is dropped rather than proposed: a span of no duration
    # names no region, and `propose_span` refuses one.
    word_extents = [word.extent or (0.0, 0.0) for word in lexical]
    all_grouped = group_extents_into_runs(word_extents)
    grouped = [
        (start, end, members)
        for start, end, members in all_grouped
        if clamp_extent((start, end), plain)[1] > clamp_extent((start, end), plain)[0]
    ]
    if len(grouped) < len(all_grouped):
        notes.append(f"{len(all_grouped) - len(grouped)} run(s) of words placed at one instant name no extent")
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
                notes.append(f"the classifier disconfirms span {start:.2f}-{end:.2f}s (speech coverage {coverage:.2f})")
        if stoi_floor is None or si_sdr_floor is None or "unmeasured" in squim:
            squim_vote = "not_evaluated"
        else:
            squim_ok = squim["stoi"] >= float(stoi_floor) and squim["si_sdr"] >= float(si_sdr_floor)
            squim_vote = "confirm" if squim_ok else "disconfirm"
        if {yamnet_vote, squim_vote} <= {"confirm", "disconfirm"} and squim_vote != yamnet_vote:
            notes.append(
                f"instruments disagree on span {start:.2f}-{end:.2f}s: classifier {yamnet_vote}, squim {squim_vote}"
            )
        corroboration.append({"yamnet_coverage": coverage, "yamnet_vote": yamnet_vote, "squim_vote": squim_vote})

    # Step 4 — the speakers are PREPROCESS's whole-file derivative, read rather than re-measured.
    # This branch's own pass ran pyannote over the lexical word hull, so a voice outside it — before
    # the participant started, after they stopped, or inside a pause — could not be seen at all.
    read = _read_diarization(store, run_dir, config)
    diarize_act = store.activity(
        node=NODE,
        step="diarize",
        parameters={"read": None if read is None else read.measurement, "rerun": False},
    )
    store.was_associated_with(diarize_act, software)

    speaker_count: int | None
    diarization_state: Any
    speaker_segments: list[tuple[str, str, tuple[float, float]]] = []  # (entity_id, speaker, extent)
    if read is None:
        speaker_count = None
        diarization_state = "derivative_absent"
        notes.append("no whole-file diarization derivative is in the store; this branch reads one and runs none")
    else:
        store.used(diarize_act, read.measurement_id)
        for start, end, label in read.segments:
            extent = clamp_extent((start, end), plain)
            if extent[1] <= extent[0]:
                continue
            speaker_id = store.entity(
                prov_type="speaker",
                extent=extent,
                attributes={"speaker": label, "diarizer": read.model, "signal": read.signal},
            )
            store.was_generated_by(speaker_id, diarize_act)
            store.was_attributed_to(speaker_id, software)
            store.was_derived_from(speaker_id, read.measurement_id)
            view.append(speaker_id)
            speaker_segments.append((speaker_id, label, extent))
        speaker_count = len({speaker for _, speaker, _ in speaker_segments})
        diarization_state = {
            "read": read.measurement,
            "signal": read.signal,
            "model": read.model,
            "exclusive": read.exclusive,
            "n_segments": len(speaker_segments),
        }
        if read.n_speakers != speaker_count:
            notes.append(f"the derivative records {read.n_speakers} speaker(s) and its segments carry {speaker_count}")

    second = config.get("speech.second_diarizer")
    second_record: Any = "not_consulted"
    if speaker_count is not None and speaker_count != 1:
        notes.append(f"speaker count {speaker_count} != 1")
        if second is not None and read is not None:
            second_model = _second_diarizer_model(str(second))
            second_agent = store.agent(
                agent_type="model", model_id=str(second_model.path_or_uri), commit_sha=second_model.commit_sha
            )
            second_act = store.activity(
                node=NODE,
                step="second_diarizer",
                parameters={"model": str(second_model.path_or_uri), "signal": read.signal},
            )
            store.was_associated_with(second_act, second_agent)
            store.used(second_act, read.measurement_id)
            # The corroborator reads the signal the derivative was measured on, not this branch's
            # own stream: two counts over two different signals corroborate nothing.
            second_stream_id, second_audio = resolve_stream(store, run_dir, read.signal)
            store.used(second_act, second_stream_id)
            [second_segments] = diarize_audios([second_audio], model=second_model)
            second_count = len({segment.speaker for segment in second_segments})
            second_record = {
                "model": str(second),
                "count": second_count,
                "agrees": second_count == speaker_count,
            }
            if second_count != speaker_count:
                notes.append(f"second diarizer counts {second_count} speakers against {speaker_count}")

    # Step 5 — separation: measurement-gated, and neither backend is selected by default. It runs
    # over the whole stream, because the diarization that gates it is now a whole-file reading.
    backend = config.get("speech.separation_backend")
    sound_class = config.get("speech.separation_sound_class")
    separation_state: Any
    separated: list[Audio] = []
    stream_span = (0.0, plain.waveform.shape[-1] / sampling_rate)
    if speaker_count is None:
        separation_state = "no_speaker_count"
    elif speaker_count < SEPARABLE_SOURCES:
        separation_state = "not_needed"
    elif backend is None:
        separation_state = "not_selected"
    elif speaker_count > SEPARABLE_SOURCES:
        separation_state = f"count_{speaker_count}_exceeds_backend"
        notes.append(f"separation cannot serve {speaker_count} speakers; the checkpoints separate exactly 2")
    elif str(backend) == UNASDIFF_BACKEND and sound_class is None:
        separation_state = "unconditioned_sound_slot_unavailable"
        notes.append(
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
            [plain],
            model=None,
            n_sources=SEPARABLE_SOURCES,
            mode="speech_sound",
            source_classes=[str(sound_class)],
        )[0]
    else:
        separator = _clearvoice_model(f"{CLEARVOICE_ORG}/{backend}")
        separation_state = {"backend": str(backend), "n_sources": SEPARABLE_SOURCES}
        separated = separate_audios([plain], model=separator, n_sources=SEPARABLE_SOURCES)[0]

    if separated:
        separate_act = store.activity(
            node=NODE, step="separate", parameters={"backend": str(backend), "extent": list(stream_span)}
        )
        store.was_associated_with(separate_act, software)
        store.used(separate_act, plain_id)
        for position, stream_audio in enumerate(separated):
            meta = dict(stream_audio.metadata.get("clearvoice") or {})
            index = int(meta.get("source_index", position))
            path, written = write_stream(stream_audio, run_dir, f"separated_{index}")
            stream_id = store.entity(
                prov_type="stream",
                extent=stream_span,
                attributes={
                    "name": f"separated_{index}",
                    **path_attributes(path, run_dir),
                    "sampling_rate": int(stream_audio.sampling_rate),
                    "channels": 1,
                    "source_index": index,
                    "backend": str(backend),
                    "input_norm_scalar": meta.get("input_norm_scalar"),
                    "separation_model": meta.get("model"),
                    "separation_commit": meta.get("commit"),
                    "write_gain": written.gain,
                },
            )
            store.was_generated_by(stream_id, separate_act)
            store.was_attributed_to(stream_id, software)
            store.was_derived_from(stream_id, plain_id)
            view.append(stream_id)

    # Step 6 — identify: words to speakers by timing, and the target by enrollment. Resolving
    # across speakers produces an aggregated span — one per contiguous run of words the diarization
    # gives to the same speaker — proposed in this branch's own family and leaving every word
    # untouched. There is no per-word assertion: `attribute` was never one of the contract's verbs.
    identify = store.activity(node=NODE, step="identify", parameters={})
    store.was_associated_with(identify, software)
    word_speakers: list[str | None] = []
    word_notes: list[str | None] = []
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
        word_notes.append(note)

    turn_proposals: list[Proposal] = []
    for position, run in enumerate(_speaker_runs(words, word_speakers, word_notes)):
        run_extent = clamp_extent(
            (
                min(word.extent or (0.0, 0.0) for word in run.words)[0],
                max((word.extent or (0.0, 0.0))[1] for word in run.words),
            ),
            plain,
        )
        if run_extent[1] <= run_extent[0]:
            notes.append(f"a run of {len(run.words)} word(s) attributed to {run.speaker} names no extent")
            continue
        sources = [
            segment_id for segment_id, speaker_label, extent in speaker_segments if _overlaps(run_extent, extent)
        ]
        turn_proposals.append(
            MINT(
                f"speaker_turn_{position}",
                run_extent,
                *(word.id for word in run.words),
                *sources,
                speaker=run.speaker,
                note=run.note,
                words_n=len(run.words),
                stream=plain_id,
            )
        )
    view.extend(propose_spans(store, identify, software, turn_proposals))

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

        model_id, revision, cut = (
            values["enrollment_model_id"],
            values["enrollment_revision"],
            values["target_match_cosine"],
        )
        unreadable = model_id is None or revision is None or cut is None
        refusal = None if unreadable else enrollment.refusal_against(model_id, revision)
        if unreadable:
            notes.append(
                "an enrollment was given and speech.enrollment_model or speech.target_match_cosine "
                "is unmeasured, so no probe was embedded"
            )
        elif refusal is not None:
            notes.append(refusal)
        elif speaker_segments:
            probe = _embedding_model(str(model_id), str(revision))
            labels: list[str] = []
            audios: list[Audio] = []
            for label in sorted({speaker for _, speaker, _ in speaker_segments}):
                # Only the audio this speaker holds alone: the shared derivative keeps pyannote's
                # overlapping view, so an overlapped region carries two voices and belongs to the
                # probe of neither.
                slices = [
                    plain.waveform[:, int(start * sampling_rate) : int(end * sampling_rate)]
                    for start, end in _exclusive_slices(label, speaker_segments)
                    if int(end * sampling_rate) > int(start * sampling_rate)
                ]
                if not slices:
                    notes.append(f"speaker {label} holds no audio alone, so no probe can be embedded for them")
                    continue
                labels.append(label)
                audios.append(Audio(waveform=torch.cat(slices, dim=1), sampling_rate=sampling_rate))
            embedding_agent = store.agent(
                agent_type="model", model_id=str(probe.path_or_uri), commit_sha=probe.commit_sha
            )
            store.was_associated_with(identify, embedding_agent)
            embeddings = extract_speaker_embeddings_from_audios(audios, model=probe)
            enrolled = torch.tensor(enrollment.vector, dtype=torch.float32)
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
                notes.append("an enrollment was given and no speaker matches it")

    # The span elements, proposed rather than written: a branch mints into its own family, naming
    # what each extent came from, and never edits a span another node proposed.
    run_proposals: list[Proposal] = []
    for position, ((start, end), (_, _, members)) in enumerate(zip(span_extents, grouped)):
        owners = {word_speakers[lexical_index[index]] for index in members}
        attributed_to = owners.pop() if len(owners) == 1 else None
        priors = [
            prior.id for prior in prior_spans if prior.extent is not None and _overlaps((start, end), prior.extent)
        ]
        run_proposals.append(
            MINT(
                f"speech_run_{position}",
                (start, end),
                consensus.id,
                *(lexical[index].id for index in members),
                *priors,
                words_n=len(members),
                attributed_to=attributed_to,
                nontarget=None if target_speaker is None or attributed_to is None else attributed_to != target_speaker,
                **corroboration[position],
            )
        )
    span_ids = propose_spans(store, corroborate, software, run_proposals)
    view.extend(span_ids)

    # Step 7 — PII: one scan over the consensus transcript and each recognizer's own transcript.
    haystacks: list[tuple[str, str, list[int]]] = [("consensus", transcript_text, list(range(len(words))))]
    for name in source_names:
        positions = [position for position, word in enumerate(words) if name in word.attributes["readings"]]
        haystacks.append((name, str(hypotheses[name].attributes["transcript"]), positions))
    pii_act = store.activity(
        node=NODE, step="pii", parameters={"text": ["consensus_transcript", *(f"asr:{name}" for name in source_names)]}
    )
    store.was_associated_with(pii_act, software)
    store.used(pii_act, consensus.id)
    for name in source_names:
        store.used(pii_act, hypotheses[name].id)
    raw_scans = scan_for_pii([text for _, text, _ in haystacks])
    scans: list[PiiScan] = raw_scans if isinstance(raw_scans, list) else [raw_scans]
    failures: dict[str, str] = {}
    scanned_by: set[str] = set()
    findings: list[dict[str, Any]] = []
    recorded: set[tuple[str, int, int]] = set()
    for (haystack, _, positions), scan in zip(haystacks, scans):
        failures.update(scan.failures)
        scanned_by.update(scan.detectors_used)
        tokens = [_reading(words[position], haystack) for position in positions]
        for finding in scan.spans:
            # Locate and mark every occurrence of this finding, not just its first (branch-speech.md §7).
            located = [(positions[first], positions[last]) for first, last in _locate(str(finding.text or ""), tokens)]
            if not located:
                notes.append(f"pii_unlocated ({finding.category})")
                occurrences = [(0, len(words) - 1)]
            else:
                occurrences = located
            for first, last in occurrences:
                if (str(finding.category), first, last) in recorded:
                    continue
                recorded.add((str(finding.category), first, last))
                covered = list(range(first, last + 1))
                # A finding nothing in the transcript places covers the whole of it: the redaction
                # that reads this must not be narrower than the text the detector was given.
                extent = _timings_hull(words, list(range(len(words)))) if not located else _timings_hull(words, covered)
                sources = sorted({str(name) for index in covered for name in words[index].attributes["sources"]})
                speakers = {word_speakers[index] for index in covered}
                resolved = len(speakers) == 1 and None not in speakers
                pii_id = store.entity(
                    prov_type="pii",
                    extent=extent,
                    attributes={
                        "category": finding.category,
                        "source": finding.source,
                        "haystack": haystack,
                        "sources": sources,
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
    notes.extend(_pii_notes(findings, failures, missing, target_speaker))
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

    # The report: the spans are in the store, the conformance is the mode's, and the observations
    # below are named rather than scored. No outcome is written here or anywhere in this branch.
    detail: dict[str, Any] = {
        "speaker_count": speaker_count,
        "diarization": diarization_state,
        "expectation": {
            "mode": mode,
            "task_family": declared_family,
            "spans_n": len(result.components),
            "findings_n": len(expectation_findings),
        },
        **syllable,
        "words_n": len(lexical),
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
        "notes": notes,
    }
    if target_speaker is not None:
        detail["target_speaker"] = target_speaker
    if enrollment_id is not None:
        detail["enrollment_id"] = enrollment_id
    report_id, report = write_report(
        store,
        proximity_act,
        software,
        node=NODE,
        kind="speech",
        conformance=result.done,
        conformance_of=TASK,
        deviations=deviation_names(expectation_findings),
        unmeasured=tuple(params.missing),
        in_family=mode == "align",
        detail=detail,
    )
    view.append(report_id)
    return BranchResult(report=report, view=tuple(view), report_entity_id=report_id)
