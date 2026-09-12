"""PREPROCESS — one conditioning pass, every shared derivative written to the store.

Every model that answers a whole-file question runs here: YAMNet, AST and HeAR alike. No later node
re-runs one. The recognizers, the aligner, SQUIM, level and the window classifiers read the plain
resampled signal; the envelope, spans, spectrograms, gammatone and the phonation pass read the
pre-emphasised one; ``disruptions_file`` reads the original recording. This node takes no pass/flag/
fail decision of its own — but it is not guaranteed to complete. Each block still runs in its own
try/except, and a block whose config value is unmeasured (a null default) or whose own upstream
prerequisite is missing from the store still records that derivative ``absent`` and moves on, exactly
as before. Any other exception is different: every remaining block still runs (this pass is meant to
be robust, not to abort early), but once the loop finishes, ``preprocess`` raises one exception
summarizing every such failure instead of returning normally — steps here do not get to silently
swallow a bug. ``run_triage`` treats that raise the same way it treats any other node erroring:
TAXONOMY, routing and every branch are skipped, and the file goes straight to VERDICT with the
failure as its reason. Every parameter's derivation is in ``data/config/default.yaml``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from importlib.metadata import version as _dist_version
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from senselab.audio.data_structures import Audio, AudioHints
from senselab.audio.tasks.classification.api import classify_audios
from senselab.audio.tasks.classification.label_scores import label_scores
from senselab.audio.tasks.classification.yamnet import SpanTooShortForYAMNet, span_yamnet_input
from senselab.audio.tasks.clearvoice import clearvoice_provenance
from senselab.audio.tasks.clipping.api import detect_clip_events
from senselab.audio.tasks.disruptions.api import detect_disruptions
from senselab.audio.tasks.envelope.api import (
    ButterworthSmoothing,
    MedianSmoothing,
    PercentileSmoothing,
    dynamic_range_normalize,
    global_floor_dbfs,
    hilbert_envelope_dbfs,
)
from senselab.audio.tasks.features_extraction.ppg import (
    PHONEME_LABELS,
    PPGS_SAMPLE_RATE,
    PpgsPosteriorgramUnavailable,
    extract_ppgs_from_audios,
    require_posteriorgram,
    to_frame_major_posteriorgram,
)
from senselab.audio.tasks.features_extraction.praat_parselmouth import (
    extract_praat_parselmouth_features_from_audios,
)
from senselab.audio.tasks.features_extraction.torchaudio import extract_spectrogram_from_audios
from senselab.audio.tasks.features_extraction.torchaudio_squim import (
    extract_objective_quality_features_from_audios,
)
from senselab.audio.tasks.gammatone.api import gammatone_filterbank
from senselab.audio.tasks.health_acoustics.api import detect_health_acoustic_events
from senselab.audio.tasks.health_acoustics.hear import (
    HEAR_MODEL_ID,
    HEAR_REVISION,
    HEAR_WINDOW_SECONDS,
    hear_window_extent,
    span_hear_input,
)
from senselab.audio.tasks.phonation.api import derive_f0_range, f0_track, formant_track
from senselab.audio.tasks.preprocessing.preprocessing import resample_audios
from senselab.audio.tasks.spans.api import (
    NoContrast,
    Span,
    group_extents_into_runs,
    propose_spans,
    rank_cut_level,
    segments_between_change_points,
)
from senselab.audio.tasks.spectral_continuity.api import spectral_continuity
from senselab.audio.tasks.speech_enhancement.api import enhance_audios
from senselab.audio.tasks.speech_enhancement.residual import band_energy_fractions, compute_residual
from senselab.audio.tasks.speech_to_text.api import transcribe_audios
from senselab.audio.workflows.audio_analysis.level import integrated_lufs
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.consensus import (
    ROUTINE,
    SOURCE_ORDER,
    SourceHypothesis,
    align_sources,
    render_transcript,
    vocabulary_key,
    word_attributes,
)
from senselab.audio.workflows.triage.label_membership import (
    LabelMembership,
    load_label_membership,
    optional_label_membership,
)
from senselab.audio.workflows.triage.nodes.common import (
    NodeResult,
    describe_exception,
    live_entities,
    path_attributes,
    resolve_stream,
    software_agent,
    write_stream,
    write_verdict,
)
from senselab.audio.workflows.triage.nodes.common import (
    write_measurement as _measurement,
)
from senselab.audio.workflows.triage.nodes.quality import (
    CLIP_AMPLITUDE_MEASUREMENT,
    CLIP_FAMILY,
    CLIP_LEVELS,
    UNCLIPPED_LOUDER_N,
    clip_spans,
)
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.data_structures import HFModel
from senselab.utils.prov_store import CHECKSUM_KEY, PATH_KEY, Entity, ProvStore, file_digest

NODE = "PREPROCESS"
CRISPERWHISPER_ID = "nyralabs/CrisperWhisper2.0_turbo"
QWEN_ID = "Qwen/Qwen3-ASR-1.7B"
QWEN_TIMESTAMP_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
AST_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
YAMNET_MODEL_URI = "https://tfhub.dev/google/yamnet/1"
FRCRN_ID = "alibabasglab/FRCRN_SE_16K"
PPGS_MODEL_ID = "interactiveaudiolab/ppgs"
PPG_MEASUREMENT = "ppg_posteriorgram"
PRAAT_MEASUREMENT = "praat_features"
PHONATION_TRACKS_MEASUREMENT = "phonation_tracks"


def _crisperwhisper_model() -> HFModel:
    """The CrisperWhisper model spec; its commit resolves at construction."""
    return HFModel(path_or_uri=CRISPERWHISPER_ID, revision="main")


def _qwen_model() -> HFModel:
    """The Qwen3-ASR model spec; its commit resolves at construction."""
    return HFModel(path_or_uri=QWEN_ID, revision="main")


def _ast_model() -> HFModel:
    """The AST model spec; its commit resolves at construction."""
    return HFModel(path_or_uri=AST_ID, revision="main")


def _frcrn_model() -> HFModel:
    """The FRCRN enhancement model spec; its commit resolves at construction."""
    return HFModel(path_or_uri=FRCRN_ID, revision="main")


@dataclass(frozen=True)
class PreprocessResult(NodeResult):
    """PREPROCESS's result.

    Attributes:
        absent: Names of derivatives that could not be computed and are absent from the store.
    """

    absent: tuple[str, ...]


def _bound_to_duration(start: float, end: float, duration_s: float) -> tuple[float, float] | None:
    """Bound one timed span by the duration of the stream it was timed against.

    Args:
        start: The span's start, in seconds.
        end: The span's end, in seconds.
        duration_s: The duration the stream decoded to, in seconds.

    Returns:
        The span with its end bound by ``duration_s``, or None when its start is at or past
        ``duration_s``, where it names no part of the stream at all.
    """
    if start >= duration_s:
        return None
    return start, min(end, duration_s)


def _merge_intervals(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """The disjoint, sorted intervals covering the union of ``spans``.

    Args:
        spans: ``(start, end)`` pairs, in any order; a pair with ``end <= start`` is dropped.

    Returns:
        The merged, non-overlapping intervals, sorted by start.
    """
    ordered = sorted((float(start), float(end)) for start, end in spans if end > start)
    if not ordered:
        return []
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _interval_overlap_fraction(start: float, end: float, regions: list[tuple[float, float]]) -> float:
    """The fraction of ``[start, end)`` covered by the union of ``regions``.

    Args:
        start: The window's start, in seconds.
        end: The window's end, in seconds.
        regions: Disjoint ``(start, end)`` intervals, as returned by :func:`_merge_intervals`.

    Returns:
        The covered fraction, in ``[0.0, 1.0]``. 0.0 when the window has zero or negative duration.
    """
    duration = end - start
    if duration <= 0:
        return 0.0
    covered = sum(max(0.0, min(end, r_end) - max(start, r_start)) for r_start, r_end in regions)
    return min(1.0, covered / duration)


def _pooled_label_scores(windows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-label mean and max score, and the window count each label appeared in.

    Args:
        windows: Classifier windows in the shape ``label_scores`` reads.

    Returns:
        ``{label: {"mean_score", "max_score", "n_windows"}}``, ranked by descending mean score.
    """
    pooled: dict[str, list[float]] = {}
    for window in windows:
        for pair in label_scores(window):
            for label, score in pair.items():
                pooled.setdefault(label, []).append(float(score))
    labels = {
        label: {"mean_score": sum(scores) / len(scores), "max_score": max(scores), "n_windows": len(scores)}
        for label, scores in pooled.items()
    }
    return dict(sorted(labels.items(), key=lambda item: -item[1]["mean_score"]))


def _confident_labels(window: dict[str, Any], membership: LabelMembership) -> dict[str, float]:
    """The labels this window carries, each with the score behind it.

    Args:
        window: A classifier window, in the shape ``label_scores`` reads.
        membership: The rule read from ``windows.<classifier>``.

    Returns:
        ``{label: score}`` over the members, in descending score order. Empty is a window nothing
        cleared, which is a different fact from a window that was never classified.
    """
    return membership.members(_raw_label_scores(window))


def _raw_label_scores(window: dict[str, Any]) -> dict[str, float]:
    """Return every valid classifier probability in its native ranked order.

    ``scores`` on a stored ``*_window`` measurement remains the thresholded decision subset.
    ``raw_scores`` is the complete model output for that same window, so presentation and later
    analysis do not reuse a decision threshold as a data-retention threshold.
    """
    return {label: score for pair in label_scores(window) for label, score in pair.items()}


def _classify_spans_in_batch(
    inputs: list[Audio],
    classify: Callable[[list[Audio]], list[list[dict[str, Any]]]],
) -> list[tuple[list[dict[str, Any]] | None, str | None]]:
    """Classify every span's input in one call, falling back to one call per span if that fails.

    Args:
        inputs: One prepared audio per span, in span order.
        classify: The batch call. Takes every input at once and returns one window list per input.

    Returns:
        One ``(windows, failure)`` pair per input, in the same order. Exactly one of the two is
        ``None``: ``windows`` is the classifier's output for that span, ``failure`` the exception
        type that stopped it.
    """
    if not inputs:
        return []
    try:
        batched = classify(inputs)
    except Exception as err:  # noqa: BLE001 — a batch that fails as a whole is retried per span below
        batch_failure = type(err).__name__
    else:
        if len(batched) == len(inputs):
            return [(windows, None) for windows in batched]
        batch_failure = f"batch returned {len(batched)} results for {len(inputs)} spans"
    results: list[tuple[list[dict[str, Any]] | None, str | None]] = []
    for item in inputs:
        try:
            results.append((classify([item])[0], None))
        except Exception as err:  # noqa: BLE001 — now each span's own failure is its own fact
            results.append((None, f"{type(err).__name__} (after batch failed: {batch_failure})"))
    return results


def _span_window_attributes(
    *,
    name: str,
    classifier: str,
    span_id: str,
    raw_window: dict[str, Any],
    membership: LabelMembership | None,
    extra: dict[str, Any],
) -> dict[str, Any]:
    """One per-span classifier window's attributes, labelled only when a membership rule exists.

    ``raw_scores`` is written whatever the configuration says, because the model ran and its output
    is a measurement. ``labels`` and ``scores`` are a decision taken over that measurement, so they
    appear only when a rule was configured, and ``labelled`` says which case a reader is looking at:
    absent labels with ``labelled`` False is "no threshold was set", an empty ``labels`` with
    ``labelled`` True is "nothing cleared the bar".

    Args:
        name: The measurement name, ``"span_hear"`` or ``"span_yamnet"``.
        classifier: The classifier's own name.
        span_id: The span this window was cut from.
        raw_window: The classifier's own output for the window.
        membership: The rule read from ``windows.<classifier>``, or None while its floor is null.
        extra: Attributes particular to one caller.

    Returns:
        The attribute mapping.
    """
    attributes: dict[str, Any] = {
        "name": name,
        "classifier": classifier,
        "signal": "plain",
        "span_id": span_id,
        "raw_scores": _raw_label_scores(raw_window),
        "default_threshold": membership.floor if membership is not None else None,
        "label_top_k": membership.top_k if membership is not None else None,
        "labelled": membership is not None,
        "isolated_span": True,
        **extra,
    }
    if membership is not None:
        members = _confident_labels(raw_window, membership)
        attributes["labels"] = list(members)
        attributes["scores"] = members
    return attributes


def _covering_window_attribution(
    span_extent: tuple[float, float], windows: list[dict[str, Any]]
) -> tuple[dict[str, float], int, float] | None:
    """The overlap-weighted mean of every whole-file window covering a short span.

    ``score[label] = sum(score_w * overlap_w) / sum(overlap_w)`` over the windows whose extent
    intersects the span at all, ``overlap_w`` the intersection in seconds.

    Args:
        span_extent: The span's own ``(start, end)`` in seconds.
        windows: Whole-file YAMNet windows, each carrying ``start``, ``end`` and ``label_scores``.

    Returns:
        ``(scores, covering_windows_n, covering_seconds)``, or None when no window overlaps the
        span at all.
    """
    start, end = span_extent
    covering_seconds = 0.0
    weighted: dict[str, float] = {}
    covering_windows_n = 0
    for window in windows:
        w_start, w_end = float(window["start"]), float(window["end"])
        overlap = min(end, w_end) - max(start, w_start)
        if overlap <= 0.0:
            continue
        covering_windows_n += 1
        covering_seconds += overlap
        for label, score in _raw_label_scores(window).items():
            weighted[label] = weighted.get(label, 0.0) + score * overlap
    if covering_windows_n == 0:
        return None
    scores = {label: value / covering_seconds for label, value in weighted.items()}
    return scores, covering_windows_n, covering_seconds


def _activity(store: ProvStore, step: str, parameters: dict[str, Any], reads: tuple[str, ...], agent_id: str) -> str:
    """One PREPROCESS sub-activity, associated with its agent and with its reads recorded.

    Args:
        store: The provenance store.
        step: The step's name.
        parameters: The values the step ran with.
        reads: Entities the step read.
        agent_id: The agent answerable for it.

    Returns:
        The activity's id.
    """
    activity_id = store.activity(node=NODE, step=step, parameters=parameters)
    store.was_associated_with(activity_id, agent_id)
    for entity_id in reads:
        store.used(activity_id, entity_id)
    return activity_id


@dataclass(frozen=True)
class ClipAmplitudes:
    """The amplitude facts a clip span's consistency is read against.

    Attributes:
        levels: Per span, in the order the extents were given, the peak absolute amplitude inside
            it; None where the extent names no sample of the signal.
        louder_counts: Per span, how many unclipped samples exceed that span's own level. Zero for
            a span whose level is None.
        unclipped_peak: The loudest unclipped sample's absolute amplitude, or None when the guarded
            spans leave no sample of the signal outside them.
        unclipped_peak_time_s: Where that sample sits, in seconds, or None.
        unclipped_samples_n: How many samples are unclipped evidence.
    """

    levels: tuple[float | None, ...]
    louder_counts: tuple[int, ...]
    unclipped_peak: float | None
    unclipped_peak_time_s: float | None
    unclipped_samples_n: int


def clip_amplitudes(audio: Audio, extents: Sequence[tuple[float, float]], *, guard_samples: int) -> ClipAmplitudes:
    """Measure each clip span's level and the loudest sample outside every one of them.

    Read from the channel-averaged signal ``detect_clip_events`` reads, at the sampling rate the
    extents are stated against. See ``specs/20260912-quality-clip-consistency/design.md``.

    Args:
        audio: The signal the extents were detected on.
        extents: Each clip span's ``(start, end)``, in seconds.
        guard_samples: How many samples each side of a span are excluded with it.

    Returns:
        The per-span levels and counts and the whole-file unclipped reading.
    """
    x = np.asarray(audio.waveform, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=0)
    magnitude = np.abs(x)
    sampling_rate = int(audio.sampling_rate)
    samples_n = int(magnitude.shape[0])
    ranges = [
        (
            max(0, min(int(round(start * sampling_rate)), samples_n)),
            max(0, min(int(round(end * sampling_rate)), samples_n)),
        )
        for start, end in extents
    ]
    excluded = np.zeros(samples_n, dtype=bool)
    for first, stop in ranges:
        excluded[max(0, first - guard_samples) : min(samples_n, stop + guard_samples)] = True
    unclipped = np.flatnonzero(~excluded)
    ordered = np.sort(magnitude[unclipped]) if unclipped.size else np.empty(0)
    levels = [float(magnitude[first:stop].max()) if stop > first else None for first, stop in ranges]
    return ClipAmplitudes(
        levels=tuple(levels),
        louder_counts=tuple(
            0 if level is None else int(ordered.size - np.searchsorted(ordered, level, side="right"))
            for level in levels
        ),
        unclipped_peak=float(ordered[-1]) if ordered.size else None,
        unclipped_peak_time_s=(
            float(unclipped[int(np.argmax(magnitude[unclipped]))] / sampling_rate) if unclipped.size else None
        ),
        unclipped_samples_n=int(unclipped.size),
    )


def write_clip_amplitudes(
    store: ProvStore,
    activity_id: str,
    agent_id: str,
    *,
    audio: Audio,
    spans: Sequence[Entity],
    signal: str,
    guard_samples: int,
    derived_from: tuple[str, ...] = (),
) -> str:
    """Measure the amplitudes of already-written clip spans and store them as one measurement.

    Every level QUALITY reads lives here, keyed by span id, beside the whole-file values. Nothing is
    stamped on the spans themselves, so this can be appended to a store whose spans already exist.

    Args:
        store: The provenance store.
        activity_id: The activity that measured the amplitudes.
        agent_id: The agent answerable for them.
        audio: The signal the spans were detected on.
        spans: The clip span entities, each carrying the extent to measure over.
        signal: The stream name the measurement is stated against.
        guard_samples: How many samples each side of a span are excluded from unclipped evidence.
        derived_from: Entities the measurement derives from, beside the spans themselves.

    Returns:
        The measurement entity's id.

    Raises:
        ValueError: If any span carries no extent, so there is nothing to measure it over.
    """
    extents: list[tuple[float, float]] = []
    for span in spans:
        if span.extent is None:
            raise ValueError(f"clip span {span.id} carries no extent; there are no samples to measure")
        extents.append((float(span.extent[0]), float(span.extent[1])))
    amplitudes = clip_amplitudes(audio, extents, guard_samples=guard_samples)
    span_ids = tuple(span.id for span in spans)
    return _measurement(
        store,
        activity_id,
        agent_id,
        name=CLIP_AMPLITUDE_MEASUREMENT,
        signal=signal,
        attributes={
            "unclipped_peak": amplitudes.unclipped_peak,
            "unclipped_peak_time_s": amplitudes.unclipped_peak_time_s,
            "unclipped_samples_n": amplitudes.unclipped_samples_n,
            "edge_guard_samples": int(guard_samples),
            "clip_spans_n": len(span_ids),
            CLIP_LEVELS: dict(zip(span_ids, amplitudes.levels)),
            UNCLIPPED_LOUDER_N: dict(zip(span_ids, amplitudes.louder_counts)),
        },
        derived_from=(*derived_from, *span_ids),
    )


def write_clip_spans(
    store: ProvStore,
    activity_id: str,
    agent_id: str,
    *,
    audio: Audio,
    extents: Sequence[tuple[float, float]],
    signal: str,
    guard_samples: int,
    derived_from: tuple[str, ...] = (),
) -> tuple[list[str], str]:
    """Write the clip spans and, beside them, the amplitude reading QUALITY checks them against.

    The two are written together because the waveform is in hand exactly once: QUALITY reads stored
    outputs and never decodes audio of its own. A span carries what was asserted — its family, its
    signal and its extent — and no amplitude; :func:`write_clip_amplitudes` holds those.

    Args:
        store: The provenance store.
        activity_id: The activity that detected the spans.
        agent_id: The agent answerable for them.
        audio: The signal the spans were detected on.
        extents: Each span's ``(start, end)``, in seconds, in time order.
        signal: The stream name the spans and the measurement are stated against.
        guard_samples: How many samples each side of a span are excluded from unclipped evidence.
        derived_from: Entities the spans and the measurement derive from.

    Returns:
        The span ids, in the order the extents were given, and the measurement's id.
    """
    spans: list[Entity] = []
    for extent in extents:
        span_id = store.entity(
            prov_type="span",
            extent=extent,
            attributes={"family": CLIP_FAMILY, "signal": signal},
        )
        store.was_generated_by(span_id, activity_id)
        store.was_attributed_to(span_id, agent_id)
        for source_id in derived_from:
            store.was_derived_from(span_id, source_id)
        spans.append(store.get_entity(span_id))
    measurement_id = write_clip_amplitudes(
        store,
        activity_id,
        agent_id,
        audio=audio,
        spans=spans,
        signal=signal,
        guard_samples=guard_samples,
        derived_from=derived_from,
    )
    return [span.id for span in spans], measurement_id


def ppg_model_agent(store: ProvStore) -> str:
    """The ppgs agent: a model whose checkpoint ships with the PyPI release, so no commit resolves.

    Args:
        store: The provenance store.

    Returns:
        The agent's id.
    """
    return store.agent(
        agent_type="model",
        model_id=PPGS_MODEL_ID,
        unresolved_reason="ppgs ships its checkpoint with the PyPI release; no commit exists to resolve",
    )


def ppg_input(store: ProvStore, run_dir: Path) -> tuple[str, Audio]:
    """The ``enhanced`` stream, conditioned as ppgs reads it: mono at :data:`PPGS_SAMPLE_RATE`.

    The one place the posteriorgram's input is prepared, so a batching caller and the block itself
    hand the model the same samples.

    Args:
        store: The provenance store, read for the live ``enhanced`` stream entity.
        run_dir: The run directory the stream's sidecar path is relative to.

    Returns:
        The stream entity's id and the conditioned audio.

    Raises:
        LookupError: If no live ``enhanced`` stream is in the store.
    """
    enhanced_id, audio = resolve_stream(store, run_dir, "enhanced")
    if audio.waveform.shape[0] != 1:
        audio = Audio(waveform=audio.waveform.mean(dim=0, keepdim=True), sampling_rate=audio.sampling_rate)
    if int(audio.sampling_rate) != PPGS_SAMPLE_RATE:
        [audio] = resample_audios([audio], PPGS_SAMPLE_RATE)
    return enhanced_id, audio


def write_ppg_posteriorgram(
    store: ProvStore,
    *,
    run_dir: Path,
    enhanced_id: str,
    audio: Audio,
    posteriorgram: torch.Tensor,
) -> str:
    """Persist one posteriorgram beside the run and register the measurement that names it.

    The array is written frame-major in float16 to ``derivatives/ppg_posteriorgram.npz`` with the
    phoneme order beside it; the entity carries the path, its SHA-256, its size and its shape, never
    the array.

    Args:
        store: The provenance store.
        run_dir: The run directory the sidecar is written under.
        enhanced_id: The ``enhanced`` stream entity the posteriorgram was measured on.
        audio: The conditioned audio the model read, for the frame rate.
        posteriorgram: The tensor ppgs returned, in any of its layouts.

    Returns:
        The measurement entity's id.
    """
    frame_major = to_frame_major_posteriorgram(posteriorgram)
    frames, phonemes = int(frame_major.shape[0]), int(frame_major.shape[1])
    duration_s = audio.waveform.shape[-1] / int(audio.sampling_rate)
    agent = ppg_model_agent(store)
    activity = _activity(store, "ppg_posteriorgram", {"model": PPGS_MODEL_ID}, (enhanced_id,), agent)
    relative = f"derivatives/{PPG_MEASUREMENT}.npz"
    np.savez(
        run_dir / relative,
        posteriorgram=frame_major.numpy().astype(np.float16),
        phonemes=np.asarray(PHONEME_LABELS, dtype=np.str_),
        seconds_per_frame=np.float64(duration_s / frames if frames else np.nan),
        duration_s=np.float64(duration_s),
        sampling_rate=np.int64(audio.sampling_rate),
    )
    return _measurement(
        store,
        activity,
        agent,
        name=PPG_MEASUREMENT,
        signal="enhanced",
        extent=(0.0, duration_s),
        attributes={
            **path_attributes(relative, run_dir),
            "frames": frames,
            "n_phonemes": phonemes,
            "phonemes": list(PHONEME_LABELS),
            "seconds_per_frame": duration_s / frames if frames else None,
            "sampling_rate": int(audio.sampling_rate),
            "dtype": "float16",
            "layout": "frames_by_phonemes",
        },
        derived_from=(enhanced_id,),
    )


def ppg_posteriorgram(store: ProvStore, *, run_dir: Path) -> str:
    """The phonetic posteriorgram over the ``enhanced`` stream, to one npz sidecar.

    Args:
        store: The provenance store.
        run_dir: The run directory the sidecar is written under.

    Returns:
        The measurement entity's id.

    Raises:
        LookupError: If no live ``enhanced`` stream is in the store.
        PpgsPosteriorgramUnavailable: If the model produced no posteriorgram for this recording.
    """
    enhanced_id, audio = ppg_input(store, run_dir)
    [result] = extract_ppgs_from_audios([audio])
    return write_ppg_posteriorgram(
        store,
        run_dir=run_dir,
        enhanced_id=enhanced_id,
        audio=audio,
        posteriorgram=require_posteriorgram(result),
    )


def _praat_scalar(value: Any) -> Any:  # noqa: ANN401 — Praat's own value, of whatever type it returned
    """One Praat feature as the store takes it: a finite float, or None for a non-finite one."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
        return value
    number = float(value)
    return number if np.isfinite(number) else None


def praat_features(store: ProvStore, config: TriageConfig, *, run_dir: Path) -> str:
    """Praat/Parselmouth's whole-file feature set over the ``enhanced`` stream.

    Every scalar is an attribute of the measurement: the set is forty numbers, small enough that a
    sidecar would only add an indirection. A non-finite scalar is recorded as null, JSON's only
    representation of a number Praat could not place.

    Args:
        store: The provenance store.
        config: The triage configuration.
        run_dir: The run directory the ``enhanced`` stream's path is relative to.

    Returns:
        The measurement entity's id.

    Raises:
        LookupError: If no live ``enhanced`` stream is in the store.
    """
    parameters: dict[str, Any] = {
        "time_step_s": float(config.require("praat_features.time_step_s")),
        "window_length_s": float(config.require("praat_features.window_length_s")),
    }
    enhanced_id, audio = resolve_stream(store, run_dir, "enhanced")
    software = software_agent(store)
    activity = _activity(store, PRAAT_MEASUREMENT, parameters, (enhanced_id,), software)
    [features] = extract_praat_parselmouth_features_from_audios(
        [audio], time_step=parameters["time_step_s"], window_length=parameters["window_length_s"]
    )
    scalars = {name: _praat_scalar(value) for name, value in sorted(features.items())}
    return _measurement(
        store,
        activity,
        software,
        name=PRAAT_MEASUREMENT,
        signal="enhanced",
        attributes={**parameters, "n_features": len(scalars), "features": scalars},
        derived_from=(enhanced_id,),
    )


def sharp_stream(store: ProvStore, run_dir: Path) -> tuple[str, Audio, str]:
    """The pre-emphasised stream when the run wrote one, else ``plain``, with the name it goes by.

    Args:
        store: The provenance store.
        run_dir: The run directory stream paths are relative to.

    Returns:
        The stream entity's id, its audio, and the signal name a measurement over it states.

    Raises:
        LookupError: If the store holds neither stream.
    """
    try:
        stream_id, audio = resolve_stream(store, run_dir, "preemphasised")
    except LookupError:
        stream_id, audio = resolve_stream(store, run_dir, "plain")
        return stream_id, audio, "plain"
    return stream_id, audio, "preemphasised"


def phonation_tracks(store: ProvStore, config: TriageConfig, *, run_dir: Path) -> str:
    """F0 over the pre-emphasised stream and the first four formants over ``plain``, per frame.

    Both streams are read back out of the store rather than taken from a conditioning pass's own
    arrays, so this pass and an extend pass over a finished run hand the trackers the same samples
    and write the same entity.

    Args:
        store: The provenance store, read for the live ``plain`` and pre-emphasised streams.
        config: The triage configuration.
        run_dir: The run directory the npz sidecar is written under.

    Returns:
        The measurement entity's id.

    Raises:
        LookupError: If no live ``plain`` stream is in the store.
        ValueError: If ``voice.f0_search_range_hz`` is unmeasured.
    """
    search = config.require("voice.f0_search_range_hz")
    plain_id, plain = resolve_stream(store, run_dir, "plain")
    sharp_id, sharp, sharp_signal = sharp_stream(store, run_dir)
    f0_min_hz, f0_max_hz = derive_f0_range(plain, search_floor_hz=float(search[0]), search_ceiling_hz=float(search[1]))
    parameters: dict[str, Any] = {
        "hop_s": float(config.require("phonation_spans.hop_s")),
        "max_formants": int(config.require("phonation_spans.max_formants")),
        "formant_max_hz": float(config.require("phonation_spans.formant_max_hz")),
        "formant_window_s": float(config.require("phonation_spans.formant_window_s")),
        "formant_preemphasis_hz": float(config.require("phonation_spans.formant_preemphasis_hz")),
        "f0_min_hz": f0_min_hz,
        "f0_max_hz": f0_max_hz,
    }
    times, f0_hz, strength = f0_track(sharp, f0_min_hz=f0_min_hz, f0_max_hz=f0_max_hz, hop_s=parameters["hop_s"])
    formants = formant_track(
        plain,
        hop_s=parameters["hop_s"],
        max_formants=parameters["max_formants"],
        formant_max_hz=parameters["formant_max_hz"],
        window_s=parameters["formant_window_s"],
        preemphasis_hz=parameters["formant_preemphasis_hz"],
    )
    software = software_agent(store)
    activity = _activity(store, PHONATION_TRACKS_MEASUREMENT, parameters, (sharp_id, plain_id), software)
    (run_dir / "derivatives").mkdir(parents=True, exist_ok=True)
    np.savez(
        run_dir / "derivatives" / f"{PHONATION_TRACKS_MEASUREMENT}.npz",
        times_s=times,
        f0_hz=f0_hz,
        strength=strength,
        formant_times_s=formants.times_s,
        f1_hz=formants.f_hz[0],
        f2_hz=formants.f_hz[1],
        f3_hz=formants.f_hz[2],
        f4_hz=formants.f_hz[3],
        f1_bw_hz=formants.bandwidth_hz[0],
        f2_bw_hz=formants.bandwidth_hz[1],
        f3_bw_hz=formants.bandwidth_hz[2],
        f4_bw_hz=formants.bandwidth_hz[3],
    )
    return _measurement(
        store,
        activity,
        software,
        name=PHONATION_TRACKS_MEASUREMENT,
        signal=sharp_signal,
        attributes={
            "hop_s": parameters["hop_s"],
            "f0_min_hz": f0_min_hz,
            "f0_max_hz": f0_max_hz,
            "f0_signal": sharp_signal,
            "formant_signal": "plain",
        },
        derived_from=(sharp_id, plain_id),
    )


def extend_clip_amplitudes(store: ProvStore, config: TriageConfig, *, run_dir: Path) -> str | None:
    """Measure the clip-amplitude measurement a finished run's clip spans have none of.

    The spans are read back rather than re-detected, so nothing else PREPROCESS produced is
    recomputed and no existing entity is touched. The source audio is the ``recording`` stream's
    own file, which ADMIT recorded the absolute path and digest of.

    Neither the activity nor the measurement carries a path, a timestamp or any other value that
    varies between two readings of the same file, so a second call writes records the store already
    holds and is a set-union no-op. A caller that skips a run already carrying the measurement is
    saving the decode, not buying the convergence.

    Args:
        store: The finished run's store, read under the run's own id.
        config: The triage configuration, read for ``quality.clip_edge_guard_samples``.
        run_dir: The run directory stream paths are relative to.

    Returns:
        The measurement entity's id, or None when the store carries no clip span to measure.

    Raises:
        LookupError: If no live ``recording`` stream is in the store.
        ValueError: If the recording's bytes no longer digest to what ADMIT recorded, so the
            amplitudes would not be those of the signal the spans were detected on.
    """
    signal = "recording"
    spans = clip_spans(store, signal)
    if not spans:
        return None
    guard_samples = int(config.require("quality.clip_edge_guard_samples"))
    stream_id, audio = resolve_stream(store, run_dir, signal)
    _check_recording_unchanged(store.get_entity(stream_id))
    parameters: dict[str, Any] = {
        "signal": signal,
        "clip_edge_guard_samples": guard_samples,
        "clip_spans_n": len(spans),
    }
    software = software_agent(store)
    activity = _activity(
        store, CLIP_AMPLITUDE_MEASUREMENT, parameters, (stream_id, *(span.id for span in spans)), software
    )
    return write_clip_amplitudes(
        store,
        activity,
        software,
        audio=audio,
        spans=spans,
        signal=signal,
        guard_samples=guard_samples,
        derived_from=(stream_id,),
    )


def _check_recording_unchanged(stream: Entity) -> None:
    """Refuse a recording whose bytes differ from the ones the clip spans were detected on.

    Args:
        stream: The ``recording`` stream entity, carrying ADMIT's ``path`` and digest.

    Raises:
        ValueError: If the file now digests to something else.
    """
    recorded = stream.attributes.get(CHECKSUM_KEY)
    path = stream.attributes.get(PATH_KEY)
    if not recorded or not path:
        return
    current, reason = file_digest(path)
    if current is None:
        raise ValueError(f"{path} cannot be digested ({reason}); its clip spans have no signal to be measured against")
    if current != recorded:
        raise ValueError(
            f"{path} now digests to {current}, not the {recorded} ADMIT read; its clip spans were "
            "detected on other bytes and amplitudes measured here would not belong to them"
        )


def preprocess(  # noqa: C901 — one block per derivative, each independent
    store: ProvStore,
    source: Audio,
    config: TriageConfig,
    hint: AudioHints | None = None,
    *,
    run_dir: Path,
) -> PreprocessResult:
    """Condition the admitted audio and write every derivative to the store.

    Args:
        store: The provenance store, already holding ADMIT's ``recording`` stream.
        source: The audio ADMIT returned, as supplied.
        config: The triage configuration.
        hint: Accepted for the shared node shape; not read.
        run_dir: Where the streams and sidecars are written.

    Returns:
        A pass verdict (PREPROCESS has no fail and no flag), the view over what was written, and the
        names of derivatives that are absent.

    Raises:
        RuntimeError: One or more blocks raised something other than a null-config ``ValueError`` or
            a missing-prerequisite ``LookupError`` — every block was still attempted, but this
            propagates instead of a normal return, once every block has had its turn.
    """
    software = software_agent(store)
    (run_dir / "streams").mkdir(parents=True, exist_ok=True)
    (run_dir / "derivatives").mkdir(parents=True, exist_ok=True)

    recording_ids = [
        e.id
        for e in store.entities("stream")
        if e.attributes.get("name") == "recording" and not store.is_invalidated(e.id)
    ]
    target_hz = int(config.require("resample.target_hz"))
    preemph_enabled = bool(config.require("preemphasis.enabled"))
    coefficient = float(config.require("preemphasis.coefficient"))

    condition = store.activity(
        node=NODE,
        step="condition",
        parameters={
            "target_hz": target_hz,
            "downmix": "mean",
            "preemphasis_enabled": preemph_enabled,
            "coefficient": coefficient,
        },
    )
    store.was_associated_with(condition, software)
    for recording_id in recording_ids:
        store.used(condition, recording_id)

    mono = Audio(waveform=source.waveform.mean(dim=0, keepdim=True), sampling_rate=source.sampling_rate)
    [plain] = resample_audios([mono], target_hz)
    peak = float(plain.waveform.abs().max())
    peak_scale = 1.0 if peak <= 1.0 else 1.0 / peak
    if peak_scale != 1.0:
        plain = Audio(waveform=plain.waveform * peak_scale, sampling_rate=target_hz)
    duration_s = plain.waveform.shape[-1] / target_hz
    plain_path, plain_report = write_stream(plain, run_dir, "plain")
    plain_id = store.entity(
        prov_type="stream",
        extent=(0.0, duration_s),
        attributes={
            "name": "plain",
            **path_attributes(plain_path, run_dir),
            "sampling_rate": target_hz,
            "channels": 1,
            "peak_scale": peak_scale,
            "write_gain": plain_report.gain,
        },
    )
    store.was_generated_by(plain_id, condition)
    store.was_attributed_to(plain_id, software)
    for recording_id in recording_ids:
        store.was_derived_from(plain_id, recording_id)

    if preemph_enabled:
        x = plain.waveform
        emphasised = torch.cat([x[:, :1], x[:, 1:] - coefficient * x[:, :-1]], dim=1)
        sharp = Audio(waveform=emphasised, sampling_rate=target_hz)
        sharp_path, sharp_report = write_stream(sharp, run_dir, "preemphasised")
        sharp_id = store.entity(
            prov_type="stream",
            extent=(0.0, duration_s),
            attributes={
                "name": "preemphasised",
                **path_attributes(sharp_path, run_dir),
                "sampling_rate": target_hz,
                "channels": 1,
                "coefficient": coefficient,
                "write_gain": sharp_report.gain,
            },
        )
        store.was_generated_by(sharp_id, condition)
        store.was_attributed_to(sharp_id, software)
        store.was_derived_from(sharp_id, plain_id)
        sharp_signal = "preemphasised"
    else:
        sharp, sharp_id, sharp_signal = plain, plain_id, "plain"

    absent: dict[str, str] = {}
    derivatives: dict[str, Any] = {}
    view: list[str] = [plain_id] + ([sharp_id] if sharp_id != plain_id else [])
    state: dict[str, Any] = {}

    def _step(step: str, parameters: dict[str, Any], reads: tuple[str, ...], agent_id: str) -> str:
        """One sub-activity, associated and with its reads recorded."""
        return _activity(store, step, parameters, reads, agent_id)

    def _clip_spans() -> None:
        """Clip-event spans over the ORIGINAL recording (ClipDaT), before any normalization runs.

        One ``clip_amplitude`` measurement beside the spans carries the loudest unclipped sample,
        where it sits, and each span's own peak keyed by its id. QUALITY's consistency check reads
        those numbers; measuring them here is what lets it read no audio.
        """
        if not recording_ids:
            raise LookupError("no recording stream in the store")
        parameters: dict[str, Any] = {
            "near_threshold": float(config.require("clipping.near_threshold")),
            "leniency_samples": int(config.require("clipping.leniency_samples")),
            "minimum_extreme": float(config.require("clipping.minimum_extreme")),
            "merge_gap_ms": float(config.require("clipping.merge_gap_ms")),
            "clip_edge_guard_samples": int(config.require("quality.clip_edge_guard_samples")),
        }
        activity = _step("clip_spans", parameters, (recording_ids[-1],), software)
        sr = int(source.sampling_rate)
        events = detect_clip_events(
            source,
            near_threshold=parameters["near_threshold"],
            leniency_samples=parameters["leniency_samples"],
            minimum_extreme=parameters["minimum_extreme"],
        )
        merge_gap_samples = parameters["merge_gap_ms"] * sr / 1000.0
        kept = sorted(events, key=lambda event: event.start_sample)
        merged: list[list[int]] = []
        for event in kept:
            if merged and event.start_sample - merged[-1][1] <= merge_gap_samples:
                merged[-1][1] = max(merged[-1][1], event.end_sample)
            else:
                merged.append([event.start_sample, event.end_sample])
        extents = [(start_sample / sr, (end_sample + 1) / sr) for start_sample, end_sample in merged]
        span_ids, amplitude_id = write_clip_spans(
            store,
            activity,
            software,
            audio=source,
            extents=extents,
            signal="recording",
            guard_samples=parameters["clip_edge_guard_samples"],
            derived_from=(recording_ids[-1],),
        )
        derivatives["clip_spans"] = span_ids
        derivatives[CLIP_AMPLITUDE_MEASUREMENT] = amplitude_id
        view.extend(span_ids)
        view.append(amplitude_id)
        state["clip_span_extents"] = extents

    def _envelope() -> None:
        """`energy_envelope` and its floor, over the pre-emphasised signal -- the primary span signal.

        Primary rather than the normalized signal: AGC is an optional, unvalidated step, and
        measured directly on real recordings it can compress
        local dynamic range enough that no peak clears any reasonable `k_db` at all (a five-breath
        recording's rise-over-floor topped out at 9 dB post-normalization against 23 dB pre-). The
        pre-emphasised envelope needs no optional step to exist, so `_spans` below always has a
        signal to propose from; `_normalized_envelope`'s spans, where available, only ever add
        candidates this pass missed, never replace it.
        """
        parameters = {
            "lowpass_hz": float(config.require("envelope.lowpass_hz")),
            "filter_order": int(config.require("envelope.filter_order")),
            "floor_percentile": float(config.require("floor.percentile")),
        }
        activity = _step("energy_envelope", parameters, (sharp_id,), software)
        envelope = hilbert_envelope_dbfs(
            sharp,
            smoothing=ButterworthSmoothing(cutoff_hz=parameters["lowpass_hz"], order=int(parameters["filter_order"])),
        )
        floor = global_floor_dbfs(envelope, percentile=parameters["floor_percentile"])
        np.savez(
            run_dir / "derivatives" / "energy_envelope.npz",
            envelope_dbfs=envelope,
            floor_dbfs=np.full_like(envelope, floor),
        )
        entity_id = _measurement(
            store,
            activity,
            software,
            name="energy_envelope",
            signal=sharp_signal,
            attributes={**path_attributes("derivatives/energy_envelope.npz", run_dir), "sampling_rate": target_hz},
            derived_from=(sharp_id,),
        )
        derivatives["energy_envelope"] = entity_id
        view.append(entity_id)
        state.update(envelope=envelope, floor=floor, envelope_id=entity_id)

    def _normalized_envelope() -> None:
        """The dynamically-normalized signal's own envelope and floor, over the pre-emphasised signal.

        Supplementary, not primary (see `_envelope` above): a quiet event AGC boosted to be
        detectable is a real candidate `_spans` should not miss, so its spans are added wherever they
        do not already overlap one the pre-emphasised pass found — never used to replace that pass,
        because AGC can also destroy contrast the raw signal still carries.

        The macro and micro envelopes inside ``dynamic_range_normalize`` are smoothed with
        :class:`~senselab.audio.tasks.envelope.api.MedianSmoothing`: a median cannot overshoot past a
        transient the way a resonant Butterworth does, which is what a word's onset is to this
        envelope. The final envelope this measurement stores goes one step further and uses
        :class:`~senselab.audio.tasks.envelope.api.PercentileSmoothing`: a median (its 50th
        percentile) still averages a real peak down toward the window's centre, and a plain rolling
        maximum overcorrects the other way — one loud sample pins the whole window to its height and
        holds it there after the sound has already ended, smearing a peak sideways in time. A high
        percentile (90th) sits close to the true peak without either failure, verified on real
        speech in this session's own diagnostics.

        The gain curve's own smoothing (``gain_smoothing``) is median too, not the Butterworth it
        shipped with: a resonant lowpass cannot settle to a short event's own correct gain within
        the event, so a ~150 ms burst spent most of its duration at several-hundred-percent excess
        gain rather than at a brief, edge-localized ringing artifact — raising the cutoff did not
        fix it, since the residual is the filter's own lag behind the macro-level transition
        upstream of it, not insufficient bandwidth. A short median settles to the correct plateau
        immediately, at the cost of a bounded transition rather than a ramped one.
        """
        parameters: dict[str, Any] = {
            "macro_smoothing_window_s": float(config.require("normalization.macro_smoothing.window_s")),
            "micro_smoothing_window_s": float(config.require("normalization.micro_smoothing.window_s")),
            "target_dr_db": float(config.require("normalization.target_dr_db")),
            "compression_ratio": float(config.require("normalization.compression_ratio")),
            "macro_target_dbfs": float(config.require("normalization.macro_target_dbfs")),
            "gain_smoothing_window_s": float(config.require("normalization.gain_smoothing.window_s")),
            "floor_dbfs": float(config.require("normalization.floor_dbfs")),
            "ceiling": float(config.require("normalization.ceiling")),
            "envelope_smoothing_window_s": float(config.require("normalization.envelope_smoothing.window_s")),
            "envelope_smoothing_percentile": float(config.require("normalization.envelope_smoothing.percentile")),
            "floor_percentile": float(config.require("floor.percentile")),
        }
        activity = _step("normalized_envelope", parameters, (sharp_id,), software)
        normalized = dynamic_range_normalize(
            sharp,
            macro_smoothing=MedianSmoothing(window_s=parameters["macro_smoothing_window_s"]),
            micro_smoothing=MedianSmoothing(window_s=parameters["micro_smoothing_window_s"]),
            target_dr_db=parameters["target_dr_db"],
            compression_ratio=parameters["compression_ratio"],
            macro_target_dbfs=parameters["macro_target_dbfs"],
            gain_smoothing=MedianSmoothing(window_s=parameters["gain_smoothing_window_s"]),
            floor_dbfs=parameters["floor_dbfs"],
            ceiling=parameters["ceiling"],
        )
        normalized_path, normalized_report = write_stream(normalized, run_dir, "normalized")
        normalized_id = store.entity(
            prov_type="stream",
            extent=(0.0, duration_s),
            attributes={
                "name": "normalized",
                **path_attributes(normalized_path, run_dir),
                "sampling_rate": target_hz,
                "channels": 1,
                "write_gain": normalized_report.gain,
            },
        )
        store.was_generated_by(normalized_id, activity)
        store.was_attributed_to(normalized_id, software)
        store.was_derived_from(normalized_id, sharp_id)
        envelope = hilbert_envelope_dbfs(
            normalized,
            smoothing=PercentileSmoothing(
                window_s=parameters["envelope_smoothing_window_s"],
                percentile=parameters["envelope_smoothing_percentile"],
            ),
        )
        floor = global_floor_dbfs(envelope, percentile=parameters["floor_percentile"])
        np.savez(
            run_dir / "derivatives" / "normalized_envelope.npz",
            envelope_dbfs=envelope,
            floor_dbfs=np.full_like(envelope, floor),
        )
        entity_id = _measurement(
            store,
            activity,
            software,
            name="normalized_envelope",
            signal="normalized",
            attributes={
                **path_attributes("derivatives/normalized_envelope.npz", run_dir),
                "sampling_rate": target_hz,
            },
            derived_from=(normalized_id,),
        )
        derivatives["normalized_envelope"] = entity_id
        view.append(normalized_id)
        view.append(entity_id)
        state.update(
            normalized_id=normalized_id,
            normalized_audio=normalized,
            normalized_envelope=envelope,
            normalized_floor=floor,
            normalized_envelope_id=entity_id,
        )

    def _spans() -> None:
        """Foreground-energy candidate spans, flagged where they overlap a clip.

        Four sources, applied in this order, each adding only spans over ground no earlier source
        covers: primary (the pre-emphasised amplitude envelope), supplementary (the normalized
        envelope, where that derivative exists), continuity (``continuity_trace``, where that
        derivative exists) and ASR (the consensus transcript's word timings, grouped where they
        touch). A later candidate overlapping a kept span is recorded on that
        span's ``corroborated_by`` attribute rather than dropped; the attribute is absent, not
        empty, on a span nothing corroborated, and creates no provenance edge because a
        corroborating candidate never becomes its own entity. ``contains_clip`` is the only flag
        this pass asserts.

        The measurements behind the source set and its ordering are in
        ``specs/20260904-preprocess-taxonomy-figure/design.md`` and
        ``specs/20260817-triage-workflow-dag/benchmarks/preprocess-params.md``.
        """
        if "envelope" not in state:
            raise LookupError("energy_envelope is absent")
        k_db = float(config.require("spans.k_db"))
        parameters: dict[str, Any] = {
            "k_db": k_db,
            "transition_window_ms": int(config.require("spans.transition_window_ms")),
            "min_duration_ms": int(config.require("spans.min_duration_ms")),
            "min_separation_ms": int(config.require("spans.min_separation_ms")),
            "continuity_cut_percentile": float(config.require("spans.continuity_cut_percentile")),
            "continuity_min_duration_ms": int(config.require("spans.continuity_min_duration_ms")),
        }
        reads = [state["envelope_id"]]
        if "normalized_envelope" in state:
            reads.append(state["normalized_envelope_id"])
        if "continuity_trace" in state:
            reads.append(derivatives["continuity_trace"])
        if "consensus" in state:
            reads.append(state["consensus_id"])
        activity = _step("spans", parameters, tuple(reads), software)

        def _propose(envelope: np.ndarray, floor: float, *, gate: float, min_duration_ms: int) -> list[Span]:
            proposed = propose_spans(
                envelope,
                floor,
                target_hz,
                k_db=gate,
                transition_window_ms=parameters["transition_window_ms"],
                min_duration_ms=min_duration_ms,
                min_separation_ms=parameters["min_separation_ms"],
            )
            return [] if isinstance(proposed, NoContrast) else proposed

        def _measure_fields(span: Span, measure: str) -> dict[str, Any]:
            if measure == "amplitude":
                return {"peak_over_floor_db": span.peak_over_floor_db, "k_db": k_db}
            if measure == "continuity":
                return {"continuity_cut_percentile": parameters["continuity_cut_percentile"]}
            return {}

        corroboration: dict[int, list[dict[str, Any]]] = {}

        def _novel(candidates: list[Span], covered: list[Span], *, measure: str, signal: str) -> list[Span]:
            """Candidates over new ground, kept; candidates over already-covered ground, recorded.

            A candidate that overlaps one or more spans in ``covered`` is not discarded: it is
            attached to every span it overlaps as a ``corroborated_by`` entry (keyed by object
            identity, since ``covered`` spans are freshly built each call and never value-equal by
            accident), so a later source agreeing with an earlier one stays visible rather than
            being silently thrown away. Only a candidate with zero overlap becomes a new span.
            """
            kept: list[Span] = []
            for candidate in candidates:
                overlapping = [o for o in covered if candidate.start < o.end and candidate.end > o.start]
                if overlapping:
                    record = {
                        "measure": measure,
                        "signal": signal,
                        "start": candidate.start,
                        "end": candidate.end,
                        "merged_proposals": candidate.merged_proposals,
                        **_measure_fields(candidate, measure),
                    }
                    for owner in overlapping:
                        corroboration.setdefault(id(owner), []).append(record)
                else:
                    kept.append(candidate)
            return kept

        primary = _propose(
            state["envelope"],
            state["floor"],
            gate=k_db,
            min_duration_ms=parameters["min_duration_ms"],
        )
        supplement: list[Span] = []
        if "normalized_envelope" in state:
            secondary = _propose(
                state["normalized_envelope"],
                state["normalized_floor"],
                gate=k_db,
                min_duration_ms=parameters["min_duration_ms"],
            )
            supplement = _novel(secondary, primary, measure="amplitude", signal="normalized")

        continuity: list[Span] = []
        if "continuity_trace" in state:
            continuity_candidates = segments_between_change_points(
                state["continuity_trace"],
                target_hz,
                cut_percentile=parameters["continuity_cut_percentile"],
                min_duration_ms=parameters["continuity_min_duration_ms"],
            )
            continuity = _novel(continuity_candidates, primary + supplement, measure="continuity", signal=sharp_signal)

        asr: list[Span] = []
        if "consensus" in state:
            word_extents = [word.extent for word in state["consensus"] if not word.bracketed]
            asr_candidates = [
                Span(start=start, end=end, peak_over_floor_db=float("nan"), merged_proposals=len(members))
                for start, end, members in group_extents_into_runs(word_extents)
            ]
            asr = _novel(asr_candidates, primary + supplement + continuity, measure="asr", signal="consensus")

        combined: list[tuple[Span, str, str, str]] = [
            (span, sharp_signal, state["envelope_id"], "amplitude") for span in primary
        ]
        combined += [(span, "normalized", state["normalized_envelope_id"], "amplitude") for span in supplement]
        combined += [(span, sharp_signal, state["envelope_id"], "continuity") for span in continuity]
        combined += [(span, "consensus", state["consensus_id"], "asr") for span in asr]

        if not combined:
            entity_id = _measurement(
                store,
                activity,
                software,
                name="spans_no_contrast",
                signal=sharp_signal,
                attributes={"k_db": k_db, "reason": "no peak rose above any gate on any signal or measure"},
                derived_from=(state["envelope_id"],),
            )
            derivatives["spans_no_contrast"] = entity_id
            view.append(entity_id)
            return
        clip_extents = state.get("clip_span_extents") or []
        span_ids: list[str] = []
        for span, signal_name, source_id, measure in combined:
            contains_clip = any(span.start < end and span.end > start for start, end in clip_extents)
            attributes: dict[str, Any] = {
                "signal": signal_name,
                "measure": measure,
                "merged_proposals": span.merged_proposals,
                "contains_clip": contains_clip,
                **_measure_fields(span, measure),
            }
            corroborated_by = corroboration.get(id(span))
            if corroborated_by:
                attributes["corroborated_by"] = corroborated_by
            span_id = store.entity(prov_type="span", extent=(span.start, span.end), attributes=attributes)
            store.was_generated_by(span_id, activity)
            store.was_attributed_to(span_id, software)
            store.was_derived_from(span_id, source_id)
            span_ids.append(span_id)
        # Everything the proposers left over is itself a span. The gaps are where the recording's
        # background lives, and the per-span classifiers run over whatever is in `span_ids`, so
        # naming them here is what gets the background measured at all rather than never looked at.
        min_gap_s = parameters["min_duration_ms"] / 1000.0
        covered = sorted((span.start, span.end) for span, _, _, _ in combined)
        gaps: list[tuple[float, float]] = []
        cursor = 0.0
        for start, end in covered:
            if start - cursor >= min_gap_s:
                gaps.append((cursor, start))
            cursor = max(cursor, end)
        if duration_s - cursor >= min_gap_s:
            gaps.append((cursor, duration_s))
        for start, end in gaps:
            contains_clip = any(start < clip_end and end > clip_start for clip_start, clip_end in clip_extents)
            gap_id = store.entity(
                prov_type="span",
                extent=(start, end),
                attributes={
                    "signal": sharp_signal,
                    "measure": "gap",
                    "merged_proposals": 0,
                    "contains_clip": contains_clip,
                },
            )
            store.was_generated_by(gap_id, activity)
            store.was_attributed_to(gap_id, software)
            store.was_derived_from(gap_id, state["envelope_id"])
            span_ids.append(gap_id)

        derivatives["spans"] = span_ids
        view.extend(span_ids)
        state["span_ids"] = span_ids

    def _scores(name: str, agent_id: str, activity_step: str, run: Callable[[], list[dict[str, Any]]]) -> None:
        """Run one classifier and store its verbatim windows; no threshold is read here (V3)."""
        activity = _step(activity_step, {}, (plain_id,), agent_id)
        windows = run()
        path = f"derivatives/{name}.json"
        (run_dir / path).write_text(json.dumps(windows))
        entity_id = _measurement(
            store,
            activity,
            agent_id,
            name=name,
            signal="plain",
            attributes={
                "classifier": name.removesuffix("_scores"),
                **path_attributes(path, run_dir),
                "n_windows": len(windows),
                "win_length_s": float(windows[0]["win_length"]) if windows else None,
                "hop_s": float(windows[0]["hop_length"]) if windows else None,
            },
            derived_from=(plain_id,),
        )
        derivatives[name] = entity_id
        view.append(entity_id)
        state[name] = windows
        state[name + "_id"] = entity_id

    def _windows(classifier: str) -> None:
        """Fold the thresholds over one classifier's stored scores into per-window label sets."""
        scores_name = f"{classifier}_scores"
        if scores_name not in state:
            raise LookupError(f"{scores_name} is absent")
        membership = load_label_membership(config, classifier)
        activity = _step(
            f"{classifier}_windows",
            {
                "default_threshold": membership.floor,
                "label_top_k": membership.top_k,
                "label_thresholds": dict(membership.label_floors),
            },
            (state[scores_name + "_id"],),
            software,
        )
        raw = state[scores_name]
        window_ids: list[str] = []
        windows_by_label: dict[str, list[str]] = {}
        fired: dict[str, float] = {}
        for raw_window in raw:
            raw_scores = _raw_label_scores(raw_window)
            members = _confident_labels(raw_window, membership)
            window_id = store.entity(
                prov_type="measurement",
                extent=(float(raw_window["start"]), float(raw_window["end"])),
                attributes={
                    "name": f"{classifier}_window",
                    "classifier": classifier,
                    "signal": "plain",
                    "labels": list(members),
                    "scores": members,
                    "raw_scores": raw_scores,
                },
            )
            store.was_generated_by(window_id, activity)
            store.was_attributed_to(window_id, software)
            store.was_derived_from(window_id, state[scores_name + "_id"])
            window_ids.append(window_id)
            for label in members:
                windows_by_label.setdefault(label, []).append(window_id)
                if label in membership.label_floors:
                    fired[label] = membership.label_floors[label]
        entity_id = _measurement(
            store,
            activity,
            software,
            name=f"{classifier}_windows",
            signal="plain",
            attributes={
                "classifier": classifier,
                "labels": sorted(windows_by_label),
                "windows_by_label": windows_by_label,
                "n_windows": len(raw),
                "win_length_s": float(raw[0]["win_length"]) if raw else None,
                "hop_s": float(raw[0]["hop_length"]) if raw else None,
                "default_threshold": membership.floor,
                "label_top_k": membership.top_k,
                "label_thresholds": fired,
            },
            derived_from=(state[scores_name + "_id"],),
        )
        derivatives[f"{classifier}_windows"] = entity_id
        view.append(entity_id)
        view.extend(window_ids)

    def _yamnet_scores() -> None:
        """YAMNet on its own native grid; `win_length`/`hop_length` are ignored by this backend."""
        _scores(
            "yamnet_scores",
            store.agent(
                agent_type="model",
                model_id=YAMNET_MODEL_URI,
                unresolved_reason="TF-Hub URL pin; no commit exists to resolve",
            ),
            "yamnet",
            lambda: classify_audios([plain], model="yamnet", top_k=int(config.require("yamnet.top_k")))[0],
        )

    def _ast_scores() -> None:
        """AST at the configured window and hop, over its whole label space (C1, C2)."""
        model = _ast_model()
        _scores(
            "ast_scores",
            store.agent(agent_type="model", model_id=str(model.path_or_uri), commit_sha=model.commit_sha),
            "ast",
            lambda: classify_audios(
                [plain],
                model=model,
                win_length=float(config.require("windows.ast.win_length_s")),
                hop_length=float(config.require("windows.ast.hop_s")),
                top_k=int(config.require("windows.ast.top_k")),
                function_to_apply="sigmoid",
            )[0],
        )

    def _hear_scores() -> None:
        """HeAR at its model-imposed 2 s window and the configured hop; `top_k=None` keeps all eight."""
        _scores(
            "hear_scores",
            store.agent(agent_type="model", model_id=HEAR_MODEL_ID, commit_sha=HEAR_REVISION),
            "hear",
            lambda: detect_health_acoustic_events(
                [plain], hop_length=float(config.require("windows.hear.hop_s")), top_k=None
            )[0],
        )

    def _silence() -> None:
        """The Silence projection of the stored YAMNet scores."""
        if "yamnet_scores" not in state:
            raise LookupError("yamnet_scores is absent")
        threshold = float(config.require("yamnet.silence_threshold"))
        activity = _step("silence", {"threshold": threshold}, (state["yamnet_scores_id"],), software)
        rows = []
        for window in state["yamnet_scores"]:
            score = 0.0
            for pair in label_scores(window):
                if "Silence" in pair:
                    score = float(pair["Silence"])
                    break
            rows.append(
                {"start": window["start"], "end": window["end"], "score": score, "is_silence": score >= threshold}
            )
        entity_id = _measurement(
            store,
            activity,
            software,
            name="silence",
            signal="plain",
            attributes={"threshold": threshold, "windows": rows},
            derived_from=(state["yamnet_scores_id"],),
        )
        derivatives["silence"] = entity_id
        view.append(entity_id)

    def _level() -> None:
        """File-level peak dBFS, RMS dBFS and LUFS on the plain signal."""
        activity = _step("level", {}, (plain_id,), software)
        x = plain.waveform.squeeze(0).numpy()
        peak_dbfs = float(20.0 * np.log10(max(float(np.abs(x).max()), 1e-12)))
        rms_dbfs = float(20.0 * np.log10(max(float(np.sqrt(np.mean(x**2))), 1e-12)))
        lufs = float(integrated_lufs(x, target_hz))
        entity_id = _measurement(
            store,
            activity,
            software,
            name="level",
            signal="plain",
            attributes={"peak_dbfs": peak_dbfs, "rms_dbfs": rms_dbfs, "lufs": lufs},
            derived_from=(plain_id,),
        )
        derivatives["level"] = entity_id
        view.append(entity_id)

    def _disruptions_file() -> None:
        """Clipping, dropouts, discontinuities, DC and ZCR over the whole ORIGINAL recording."""
        if not recording_ids:
            raise LookupError("no recording stream in the store")
        parameters: dict[str, Any] = {
            "clip_headroom": float(config.require("disruptions.clip_headroom")),
            "min_clip_run": int(config.require("disruptions.min_clip_run")),
            "min_dropout_ms": float(config.require("disruptions.min_dropout_ms")),
            "discontinuity_local_factor": float(config.require("disruptions.discontinuity_local_factor")),
            "discontinuity_window_ms": float(config.require("disruptions.discontinuity_window_ms")),
        }
        activity = _step("disruptions_file", parameters, (recording_ids[-1],), software)
        original_duration = source.waveform.shape[-1] / int(source.sampling_rate)
        found = detect_disruptions(source, 0.0, original_duration, **parameters)
        counts = {key: value for key, value in asdict(found).items() if key not in ("start", "end")}
        entity_id = _measurement(
            store,
            activity,
            software,
            name="disruptions_file",
            signal="recording",
            attributes={**counts, "sampling_rate": int(source.sampling_rate)},
            derived_from=(recording_ids[-1],),
        )
        derivatives["disruptions_file"] = entity_id
        view.append(entity_id)

    def _squim_for(name: str, span_ids: list[str]) -> None:
        """One objective-head measure assertion per span in ``span_ids``; refusals recorded, never padded."""
        if not span_ids:
            raise LookupError("spans are absent")
        agent = store.agent(
            agent_type="model",
            model_id="torchaudio SQUIM_OBJECTIVE",
            unresolved_reason=f"bundled torchaudio weights, version {_dist_version('torchaudio')}",
        )
        activity = _step(name, {}, tuple(span_ids), agent)
        assertion_ids: list[str] = []
        for span_id in span_ids:
            span = store.get_entity(span_id)
            start, end = span.extent or (0.0, 0.0)
            segment = Audio(
                waveform=plain.waveform[:, int(start * target_hz) : int(end * target_hz)],
                sampling_rate=target_hz,
            )
            try:
                [scores] = extract_objective_quality_features_from_audios([segment])
                attributes: dict[str, Any] = {
                    "verb": "measure",
                    "name": name,
                    "stoi": float(scores["stoi"]),
                    "pesq": float(scores["pesq"]),
                    "si_sdr": float(scores["si_sdr"]),
                }
            except Exception as err:  # noqa: BLE001 — a span SQUIM refuses is unmeasured, not padded
                attributes = {"verb": "measure", "name": name, "unmeasured": type(err).__name__}
            assertion_id = store.entity(prov_type="assertion", extent=span.extent, attributes=attributes)
            store.was_generated_by(assertion_id, activity)
            store.was_attributed_to(assertion_id, agent)
            store.was_derived_from(assertion_id, span_id)
            assertion_ids.append(assertion_id)
        derivatives[name] = assertion_ids
        view.extend(assertion_ids)

    def _squim() -> None:
        """SQUIM over the spans, on the plain signal -- recording quality, not any span's own gain."""
        _squim_for("squim", state.get("span_ids") or [])

    def _mark_unmeasured(activity: str, agent_id: str, span: Entity, name: str, reason: str) -> str:
        """Record one span as attempted but unmeasured, so its absence is a fact, not a silence."""
        assertion_id = store.entity(
            prov_type="assertion",
            extent=span.extent,
            attributes={"verb": "measure", "name": name, "unmeasured": reason},
        )
        store.was_generated_by(assertion_id, activity)
        store.was_attributed_to(assertion_id, agent_id)
        store.was_derived_from(assertion_id, span.id)
        return assertion_id

    def _span_hear() -> None:
        """Per-span HeAR re-evaluation of the spans, raw scores only — no labelling decision.

        Reuses the same per-span windowing AIRWAY uses for its own candidates (a short span is
        centred in a silent 2 s buffer; a long span is passed through and HeAR's native windows are
        placed back on the recording's own timeline) for the reason AIRWAY's own docstring already
        gives: a whole-file HeAR window is the wrong instrument for an isolated candidate. Runs over
        the plain signal, like ``_squim_for`` -- HeAR already carries its own internal preprocessing,
        so handing it our own dynamic-range-normalized signal on top is redundant at best and
        distorting at worst. No longer gated on normalization: this re-evaluation needs no normalized
        signal to exist at all.
        """
        span_ids = state.get("span_ids") or []
        if not span_ids:
            raise LookupError("spans are absent")
        agent = store.agent(agent_type="model", model_id=HEAR_MODEL_ID, commit_sha=HEAR_REVISION)
        activity = _step("span_hear", {}, tuple(span_ids), agent)
        membership = optional_label_membership(config, "hear")
        result_ids: list[str] = []
        prepared: list[Audio] = []
        prepared_for: list[str] = []
        for span_id in span_ids:
            span = store.get_entity(span_id)
            extent = span.extent or (0.0, 0.0)
            try:
                prepared.append(span_hear_input(plain, extent))
            except Exception as err:  # noqa: BLE001 — a span HeAR cannot be given is unmeasured, not padded
                result_ids.append(_mark_unmeasured(activity, agent, span, "span_hear", type(err).__name__))
                continue
            prepared_for.append(span_id)
        classified = _classify_spans_in_batch(
            prepared,
            lambda batch: detect_health_acoustic_events(batch, hop_length=HEAR_WINDOW_SECONDS, top_k=None),
        )
        for span_id, (raw_windows, failure) in zip(prepared_for, classified):
            span = store.get_entity(span_id)
            extent = span.extent or (0.0, 0.0)
            if failure is not None:
                result_ids.append(_mark_unmeasured(activity, agent, span, "span_hear", failure))
                continue
            if not raw_windows:
                result_ids.append(_mark_unmeasured(activity, agent, span, "span_hear", "no_native_window"))
                continue
            for raw_window in raw_windows:
                window_extent = hear_window_extent(extent, raw_window)
                window_id = store.entity(
                    prov_type="measurement",
                    extent=window_extent,
                    attributes=_span_window_attributes(
                        name="span_hear",
                        classifier="hear",
                        span_id=span_id,
                        raw_window=raw_window,
                        membership=membership,
                        extra={"input_window_s": HEAR_WINDOW_SECONDS},
                    ),
                )
                store.was_generated_by(window_id, activity)
                store.was_attributed_to(window_id, agent)
                store.was_derived_from(window_id, span_id)
                result_ids.append(window_id)
        derivatives["span_hear"] = result_ids
        view.extend(result_ids)

    def _span_yamnet() -> None:
        """Per-span YAMNet, raw scores only — no labelling decision.

        A span at least :data:`~senselab.audio.tasks.classification.yamnet.YAMNET_WINDOW_SECONDS`
        long is classified directly, letting YAMNet place its own native windows over it. A shorter
        span is never classified directly: its score is the overlap-weighted mean of the whole-file
        ``yamnet_scores`` windows that cover it (:func:`_covering_window_attribution`) -- those
        windows are real, unpadded audio, already computed earlier in this node. A short span with
        nothing covering it (only possible when the whole-file pass itself is absent) is recorded
        unmeasured rather than scored. Runs over the plain signal, like ``_squim_for`` -- YAMNet
        already carries its own internal preprocessing, so our own dynamic-range-normalized signal
        on top is redundant at best and distorting at worst. No longer gated on normalization: this
        re-evaluation needs no normalized signal to exist at all.
        """
        span_ids = state.get("span_ids") or []
        if not span_ids:
            raise LookupError("spans are absent")
        agent = store.agent(
            agent_type="model",
            model_id=YAMNET_MODEL_URI,
            unresolved_reason="TF-Hub URL pin; no commit exists to resolve",
        )
        activity = _step("span_yamnet", {}, tuple(span_ids), agent)
        membership = optional_label_membership(config, "yamnet")
        top_k = int(config.require("yamnet.top_k"))
        whole_file_windows: list[dict[str, Any]] | None = state.get("yamnet_scores")

        result_ids: list[str] = []
        native_ids: list[str] = []
        native_prepared: list[Audio] = []
        for span_id in span_ids:
            span = store.get_entity(span_id)
            extent = span.extent or (0.0, 0.0)
            try:
                audio = span_yamnet_input(plain, extent)
            except SpanTooShortForYAMNet:
                if whole_file_windows is None:
                    result_ids.append(_mark_unmeasured(activity, agent, span, "span_yamnet", "yamnet_scores_absent"))
                    continue
                attribution = _covering_window_attribution(extent, whole_file_windows)
                if attribution is None:
                    result_ids.append(_mark_unmeasured(activity, agent, span, "span_yamnet", "no_covering_window"))
                    continue
                scores, covering_windows_n, covering_seconds = attribution
                ranked = sorted(scores.items(), key=lambda item: -item[1])
                attributed_window: dict[str, Any] = {"label_scores": [{label: score} for label, score in ranked]}
                window_id = store.entity(
                    prov_type="measurement",
                    extent=extent,
                    attributes=_span_window_attributes(
                        name="span_yamnet",
                        classifier="yamnet",
                        span_id=span_id,
                        raw_window=attributed_window,
                        membership=membership,
                        extra={
                            "attribution": "covering_windows",
                            "covering_windows_n": covering_windows_n,
                            "covering_seconds": covering_seconds,
                        },
                    ),
                )
                store.was_generated_by(window_id, activity)
                store.was_attributed_to(window_id, agent)
                store.was_derived_from(window_id, span_id)
                result_ids.append(window_id)
                continue
            except Exception as err:  # noqa: BLE001 — a span YAMNet cannot be given is unmeasured, not padded
                result_ids.append(_mark_unmeasured(activity, agent, span, "span_yamnet", type(err).__name__))
                continue
            native_prepared.append(audio)
            native_ids.append(span_id)

        classified = _classify_spans_in_batch(
            native_prepared,
            lambda batch: classify_audios(batch, model="yamnet", top_k=top_k),
        )
        for span_id, (raw_windows, failure) in zip(native_ids, classified):
            span = store.get_entity(span_id)
            start, end = span.extent or (0.0, 0.0)
            if failure is not None:
                result_ids.append(_mark_unmeasured(activity, agent, span, "span_yamnet", failure))
                continue
            if not raw_windows:
                result_ids.append(_mark_unmeasured(activity, agent, span, "span_yamnet", "no_native_window"))
                continue
            for raw_window in raw_windows:
                window_extent = (
                    start + float(raw_window["start"]),
                    min(start + float(raw_window["end"]), end),
                )
                window_id = store.entity(
                    prov_type="measurement",
                    extent=window_extent,
                    attributes=_span_window_attributes(
                        name="span_yamnet",
                        classifier="yamnet",
                        span_id=span_id,
                        raw_window=raw_window,
                        membership=membership,
                        extra={"attribution": "native"},
                    ),
                )
                store.was_generated_by(window_id, activity)
                store.was_attributed_to(window_id, agent)
                store.was_derived_from(window_id, span_id)
                result_ids.append(window_id)
        derivatives["span_yamnet"] = result_ids
        view.extend(result_ids)

    def _asr(
        name: str,
        factory: Callable[[], HFModel],
        source_kind: str,
        timing_model: str | None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """One recognizer: its transcript and its own word list, retained as the consensus's evidence.

        No ``word`` entity is written here; PREPROCESS writes those once, over the consensus.
        """
        model = factory()
        agent = store.agent(agent_type="model", model_id=str(model.path_or_uri), commit_sha=model.commit_sha)
        activity = _step(
            name, {"model": str(model.path_or_uri), **{k: str(v) for k, v in kwargs.items()}}, (plain_id,), agent
        )
        [line] = transcribe_audios([plain], model=model, **kwargs)
        words: list[dict[str, Any]] = []
        untimed_chunks_n = 0
        out_of_bounds_chunks_n = 0
        for chunk in line.chunks or []:
            if chunk.start is None or chunk.end is None:
                untimed_chunks_n += 1
                continue
            span = _bound_to_duration(float(chunk.start), float(chunk.end), duration_s)
            if span is None:
                out_of_bounds_chunks_n += 1
                continue
            words.append({"text": chunk.text, "start": span[0], "end": span[1], "score": chunk.score})
        meta: dict[str, Any] = {
            "role": "asr_hypothesis",
            "source": name,
            "model_id": str(model.path_or_uri),
            "commit_sha": model.commit_sha,
            "transcript": line.text or "",
            "words": words,
            "n_words": len(words),
            "untimed_chunks_n": untimed_chunks_n,
            "out_of_bounds_chunks_n": out_of_bounds_chunks_n,
            "timestamp_source": source_kind,
            "timestamp_model": timing_model,
            "duration_s": duration_s,
        }
        entity_id = _measurement(
            store, activity, agent, name=name, signal="plain", attributes=meta, derived_from=(plain_id,)
        )
        derivatives[name] = entity_id
        view.append(entity_id)

    def _consensus() -> None:
        """The consensus stream over every ``asr_hypothesis`` measurement in the store, plus its words."""
        hypotheses: dict[str, Entity] = {}
        for measurement in live_entities(store, "measurement"):
            if measurement.attributes.get("role") == "asr_hypothesis":
                hypotheses[str(measurement.attributes["source"])] = measurement
        sources = [
            SourceHypothesis(
                name=source,
                words=tuple(
                    (str(word["text"]), float(word["start"]), float(word["end"]))
                    for word in measurement.attributes["words"]
                ),
                timestamp_source=str(measurement.attributes["timestamp_source"]),
                timestamp_model=measurement.attributes.get("timestamp_model"),
            )
            for source, measurement in hypotheses.items()
        ]
        onomatopoeic = {vocabulary_key(str(token)) for token in (config.get("words.onomatopoeic_tokens") or [])}
        consensus = align_sources(sources, onomatopoeic=onomatopoeic)
        names = [row["name"] for row in consensus.provenance["sources"]]
        measurement_ids = tuple(hypotheses[name].id for name in names)
        activity = _step(
            "consensus", {"routine": ROUTINE, "source_order": SOURCE_ORDER, "sources": names}, measurement_ids, software
        )
        word_ids: list[str] = []
        for word in consensus.words:
            word_id = store.entity(prov_type="word", extent=word.extent, attributes=word_attributes(word))
            store.was_generated_by(word_id, activity)
            store.was_attributed_to(word_id, software)
            for source in word.sources:
                store.was_derived_from(word_id, hypotheses[source].id)
            word_ids.append(word_id)
        rows: list[dict[str, Any]] = []
        for row in consensus.provenance["sources"]:
            hypothesis = hypotheses[row["name"]]
            generating = store.generated_by(hypothesis.id)
            agents = store.associated_with(generating) if generating is not None else []
            rows.append(
                {
                    **row,
                    "model_id": hypothesis.attributes.get("model_id"),
                    "commit_sha": hypothesis.attributes.get("commit_sha"),
                    "measurement_id": hypothesis.id,
                    "agent_id": agents[0] if agents else None,
                }
            )
        entity_id = _measurement(
            store,
            activity,
            software,
            name="consensus_transcript",
            signal="plain",
            attributes={
                "role": "consensus",
                **consensus.provenance,
                "sources": rows,
                "word_ids": word_ids,
                "text": render_transcript(consensus.words, strong=("", "")),
            },
            derived_from=measurement_ids,
        )
        derivatives["consensus_transcript"] = entity_id
        view.append(entity_id)
        view.extend(word_ids)
        state.update(consensus=consensus.words, consensus_id=entity_id)

    def _phonation_tracks() -> None:
        """F0 and formant tracks over the whole stream — measured once, localised nowhere.

        Sustained-phonation and glide span *detection* used to happen here; it has moved to
        TAXONOMY, which reads this measurement back and runs the same proposal functions over it.
        This block keeps only the part that is a measurement, and it reads its streams back out of
        the store, so a finished run can gain the tracks without conditioning being replayed.
        """
        entity_id = phonation_tracks(store, config, run_dir=run_dir)
        derivatives[PHONATION_TRACKS_MEASUREMENT] = entity_id
        view.append(entity_id)

    def _spectrogram(name: str, window_key: str) -> None:
        """One STFT power spectrogram, window and hop from the config, n_fft = win_length (decision N7).

        Stores the magnitude (``sqrt`` of this transform's power output) and the hop it was computed
        at into ``state`` under this block's own name, alongside writing the usual npz/measurement --
        so a later block (``_spans``'s continuity source, for the wideband case) can reuse the same
        array rather than recomputing an independent STFT with merely matching parameters. Harmless
        for the narrowband case, which nothing currently reads back out of ``state``.
        """
        window_ms = float(config.require(window_key))
        hop_ms = float(config.require("spectrogram.hop_ms"))
        win_length = int(target_hz * window_ms / 1000.0)
        hop_length = int(target_hz * hop_ms / 1000.0)
        parameters = {"win_length": win_length, "hop_length": hop_length, "n_fft": win_length}
        activity = _step(name, parameters, (sharp_id,), software)
        [result] = extract_spectrogram_from_audios(
            [sharp], n_fft=win_length, win_length=win_length, hop_length=hop_length
        )
        power = result["spectrogram"].numpy()
        np.savez(run_dir / "derivatives" / f"{name}.npz", spectrogram=power)
        entity_id = _measurement(
            store,
            activity,
            software,
            name=name,
            signal=sharp_signal,
            attributes={**path_attributes(f"derivatives/{name}.npz", run_dir), **parameters},
            derived_from=(sharp_id,),
        )
        derivatives[name] = entity_id
        view.append(entity_id)
        state[f"{name}_magnitude"] = np.sqrt(np.maximum(power, 0.0))
        state[f"{name}_hop_s"] = hop_ms / 1000.0

    def _continuity_trace() -> None:
        """The frame-to-frame spectral similarity over the narrowband spectrogram, to one npz sidecar.

        Written as its own derivative rather than left in ``_spans``' local scope, so a reader draws
        the trace these spans were proposed from. ``cut_level`` records where the rank cut fell.
        """
        if "spectrogram_narrowband_magnitude" not in state:
            raise LookupError("spectrogram_narrowband is absent")
        if "envelope" not in state:
            raise LookupError("energy_envelope is absent")
        parameters: dict[str, Any] = {
            "cut_percentile": float(config.require("spans.continuity_cut_percentile")),
            "envelope_lowpass_hz": float(config.require("envelope.lowpass_hz")),
            "envelope_filter_order": int(config.require("envelope.filter_order")),
        }
        reads = (derivatives["spectrogram_narrowband"], state["envelope_id"])
        activity = _step("continuity_trace", parameters, reads, software)
        trace = spectral_continuity(
            state["spectrogram_narrowband_magnitude"],
            hop_s=state["spectrogram_narrowband_hop_s"],
            sampling_rate=target_hz,
            n_samples=len(state["envelope"]),
            smoothing=ButterworthSmoothing(
                cutoff_hz=parameters["envelope_lowpass_hz"], order=parameters["envelope_filter_order"]
            ),
        )
        cut_level = rank_cut_level(trace, cut_percentile=parameters["cut_percentile"])
        np.savez(run_dir / "derivatives" / "continuity_trace.npz", continuity=trace)
        entity_id = _measurement(
            store,
            activity,
            software,
            name="continuity_trace",
            signal=sharp_signal,
            attributes={
                **path_attributes("derivatives/continuity_trace.npz", run_dir),
                "sampling_rate": target_hz,
                "cut_percentile": parameters["cut_percentile"],
                "cut_level": cut_level,
            },
            derived_from=reads,
        )
        derivatives["continuity_trace"] = entity_id
        view.append(entity_id)
        state["continuity_trace"] = trace

    def _gammatone() -> None:
        """The ERB-spaced filterbank energies, to one npz sidecar."""
        parameters: dict[str, Any] = {
            "n_channels": int(config.require("gammatone.n_channels")),
            "low_hz": float(config.require("gammatone.low_hz")),
            "high_hz": float(config.require("gammatone.high_hz")),
            "hop_s": float(config.require("gammatone.hop_s")),
        }
        activity = _step("gammatone", parameters, (sharp_id,), software)
        centre_frequencies, energy_db = gammatone_filterbank(
            sharp,
            n_channels=parameters["n_channels"],
            low_hz=parameters["low_hz"],
            high_hz=parameters["high_hz"],
            hop_s=parameters["hop_s"],
        )
        np.savez(
            run_dir / "derivatives" / "gammatone.npz",
            centre_frequencies_hz=centre_frequencies,
            energy_db=energy_db,
        )
        entity_id = _measurement(
            store,
            activity,
            software,
            name="gammatone",
            signal=sharp_signal,
            attributes={**path_attributes("derivatives/gammatone.npz", run_dir), "hop_s": parameters["hop_s"]},
            derived_from=(sharp_id,),
        )
        derivatives["gammatone"] = entity_id
        view.append(entity_id)

    def _ppg_posteriorgram() -> None:
        """The phonetic posteriorgram over the ``enhanced`` stream, to one npz sidecar.

        Reads the stream back out of the store rather than out of ``state``, so this pass and an
        extend pass over a finished run hand the model the same samples and write the same entity.
        """
        entity_id = ppg_posteriorgram(store, run_dir=run_dir)
        derivatives[PPG_MEASUREMENT] = entity_id
        view.append(entity_id)

    def _praat_features() -> None:
        """Praat's whole-file feature set over the ``enhanced`` stream, read back from the store."""
        entity_id = praat_features(store, config, run_dir=run_dir)
        derivatives[PRAAT_MEASUREMENT] = entity_id
        view.append(entity_id)

    def _speech_regions() -> tuple[list[tuple[float, float]], str]:
        """Speech regions for the residual's ``speech_overlap``, and which source produced them.

        The union of the consensus's lexical words' per-source ``timings`` when a consensus
        transcript exists (even one with no lexical words, which returns no regions but still
        names that source); the union of this pass's own amplitude-source spans otherwise.
        """
        words = state.get("consensus")
        if words is not None:
            spans = [span for word in words if not word.bracketed for span in word.timings.values()]
            return _merge_intervals(spans), "consensus_transcript"
        amplitude_spans: list[tuple[float, float]] = []
        for span_id in state.get("span_ids") or []:
            span = store.get_entity(span_id)
            if span.attributes.get("measure") == "amplitude" and span.extent is not None:
                amplitude_spans.append(span.extent)
        return _merge_intervals(amplitude_spans), "amplitude_spans"

    def _residual() -> None:
        """Background residual and its paired enhancement: ``plain`` and a lag-aligned FRCRN pass.

        Gated by ``residual.enabled``: FRCRN_SE_16K runs on ``plain``, is cross-correlation
        aligned to it (``residual.max_lag_ms`` search via
        :func:`~senselab.audio.tasks.speech_enhancement.residual.compute_residual`), and the aligned
        enhancement is written as its own stream (``enhanced``) alongside the least-squares
        gain-fitted ``residual = plain - g*enhanced``. Both are written whenever this block runs, so
        a downstream consumer -- or a person listening back -- can see what FRCRN actually produced,
        not only what was left after subtracting it.

        No energy-fraction gate decides whether either stream is written. This block measures; it
        does not judge whether a residual means "background" -- that question is answered by what
        ``enhanced`` and ``residual`` were each classified as (``enhanced_yamnet``/``ast``/``hear``
        and ``residual_yamnet``/``ast``/``hear`` below), not by a threshold on an energy ratio
        computed before any classifier has run. FRCRN itself being unavailable or raising is still a
        gate PREPROCESS survives (recorded as a ``ValueError`` the outer loop attributes to this
        block), because there is nothing to measure at all in that case.

        FRCRN is a speech-enhancement model; ``plain - g*enhanced`` reads as "the noise that was
        removed" only where there was speech for it to separate from noise. The residual is written
        regardless of whether speech is present -- this is a recorded precondition on what it
        *means*, not a gate on whether it is produced -- via ``speech_present``, ``n_consensus_words``
        and ``speech_coverage_fraction`` (the consensus's lexical words' own per-source timings,
        unioned, over the stream's duration). When no consensus transcript exists at all, the latter
        two are ``None`` (unmeasured) rather than 0, and ``speech_present`` is False.
        """
        if not bool(config.require("residual.enabled")):
            raise ValueError("residual.enabled is false")
        max_lag_ms = float(config.require("residual.max_lag_ms"))
        bands_hz = [(float(band[0]), float(band[1])) for band in config.require("residual.bands_hz")]
        model = _frcrn_model()
        agent = store.agent(agent_type="model", model_id=str(model.path_or_uri), commit_sha=model.commit_sha)
        parameters: dict[str, Any] = {
            "model": str(model.path_or_uri),
            "max_lag_ms": max_lag_ms,
            "bands_hz": bands_hz,
        }
        activity = _step("residual", parameters, (plain_id,), agent)
        try:
            [enhanced] = enhance_audios([plain], model=model)
        except Exception as err:  # noqa: BLE001 — FRCRN unavailable is a gate, not a crash
            raise ValueError(f"FRCRN enhancement unavailable: {describe_exception(err)}") from err
        if int(enhanced.sampling_rate) != target_hz:
            [enhanced] = resample_audios([enhanced], target_hz)
        provenance = clearvoice_provenance(enhanced)
        model_id, commit_sha = provenance if provenance is not None else (str(model.path_or_uri), model.commit_sha)

        ref = plain.waveform.squeeze(0).to(torch.float64).numpy()
        sig = enhanced.waveform.squeeze(0).to(torch.float64).numpy()
        computation = compute_residual(ref, sig, target_hz, max_lag_ms=max_lag_ms)

        enhanced_duration_s = computation.signal_aligned.shape[-1] / target_hz
        enhanced_audio = Audio(
            waveform=torch.from_numpy(computation.signal_aligned.astype(np.float32)).unsqueeze(0),
            sampling_rate=target_hz,
        )
        enhanced_path, enhanced_report = write_stream(enhanced_audio, run_dir, "enhanced")
        enhanced_id = store.entity(
            prov_type="stream",
            extent=(0.0, enhanced_duration_s),
            attributes={
                "name": "enhanced",
                **path_attributes(enhanced_path, run_dir),
                "sampling_rate": target_hz,
                "channels": 1,
                "write_gain": enhanced_report.gain,
            },
        )
        store.was_generated_by(enhanced_id, activity)
        store.was_attributed_to(enhanced_id, agent)
        store.was_derived_from(enhanced_id, plain_id)

        residual_duration_s = computation.residual.shape[-1] / target_hz
        residual_audio = Audio(
            waveform=torch.from_numpy(computation.residual.astype(np.float32)).unsqueeze(0), sampling_rate=target_hz
        )
        residual_path, residual_report = write_stream(residual_audio, run_dir, "residual")
        residual_id = store.entity(
            prov_type="stream",
            extent=(0.0, residual_duration_s),
            attributes={
                "name": "residual",
                **path_attributes(residual_path, run_dir),
                "sampling_rate": target_hz,
                "channels": 1,
                "write_gain": residual_report.gain,
            },
        )
        store.was_generated_by(residual_id, activity)
        store.was_attributed_to(residual_id, agent)
        store.was_derived_from(residual_id, plain_id)

        peak_dbfs = float(20.0 * np.log10(max(float(np.abs(computation.residual).max()), 1e-12)))
        rms_dbfs = float(20.0 * np.log10(max(float(np.sqrt(np.mean(np.square(computation.residual)))), 1e-12)))
        regions, region_source = _speech_regions()
        n_consensus_words: int | None
        if region_source == "consensus_transcript":
            n_consensus_words = sum(1 for word in state["consensus"] if not word.bracketed)
            speech_coverage_fraction: float | None = (
                sum(end - start for start, end in regions) / residual_duration_s if residual_duration_s > 0 else 0.0
            )
            speech_present = n_consensus_words > 0
        else:
            n_consensus_words = None
            speech_coverage_fraction = None
            speech_present = False
        entity_id = _measurement(
            store,
            activity,
            agent,
            name="residual",
            signal="residual",
            attributes={
                "model_id": model_id,
                "commit_sha": commit_sha,
                "lag_samples": computation.lag_samples,
                "lag_ms": computation.lag_ms,
                "gain": computation.gain,
                "gain_db": computation.gain_db,
                "energy_fraction": computation.residual_energy_fraction,
                "enhanced_energy_fraction": computation.signal_energy_fraction,
                "correlation_enhanced": computation.correlation_signal,
                "correlation_residual": computation.correlation_residual,
                "peak_dbfs": peak_dbfs,
                "rms_dbfs": rms_dbfs,
                "bands": band_energy_fractions(computation.residual, target_hz, bands_hz),
                "speech_present": speech_present,
                "n_consensus_words": n_consensus_words,
                "speech_coverage_fraction": speech_coverage_fraction,
            },
            derived_from=(residual_id, enhanced_id),
        )
        derivatives["residual"] = entity_id
        view.append(enhanced_id)
        view.append(residual_id)
        view.append(entity_id)
        state.update(
            enhanced_id=enhanced_id,
            enhanced_audio=enhanced_audio,
            residual_id=residual_id,
            residual_audio=residual_audio,
            residual_duration_s=residual_duration_s,
            speech_regions=regions,
            speech_overlap_source=region_source,
        )

    def _stream_classifier_scores(
        prefix: str, name: str, agent_id: str, activity_step: str, run: Callable[[], list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        """One classifier's whole-file windows over the ``enhanced`` or ``residual`` stream.

        Args:
            prefix: ``"enhanced"`` or ``"residual"`` -- which stream this classifier ran over.
            name: The measurement name, e.g. ``"residual_yamnet_scores"``.
            agent_id: The classifier's own agent.
            activity_step: The activity step name.
            run: The classify call, over ``state[f"{prefix}_audio"]``.
        """
        audio_key = f"{prefix}_audio"
        id_key = f"{prefix}_id"
        if audio_key not in state:
            raise LookupError(f"{prefix} is absent")
        activity = _step(activity_step, {}, (state[id_key],), agent_id)
        windows = run()
        regions = state["speech_regions"]
        for window in windows:
            window["speech_overlap"] = _interval_overlap_fraction(float(window["start"]), float(window["end"]), regions)
        path = f"derivatives/{name}.json"
        (run_dir / path).write_text(json.dumps(windows))
        entity_id = _measurement(
            store,
            activity,
            agent_id,
            name=name,
            signal=prefix,
            attributes={
                "classifier": name.removeprefix(f"{prefix}_").removesuffix("_scores"),
                **path_attributes(path, run_dir),
                "n_windows": len(windows),
                "win_length_s": float(windows[0]["win_length"]) if windows else None,
                "hop_s": float(windows[0]["hop_length"]) if windows else None,
                "speech_overlap_source": state["speech_overlap_source"],
            },
            derived_from=(state[id_key],),
        )
        derivatives[name] = entity_id
        view.append(entity_id)
        state[name] = windows
        state[f"{name}_id"] = entity_id
        return windows

    def _stream_classifier_summaries(
        prefix: str, classifier: str, agent_id: str, windows: list[dict[str, Any]], reads: tuple[str, ...]
    ) -> None:
        """Two label summaries over one stream's classifier windows: every window, speech-free ones.

        The second is exactly the windows whose ``speech_overlap`` is 0.0 -- no new threshold is
        introduced. Comparing the two is the check for the enhancement model's own speech-shaped
        artefact: a label whose peak collapses once the speech-overlapping windows are excluded was
        that artefact, not the background.
        """
        activity = _step(f"{prefix}_{classifier}_summary", {}, reads, agent_id)
        speech_free = [window for window in windows if float(window.get("speech_overlap", 0.0)) == 0.0]
        for suffix, subset in (("all", windows), ("speech_free", speech_free)):
            name = f"{prefix}_{classifier}_summary_{suffix}"
            entity_id = _measurement(
                store,
                activity,
                agent_id,
                name=name,
                signal=prefix,
                attributes={
                    "classifier": classifier,
                    "n_windows": len(subset),
                    "n_windows_total": len(windows),
                    "labels": _pooled_label_scores(subset),
                },
                derived_from=reads,
            )
            derivatives[name] = entity_id
            view.append(entity_id)

    def _stream_yamnet(prefix: str) -> None:
        """YAMNet's whole-file windows over one stream, plus its speech-overlap label summaries."""
        agent = store.agent(
            agent_type="model",
            model_id=YAMNET_MODEL_URI,
            unresolved_reason="TF-Hub URL pin; no commit exists to resolve",
        )
        name = f"{prefix}_yamnet_scores"
        windows = _stream_classifier_scores(
            prefix,
            name,
            agent,
            f"{prefix}_yamnet",
            lambda: classify_audios(
                [state[f"{prefix}_audio"]], model="yamnet", top_k=int(config.require("yamnet.top_k"))
            )[0],
        )
        _stream_classifier_summaries(prefix, "yamnet", agent, windows, (state[f"{name}_id"],))

    def _stream_ast(prefix: str) -> None:
        """AST's whole-file windows over one stream, plus its speech-overlap label summaries."""
        model = _ast_model()
        agent = store.agent(agent_type="model", model_id=str(model.path_or_uri), commit_sha=model.commit_sha)
        name = f"{prefix}_ast_scores"
        windows = _stream_classifier_scores(
            prefix,
            name,
            agent,
            f"{prefix}_ast",
            lambda: classify_audios(
                [state[f"{prefix}_audio"]],
                model=model,
                win_length=float(config.require("windows.ast.win_length_s")),
                hop_length=float(config.require("windows.ast.hop_s")),
                top_k=int(config.require("windows.ast.top_k")),
                function_to_apply="sigmoid",
            )[0],
        )
        _stream_classifier_summaries(prefix, "ast", agent, windows, (state[f"{name}_id"],))

    def _stream_hear(prefix: str) -> None:
        """HeAR's whole-file windows over one stream, plus its speech-overlap label summaries."""
        agent = store.agent(agent_type="model", model_id=HEAR_MODEL_ID, commit_sha=HEAR_REVISION)
        name = f"{prefix}_hear_scores"
        windows = _stream_classifier_scores(
            prefix,
            name,
            agent,
            f"{prefix}_hear",
            lambda: detect_health_acoustic_events(
                [state[f"{prefix}_audio"]], hop_length=float(config.require("windows.hear.hop_s")), top_k=None
            )[0],
        )
        _stream_classifier_summaries(prefix, "hear", agent, windows, (state[f"{name}_id"],))

    blocks: list[tuple[str, Callable[[], None]]] = [
        ("clip_spans", _clip_spans),
        ("yamnet_scores", _yamnet_scores),
        ("yamnet_windows", lambda: _windows("yamnet")),
        ("silence", _silence),
        ("ast_scores", _ast_scores),
        ("ast_windows", lambda: _windows("ast")),
        ("hear_scores", _hear_scores),
        ("hear_windows", lambda: _windows("hear")),
        ("level", _level),
        ("disruptions_file", _disruptions_file),
        ("asr_crisperwhisper", lambda: _asr("asr_crisperwhisper", _crisperwhisper_model, "native", None)),
        (
            "asr_qwen",
            lambda: _asr("asr_qwen", _qwen_model, "bundled_aligner", QWEN_TIMESTAMP_MODEL, return_timestamps=True),
        ),
        ("consensus_transcript", _consensus),
        (PHONATION_TRACKS_MEASUREMENT, _phonation_tracks),
        ("energy_envelope", _envelope),
        ("normalized_envelope", _normalized_envelope),
        ("spectrogram_wideband", lambda: _spectrogram("spectrogram_wideband", "spectrogram.wideband_window_ms")),
        (
            "spectrogram_narrowband",
            lambda: _spectrogram("spectrogram_narrowband", "spectrogram.narrowband_window_ms"),
        ),
        ("continuity_trace", _continuity_trace),
        ("spans", _spans),
        ("residual", _residual),
        ("enhanced_yamnet", lambda: _stream_yamnet("enhanced")),
        ("enhanced_ast", lambda: _stream_ast("enhanced")),
        ("enhanced_hear", lambda: _stream_hear("enhanced")),
        ("residual_yamnet", lambda: _stream_yamnet("residual")),
        ("residual_ast", lambda: _stream_ast("residual")),
        ("residual_hear", lambda: _stream_hear("residual")),
        ("squim", _squim),
        ("span_hear", _span_hear),
        ("span_yamnet", _span_yamnet),
        ("gammatone", _gammatone),
        (PPG_MEASUREMENT, _ppg_posteriorgram),
        (PRAAT_MEASUREMENT, _praat_features),
    ]
    hard_failures: list[tuple[str, str]] = []
    for name, block in blocks:
        try:
            block()
        except (ValueError, LookupError) as err:
            # A null/unmeasured config value, or a block's own missing upstream prerequisite —
            # both are cascading absences, not new failures.
            absent[name] = describe_exception(err)
        except Exception as err:  # noqa: BLE001 — classified below; every remaining block still runs
            absent[name] = describe_exception(err)
            hard_failures.append((name, describe_exception(err)))

    if hard_failures:
        summary = "; ".join(f"{name}: {message}" for name, message in hard_failures)
        raise RuntimeError(f"PREPROCESS: {len(hard_failures)} block(s) failed unexpectedly: {summary}")

    verdict_id, verdict = write_verdict(
        store,
        condition,
        software,
        node=NODE,
        outcome=Outcome.PASS,
        kind=None,
        why="conditioning complete; absent derivatives are listed",
        detail={"absent": dict(sorted(absent.items())), "derivatives": derivatives},
    )
    view.append(verdict_id)
    return PreprocessResult(
        verdict=verdict, view=tuple(view), verdict_entity_id=verdict_id, absent=tuple(sorted(absent))
    )
