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
from typing import Any, Callable

import numpy as np
import torch

from senselab.audio.data_structures import Audio, AudioHints
from senselab.audio.tasks.classification.api import classify_audios
from senselab.audio.tasks.classification.label_scores import label_scores
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
from senselab.audio.tasks.phonation.api import f0_track, formant_track
from senselab.audio.tasks.preprocessing.preprocessing import resample_audios
from senselab.audio.tasks.spans.api import (
    NoContrast,
    Span,
    group_extents_into_runs,
    propose_spans,
    segments_between_change_points,
)
from senselab.audio.tasks.spectral_continuity.api import spectral_continuity
from senselab.audio.tasks.speech_to_text.api import transcribe_audios
from senselab.audio.workflows.audio_analysis.asr import fuse_consensus_words
from senselab.audio.workflows.audio_analysis.level import integrated_lufs
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.common import (
    NodeResult,
    describe_exception,
    software_agent,
    write_verdict,
)
from senselab.audio.workflows.triage.nodes.common import (
    write_measurement as _measurement,
)
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.data_structures import HFModel
from senselab.utils.prov_store import Entity, ProvStore

NODE = "PREPROCESS"
CRISPERWHISPER_ID = "nyralabs/CrisperWhisper2.0_turbo"
QWEN_ID = "Qwen/Qwen3-ASR-1.7B"
QWEN_TIMESTAMP_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
AST_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
YAMNET_MODEL_URI = "https://tfhub.dev/google/yamnet/1"


def _crisperwhisper_model() -> HFModel:
    """The CrisperWhisper model spec; its commit resolves at construction."""
    return HFModel(path_or_uri=CRISPERWHISPER_ID, revision="main")


def _qwen_model() -> HFModel:
    """The Qwen3-ASR model spec; its commit resolves at construction."""
    return HFModel(path_or_uri=QWEN_ID, revision="main")


def _ast_model() -> HFModel:
    """The AST model spec; its commit resolves at construction."""
    return HFModel(path_or_uri=AST_ID, revision="main")


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


def _norm_token(token: str) -> str:
    """A token normalised for vocabulary matching: casefolded, edge punctuation stripped."""
    return token.casefold().strip(".,;:!?\"'()")


def _as_non_word(text: str, onomatopoeic: set[str]) -> tuple[str | None, str | None]:
    """The bracketed form of a non-lexical token, or ``(None, None)`` when the token is a word.

    Args:
        text: The token as the recognizer produced it.
        onomatopoeic: The normalised ``words.onomatopoeic_tokens`` vocabulary; empty while it is null.

    Returns:
        ``(bracketed, origin)`` where ``origin`` is ``"bracketed"`` or ``"onomatopoeic"``, or
        ``(None, None)``.
    """
    stripped = text.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        return stripped, "bracketed"
    normalised = _norm_token(stripped)
    if normalised and normalised in onomatopoeic:
        return f"[{normalised.upper()}]", "onomatopoeic"
    return None, None


def _confident_labels(
    window: dict[str, Any], default_threshold: float, label_thresholds: dict[str, float]
) -> dict[str, float]:
    """The labels this window is confident of, each with the score behind it.

    A label is a member iff its score clears its own threshold — ``label_thresholds[label]`` where
    one exists, ``default_threshold`` otherwise. The result may be empty, which is a window nobody's
    threshold cleared and is a different fact from a window that was never classified.

    Args:
        window: A classifier window, in the shape ``label_scores`` reads.
        default_threshold: The threshold for a label with no entry of its own.
        label_thresholds: Per-label thresholds.

    Returns:
        ``{label: score}`` over the members, in descending score order.
    """
    members: dict[str, float] = {}
    for pair in label_scores(window):
        for label, score in pair.items():
            if float(score) >= float(label_thresholds.get(label, default_threshold)):
                members[label] = float(score)
    return dict(sorted(members.items(), key=lambda item: -item[1]))


def _raw_label_scores(window: dict[str, Any]) -> dict[str, float]:
    """Return every valid classifier probability in its native ranked order.

    ``scores`` on a stored ``*_window`` measurement remains the thresholded decision subset.
    ``raw_scores`` is the complete model output for that same window, so presentation and later
    analysis do not reuse a decision threshold as a data-retention threshold.
    """
    return {label: score for pair in label_scores(window) for label, score in pair.items()}


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
    plain.save_to_file(str(run_dir / "streams" / "plain.wav"))
    plain_id = store.entity(
        prov_type="stream",
        extent=(0.0, duration_s),
        attributes={
            "name": "plain",
            "path": "streams/plain.wav",
            "sampling_rate": target_hz,
            "channels": 1,
            "peak_scale": peak_scale,
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
        sharp.save_to_file(str(run_dir / "streams" / "preemphasised.wav"))
        sharp_id = store.entity(
            prov_type="stream",
            extent=(0.0, duration_s),
            attributes={
                "name": "preemphasised",
                "path": "streams/preemphasised.wav",
                "sampling_rate": target_hz,
                "channels": 1,
                "coefficient": coefficient,
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
        activity_id = store.activity(node=NODE, step=step, parameters=parameters)
        store.was_associated_with(activity_id, agent_id)
        for entity_id in reads:
            store.used(activity_id, entity_id)
        return activity_id

    def _clip_spans() -> None:
        """Clip-event spans over the ORIGINAL recording (ClipDaT), before any normalization runs."""
        if not recording_ids:
            raise LookupError("no recording stream in the store")
        parameters: dict[str, Any] = {
            "near_threshold": float(config.require("clipping.near_threshold")),
            "leniency_samples": int(config.require("clipping.leniency_samples")),
            "minimum_extreme": float(config.require("clipping.minimum_extreme")),
            "merge_gap_ms": float(config.require("clipping.merge_gap_ms")),
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
        span_ids: list[str] = []
        extents: list[tuple[float, float]] = []
        for start_sample, end_sample in merged:
            extent = (start_sample / sr, (end_sample + 1) / sr)
            span_id = store.entity(
                prov_type="span",
                extent=extent,
                attributes={"family": "clip", "signal": "recording"},
            )
            store.was_generated_by(span_id, activity)
            store.was_attributed_to(span_id, software)
            store.was_derived_from(span_id, recording_ids[-1])
            span_ids.append(span_id)
            extents.append(extent)
        derivatives["clip_spans"] = span_ids
        view.extend(span_ids)
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
            attributes={"path": "derivatives/energy_envelope.npz", "sampling_rate": target_hz},
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
        normalized.save_to_file(str(run_dir / "streams" / "normalized.wav"))
        normalized_id = store.entity(
            prov_type="stream",
            extent=(0.0, duration_s),
            attributes={
                "name": "normalized",
                "path": "streams/normalized.wav",
                "sampling_rate": target_hz,
                "channels": 1,
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
            attributes={"path": "derivatives/normalized_envelope.npz", "sampling_rate": target_hz},
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

        Primary: proposed from the pre-emphasised amplitude envelope, always available.
        Supplementary: proposed from the normalized amplitude envelope where that derivative exists,
        kept only where it does not already overlap a primary span. Continuity: proposed from
        spectral_continuity's frame-to-frame spectral similarity over the wideband spectrogram
        block's own magnitude output (reused directly, not recomputed at merely-matching parameters
        -- see `_spectrogram`) -- wideband rather than narrowband for its shorter analysis window
        (spectrogram.wideband_window_ms against .narrowband_window_ms, same shared hop_ms), giving
        continuity finer temporal resolution to place an onset/offset transition against, at the cost
        of frequency resolution continuity's own frame-to-frame comparison does not need as much as a
        band-energy measurement would. Smoothed with the same `ButterworthSmoothing(cutoff_hz=
        envelope.lowpass_hz, order=envelope.filter_order)` the primary amplitude envelope itself
        uses -- not a separately-chosen scheme -- so both measures agree on what counts as the same
        continuous event at the same time-constant before either is compared against or deduplicated
        with the other. Kept only where it does not already overlap a primary or
        supplementary span -- a sustained tonal/harmonic production (a glide, a held vowel) can hold
        a stable spectral shape well before its amplitude clears any gate. Absent whenever
        `spectrogram_wideband` itself is, the same "each block is independent" contract every
        other dependency in this node already follows. ASR: the consensus transcript's own word
        timings (`_consensus`), grouped into runs by `speech.word_gap_ms` via
        :func:`~senselab.audio.tasks.spans.api.group_extents_into_runs` -- the same grouping SPEECH
        uses for its own word-timing spans, shared rather than duplicated. No threshold or floor: a
        recognizer transcribing a stretch as speech is itself the evidence, not a measure needing a
        gate. Absent whenever `consensus` itself is (both recognizers failed).

        Priority order is primary, then supplementary, then continuity, then ASR: each source only
        ever adds a *new* span over ground no earlier source already covers -- one span entity per
        covered stretch, never two near-duplicates for the same region. A later source's candidate
        that overlaps an already-kept span is not silently dropped, though: it is recorded on that
        span's own `corroborated_by` attribute (one entry per overlapping candidate, carrying that
        candidate's own `measure`/`signal`/extent/its own contrast field), so agreement between
        independent sources over the same stretch stays visible and queryable rather than being
        implied only by which single source happened to propose first. `corroborated_by` is absent
        (not an empty list) on a span nothing else corroborated. This is attribute metadata only --
        no `was_derived_from` provenance edge is created for a corroborating candidate, since it
        never becomes its own store entity. A span carries no notion of which downstream branch it
        is "for"; `contains_clip` is the only flag this pass asserts.
        """
        if "envelope" not in state:
            raise LookupError("energy_envelope is absent")
        k_db = float(config.require("spans.k_db"))
        parameters: dict[str, Any] = {
            "k_db": k_db,
            "floor_margin_db": float(config.require("spans.floor_margin_db")),
            "transition_window_ms": int(config.require("spans.transition_window_ms")),
            "min_duration_ms": int(config.require("spans.min_duration_ms")),
            "min_separation_ms": int(config.require("spans.min_separation_ms")),
            "continuity_cut_percentile": float(config.require("spans.continuity_cut_percentile")),
            "continuity_min_duration_ms": int(config.require("spans.continuity_min_duration_ms")),
            "envelope_lowpass_hz": float(config.require("envelope.lowpass_hz")),
            "envelope_filter_order": int(config.require("envelope.filter_order")),
        }
        word_gap_ms = config.get("speech.word_gap_ms")
        if word_gap_ms is not None:
            parameters["word_gap_ms"] = float(word_gap_ms)
        reads = [state["envelope_id"]]
        if "normalized_envelope" in state:
            reads.append(state["normalized_envelope_id"])
        if "spectrogram_narrowband_magnitude" in state:
            reads.append(derivatives.get("spectrogram_narrowband", state["envelope_id"]))
        if "consensus" in state and word_gap_ms is not None:
            reads.append(state["consensus_id"])
        activity = _step("spans", parameters, tuple(reads), software)

        def _propose(
            envelope: np.ndarray, floor: float, *, gate: float, margin: float, min_duration_ms: int
        ) -> list[Span]:
            proposed = propose_spans(
                envelope,
                floor,
                target_hz,
                k_db=gate,
                floor_margin_db=margin,
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
            return {"word_gap_ms": parameters["word_gap_ms"]}

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
            margin=parameters["floor_margin_db"],
            min_duration_ms=parameters["min_duration_ms"],
        )
        supplement: list[Span] = []
        if "normalized_envelope" in state:
            secondary = _propose(
                state["normalized_envelope"],
                state["normalized_floor"],
                gate=k_db,
                margin=parameters["floor_margin_db"],
                min_duration_ms=parameters["min_duration_ms"],
            )
            supplement = _novel(secondary, primary, measure="amplitude", signal="normalized")

        continuity: list[Span] = []
        if "spectrogram_narrowband_magnitude" in state:
            continuity_trace = spectral_continuity(
                state["spectrogram_narrowband_magnitude"],
                hop_s=state["spectrogram_narrowband_hop_s"],
                sampling_rate=target_hz,
                n_samples=len(state["envelope"]),
                smoothing=ButterworthSmoothing(
                    cutoff_hz=parameters["envelope_lowpass_hz"], order=parameters["envelope_filter_order"]
                ),
            )
            continuity_candidates = segments_between_change_points(
                continuity_trace,
                target_hz,
                cut_percentile=parameters["continuity_cut_percentile"],
                min_duration_ms=parameters["continuity_min_duration_ms"],
            )
            continuity = _novel(continuity_candidates, primary + supplement, measure="continuity", signal=sharp_signal)

        asr: list[Span] = []
        if "consensus" in state and word_gap_ms is not None:
            word_extents = [(float(word["start"]), float(word["end"])) for word in state["consensus"]]
            asr_candidates = [
                Span(start=start, end=end, peak_over_floor_db=float("nan"), merged_proposals=len(members))
                for start, end, members in group_extents_into_runs(word_extents, parameters["word_gap_ms"])
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
                "path": path,
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
        default_threshold = float(config.require(f"windows.{classifier}.default_threshold"))
        label_thresholds = {
            str(label): float(value)
            for label, value in config.require(f"windows.{classifier}.label_thresholds").items()
        }
        activity = _step(
            f"{classifier}_windows",
            {"default_threshold": default_threshold, "label_thresholds": label_thresholds},
            (state[scores_name + "_id"],),
            software,
        )
        raw = state[scores_name]
        window_ids: list[str] = []
        windows_by_label: dict[str, list[str]] = {}
        fired: dict[str, float] = {}
        for raw_window in raw:
            raw_scores = _raw_label_scores(raw_window)
            members = _confident_labels(raw_window, default_threshold, label_thresholds)
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
                if label in label_thresholds:
                    fired[label] = label_thresholds[label]
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
                "default_threshold": default_threshold,
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
        default_threshold = float(config.require("windows.hear.default_threshold"))
        label_thresholds = {
            str(label): float(value) for label, value in (config.get("windows.hear.label_thresholds") or {}).items()
        }
        result_ids: list[str] = []
        for span_id in span_ids:
            span = store.get_entity(span_id)
            extent = span.extent or (0.0, 0.0)
            try:
                input_audio = span_hear_input(plain, extent)
                raw_windows = detect_health_acoustic_events([input_audio], hop_length=HEAR_WINDOW_SECONDS, top_k=None)[
                    0
                ]
            except Exception as err:  # noqa: BLE001 — a span HeAR refuses is unmeasured, not padded
                result_ids.append(_mark_unmeasured(activity, agent, span, "span_hear", type(err).__name__))
                continue
            if not raw_windows:
                result_ids.append(_mark_unmeasured(activity, agent, span, "span_hear", "no_native_window"))
                continue
            for raw_window in raw_windows:
                members = _confident_labels(raw_window, default_threshold, label_thresholds)
                raw_scores = _raw_label_scores(raw_window)
                window_extent = hear_window_extent(extent, raw_window)
                window_id = store.entity(
                    prov_type="measurement",
                    extent=window_extent,
                    attributes={
                        "name": "span_hear",
                        "classifier": "hear",
                        "signal": "plain",
                        "span_id": span_id,
                        "labels": list(members),
                        "scores": members,
                        "raw_scores": raw_scores,
                        "input_window_s": HEAR_WINDOW_SECONDS,
                        "isolated_span": True,
                    },
                )
                store.was_generated_by(window_id, activity)
                store.was_attributed_to(window_id, agent)
                store.was_derived_from(window_id, span_id)
                result_ids.append(window_id)
        derivatives["span_hear"] = result_ids
        view.extend(result_ids)

    def _span_yamnet() -> None:
        """Per-span YAMNet over the spans, raw scores only — no labelling decision.

        Unlike HeAR, YAMNet has no fixed-window constraint (its own native ~0.96 s grid runs over
        whatever length it is given), so a span is classified directly rather than buffered. Runs
        over the plain signal, like ``_squim_for`` -- YAMNet already carries its own internal
        preprocessing, so our own dynamic-range-normalized signal on top is redundant at best and
        distorting at worst. No longer gated on normalization: this re-evaluation needs no
        normalized signal to exist at all.
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
        default_threshold = float(config.require("windows.yamnet.default_threshold"))
        label_thresholds = {
            str(label): float(value) for label, value in (config.get("windows.yamnet.label_thresholds") or {}).items()
        }
        top_k = int(config.require("yamnet.top_k"))
        result_ids: list[str] = []
        for span_id in span_ids:
            span = store.get_entity(span_id)
            start, end = span.extent or (0.0, 0.0)
            segment = Audio(
                waveform=plain.waveform[:, int(start * target_hz) : int(end * target_hz)],
                sampling_rate=target_hz,
            )
            try:
                raw_windows = classify_audios([segment], model="yamnet", top_k=top_k)[0]
            except Exception as err:  # noqa: BLE001 — a span YAMNet refuses is unmeasured, not padded
                result_ids.append(_mark_unmeasured(activity, agent, span, "span_yamnet", type(err).__name__))
                continue
            if not raw_windows:
                result_ids.append(_mark_unmeasured(activity, agent, span, "span_yamnet", "no_native_window"))
                continue
            for raw_window in raw_windows:
                members = _confident_labels(raw_window, default_threshold, label_thresholds)
                raw_scores = _raw_label_scores(raw_window)
                window_extent = (start + float(raw_window["start"]), start + float(raw_window["end"]))
                window_id = store.entity(
                    prov_type="measurement",
                    extent=window_extent,
                    attributes={
                        "name": "span_yamnet",
                        "classifier": "yamnet",
                        "signal": "plain",
                        "span_id": span_id,
                        "labels": list(members),
                        "scores": members,
                        "raw_scores": raw_scores,
                        "isolated_span": True,
                    },
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

        No ``word`` entity is written here. PREPROCESS writes those once, over the consensus, so a
        consumer never has to disambiguate two populations of ``word`` by generating activity.
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
            "recognizer": str(model.path_or_uri),
            "transcript": line.text or "",
            "words": words,
            "untimed_chunks_n": untimed_chunks_n,
            "out_of_bounds_chunks_n": out_of_bounds_chunks_n,
            "timestamp_source": source_kind,
        }
        if timing_model is not None:
            meta["timestamp_model"] = timing_model
        entity_id = _measurement(
            store, activity, agent, name=name, signal="plain", attributes=meta, derived_from=(plain_id,)
        )
        derivatives[name] = entity_id
        view.append(entity_id)
        state[name] = line
        state[name + "_id"] = entity_id

    def _consensus() -> None:
        """The consensus over both recognizers, by the audio-analysis routine, plus its word entities."""
        if "asr_crisperwhisper" not in state or "asr_qwen" not in state:
            raise LookupError("both recognizers are needed")
        activity = _step(
            "consensus",
            {
                "systems": [CRISPERWHISPER_ID, QWEN_ID],
                "routine": "fuse_consensus_words",
                "timing_authority": "consensus_asr",
            },
            (state["asr_crisperwhisper_id"], state["asr_qwen_id"]),
            software,
        )
        fused, provenance = fuse_consensus_words(
            {CRISPERWHISPER_ID: state["asr_crisperwhisper"], QWEN_ID: state["asr_qwen"]}
        )
        if not provenance:
            provenance = {"operator": "consensus_words/resample", "sources": [], "n_words": 0}
        onomatopoeic = {_norm_token(str(token)) for token in (config.get("words.onomatopoeic_tokens") or [])}
        word_ids: list[str] = []
        event_ids: list[str] = []
        kept: list[dict[str, Any]] = []
        for entry in fused:
            span = _bound_to_duration(float(entry["start"]), float(entry["end"]), duration_s)
            if span is None:
                continue
            text = str(entry.get("text") or "")
            recognizers = [str(s) for s in (entry.get("sources") or [])]
            bracketed, origin = _as_non_word(text, onomatopoeic)
            if bracketed is not None:
                event_id = store.entity(
                    prov_type="event",
                    extent=span,
                    attributes={
                        "bracketed": bracketed,
                        "raw": text,
                        "origin": origin,
                        "recognizers": recognizers,
                    },
                )
                store.was_generated_by(event_id, activity)
                store.was_attributed_to(event_id, software)
                event_ids.append(event_id)
                continue
            word_id = store.entity(
                prov_type="word",
                extent=span,
                attributes={
                    "text": text,
                    "confidence": entry.get("confidence"),
                    "existence_confidence": entry.get("existence_confidence"),
                    "temporal_confidence": entry.get("temporal_confidence"),
                    "coverage": entry.get("coverage"),
                    "recognizers": recognizers,
                    "timing_sources": entry.get("timing_sources"),
                    "index": len(kept),
                },
            )
            store.was_generated_by(word_id, activity)
            store.was_attributed_to(word_id, software)
            word_ids.append(word_id)
            kept.append({**entry, "start": span[0], "end": span[1]})
        entity_id = _measurement(
            store,
            activity,
            software,
            name="consensus_transcript",
            signal="plain",
            attributes={
                "words": kept,
                "provenance": provenance,
                "systems": [CRISPERWHISPER_ID, QWEN_ID],
                "timing_authority": "consensus_asr",
                "word_ids": word_ids,
                "event_ids": event_ids,
                "text": " ".join(str(entry.get("text") or "") for entry in kept),
            },
            derived_from=(state["asr_crisperwhisper_id"], state["asr_qwen_id"]),
        )
        derivatives["consensus_transcript"] = entity_id
        view.append(entity_id)
        view.extend(word_ids)
        view.extend(event_ids)
        state.update(consensus=kept, consensus_id=entity_id, consensus_word_ids=word_ids)

    def _phonation_tracks() -> None:
        """F0 and formant tracks over the whole stream — measured once, localised nowhere.

        Sustained-phonation and glide span *detection* used to happen here; it has moved to
        TAXONOMY (owner-directed), which reads this measurement back and runs the same proposal
        functions over it, so a decision about which stretch counts as phonation is no longer made
        during conditioning. This block keeps only the part that is a measurement: F0 over the
        pre-emphasised stream, and the first four formants and their bandwidths over ``plain``, per
        frame. It no longer depends on the consensus transcript at all — word-aligned phonation
        proposal is TAXONOMY's concern now, not a reason for this measurement to wait on ASR.
        """
        f0_range = config.require("voice.f0_range_hz")
        f0_min_hz, f0_max_hz = float(f0_range[0]), float(f0_range[1])
        parameters: dict[str, Any] = {
            "hop_s": float(config.require("phonation_spans.hop_s")),
            "max_formants": int(config.require("phonation_spans.max_formants")),
            "formant_max_hz": float(config.require("phonation_spans.formant_max_hz")),
            "formant_window_s": float(config.require("phonation_spans.formant_window_s")),
            "formant_preemphasis_hz": float(config.require("phonation_spans.formant_preemphasis_hz")),
            "f0_min_hz": f0_min_hz,
            "f0_max_hz": f0_max_hz,
        }
        activity = _step("phonation_tracks", parameters, (sharp_id, plain_id), software)
        times, f0_hz, strength = f0_track(sharp, f0_min_hz=f0_min_hz, f0_max_hz=f0_max_hz, hop_s=parameters["hop_s"])
        formants = formant_track(
            plain,
            hop_s=parameters["hop_s"],
            max_formants=parameters["max_formants"],
            formant_max_hz=parameters["formant_max_hz"],
            window_s=parameters["formant_window_s"],
            preemphasis_hz=parameters["formant_preemphasis_hz"],
        )
        np.savez(
            run_dir / "derivatives" / "phonation_tracks.npz",
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
        entity_id = _measurement(
            store,
            activity,
            software,
            name="phonation_tracks",
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
        derivatives["phonation_tracks"] = entity_id
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
            attributes={"path": f"derivatives/{name}.npz", **parameters},
            derived_from=(sharp_id,),
        )
        derivatives[name] = entity_id
        view.append(entity_id)
        state[f"{name}_magnitude"] = np.sqrt(np.maximum(power, 0.0))
        state[f"{name}_hop_s"] = hop_ms / 1000.0

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
            attributes={"path": "derivatives/gammatone.npz", "hop_s": parameters["hop_s"]},
            derived_from=(sharp_id,),
        )
        derivatives["gammatone"] = entity_id
        view.append(entity_id)

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
        ("phonation_tracks", _phonation_tracks),
        ("energy_envelope", _envelope),
        ("normalized_envelope", _normalized_envelope),
        ("spectrogram_wideband", lambda: _spectrogram("spectrogram_wideband", "spectrogram.wideband_window_ms")),
        (
            "spectrogram_narrowband",
            lambda: _spectrogram("spectrogram_narrowband", "spectrogram.narrowband_window_ms"),
        ),
        ("spans", _spans),
        ("squim", _squim),
        ("span_hear", _span_hear),
        ("span_yamnet", _span_yamnet),
        ("gammatone", _gammatone),
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
