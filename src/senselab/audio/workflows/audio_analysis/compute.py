"""Public entry point for the three-axis comparator workflow.

``compute_uncertainty_axes`` is the only function callers should typically need. It reads
the in-memory ``passes`` summary produced by analyze_audio's per-task pipeline and returns
the nine ``AxisResult`` objects (3 axes × 2 passes + 3 raw_vs_enhanced deltas) plus an
``incomparable_reasons`` dict for the disagreements index.

**Harvest / aggregate split** (spec ``20260723-225523-dynamic-uncertainty-workflow``
research.md D8, FR-006): the expensive, model-touching phase lives in ``harvest_pass``
(embedding extraction + clustering, frame posteriors, Brouhaha quality, sound sources,
per-axis vote harvesting) and returns a :class:`~..votes.PassHarvest`; the cheap, pure
fold into ``AxisResult`` rows lives in :func:`senselab.audio.workflows.audio_analysis.votes.aggregate_pass`.
Re-scoring with a different aggregator therefore requires no model inference:

    harvests = {pl: harvest_pass(...)[0] for pl in passes}
    axis_results = {(pl, ax): r for pl, h in harvests.items()
                    for ax, r in aggregate_pass(h, aggregator="mean", params=params).items()}

**Legacy mutation note**: with the default ``mutate_passes=True`` this function still
injects the synthetic ``embedding_silhouette/...`` diar source into each pass's
``diarization.by_model`` block (consumed by the timeline plot and by callers that re-read
``passes``). Pass ``mutate_passes=False`` for a side-effect-free call — the synthetic
source is then only visible via ``PassHarvest.synthetic_diarization``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from senselab.audio.data_structures import Audio
from senselab.audio.workflows.audio_analysis.embeddings import (
    WindowEmbedding,
    extract_per_window_embeddings,
)
from senselab.audio.workflows.audio_analysis.grid import BucketGrid
from senselab.audio.workflows.audio_analysis.harvesters import (
    classification_top1_in_window,
    classification_windows,
)
from senselab.audio.workflows.audio_analysis.identity import harvest_identity_votes
from senselab.audio.workflows.audio_analysis.presence import harvest_presence_votes
from senselab.audio.workflows.audio_analysis.reliability import (
    combined_weights,
    signal_stability,
)
from senselab.audio.workflows.audio_analysis.reliability import (
    signal_names as _signal_names,
)
from senselab.audio.workflows.audio_analysis.types import (
    AxisResult,
    UncertaintyAxis,
    UncertaintyRow,
)
from senselab.audio.workflows.audio_analysis.utterance import harvest_utterance_votes
from senselab.audio.workflows.audio_analysis.votes import (
    PassHarvest,
    aggregate_pass,
    compute_pass_deltas,
)


def harvest_pass(
    *,
    pass_label: str,
    pass_summary: dict[str, Any],
    per_pass_audio: Audio | None,
    grid: BucketGrid,
    speaker_embedding_models: list[str],
    speech_presence_labels: list[str],
    utterance_grid: BucketGrid | None = None,
    presence_grid: BucketGrid | None = None,
    scene_quality: bool = True,
    sound_sources: bool = True,
    embedding_window_s: float = 2.0,
    embedding_hop_s: float = 1.0,
    same_speaker_floor: float = 0.30,
    diff_speaker_floor: float = 0.70,
    cluster_cosine_threshold: float = 0.5,
    clustering_algorithm: str = "spectral",
    calibration: dict[str, Any] | None = None,
) -> tuple[PassHarvest, dict[str, list[WindowEmbedding]], dict[str, str]]:
    """Run every model-touching step for one pass and return its harvested votes.

    Does NOT mutate ``pass_summary``: the synthetic embedding-silhouette diar source is
    harvested from an augmented local copy and reported via
    ``PassHarvest.synthetic_diarization`` for callers that want to persist / plot it.

    Returns:
        ``(harvest, per_window_embeddings, incomparable_reasons)`` where
        incomparable_reasons keys are already ``"<pass>/..."``-prefixed.
    """
    incomparable_reasons: dict[str, str] = {}

    # Windowed speaker embeddings — independent check on diar segmentation.
    per_window_embeddings: dict[str, list[WindowEmbedding]] = {}
    emb_failures: dict[str, str] = {}
    if per_pass_audio is not None and speaker_embedding_models:
        try:
            per_window_embeddings = extract_per_window_embeddings(
                audio=per_pass_audio,
                models=speaker_embedding_models,
                window_s=embedding_window_s,
                hop_s=embedding_hop_s,
                failures=emb_failures,
            )
        except Exception as exc:  # noqa: BLE001
            incomparable_reasons[f"{pass_label}/identity/across_time"] = f"speaker-embedding extraction failed: {exc!r}"
            per_window_embeddings = {}
    else:
        if not speaker_embedding_models:
            incomparable_reasons[f"{pass_label}/identity/embeddings"] = (
                "no embedding models configured — silhouette / cosine validation disabled"
            )
        elif per_pass_audio is None:
            incomparable_reasons[f"{pass_label}/identity/embeddings"] = (
                "no Audio object available for this pass — embedding extraction skipped"
            )
    for emb_model_id, emb_msg in emb_failures.items():
        incomparable_reasons[f"{pass_label}/identity/embeddings/{emb_model_id}"] = emb_msg

    # Cluster windowed embeddings to estimate the pass's speaker count and
    # synthesize an embedding-derived diarization source. The result feeds
    # both the presence axis (per-window silhouette voter) and the
    # diarization stack (the synthetic source becomes another diar voter
    # for the identity axis and another stripe in the timeline plot).
    emb_cluster: dict[str, Any] | None = None
    if per_window_embeddings:
        from senselab.audio.workflows.audio_analysis.embeddings import (
            cluster_pass_speakers as _cluster_pass_speakers,
        )

        cluster_failures: dict[str, str] = {}
        for emb_model_id in sorted(per_window_embeddings):
            entries = per_window_embeddings[emb_model_id]
            if not entries:
                continue
            speech_mask = _speech_window_mask(
                entries=entries,
                pass_summary=pass_summary,
                speech_presence_labels=speech_presence_labels,
            )
            emb_cluster = _cluster_pass_speakers(
                entries,
                failures=cluster_failures,
                failure_key=f"clustering/{emb_model_id}",
                is_speech_per_window=speech_mask,
                algorithm=clustering_algorithm,
            )
            if emb_cluster is not None:
                emb_cluster["embedding_model"] = emb_model_id
                emb_cluster["windows"] = entries
                break
        for k, msg in cluster_failures.items():
            incomparable_reasons[f"{pass_label}/identity/{k}"] = msg
        if emb_cluster is None and per_window_embeddings:
            incomparable_reasons[f"{pass_label}/identity/embedding_clustering"] = (
                "all embedding models too sparse / failed clustering — no n_speakers estimate"
            )

    # Build the synthetic embedding-derived diar source, harvested from an
    # augmented LOCAL copy of the pass summary (no caller mutation here).
    synthetic_diarization: dict[str, Any] | None = None
    harvest_summary = pass_summary
    if emb_cluster is not None:
        synthetic_segments: list[Any] = []
        entries = emb_cluster["windows"]
        labels = emb_cluster["labels"]
        for i, w in enumerate(entries):
            cluster_id = labels.get(i, "NOISE")
            if cluster_id == "NOISE":
                continue
            synthetic_segments.append(
                {
                    "start": float(w.start_s),
                    "end": float(w.end_s),
                    "speaker": cluster_id,
                }
            )
        synthetic_diar_id = f"embedding_silhouette/{emb_cluster['embedding_model']}"
        synthetic_block = {
            "status": "ok",
            "result": [synthetic_segments],
            "n_speakers": emb_cluster["n_speakers"],
            "best_silhouette": emb_cluster.get("best_silhouette"),
            "is_synthetic": True,
        }
        synthetic_diarization = {synthetic_diar_id: synthetic_block}
        diar_block = pass_summary.get("diarization") or {}
        by_model = dict(diar_block.get("by_model") or {})
        by_model[synthetic_diar_id] = synthetic_block
        harvest_summary = {**pass_summary, "diarization": {**diar_block, "by_model": by_model}}

    align_by_model = ((harvest_summary.get("alignment") or {}).get("by_model")) or {}
    ppg_block_raw = harvest_summary.get("ppgs") or harvest_summary.get("ppg")
    if ppg_block_raw is None:
        # PPG was opted out (e.g. ``--ppg`` not passed). The user explicitly
        # chose not to compute it; treat as a known-missing sub-signal.
        incomparable_reasons[f"{pass_label}/utterance/ppg"] = "PPG opt-in not provided"
        ppg_block: dict[str, Any] = {}
    elif not (isinstance(ppg_block_raw, dict) and ppg_block_raw.get("status") == "ok"):
        # PPG ran but failed (model crash, OOM, missing dependency). Surface
        # the actual status so the disagreements report distinguishes
        # "user opted out" from "we tried and it broke".
        status = ppg_block_raw.get("status", "unknown") if isinstance(ppg_block_raw, dict) else "non_dict_payload"
        error_msg = ppg_block_raw.get("error") if isinstance(ppg_block_raw, dict) else None
        reason = f"PPG extraction status={status!r}"
        if error_msg:
            reason += f" error={str(error_msg)[:160]!r}"
        incomparable_reasons[f"{pass_label}/utterance/ppg"] = reason
        ppg_block = {}
    else:
        ppg_block = ppg_block_raw

    # ── presence harvest inputs ──
    pres_grid = presence_grid if presence_grid is not None else grid

    # Frame-level speech posteriors (US3): segmentation-3.0 raw scores + the
    # Brouhaha VAD head, as continuous fine-resolution presence voters.
    frame_voters: dict[str, Any] = {}
    frame_posteriors_provenance: dict[str, Any] = {}
    brouhaha_frames = None
    if per_pass_audio is not None:
        from senselab.audio.tasks.voice_activity_detection.frame_posteriors import (
            SEGMENTATION_MODEL_ID,
            SEGMENTATION_REVISION,
            extract_speech_frame_posteriors,
        )

        seg_fp = extract_speech_frame_posteriors([per_pass_audio])[0]
        if seg_fp is not None:
            frame_voters["frame_segmentation"] = seg_fp
        frame_posteriors_provenance["segmentation"] = {
            "id": SEGMENTATION_MODEL_ID,
            "revision": SEGMENTATION_REVISION,
            "available": seg_fp is not None,
        }

    # Scene quality (US1): per-bucket SNR / clipping / reverb / bandwidth
    # degradation + estimator-spread uncertainty; additive on presence rows.
    quality_by_bucket: dict[tuple[float, float], dict[str, Any]] = {}
    scene_quality_provenance: dict[str, Any] = {"enabled": bool(scene_quality)}
    if scene_quality and per_pass_audio is not None:
        from senselab.audio.tasks.scene_quality import extract_brouhaha_frames
        from senselab.audio.tasks.scene_quality.brouhaha import BROUHAHA_MODEL_ID, BROUHAHA_REVISION
        from senselab.audio.tasks.voice_activity_detection.frame_posteriors import FramePosterior
        from senselab.audio.workflows.audio_analysis.quality import (
            QUALITY_ANALYSIS_HOP_S,
            QUALITY_ANALYSIS_WIN_S,
            harvest_quality_scores,
        )

        brouhaha_frames = extract_brouhaha_frames([per_pass_audio])[0]
        for q in harvest_quality_scores(
            audio=per_pass_audio, brouhaha=brouhaha_frames, grid=pres_grid, calibration=calibration
        ):
            quality_by_bucket[(round(q["start"], 6), round(q["end"], 6))] = q
        # Reuse the Brouhaha VAD head as a second frame-posterior presence voter.
        if brouhaha_frames is not None:
            frame_voters["frame_brouhaha_vad"] = FramePosterior(
                probs=brouhaha_frames.vad, frame_hop_s=brouhaha_frames.frame_hop_s
            )
        scene_quality_provenance.update(
            {
                "analysis_win_length": QUALITY_ANALYSIS_WIN_S,
                "analysis_hop_length": QUALITY_ANALYSIS_HOP_S,
                "calibration_version": (calibration or {}).get("calibration_version"),
                "model": {
                    "id": BROUHAHA_MODEL_ID,
                    "revision": BROUHAHA_REVISION,
                    "available": brouhaha_frames is not None,
                },
            }
        )

    presence_votes = harvest_presence_votes(
        pass_summary=harvest_summary,
        grid=pres_grid,
        speech_presence_labels=speech_presence_labels,
        alignment_by_model=align_by_model,
        per_window_embeddings=per_window_embeddings,
        frame_posteriors=frame_voters or None,
    )

    # Sound sources (US2): per-bucket AudioSet→category masses from AST/YAMNet.
    source_by_bucket: dict[tuple[float, float], dict[str, Any]] = {}
    sound_sources_provenance: dict[str, Any] = {"enabled": bool(sound_sources)}
    if sound_sources:
        from senselab.audio.workflows.audio_analysis.sound_sources import (
            harvest_source_categories,
            load_source_category_map,
        )

        for s in harvest_source_categories(pass_summary=harvest_summary, grid=pres_grid):
            source_by_bucket[(round(s["start"], 6), round(s["end"], 6))] = s
        sound_sources_provenance["category_map_version"] = load_source_category_map().get("version")

    # ── identity harvest ──
    # Prefer per-pass empirical calibration learned from this pass's embedding
    # clusters; fall back to the CLI defaults when clustering didn't produce
    # useful per-pass anchors.
    same_floor_eff = same_speaker_floor
    diff_floor_eff = diff_speaker_floor
    if isinstance(emb_cluster, dict):
        sf = emb_cluster.get("empirical_same_speaker_floor")
        df = emb_cluster.get("empirical_diff_speaker_floor")
        if isinstance(sf, (int, float)) and isinstance(df, (int, float)) and df > sf:
            same_floor_eff = float(sf)
            diff_floor_eff = float(df)
    identity_votes = harvest_identity_votes(
        pass_summary=harvest_summary,
        grid=grid,
        per_window_embeddings=per_window_embeddings,
        same_speaker_floor=same_floor_eff,
        diff_speaker_floor=diff_floor_eff,
        cluster_cosine_threshold=cluster_cosine_threshold,
    )

    # ── utterance harvest ──
    utt_grid = utterance_grid if utterance_grid is not None else grid
    utterance_votes = harvest_utterance_votes(
        pass_summary=harvest_summary,
        grid=utt_grid,
        ppg_block=ppg_block,
        alignment_by_model=align_by_model,
    )

    harvest = PassHarvest(
        pass_label=pass_label,
        presence_votes=presence_votes,
        identity_votes=identity_votes,
        utterance_votes=utterance_votes,
        quality_by_bucket=quality_by_bucket,
        source_by_bucket=source_by_bucket,
        grids={
            "presence": {"win_length": pres_grid.win_length, "hop_length": pres_grid.hop_length},
            "identity": {"win_length": grid.win_length, "hop_length": grid.hop_length},
            "utterance": {"win_length": utt_grid.win_length, "hop_length": utt_grid.hop_length},
        },
        provenance_extras={
            "scene_quality": scene_quality_provenance,
            "sound_sources": sound_sources_provenance,
            "frame_posteriors": frame_posteriors_provenance,
        },
        synthetic_diarization=synthetic_diarization,
    )
    return harvest, per_window_embeddings, incomparable_reasons


def compute_uncertainty_axes(
    *,
    passes: dict[str, dict[str, Any]],
    grid: BucketGrid,
    params: dict[str, Any],
    audio: dict[str, Audio],
    speaker_embedding_models: list[str],
    aggregator: str,
    speech_presence_labels: list[str],
    utterance_grid: BucketGrid | None = None,
    presence_grid: BucketGrid | None = None,
    scene_quality: bool = True,
    sound_sources: bool = True,
    embedding_window_s: float = 2.0,
    embedding_hop_s: float = 1.0,
    same_speaker_floor: float = 0.30,
    diff_speaker_floor: float = 0.70,
    cluster_cosine_threshold: float = 0.5,
    clustering_algorithm: str = "spectral",
    mutate_passes: bool = True,
    harvests_out: dict[str, Any] | None = None,
    calibration: dict[str, Any] | None = None,
) -> tuple[dict[tuple[str, UncertaintyAxis], AxisResult], dict[str, str], dict[str, dict[str, list[WindowEmbedding]]]]:
    """Compute per-pass and raw-vs-enhanced uncertainty rows for all three axes.

    Thin wrapper over :func:`harvest_pass` (expensive, model-touching) +
    :func:`~..votes.aggregate_pass` (pure). Behavior, outputs, and — with the default
    ``mutate_passes=True`` — the legacy synthetic-diar-source injection into the
    caller's ``passes`` dict are unchanged from the pre-split implementation.

    Args:
        passes: Mapping ``{pass_label → pass_summary}`` where each pass_summary is the
            same dict-of-dicts shape produced by analyze_audio's run_pass (keyed by task,
            then by ``"by_model"`` for multi-model tasks). Pass labels are typically
            ``"raw_16k"`` and ``"enhanced_16k"``.
        grid: Bucket grid (FR-010).
        params: Comparator-relevant CLI flags — recorded into each row's parquet
            provenance for reproducibility.
        audio: Per-pass ``Audio`` objects, used to slice waveforms for per-segment
            speaker embedding extraction.
        speaker_embedding_models: Model ids for ECAPA / ResNet — typically the same set
            already configured via ``--speaker-embedding-models``.
        aggregator: One of ``min`` / ``mean`` / ``harmonic_mean`` /
            ``disagreement_weighted`` (FR-004).
        speech_presence_labels: AudioSet labels that count as "speech-present" for AST /
            YAMNet contributions to the presence axis.
        utterance_grid: Optional separate bucket grid for the utterance axis (typically
            wider + overlapping than the shared grid so most words land fully inside at
            least one bucket). When ``None``, the shared ``grid`` is reused for utterance.
        presence_grid: Optional separate (typically finer) bucket grid for the presence
            axis, so brief events can be localized from continuous frame posteriors. When
            ``None``, the shared ``grid`` is reused (preserving legacy behavior); the CLI
            defaults it to 0.1 s / 0.02 s. Quality and source columns are computed on this
            same presence grid so they align with the presence rows.
        scene_quality: When True (default), compute per-bucket audio-quality degradation
            scores (SNR / clipping / reverb / bandwidth + estimator-spread uncertainty)
            via Brouhaha + existing DSP metrics and attach them as additive columns on
            the presence rows. Null-safe when the model / audio is unavailable (FR-023).
        sound_sources: When True (default), map the AST / YAMNet AudioSet scores into
            per-bucket source-category masses (speech / people / machine / environment)
            + dominant category, attached as additive columns on the presence rows.
            Null when neither classifier ran (FR-023).
        embedding_window_s: Window length (seconds) for fixed-grid speaker-embedding
            extraction. Defaults to 2.0 s (ECAPA's recommended minimum).
        embedding_hop_s: Window hop (seconds) for fixed-grid speaker-embedding
            extraction. Defaults to 1.0 s.
        same_speaker_floor: Identity calibration anchor — raw cosine distance
            ≤ this is treated as confidently same-speaker.
        diff_speaker_floor: Identity calibration anchor — raw cosine distance
            ≥ this is treated as confidently different-speaker.
        cluster_cosine_threshold: Cosine similarity threshold for clustering
            ``(diar_model, raw_label)`` into pass-wide speaker IDs (used by
            the cross-model agreement sub-signal and the plot's color map).
        clustering_algorithm: ``"spectral"`` (default) or ``"kmeans"`` for the
            embedding-window clustering step. Spectral clustering on a
            precomputed cosine-similarity affinity handles non-convex speaker
            clusters better than k-means; falls back automatically to k-means
            on per-k failure.
        mutate_passes: When True (default, legacy behavior) inject each pass's
            synthetic ``embedding_silhouette/...`` diar source into the caller's
            ``passes`` dict so downstream consumers (timeline plot) see it. When
            False the caller's dict is left untouched.
        harvests_out: Optional dict populated with ``{pass_label: PassHarvest}`` as
            each pass is harvested. An out-parameter rather than a fourth return
            value so no existing caller's tuple arity changes; the adaptive loop's
            in-process path (T040) needs the harvests the parquets were built from.
        calibration: Optional flat runtime calibration dict (US5 — see
            ``calibration.profile_to_runtime``): dB→[0,1] anchors consumed by the
            quality harvest. Aggregator-side temperatures travel separately via
            ``params["calibration"]``; pass the same dict in both places.

    Returns:
        ``(axis_results, incomparable_reasons, per_window_embeddings_by_pass)`` where:

        - axis_results maps ``(pass_label, axis)`` → AxisResult.
        - incomparable_reasons maps ``"<pass>/<axis>/<sub-signal>"`` → human-readable
          reason for surfacing in ``disagreements.json``.
        - per_window_embeddings_by_pass maps ``pass_label`` →
          ``{embedding_model_id → [WindowEmbedding, ...]}``. The window grid is
          uniform (fixed ``embedding_window_s`` / ``embedding_hop_s``) and shared
          across embedding models so adjacent-window cosine distance is a
          model-free indicator of speaker change — independent of any diarization.
    """
    axis_results: dict[tuple[str, UncertaintyAxis], AxisResult] = {}
    incomparable_reasons: dict[str, str] = {}
    per_window_embeddings_by_pass: dict[str, dict[str, list[WindowEmbedding]]] = {}
    harvests_by_label: dict[str, PassHarvest] = {}

    for pass_label in sorted(passes.keys()):
        pass_summary = passes.get(pass_label) or {}
        harvest, per_window_embeddings, pass_reasons = harvest_pass(
            pass_label=pass_label,
            pass_summary=pass_summary,
            per_pass_audio=audio.get(pass_label),
            grid=grid,
            speaker_embedding_models=speaker_embedding_models,
            speech_presence_labels=speech_presence_labels,
            utterance_grid=utterance_grid,
            presence_grid=presence_grid,
            scene_quality=scene_quality,
            sound_sources=sound_sources,
            embedding_window_s=embedding_window_s,
            embedding_hop_s=embedding_hop_s,
            same_speaker_floor=same_speaker_floor,
            diff_speaker_floor=diff_speaker_floor,
            cluster_cosine_threshold=cluster_cosine_threshold,
            clustering_algorithm=clustering_algorithm,
            calibration=calibration,
        )
        per_window_embeddings_by_pass[pass_label] = per_window_embeddings
        incomparable_reasons.update(pass_reasons)

        # Legacy side effect (opt-out via mutate_passes=False): make the synthetic
        # diar source visible to callers that re-read ``passes`` (timeline plot).
        if mutate_passes and harvest.synthetic_diarization:
            diar_block = pass_summary.get("diarization") or {}
            by_model = diar_block.get("by_model") or {}
            by_model.update(harvest.synthetic_diarization)
            pass_summary["diarization"] = {**diar_block, "by_model": by_model}

        if harvests_out is not None:
            harvests_out[pass_label] = harvest
        harvests_by_label[pass_label] = harvest

    # Reliability is measured across passes, so aggregation waits until every pass is
    # harvested. Raw and enhanced are the same recording under a transform, so each
    # signal's two answers are already a stability sample — a signal that contradicts
    # itself between them has not earned an equal vote. Aggregation is pure, so deferring
    # it changes nothing else.
    # Both gates, as in adaptive/influence: perturbation stability asks whether a signal
    # agrees with itself, the derivation gate whether its agreement with another signal is
    # independent evidence at all.
    reliability_by_axis = {
        axis: combined_weights(
            signal_stability(harvests_by_label, axis=axis),
            _signal_names(harvests_by_label, axis=axis),
        )
        for axis in ("presence", "identity", "utterance")
    }
    for pass_label, harvest in sorted(harvests_by_label.items()):
        for axis, result in aggregate_pass(
            harvest, aggregator=aggregator, params=params, signal_reliability=reliability_by_axis
        ).items():
            axis_results[(pass_label, axis)] = result  # type: ignore[index]

    # ── raw_vs_enhanced deltas ──
    # The current public delta is "raw_16k vs enhanced_16k". A 3rd pass (e.g. a
    # second enhancement variant) is computed alongside but not delta'd here —
    # extend by adding a generic pass-pair loop if multi-pass deltas are needed.
    if "raw_16k" in passes and "enhanced_16k" in passes:
        for axis in ("presence", "identity", "utterance"):
            raw_rows = axis_results[("raw_16k", axis)].rows  # type: ignore[index]
            enh_rows = axis_results[("enhanced_16k", axis)].rows  # type: ignore[index]
            delta_rows = compute_pass_deltas(raw_rows, enh_rows, axis, aggregator)
            axis_results[("raw_vs_enhanced", axis)] = AxisResult(  # type: ignore[index]
                pass_label="raw_vs_enhanced",
                axis=axis,  # type: ignore[arg-type]
                rows=delta_rows,
                provenance={
                    "axis": axis,
                    "pass": "raw_vs_enhanced",
                    "grid": {"win_length": grid.win_length, "hop_length": grid.hop_length},
                    "comparator_params": params,
                    "contributing_model_set": sorted({m for r in (raw_rows + enh_rows) for m in r.contributing_models}),
                },
            )

    return axis_results, incomparable_reasons, per_window_embeddings_by_pass


# Backward-compatible alias — the delta math moved (verbatim) to votes.compute_pass_deltas.
def _compute_raw_vs_enhanced_delta(
    raw_rows: list[UncertaintyRow],
    enh_rows: list[UncertaintyRow],
    axis: str,
    aggregator: str,
) -> list[UncertaintyRow]:
    """Deprecated alias for :func:`~..votes.compute_pass_deltas` (kept for importers)."""
    return compute_pass_deltas(raw_rows, enh_rows, axis, aggregator)


def _speech_window_mask(
    *,
    entries: list[WindowEmbedding],
    pass_summary: dict[str, Any],
    speech_presence_labels: list[str],
) -> list[bool] | None:
    """Build a per-embedding-window boolean mask of "is this window speech?".

    **YAMNet veto, not a fallback ladder.** When YAMNet is available, its top-1
    label decision is authoritative — even if loudness is high, AST disagrees,
    or both. AST is only consulted when YAMNet is unavailable; openSMILE
    loudness is consulted only when both classifiers are unavailable. This is
    deliberate: YAMNet is trained on the AudioSet hierarchy with explicit
    speech labels; AST is broader-coverage but noisier on speech specifically;
    loudness is recording-conditional. Trusting YAMNet's "Music" / "Vehicle"
    over a loud-but-non-speech window is the right call for our use case.

    Tradeoff: YAMNet has known confusions (e.g. tagging child voices as "Music"
    or "Singing"). When that happens, a real-speech window gets dropped from
    clustering. The mitigation is upstream: tune ``speech_presence_labels`` to
    include the singing / NORP labels you care about. We deliberately do NOT
    let loudness override YAMNet here, because that would break the
    silent-room-detection guarantee callers rely on (a loud window of music
    must not be allowed to claim "speech" status).

    Returns ``None`` when none of the three signals are available, in which
    case the caller falls back to legacy behavior (cluster every non-zero-norm
    window).
    """
    ast_block = pass_summary.get("ast") or {}
    yam_block = pass_summary.get("yamnet") or {}
    ast_ok = isinstance(ast_block, dict) and ast_block.get("status") == "ok"
    yam_ok = isinstance(yam_block, dict) and yam_block.get("status") == "ok"

    feat_block = pass_summary.get("features") or {}
    feat_result = feat_block.get("result") if isinstance(feat_block, dict) else None
    opensmile_rows: list[dict[str, Any]] = feat_result.get("opensmile", []) if isinstance(feat_result, dict) else []

    if not (ast_ok or yam_ok or opensmile_rows):
        return None

    def _native_grid(block: dict[str, Any]) -> tuple[float, float]:
        windows = classification_windows(block.get("result"))
        if not windows or not isinstance(windows[0], dict):
            return 1.0, 1.0
        w = windows[0]
        win_len = float(w.get("win_length", 0) or 0) or float(w.get("end", 0) - w.get("start", 0))
        hop_len = float(w.get("hop_length", 0) or 0) or win_len
        if win_len <= 0:
            win_len = 1.0
        if hop_len <= 0:
            hop_len = win_len
        return win_len, hop_len

    ast_hop = _native_grid(ast_block)[1] if ast_ok else 0.0
    yam_hop = _native_grid(yam_block)[1] if yam_ok else 0.0

    loudness_q25: float | None = None
    if opensmile_rows:
        vals: list[float] = []
        for r in opensmile_rows:
            v = r.get("Loudness_sma3")
            if v is None:
                continue
            try:
                vf = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(vf):
                vals.append(vf)
        if len(vals) >= 100:  # ~1 s of opensmile frames
            loudness_q25 = float(np.percentile(vals, 25))

    allow = set(speech_presence_labels)
    mask: list[bool] = []
    for w in entries:
        center = 0.5 * (w.start_s + w.end_s)
        # YAMNet is authoritative when available — it's the canonical
        # AudioSet speech-presence detector.
        if yam_ok:
            idx = max(0, int(round(center / yam_hop))) if yam_hop > 0 else 0
            label, _, _ = classification_top1_in_window(yam_block.get("result"), idx)
            if label is not None:
                mask.append(label in allow)
                continue
        # Fall back to AST when YAMNet unavailable.
        if ast_ok:
            idx = max(0, int(round(center / ast_hop))) if ast_hop > 0 else 0
            label, _, _ = classification_top1_in_window(ast_block.get("result"), idx)
            if label is not None:
                mask.append(label in allow)
                continue
        # Final fallback: openSMILE loudness threshold.
        if loudness_q25 is not None and opensmile_rows:
            vals_in: list[float] = []
            for r in opensmile_rows:
                rs = r.get("start") or r.get("frameTime") or r.get("time")
                re_ = r.get("end")
                try:
                    rs_f = float(rs) if rs is not None else None
                    re_f = float(re_) if re_ is not None else (rs_f + 0.01 if rs_f is not None else None)
                except (TypeError, ValueError):
                    continue
                if rs_f is None or re_f is None:
                    continue
                if rs_f < w.end_s and re_f > w.start_s:
                    v = r.get("Loudness_sma3")
                    if v is None:
                        continue
                    try:
                        vf = float(v)
                    except (TypeError, ValueError):
                        continue
                    vals_in.append(vf)
            if vals_in:
                mean_loud = sum(vals_in) / len(vals_in)
                mask.append(mean_loud >= loudness_q25)
                continue
        mask.append(True)
    return mask
