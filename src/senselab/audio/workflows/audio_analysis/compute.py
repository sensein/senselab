"""Public entry point for the three-axis comparator workflow.

``compute_uncertainty_axes`` is the only function callers should typically need. It reads
the in-memory ``passes`` summary produced by analyze_audio's per-task pipeline and returns
the nine ``AxisResult`` objects (3 axes × 2 passes + 3 raw_vs_enhanced deltas) plus an
``incomparable_reasons`` dict for the disagreements index.

**Note**: this function MUTATES the caller's ``passes`` dict to inject the synthetic
``embedding_silhouette/...`` diar source into each pass's ``diarization.by_model`` block.
That source is consumed by the identity-axis harvester and the timeline plot. Callers
that retain a reference to ``passes`` will see the synthetic source after this call.
Pass a deep-copy if you need the input unchanged.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from senselab.audio.data_structures import Audio
from senselab.audio.workflows.audio_analysis.aggregate import (
    aggregate_identity,
    aggregate_presence,
    aggregate_utterance,
    presence_p_voice,
)
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
from senselab.audio.workflows.audio_analysis.presence import (
    harvest_presence_votes,
    reference_grid_and_speech_mask,
)
from senselab.audio.workflows.audio_analysis.types import (
    AxisResult,
    UncertaintyAxis,
    UncertaintyRow,
)
from senselab.audio.workflows.audio_analysis.utterance import harvest_utterance_votes


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
    embedding_window_s: float = 2.0,
    embedding_hop_s: float = 1.0,
    same_speaker_floor: float = 0.30,
    diff_speaker_floor: float = 0.70,
    cluster_cosine_threshold: float = 0.5,
    clustering_algorithm: str = "spectral",
) -> tuple[dict[tuple[str, UncertaintyAxis], AxisResult], dict[str, str], dict[str, dict[str, list[WindowEmbedding]]]]:
    """Compute per-pass and raw-vs-enhanced uncertainty rows for all three axes.

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

    # Per-pass windowed embeddings — uniform fixed grid, no diarization dependency.
    per_window_embeddings_by_pass: dict[str, dict[str, list[WindowEmbedding]]] = {}

    pass_labels = sorted(passes.keys())
    for pass_label in pass_labels:
        pass_summary = passes.get(pass_label) or {}
        per_pass_audio = audio.get(pass_label)
        # Windowed speaker embeddings — independent check on diar segmentation.
        emb_failures: dict[str, str] = {}
        if per_pass_audio is not None and speaker_embedding_models:
            try:
                per_window_embeddings_by_pass[pass_label] = extract_per_window_embeddings(
                    audio=per_pass_audio,
                    models=speaker_embedding_models,
                    window_s=embedding_window_s,
                    hop_s=embedding_hop_s,
                    failures=emb_failures,
                )
            except Exception as exc:  # noqa: BLE001
                incomparable_reasons[f"{pass_label}/identity/across_time"] = (
                    f"speaker-embedding extraction failed: {exc!r}"
                )
                per_window_embeddings_by_pass[pass_label] = {}
        else:
            per_window_embeddings_by_pass[pass_label] = {}
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
        if per_window_embeddings_by_pass[pass_label]:
            from senselab.audio.workflows.audio_analysis.embeddings import (
                cluster_pass_speakers as _cluster_pass_speakers,
            )

            cluster_failures: dict[str, str] = {}
            # Compute the speech mask once on the shared window grid (it depends
            # on window times, not the embedding vectors, so it applies to every
            # model) — same helper the speaker_profile build uses (FR-002).
            _, speech_mask = reference_grid_and_speech_mask(
                per_window_embeddings_by_pass[pass_label],
                pass_summary=pass_summary,
                speech_presence_labels=speech_presence_labels,
            )
            for emb_model_id in sorted(per_window_embeddings_by_pass[pass_label]):
                entries = per_window_embeddings_by_pass[pass_label][emb_model_id]
                if not entries:
                    continue
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
            if emb_cluster is None and per_window_embeddings_by_pass[pass_label]:
                incomparable_reasons[f"{pass_label}/identity/embedding_clustering"] = (
                    "all embedding models too sparse / failed clustering — no n_speakers estimate"
                )

        # Inject the embedding-derived diar source into the pass summary so
        # downstream harvesters see it as just another diar voter.
        if emb_cluster is not None:
            diar_block = pass_summary.get("diarization") or {}
            by_model = diar_block.get("by_model") or {}
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
            by_model[synthetic_diar_id] = {
                "status": "ok",
                "result": [synthetic_segments],
                "n_speakers": emb_cluster["n_speakers"],
                "best_silhouette": emb_cluster.get("best_silhouette"),
                "is_synthetic": True,
            }
            pass_summary["diarization"] = {**diar_block, "by_model": by_model}

        align_by_model = ((pass_summary.get("alignment") or {}).get("by_model")) or {}
        ppg_block_raw = pass_summary.get("ppgs") or pass_summary.get("ppg")
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

        # ── presence ──
        presence_votes = harvest_presence_votes(
            pass_summary=pass_summary,
            grid=grid,
            speech_presence_labels=speech_presence_labels,
            alignment_by_model=align_by_model,
            per_window_embeddings=per_window_embeddings_by_pass.get(pass_label),
        )
        presence_rows: list[UncertaintyRow] = []
        # Map bucket-start → presence p_voice so identity / utterance can
        # mask their per-bucket uncertainty by "we are confident there's
        # speech here" (low p_voice → confident silence → don't trust the
        # axis-specific signals → mask down).
        presence_p_voice_by_bucket: dict[tuple[float, float], float] = {}
        for bucket in presence_votes:
            u = aggregate_presence(bucket["votes"])
            p_v = presence_p_voice(bucket["votes"])
            if u is None and not bucket["votes"]:
                continue
            # Presence axis itself isn't masked — it IS the mask source.
            presence_rows.append(
                UncertaintyRow(
                    start=bucket["start"],
                    end=bucket["end"],
                    axis="presence",
                    aggregated_uncertainty=u,
                    contributing_models=sorted(bucket["votes"].keys()),
                    model_votes=bucket["votes"],
                    comparison_status="ok" if u is not None else "incomparable",
                    raw_aggregated_uncertainty=u,
                    intensity_weight=1.0,
                )
            )
            if p_v is not None:
                presence_p_voice_by_bucket[(round(bucket["start"], 6), round(bucket["end"], 6))] = float(p_v)

        def _intensity_mask(start: float, end: float) -> float:
            """Map presence p_voice → mask weight in [0, 1].

            ``p_voice >= 0.5`` (likely voice or uncertain) → mask = 1.0 (full
            weight, don't downweight). ``p_voice < 0.5`` → linear ramp down to
            0.0 at ``p_voice == 0`` (confident silence → mask out the bucket).
            Falls back to 1.0 (no masking) when presence didn't run.
            """
            p = presence_p_voice_by_bucket.get((round(start, 6), round(end, 6)))
            if p is None:
                return 1.0
            if p >= 0.5:
                return 1.0
            return max(0.0, min(1.0, p / 0.5))

        axis_results[(pass_label, "presence")] = AxisResult(
            pass_label=pass_label,  # type: ignore[arg-type]
            axis="presence",
            rows=presence_rows,
            provenance={
                "axis": "presence",
                "pass": pass_label,
                "grid": {"win_length": grid.win_length, "hop_length": grid.hop_length},
                "comparator_params": params,
                "contributing_model_set": sorted({m for b in presence_votes for m in b["votes"]}),
            },
        )

        # ── identity ──
        # Prefer per-pass empirical calibration learned from this pass's
        # embedding clusters (within-cluster vs between-cluster cosine
        # distance distributions). Falls back to the CLI defaults when
        # clustering didn't produce useful per-pass anchors.
        same_floor_eff = same_speaker_floor
        diff_floor_eff = diff_speaker_floor
        if isinstance(emb_cluster, dict):
            sf = emb_cluster.get("empirical_same_speaker_floor")
            df = emb_cluster.get("empirical_diff_speaker_floor")
            if isinstance(sf, (int, float)) and isinstance(df, (int, float)) and df > sf:
                same_floor_eff = float(sf)
                diff_floor_eff = float(df)
        identity_votes = harvest_identity_votes(
            pass_summary=pass_summary,
            grid=grid,
            per_window_embeddings=per_window_embeddings_by_pass[pass_label],
            same_speaker_floor=same_floor_eff,
            diff_speaker_floor=diff_floor_eff,
            cluster_cosine_threshold=cluster_cosine_threshold,
        )
        identity_rows: list[UncertaintyRow] = []
        for bucket in identity_votes:
            u_raw = aggregate_identity(bucket["votes"], raw_vs_enh=None, aggregator=aggregator)
            if u_raw is None and not bucket["votes"]:
                continue
            mask = _intensity_mask(bucket["start"], bucket["end"])
            # Don't multiply mask into ``aggregated_uncertainty`` — that would
            # corrupt the parquet's primary column (a real high-disagreement
            # bucket in a low-loudness region would be silently demoted, and
            # ``disagreements.json`` ranks on this column). Instead keep the
            # raw value here and expose ``intensity_weight`` so downstream
            # consumers (global_summary, disagreements ranking tiebreaker) can
            # apply the mask in their own aggregation step.
            identity_rows.append(
                UncertaintyRow(
                    start=bucket["start"],
                    end=bucket["end"],
                    axis="identity",
                    aggregated_uncertainty=u_raw,
                    contributing_models=sorted(bucket["votes"].keys()),
                    model_votes=bucket["votes"],
                    comparison_status="ok" if u_raw is not None else "incomparable",
                    raw_aggregated_uncertainty=u_raw,
                    intensity_weight=mask,
                )
            )
        axis_results[(pass_label, "identity")] = AxisResult(
            pass_label=pass_label,  # type: ignore[arg-type]
            axis="identity",
            rows=identity_rows,
            provenance={
                "axis": "identity",
                "pass": pass_label,
                "grid": {"win_length": grid.win_length, "hop_length": grid.hop_length},
                "comparator_params": params,
                "contributing_model_set": sorted({m for b in identity_votes for m in b["votes"]}),
            },
        )

        # ── utterance ──
        # Utterance gets its own (typically wider, overlapping) grid so most words
        # land fully inside at least one bucket. Defaults to the shared grid when
        # the caller doesn't supply one.
        utt_grid = utterance_grid if utterance_grid is not None else grid
        utterance_votes = harvest_utterance_votes(
            pass_summary=pass_summary,
            grid=utt_grid,
            ppg_block=ppg_block,
            alignment_by_model=align_by_model,
        )
        utterance_rows: list[UncertaintyRow] = []
        for bucket in utterance_votes:
            u_raw = aggregate_utterance(bucket["votes"], aggregator=aggregator)
            if u_raw is None and not bucket["votes"]:
                continue
            # Utterance buckets may use a different (typically wider) grid
            # than presence; average the mask across presence buckets that
            # OVERLAP this utterance bucket (rather than picking the closest
            # single one — the closest-only rule would ignore a half-voice
            # half-silence bucket).
            mask_values = []
            for (s_p, e_p), p_v in presence_p_voice_by_bucket.items():
                if s_p < bucket["end"] and e_p > bucket["start"]:
                    mask_values.append(1.0 if p_v >= 0.5 else max(0.0, min(1.0, p_v / 0.5)))
            mask = sum(mask_values) / len(mask_values) if mask_values else 1.0
            utterance_rows.append(
                UncertaintyRow(
                    start=bucket["start"],
                    end=bucket["end"],
                    axis="utterance",
                    aggregated_uncertainty=u_raw,
                    contributing_models=sorted(bucket["votes"].keys()),
                    model_votes=bucket["votes"],
                    comparison_status="ok" if u_raw is not None else "incomparable",
                    raw_aggregated_uncertainty=u_raw,
                    intensity_weight=mask,
                )
            )
        axis_results[(pass_label, "utterance")] = AxisResult(
            pass_label=pass_label,  # type: ignore[arg-type]
            axis="utterance",
            rows=utterance_rows,
            provenance={
                "axis": "utterance",
                "pass": pass_label,
                "grid": {"win_length": utt_grid.win_length, "hop_length": utt_grid.hop_length},
                "comparator_params": params,
                "contributing_model_set": sorted({m for b in utterance_votes for m in b["votes"]}),
            },
        )

    # ── raw_vs_enhanced deltas ──
    # The current public delta is "raw_16k vs enhanced_16k". A 3rd pass (e.g. a
    # second enhancement variant) is computed alongside but not delta'd here —
    # extend by adding a generic pass-pair loop if multi-pass deltas are needed.
    if "raw_16k" in passes and "enhanced_16k" in passes:
        for axis in ("presence", "identity", "utterance"):
            raw_rows = axis_results[("raw_16k", axis)].rows  # type: ignore[index]
            enh_rows = axis_results[("enhanced_16k", axis)].rows  # type: ignore[index]
            delta_rows = _compute_raw_vs_enhanced_delta(raw_rows, enh_rows, axis, aggregator)
            axis_results[("raw_vs_enhanced", axis)] = AxisResult(
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


def _compute_raw_vs_enhanced_delta(
    raw_rows: list[UncertaintyRow],
    enh_rows: list[UncertaintyRow],
    axis: str,
    aggregator: str,
) -> list[UncertaintyRow]:
    """Pair raw and enhanced rows by (start, end) and emit a delta row per shared bucket.

    The delta row's ``aggregated_uncertainty`` is the absolute difference between the
    two passes' aggregated uncertainties on the same bucket, clipped to [0, 1]. Buckets
    present in only one pass produce ``comparison_status="one_sided"`` with
    ``aggregated_uncertainty=None``.
    """
    raw_by_bucket = {(r.start, r.end): r for r in raw_rows}
    enh_by_bucket = {(r.start, r.end): r for r in enh_rows}
    bucket_keys = sorted(set(raw_by_bucket) | set(enh_by_bucket))
    out: list[UncertaintyRow] = []
    for key in bucket_keys:
        raw_row = raw_by_bucket.get(key)
        enh_row = enh_by_bucket.get(key)
        # Build a combined model_votes dict tagged by pass.
        votes: dict[str, dict[str, Any]] = {}
        if raw_row is not None:
            for m, v in raw_row.model_votes.items():
                votes[f"raw_16k::{m}"] = v
        if enh_row is not None:
            for m, v in enh_row.model_votes.items():
                votes[f"enhanced_16k::{m}"] = v

        if raw_row is None or enh_row is None:
            # One-sided delta: carry forward whatever raw/enh values are
            # available so downstream consumers can still tell which pass had
            # the data. Mask the delta row's intensity_weight by the maximum
            # of the available side(s) — if either pass thinks there's voice,
            # the delta is worth attending to.
            present = raw_row if raw_row is not None else enh_row
            iw = present.intensity_weight if present and present.intensity_weight is not None else None
            ra_raw = raw_row.raw_aggregated_uncertainty if raw_row else None
            enh_raw = enh_row.raw_aggregated_uncertainty if enh_row else None
            out.append(
                UncertaintyRow(
                    start=key[0],
                    end=key[1],
                    axis=axis,  # type: ignore[arg-type]
                    aggregated_uncertainty=None,
                    contributing_models=sorted(votes.keys()),
                    model_votes=votes,
                    comparison_status="one_sided",
                    raw_aggregated_uncertainty=ra_raw if ra_raw is not None else enh_raw,
                    intensity_weight=iw,
                )
            )
            continue
        ra = raw_row.aggregated_uncertainty
        ea = enh_row.aggregated_uncertainty
        if ra is None or ea is None:
            delta = None
            status = "incomparable"
        else:
            delta = max(0.0, min(1.0, abs(ra - ea)))
            status = "ok"
        # Delta uses the MAXIMUM of the two passes' intensity weights — if
        # either pass thought voice was present in the bucket, the delta is
        # worth attending to (don't downweight just because one pass thought
        # the bucket was silent).
        raw_iw = raw_row.intensity_weight if raw_row.intensity_weight is not None else 1.0
        enh_iw = enh_row.intensity_weight if enh_row.intensity_weight is not None else 1.0
        delta_iw = max(raw_iw, enh_iw)
        out.append(
            UncertaintyRow(
                start=key[0],
                end=key[1],
                axis=axis,  # type: ignore[arg-type]
                aggregated_uncertainty=delta,
                contributing_models=sorted(votes.keys()),
                model_votes=votes,
                comparison_status=status,  # type: ignore[arg-type]
                raw_aggregated_uncertainty=delta,
                intensity_weight=delta_iw,
            )
        )
    return out


# NOTE: ``speech_window_mask_for_file`` (formerly the private ``_speech_window_mask``
# defined here) was promoted to ``presence.py`` so the speaker_profile package can
# reuse it as the build-time speech gate (T009a, FR-002). It is imported above.
