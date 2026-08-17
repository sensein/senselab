"""Public entry point for the uncertainty-axis comparator workflow.

``compute_uncertainty_axes`` is the only function callers should typically need. It reads the
in-memory ``passes`` summary produced by analyze_audio's per-task pipeline and returns the L1
per-signal results (one per ``(pass, signal)``) plus one fused axis per ``axes.AXIS_NAMES`` entry,
plus an ``incomparable_reasons`` dict for the disagreements index. Counted from the declaration
rather than written out: "three" was wrong from the moment ``background_mask`` became the fourth.

**One grid, every axis** (D-24). ``grid`` applies to all of them, and there is deliberately no
per-axis override — the three that existed produced four grids sharing zero bucket keys, so every
cross-axis coupling in the pipeline ran and did nothing.

**There is no per-pass axis.** An axis aggregates across signals *and* across passes: a pass is an
input dimension to the fold, never an index on its output. So harvest and link are per pass, and
the fold happens exactly once per axis, in ``fuse.fuse_axis``, with every pass in hand. The
previous 9-cell ``(pass × axis)`` grid — six per-pass axes plus three ``raw_vs_enhanced`` deltas
obtained by subtracting two of them — is gone; perturbation stability is measured per *signal* by
``reliability.signal_stability``, which is what actually feeds the fusion weights.

**Harvest / link split** (spec ``20260723-225523-dynamic-uncertainty-workflow`` research.md D8,
FR-006): the expensive, model-touching phase lives in ``harvest_pass`` (embedding extraction +
clustering, frame posteriors, Brouhaha quality, sound sources, per-axis vote harvesting) and
returns a :class:`~..votes.PassHarvest`; the cheap, pure link into per-signal rows and belief
buckets lives in :func:`senselab.audio.workflows.audio_analysis.votes.link_pass`. Re-scoring with a
different aggregator therefore requires no model inference:

    harvests = {pl: harvest_pass(...)[0] for pl in passes}
    linked = {pl: link_pass(h, params=params) for pl, h in harvests.items()}

**Legacy mutation note**: with the default ``mutate_passes=True`` this function still
injects the synthetic ``embedding_silhouette/...`` diar source into each pass's
``diarization.by_model`` block (consumed by the timeline plot and by callers that re-read
``passes``). Pass ``mutate_passes=False`` for a side-effect-free call — the synthetic
source is then only visible via ``PassHarvest.synthetic_diarization``.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from senselab.audio.data_structures import Audio
from senselab.audio.workflows.audio_analysis.asr import harvest_asr_votes
from senselab.audio.workflows.audio_analysis.axes import AXIS_NAMES, HARVESTED_AXES, passes_for_axis
from senselab.audio.workflows.audio_analysis.embeddings import (
    WindowEmbedding,
    extract_per_window_embeddings,
)
from senselab.audio.workflows.audio_analysis.fuse import SnrGate, fuse_axis
from senselab.audio.workflows.audio_analysis.grid import BucketGrid
from senselab.audio.workflows.audio_analysis.harvesters import (
    classification_top1_in_window,
    classification_windows,
)
from senselab.audio.workflows.audio_analysis.perturbations import IDENTITY_NAME
from senselab.audio.workflows.audio_analysis.reliability import (
    measured_weights,
    reliability_from_stability,
    signal_stability,
    stability_rows,
)
from senselab.audio.workflows.audio_analysis.reliability import (
    signal_names as _signal_names,
)
from senselab.audio.workflows.audio_analysis.signal import resolved_commit_sha
from senselab.audio.workflows.audio_analysis.speaker import harvest_speaker_votes
from senselab.audio.workflows.audio_analysis.speech_presence import harvest_speech_presence_evidence
from senselab.audio.workflows.audio_analysis.speech_presence_link import (
    policy_from_params,
    votes_for_harvest,
)
from senselab.audio.workflows.audio_analysis.support import (
    evidence_signal_names,
    informative_evidence,
    signal_support,
)
from senselab.audio.workflows.audio_analysis.types import (
    FusedAxis,
    SignalResult,
    UncertaintyAxis,
)
from senselab.audio.workflows.audio_analysis.votes import (
    PassHarvest,
    apply_scene_coupling,
    link_pass,
)


def harvest_pass(
    *,
    perturbation: str,
    pass_summary: dict[str, Any],
    per_pass_audio: Audio | None,
    grid: BucketGrid,
    speaker_embedding_models: list[str],
    speech_presence_labels: list[str],
    scene_quality: bool = True,
    sound_sources: bool = True,
    embedding_window_s: float = 2.0,
    embedding_hop_s: float = 1.0,
    same_speaker_floor: float = 0.30,
    diff_speaker_floor: float = 0.70,
    cluster_cosine_threshold: float = 0.5,
    clustering_algorithm: str = "spectral",
    task_type: str | None = None,
    calibration: dict[str, Any] | None = None,
    speech_presence_policy: Any = None,  # noqa: ANN401 — SpeechPresencePolicy
) -> tuple[PassHarvest, dict[str, list[WindowEmbedding]], dict[str, str]]:
    """Run every model-touching step for one pass and return its harvested votes.

    **One ``grid``, every axis.** There is no per-axis grid parameter, because there is no axis
    whose question is answered at a different spacing: presence and the mask are cut on it, the
    speaker votes are cast on it, and the asr axis resamples the consensus word fold onto it. The
    three that used to exist produced four grids that shared no bucket key, which silently disabled
    every cross-axis coupling in the pipeline.

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
            incomparable_reasons[f"{perturbation}/speaker/across_time"] = (
                f"speaker-embedding extraction failed: {exc!r}"
            )
            per_window_embeddings = {}
    else:
        if not speaker_embedding_models:
            incomparable_reasons[f"{perturbation}/speaker/embeddings"] = (
                "no embedding models configured — silhouette / cosine validation disabled"
            )
        elif per_pass_audio is None:
            incomparable_reasons[f"{perturbation}/speaker/embeddings"] = (
                "no Audio object available for this pass — embedding extraction skipped"
            )
    for emb_model_id, emb_msg in emb_failures.items():
        incomparable_reasons[f"{perturbation}/speaker/embeddings/{emb_model_id}"] = emb_msg

    # Cluster windowed embeddings to estimate the pass's speaker count and
    # synthesize an embedding-derived diarization source. The result feeds
    # both the speech_presence axis (per-window silhouette voter) and the
    # diarization stack (the synthetic source becomes another diar voter
    # for the speaker axis and another stripe in the timeline plot).
    # One per embedder that clustered successfully. Each is a diarization model in its own right
    # (D-20) — its spans go into ``diarization.by_model`` beside pyannote's and sortformer's, and every
    # consumer reads them there. There is deliberately **no pass-level representative**: the only thing
    # ever taken from one was a cosine calibration band, which is per-embedder, so a representative
    # meant applying one embedder's band to another's distances.
    emb_clusters: list[dict[str, Any]] = []
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
                # One diarizer per embedder. A ``break`` here kept only whichever embedder sorted
                # first and discarded the rest, so a run with ecapa *and* resnet produced three
                # diarization models where there are four — and the discarded one survived only as a
                # ``::`` verifier, which is how "embedding" came to look like a separate dimension
                # rather than the basis of a model. Clustering over resnet vectors is a diarizer in
                # exactly the sense clustering over ecapa vectors is (D-20).
                emb_clusters.append(emb_cluster)
        for k, msg in cluster_failures.items():
            incomparable_reasons[f"{perturbation}/speaker/{k}"] = msg
        if not emb_clusters and per_window_embeddings:
            incomparable_reasons[f"{perturbation}/speaker/embedding_clustering"] = (
                "all embedding models too sparse / failed clustering — no n_speakers estimate"
            )

    # Build the synthetic embedding-derived diar source, harvested from an
    # augmented LOCAL copy of the pass summary (no caller mutation here).
    synthetic_diarization: dict[str, Any] | None = None
    harvest_summary = pass_summary
    if emb_clusters:
        synthetic_diarization = {}
        for cluster in emb_clusters:
            synthetic_segments: list[Any] = []
            entries = cluster["windows"]
            labels = cluster["labels"]
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
            synthetic_diarization[f"embedding_silhouette/{cluster['embedding_model']}"] = {
                "status": "ok",
                "result": [synthetic_segments],
                "n_speakers": cluster["n_speakers"],
                "best_silhouette": cluster.get("best_silhouette"),
                "is_synthetic": True,
            }
        diar_block = pass_summary.get("diarization") or {}
        by_model = dict(diar_block.get("by_model") or {})
        by_model.update(synthetic_diarization)
        harvest_summary = {**pass_summary, "diarization": {**diar_block, "by_model": by_model}}

    align_by_model = ((harvest_summary.get("alignment") or {}).get("by_model")) or {}

    # ── speech_presence harvest inputs ──
    # Frame-level speech posteriors (US3): segmentation-3.0 raw scores + the
    # Brouhaha VAD head, as continuous fine-resolution speech_presence voters.
    frame_voters: dict[str, Any] = {}
    frame_posteriors_provenance: dict[str, Any] = {}
    brouhaha_frames = None
    # No ``segmentation-3.0`` frame voter. It was here as a *speech* voter, and brouhaha's VAD head
    # (added below) is the same kind of evidence — a continuous per-frame speech probability. What
    # segmentation-3.0 uniquely carried was per-speaker channels, and every consumer of those has
    # moved to diarizer spans (D-19): the count to ``occupancy``, the binding to
    # ``identity_binding``. The link dispatch is structural (on ``frame_mean``, not on a model
    # name), so removing this voter needs no registry change.

    # Scene quality (US1): per-bucket SNR / clipping / reverb / bandwidth
    # degradation + estimator-spread uncertainty; additive on speech_presence rows.
    quality_by_bucket: dict[tuple[float, float], dict[str, Any]] = {}
    scene_quality_provenance: dict[str, Any] = {"enabled": bool(scene_quality)}
    if scene_quality and per_pass_audio is not None:
        from senselab.audio.tasks.scene_quality import extract_brouhaha_frames
        from senselab.audio.tasks.scene_quality.brouhaha import BROUHAHA_MODEL_ID, BROUHAHA_REVISION
        from senselab.audio.tasks.voice_activity_detection.frame_posteriors import FramePosterior
        from senselab.audio.workflows.audio_analysis.quality import (
            QUALITY_ANALYSIS_HOP_S,
            QUALITY_ANALYSIS_WIN_S,
            harvest_quality_measurements,
        )

        brouhaha_frames = extract_brouhaha_frames([per_pass_audio])[0]
        # No ``calibration`` here: L1 measures in dB / hertz / proportion, and the anchors that
        # turn those into degradation scores are applied by ``aggregate_pass`` at L2.
        for q in harvest_quality_measurements(audio=per_pass_audio, brouhaha=brouhaha_frames, grid=grid):
            quality_by_bucket[(round(q["start"], 6), round(q["end"], 6))] = q
        scene_quality_provenance.update(
            {
                "analysis_win_length": QUALITY_ANALYSIS_WIN_S,
                "analysis_hop_length": QUALITY_ANALYSIS_HOP_S,
                "calibration_version": (calibration or {}).get("calibration_version"),
                "model": {
                    "id": BROUHAHA_MODEL_ID,
                    # Both, deliberately (mirrors StageContext.provenance_for): "revision" is the
                    # ref Brouhaha is pinned to, "commit_sha" is the immutable commit that ref
                    # resolved to for this run. This used to record only the ref, indistinguishable
                    # from a moving "main" that happened to answer the same way on a given day.
                    # Resolved only when the model actually loaded — a model that never loaded has
                    # no commit to name, and resolving one anyway would spend a Hub round-trip on
                    # every null-quality run for no benefit.
                    "revision": BROUHAHA_REVISION,
                    "commit_sha": (
                        resolved_commit_sha(BROUHAHA_MODEL_ID, BROUHAHA_REVISION)
                        if brouhaha_frames is not None
                        else None
                    ),
                    "available": brouhaha_frames is not None,
                },
            }
        )

    # Brouhaha's VAD head as a frame-posterior speech voter. **Not gated on ``scene_quality``**: by
    # D-20 one forward pass yields three signals — ``(speech, brouhaha/VAD)``, ``(snr, brouhaha)`` and
    # ``(c50, brouhaha)`` — and sharing a pass is an implementation fact, not a reason to share a
    # flag. It was inside the quality gate, and once ``segmentation-3.0`` stopped being a voter that
    # left **zero** frame voters whenever a caller disabled scene quality: the speech axis silently
    # lost its frame evidence to a quality switch. Found by a test that passes ``scene_quality=False``.
    if per_pass_audio is not None:
        from senselab.audio.tasks.scene_quality import extract_brouhaha_frames as _brouhaha_for_vad
        from senselab.audio.tasks.scene_quality.brouhaha import (
            BROUHAHA_MODEL_ID as _VAD_MODEL_ID,
        )
        from senselab.audio.tasks.scene_quality.brouhaha import (
            BROUHAHA_REVISION as _VAD_REVISION,
        )
        from senselab.audio.tasks.voice_activity_detection.frame_posteriors import FramePosterior

        vad_frames = brouhaha_frames if brouhaha_frames is not None else _brouhaha_for_vad([per_pass_audio])[0]
        if vad_frames is not None:
            # A VAD head is genuinely one channel, so ``single`` is a declaration, not a collapse.
            frame_voters["frame_brouhaha_vad"] = FramePosterior(
                activations=np.asarray(vad_frames.vad, dtype=np.float64)[:, None],
                frame_hop_s=vad_frames.frame_hop_s,
                channel_format="single",
            )
        # Recorded whether or not the model loaded: a voter with no provenance is a number nobody can
        # reproduce, and ``available: False`` is the only way "the model was unreachable" stays
        # distinguishable from "this voter was never asked for".
        frame_posteriors_provenance["brouhaha_vad"] = {
            "id": _VAD_MODEL_ID,
            # Same reasoning as the scene-quality "model" block above: keep the ref, add the
            # resolved commit rather than replacing one with the other, and only resolve when the
            # model actually loaded.
            "revision": _VAD_REVISION,
            "commit_sha": (resolved_commit_sha(_VAD_MODEL_ID, _VAD_REVISION) if vad_frames is not None else None),
            "available": vad_frames is not None,
        }

    speech_presence_evidence = harvest_speech_presence_evidence(
        pass_summary=harvest_summary,
        grid=grid,
        speech_presence_labels=speech_presence_labels,
        alignment_by_model=align_by_model,
        frame_posteriors=frame_voters or None,
        # Audio for the absolute acoustic signals (LUFS, level-above-floor). Both are
        # whole-recording measurements, so they cannot be recovered from the per-frame openSMILE
        # table the way the percentile-ranked signals they replace were (D-3).
        waveform=(per_pass_audio.waveform.detach().cpu().numpy() if per_pass_audio is not None else None),
        sampling_rate=(int(per_pass_audio.sampling_rate) if per_pass_audio is not None else None),
    )

    # Sound sources (US2): per-bucket AudioSet→category masses from AST/YAMNet.
    source_by_bucket: dict[tuple[float, float], dict[str, Any]] = {}
    sound_sources_provenance: dict[str, Any] = {"enabled": bool(sound_sources)}
    if sound_sources:
        from senselab.audio.workflows.audio_analysis.sound_sources import (
            harvest_source_categories,
            load_source_category_map,
        )

        for s in harvest_source_categories(pass_summary=harvest_summary, grid=grid):
            source_by_bucket[(round(s["start"], 6), round(s["end"], 6))] = s
        sound_sources_provenance["category_map_version"] = load_source_category_map().get("version")

    # ── background-mask harvest ──
    # Derived from the presence evidence already gathered, not re-measured: the mask shares the
    # presence grid (D-24 correction), so one measurement serves both axes and they cannot drift.
    # Which reading indicates *target* activity depends on ``task_type`` — in a breathing task speech
    # indicates target *absence*, which is how a collected breath came to be reported as background.
    from senselab.audio.workflows.audio_analysis.mask_harvest import harvest_background_mask_evidence

    _speech_by_bucket: dict[tuple[float, float], float] = {}
    _words_by_bucket: dict[tuple[float, float], float] = {}
    _occupancy_by_bucket: dict[tuple[float, float], float] = {}
    # Which presence signals each mask reading maxes over. Recorded because the mask's three
    # "signals" are folds of presence's own evidence, and a fold renames its sources out of
    # visibility: ``measure_axis_overlap`` compares what an axis rests on, so without this the
    # coupling ``speech_presence <- axis::background_mask`` reads overlap 0.00 and enters at full
    # weight while being, in substance, presence's own evidence returning under another name.
    _mask_sources: dict[str, dict[tuple[float, float], set[str]]] = {"speech": {}, "words": {}, "speakers": {}}
    for row in speech_presence_evidence:
        key = (round(float(row["start"]), 6), round(float(row["end"]), 6))
        for name, ev in (row.get("evidence") or {}).items():
            if not isinstance(ev, Mapping):
                continue
            if "frame_mean" in ev and ev["frame_mean"] is not None:
                _speech_by_bucket[key] = max(_speech_by_bucket.get(key, 0.0), float(ev["frame_mean"]))
                _mask_sources["speech"].setdefault(key, set()).add(str(name))
            elif ev.get("word_overlap_s") is not None:
                _words_by_bucket[key] = max(_words_by_bucket.get(key, 0.0), float(ev["word_overlap_s"]))
                _mask_sources["words"].setdefault(key, set()).add(str(name))
            elif ev.get("covered_fraction") is not None:
                _occupancy_by_bucket[key] = max(_occupancy_by_bucket.get(key, 0.0), float(ev["covered_fraction"]))
                _mask_sources["speakers"].setdefault(key, set()).add(str(name))
    background_mask_evidence = harvest_background_mask_evidence(
        duration_s=float(harvest_summary.get("duration_s", 0.0) or 0.0),
        grid=grid,
        task_type=task_type,
        speech_by_bucket=_speech_by_bucket,
        word_coverage_by_bucket=_words_by_bucket,
        speaker_occupancy_by_bucket=_occupancy_by_bucket,
        sources_by_bucket=_mask_sources,
        policy=speech_presence_policy,
    )

    # ── speaker harvest ──
    # Prefer per-pass empirical calibration learned from this pass's embedding
    # clusters; fall back to the CLI defaults when clustering didn't produce
    # useful per-pass anchors.
    # One band per embedder, from that embedder's own clustering. A cosine band is a property of the
    # embedding space, not of the pass: ecapa's same/different separation is not resnet's, and the
    # single pass-level pair this replaces calibrated *every* embedder's distances with whichever
    # model the clustering loop happened to keep. An embedder that produced no usable band is absent
    # here and falls back to the CLI defaults, which is the honest reading of "not measured".
    speaker_floors: dict[str, tuple[float, float]] = {}
    for cluster in emb_clusters:
        sf = cluster.get("empirical_same_speaker_floor")
        df = cluster.get("empirical_diff_speaker_floor")
        if isinstance(sf, (int, float)) and isinstance(df, (int, float)) and df > sf:
            speaker_floors[str(cluster["embedding_model"])] = (float(sf), float(df))
    # ── the consensus word fold, once per pass ──
    # Two axes read these words: the asr axis resamples their accuracy, and the speaker axis reads
    # their ``temporal_confidence`` as word-location doubt (word boundaries are what assign a word to
    # a speaker's span). The fold runs g2p per word pair, so folding it twice would double that cost
    # for one answer — and would let the two axes disagree about a fold neither of them owns.
    from senselab.audio.workflows.audio_analysis.asr import fuse_consensus_words
    from senselab.audio.workflows.audio_analysis.harvesters import resolve_asr_result

    asr_blocks = (harvest_summary.get("asr") or {}).get("by_model") or {}
    asr_resolved = {
        m: resolve_asr_result(b, align_by_model.get(m))
        for m, b in asr_blocks.items()
        if isinstance(b, dict) and b.get("status") == "ok"
    }
    # ``policy=`` is not optional here even though it defaults to None: without it the fold silently
    # falls back to ``fuse_consensus_words``'s own 0.3 / 0.15, and ``linking.asr_slot_overlap`` /
    # ``asr_slot_mid_tol_s`` never reach the word-slot join on the only production path (F-162). The
    # packaged defaults happen to equal the fallbacks, so the bug is inert today and shows up only
    # when someone changes the config and nothing happens -- which is the worse failure, because the
    # run's own provenance then records the value that ran and the config records a different one.
    consensus_fold = fuse_consensus_words(asr_resolved, policy=speech_presence_policy)

    speaker_votes = harvest_speaker_votes(
        pass_summary=harvest_summary,
        grid=grid,
        per_window_embeddings=per_window_embeddings,
        same_speaker_floor=same_speaker_floor,
        diff_speaker_floor=diff_speaker_floor,
        speaker_floors=speaker_floors,
        cluster_cosine_threshold=cluster_cosine_threshold,
        fused_words=consensus_fold[0],
    )

    # ── asr harvest ──
    asr_votes = harvest_asr_votes(
        pass_summary=harvest_summary,
        grid=grid,
        alignment_by_model=align_by_model,
        fused=consensus_fold,
    )

    harvest = PassHarvest(
        perturbation=perturbation,
        speech_presence_evidence=speech_presence_evidence,
        speaker_votes=speaker_votes,
        asr_votes=asr_votes,
        quality_by_bucket=quality_by_bucket,
        source_by_bucket=source_by_bucket,
        # Every axis, and the same pair for each — recorded per axis rather than once because the
        # *claim* being made is per axis, and a reader checking "are these on one grid?" must be
        # able to see the four answers rather than infer them from a single entry. Built from
        # ``AXIS_NAMES`` so an axis cannot be missing from it: ``background_mask`` was, while being
        # cut on the presence grid, so nothing recorded what the mask's rows were spaced at.
        grids={axis_name: {"win_length": grid.win_length, "hop_length": grid.hop_length} for axis_name in AXIS_NAMES},
        # Carried so the aggregate phase can compare a measured roll-off against Nyquist without
        # touching audio, which would break its purity guarantee.
        sampling_rate=int(per_pass_audio.sampling_rate) if per_pass_audio is not None else 16000,
        # L1 vectors. The clustering over them is L2's (D-7), so they travel rather than their
        # conclusion — which also lets a later stage reuse the same cluster assignment for
        # speaker label repair instead of re-deriving one that could disagree.
        per_window_embeddings=dict(per_window_embeddings or {}),
        # Carried so fusion can bind fused speaker ids to each diarizer's own labels (C2) without
        # re-running a model, which would defeat the harvest/aggregate split. The channels this
        # replaces were permutation-arbitrary and could not name anyone (D-19).
        diarization_by_model=dict((harvest_summary.get("diarization") or {}).get("by_model") or {}),
        background_mask_evidence=background_mask_evidence,
        provenance_extras={
            "scene_quality": scene_quality_provenance,
            "sound_sources": sound_sources_provenance,
            "frame_posteriors": frame_posteriors_provenance,
        },
        synthetic_diarization=synthetic_diarization,
    )
    return harvest, per_window_embeddings, incomparable_reasons


def _signal_multiplicity(buckets_by_pass: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, int]:
    """How many independent sources each signal folds, over every bucket and pass of one axis.

    Max rather than mean or sum: a signal's source count is a property of how it is *built*, so the
    bucket where every source reported is the one that says what it is. Buckets where a source fell
    silent are that source being silent, not the signal having fewer of them — and summing across
    passes would report four diarizers as eight, since a source is the same source under a transform.

    Args:
        buckets_by_pass: ``{perturbation → harvested buckets}`` for one axis.

    Returns:
        ``{signal → source count}``. A signal that declares nothing is absent, and
        ``measured_weights`` then leaves it at full corroboration — a factor never gathered must not
        act as a discount.
    """
    from senselab.audio.workflows.audio_analysis.fuse import per_signal_sources

    out: dict[str, int] = {}
    for buckets in (buckets_by_pass or {}).values():
        for bucket in buckets or []:
            if not isinstance(bucket, Mapping):
                continue
            for signal, count in per_signal_sources(bucket).items():
                if count > out.get(signal, 0):
                    out[signal] = count
    return out


def compute_uncertainty_axes(
    *,
    passes: dict[str, dict[str, Any]],
    grid: BucketGrid,
    params: dict[str, Any],
    audio: dict[str, Audio],
    speaker_embedding_models: list[str],
    aggregator: str,
    speech_presence_labels: list[str],
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
    weights_out: dict[str, Any] | None = None,
    stability_out: dict[str, Any] | None = None,
    linked_out: dict[str, Any] | None = None,
    calibration: dict[str, Any] | None = None,
    snr_floor_db: float,
    snr_gated_passes: frozenset[str],
) -> tuple[
    dict[str, dict[str, SignalResult]],
    dict[str, FusedAxis],
    dict[str, str],
    dict[str, dict[str, list[WindowEmbedding]]],
]:
    """Measure every signal on every pass, then fuse each axis once across all of them.

    Thin wrapper over :func:`harvest_pass` (expensive, model-touching) + :func:`~..votes.link_pass`
    (pure) + :func:`~..fuse.fuse_axis` (pure). With the default ``mutate_passes=True`` the legacy
    synthetic-diar-source injection into the caller's ``passes`` dict is unchanged.

    Args:
        passes: Mapping ``{perturbation → pass_summary}`` where each pass_summary is the
            same dict-of-dicts shape produced by analyze_audio's run_pass (keyed by task,
            then by ``"by_model"`` for multi-model tasks). Pass labels are typically
            ``"raw"`` and ``"enhanced"``.
        grid: The run's one bucket grid — every axis is harvested on it, so row *i* of one axis is
            row *i* of another and cross-axis coupling needs no projection. Defaults to
            :data:`~.grid.DEFAULT_TIME_GRID`; there is deliberately no per-axis override.
        params: Comparator-relevant CLI flags — recorded into each row's parquet
            provenance for reproducibility.
        audio: Per-pass ``Audio`` objects, used to slice waveforms for per-segment
            speaker embedding extraction.
        speaker_embedding_models: Model ids for ECAPA / ResNet — typically the same set
            already configured via ``--speaker-embedding-models``.
        aggregator: One of ``min`` / ``mean`` / ``harmonic_mean`` /
            ``disagreement_weighted`` (FR-004).
        weights_out: When given, receives ``{axis → {signal → measured weight}}`` — the
            weights the fold applied, so level 2 can fuse with the same numbers
            rather than recomputing them and drifting apart silently.
        linked_out: When given, receives ``{perturbation → LinkedPass}`` — the linked belief
            buckets the fold consumed, including the synthetic cross-signal blocks. An
            out-parameter so the return type stays the four products a consumer needs.
        stability_out: When given, receives ``{"instability": {axis → {signal → mean |Δ|}},
            "per_bucket": {axis → {signal → rows}}}`` — the perturbation evidence behind those
            weights, so the number that discounted a signal is inspectable rather than only
            recomputable.
        speech_presence_labels: AudioSet labels that count as "speech-present" for AST /
            YAMNet contributions to the speech_presence axis.
        scene_quality: When True (default), compute per-bucket audio-quality degradation
            scores (SNR / clipping / reverb / bandwidth + estimator-spread uncertainty)
            via Brouhaha + existing DSP metrics and attach them as additive columns on
            the speech_presence rows. Null-safe when the model / audio is unavailable (FR-023).
        sound_sources: When True (default), map the AST / YAMNet AudioSet scores into
            per-bucket source-category masses (speech / people / machine / environment)
            + dominant category, attached as additive columns on the speech_presence rows.
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
        harvests_out: Optional dict populated with ``{perturbation: PassHarvest}`` as
            each pass is harvested. An out-parameter rather than a fourth return
            value so no existing caller's tuple arity changes; the adaptive loop's
            in-process path (T040) needs the harvests the parquets were built from.
        calibration: Optional flat runtime calibration dict (US5 — see
        speech_presence_policy: The run's L1→L2 linking policy, from the config's ``linking:``
            block. Supplies the background mask's three thresholds. ``None`` uses the module
            defaults, which are the same values.
            ``calibration.profile_to_runtime``): dB→[0,1] anchors consumed by the
            quality harvest. Aggregator-side temperatures travel separately via
            ``params["calibration"]``; pass the same dict in both places.

        snr_floor_db: SNR below which a repair perturbation is admitted to the fold, from
            ``triage.snr_floor_db`` — one number shared with the run-level ``enhancement.mode:
            auto`` decision rather than a second threshold that could drift from it.
        snr_gated_passes: Names of the perturbations whose readings only count where the recording
            is degraded — those whose ``Perturbation.admission_requires_low_snr`` is true. Required
            rather than inferred from the name: "anything that is not ``raw``" would gate the
            invariance probes too, and those are meaningful at every SNR by construction. Pass
            ``frozenset()`` for a run with nothing to gate.

    Returns:
        ``(signal_results_by_pass, fused_axes, incomparable_reasons, per_window_embeddings_by_pass)``
        where:

        - signal_results_by_pass maps ``perturbation → {signal → SignalResult}`` — the L1
          evidence, in native units, with no axis anywhere.
        - fused_axes maps ``axis → FusedAxis`` — round 0 of the single fold, with the pass
          dimension appearing only as each row's ``contributing_passes`` list.
        - incomparable_reasons maps ``"<pass>/<axis>/<sub-signal>"`` → human-readable
          reason for surfacing in ``disagreements.json``.
        - per_window_embeddings_by_pass maps ``perturbation`` →
          ``{embedding_model_id → [WindowEmbedding, ...]}``. The window grid is
          uniform (fixed ``embedding_window_s`` / ``embedding_hop_s``) and shared
          across embedding models so adjacent-window cosine distance is a
          model-free indicator of speaker change — independent of any diarization.
    """
    incomparable_reasons: dict[str, str] = {}
    per_window_embeddings_by_pass: dict[str, dict[str, list[WindowEmbedding]]] = {}
    harvests_by_label: dict[str, PassHarvest] = {}

    for perturbation in sorted(passes.keys()):
        pass_summary = passes.get(perturbation) or {}
        harvest, per_window_embeddings, pass_reasons = harvest_pass(
            perturbation=perturbation,
            pass_summary=pass_summary,
            per_pass_audio=audio.get(perturbation),
            grid=grid,
            speaker_embedding_models=speaker_embedding_models,
            speech_presence_labels=speech_presence_labels,
            # The same policy the link phase applies, so the mask's thresholds and the presence
            # votes' thresholds cannot come from two different places.
            speech_presence_policy=policy_from_params(params),
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
        per_window_embeddings_by_pass[perturbation] = per_window_embeddings
        incomparable_reasons.update(pass_reasons)

        # Legacy side effect (opt-out via mutate_passes=False): make the synthetic
        # diar source visible to callers that re-read ``passes`` (timeline plot).
        if mutate_passes and harvest.synthetic_diarization:
            diar_block = pass_summary.get("diarization") or {}
            by_model = diar_block.get("by_model") or {}
            by_model.update(harvest.synthetic_diarization)
            pass_summary["diarization"] = {**diar_block, "by_model": by_model}

        if harvests_out is not None:
            harvests_out[perturbation] = harvest
        harvests_by_label[perturbation] = harvest

    # Reliability is measured across passes, so aggregation waits until every pass is
    # harvested. Raw and enhanced are the same recording under a transform, so each
    # signal's two answers are already a stability sample — a signal that contradicts
    # itself between them has not earned an equal vote. Aggregation is pure, so deferring
    # it changes nothing else.
    # Both factors measured, no signal named: perturbation stability asks whether a signal
    # agrees with itself under a transform, physical support whether the audio carries what
    # it claimed. Support is measured once on the unmodified pass — a speaker claimed where
    # there is no speech is a fact about the recording, not about the transform.
    # Linking first, once: stability compares *beliefs* across passes, and the presence harvest
    # holds measurements. Deriving the weight from the same linked value the fold consumes is what
    # stops the two being computed from different things.
    linked_by_pass = {
        perturbation: link_pass(harvest, params=params) for perturbation, harvest in sorted(harvests_by_label.items())
    }
    signal_results_by_pass = {label: dict(linked.signal_results) for label, linked in linked_by_pass.items()}
    if linked_out is not None:
        linked_out.clear()
        linked_out.update(linked_by_pass)
    # ``passes_for_axis`` rather than every pass: an axis whose question is about the recording as read
    # folds only the identity (``axes.IDENTITY_ONLY_AXES``). The mask's own stage already refuses the
    # enhanced variant; its axis used to accept it.
    buckets_by_axis_pass = {
        axis: {
            label: linked_by_pass[label].buckets_by_axis.get(axis, [])
            for label in passes_for_axis(axis, linked_by_pass)
        }
        for axis in HARVESTED_AXES
    }

    # Support is measured on the recording when it is present: the identity is the one
    # perturbation every run has, so it is the only defensible default.
    support_label = IDENTITY_NAME if IDENTITY_NAME in linked_by_pass else next(iter(linked_by_pass), None)
    speech_presence_buckets = (
        buckets_by_axis_pass["speech_presence"].get(support_label, []) if support_label is not None else []
    )
    support = signal_support(
        speech_presence_buckets,
        evidence_signals=sorted(
            informative_evidence(speech_presence_buckets, sorted(evidence_signal_names(speech_presence_buckets)))
        ),
    )
    if weights_out is not None:
        weights_out.clear()
    instability_by_axis = {
        axis: signal_stability(harvests_by_label, axis=axis, buckets_by_pass=buckets_by_axis_pass[axis])
        for axis in HARVESTED_AXES
    }
    reliability_by_axis = {
        axis: measured_weights(
            instability_by_axis[axis],
            support,
            _signal_names(harvests_by_label, axis=axis),
            multiplicity=_signal_multiplicity(buckets_by_axis_pass[axis]),
        )
        for axis in HARVESTED_AXES
    }
    if stability_out is not None:
        stability_out.clear()
        stability_out.update(
            {
                "instability": {axis: dict(v) for axis, v in instability_by_axis.items()},
                "per_bucket": {axis: stability_rows(buckets_by_axis_pass[axis]) for axis in HARVESTED_AXES},
            }
        )
    if weights_out is not None:
        # Exposed so level 2 can fuse with the same weights the diagnostics used; recomputing
        # them there would let the two drift apart silently. The per-factor basis rides
        # alongside under a reserved key so a discounted signal records *why*.
        weights_out.update(reliability_by_axis)
        stability = {axis: reliability_from_stability(v) for axis, v in instability_by_axis.items()}
        weights_out["__basis__"] = {
            axis: {
                signal: {
                    "stability": round(float(stability[axis].get(signal, 1.0)), 6),
                    "support": round(float(support.get(signal.split("::", 1)[0], 1.0)), 6),
                }
                for signal in reliability_by_axis[axis]
            }
            for axis in reliability_by_axis
        }

    # One fold per axis, over every pass at once. ``fuse_axis`` averages a signal's readings
    # across passes before weighting, so the passes are an input dimension here exactly as the
    # signals are — and appear on the output only as the ``contributing_passes`` column.
    basis = (weights_out or {}).get("__basis__") or {}
    # One constructor, shared with ``fuse.write_final_uncertainty``, so the two folds of these same
    # harvests cannot end up differently gated under one axis name.
    snr_gate = SnrGate.build(harvests_by_label, floor_db=snr_floor_db, gated_passes=snr_gated_passes)
    fused_axes: dict[str, FusedAxis] = {}
    for axis in HARVESTED_AXES:
        rows = fuse_axis(
            buckets_by_axis_pass[axis],
            weights=reliability_by_axis.get(axis, {}),
            aggregator=aggregator,
            weight_basis=basis.get(axis, {}),
            round_index=0,
            snr_gate=snr_gate,
        )
        fused_axes[axis] = FusedAxis(
            axis=axis,  # type: ignore[arg-type]
            rows=rows,
            provenance={
                "axis": axis,
                "grid": {
                    label: dict(linked.provenance.get("grids", {}).get(axis, {}))
                    for label, linked in linked_by_pass.items()
                },
                "comparator_params": params,
                "aggregator": aggregator,
                "passes": sorted(linked_by_pass),
                **(
                    {
                        "speech_presence_policy": next(iter(linked_by_pass.values())).provenance.get(
                            "speech_presence_policy"
                        )
                    }
                    if axis == "speech_presence" and linked_by_pass
                    else {}
                ),
            },
        )

    _attach_scene_measurements(fused_axes, linked_by_pass)
    _apply_scene_coupling(fused_axes, params)
    # No transcript column on the asr rows. It used to carry ``{pass::model → {text}}`` per bucket,
    # rebuilt from the same words ``final/transcript.json`` publishes at word resolution — and it is
    # what made this axis need a 1.0 s window, because a fully-contained per-bucket text read
    # returns nothing from a bucket narrower than a word. The reader that wanted it
    # (``labelstudio``) now gets the word-level ``final__consensus_transcript__text`` track that
    # ``adaptive.ls_final`` builds from the transcript itself.

    return signal_results_by_pass, fused_axes, incomparable_reasons, per_window_embeddings_by_pass


def _attach_scene_measurements(
    fused_axes: dict[str, FusedAxis],
    linked_by_pass: dict[str, Any],
) -> None:
    """Carry the per-bucket scene measurements and derived scores onto the fused presence rows.

    The measurements are per pass and per signal (L1's business); the *scores* are anchored
    against a calibration profile and are therefore L2's. Folding the two passes here — a mean,
    named — is the same treatment every other signal gets, rather than picking a winning pass.
    """
    presence = fused_axes.get("speech_presence")
    if presence is None:
        return
    measured: dict[tuple[float, float], dict[str, list[float]]] = {}
    scored: dict[tuple[float, float], dict[str, list[float]]] = {}
    labelled: dict[tuple[float, float], dict[str, Any]] = {}
    dominant: dict[tuple[float, float], list[str]] = {}
    for linked in linked_by_pass.values():
        for result in linked.signal_results.values():
            if result.signal not in ("scene_quality", "sound_sources"):
                continue
            for row in result.rows:
                key = (round(row.start, 6), round(row.end, 6))
                slot = measured.setdefault(key, {})
                for name, value in row.measurement.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        slot.setdefault(str(name), []).append(float(value))
                    elif name == "src_dominant" and isinstance(value, str):
                        dominant.setdefault(key, []).append(value)
        for key, scores in linked.quality_scores.items():
            slot = scored.setdefault(key, {})
            for name, value in scores.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    slot.setdefault(str(name), []).append(float(value))
                else:
                    # A non-numeric entry names *which* estimator produced the score. It is a
                    # fact about the measurement, not a value to average, so it rides through.
                    labelled.setdefault(key, {})[str(name)] = value
    for row in presence.rows:
        key = (round(float(row["start"]), 6), round(float(row["end"]), 6))
        for name, values in (measured.get(key) or {}).items():
            row[name] = sum(values) / len(values)
        for name, values in (scored.get(key) or {}).items():
            row[name] = sum(values) / len(values)
        for name, label in (labelled.get(key) or {}).items():
            row[name] = label
        names = dominant.get(key)
        if names:
            row["src_dominant"] = max(sorted(set(names)), key=names.count)
    presence.provenance["scene_measurement_fold"] = "mean over passes reporting the bucket"


def _apply_scene_coupling(fused_axes: dict[str, FusedAxis], params: dict[str, Any]) -> None:
    """Apply the scene→asr coupling to round 0's rows (see :func:`votes.apply_scene_coupling`).

    The shared implementation lives in ``votes`` because ``write_final_uncertainty`` has to apply
    it to *every* round: its rounds re-fold each axis from the harvests, so a coupling applied only
    here was overwritten before anything was written to disk.
    """
    asr = fused_axes.get("asr")
    presence = fused_axes.get("speech_presence")
    if asr is None or presence is None:
        return
    asr.provenance["asr_scene_coupling"] = apply_scene_coupling(asr.rows, presence.rows, params)


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
        # AudioSet speech-speech_presence detector.
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
