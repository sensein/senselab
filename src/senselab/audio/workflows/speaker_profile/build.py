"""Build a SpeakerProfile from a subject's audio files.

This module hosts the per-file speech-window extractor (T009 / FR-002 / FR-008
gating) and — in subsequent tasks — the cross-file aggregation, confidence
policy, per-file keep/drop decisions, session-weighting refinement, and the
``build_speaker_profile`` orchestration entrypoint (T012–T016).

Phase 2 deliverable in this file:

- :class:`TaggedWindowEmbedding` — the file-tagged per-window embedding used
  for leave-one-file-out scoring later (FR-012).
- :func:`extract_speech_windows_for_file` — locate speech via a
  best-available presence gate (diarization + scene-speech mask + loudness;
  opportunistic Whisper / PPG when already cached; never triggers ASR/PPG
  solely to gate) and return ≥~1s windows per model tagged with the source
  ``file_id``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from senselab.audio.data_structures import Audio
from senselab.audio.workflows.audio_analysis.embeddings import (
    WindowEmbedding,
    _empirical_calibration_band,
    cluster_pass_speakers,
    extract_per_window_embeddings,
)
from senselab.audio.workflows.audio_analysis.presence import speech_window_mask_for_file
from senselab.audio.workflows.speaker_profile import constants as C
from senselab.audio.workflows.speaker_profile.cache import audio_signature, senselab_version
from senselab.audio.workflows.speaker_profile.io import SCHEMA_VERSION, save_profile
from senselab.audio.workflows.speaker_profile.types import (
    ClusterStats,
    ProfileConfidence,
    ProfileParams,
    ProfileSourceFile,
    SpeakerProfile,
)
from senselab.utils.data_structures import DeviceType

# Default AudioSet labels we treat as "speech-present" for the build-time gate.
# These mirror ``scripts/analyze_audio.py``'s default — keeping them in sync
# means a profile built here uses the same speech definition the identity-axis
# clustering uses inside ``analyze_audio`` (FR-002 wording: "the same signal
# the clustering step consumes").
DEFAULT_SPEECH_PRESENCE_LABELS: tuple[str, ...] = (
    "Speech",
    "Conversation",
    "Narration, monologue",
    "Female speech, woman speaking",
    "Male speech, man speaking",
    "Child speech, kid speaking",
    "Speech synthesizer",
)


@dataclass(slots=True, frozen=True)
class TaggedWindowEmbedding:
    """A :class:`WindowEmbedding` tagged with its source ``file_id`` and model.

    The ``file_id`` tag is what enables leave-one-file-out scoring at compare
    time (FR-012 / R5): when scoring recording *F*, we exclude all tagged
    windows whose ``file_id == F`` from the centroid.
    """

    file_id: str
    model_id: str
    window: WindowEmbedding


def extract_speech_windows_for_file(
    *,
    audio: Audio,
    file_id: str,
    pass_summary: dict[str, Any],
    embedding_models: Sequence[str] = C.DEFAULT_EMBEDDING_MODELS,
    device: DeviceType | None = None,
    profile_window_s: float = C.PROFILE_WINDOW_S,
    profile_hop_s: float = C.PROFILE_HOP_S,
    speech_presence_labels: Sequence[str] = DEFAULT_SPEECH_PRESENCE_LABELS,
    failures: dict[str, str] | None = None,
) -> tuple[list[TaggedWindowEmbedding], dict[str, Any]]:
    """Locate speech windows in one file and embed each per model.

    The gate is **best-available presence**: it consumes the cached
    AST/YAMNet/openSMILE outputs already present in ``pass_summary`` via the
    promoted :func:`speech_window_mask_for_file` helper (T009a). The caller
    is responsible for assembling ``pass_summary`` from whichever tasks they
    have already cached — this function never triggers ASR/PPG itself.

    Args:
        audio: Mono 16 kHz ``Audio`` for one file. The window grid is anchored
            to the audio duration, so audios shorter than ``profile_window_s``
            contribute nothing.
        file_id: Stable identifier for the source file; tagged onto every
            returned window so leave-one-file-out scoring (FR-012) can later
            exclude this file's contribution.
        pass_summary: Dict shaped like ``analyze_audio``'s per-pass summary;
            this function reads ``ast``, ``yamnet``, and ``features.opensmile``
            from it to build the speech mask. Pass an empty/partial dict to
            fall back to "every non-zero window" (the legacy behavior of
            ``speech_window_mask_for_file``).
        embedding_models: HF model ids for the embedding consensus
            (default: ECAPA + ResNet + WavLM — FR-018).
        device: Optional compute device override.
        profile_window_s: Long-window length for centroid-quality embeddings
            (default from ``constants.py``; FR-002 — windows are ≥~1 s
            contiguous speech by construction).
        profile_hop_s: Hop between consecutive long windows.
        speech_presence_labels: AudioSet labels the gate treats as speech.
        failures: Optional dict to populate with per-model load/embed failure
            reasons (mirrors the existing audio_analysis ``failures`` pattern).

    Returns:
        ``(tagged_windows, info)`` where ``tagged_windows`` is the flat list of
        speech-windows tagged with ``file_id`` and ``model_id``, and ``info``
        is a small bookkeeping dict (``speech_seconds``, ``windows_total``,
        ``windows_kept``, ``windows_dropped_non_speech``, ``drop_reason`` if
        the file contributed nothing).
    """
    sr = audio.sampling_rate
    duration_s = audio.waveform.shape[-1] / sr if sr else 0.0

    info: dict[str, Any] = {
        "file_id": file_id,
        "duration_s": float(duration_s),
        "speech_seconds": 0.0,
        "windows_total": 0,
        "windows_kept": 0,
        "windows_dropped_non_speech": 0,
        "drop_reason": None,
    }

    # Hard floor: file shorter than the long window grid can't contribute.
    if duration_s < profile_window_s:
        info["drop_reason"] = "audio_too_short"
        return [], info

    # Extract per-window embeddings per model. The function builds the same
    # window grid for every model, so the speech mask we compute once applies
    # to all of them.
    per_model_windows: dict[str, list[WindowEmbedding]] = extract_per_window_embeddings(
        audio=audio,
        models=list(embedding_models),
        window_s=profile_window_s,
        hop_s=profile_hop_s,
        device=device,
        failures=failures,
    )
    if not per_model_windows or not any(per_model_windows.values()):
        info["drop_reason"] = "no_embedding_windows"
        return [], info

    # Use the first model's window grid for the speech mask (they share the
    # same grid by construction in extract_per_window_embeddings).
    reference_windows: list[WindowEmbedding] = next((w for w in per_model_windows.values() if w), [])
    mask: list[bool] | None = speech_window_mask_for_file(
        entries=reference_windows,
        pass_summary=pass_summary,
        speech_presence_labels=list(speech_presence_labels),
    )
    # ``None`` → no AST/YAMNet/loudness available; keep every window (legacy
    # behavior matches what cluster_pass_speakers does without a mask).
    if mask is None:
        mask = [True] * len(reference_windows)

    info["windows_total"] = len(reference_windows)

    tagged: list[TaggedWindowEmbedding] = []
    kept_window_seconds: list[float] = []
    for i, w in enumerate(reference_windows):
        if i >= len(mask) or not mask[i]:
            info["windows_dropped_non_speech"] += 1
            continue
        for model_id, windows in per_model_windows.items():
            if i >= len(windows):
                continue
            mw = windows[i]
            if mw.vector.size == 0:
                continue
            tagged.append(TaggedWindowEmbedding(file_id=file_id, model_id=model_id, window=mw))
        kept_window_seconds.append(float(w.end_s) - float(w.start_s))

    info["windows_kept"] = len(kept_window_seconds)
    info["speech_seconds"] = float(sum(kept_window_seconds))
    if info["windows_kept"] == 0 and info["drop_reason"] is None:
        info["drop_reason"] = "no_speech_windows"

    return tagged, info


# ──────────────────────────────────────────────────────────────────────────
# Cross-file dominant-cluster aggregation → per-model centroid + calibration band.
#
# The pooled, file-tagged windows are clustered ONCE using a reference model
# (the first available embedding model — typically ECAPA), reusing the existing
# contamination-tolerant ``cluster_pass_speakers`` (outlier rejection,
# silhouette gating, merge-close-clusters). The dominant cluster's windows then
# define, per model, an L2-normalized centroid and an empirical calibration
# band. Minority clusters (intruders, noise) are discarded — that is what gives
# the profile its tolerance to a moderate fraction of contaminating audio.
# ──────────────────────────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class _Slot:
    """One aligned time-window across models for a single file.

    ``extract_per_window_embeddings`` builds the same window grid for every
    model, so a ``(file_id, start_s, end_s)`` triple identifies the same slice
    of audio in each model's output. We group the flat ``TaggedWindowEmbedding``
    list back into slots so the centroid for every model is taken over the
    *same* set of dominant-cluster windows.
    """

    file_id: str
    start_s: float
    end_s: float
    vectors: dict[str, np.ndarray]  # model_id -> raw embedding vector

    @property
    def duration_s(self) -> float:
        return float(self.end_s) - float(self.start_s)


@dataclass(slots=True)
class AggregationResult:
    """Output of :func:`aggregate_dominant_cluster`."""

    centroids: dict[str, list[float]]
    calibration_band: dict[str, tuple[float, float]]
    dominant_cluster: ClusterStats
    runner_up_cluster: ClusterStats | None
    aggregate_speech_seconds: float
    # file_id -> (windows_in_dominant_cluster, speech_seconds_in_dominant_cluster)
    per_file_dominant: dict[str, tuple[int, float]]
    # file_id -> count of windows that entered clustering (contributed a reference vector)
    per_file_clustered: dict[str, int]


def _slot_key(file_id: str, start_s: float, end_s: float) -> tuple[str, int, int]:
    """Stable, float-tolerant key identifying one aligned window across models."""
    return (file_id, int(round(float(start_s) * 1000)), int(round(float(end_s) * 1000)))


def _unit(v: np.ndarray) -> np.ndarray | None:
    """L2-normalize ``v`` to a unit vector, or ``None`` if it has zero norm."""
    arr = np.asarray(v, dtype=np.float64).flatten()
    if arr.size == 0:
        return None
    norm = float(np.linalg.norm(arr))
    if norm <= 0:
        return None
    return arr / norm


def _weighted_unit_centroid(
    vectors: Sequence[np.ndarray],
    weights: Sequence[float],
) -> list[float]:
    """Weighted spherical mean: normalize each vector, weight, sum, re-normalize."""
    acc: np.ndarray | None = None
    for v, w in zip(vectors, weights, strict=False):
        u = _unit(v)
        if u is None or w <= 0:
            continue
        acc = (w * u) if acc is None else (acc + w * u)
    if acc is None:
        return []
    norm = float(np.linalg.norm(acc))
    if norm <= 0:
        return []
    return [float(x) for x in (acc / norm)]


def _session_weight(
    file_id: str,
    prefer_session: str | None,
    session_of_file: Mapping[str, str | None] | None,
) -> float:
    """Up-weight a window when its file's session matches ``prefer_session``."""
    if not prefer_session or session_of_file is None:
        return 1.0
    return C.SESSION_PREFERENCE_WEIGHT if session_of_file.get(file_id) == prefer_session else 1.0


def aggregate_dominant_cluster(
    tagged_windows: Sequence[TaggedWindowEmbedding],
    *,
    embedding_models: Sequence[str] = C.DEFAULT_EMBEDDING_MODELS,
    prefer_session: str | None = None,
    session_of_file: Mapping[str, str | None] | None = None,
    n_clusters_max: int = C.N_CLUSTERS_MAX,
    min_windows_for_clustering: int = C.MIN_WINDOWS_FOR_CLUSTERING,
    coherent_silhouette_threshold: float = C.COHERENT_SILHOUETTE_THRESHOLD,
    algorithm: str = C.CLUSTERING_ALGORITHM,
    failures: dict[str, str] | None = None,
) -> AggregationResult | None:
    """Cluster pooled windows and aggregate the dominant cluster into per-model centroids.

    The pooled windows are already speech-gated and carry no identity assignment;
    clustering on the reference model is what separates the target voice from
    contaminating voices.

    Args:
        tagged_windows: Pooled, file-tagged windows from
            :func:`extract_speech_windows_for_file` across all of a subject's files.
        embedding_models: Ordered model ids; the first present in the pool is the
            reference model whose vectors drive the clustering metric.
        prefer_session: Optional session id whose windows are up-weighted in
            dominant-cluster selection and centroid direction (never in the
            reported speech seconds). ``None`` means unweighted.
        session_of_file: Map of ``file_id -> session_id`` used to resolve
            ``prefer_session``. Ignored when ``prefer_session`` is ``None``.
        n_clusters_max: Maximum number of speaker clusters to consider.
        min_windows_for_clustering: Below this window count, fall back to a single
            cluster instead of partitioning.
        coherent_silhouette_threshold: Silhouette floor separating the
            multi-cluster regime from the single-cluster regime.
        algorithm: Clustering algorithm (``"spectral"`` or ``"kmeans"``).
        failures: Optional dict for surfacing skip reasons.

    Returns:
        An :class:`AggregationResult`, or ``None`` when there are no usable
        windows at all. A pooled set that clusters entirely to ``NOISE`` returns
        a result whose ``dominant_cluster`` has zero windows / seconds and empty
        ``centroids`` (the caller maps that to ``confidence="insufficient"``).
    """
    # 1. Re-assemble flat tagged windows into aligned per-slot records.
    slots_by_key: dict[tuple[str, int, int], _Slot] = {}
    slot_order: list[tuple[str, int, int]] = []
    for tw in tagged_windows:
        key = _slot_key(tw.file_id, tw.window.start_s, tw.window.end_s)
        slot = slots_by_key.get(key)
        if slot is None:
            slot = _Slot(file_id=tw.file_id, start_s=tw.window.start_s, end_s=tw.window.end_s, vectors={})
            slots_by_key[key] = slot
            slot_order.append(key)
        slot.vectors[tw.model_id] = np.asarray(tw.window.vector, dtype=np.float64).flatten()

    if not slot_order:
        return None

    slots: list[_Slot] = [slots_by_key[k] for k in slot_order]

    # 2. Choose the reference model: first requested model that any slot carries.
    reference_model: str | None = None
    for m in embedding_models:
        if any(m in s.vectors for s in slots):
            reference_model = m
            break
    if reference_model is None:
        if failures is not None:
            failures["profile_aggregation"] = "no requested embedding model present in pooled windows"
        return None

    ref_slots = [s for s in slots if reference_model in s.vectors]
    ref_entries = [
        WindowEmbedding(start_s=s.start_s, end_s=s.end_s, vector=s.vectors[reference_model]) for s in ref_slots
    ]

    cluster_res = cluster_pass_speakers(
        ref_entries,
        n_clusters_max=n_clusters_max,
        min_windows_for_clustering=min_windows_for_clustering,
        coherent_silhouette_threshold=coherent_silhouette_threshold,
        algorithm=algorithm,
        failures=failures,
        failure_key="profile_clustering",
    )
    if cluster_res is None:
        if failures is not None:
            failures.setdefault("profile_aggregation", "clustering returned no result (sklearn unavailable?)")
        return None

    labels: dict[int, str] = cluster_res.get("labels", {})
    best_silhouette: float | None = cluster_res.get("best_silhouette")

    # 3. Group reference slots by cluster label, ignoring NOISE.
    cluster_to_idx: dict[str, list[int]] = {}
    for i in range(len(ref_slots)):
        lbl = labels.get(i, "NOISE")
        if lbl == "NOISE":
            continue
        cluster_to_idx.setdefault(lbl, []).append(i)

    per_file_clustered: dict[str, int] = {}
    for s in ref_slots:
        per_file_clustered[s.file_id] = per_file_clustered.get(s.file_id, 0) + 1

    def _raw_seconds(idxs: Sequence[int]) -> float:
        return float(sum(ref_slots[i].duration_s for i in idxs))

    def _weighted_seconds(idxs: Sequence[int]) -> float:
        return float(
            sum(
                ref_slots[i].duration_s * _session_weight(ref_slots[i].file_id, prefer_session, session_of_file)
                for i in idxs
            )
        )

    n_clustered = sum(len(idxs) for idxs in cluster_to_idx.values())

    if not cluster_to_idx or n_clustered == 0:
        # Everything fell to NOISE — no usable dominant voice.
        return AggregationResult(
            centroids={},
            calibration_band={},
            dominant_cluster=ClusterStats(n_windows=0, speech_seconds=0.0, silhouette=best_silhouette, share=0.0),
            runner_up_cluster=None,
            aggregate_speech_seconds=0.0,
            per_file_dominant={},
            per_file_clustered=per_file_clustered,
        )

    # 4. Dominant = cluster with most session-weighted speech; tie-break on raw
    #    seconds then window count so selection is deterministic.
    ordered = sorted(
        cluster_to_idx.items(),
        key=lambda kv: (_weighted_seconds(kv[1]), _raw_seconds(kv[1]), len(kv[1])),
        reverse=True,
    )
    dominant_label, dominant_idx = ordered[0]
    runner_up_label, runner_up_idx = ordered[1] if len(ordered) > 1 else (None, [])

    # 5. Per-model centroid + calibration band over the clustered slots.
    label_to_int = {lbl: n for n, lbl in enumerate(sorted(cluster_to_idx))}
    centroids: dict[str, list[float]] = {}
    calibration_band: dict[str, tuple[float, float]] = {}
    for model in embedding_models:
        dom_vectors = [ref_slots[i].vectors[model] for i in dominant_idx if model in ref_slots[i].vectors]
        dom_weights = [
            _session_weight(ref_slots[i].file_id, prefer_session, session_of_file)
            for i in dominant_idx
            if model in ref_slots[i].vectors
        ]
        centroid = _weighted_unit_centroid(dom_vectors, dom_weights)
        if not centroid:
            continue
        centroids[model] = centroid

        # Calibration band over ALL clustered slots that carry this model.
        band_units: list[np.ndarray] = []
        band_labels: list[int] = []
        for lbl, idxs in cluster_to_idx.items():
            for i in idxs:
                if model not in ref_slots[i].vectors:
                    continue
                u = _unit(ref_slots[i].vectors[model])
                if u is None:
                    continue
                band_units.append(u)
                band_labels.append(label_to_int[lbl])
        if len(band_units) >= 2:
            same_floor, diff_floor = _empirical_calibration_band(np.stack(band_units, axis=0), np.asarray(band_labels))
        else:
            same_floor, diff_floor = C.SAME_SPEAKER_FLOOR_FALLBACK, C.DIFF_SPEAKER_FLOOR_FALLBACK
        calibration_band[model] = (same_floor, diff_floor)

    dominant_seconds = _raw_seconds(dominant_idx)
    dominant_stats = ClusterStats(
        n_windows=len(dominant_idx),
        speech_seconds=dominant_seconds,
        silhouette=best_silhouette,
        share=(len(dominant_idx) / n_clustered) if n_clustered else 0.0,
    )
    runner_up_stats: ClusterStats | None = None
    if runner_up_label is not None:
        runner_up_stats = ClusterStats(
            n_windows=len(runner_up_idx),
            speech_seconds=_raw_seconds(runner_up_idx),
            silhouette=best_silhouette,
            share=(len(runner_up_idx) / n_clustered) if n_clustered else 0.0,
        )

    per_file_dominant: dict[str, tuple[int, float]] = {}
    for i in dominant_idx:
        fid = ref_slots[i].file_id
        n_prev, s_prev = per_file_dominant.get(fid, (0, 0.0))
        per_file_dominant[fid] = (n_prev + 1, s_prev + ref_slots[i].duration_s)

    return AggregationResult(
        centroids=centroids,
        calibration_band=calibration_band,
        dominant_cluster=dominant_stats,
        runner_up_cluster=runner_up_stats,
        aggregate_speech_seconds=dominant_seconds,
        per_file_dominant=per_file_dominant,
        per_file_clustered=per_file_clustered,
    )


# ──────────────────────────────────────────────────────────────────────────
# Confidence policy.
# ──────────────────────────────────────────────────────────────────────────


def decide_confidence(
    *,
    dominant_speech_seconds: float,
    runner_up_speech_seconds: float,
    has_dominant: bool,
    min_confident_speech_s: float = C.MIN_CONFIDENT_SPEECH_S,
    target_confident_speech_s: float = C.TARGET_CONFIDENT_SPEECH_S,
    ambiguity_share_ratio: float = C.AMBIGUITY_SHARE_RATIO,
) -> ProfileConfidence:
    """Map aggregation outcome → a :data:`ProfileConfidence` label.

    Policy (honoring the schema invariant
    ``confidence=="low" ⟺ 0 < aggregate < min_confident_speech_s`` with a
    coherent cluster):

    - **insufficient** — no coherent dominant cluster, or zero aggregate speech.
    - **ambiguous** — a runner-up cluster rivals the dominant one
      (``runner_up / dominant >= ambiguity_share_ratio``); takes precedence
      because the identity of the target voice is in doubt regardless of duration.
    - **low** — coherent dominant cluster but ``0 < aggregate < min``.
    - **ok** — coherent dominant cluster with ``aggregate >= min``.

    ``target_confident_speech_s`` is not the ok/low boundary (the schema fixes
    that at ``min``); it is stamped into the artifact params as the
    "comfortably confident" level for downstream consumers and threshold tuning.
    """
    if not has_dominant or dominant_speech_seconds <= 0:
        return "insufficient"
    if (
        runner_up_speech_seconds > 0
        and dominant_speech_seconds > 0
        and (runner_up_speech_seconds / dominant_speech_seconds) >= ambiguity_share_ratio
    ):
        return "ambiguous"
    if dominant_speech_seconds < min_confident_speech_s:
        return "low"
    return "ok"


# ──────────────────────────────────────────────────────────────────────────
# Per-file keep/drop decisions + ProfileSourceFile usage records.
# ──────────────────────────────────────────────────────────────────────────

# Map the extractor's window-level drop reasons onto the artifact's documented
# ``ProfileSourceFile.drop_reason`` vocabulary.
_DROP_REASON_MAP: dict[str, str] = {
    "audio_too_short": "insufficient_speech",
    "no_embedding_windows": "insufficient_speech",
    "no_speech_windows": "non_speech_task",
}


def build_source_records(
    *,
    file_infos: Sequence[dict[str, Any]],
    aggregation: AggregationResult | None,
    audio_signatures: Mapping[str, str],
    session_of_file: Mapping[str, str | None] | None = None,
) -> list[ProfileSourceFile]:
    """One :class:`ProfileSourceFile` per ingested file (kept or dropped).

    A file is **kept** iff it contributed ≥1 window to the dominant cluster.
    Files that produced no usable windows carry their extractor drop reason
    (mapped to the artifact vocabulary); files that clustered but landed outside
    the dominant cluster get ``"outside_dominant_cluster"``.
    """
    per_file_dominant = aggregation.per_file_dominant if aggregation else {}
    per_file_clustered = aggregation.per_file_clustered if aggregation else {}

    records: list[ProfileSourceFile] = []
    for info in file_infos:
        file_id = str(info.get("file_id", ""))
        session = session_of_file.get(file_id) if session_of_file else None
        sig = audio_signatures.get(file_id, "")

        dom_windows, dom_seconds = per_file_dominant.get(file_id, (0, 0.0))
        if dom_windows > 0:
            records.append(
                ProfileSourceFile(
                    file_id=file_id,
                    audio_signature=sig,
                    session_id=session,
                    speech_seconds_used=float(dom_seconds),
                    windows_used=int(dom_windows),
                    kept=True,
                    drop_reason=None,
                )
            )
            continue

        # Not in the dominant cluster — decide why.
        if per_file_clustered.get(file_id, 0) > 0:
            drop_reason = "outside_dominant_cluster"
        else:
            raw_reason = info.get("drop_reason")
            drop_reason = (
                _DROP_REASON_MAP.get(str(raw_reason), str(raw_reason)) if raw_reason else "insufficient_speech"
            )
        records.append(
            ProfileSourceFile(
                file_id=file_id,
                audio_signature=sig,
                session_id=session,
                speech_seconds_used=0.0,
                windows_used=0,
                kept=False,
                drop_reason=drop_reason,
            )
        )
    return records


# ──────────────────────────────────────────────────────────────────────────
# Orchestration entrypoint.
# ──────────────────────────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class ProfileInput:
    """One file fed to :func:`build_speaker_profile`.

    ``pass_summary`` carries whatever speech-gating signals the caller already
    has cached (AST / YAMNet / openSMILE); an empty dict makes the gate fall
    back to "every non-zero window" (see :func:`extract_speech_windows_for_file`).
    """

    audio: Audio
    file_id: str
    session_id: str | None = None
    pass_summary: dict[str, Any] | None = None


def build_speaker_profile(
    subject_id: str,
    inputs: Sequence[ProfileInput],
    *,
    embedding_models: Sequence[str] = C.DEFAULT_EMBEDDING_MODELS,
    profile_window_s: float = C.PROFILE_WINDOW_S,
    profile_hop_s: float = C.PROFILE_HOP_S,
    min_confident_speech_s: float = C.MIN_CONFIDENT_SPEECH_S,
    target_confident_speech_s: float = C.TARGET_CONFIDENT_SPEECH_S,
    ambiguity_share_ratio: float = C.AMBIGUITY_SHARE_RATIO,
    prefer_session: str | None = None,
    device: DeviceType | None = None,
    output: Path | None = None,
) -> SpeakerProfile:
    """Build exactly one contamination-tolerant profile for ``subject_id``.

    Ties together per-file speech-window extraction, cross-file dominant-cluster
    aggregation, the confidence policy, and per-file usage records into a single
    :class:`SpeakerProfile`. When ``output`` is given the artifact is also
    written via :func:`io.save_profile`.

    Per-file and per-model failures are non-fatal: they are recorded in
    ``provenance["failures"]`` and the affected file's ``drop_reason``, never
    raised (matches the existing ``failures`` pattern).
    """
    failures: dict[str, str] = {}
    session_of_file: dict[str, str | None] = {inp.file_id: inp.session_id for inp in inputs}
    audio_signatures: dict[str, str] = {}

    pooled: list[TaggedWindowEmbedding] = []
    file_infos: list[dict[str, Any]] = []
    for inp in inputs:
        try:
            audio_signatures[inp.file_id] = audio_signature(inp.audio)
        except Exception as exc:  # noqa: BLE001
            failures[f"signature/{inp.file_id}"] = repr(exc)
            audio_signatures[inp.file_id] = ""
        tagged, info = extract_speech_windows_for_file(
            audio=inp.audio,
            file_id=inp.file_id,
            pass_summary=inp.pass_summary or {},
            embedding_models=embedding_models,
            device=device,
            profile_window_s=profile_window_s,
            profile_hop_s=profile_hop_s,
            failures=failures,
        )
        pooled.extend(tagged)
        file_infos.append(info)

    aggregation = aggregate_dominant_cluster(
        pooled,
        embedding_models=embedding_models,
        prefer_session=prefer_session,
        session_of_file=session_of_file,
        failures=failures,
    )

    has_dominant = aggregation is not None and bool(aggregation.centroids)
    runner_up_seconds = (
        aggregation.runner_up_cluster.speech_seconds
        if aggregation and aggregation.runner_up_cluster is not None
        else 0.0
    )
    aggregate_seconds = aggregation.aggregate_speech_seconds if aggregation else 0.0

    confidence = decide_confidence(
        dominant_speech_seconds=aggregate_seconds,
        runner_up_speech_seconds=runner_up_seconds,
        has_dominant=has_dominant,
        min_confident_speech_s=min_confident_speech_s,
        target_confident_speech_s=target_confident_speech_s,
        ambiguity_share_ratio=ambiguity_share_ratio,
    )

    sources = build_source_records(
        file_infos=file_infos,
        aggregation=aggregation,
        audio_signatures=audio_signatures,
        session_of_file=session_of_file,
    )

    params = ProfileParams(
        embedding_models=list(embedding_models),
        profile_window_s=profile_window_s,
        profile_hop_s=profile_hop_s,
        detect_window_s=C.DETECT_WINDOW_S,
        detect_hop_s=C.DETECT_HOP_S,
        min_confident_speech_s=min_confident_speech_s,
        target_confident_speech_s=target_confident_speech_s,
        ambiguity_share_ratio=ambiguity_share_ratio,
        prefer_session=prefer_session,
    )

    # The runner-up is only meaningful as the ambiguity witness (schema invariant
    # ``runner_up_cluster is non-null iff confidence == "ambiguous"``).
    runner_up_cluster = aggregation.runner_up_cluster if (aggregation and confidence == "ambiguous") else None
    dominant_cluster = (
        aggregation.dominant_cluster
        if aggregation
        else ClusterStats(n_windows=0, speech_seconds=0.0, silhouette=None, share=0.0)
    )

    provenance: dict[str, Any] = {
        "senselab_version": senselab_version(),
        "schema_version": SCHEMA_VERSION,
        "cache_key_basis": f"module:{__name__}",
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reference_model": next(iter(aggregation.centroids), None) if aggregation else None,
    }
    if failures:
        provenance["failures"] = dict(failures)

    profile = SpeakerProfile(
        subject_id=subject_id,
        centroids=aggregation.centroids if aggregation else {},
        confidence=confidence,
        aggregate_speech_seconds=aggregate_seconds,
        dominant_cluster=dominant_cluster,
        runner_up_cluster=runner_up_cluster,
        calibration_band=aggregation.calibration_band if aggregation else {},
        sources=sources,
        params=params,
        provenance=provenance,
    )

    if output is not None:
        save_profile(profile, Path(output))

    return profile
