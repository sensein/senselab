"""Per-window speaker-embedding extraction.

Slices the pass audio into uniform fixed-duration windows (default 2.0 s with 1.0 s
hop) and runs each requested embedding model on every window. Output is a list of
``(start_s, end_s, vector_np)`` per model, written to disk by the caller and consumed
by the speaker workflow.

Why windows, not diarization segments
-------------------------------------

The embedding signal is meant to be an **independent** check on the diarization model's
segmentation. If the diarizer merges two speakers into one segment, segment-anchored
embeddings inherit that mistake. Per-window embeddings are computed on a uniform time
grid that doesn't depend on any model's segmentation, so cosine distance between
adjacent windows is a model-free indicator of speaker change.

Why 1.0 s / 0.5 s defaults
--------------------------

ECAPA / ResNet are trained on multi-second utterances and SpeechBrain's pipeline
forwards short clips through their stat-pooling layers without internal padding,
so the embedding is *noisier* below 1 s — but still functional. We trade some
embedding precision for finer temporal resolution: a 1 s window with a 0.5 s
hop gives one embedding per 0.5 s bucket, eliminating the same-window dedup
that previously dropped half of consecutive same-cluster comparisons. Down-
stream calibration via ``calibrate_cosine_uncertainty`` already accounts for
the noisier same-speaker baseline. The caller can override via ``window_s``
and ``hop_s``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.speaker_embeddings import extract_speaker_embeddings_from_audios
from senselab.utils.data_structures import DeviceType, SpeechBrainModel


@dataclass(frozen=True)
class WindowEmbedding:
    """One embedding vector for a fixed time window."""

    start_s: float
    end_s: float
    vector: np.ndarray


def _slice_audio(audio: Audio, start_s: float, end_s: float) -> Audio:
    """Return a new ``Audio`` covering ``[start_s, end_s]`` (clamped to audio bounds).

    Assumes the input is shaped ``(channels, samples)``. Multichannel audio is
    sliced unchanged — the embedding backend's own preprocessor handles it.
    """
    sr = audio.sampling_rate
    n_samples = audio.waveform.shape[-1]
    start_sample = max(0, int(start_s * sr))
    end_sample = min(n_samples, max(start_sample + 1, int(end_s * sr)))
    waveform = audio.waveform[:, start_sample:end_sample]
    return Audio(waveform=waveform, sampling_rate=sr)


def _window_starts(duration_s: float, window_s: float, hop_s: float) -> list[float]:
    """Return window start times spanning ``[0, duration_s]``.

    The last window is anchored to ``duration_s - window_s`` (or 0) so it always
    covers exactly ``window_s`` seconds — the embedding model needs the full
    minimum-length input.
    """
    if duration_s <= 0 or window_s <= 0 or hop_s <= 0:
        return []
    if duration_s <= window_s:
        return [0.0]
    starts: list[float] = []
    t = 0.0
    while t + window_s <= duration_s + 1e-9:
        starts.append(t)
        t += hop_s
    # Pin the last window to the audio tail so we don't drop the final speaker turn.
    last = max(0.0, duration_s - window_s)
    if not starts or starts[-1] < last - 1e-6:
        starts.append(last)
    return starts


def extract_per_window_embeddings(
    *,
    audio: Audio,
    models: list[str],
    window_s: float = 1.0,
    hop_s: float = 0.5,
    device: DeviceType | None = None,
    failures: dict[str, str] | None = None,
) -> dict[str, list[WindowEmbedding]]:
    """Run each embedding model on every fixed window of ``audio``.

    Args:
        audio: The pass's full ``Audio`` object.
        models: HuggingFace model ids for the embedding backends (ECAPA, ResNet, ...).
        window_s: Window length in seconds. Defaults to 1.0.
        hop_s: Window hop in seconds. Defaults to 0.5.
        device: Optional device override.
        failures: Optional dict the function will populate with
            ``{model_id → reason}`` for any model that produced an empty result.
            The caller can fold these into ``incomparable_reasons`` so silent
            empty-vote behavior is auditable.

    Returns:
        ``{model_id → [WindowEmbedding, ...]}``. Each list shares the same window
        grid across models, so ``out[m_a][i]`` and ``out[m_b][i]`` cover the same
        time span. A model that fails to load returns ``[]`` (and writes a reason
        into ``failures`` when provided).
    """
    if not models:
        return {}
    if audio.waveform.ndim < 2:
        raise ValueError(
            f"extract_per_window_embeddings expects a (channels, samples) waveform; "
            f"got shape {tuple(audio.waveform.shape)}."
        )
    duration_s = audio.waveform.shape[-1] / audio.sampling_rate
    starts = _window_starts(duration_s, window_s, hop_s)
    if not starts:
        msg = (
            f"audio duration ({duration_s:.3f} s) is shorter than the embedding "
            f"window ({window_s} s); no window grid producible"
        )
        print(f"warn: {msg}", file=sys.stderr)
        if failures is not None:
            for m in models:
                failures[m] = msg
        return {m: [] for m in models}

    audio_slices: list[Audio] = []
    spans: list[tuple[float, float]] = []
    for s in starts:
        e = min(duration_s, s + window_s)
        audio_slices.append(_slice_audio(audio, s, e))
        spans.append((s, e))

    out: dict[str, list[WindowEmbedding]] = {}
    for model_id in models:
        try:
            sb_model: SpeechBrainModel = SpeechBrainModel(path_or_uri=model_id, revision="main")
            tensors = extract_speaker_embeddings_from_audios(audios=audio_slices, model=sb_model, device=device)
            entries: list[WindowEmbedding] = []
            for (start_s, end_s), t in zip(spans, tensors, strict=False):
                vec = _flatten_to_1d(t)
                entries.append(WindowEmbedding(start_s=start_s, end_s=end_s, vector=vec))
            out[model_id] = entries
        except Exception as exc:  # noqa: BLE001
            # Surface per-model failure to the caller via stderr so the user can
            # tell "model crashed" from "audio too short" — both produce an empty
            # list, but only one is a real configuration problem.
            msg = f"model failed during extraction: {exc!r}"
            print(
                f"warn: speaker-embedding model {model_id!r} {msg}",
                file=sys.stderr,
            )
            if failures is not None:
                failures[model_id] = msg
            out[model_id] = []
    return out


def _flatten_to_1d(t: Any) -> np.ndarray:  # noqa: ANN401
    """Coerce an embedding tensor / array to a 1-D ``float32`` numpy vector.

    SpeechBrain returns ``(1, D)``, ``(B, D)``, or ``(K, D)`` depending on the
    backend; ``.squeeze()`` would silently leave a 2-D ``(K, D)`` if K>1, which
    the cosine helper then sees as a length-mismatch and drops to ``None``. Here
    we always reshape to a single 1-D vector — when the backend returns multiple
    candidate vectors we average them (the closest the workflow can do without
    backend-specific knowledge).
    """
    if t is None:
        return np.zeros(0, dtype=np.float32)
    if isinstance(t, torch.Tensor):
        arr = t.detach().cpu().numpy()
    else:
        arr = np.asarray(t)
    if arr.size == 0:
        return np.zeros(0, dtype=np.float32)
    while arr.ndim > 1 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim > 1:
        # Multiple candidate vectors → average to a single representative.
        arr = arr.mean(axis=tuple(range(arr.ndim - 1)))
    return arr.astype(np.float32, copy=False)


def window_embedding_at(
    entries: list[WindowEmbedding],
    t: float,
) -> WindowEmbedding | None:
    """Return the window whose midpoint is closest to ``t``.

    Used by the plot's adjacent-cosine row (build a uniform per-window series).
    Prefer ``window_index_at`` when the caller needs to detect "same window as
    last lookup" — index equality is the only way to tell if two consecutive
    buckets inherit the same window's embedding (which makes their cosine
    distance trivially zero and meaningless).
    """
    idx = window_index_at(entries, t)
    return entries[idx] if idx is not None else None


def window_index_at(
    entries: list[WindowEmbedding],
    t: float,
) -> int | None:
    """Return the index of the window whose midpoint is closest to ``t``.

    Ties (e.g. ``t`` exactly between two adjacent overlapping windows) resolve
    to the earlier window deterministically.
    """
    if not entries:
        return None
    best_idx: int | None = None
    best_dist = float("inf")
    for i, w in enumerate(entries):
        center = 0.5 * (w.start_s + w.end_s)
        d = abs(center - t)
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx


def cluster_pass_speakers(
    entries: list[WindowEmbedding],
    *,
    n_clusters_max: int = 6,
    min_windows_for_clustering: int = 4,
    coherent_silhouette_threshold: float = 0.10,
    failures: dict[str, str] | None = None,
    failure_key: str = "embedding_clustering",
    is_speech_per_window: list[bool] | None = None,
    algorithm: str = "spectral",
) -> dict[str, Any] | None:
    r"""Estimate ``n_speakers`` for the pass and return per-window cluster + silhouette info.

    Args:
        entries: Per-window embeddings to cluster.
        n_clusters_max: Maximum number of speaker clusters to consider.
        min_windows_for_clustering: Minimum window count needed to attempt clustering.
        coherent_silhouette_threshold: Silhouette score below which we treat the
            best k-cluster solution as ``n_speakers <= 1`` (no inter-cluster
            separation).
        failures: Optional caller-supplied dict for surfacing skip reasons.
        failure_key: Key under which to record any failure.
        is_speech_per_window: Boolean mask per ``entries`` index indicating
            whether the window contains speech (per YAMNet / AST / loudness).
            When provided, only ``True`` windows participate in clustering;
            ``False`` windows get ``cluster_id="NOISE"`` and ``p_voice = 0.0``.
            This stops silent / background windows from being counted as a
            "speaker" — the bug that previously inflated ``n_speakers`` on
            recordings with long silent stretches. ``None`` clusters every
            non-zero-norm window (legacy behavior).
        algorithm: ``"spectral"`` (default) or ``"kmeans"``. Spectral
            clustering on a precomputed cosine-similarity affinity matrix
            handles non-convex cluster shapes better than k-means and is the
            standard choice for speaker diarization. K-means stays as the
            fallback when sklearn's ``SpectralClustering`` errors (e.g.,
            degenerate affinity matrix).

    Algorithm:

    1. **Filter windows** — drop zero-norm vectors and (if ``is_speech_per_window``
       is supplied) non-speech windows.
    2. **Sweep k = 2 … min(n_clusters_max, n_speech − 1)** running the chosen
       clustering algorithm on L2-normalized embedding vectors.
    3. **Pick k\* = argmax silhouette_score** across the sweep.
    4. **If best silhouette ≥ coherent_silhouette_threshold** → multi-cluster regime:
       ``n_speakers = k\*``; per-window ``p_voice`` is the rescaled silhouette.
    5. **Else** → single-cluster regime (1 if the speech windows cluster
       tightly around the mean, 0 if even that fails).

    The output ``labels`` use ``"spk0"``, ``"spk1"``, … — this component's own speaker
    labels, in the same role as pyannote's ``SPEAKER_00``. They are distinct from the
    pass-wide cluster ids (``C0``…) that harmonise labels across diar models, and from the
    fused speaker ids (``S0``…) in ``final/speakers.json``; three namespaces that all read
    as "S0" made the label-correspondence table unreadable on a real run. Non-speech
    or zero-norm windows are tagged ``"NOISE"``. ``p_voice`` is keyed by the
    window index in ``entries``.

    Returns ``None`` when too few windows to cluster, or sklearn unavailable.
    """
    try:
        from sklearn.cluster import KMeans, SpectralClustering
        from sklearn.metrics import silhouette_samples, silhouette_score
    except ImportError as exc:
        msg = f"sklearn unavailable for clustering: {exc!r}"
        print(f"warn: cluster_pass_speakers skipped: {msg}", file=sys.stderr)
        if failures is not None:
            failures[failure_key] = msg
        return None

    vectors: list[np.ndarray] = []
    valid_indices: list[int] = []
    for i, w in enumerate(entries):
        if w.vector.size == 0:
            continue
        norm = float(np.linalg.norm(w.vector))
        if norm <= 0:
            continue
        # Speech mask filter — non-speech windows skip clustering entirely so
        # they don't get counted as a speaker cluster.
        if is_speech_per_window is not None and i < len(is_speech_per_window):
            if not is_speech_per_window[i]:
                continue
        vectors.append(np.asarray(w.vector, dtype=np.float64) / norm)
        valid_indices.append(i)

    cluster_labels: dict[int, str] = {}
    p_voice: dict[int, float] = {}
    for i in range(len(entries)):
        if i not in valid_indices:
            cluster_labels[i] = "NOISE"
            p_voice[i] = 0.0

    if not valid_indices:
        msg = "no valid speech embedding windows (all filtered as zero-norm or non-speech)"
        print(f"warn: cluster_pass_speakers: {msg}", file=sys.stderr)
        if failures is not None:
            failures[failure_key] = msg
        return {
            "n_speakers": 0,
            "best_silhouette": None,
            "labels": cluster_labels,
            "p_voice": p_voice,
            "valid_indices": [],
        }

    # Single-asr / quiet-environment path: when we have ≥1 valid speech
    # window but fewer than ``min_windows_for_clustering``, there's nothing to
    # partition — the system should report exactly one speaker, not skip. This
    # is the "single word in an otherwise quiet recording" case.
    if len(vectors) < min_windows_for_clustering:
        for idx in valid_indices:
            cluster_labels[idx] = "spk0"
            p_voice[idx] = 1.0
        band = _within_cluster_band(np.stack(vectors, axis=0)) if vectors else None
        same_floor, diff_floor = band if band is not None else (None, None)
        return {
            "n_speakers": 1,
            "best_silhouette": None,
            "labels": cluster_labels,
            "p_voice": p_voice,
            "valid_indices": list(valid_indices),
            "empirical_same_speaker_floor": same_floor,
            "empirical_diff_speaker_floor": diff_floor,
            "single_window_mode": True,
        }
    X = np.stack(vectors, axis=0)

    algorithm_used: str | None = None

    def _fit_predict(k: int) -> np.ndarray | None:
        """Fit the chosen clustering algorithm and return labels, or ``None`` on failure."""
        nonlocal algorithm_used
        if algorithm == "spectral":
            try:
                # Precomputed cosine-similarity affinity. SpectralClustering
                # requires non-negative weights; clamp negative similarities
                # to 0 (they correspond to anti-aligned embeddings — rare for
                # speaker embeddings but possible).
                affinity = np.maximum(X @ X.T, 0.0)
                sc = SpectralClustering(
                    n_clusters=k,
                    affinity="precomputed",
                    assign_labels="kmeans",
                    random_state=0,
                    n_init=5,
                )
                labels = sc.fit_predict(affinity)
                algorithm_used = "spectral"
                return labels
            except Exception as exc:  # noqa: BLE001
                msg = f"spectral clustering at k={k} failed ({exc!r}); falling back to k-means"
                print(f"warn: {msg}", file=sys.stderr)
                if failures is not None:
                    failures[f"{failure_key}/spectral_k{k}"] = msg
        try:
            km = KMeans(n_clusters=k, n_init=5, random_state=0)
            labels = km.fit_predict(X)
            algorithm_used = "kmeans" if algorithm_used != "spectral" else algorithm_used
            return labels
        except Exception as exc:  # noqa: BLE001
            msg = f"clustering at k={k} failed: {exc!r}"
            print(f"warn: {msg}", file=sys.stderr)
            if failures is not None:
                failures[f"{failure_key}/algorithm_k{k}"] = msg
            return None

    best_k = 1
    best_overall = -1.0
    best_labels: np.ndarray | None = None
    k_max = min(n_clusters_max, len(vectors) - 1)
    # Tiny clusters (< this fraction of all clustered windows) are treated as
    # outliers, not speakers — silhouette can otherwise reward solutions where
    # a handful of cross-talk / overlap / mislabeled windows props up the
    # score by sitting in their own corner of the embedding space.
    #
    # Floor of 2 (not 5) so the size gate is reachable even on tiny passes.
    # On a 6-window pass, 0.10 * 6 = 0.6 → max(2, 1) = 2 windows per cluster
    # (k=2 needs ≥4 windows total). On a typical 60-window pass, max(2, 6) = 6
    # rejects 3-window outliers without dropping real speakers.
    min_cluster_fraction = 0.10
    min_cluster_size = max(2, int(round(min_cluster_fraction * len(vectors))))
    rejected_for_min_size = 0
    rejected_for_silhouette = 0
    for k in range(2, k_max + 1):
        labels = _fit_predict(k)
        if labels is None:
            continue
        unique_labels, counts = np.unique(labels, return_counts=True)
        if len(unique_labels) < 2:
            rejected_for_silhouette += 1
            continue
        # Reject the partition if any cluster is too small to be a speaker.
        if int(counts.min()) < min_cluster_size:
            rejected_for_min_size += 1
            continue
        try:
            score = silhouette_score(X, labels, metric="cosine")
        except ValueError:
            continue
        if score > best_overall:
            best_overall = score
            best_k = k
            best_labels = labels
    if best_labels is None and rejected_for_min_size > 0 and failures is not None:
        # Surface the silent-fall-through case: every candidate k produced a
        # partition with at least one too-small cluster. The caller might see
        # n_speakers=1 here when the audio actually has multiple speakers but
        # one of them spoke too briefly to clear the size floor.
        failures[f"{failure_key}/all_partitions_under_min_size"] = (
            f"all {rejected_for_min_size} candidate k partitions had a cluster < {min_cluster_size} windows; "
            f"falling through to single-cluster regime"
        )

    if best_labels is not None and best_overall >= coherent_silhouette_threshold:
        # Same-speaker post-merge step. Spectral / k-means at k≥2 sometimes
        # splits one speaker's recording into sub-clusters when prosody /
        # distance / phonation varies. Centroids with cosine similarity
        # ≥ ``merge_threshold`` are taken to be the same speaker and merged.
        # This pass does NOT assume any target speaker count — it only
        # collapses pairs that look like the same person by cosine.
        #
        # Threshold tradeoff:
        #   - ECAPA same-speaker centroid cos_sim is typically 0.6+ on adult
        #     VoxCeleb when phonetic content is reasonably matched.
        #   - Different speakers with similar timbre (same gender + age,
        #     children, family resemblance) can sit at cos_sim ~0.30 —
        #     these are genuinely different people and must NOT be merged.
        #
        # 0.55 keeps such similar-voice distinct speakers separated while
        # still folding in clear-cut same-speaker prosodic outliers. Audio
        # with heavily varying within-speaker prosody (impressions, extreme
        # distance changes) may still over-split; lower the threshold only
        # after re-validating against any similar-voice pair you care about.
        merge_threshold = 0.55
        best_labels = _merge_close_clusters(X, best_labels, merge_threshold=merge_threshold)
        unique_after_merge = sorted(set(int(x) for x in best_labels))
        n_speakers_final = len(unique_after_merge)
        # Renumber to spk0..spk(n-1) so downstream label sets are dense.
        relabel = {old: new for new, old in enumerate(unique_after_merge)}
        best_labels = np.array([relabel[int(x)] for x in best_labels], dtype=int)
        try:
            per_sample = silhouette_samples(X, best_labels, metric="cosine") if n_speakers_final >= 2 else None
        except ValueError:
            per_sample = None
        for vi, idx in enumerate(valid_indices):
            cluster_labels[idx] = f"spk{int(best_labels[vi])}"
            if per_sample is None:
                p_voice[idx] = 1.0 if n_speakers_final == 1 else 0.5
            else:
                p_voice[idx] = max(0.0, min(1.0, 0.5 * (float(per_sample[vi]) + 1.0)))
        # Empirical per-pass calibration band for the speaker-axis cosine
        # validation: ``same_floor`` = 75th percentile of within-cluster
        # pairwise cos_dist (most same-speaker pairs are below this),
        # ``diff_floor`` = 25th percentile of between-cluster pairwise cos_dist
        # (most different-speaker pairs are above this). Falls back to the
        # default fixed band when too few pairs.
        # Sequential first: it measures the statistic the harvester computes. All-pairs is
        # kept as a fallback for passes whose turn structure gives too few sequential
        # comparisons to anchor on.
        band = _sequential_calibration_band(X, best_labels) or _empirical_calibration_band(X, best_labels)
        same_floor, diff_floor = band if band is not None else (None, None)
        return {
            "n_speakers": n_speakers_final,
            "best_silhouette": float(best_overall),
            "labels": cluster_labels,
            "p_voice": p_voice,
            "valid_indices": list(valid_indices),
            "empirical_same_speaker_floor": same_floor,
            "empirical_diff_speaker_floor": diff_floor,
            "algorithm": algorithm_used or algorithm,
            "merged_from_k": int(best_k) if int(best_k) != n_speakers_final else None,
        }

    # Single-cluster regime: no clear inter-cluster separation.
    centroid = X.mean(axis=0)
    centroid_norm = float(np.linalg.norm(centroid))
    if centroid_norm == 0:
        # All-zero pass — no detectable voice.
        for idx in valid_indices:
            cluster_labels[idx] = "NOISE"
            p_voice[idx] = 0.0
        return {
            "n_speakers": 0,
            "best_silhouette": float(best_overall) if best_overall > -1.0 else None,
            "labels": cluster_labels,
            "p_voice": p_voice,
            "valid_indices": list(valid_indices),
        }
    centroid_unit = centroid / centroid_norm
    sims = X @ centroid_unit
    mean_sim = float(np.mean(sims))
    if mean_sim >= coherent_silhouette_threshold:
        # Single coherent speaker.
        for vi, idx in enumerate(valid_indices):
            cluster_labels[idx] = "spk0"
            # cos sim → [0,1] via (s+1)/2.
            p_voice[idx] = max(0.0, min(1.0, 0.5 * (float(sims[vi]) + 1.0)))
        band = _within_cluster_band(X)
        same_floor, diff_floor = band if band is not None else (None, None)
        return {
            "n_speakers": 1,
            "best_silhouette": float(best_overall) if best_overall > -1.0 else None,
            "labels": cluster_labels,
            "p_voice": p_voice,
            "valid_indices": list(valid_indices),
            "empirical_same_speaker_floor": same_floor,
            "empirical_diff_speaker_floor": diff_floor,
        }
    # No coherent cluster — likely noise / silence dominated.
    for vi, idx in enumerate(valid_indices):
        cluster_labels[idx] = "NOISE"
        p_voice[idx] = max(0.0, min(1.0, 0.5 * (float(sims[vi]) + 1.0)))
    return {
        "n_speakers": 0,
        "best_silhouette": float(best_overall) if best_overall > -1.0 else None,
        "labels": cluster_labels,
        "p_voice": p_voice,
        "valid_indices": list(valid_indices),
    }


def _merge_close_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    merge_threshold: float = 0.55,
) -> np.ndarray:
    """Iteratively merge the closest cluster pair while their centroid cos_sim ≥ threshold.

    Speaker embedding clusterings (k-means / spectral) sometimes split one
    speaker into prosodic sub-clusters — long continuous passage vs brief
    asr, near-mic vs far-mic, etc. This pass collapses those back into
    one. Two cluster centroids with cosine similarity ≥ ``merge_threshold``
    are taken to be the same speaker; we merge them and re-evaluate. Stops
    when every remaining pair is below the threshold (i.e. genuinely
    different speakers).

    Vectors in ``X`` are assumed L2-normalized.
    """
    if X.shape[0] == 0 or labels.size == 0:
        return labels
    labels = labels.copy()
    while True:
        unique = sorted(set(int(x) for x in labels))
        if len(unique) < 2:
            return labels
        centroids: dict[int, np.ndarray] = {}
        for u in unique:
            mask = labels == u
            c = X[mask].mean(axis=0)
            n = float(np.linalg.norm(c))
            centroids[u] = (c / n) if n > 0 else c
        best_pair: tuple[int, int] | None = None
        best_sim = merge_threshold
        for i, ui in enumerate(unique):
            ci = centroids[ui]
            if ci.size == 0 or float(np.linalg.norm(ci)) == 0:
                continue
            for uj in unique[i + 1 :]:
                cj = centroids[uj]
                if cj.size == 0 or float(np.linalg.norm(cj)) == 0:
                    continue
                sim = float(np.dot(ci, cj))
                if sim >= best_sim:
                    best_sim = sim
                    best_pair = (ui, uj)
        if best_pair is None:
            return labels
        # Merge the higher-numbered cluster into the lower one. (Cluster
        # numbers carry no meaning here; a deterministic choice is enough.)
        keep, drop = sorted(best_pair)
        labels[labels == drop] = keep


def _empirical_calibration_band(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    min_pairs: int = 5,
) -> tuple[float, float] | None:
    """Estimate ``(same_speaker_floor, diff_speaker_floor)`` from clustered embeddings.

    Walks all within-cluster pairs and all between-cluster pairs, computes
    cosine distances, and uses percentile anchors:

    - ``same_floor = quantile(within, 0.75)``: most same-speaker pairs sit
      below; calibrate so cos_dist at this anchor → uncertainty 0.
    - ``diff_floor = quantile(between, 0.25)``: most different-speaker pairs
      sit above; cos_dist at this anchor → uncertainty 1.

    Returns ``None`` when the pass cannot support a band — too few pairs, or within- and
    between-cluster distances that overlap. Substituting the literature defaults there was
    the defect this replaces: on a real recording ECAPA's within-speaker distances had
    median 0.874 against 0.966 between, overlapping almost entirely, and the [0.30, 0.70]
    fallback sat below *every* distance the embedding produced — so every same-speaker
    comparison scored as maximally doubtful and the axis reported high uncertainty on
    buckets where every diarizer agreed. A sub-signal that cannot be calibrated must drop
    out (FR-007), not vote with fabricated confidence.

    Vectors in ``X`` are assumed L2-normalized (unit-norm), which matches
    ``cluster_pass_speakers``'s preprocessing — cosine distance reduces to
    ``1 − x · y`` directly.
    """
    if X.shape[0] < 2:
        return None
    sim_matrix = X @ X.T  # shape (N, N), values in [-1, 1] for unit vectors
    n = X.shape[0]
    within_dists: list[float] = []
    between_dists: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(max(0.0, min(1.0, 1.0 - sim_matrix[i, j])))
            if labels[i] == labels[j]:
                within_dists.append(d)
            else:
                between_dists.append(d)
    if len(within_dists) < min_pairs or len(between_dists) < min_pairs:
        return None
    same_floor = float(np.quantile(within_dists, 0.75))
    diff_floor = float(np.quantile(between_dists, 0.25))
    # Ensure ordering (same < diff). Pathological clusters can give within > between
    # — fall back to defaults rather than report nonsense.
    if same_floor >= diff_floor:
        return None
    # Clamp only enough to keep the band ordered and inside [0.05, 0.95]. An earlier
    # version pinned same_floor to at most 0.50 and diff_floor to at least 0.50, which
    # presumes the two distributions straddle 0.5. Measured on a real two-speaker
    # recording, ECAPA's *within*-speaker distance over 0.5 s buckets has a median of
    # 0.543 — above that cap — so the presumption fails at this time scale and the cap
    # would drag the anchor back below anything the embeddings actually produce.
    same_floor = max(0.05, min(0.95, same_floor))
    diff_floor = max(0.05, min(0.95, diff_floor))
    if diff_floor - same_floor < 0.05:
        return None
    return same_floor, diff_floor


def _sequential_calibration_band(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    min_pairs: int = 5,
) -> tuple[float, float] | None:
    """Calibration anchors measured on the comparison the speaker harvester actually makes.

    The harvester never compares all window pairs. It compares each bucket to the *most
    recent prior* bucket carrying the same speaker label, and — for change claims — to the
    immediately preceding bucket. Anchors must describe that statistic, because it is a
    different distribution from the all-pairs one.

    How different, measured on a two-speaker recording with the diarizer's turns as ground
    truth: over all pairs, ECAPA's within- and between-speaker distances overlap (within q75
    0.919 against between q25 0.916) and no ordered band exists at all. Over the sequential
    comparison, the same embeddings separate cleanly (within q75 0.646, between q25 0.915).
    Nearest-centroid classification recovers the diarizer's labels at 98.5%, so the embedding
    discriminates perfectly well — calibrating on all-pairs merely measured the wrong thing,
    discarded the usable band, and fell back to a fixed one sitting below every distance the
    embedding produces. The axis then reported high uncertainty on buckets where every
    diarizer agreed.

    Args:
        X: Unit-norm embedding vectors in time order.
        labels: Cluster label per row, aligned with ``X``.
        min_pairs: Minimum comparisons of each kind before anchors are trusted.

    Returns:
        ``(same_speaker_floor, diff_speaker_floor)``, or ``None`` when the pass supports no
        ordered band — too few comparisons, or distributions that genuinely overlap.
    """
    if X.shape[0] < 2:
        return None
    within: list[float] = []
    between: list[float] = []
    last_seen: dict[Any, int] = {}
    for i in range(X.shape[0]):
        label = labels[i]
        for other, idx in last_seen.items():
            d = float(max(0.0, min(1.0, 1.0 - float(X[i] @ X[idx]))))
            (within if other == label else between).append(d)
        last_seen[label] = i
    if len(within) < min_pairs or len(between) < min_pairs:
        return None
    same_floor = float(np.quantile(within, 0.75))
    diff_floor = float(np.quantile(between, 0.25))
    if same_floor >= diff_floor:
        return None
    same_floor = max(0.05, min(0.95, same_floor))
    diff_floor = max(0.05, min(0.95, diff_floor))
    if diff_floor - same_floor < 0.05:
        return None
    return same_floor, diff_floor


def _within_cluster_band(
    X: np.ndarray,
    *,
    fallback_diff_floor: float = 0.70,
    min_pairs: int = 5,
) -> tuple[float, float] | None:
    """Calibration band for a pass whose windows all belong to one speaker.

    With a single cluster there are no between-speaker pairs, so only the same-speaker
    anchor can be measured and the different-speaker floor keeps its default. Measuring the
    one is what matters: a fixed same-speaker floor of 0.30 is unreachable for short-bucket
    embeddings, which leaves the speaker axis unable to report low uncertainty even when
    every diarizer agrees on an unchanging speaker.
    """
    if X.shape[0] < 2:
        return None
    sim = X @ X.T
    n = X.shape[0]
    dists = [float(max(0.0, min(1.0, 1.0 - sim[i, j]))) for i in range(n) for j in range(i + 1, n)]
    if len(dists) < min_pairs:
        return None
    same_floor = max(0.05, min(0.95, float(np.quantile(dists, 0.75))))
    diff_floor = fallback_diff_floor
    if diff_floor - same_floor < 0.05:
        # The speaker's own spread reaches the different-speaker anchor. Push the anchor
        # out rather than returning an inverted band that would score every comparison.
        diff_floor = min(0.95, same_floor + 0.20)
    return same_floor, diff_floor


def silhouette_voice_score(
    entries: list[WindowEmbedding],
    *,
    n_clusters_max: int = 6,
    min_windows_for_clustering: int = 4,
) -> dict[int, float] | None:
    """Backwards-compatible thin wrapper returning just the per-window p_voice map.

    Prefer ``cluster_pass_speakers`` for new code — it also exposes the
    estimated speaker count and per-window cluster labels (used to synthesise
    an embedding-derived diarization source).
    """
    res = cluster_pass_speakers(
        entries,
        n_clusters_max=n_clusters_max,
        min_windows_for_clustering=min_windows_for_clustering,
    )
    return None if res is None else res["p_voice"]


def calibrate_cosine_uncertainty(
    cos_dist: float,
    *,
    same_speaker_floor: float = 0.30,
    diff_speaker_floor: float = 0.70,
    direction: str = "same",
) -> float:
    """Map raw cosine distance to a calibrated uncertainty in ``[0, 1]``.

    The raw cosine distance between two ECAPA / ResNet asr embeddings sits
    in a noise floor of roughly 0.1–0.3 even for the same speaker (phonetic
    variation), so a small distance is **not** strong evidence of speaker. The
    EER decision boundary on VoxCeleb is around 0.4–0.5; distances above ~0.7
    are confidently different speakers. This helper maps the raw distance onto
    the [same_speaker_floor, diff_speaker_floor] calibration band.

    Args:
        cos_dist: Raw ``1 − cos_sim`` from two embeddings.
        same_speaker_floor: Distance at or below which we're confident the two
            embeddings come from the same speaker. Defaults to 0.30 (typical
            ECAPA same-speaker noise level).
        diff_speaker_floor: Distance at or above which we're confident they're
            different speakers. Defaults to 0.70 (well above the EER region).
        direction: ``"same"`` if the diar model claimed *same speaker* (high
            distance is the disagreement); ``"diff"`` if the model claimed a
            *speaker change* (low distance is the disagreement). The returned
            value is uncertainty — higher means the audio contradicts the
            claim more strongly.

    Returns:
        Calibrated uncertainty in ``[0, 1]``.
    """
    if cos_dist <= same_speaker_floor:
        consistency = 1.0  # confidently same speaker
    elif cos_dist >= diff_speaker_floor:
        consistency = 0.0  # confidently different
    else:
        # Linear interpolation in the calibration band.
        span = diff_speaker_floor - same_speaker_floor
        consistency = max(0.0, min(1.0, 1.0 - (cos_dist - same_speaker_floor) / span))
    if direction == "same":
        # Claim was "same speaker"; uncertainty rises as audio shows different.
        return 1.0 - consistency
    if direction == "diff":
        # Claim was "different speaker"; uncertainty rises as audio shows same.
        return consistency
    raise ValueError(f"direction must be 'same' or 'diff', got {direction!r}")
