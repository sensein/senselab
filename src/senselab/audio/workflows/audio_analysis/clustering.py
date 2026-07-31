"""Embedding-based clustering of diarization labels into pass-wide speaker IDs.

Each diarization model invents its own speaker labels (pyannote ``SPEAKER_00``,
sortformer ``speaker_2``, etc.). To compare across diar models — and to
recognize that two models naming the same person differently are actually in
agreement — we cluster ``(diar_model, raw_label)`` pairs by their mean speaker
embedding. The output map ``{(diar_model, raw_label) → cluster_id}`` is the
shared "speaker identity" used by:

- ``identity`` axis: cross-model agreement signal compares cluster_id, not raw
  labels, so naming-convention differences don't fake disagreement.
- timeline plot: speaker color is per cluster_id, not per raw label, so a
  speaker keeps the same color across diar models and across passes.

A diar bucket where no model detected a speaker contributes no segment, so
it has no entry in this map. The harvester's ``<silent>`` pseudo-label is
mapped to a fixed ``"SIL"`` cluster outside this module — silence is its own
universal cluster regardless of pass / model.
"""

from __future__ import annotations

from typing import Any, TypeVar

import numpy as np

from senselab.audio.workflows.audio_analysis.embeddings import WindowEmbedding

K = TypeVar("K")


def _seg_attr(seg: Any, name: str) -> Any:  # noqa: ANN401
    """Tolerant getter for ScriptLine vs dict shapes."""
    if isinstance(seg, dict):
        return seg.get(name)
    return getattr(seg, name, None)


def _diar_segments(block: Any) -> list[Any]:  # noqa: ANN401
    """Extract the segment list from a diar block, handling List[List[ScriptLine]] wrap."""
    if not (isinstance(block, dict) and block.get("status") == "ok"):
        return []
    res = block.get("result")
    if not (isinstance(res, list) and res):
        return []
    inner = res[0] if isinstance(res[0], list) else res
    return list(inner) if isinstance(inner, list) else []


def _mean_window_embedding_over_segments(
    segs: list[Any],
    windows: list[WindowEmbedding],
) -> np.ndarray | None:
    """Average window-embedding vectors whose midpoints fall inside any of ``segs``."""
    if not segs or not windows:
        return None
    accum: list[np.ndarray] = []
    for w in windows:
        wc = 0.5 * (w.start_s + w.end_s)
        for seg in segs:
            ls = _seg_attr(seg, "start")
            le = _seg_attr(seg, "end")
            if ls is None or le is None:
                continue
            if float(ls) <= wc <= float(le):
                arr = np.asarray(w.vector, dtype=np.float64).flatten()
                if arr.size > 0:
                    accum.append(arr)
                break
    if not accum:
        return None
    return np.stack(accum, axis=0).mean(axis=0)


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float | None:
    """Cosine similarity between two equal-length 1-D arrays. None on bad input."""
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return None
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return None
    return float(np.dot(a, b) / (na * nb))


def assign_unified_clusters_with_seed_phase(
    seed_groups: list[list[tuple[K, np.ndarray]]],
    other_items: list[tuple[K, np.ndarray]],
    *,
    cosine_threshold: float = 0.5,
    cross_group_threshold: float = 0.75,
) -> dict[K, str]:
    """Group-aware seed phase + frozen-pool fallback.

    ``seed_groups`` is a list of groups. Each inner list contains
    ``(key, mean_emb)`` pairs that originated from the SAME per-source
    clustering — they have already been validated as distinct clusters by
    that source's min-size + silhouette guards and **MUST NOT** merge with
    each other inside this function. Items from DIFFERENT groups may merge
    when their centroid cosine similarity ≥ ``cross_group_threshold`` —
    that's the cross-pass / cross-source unification (e.g. raw-pass Peter
    matched to enhanced-pass Peter; both at cos_sim ~0.9+).

    Why two thresholds:
      - ``cross_group_threshold`` (default 0.75) governs match-across-groups
        (raw vs enhanced, ECAPA-clustering vs ResNet-clustering). Same
        speaker across passes is typically cos_sim 0.85+, different
        speakers within a pass sit around 0.30-0.50, so 0.75 cleanly
        separates them.
      - ``cosine_threshold`` (default 0.5) governs ``other_items`` matching
        — used for pyannote / sortformer labels to snap to an existing seed
        when their mean embedding clears the bar. Lower threshold here is
        intentional: those models segment differently than the synthetic
        source, so their per-label means can be noisier.

    Phase 1 — *seed groups*: walk each group; each ``(key, mean_emb)`` in
    that group is assigned a NEW centroid (within-group items never share
    a cluster id). After the group is consumed, walk a second time across
    the existing centroid pool — if any pair of centroids have cos_sim ≥
    ``cross_group_threshold``, merge them. This is how raw_Peter and
    enh_Peter end up in the same unified cluster.

    Phase 2 — *frozen pool*: each ``(key, mean_emb)`` in ``other_items``
    snaps to the closest seed centroid (no threshold once the pool is
    seeded). If no centroid exists yet, fall back to the legacy
    ``cosine_threshold`` rule so the function degrades gracefully when no
    seeds were provided.
    """
    out: dict[K, str] = {}
    centroids: list[np.ndarray] = []
    # Map cluster_id (C0..) → list of centroid indices that belong to it.
    # We append to ``centroids`` strictly per (key) but the cluster_id may be
    # reused when a cross-group match fires.
    centroid_to_cluster: list[int] = []  # parallel to centroids — cluster id of each centroid
    next_cluster_id = 0

    def _add_centroid_as_new() -> int:
        nonlocal next_cluster_id
        cid = next_cluster_id
        next_cluster_id += 1
        return cid

    for group in seed_groups:
        # Each member of this group gets its own new centroid; never merge
        # within the group. After all members are added, attempt cross-group
        # merges by walking the existing centroid pool.
        added_this_group: list[int] = []  # indices into ``centroids``
        for key, mean_emb in group:
            if mean_emb is None or mean_emb.size == 0:
                continue
            # Try to match against centroids that were already in the pool
            # BEFORE this group started (i.e. from earlier groups). Skip
            # centroids that were added by *this* group — they're guaranteed
            # to be distinct per the upstream clusterer.
            existing_count = len(centroids) - len(added_this_group)
            best_idx = None
            best_sim = cross_group_threshold
            for ci in range(existing_count):
                sim = _cos_sim(mean_emb, centroids[ci])
                if sim is not None and sim >= best_sim:
                    best_sim = sim
                    best_idx = ci
            if best_idx is not None:
                centroids.append(mean_emb)
                centroid_to_cluster.append(centroid_to_cluster[best_idx])
                added_this_group.append(len(centroids) - 1)
                out[key] = f"C{centroid_to_cluster[best_idx]}"
            else:
                cid = _add_centroid_as_new()
                centroids.append(mean_emb)
                centroid_to_cluster.append(cid)
                added_this_group.append(len(centroids) - 1)
                out[key] = f"C{cid}"

    seed_phase_done = bool(centroids)
    for key, mean_emb in other_items:
        if mean_emb is None or mean_emb.size == 0:
            continue
        best_idx = None
        # When the seed pool exists, snap to closest (no threshold). Without
        # seeds, fall back to ``cosine_threshold`` so legacy callers still
        # get reasonable behavior.
        best_sim = -1.0 if seed_phase_done else cosine_threshold
        for ci, c in enumerate(centroids):
            sim = _cos_sim(mean_emb, c)
            if sim is not None and sim >= best_sim:
                best_sim = sim
                best_idx = ci
        if best_idx is None:
            if seed_phase_done:
                out[key] = "?"
            else:
                cid = _add_centroid_as_new()
                centroids.append(mean_emb)
                centroid_to_cluster.append(cid)
                out[key] = f"C{cid}"
        else:
            out[key] = f"C{centroid_to_cluster[best_idx]}"
    return out
