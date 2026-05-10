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
    seed_items: list[tuple[K, np.ndarray]],
    other_items: list[tuple[K, np.ndarray]],
    *,
    cosine_threshold: float = 0.5,
) -> dict[K, str]:
    """Two-phase greedy centroid assignment with the seed pool frozen after phase 1.

    Phase 1 — *seed*: each ``(key, mean_emb)`` in ``seed_items`` either matches
    an existing centroid (cos_sim ≥ ``cosine_threshold``) or starts a new one.
    The cluster pool grows freely.

    Phase 2 — *frozen*: each ``(key, mean_emb)`` in ``other_items`` MUST snap
    to one of the seed centroids (closest by cosine, no threshold). If no
    centroid exists at all when phase 2 starts, the item starts its own — but
    once a single seed centroid is established, ``other_items`` can never
    spawn new ones; instead they're tagged ``"?"`` if they fail every cosine
    check (this is impossible without a threshold, but kept for safety).

    Returns ``{key → cluster_id}`` with cluster_ids of the form ``"S0"``,
    ``"S1"``, ... — same labels both phases use.

    The intended use: ``seed_items`` carries the synthetic
    ``embedding_silhouette/...`` diar source's per-cluster mean embeddings
    (already validated by ``cluster_pass_speakers``'s min-size + silhouette
    guards), and ``other_items`` carries pyannote / sortformer mean embeddings.
    Frozen-pool semantics prevent a noisy 3-segment third "speaker" from
    inflating the unified speaker count beyond what the embedding source
    decided.
    """
    out: dict[K, str] = {}
    centroids: list[np.ndarray] = []
    for key, mean_emb in seed_items:
        if mean_emb is None or mean_emb.size == 0:
            continue
        best_idx = None
        best_sim = cosine_threshold
        for ci, c in enumerate(centroids):
            sim = _cos_sim(mean_emb, c)
            if sim is not None and sim >= best_sim:
                best_sim = sim
                best_idx = ci
        if best_idx is None:
            centroids.append(mean_emb)
            out[key] = f"S{len(centroids) - 1}"
        else:
            out[key] = f"S{best_idx}"
    seed_phase_done = bool(centroids)
    for key, mean_emb in other_items:
        if mean_emb is None or mean_emb.size == 0:
            continue
        best_idx = None
        # When the seed phase produced any centroid, every other_item must
        # match the closest one (no threshold). Without seeds, fall back to
        # the cosine_threshold rule so the function still does something
        # reasonable on a no-seed pass (legacy behavior).
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
                centroids.append(mean_emb)
                out[key] = f"S{len(centroids) - 1}"
        else:
            out[key] = f"S{best_idx}"
    return out


def cluster_speaker_labels_by_embedding(
    diar_blocks: dict[str, Any],
    per_window_embeddings: dict[str, list[WindowEmbedding]],
    *,
    cosine_threshold: float = 0.5,
    failures: dict[str, str] | None = None,
    failure_key: str = "speaker_label_clustering",
) -> dict[tuple[str, str], str]:
    """Return ``{(diar_model, raw_label) → cluster_id}`` for one pass.

    For each (diar_model, raw_label), computes the mean window-embedding across
    all segments labelled with that raw_label, then greedily assigns each
    (model, label) to an existing cluster whose centroid has
    ``cos_sim ≥ cosine_threshold`` — or starts a new cluster.

    Picks the alphabetically first embedding model in ``per_window_embeddings``
    (typically ECAPA before ResNet) — we only need one to define a metric, and
    using both at once would require multi-modal centroid logic that adds
    little discriminative power for typical 2–4-speaker audio.

    When no embedding model is available, falls back to ``raw_label`` as
    cluster id (so cross-naming agreement can't be detected, but the system
    degrades gracefully).
    """
    out: dict[tuple[str, str], str] = {}
    if not per_window_embeddings or not any(per_window_embeddings.values()):
        msg = "no embedding windows available — falling back to raw label identity (cross-naming agreement unavailable)"
        import sys as _sys

        print(f"warn: cluster_speaker_labels_by_embedding: {msg}", file=_sys.stderr)
        if failures is not None:
            failures[failure_key] = msg
        for m, block in diar_blocks.items():
            for seg in _diar_segments(block):
                spk = str(_seg_attr(seg, "speaker") or "?")
                out.setdefault((m, spk), spk)
        return out

    emb_model = sorted(per_window_embeddings)[0]
    windows = per_window_embeddings[emb_model]

    # Build (key, mean_emb) lists split into seed (synthetic embedding_silhouette
    # source) vs other (pyannote / sortformer / etc). The shared helper enforces
    # the freeze invariant.
    seed_items: list[tuple[tuple[str, str], np.ndarray]] = []
    other_items: list[tuple[tuple[str, str], np.ndarray]] = []
    for m, block in diar_blocks.items():
        segs_by_label: dict[str, list[Any]] = {}
        for seg in _diar_segments(block):
            spk = str(_seg_attr(seg, "speaker") or "?")
            segs_by_label.setdefault(spk, []).append(seg)
        is_seed = m.startswith("embedding_silhouette/")
        for raw_label, segs in segs_by_label.items():
            mean_emb = _mean_window_embedding_over_segments(segs, windows)
            if mean_emb is None or mean_emb.size == 0:
                # No audio support — fall back to raw_label as cluster id.
                out[(m, raw_label)] = raw_label
                continue
            (seed_items if is_seed else other_items).append(((m, raw_label), mean_emb))

    out.update(assign_unified_clusters_with_seed_phase(seed_items, other_items, cosine_threshold=cosine_threshold))
    return out
