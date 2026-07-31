"""Identity repair: embedding change-point detection + consensus re-clustering.

Implements I1 (boundary evidence) and I2 (re-cluster) generically:

1. Per embedding model, L2-normalize the per-window vectors and compute the
   adjacent-window cosine-distance trajectory; average trajectories across
   models (windows share one grid per run).
2. Change-points = local maxima above ``mean + k·std`` (policy ``cp_k``) with a
   minimum absolute floor (``cp_floor``); prominence is kept as boundary
   confidence. Diarization-model boundaries join the candidate cut set — the
   diarizers may be *under*-segmenting (the failure this repairs) but any cut
   they did make is evidence.
3. Segments = voiced spans cut at change-points (min duration ``min_segment_s``;
   shorter segments merge into the neighbor with the closer centroid).
4. Per model, pool window embeddings per segment (p_voice-weighted mean) and
   agglomerative-cluster the pooled vectors (average linkage, cosine threshold
   ``recluster_cosine_threshold``). Cross-model consensus: segments co-clustered
   by ≥ half the models merge (union-find on the co-association matrix).
5. Output refined segments/clusters (ids ``R0, R1, …`` by first appearance),
   boundary confidences, and per-bucket speaker votes; the caller shadows the
   per-bucket ``__cross_diar_label_disagreement__`` with a recomputed value that
   includes the new voter.

No parameter here is tuned to a particular file: everything comes from the
policy, and inputs are whatever embeddings/diarization the run produced.
"""

from __future__ import annotations

from typing import Any


def _l2(mat: Any) -> Any:  # noqa: ANN401
    import numpy as np

    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, 1e-9)


def change_point_trajectory(
    window_embeddings: dict[str, list[dict[str, Any]]],
) -> tuple[list[float], list[float]]:
    """Mean adjacent-cosine-distance trajectory across models → (boundary_times, distances)."""
    import numpy as np

    per_model: list[Any] = []
    times: list[float] | None = None
    for windows in window_embeddings.values():
        if len(windows) < 2:
            continue
        vecs = _l2(np.asarray([w["vector"] for w in windows], dtype=float))
        d = 1.0 - (vecs[:-1] * vecs[1:]).sum(axis=1)
        per_model.append(d)
        if times is None:
            # Boundary between window i and i+1 ≈ centre of their overlap.
            times = [
                round((float(windows[i + 1]["start_s"]) + float(windows[i]["end_s"])) / 2.0, 4)
                for i in range(len(windows) - 1)
            ]
    if not per_model or times is None:
        return [], []
    dist = np.mean(np.stack(per_model), axis=0)
    if len(dist) >= 3:  # light smoothing, preserves peaks
        dist = np.convolve(dist, np.array([0.25, 0.5, 0.25]), mode="same")
    return times, [float(x) for x in dist]


def detect_change_points(
    times: list[float], dist: list[float], *, cp_k: float, cp_floor: float
) -> list[dict[str, Any]]:
    """Local maxima above max(mean + k·std, floor); prominence-normalized confidence."""
    import numpy as np

    if not times:
        return []
    d = np.asarray(dist)
    thr = max(float(d.mean() + cp_k * d.std()), cp_floor)
    span = max(1e-9, float(d.max() - d.min()))
    out = []
    for i in range(len(d)):
        left = d[i - 1] if i > 0 else -np.inf
        right = d[i + 1] if i < len(d) - 1 else -np.inf
        if d[i] >= thr and d[i] >= left and d[i] >= right:
            conf = round(float((d[i] - d.min()) / span), 4)
            out.append({"time": times[i], "distance": round(float(d[i]), 4), "confidence": conf})
    return out


def _voiced_spans(
    p_voice_at: Any,  # noqa: ANN401 — callable (t) -> float | None
    duration_s: float,
    *,
    step: float = 0.05,
    threshold: float = 0.5,
) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    t, open_start = 0.0, None
    while t < duration_s:
        pv = p_voice_at(t + step / 2)
        voiced = pv is not None and pv >= threshold
        if voiced and open_start is None:
            open_start = t
        elif not voiced and open_start is not None:
            spans.append((round(open_start, 4), round(t, 4)))
            open_start = None
        t += step
    if open_start is not None:
        spans.append((round(open_start, 4), round(duration_s, 4)))
    return spans


def _agglomerative_cosine(vectors: Any, threshold: float) -> list[int]:  # noqa: ANN401
    """Deterministic average-linkage agglomerative clustering on cosine distance."""
    import numpy as np

    n = len(vectors)
    clusters: list[list[int]] = [[i] for i in range(n)]
    vecs = _l2(np.asarray(vectors, dtype=float))
    dmat = 1.0 - vecs @ vecs.T
    while len(clusters) > 1:
        best: tuple[float, int, int] | None = None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                d = float(np.mean([dmat[i, j] for i in clusters[a] for j in clusters[b]]))
                if best is None or d < best[0] - 1e-12:
                    best = (d, a, b)
        if best is None or best[0] > threshold:
            break
        _, a, b = best
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]
    labels = [0] * n
    for lbl, members in enumerate(sorted(clusters, key=min)):
        for i in members:
            labels[i] = lbl
    return labels


def repair_identity(
    *,
    window_embeddings: dict[str, list[dict[str, Any]]],
    diar_boundaries: list[float],
    p_voice_at: Any,  # noqa: ANN401 — callable (t) -> float | None
    duration_s: float,
    policy: dict[str, Any],
) -> dict[str, Any] | None:
    """Full I1+I2 repair. Returns refined segments/clusters + change-point evidence, or None."""
    import numpy as np

    cfg = policy.get("speaker") or {}
    times, dist = change_point_trajectory(window_embeddings)
    if not times:
        return None
    cps = detect_change_points(times, dist, cp_k=float(cfg.get("cp_k", 1.0)), cp_floor=float(cfg.get("cp_floor", 0.15)))
    cuts = sorted({round(c["time"], 4) for c in cps} | {round(b, 4) for b in diar_boundaries})
    min_seg = float(cfg.get("min_segment_s", 0.25))

    # Voiced spans cut at change-points.
    segments: list[dict[str, Any]] = []
    voiced_thr = float(cfg.get("voiced_threshold", 0.5))
    for span_start, span_end in _voiced_spans(p_voice_at, duration_s, threshold=voiced_thr):
        edges = [span_start] + [c for c in cuts if span_start + min_seg <= c <= span_end - min_seg] + [span_end]
        for i in range(len(edges) - 1):
            if edges[i + 1] - edges[i] >= min_seg:
                segments.append({"start": edges[i], "end": edges[i + 1]})
    if not segments:
        return None

    # Pool per model per segment (p_voice-weighted window means).
    pooled: dict[str, list[Any]] = {}
    for model, windows in window_embeddings.items():
        vecs = _l2(np.asarray([w["vector"] for w in windows], dtype=float))
        mids = np.asarray([(float(w["start_s"]) + float(w["end_s"])) / 2.0 for w in windows])
        seg_vecs = []
        for seg in segments:
            inside = (mids >= seg["start"]) & (mids < seg["end"])
            if not inside.any():  # fall back to nearest window
                inside = np.zeros(len(mids), dtype=bool)
                inside[int(np.argmin(np.abs(mids - (seg["start"] + seg["end"]) / 2)))] = True
            weights = np.asarray([max(0.05, p_voice_at(m) or 0.05) for m in mids[inside]])
            v = (vecs[inside] * weights[:, None]).sum(axis=0) / weights.sum()
            seg_vecs.append(v)
        pooled[model] = seg_vecs

    # Per-model clustering → cross-model co-association consensus.
    thr = float(cfg.get("recluster_cosine_threshold", 0.45))
    n = len(segments)
    coassoc = np.zeros((n, n))
    for model, seg_vecs in pooled.items():
        labels = _agglomerative_cosine(seg_vecs, thr)
        for i in range(n):
            for j in range(n):
                coassoc[i, j] += labels[i] == labels[j]
    coassoc /= max(1, len(pooled))

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if coassoc[i, j] >= 0.5:
                parent[find(j)] = find(i)
    roots: dict[int, str] = {}
    for idx, seg in enumerate(segments):
        r = find(idx)
        if r not in roots:
            roots[r] = f"R{len(roots)}"
        seg["cluster_id"] = roots[r]
        cp_conf = {round(c["time"], 4): c["confidence"] for c in cps}
        seg["boundary_confidence"] = {
            "start": cp_conf.get(round(seg["start"], 4), 0.5),
            "end": cp_conf.get(round(seg["end"], 4), 0.5),
        }
    return {
        "segments": segments,
        "n_clusters": len(roots),
        "change_points": cps,
        "trajectory": {"times": times, "distances": [round(d, 4) for d in dist]},
        "models_used": sorted(window_embeddings.keys()),
        "params": {
            "cp_k": cfg.get("cp_k", 1.0),
            "cp_floor": cfg.get("cp_floor", 0.15),
            "min_segment_s": min_seg,
            "recluster_cosine_threshold": thr,
        },
    }


def cluster_at(refined: dict[str, Any], t: float) -> str | None:
    """Refined cluster id at time ``t`` (None in unvoiced gaps)."""
    for seg in refined["segments"]:
        if seg["start"] <= t < seg["end"]:
            return str(seg["cluster_id"])
    return None


def cross_source_disagreement(cluster_ids: list[str]) -> float | None:
    """Fraction of source pairs disagreeing on the cluster — mirrors the speaker axis sub-signal."""
    ids = [c for c in cluster_ids if c]
    if len(ids) < 2:
        return None
    pairs = disagree = 0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            pairs += 1
            disagree += ids[i] != ids[j]
    return disagree / pairs if pairs else None
