"""Identity axis vote harvesters — "was it the same speaker?".

The identity question splits into three independent diagnostic checks per bucket:

1. **Same-speaker claim validation** — when a diar model labels this bucket the
   same speaker (after embedding-clustering across diar models) as some prior
   bucket on its track, does the audio embedding confirm? Cosine distance to
   the most recent prior same-cluster embedding, calibrated against typical
   ECAPA / ResNet same-speaker noise floor and different-speaker EER.
2. **Speaker-change claim validation** — when a diar model says this bucket is a
   different speaker (or transitions between speech and silence) from the
   immediately prior bucket, does the audio embedding confirm? Cosine distance
   to the immediately prior bucket's embedding, calibrated.
3. **Cross-diar-model agreement** — do the active diar models agree on the
   speaker for this bucket (after embedding-clustering)? 0 = all agree;
   1 = all disagree. Pyannote ``SPEAKER_00`` and sortformer ``speaker_2`` end
   up in the same cluster when their embeddings match, so naming-convention
   differences don't fake disagreement.

Speaker count handling
----------------------

The harvester handles 1, 2, or many speakers identically — same-label tracking
is per ``(diar_model, cluster_id)`` so each speaker has its own history, and
cross-model agreement is computed pairwise across whatever diar models are
active.

"No speaker" handling
---------------------

When a diar model returns no segment for a bucket, we treat the absence as a
``"<silent>"`` pseudo-cluster. That way:

- Two consecutive silent buckets on the same model count as "same speaker" with
  no embedding comparison.
- A silent → speaking transition counts as a real change claim.
- Cross-model disagreement fires when one model says "<silent>" and another
  identifies a speaker — they fundamentally disagree on whether anyone is there.

Same-window dedup
-----------------

When two buckets share an embedding window (a 2 s / 1 s grid covers up to four
0.5 s buckets per window), their embedding vectors are identical and any cosine
comparison returns 0 — that's an artifact, not a confirmation. Both same-label
and change sub-signals skip emitting a value in that case (None drops out of
the aggregator per FR-007).
"""

from __future__ import annotations

from typing import Any

from senselab.audio.workflows.audio_analysis.clustering import cluster_speaker_labels_by_embedding
from senselab.audio.workflows.audio_analysis.embeddings import (
    WindowEmbedding,
    calibrate_cosine_uncertainty,
    window_index_at,
)
from senselab.audio.workflows.audio_analysis.grid import BucketGrid
from senselab.audio.workflows.audio_analysis.harvesters import diar_speaker_label_in_window

SILENT_CLUSTER_ID = "SIL"


def _cosine_similarity(a: list[float], b: list[float]) -> float | None:
    """Cosine similarity between two equal-length vectors. Returns None on bad inputs."""
    if not a or not b or len(a) != len(b):
        return None
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return None
    return dot / (norm_a * norm_b)


def _cos_dist(a: list[float], b: list[float]) -> float | None:
    """Cosine distance ``1 − cos_sim`` clipped to ``[0, 1]``. None on bad inputs."""
    sim = _cosine_similarity(a, b)
    if sim is None:
        return None
    return max(0.0, min(1.0, 1.0 - sim))


def _embedding_for_bucket(
    per_window_embeddings: dict[str, list[WindowEmbedding]],
    model_id: str,
    bucket_center_s: float,
) -> tuple[int, list[float]] | None:
    """Return ``(window_index, vector)`` for the window covering ``bucket_center_s``."""
    entries = per_window_embeddings.get(model_id) or []
    idx = window_index_at(entries, bucket_center_s)
    if idx is None:
        return None
    w = entries[idx]
    if w.vector.size == 0:
        return None
    return idx, [float(x) for x in w.vector.tolist()]


def harvest_identity_votes(
    *,
    pass_summary: dict[str, Any],
    grid: BucketGrid,
    per_window_embeddings: dict[str, list[WindowEmbedding]],
    same_speaker_floor: float = 0.30,
    diff_speaker_floor: float = 0.70,
    cluster_cosine_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Yield ``{"start", "end", "votes"}`` per bucket for the identity axis.

    Args:
        pass_summary: Per-task summary for one pass (diarization, alignment, etc.).
        grid: Bucket grid.
        per_window_embeddings: ``{embedding_model_id → [WindowEmbedding, ...]}``.
        same_speaker_floor: Cosine distance ≤ this is treated as confidently
            same-speaker (uncertainty 0 for same-claim, 1 for change-claim).
        diff_speaker_floor: Cosine distance ≥ this is treated as confidently
            different-speaker (uncertainty 1 for same-claim, 0 for change-claim).
        cluster_cosine_threshold: Cosine similarity threshold for clustering
            (diar_model, raw_label) into pass-wide speaker IDs. 0.5 is roughly
            the EER for ECAPA on VoxCeleb.

    Returns:
        List of ``{"start", "end", "votes"}`` dicts. ``votes`` shape::

            {
                "<diar_model>": {
                    "speaker_label": "<raw label or '<silent>'>",
                    "cluster_id": "S0" | "S1" | ... | "SIL",
                    "speaker_changed_from_prev": bool | None,
                },
                "<diar_model>::<embedding_model>": {
                    "diar_model": "<diar_model>",
                    "embedding_model": "<embedding_model>",
                    "embedding_cosine_within_track": float | None,
                    "same_label_uncertainty": float | None,
                    "embedding_cosine_to_prev_bucket": float | None,
                    "change_inconsistency_uncertainty": float | None,
                },
                "__cross_diar_label_disagreement__": {
                    "value": float | None,
                    "n_pairs": int,
                    "n_disagree": int,
                    "cluster_ids": dict[diar_model, cluster_id],
                },
            }
    """
    duration_s = float(pass_summary.get("duration_s", 0.0) or 0.0)
    diar_blocks = (pass_summary.get("diarization") or {}).get("by_model") or {}
    diar_ok = {m: b for m, b in diar_blocks.items() if isinstance(b, dict) and b.get("status") == "ok"}
    if duration_s <= 0:
        for block in diar_ok.values():
            res = block.get("result")
            if not (isinstance(res, list) and res):
                continue
            segs = res[0] if isinstance(res[0], list) else res
            if not isinstance(segs, list):
                continue
            for seg in segs:
                end_attr = seg.get("end") if isinstance(seg, dict) else getattr(seg, "end", None)
                if end_attr is not None:
                    try:
                        duration_s = max(duration_s, float(end_attr))
                    except (TypeError, ValueError):
                        continue

    # Cluster (diar_model, raw_label) → cluster_id once per pass. Two diar
    # models that identify the same speaker with different naming end up in
    # the same cluster — when embeddings are available. Without embeddings the
    # clusterer falls back to raw_label as cluster_id, in which case
    # cross-model "disagreement" reduces to literal-string comparison, which
    # is meaningless across diar conventions; we suppress the cross-model
    # signal in that case (set ``embeddings_available=False``).
    embeddings_available = bool(per_window_embeddings) and any(bool(v) for v in per_window_embeddings.values())
    cluster_map = cluster_speaker_labels_by_embedding(
        diar_ok,
        per_window_embeddings,
        cosine_threshold=cluster_cosine_threshold,
    )

    bucket_starts_ends = [(start, end) for start, end, _ in grid.iter_buckets(duration_s)]

    # Per-bucket raw label per diar model. None when the model emitted no
    # segment overlapping the bucket; we promote those to "<silent>" below.
    label_sequences: dict[str, list[str | None]] = {m: [] for m in diar_ok}
    for start, end in bucket_starts_ends:
        for m, block in diar_ok.items():
            label_sequences[m].append(diar_speaker_label_in_window(block.get("result"), start, end))

    # Per (diar_model, embedding_model, cluster_id): the (window_idx, embedding)
    # of the most recent prior bucket this diar model labelled with that cluster.
    prev_emb_per_track: dict[tuple[str, str, str], tuple[int, list[float]]] = {}
    # Per (diar_model, embedding_model): the (window_idx, embedding) of the
    # IMMEDIATELY PRIOR bucket — used to validate change claims.
    prev_emb_immediate: dict[tuple[str, str], tuple[int, list[float]]] = {}
    # Per diar model: the previous bucket's cluster_id (for change detection).
    prev_cluster_per_model: dict[str, str] = {}

    out: list[dict[str, Any]] = []
    for bucket_idx, (start, end) in enumerate(bucket_starts_ends):
        bucket_center = 0.5 * (start + end)
        votes: dict[str, dict[str, Any]] = {}

        cluster_this_bucket: dict[str, str] = {}

        for m in diar_ok:
            raw_label = label_sequences[m][bucket_idx]
            if raw_label is None:
                cluster_id = SILENT_CLUSTER_ID
                effective_label = "<silent>"
            else:
                cluster_id = cluster_map.get((m, raw_label), raw_label)
                effective_label = raw_label
            cluster_this_bucket[m] = cluster_id

            prev_cluster = prev_cluster_per_model.get(m)
            speaker_changed = (cluster_id != prev_cluster) if prev_cluster is not None else None

            votes[m] = {
                "speaker_label": effective_label,
                "cluster_id": cluster_id,
                "speaker_changed_from_prev": speaker_changed,
            }
            prev_cluster_per_model[m] = cluster_id

            # Silence carries no embedding signal — skip embedding sub-signals
            # for silent buckets but still update prev_cluster so transitions
            # silent ↔ speaking show up as speaker_changed in the next bucket.
            if cluster_id == SILENT_CLUSTER_ID:
                continue

            for emb_model_id in per_window_embeddings.keys():
                lookup = _embedding_for_bucket(per_window_embeddings, emb_model_id, bucket_center)
                if lookup is None:
                    continue
                window_idx, vec = lookup
                track_key = (m, emb_model_id, cluster_id)
                imm_key = (m, emb_model_id)

                # ── Same-cluster claim validation ───────────────────────
                prev_same = prev_emb_per_track.get(track_key)
                same_cos: float | None = None
                same_unc: float | None = None
                if prev_same is not None and prev_same[0] != window_idx:
                    same_cos = _cos_dist(vec, prev_same[1])
                    if same_cos is not None:
                        same_unc = calibrate_cosine_uncertainty(
                            same_cos,
                            same_speaker_floor=same_speaker_floor,
                            diff_speaker_floor=diff_speaker_floor,
                            direction="same",
                        )

                # ── Speaker-change claim validation ─────────────────────
                change_cos: float | None = None
                change_unc: float | None = None
                if speaker_changed is True:
                    prev_imm = prev_emb_immediate.get(imm_key)
                    if prev_imm is not None and prev_imm[0] != window_idx:
                        change_cos = _cos_dist(vec, prev_imm[1])
                        if change_cos is not None:
                            change_unc = calibrate_cosine_uncertainty(
                                change_cos,
                                same_speaker_floor=same_speaker_floor,
                                diff_speaker_floor=diff_speaker_floor,
                                direction="diff",
                            )

                votes[f"{m}::{emb_model_id}"] = {
                    "diar_model": m,
                    "embedding_model": emb_model_id,
                    "embedding_cosine_within_track": same_cos,
                    "same_label_uncertainty": same_unc,
                    "embedding_cosine_to_prev_bucket": change_cos,
                    "change_inconsistency_uncertainty": change_unc,
                }

                prev_emb_per_track[track_key] = (window_idx, vec)
                prev_emb_immediate[imm_key] = (window_idx, vec)

        # ── Cross-diar-model agreement ──────────────────────────────────
        # Compares cluster_ids (post-embedding-clustering) across diar models.
        # "<silent>" mismatch with a speech cluster IS a real disagreement.
        # Suppressed when no embeddings are available — without clustering,
        # different naming conventions would always look like disagreement.
        if embeddings_available and len(cluster_this_bucket) >= 2:
            models_sorted = sorted(cluster_this_bucket)
            n_pairs = 0
            n_disagree = 0
            for i in range(len(models_sorted)):
                for j in range(i + 1, len(models_sorted)):
                    n_pairs += 1
                    if cluster_this_bucket[models_sorted[i]] != cluster_this_bucket[models_sorted[j]]:
                        n_disagree += 1
            votes["__cross_diar_label_disagreement__"] = {
                "value": (n_disagree / n_pairs) if n_pairs > 0 else None,
                "n_pairs": n_pairs,
                "n_disagree": n_disagree,
                "cluster_ids": dict(cluster_this_bucket),
            }

        out.append({"start": start, "end": end, "votes": votes})

    return out
