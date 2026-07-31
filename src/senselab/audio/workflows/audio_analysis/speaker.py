"""Identity axis vote harvesters — "was it the same speaker?".

The speaker question splits into three independent diagnostic checks per bucket:

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

import math
from typing import Any, Mapping, Sequence

from senselab.audio.workflows.audio_analysis.embeddings import (
    WindowEmbedding,
    calibrate_cosine_uncertainty,
    window_index_at,
)
from senselab.audio.workflows.audio_analysis.grid import BucketGrid
from senselab.audio.workflows.audio_analysis.harmonize import harmonize_from_diarization
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


def harvest_speaker_votes(
    *,
    pass_summary: dict[str, Any],
    grid: BucketGrid,
    per_window_embeddings: dict[str, list[WindowEmbedding]],
    same_speaker_floor: float | None = 0.30,
    diff_speaker_floor: float | None = 0.70,
    cluster_cosine_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Yield ``{"start", "end", "votes"}`` per bucket for the speaker axis.

    Args:
        pass_summary: Per-task summary for one pass (diarization, alignment, etc.).
        grid: Bucket grid.
        per_window_embeddings: ``{embedding_model_id → [WindowEmbedding, ...]}``.
        same_speaker_floor: ``None`` when the pass admits no usable calibration band, in
            which case the embedding sub-signals are omitted entirely. Otherwise, cosine
            distance ≤ this is treated as confidently
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
                    "cluster_id": "C0" | "C1" | ... | "SIL",
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
    # models that identify the same speaker with different naming end up on the same common id.
    # H2 (D-6): two independent matchers — temporal overlap and embedding centroid — with their
    # disagreement recorded as the assignment's own uncertainty. The previous single-matcher
    # clusterer produced a point assignment with no way to express doubt about itself, so a wrong
    # guess about which label denotes whom propagated as fact and surfaced downstream as two models
    # "disagreeing" when they had never been correctly compared.
    harmonization = harmonize_from_diarization(
        diar_ok,
        per_window_embeddings,
        min_similarity=cluster_cosine_threshold,
    )
    cluster_map = harmonization.mapping
    # Whether the labels were actually mapped into a shared space, as opposed to each model keeping
    # its own names. Cross-model comparison is only meaningful once they were.
    harmonized = bool(cluster_map)

    # An embedding whose same- and between-speaker distances overlap cannot validate an
    # speaker claim at this window scale. Its sub-signals then drop out (FR-007) rather
    # than voting: substituting a fixed band that sits below every distance the embedding
    # produces turns "cannot tell" into "confidently different", and one such saturated
    # derived signal outvotes unanimous diarizer agreement.
    # Bound to narrowed locals rather than checked via a boolean: mypy cannot narrow
    # ``float | None`` through a separate flag, and the alternative was three ignore comments on
    # calls that are in fact guarded.
    same_floor = None if same_speaker_floor is None else float(same_speaker_floor)
    diff_floor = None if diff_speaker_floor is None else float(diff_speaker_floor)

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
            # How well-determined this model's mapping into the common space was. Carried onto the
            # vote so it reaches the axis rather than being discarded at the harmonization step: a
            # bucket whose label assignment is contested must not read as confidently identified.
            if raw_label is not None:
                assignment_unc = harmonization.uncertainty.get((m, raw_label))
                if assignment_unc is not None:
                    votes[m]["assignment_uncertainty"] = float(assignment_unc)
                agreed = harmonization.methods_agreed.get((m, raw_label))
                if agreed is not None:
                    votes[m]["assignment_methods_agreed"] = bool(agreed)
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
                    if same_cos is not None and same_floor is not None and diff_floor is not None:
                        same_unc = calibrate_cosine_uncertainty(
                            same_cos,
                            same_speaker_floor=same_floor,
                            diff_speaker_floor=diff_floor,
                            direction="same",
                        )

                # ── Speaker-change claim validation ─────────────────────
                change_cos: float | None = None
                change_unc: float | None = None
                if speaker_changed is True:
                    prev_imm = prev_emb_immediate.get(imm_key)
                    if prev_imm is not None and prev_imm[0] != window_idx:
                        change_cos = _cos_dist(vec, prev_imm[1])
                        if change_cos is not None and same_floor is not None and diff_floor is not None:
                            change_unc = calibrate_cosine_uncertainty(
                                change_cos,
                                same_speaker_floor=same_floor,
                                diff_speaker_floor=diff_floor,
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
        # Compares harmonized common-space ids across diar models. A "<silent>" mismatch against a
        # speech cluster IS a real disagreement.
        #
        # No longer suppressed when embeddings are unavailable. That suppression was correct while
        # the label mapping came from embedding centroids alone, because its no-embedding fallback
        # used the raw label string as the cluster id — so agreement reduced to comparing
        # ``SPEAKER_00`` against ``spk1``, which can only ever report disagreement. H2's overlap
        # matcher settles that case from timing evidence, so the signal is now measurable without
        # embeddings, and withholding it would discard evidence we have.
        if harmonized and len(cluster_this_bucket) >= 2:
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


# ── per-speaker structure from the harvested evidence (T098) ──────────
#
# The per-bucket axis above stays the evidence-gathering mechanism. What follows only
# *reads* what it harvested — no new inference, no model access, no I/O — and reshapes it
# from "how uncertain was speaker here?" into "which speaker was in doubt, and when?".
#
# The reshape is what makes the uncertainty actionable. A single per-bucket scalar says a
# region is contested but not who is contested in it, so no follow-up can be targeted at a
# speaker. Per-speaker rows name the subject of the doubt, which is the whole point of
# moving the axis (FR-003).


def _bucket_clusters(vote_bucket: Mapping[str, Any]) -> dict[str, str]:
    """Per diar model, the cluster it placed in this bucket."""
    votes = vote_bucket.get("votes") or {}
    if not isinstance(votes, Mapping):
        return {}
    cross = votes.get("__cross_diar_label_disagreement__")
    if isinstance(cross, Mapping) and isinstance(cross.get("cluster_ids"), Mapping):
        return {str(m): str(c) for m, c in cross["cluster_ids"].items()}
    return {
        str(model): str(entry["cluster_id"])
        for model, entry in votes.items()
        if isinstance(entry, Mapping) and entry.get("cluster_id") is not None and "::" not in str(model)
    }


def _binary_entropy(p: float) -> float:
    """Normalized Shannon entropy of a two-outcome split; 0 = unanimous, 1 = evenly split."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p)))


def per_speaker_tracks(
    speaker_votes: Sequence[Mapping[str, Any]],
    *,
    speaker_ids: Mapping[str, str] | None = None,
    silent_cluster_id: str = SILENT_CLUSTER_ID,
) -> list[dict[str, Any]]:
    """Per-speaker, per-bucket speech_presence rows derived from the harvested speaker votes.

    ``speech_presence_confidence`` is the share of diar models active in the bucket that placed
    this speaker there, and ``speech_presence_uncertainty`` the normalized entropy of that split —
    so a speaker every model agrees on carries no doubt, and one the models are evenly
    divided over carries all of it. Models reporting silence stay in the denominator: a lone
    detection among four silent models is exactly the case that must not read as certain.

    A speaker absent from a bucket gets no row there. Rows are evidence of speech_presence, and an
    absent row is not the same claim as speech_presence at confidence zero — the second would put
    a positive assertion about every speaker in every bucket into the output.

    Args:
        speaker_votes: Bucket dicts from :func:`harvest_speaker_votes`.
        speaker_ids: Optional ``{cluster_id → fused speaker id}``. A cluster with no mapping
            keeps its own id rather than being dropped: a cluster the count posterior does
            not back is still observed evidence, and dropping it would make the surplus
            invisible instead of contested.
        silent_cluster_id: The pseudo-cluster standing for "no speaker here", which is a
            bookkeeping device rather than a person and never produces a row.

    Returns:
        Rows ordered by ``(start, cluster_id)`` — the outputs are asserted byte-identical
        across runs, so ordering must not depend on dict insertion order.
    """
    mapping = dict(speaker_ids or {})
    rows: list[dict[str, Any]] = []
    for bucket in speaker_votes:
        clusters = _bucket_clusters(bucket)
        if not clusters:
            continue
        n_models = len(clusters)
        active = sorted({c for c in clusters.values() if c != silent_cluster_id})
        for cluster in active:
            sources = sorted(m for m, c in clusters.items() if c == cluster)
            share = len(sources) / n_models
            rows.append(
                {
                    "cluster_id": cluster,
                    "speaker_id": mapping.get(cluster, cluster),
                    "start": float(bucket.get("start", 0.0)),
                    "end": float(bucket.get("end", 0.0)),
                    "speech_presence_confidence": share,
                    "speech_presence_uncertainty": _binary_entropy(share),
                    "overlap_with": [c for c in active if c != cluster],
                    "contributing_sources": sources,
                }
            )
    rows.sort(key=lambda r: (r["start"], r["cluster_id"]))
    return rows


def cluster_active_time(
    speaker_votes: Sequence[Mapping[str, Any]],
    *,
    silent_cluster_id: str = SILENT_CLUSTER_ID,
) -> dict[str, float]:
    """Total time each cluster was claimed by at least one model, most active first.

    Which cluster becomes ``S0`` must be a property of the evidence rather than of
    iteration order, so ties break by cluster id.
    """
    totals: dict[str, float] = {}
    for bucket in speaker_votes:
        span = float(bucket.get("end", 0.0)) - float(bucket.get("start", 0.0))
        for cluster in {c for c in _bucket_clusters(bucket).values() if c != silent_cluster_id}:
            totals[cluster] = totals.get(cluster, 0.0) + span
    return dict(sorted(totals.items(), key=lambda kv: (-kv[1], kv[0])))


def label_correspondence_rows(
    speaker_votes: Sequence[Mapping[str, Any]],
    *,
    speaker_ids: Mapping[str, str],
    silent_cluster_id: str = SILENT_CLUSTER_ID,
) -> list[dict[str, Any]]:
    """Map each diar model's own speaker label to the fused speaker it became (FR-005).

    Every diarizer invents its own labels, so a consumer cannot act on a fused speaker id
    without knowing which of its own labels produced it. A label that appears under more
    than one cluster yields one row per cluster: collapsing to a single mapping would hide
    a genuine instability in the clustering behind a tidy-looking table.
    """
    seen: set[tuple[str, str, str]] = set()
    for bucket in speaker_votes:
        votes = bucket.get("votes") or {}
        if not isinstance(votes, Mapping):
            continue
        for model, entry in votes.items():
            if not isinstance(entry, Mapping) or "::" in str(model) or entry.get("cluster_id") is None:
                continue
            cluster = str(entry["cluster_id"])
            if cluster == silent_cluster_id:
                continue
            seen.add((str(model), str(entry.get("speaker_label") or ""), cluster))
    return [
        {
            "source": model,
            "source_label": label,
            "cluster_id": cluster,
            "speaker_id": speaker_ids.get(cluster, cluster),
        }
        for model, label, cluster in sorted(seen)
    ]
