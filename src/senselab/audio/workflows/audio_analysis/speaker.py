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
from typing import Any, Final, Mapping, Sequence

import numpy as np

from senselab.audio.workflows.audio_analysis.attribution import (
    speaker_assignment_doubt,
    target_activity_doubt,
    word_coverage,
)
from senselab.audio.workflows.audio_analysis.embeddings import (
    WindowEmbedding,
    calibrate_cosine_uncertainty,
    window_index_at,
)
from senselab.audio.workflows.audio_analysis.grid import BucketGrid
from senselab.audio.workflows.audio_analysis.harmonize import harmonize_from_diarization
from senselab.audio.workflows.audio_analysis.harvesters import diar_speaker_label_in_window
from senselab.audio.workflows.audio_analysis.joint import speaker_change_series
from senselab.audio.workflows.audio_analysis.occupancy import (
    count_posterior_in_window,
    spans_from_diarization,
)

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


CHANGE_POINT_BUCKET_REDUCTION = "max"
"""How a reporting bucket summarises the change-point boundaries inside it.

A decision rather than arithmetic, so it is named here and overridable rather than inlined as an
``argmax``. ``"max"`` is the default because a single sharp boundary surrounded by continuation *is*
a change, and averaging it against its neighbours would dilute it away — the boundaries inside one
bucket are near-duplicates of each other at a 50 ms hop, so a mean is closer to "how continuous was
this stretch" than to "did a change happen here". ``"mean"`` is available for a caller who wants the
latter question answered instead.
"""


_VOCAL_ACTIVITY: Final[tuple[str, ...]] = ("target_active", "nontarget_active")
"""Mask states that positively report a voice, whether or not it is the target's.

Named because they are the states under which the speaker axis's word gate does not apply: a mask
saying someone is vocalising outranks word absence as evidence about whether there is speech here.
The other three states — ``target_free``, ``indeterminate`` and ``None`` — report no voice, decline
to say, and "no region covered this bucket, possibly because the mask never ran" respectively.
"""


def harvest_speaker_votes(
    *,
    pass_summary: dict[str, Any],
    grid: BucketGrid,
    per_window_embeddings: dict[str, list[WindowEmbedding]],
    same_speaker_floor: float | None = 0.30,
    diff_speaker_floor: float | None = 0.70,
    speaker_floors: Mapping[str, tuple[float, float]] | None = None,
    cluster_cosine_threshold: float = 0.5,
    change_point_bucket_reduction: str = CHANGE_POINT_BUCKET_REDUCTION,
    fused_words: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Yield ``{"start", "end", "votes"}`` per bucket for the speaker axis.

    **The axis asks "how sure are we who is speaking here?"** — attribution, not change. Its two
    scored voters come from ``attribution``: ``speaker_assignment`` (do the diarizers agree who is
    here, measured over *all* the answers they gave, since absent a target embedding no speaker is
    privileged) and ``target_activity`` (do we know anyone was active at all). Both are gated by
    ``word_coverage`` **except where the mask reports a voice**: a bucket with no words has no speech
    to attribute and gets no claim, unless the region state is ``target_active`` or
    ``nontarget_active``, in which case the mask has positively measured a vocalization and neither
    of these word-independent voters may be dropped for lacking words (see :data:`_VOCAL_ACTIVITY`).
    Everything else this emits is a *measurement* other consumers read — the cluster assignments, the
    embedding cosines, the change points, the overlap distribution — and is deliberately unscored, so
    the fold sees two voters.

    It asked "was it the same speaker as before?" until 2026-08-05, scored per (diar × embedder) pair
    against embedding cosine. On a 0.1 s grid that asks ten times a second against 0.5 s embedding
    windows, and it read 0.666 on a conversation whose per-speaker presence doubt was 0.168. See
    ``specs/20260728-221507-per-speaker-identity-scene/speaker-axis-attribution-design.md``.

    Args:
        pass_summary: Per-task summary for one pass (diarization, alignment, etc.).
        grid: Bucket grid.
        per_window_embeddings: ``{embedding_model_id → [WindowEmbedding, ...]}``.
        change_point_bucket_reduction: How a bucket summarises the change-point boundaries it
            contains — ``"max"`` (default) or ``"mean"``. Named and overridable because it is a
            decision, not arithmetic: see :data:`CHANGE_POINT_BUCKET_REDUCTION`.
        same_speaker_floor: ``None`` when the pass admits no usable calibration band, in
            which case the embedding sub-signals are omitted entirely. Otherwise, cosine
            distance ≤ this is treated as confidently
            same-speaker (uncertainty 0 for same-claim, 1 for change-claim).
        diff_speaker_floor: Cosine distance ≥ this is treated as confidently
            different-speaker (uncertainty 1 for same-claim, 0 for change-claim).
        speaker_floors: ``{embedding model → (same_floor, diff_floor)}`` measured empirically for
            that model's own cosine distribution. **Per embedder, because a cosine band is a property
            of the embedding space and not of the pass**: ecapa's same/different separation is not
            resnet's, and one pass-level pair calibrated every embedder's distances with whichever
            model happened to be measured — silently, since the clustering loop kept only the first.
            An embedder absent from the map falls back to the ``*_speaker_floor`` arguments, which is
            the honest default: no empirical band was measured for it.
        cluster_cosine_threshold: Cosine similarity threshold for clustering
            (diar_model, raw_label) into pass-wide speaker IDs. 0.5 is roughly
            the EER for ECAPA on VoxCeleb.
        fused_words: The consensus words from ``asr.fuse_consensus_words``, used as a **gate** rather
            than as a voter: a bucket no word occupies has no speech to attribute, so the axis makes
            no claim there — unless the background mask reports a voice in it, which is measured
            evidence the word proxy is wrong. Word timing bounds *where* a speaker change can be; it
            is not evidence about *who*, and folding it in as doubt swamped the per-speaker term with
            ~0.223 of standing jitter (see ``attribution.word_coverage``). Empty or ``None`` disables
            the gate entirely: with no word measured anywhere, a bucket's emptiness carries no
            information, and gating on it would delete the axis rather than sharpen it.

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
                    "calibrated_same_doubt": float | None,
                    "embedding_cosine_to_prev_bucket": float | None,
                    "calibrated_change_doubt": float | None,
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
    _default_same = None if same_speaker_floor is None else float(same_speaker_floor)
    _default_diff = None if diff_speaker_floor is None else float(diff_speaker_floor)
    _floors = dict(speaker_floors or {})

    def _band(embedder: str) -> tuple[float | None, float | None]:
        """The calibration band for ``embedder``: its own if measured, the CLI default otherwise."""
        measured = _floors.get(embedder)
        if measured is None:
            return _default_same, _default_diff
        return float(measured[0]), float(measured[1])

    bucket_starts_ends = [(start, end) for start, end, _ in grid.iter_buckets(duration_s)]

    # Span sets for J1, each carrying its tool's declared speaker capacity (D-19). Derived from the
    # pass summary this harvest already has, rather than passed in: the diarizers *are* the speaker
    # evidence, so needing them handed over separately was an artifact of the count coming from one
    # model's frame channels.
    diar_spans = spans_from_diarization(diar_blocks)

    # J2 — where the voice changes, computed once per pass over the embedding windows. Boundary
    # times are on the embedding hop (50 ms by default), far finer than the reporting grid, so each
    # bucket reads the boundaries that fall inside it rather than the series being resampled: a
    # change point is an instant, and averaging it into a bucket would blunt exactly the
    # localisation the fine hop was chosen to buy.
    # No calibration band means the embeddings were *measured* not to separate speakers on this
    # pass, so the change-point signal drops out rather than borrowing the library anchors — the
    # same FR-007 rule the other embedding sub-signals follow, and for the same reason: on this
    # axis a confident derived signal outvotes unanimous diarizer agreement.
    change_by_model: dict[str, dict[str, Any]] = {}
    for emb_model in sorted(per_window_embeddings or {}):
        _cp_same, _cp_diff = _band(emb_model)
        if _cp_same is not None and _cp_diff is not None:
            series = speaker_change_series(
                per_window_embeddings.get(emb_model) or [],
                same_speaker_floor=_cp_same,
                diff_speaker_floor=_cp_diff,
            )
            if series is not None:
                change_by_model[emb_model] = series

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
                same_floor, diff_floor = _band(emb_model_id)
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

                # **L1 measurements, not L2 voters.** The cosines and their per-embedder calibrated
                # readings all stay — ``_signal_rows_from_buckets`` copies them into
                # ``L1/<pass>/signals/<diar::emb>.parquet``, which is where a measurement belongs, and
                # the per-embedder band that produced them is a measured property of that embedding
                # space.
                #
                # What changed is the *names*. They were ``same_label_uncertainty`` and
                # ``change_inconsistency_uncertainty``, which are the codebase's scored-field names
                # (``fuse._UNCERTAINTY_FIELDS``), so the fold read them as the axis's voters — asking
                # "same speaker as before?" at the grid rate against embeddings windowed ten times
                # coarser, which is what read 0.666 on a clean conversation whose per-speaker presence
                # doubt was 0.168. Under these names the fold does not read them and the axis is
                # composed from ``attribution``'s three terms instead.
                votes[f"{m}::{emb_model_id}"] = {
                    "diar_model": m,
                    "embedding_model": emb_model_id,
                    "embedding_cosine_within_track": same_cos,
                    "calibrated_same_doubt": same_unc,
                    "embedding_cosine_to_prev_bucket": change_cos,
                    "calibrated_change_doubt": change_unc,
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
            # ``cluster_ids`` is what ``_bucket_clusters`` reads, so the block stays. Its ``value``
            # does not: the axis's per-speaker term is computed from these same assignments by
            # ``attribution.speaker_assignment_doubt``, and scoring the pair-disagreement
            # fraction beside it would count one body of evidence twice (D-21 rule 6).
            votes["__cross_diar_label_disagreement__"] = {
                "n_pairs": n_pairs,
                "n_disagree": n_disagree,
                "cluster_ids": dict(cluster_this_bucket),
            }

        # J2 — a change point inside this bucket is evidence about the speaker axis.
        for emb_model, series in change_by_model.items():
            inside = (series["times"] >= start) & (series["times"] < end)
            if not inside.any():
                continue
            probs = series["p_change"][inside]
            best = (
                int(np.argmax(probs))
                if change_point_bucket_reduction == "max"
                else int(np.argmin(np.abs(probs - float(np.mean(probs)))))
            )
            # Unscored, for the same reason as the pair entries above: a change point is evidence
            # about *when* the speaker changed, not about how sure we are who is speaking. It stays
            # because ``identity_repair`` reads it to place boundaries.
            votes[f"{emb_model}::change_point"] = {
                "change_uncertainty": float(series["uncertainty"][inside][best]),
                "p_change": float(series["p_change"][inside][best]),
                "cosine_distance": float(series["distance"][inside][best]),
                "boundary_s": float(series["times"][inside][best]),
                "n_boundaries": int(inside.sum()),
                "bucket_reduction": change_point_bucket_reduction,
                # Neighbouring boundaries share almost all of their audio, so they are not
                # independent evidence; recorded so a consumer cannot treat them as such.
                "resolution_s": float(series["hop_s"]),
                "native_window_s": float(series["window_s"]),
            }

        # J1 — how many speakers are simultaneously active, from **cross-diarizer spread** rather
        # than from one model's per-channel probabilities (D-19). The old construction was a
        # Poisson-binomial over segmentation-3.0's channels, treating them as independent
        # Bernoullis; they are a powerset conversion, mutually exclusive by construction, so that
        # independence was never there. A count is still permutation-invariant, so this is
        # answerable before the speaker↔label binding D-7 hands to rounds.
        j1 = count_posterior_in_window(diar_spans, start=start, end=end) if diar_spans else None
        if j1 is not None and j1["uncertainty"] is not None:
            # **An L1 measurement, not a voter and not a synthetic block.** Unscored, because how
            # many speakers overlap is evidence about *who*, which the per-speaker term already reads
            # from the same spans — scoring both counts one body of evidence twice (D-21 rule 6).
            #
            # Named without the ``__`` prefix deliberately: ``votes._signal_rows_from_buckets`` skips
            # ``__``-prefixed entries, so under the old name this had no reader at all once it stopped
            # being scored — recorded-but-unread, which is exactly the
            # ``__pairwise_phoneme_distances__`` mistake. As a plain name it lands in
            # ``L1/<pass>/signals/overlap_count.parquet``, where a per-bucket measurement belongs, and
            # ``count_uncertainty`` keeps it out of ``fuse._UNCERTAINTY_FIELDS``.
            votes["overlap_count"] = {
                "count_uncertainty": float(j1["uncertainty"]),
                "expected_count": j1["expected_count"],
                "p_overlap": j1["p_overlap"],
                # The distribution itself, keyed by count, so a consumer can see *which* counts
                # were in contention rather than only how much doubt there was between them.
                "count_distribution": {str(k): v for k, v in j1["counts"].items()},
                "n_samples": j1["n_samples"],
                # A tool at its ceiling contributes a lower bound, so the bucket's figure inherits
                # it. Without this a bounded count reads as a settled one.
                "lower_bounded": j1["lower_bounded"],
                "censored_sources": list(j1["censored_sources"]),
                "contributing_sources": list(j1["contributing_sources"]),
            }

        out.append({"start": start, "end": end, "votes": votes})

    # ── the attribution voters ──
    # Added in a second pass because two of the three are per-bucket projections of whole-pass
    # evidence (the consensus words, the mask regions) and the third reads the cluster assignments
    # the loop above has just finished writing.
    buckets = [(round(float(b["start"]), 6), round(float(b["end"]), 6)) for b in out]
    mask_doc = ((pass_summary.get("background_mask") or {}).get("result")) or {}
    mask_regions = mask_doc.get("regions") or []
    coverage = word_coverage(list(fused_words or ()), buckets)
    activity = target_activity_doubt(mask_regions, buckets)

    for bucket_dict in out:
        key = (round(float(bucket_dict["start"]), 6), round(float(bucket_dict["end"]), 6))
        votes = bucket_dict["votes"]
        doubt, state = activity[key]
        if state == "target_free":
            # Nobody to attribute, so no claim at all. ``0.0`` would assert confident attribution
            # where no attribution was made, which is a claim of a different kind.
            bucket_dict["votes"] = {}
            continue
        if state not in _VOCAL_ACTIVITY and fused_words and coverage[key] <= 0.0:
            # **No words here, and no mask evidence of a voice, so no speech to attribute.** Word
            # timing is used to *sharpen* the question rather than to vote on it: its one job is
            # telling us when there is nothing to be uncertain about. Measured on a clean two-speaker
            # conversation, 22 of the 29 buckets the axis flagged were wordless — the inter-turn
            # silence where the four diarizers disagree about exactly where the boundary falls. There
            # is no speaker to get wrong in a gap between turns, so the axis makes no claim rather
            # than reporting the disagreement.
            #
            # Word absence is a *proxy* for speech absence, and it holds only for adult connected
            # speech: a cry, a cough or a groan is a voice with no words in it. So the proxy yields
            # to the mask wherever the mask positively reports vocal activity, because both voters
            # below are word-independent — ``speaker_assignment`` is entropy over diarizer cluster
            # assignments and ``target_activity`` is the region state itself — and neither is
            # evidence the word gate is entitled to discard. Two cases were being zeroed silently:
            # a non-lexical vocalization, which lands in ``nontarget_active``, and real speech the
            # ASR missed, which lands in ``target_active``. Everything else keeps the old reading:
            # true silence is ``target_free`` (returned above) or ``indeterminate``/``None``, and
            # ``None`` also means the mask stage never ran, so it can license nothing.
            #
            # Gated on the fold having produced **at least one word anywhere**, which is what makes
            # this bucket's emptiness a measurement rather than an absence of measurement. Two
            # failures it rules out, both of which null the entire axis silently: a run with
            # ``stages.asr: false`` (``harvest_pass`` hands over ``[]``, not ``None``, so an
            # ``is not None`` check does not catch it — an existing test did), and a fold that read no
            # words because every result shape was unconvertible, which is the defect that once left
            # the asr axis with zero contributing signals over a whole recording.
            bucket_dict["votes"] = {}
            continue
        clusters = _bucket_clusters(bucket_dict)
        assignment = speaker_assignment_doubt(clusters)
        if assignment is not None:
            votes["speaker_assignment"] = {
                "value": assignment,
                "operator": "over_speakers/entropy",
                # How many independent sources stand behind this one number, and what each said.
                # This axis has a *single* scored signal that folds every diarizer, so without these
                # the axis reads exactly as confident as one resting on four independent signals, and
                # ``epistemic_uncertainty`` reports 0.0 — not because the diarizers agreed but
                # because the spread was collapsed before the fold that measures spread saw it. The
                # same defect the asr axis had while ``consensus_words`` was its only voter.
                "n_sources": len(clusters),
                "source_outcomes": dict(sorted(clusters.items())),
            }
        if doubt is not None:
            votes["target_activity"] = {"value": doubt, "operator": "mask_region/gated_on_state", "state": state}

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
