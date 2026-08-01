"""H2: the common speaker space, and the uncertainty of constructing it.

Every diarizer names its speakers arbitrarily — ``SPEAKER_00``, ``spk0`` — and the names carry no
meaning across models. So any cross-model statement about speaker first *guesses* that two labels
denote the same person. Treated as fact, that guess makes two models which were never correctly
compared read as disagreeing, and speaker uncertainty then stays high in exactly the regions where
per-speaker speech_presence is unambiguous. That is the observation this module exists to address.

**Harmonization is therefore an estimation step and reports its own uncertainty.** Two independent
matchers run over the same labels:

- **temporal overlap** — a one-to-one assignment maximising co-occurrence duration (Hungarian);
- **embedding centroid** — a one-to-one assignment maximising mean-embedding cosine similarity.

Where they agree, the assignment is confident. Where they disagree, that disagreement *is* the
assignment uncertainty, measured with the same estimators as every other axis: normalised Shannon
entropy over the candidate targets, and weighted vote share for the winner. Neither matcher alone
can express doubt about itself, which is why both run (D-6).

Three id namespaces stay distinct because all three once rendered as ``S0``: a model's own labels
(``SPEAKER_00``, ``spk0``), the harmonized cluster produced here (``C0``), and the fused speaker id
in ``final/speakers.json`` (``S0``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from senselab.audio.workflows.audio_analysis.statistics import confidence, entropy_uncertainty

__all__ = [
    "TranscriptHarmonization",
    "TranscriptSlot",
    "harmonize_transcripts",
    "SpeakerHarmonization",
    "centroid_assignment",
    "harmonize_from_diarization",
    "harmonize_speaker_labels",
    "overlap_assignment",
]

MIN_CENTROID_SIMILARITY = 0.5
"""Cosine similarity below which a centroid pairing is not proposed at all.

Without a floor the Hungarian assignment would pair the least-dissimilar remaining labels no matter
how unlike they are, manufacturing a match between two speakers who simply both went unmatched."""


@dataclass
class SpeakerHarmonization:
    """A common speaker space plus how well-determined each assignment into it was.

    Attributes:
        mapping: ``(model, label) → common id`` (``C0``, ``C1``, …).
        confidence: ``(model, label) → P(assignment correct)``, the matchers' weighted vote share
            for the chosen target.
        uncertainty: ``(model, label) → normalised entropy`` over candidate targets, or ``None``
            when only one matcher ran — unmeasured rather than measured-and-zero.
        methods_agreed: ``(model, label) → True | False | None``; ``None`` when no second matcher
            was available to agree or disagree.
        reference_model: The model whose labels seeded the space. Its own assignments are correct
            by construction, so they are exempt from agreement checks — exposed rather than left
            implicit, because a consumer auditing agreement would otherwise read those exempt
            entries as unmeasured failures.
    """

    mapping: dict[tuple[str, str], str] = field(default_factory=dict)
    confidence: dict[tuple[str, str], float] = field(default_factory=dict)
    uncertainty: dict[tuple[str, str], Optional[float]] = field(default_factory=dict)
    methods_agreed: dict[tuple[str, str], Optional[bool]] = field(default_factory=dict)
    reference_model: Optional[str] = None

    def to_json(self) -> dict[str, Any]:
        """Serialise with ``model::label`` string keys, for the decision log and parquet."""

        def _key(k: tuple[str, str]) -> str:
            return f"{k[0]}::{k[1]}"

        return {
            "mapping": {_key(k): v for k, v in sorted(self.mapping.items())},
            "confidence": {_key(k): v for k, v in sorted(self.confidence.items())},
            "uncertainty": {_key(k): v for k, v in sorted(self.uncertainty.items())},
            "methods_agreed": {_key(k): v for k, v in sorted(self.methods_agreed.items())},
            "reference_model": self.reference_model,
        }


def _seg_fields(seg: Any) -> Optional[tuple[float, float, str]]:  # noqa: ANN401
    """Coerce a diarization segment (dict or object) to ``(start, end, label)``."""
    if isinstance(seg, dict):
        start, end = seg.get("start"), seg.get("end")
        label = seg.get("speaker", seg.get("speaker_label", seg.get("label")))
    else:
        start, end = getattr(seg, "start", None), getattr(seg, "end", None)
        label = getattr(seg, "speaker", None) or getattr(seg, "label", None)
    if start is None or end is None or label is None:
        return None
    try:
        return float(start), float(end), str(label)
    except (TypeError, ValueError):
        return None


def _labels_of(segments: Sequence[Any]) -> list[str]:
    """Sorted distinct labels, so downstream ordering never depends on segment order."""
    out: set[str] = set()
    for seg in segments:
        parsed = _seg_fields(seg)
        if parsed is not None:
            out.add(parsed[2])
    return sorted(out)


def _overlap_seconds(a: Sequence[Any], b: Sequence[Any], label_a: str, label_b: str) -> float:
    """Total time both labels are simultaneously active."""
    spans_a = [(s, e) for s, e, lab in filter(None, map(_seg_fields, a)) if lab == label_a]
    spans_b = [(s, e) for s, e, lab in filter(None, map(_seg_fields, b)) if lab == label_b]
    total = 0.0
    for s1, e1 in spans_a:
        for s2, e2 in spans_b:
            total += max(0.0, min(e1, e2) - max(s1, s2))
    return total


def _maximise(score: np.ndarray) -> list[tuple[int, int]]:
    """One-to-one pairing maximising total score.

    Uses ``scipy.optimize.linear_sum_assignment`` when available, else a greedy fallback that takes
    the highest remaining score and removes both its row and column. Greedy is not optimal but is
    still a *matching*, which is the property that matters: a plain per-row argmax would let two
    labels of one model collapse onto one label of another, silently merging two people.
    """
    if score.size == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(-score)
        return list(zip((int(r) for r in rows), (int(c) for c in cols)))
    except ImportError:
        pairs: list[tuple[int, int]] = []
        used_r: set[int] = set()
        used_c: set[int] = set()
        order = np.argsort(score, axis=None)[::-1]
        for flat in order:
            r, c = int(flat // score.shape[1]), int(flat % score.shape[1])
            if r in used_r or c in used_c:
                continue
            pairs.append((r, c))
            used_r.add(r)
            used_c.add(c)
        return sorted(pairs)


def overlap_assignment(a: Sequence[Any], b: Sequence[Any]) -> dict[str, str]:
    """Match ``a``'s labels to ``b``'s by maximising total co-occurrence duration.

    Args:
        a: Diarization segments from one model.
        b: Diarization segments from another.

    Returns:
        ``{a_label → b_label}``, one-to-one, omitting labels with no positive overlap — a pairing
        with zero shared time is not evidence of anything.
    """
    labels_a, labels_b = _labels_of(a), _labels_of(b)
    if not labels_a or not labels_b:
        return {}
    score = np.zeros((len(labels_a), len(labels_b)), dtype=np.float64)
    for i, la in enumerate(labels_a):
        for j, lb in enumerate(labels_b):
            score[i, j] = _overlap_seconds(a, b, la, lb)
    return {labels_a[i]: labels_b[j] for i, j in _maximise(score) if score[i, j] > 0.0}


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def centroid_assignment(
    a: Mapping[str, np.ndarray],
    b: Mapping[str, np.ndarray],
    *,
    min_similarity: float = MIN_CENTROID_SIMILARITY,
) -> dict[str, str]:
    """Match labels between two models by mean-embedding cosine similarity.

    Args:
        a: ``{label → centroid}`` for one model.
        b: ``{label → centroid}`` for another.
        min_similarity: Pairings below this are not proposed.

    Returns:
        ``{a_label → b_label}``, one-to-one.
    """
    labels_a, labels_b = sorted(a), sorted(b)
    if not labels_a or not labels_b:
        return {}
    score = np.zeros((len(labels_a), len(labels_b)), dtype=np.float64)
    for i, la in enumerate(labels_a):
        for j, lb in enumerate(labels_b):
            score[i, j] = _cosine(np.asarray(a[la], dtype=np.float64), np.asarray(b[lb], dtype=np.float64))
    return {labels_a[i]: labels_b[j] for i, j in _maximise(score) if score[i, j] >= float(min_similarity)}


def harmonize_speaker_labels(
    per_model_segments: Mapping[str, Sequence[Any]],
    *,
    centroids: Optional[Mapping[tuple[str, str], np.ndarray]] = None,
    min_similarity: float = MIN_CENTROID_SIMILARITY,
) -> SpeakerHarmonization:
    """Build a common speaker space across models, recording how determined each assignment was.

    Args:
        per_model_segments: ``{model → diarization segments}``.
        centroids: ``{(model, label) → mean embedding}``. When omitted only the overlap matcher
            runs, and ``methods_agreed`` / ``uncertainty`` are ``None`` throughout — one matcher
            cannot corroborate itself.
        min_similarity: Floor for the centroid matcher.

    Returns:
        A :class:`SpeakerHarmonization`.

    The first model in sorted order seeds the space, so ids do not depend on dict iteration order:
    they appear in outputs and in the decision log, and an order-dependent id would make two
    identical runs look like they disagreed.
    """
    models = sorted(per_model_segments)
    result = SpeakerHarmonization()
    if not models:
        return result

    # The reference model's labels seed the common space.
    reference = models[0]
    result.reference_model = reference
    for idx, label in enumerate(_labels_of(per_model_segments[reference])):
        key = (reference, label)
        result.mapping[key] = f"C{idx}"
        # The reference defines the space, so its assignment is correct by construction --
        # unconditionally, not contingent on whether centroids happened to be supplied -- and
        # there is nothing for a second matcher to corroborate.
        result.confidence[key] = 1.0
        result.uncertainty[key] = 0.0
        result.methods_agreed[key] = None
    next_id = len(result.mapping)

    ref_segments = per_model_segments[reference]
    ref_centroids = {lab: vec for (m, lab), vec in centroids.items() if m == reference} if centroids is not None else {}

    for model in models[1:]:
        segments = per_model_segments[model]
        by_overlap = overlap_assignment(segments, ref_segments)
        model_centroids = (
            {lab: vec for (m, lab), vec in centroids.items() if m == model} if centroids is not None else {}
        )
        by_centroid = (
            centroid_assignment(model_centroids, ref_centroids, min_similarity=min_similarity)
            if model_centroids and ref_centroids
            else {}
        )
        have_second_matcher = bool(by_centroid)

        for label in _labels_of(segments):
            key = (model, label)
            # Each matcher votes for a common id, or abstains.
            votes: dict[str, float] = {}
            targets: dict[str, Optional[str]] = {}
            for method, assignment in (("overlap", by_overlap), ("centroid", by_centroid)):
                ref_label = assignment.get(label)
                target = result.mapping.get((reference, ref_label)) if ref_label else None
                targets[method] = target
                if target is not None:
                    votes[target] = votes.get(target, 0.0) + 1.0

            if not votes:
                # Neither matcher placed this label: it is a speaker the reference did not report,
                # so it gets its own id rather than being forced onto an existing one.
                result.mapping[key] = f"C{next_id}"
                next_id += 1
                result.confidence[key] = 1.0
                result.uncertainty[key] = None
                result.methods_agreed[key] = None
                continue

            winner = max(sorted(votes), key=lambda t: votes[t])
            result.mapping[key] = winner
            picked = confidence([t == winner for t in targets.values() if t is not None])
            result.confidence[key] = 1.0 if picked is None else picked
            if have_second_matcher and targets["overlap"] is not None and targets["centroid"] is not None:
                result.methods_agreed[key] = targets["overlap"] == targets["centroid"]
                # Entropy over what the matchers proposed: agreement collapses to one outcome
                # (0.0), a split between two candidates is maximal (1.0).
                result.uncertainty[key] = entropy_uncertainty(votes)
            else:
                result.methods_agreed[key] = None
                result.uncertainty[key] = None

    return result


def harmonize_from_diarization(
    diar_blocks: Mapping[str, Any],
    per_window_embeddings: Optional[Mapping[str, Sequence[Any]]] = None,
    *,
    min_similarity: float = MIN_CENTROID_SIMILARITY,
) -> SpeakerHarmonization:
    """Adapter: harmonize labels straight from the pipeline's diarization blocks.

    Args:
        diar_blocks: ``{model → diar block}`` as carried on a pass summary.
        per_window_embeddings: ``{embedding model → windows}``; the alphabetically first model is
            used to build per-label centroids. One metric is enough to define a matcher, and using
            both at once would need multi-modal centroid logic that adds little for 2-4 speakers.
        min_similarity: Floor for the centroid matcher.

    Returns:
        A :class:`SpeakerHarmonization`. Without embeddings only the overlap matcher runs, and the
        assignment uncertainties are ``None`` throughout.

    Replaces ``clustering.cluster_speaker_labels_by_embedding``, which matched on embedding
    centroids alone and so produced a point assignment with no way to express doubt about itself.
    Its no-embeddings fallback used the raw label string as the cluster id, which makes cross-model
    "agreement" a comparison of naming conventions — the overlap matcher answers that case with
    actual evidence instead.
    """
    from senselab.audio.workflows.audio_analysis.clustering import (
        _diar_segments,
        _mean_window_embedding_over_segments,
        _seg_attr,
    )

    segments_by_model: dict[str, list[Any]] = {}
    segs_by_key: dict[tuple[str, str], list[Any]] = {}
    for model, block in diar_blocks.items():
        segs = _diar_segments(block)
        segments_by_model[str(model)] = list(segs)
        for seg in segs:
            label = str(_seg_attr(seg, "speaker") or "?")
            segs_by_key.setdefault((str(model), label), []).append(seg)

    centroids: Optional[dict[tuple[str, str], np.ndarray]] = None
    if per_window_embeddings and any(bool(v) for v in per_window_embeddings.values()):
        emb_model = sorted(per_window_embeddings)[0]
        windows = list(per_window_embeddings[emb_model])
        built: dict[tuple[str, str], np.ndarray] = {}
        for key, segs in segs_by_key.items():
            mean_emb = _mean_window_embedding_over_segments(segs, windows)
            if mean_emb is not None and np.asarray(mean_emb).size:
                built[key] = np.asarray(mean_emb, dtype=np.float64)
        # Only offer centroids when *every* label has one. A partial set would let the centroid
        # matcher assign some labels and abstain on others, and the abstentions would then read as
        # "one matcher ran" rather than "this label had no embedding support".
        if built and len(built) == len(segs_by_key):
            centroids = built

    return harmonize_speaker_labels(segments_by_model, centroids=centroids, min_similarity=min_similarity)


# ── H3: a common word space across ASR models ────────────────────────────────


def _normalise_token(text: str) -> str:
    """Casefold and strip punctuation, for deciding *agreement* only.

    Models differ in casing and punctuation convention, and those differences are not
    transcription disputes. Normalisation therefore decides whether two readings agree; it never
    replaces what a model actually said, which stays on the slot.
    """
    return "".join(ch for ch in str(text).casefold() if ch.isalnum() or ch == "'")


@dataclass
class TranscriptSlot:
    """One position in the harmonised word space.

    Attributes:
        start_s: Earliest start among the models that filled this slot.
        end_s: Latest end among them.
        words: ``{model → surface form}``; a model absent from the slot maps to ``None``.
        consensus: The majority reading, or ``None`` when no reading holds a strict majority —
            a two-way tie has no winner, and publishing either would manufacture agreement.
        disagreement: ``1 − (largest agreeing share)`` over the models that filled the slot.
    """

    start_s: float
    end_s: float
    words: dict[str, Optional[str]]
    consensus: Optional[str]
    disagreement: float


@dataclass
class TranscriptHarmonization:
    """The harmonised lattice plus the two rates H3 exists to expose.

    Attributes:
        slots: Word positions in time order.
        gap_rate: ``{model → fraction of slots this model left empty}``.
        insertion_rate: ``{model → fraction of this model's words no other model produced}``.
        reference: The model whose token sequence anchored the alignment.
    """

    slots: list[TranscriptSlot]
    gap_rate: dict[str, float]
    insertion_rate: dict[str, float]
    reference: Optional[str]


def _align_pair(a: Sequence[str], b: Sequence[str]) -> list[tuple[Optional[int], Optional[int]]]:
    """Levenshtein alignment path between two token sequences.

    Returns ``(i, j)`` pairs where either side may be ``None`` for a gap. Needed rather than the
    plain distance because H3's whole purpose is *which* positions correspond: a distance says a
    model missed a word, an alignment says which one and leaves the rest lined up.
    """
    n, m = len(a), len(b)
    cost = np.zeros((n + 1, m + 1), dtype=np.int64)
    cost[:, 0] = np.arange(n + 1)
    cost[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = cost[i - 1, j - 1] + (0 if a[i - 1] == b[j - 1] else 1)
            cost[i, j] = min(sub, cost[i - 1, j] + 1, cost[i, j - 1] + 1)

    path: list[tuple[Optional[int], Optional[int]]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and cost[i, j] == cost[i - 1, j - 1] + (0 if a[i - 1] == b[j - 1] else 1):
            path.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and cost[i, j] == cost[i - 1, j] + 1:
            path.append((i - 1, None))
            i -= 1
        else:
            path.append((None, j - 1))
            j -= 1
    path.reverse()
    return path


def harmonize_transcripts(
    by_model: Mapping[str, Sequence[tuple[float, float, str]]],
) -> TranscriptHarmonization:
    """H3: align several ASR transcripts to each other, not only to audio.

    Each transcript arrives already aligned to the audio, independently. That is not enough to
    compare them: a model that inserts or drops one word shifts every timestamp after it, so a
    time-based comparison turns a single miss into a whole tail of apparent substitutions. Aligning
    the token sequences to each other keeps the rest lined up and makes "these models produced
    different words for the same position" expressible — which a per-window WER cannot say.

    Alignment is **star-shaped**: every model is aligned to one reference, and the reference's
    positions plus each model's insertions form the slots. The reference is the model whose token
    count is the median, so an outlier transcript (a hallucinated run, a truncated decode) does not
    become the frame everything else is measured against. This is an approximation — a full
    multiple-sequence alignment would not privilege any model — and the reference is reported so a
    consumer can see which one was privileged.

    Args:
        by_model: ``{model → [(start_s, end_s, text), ...]}`` in time order.

    Returns:
        A :class:`TranscriptHarmonization`. Gap and insertion rates are per model; slot
        disagreement is ``1 − largest agreeing share`` among the models that filled it.
    """
    models = sorted(m for m, w in (by_model or {}).items() if w)
    if not models:
        return TranscriptHarmonization(slots=[], gap_rate={}, insertion_rate={}, reference=None)

    words = {m: [(float(s), float(e), str(t)) for s, e, t in by_model[m]] for m in models}
    tokens = {m: [_normalise_token(t) for _, _, t in words[m]] for m in models}

    # Median token count, so neither the longest nor the shortest transcript anchors the space.
    ordered = sorted(models, key=lambda m: len(tokens[m]))
    reference = ordered[len(ordered) // 2]

    # slot key -> {model: word index}. Reference positions are integers; a run of insertions
    # between reference positions r-1 and r is keyed (r, k) so it sorts into place.
    slot_members: dict[tuple[float, float], dict[str, int]] = {}
    for ref_pos in range(len(tokens[reference])):
        slot_members[(float(ref_pos), 0.0)] = {reference: ref_pos}

    for model in models:
        if model == reference:
            continue
        pending = 0
        last_ref = -1
        for r_idx, m_idx in _align_pair(tokens[reference], tokens[model]):
            if r_idx is not None and m_idx is not None:
                slot_members.setdefault((float(r_idx), 0.0), {})[model] = m_idx
                last_ref, pending = r_idx, 0
            elif r_idx is not None:
                last_ref, pending = r_idx, 0  # reference-only position: a gap for this model
            elif m_idx is not None:
                # Model-only position: an insertion, keyed between the surrounding reference
                # positions so a run of them sorts into place rather than collapsing onto one slot.
                pending += 1
                slot_members.setdefault((float(last_ref) + 0.5, float(pending)), {})[model] = m_idx

    slots: list[TranscriptSlot] = []
    for key in sorted(slot_members):
        members = slot_members[key]
        surfaces = {m: words[m][i][2] for m, i in members.items()}
        spans = [(words[m][i][0], words[m][i][1]) for m, i in members.items()]
        counts: dict[str, int] = {}
        for m, i in members.items():
            counts[tokens[m][i]] = counts.get(tokens[m][i], 0) + 1
        top = max(counts.values())
        winners = [tok for tok, c in counts.items() if c == top]
        # A strict majority only: with two models reading differently there is no winner, and
        # publishing either would manufacture agreement that was never observed.
        consensus_token = winners[0] if len(winners) == 1 else None
        consensus = None
        if consensus_token is not None:
            consensus = next(surfaces[m] for m, i in members.items() if tokens[m][i] == consensus_token)
        slots.append(
            TranscriptSlot(
                start_s=min(s for s, _ in spans),
                end_s=max(e for _, e in spans),
                words={m: surfaces.get(m) for m in models},
                consensus=consensus,
                disagreement=float(1.0 - top / len(members)) if members else 0.0,
            )
        )

    n_slots = len(slots) or 1
    gap_rate = {m: sum(1 for s in slots if s.words.get(m) is None) / n_slots for m in models}
    insertion_rate = {}
    for m in models:
        produced = len(words[m]) or 1
        alone = sum(
            1 for s in slots if s.words.get(m) is not None and sum(v is not None for v in s.words.values()) == 1
        )
        insertion_rate[m] = alone / produced if len(models) > 1 else 0.0
    return TranscriptHarmonization(slots=slots, gap_rate=gap_rate, insertion_rate=insertion_rate, reference=reference)
