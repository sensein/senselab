"""H2: the common speaker space, and the uncertainty of constructing it.

Every diarizer names its speakers arbitrarily — ``SPEAKER_00``, ``spk0`` — and the names carry no
meaning across models. So any cross-model statement about identity first *guesses* that two labels
denote the same person. Treated as fact, that guess makes two models which were never correctly
compared read as disagreeing, and speaker uncertainty then stays high in exactly the regions where
per-speaker presence is unambiguous. That is the observation this module exists to address.

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
    "SpeakerHarmonization",
    "centroid_assignment",
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
