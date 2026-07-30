"""Per-speaker identity uncertainty (T095-T097, FR-001 to FR-011).

The identity axis reports one uncertainty value per time bucket, answering "was it the same
speaker?" That scalar cannot express *how many* people the analysis thinks are present, and
the distinction matters: on a validation recording two diarizers each reported one speaker
for the whole clip while embedding clustering reported five distinct regions aligned to name
boundaries. The axis correctly registered high uncertainty, but a consumer reading 0.67
cannot tell "we disagree about who spoke" from "we disagree about whether this is one person
or four" — different problems with different fixes.

So identity becomes **per speaker**: a distribution over how many speakers are present, one
hypothesis per speaker with its own existence uncertainty, and a presence track per
hypothesis.

Two design commitments:

**Multi-modal disagreement is representable.** A mean or a majority vote would have reported
"one speaker, slightly uncertain" for the case above, which is precisely the wrong summary.
The posterior keeps the competing counts and names which sources backed each.

**Weight comes from the influence gates, not from counting heads.** A clustering-derived
pseudo-diarizer agreeing with the embeddings it was computed from is one computation counted
twice. Reusing ``influence.resolve_influence`` means a derived voter is attenuated by the
same rule everywhere in the loop rather than by a special case here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

from senselab.audio.workflows.audio_analysis.adaptive.influence import SOURCE_KINDS, resolve_influence
from senselab.audio.workflows.audio_analysis.identity import (
    cluster_active_time,
    label_correspondence_rows,
    per_speaker_tracks,
)

__all__ = [
    "PerSpeakerPresenceTrack",
    "PerturbationEvidence",
    "build_presence_tracks",
    "build_speaker_identity",
    "evidence_from_passes",
    "claims_from_perturbations",
    "perturbation_uncertainty",
    "SourceCountClaim",
    "SourceLabelCorrespondence",
    "SpeakerCountPosterior",
    "SpeakerHypothesis",
    "source_kind_for",
    "speaker_count_posterior",
]

SourceKind = Literal["independent", "derived"]


@dataclass(frozen=True)
class SourceCountClaim:
    """One source's claim about how many speakers are present."""

    source: str
    count: int
    uncertainty: float = 0.0
    kind: SourceKind = "independent"


@dataclass(frozen=True)
class SpeakerCountPosterior:
    """Belief over the number of speakers, with per-count attribution.

    Attributes:
        probabilities: count → probability, summing to 1.
        support: count → the sources that claimed it (FR-006).
        modal_count: highest-probability count.
        is_multimodal: whether more than one count clears the policy threshold.
        weights: source → the effective weight its claim carried.
    """

    probabilities: dict[int, float]
    support: dict[int, list[str]]
    modal_count: int
    is_multimodal: bool
    weights: dict[str, float] = field(default_factory=dict)
    converged: bool = False

    def to_json(self) -> dict[str, Any]:
        """Serialize per ``contracts/speaker-identity.md`` (JSON keys are strings)."""
        return {
            "probabilities": {str(k): round(v, 6) for k, v in sorted(self.probabilities.items())},
            "support": {str(k): sorted(v) for k, v in sorted(self.support.items())},
            "modal_count": self.modal_count,
            "is_multimodal": self.is_multimodal,
            "weights": {k: round(v, 6) for k, v in sorted(self.weights.items())},
            "converged": self.converged,
        }


def speaker_count_posterior(
    claims: Sequence[SourceCountClaim],
    *,
    gates: Mapping[str, float] | None = None,
    multimodal_threshold: float = 0.15,
) -> SpeakerCountPosterior:
    """Build the speaker-count posterior from per-source claims.

    Each source's claim carries the weight the influence gates give it: attenuated by its own
    uncertainty, and further attenuated if its labels are derived from another signal already
    in the system. Counting heads instead would let a derived voter carry the same authority
    as an independent observation of the audio.

    Args:
        claims: One claim per source.
        gates: Derivation gates; ``derived`` must sit strictly below ``independent``.
        multimodal_threshold: Probability above which a count counts as a supported mode.

    Returns:
        The posterior. Empty claims yield all mass on zero speakers — the honest reading of
        "no source reported anybody".
    """
    resolved_gates = dict(gates or {"independent": 1.0, "derived": 0.4})
    if not claims:
        return SpeakerCountPosterior(probabilities={0: 1.0}, support={0: []}, modal_count=0, is_multimodal=False)

    weights: dict[str, float] = {}
    mass: dict[int, float] = {}
    support: dict[int, list[str]] = {}
    for claim in sorted(claims, key=lambda c: c.source):
        if claim.kind not in SOURCE_KINDS:
            raise ValueError(f"unknown source kind {claim.kind!r} for {claim.source!r}")
        weight = resolve_influence(
            claim.source,
            base_weight=1.0,
            uncertainty=claim.uncertainty,
            kind=claim.kind,
            gates=resolved_gates,
        ).effective_weight
        weights[claim.source] = weight
        count = int(claim.count)
        mass[count] = mass.get(count, 0.0) + weight
        support.setdefault(count, []).append(claim.source)

    total = sum(mass.values())
    if total <= 0.0:
        # Every source was fully uncertain. Reporting a count would invent one; the honest
        # answer is that the modal claim stands with no confidence behind it.
        counts = sorted(support)
        return SpeakerCountPosterior(
            probabilities={c: 1.0 / len(counts) for c in counts},
            support=support,
            modal_count=counts[0],
            is_multimodal=len(counts) > 1,
            weights=weights,
        )

    probabilities = {c: m / total for c, m in mass.items()}
    modal = max(probabilities, key=lambda c: (probabilities[c], -c))
    modes = [c for c, p in probabilities.items() if p >= multimodal_threshold]
    return SpeakerCountPosterior(
        probabilities=probabilities,
        support=support,
        modal_count=modal,
        is_multimodal=len(modes) > 1,
        weights=weights,
    )


@dataclass(frozen=True)
class SourceLabelCorrespondence:
    """How one source's own speaker label maps to a fused hypothesis (FR-005)."""

    source: str
    source_label: str
    speaker_id: str
    kind: SourceKind
    cluster_id: str | None = None
    confidence: float | None = None

    def to_json(self) -> dict[str, Any]:
        """Serialize for ``final/speakers.json``."""
        return {
            "source": self.source,
            "source_label": self.source_label,
            "speaker_id": self.speaker_id,
            "source_kind": self.kind,
            "cluster_id": self.cluster_id,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class PerSpeakerPresenceTrack:
    """One bucket of one speaker's presence belief (FR-003)."""

    speaker_id: str
    start: float
    end: float
    presence_confidence: float | None
    presence_uncertainty: float | None
    overlap_with: list[str] = field(default_factory=list)
    contributing_sources: list[str] = field(default_factory=list)
    round: int = 0
    resolution_kind: str = "unresolved"

    def to_row(self) -> dict[str, Any]:
        """Row for ``final/per_speaker_presence.parquet``."""
        return {
            "speaker_id": self.speaker_id,
            "start": self.start,
            "end": self.end,
            "presence_confidence": self.presence_confidence,
            "presence_uncertainty": self.presence_uncertainty,
            "overlap_with": list(self.overlap_with),
            "contributing_sources": list(self.contributing_sources),
            "round": self.round,
            "resolution_kind": self.resolution_kind,
        }


@dataclass(frozen=True)
class SpeakerHypothesis:
    """One person the analysis believes is present.

    ``existence_uncertainty`` and per-bucket ``presence_uncertainty`` are deliberately
    separate (FR-004): "this speaker might not exist" and "this speaker exists but we are
    unsure where they spoke" call for different follow-up, and one number cannot say which
    is meant.
    """

    speaker_id: str
    existence_uncertainty: float
    supporting_sources: list[str]
    source_kinds: dict[str, SourceKind]
    first_seen: float | None = None
    last_seen: float | None = None
    total_active_s: float = 0.0
    converged: bool = False
    revisions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def has_independent_support(self) -> bool:
        """Whether any source proposing this speaker observes identity directly.

        A hypothesis resting only on derived sources is not thereby wrong, but a consumer
        must be able to see that it has no independent observation behind it.
        """
        return any(kind == "independent" for kind in self.source_kinds.values())

    def to_json(self) -> dict[str, Any]:
        """Serialize per ``contracts/speaker-identity.md``."""
        return {
            "speaker_id": self.speaker_id,
            "existence_uncertainty": round(self.existence_uncertainty, 6),
            "supporting_sources": sorted(self.supporting_sources),
            "source_kinds": dict(sorted(self.source_kinds.items())),
            "has_independent_support": self.has_independent_support,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "total_active_s": round(self.total_active_s, 6),
            "converged": self.converged,
            "revisions": list(self.revisions),
        }


def source_kind_for(source: str, policy: Mapping[str, Any] | None = None) -> SourceKind:
    """Resolve whether a source is an independent observer or derived (FR-007).

    Read from policy rather than hardcoded, because the classification is a judgement about
    **pipeline wiring**, not an intrinsic property of a model. The same clustering component
    would be independent in a pipeline that did not also use its embeddings to harmonise
    other sources' labels.

    The live example, recorded so the decision stays arguable: ``embedding_silhouette`` is
    marked derived because it seeds the cross-model label harmonisation — other diarizers'
    labels snap to its centroids — and the same embeddings drive same-label and change-point
    validation, so that evidence already enters the identity axis three ways. Against that:
    it runs an embedding model on the audio and clusters the result, which is a direct
    observation; and on one validation recording it reported five speakers where two
    "independent" diarizers reported one, with re-examination suggesting it was the closer
    answer. Down-weighting it may therefore suppress correct results. The gate is
    configurable precisely because that tension is unresolved and needs ground truth.

    Args:
        source: Source identifier. Matched on the prefix before ``/`` so versioned ids like
            ``embedding_silhouette/<model>`` resolve to the same kind.
        policy: Adaptive policy; the ``influence.source_kinds`` mapping is consulted.

    Returns:
        ``"independent"`` or ``"derived"``.
    """
    influence = dict((policy or {}).get("influence") or {})
    declared = dict(influence.get("source_kinds") or {})
    default: SourceKind = influence.get("default_source_kind", "independent")  # type: ignore[assignment]
    base = source.split("/", 1)[0]
    kind = declared.get(source, declared.get(base, default))
    if kind not in SOURCE_KINDS:
        raise ValueError(f"policy declares unknown source kind {kind!r} for {source!r}")
    return kind  # type: ignore[return-value]


# ── perturbation-derived reliability ───────────────────────────────────
#
# Preferred over declaring a source independent or derived by hand. A hand-set gate
# encodes a judgement about pipeline wiring; a perturbation measures what the source
# actually does. A source that returns the same answer when the input is perturbed has
# earned confidence in that answer; one that flips under mild preprocessing has not,
# whatever we would call its provenance.
#
# The pipeline already generates this evidence and was not using it that way:
#
#   - **preprocessing** -- the raw and enhanced passes are the same recording under a
#     transform, so any per-pass answer is a two-point perturbation sample already;
#   - **gain** -- the amplitude probe sweeps level, and level demonstrably moves
#     classifier output;
#   - **band limiting, cropping, additive noise at known SNR** -- further points on the
#     same axis, cheap to add because they need no new model.
#
# This matters most exactly where hand-assignment is least trustworthy. A recording
# that is genuinely hard for diarizers is one where off-the-shelf models disagree
# *with themselves* across preprocessing -- and that instability is the evidence, not
# a nuisance to be smoothed over by a constant.


@dataclass(frozen=True)
class PerturbationEvidence:
    """One source's answers to the same question under several perturbations.

    Attributes:
        source: Source identifier.
        answers: perturbation id → the answer that source gave under it. Perturbation ids
            are free-form (``"raw"``, ``"enhanced"``, ``"gain+6dB"``, ...) so any transform
            the pipeline already performs can contribute a point.
    """

    source: str
    answers: dict[str, Any]

    @property
    def n(self) -> int:
        """Number of perturbation points observed."""
        return len(self.answers)


def perturbation_uncertainty(evidence: PerturbationEvidence) -> float | None:
    """Uncertainty in ``[0, 1]`` from how much a source's answer moves under perturbation.

    Normalized Shannon entropy over the distribution of answers — the same collapse the
    presence axis already uses — so unanimity is 0 and a maximally split source is 1.
    Entropy rather than a modal fraction because *how* the disagreement is spread matters:
    two answers split evenly is a different state from one answer plus scattered outliers.

    Args:
        evidence: The source's answers across perturbations.

    Returns:
        Uncertainty in ``[0, 1]``, or ``None`` with fewer than two perturbation points —
        a single observation carries no evidence about stability, and reporting 0 there
        would award full confidence for having been asked once.
    """
    import math as _math

    if evidence.n < 2:
        return None
    counts: dict[str, int] = {}
    for answer in evidence.answers.values():
        counts[repr(answer)] = counts.get(repr(answer), 0) + 1
    if len(counts) == 1:
        return 0.0
    total = sum(counts.values())
    entropy = -sum((c / total) * _math.log(c / total) for c in counts.values())
    return min(1.0, entropy / _math.log(len(counts)))


def claims_from_perturbations(
    evidence: Sequence[PerturbationEvidence],
    *,
    policy: Mapping[str, Any] | None = None,
    fallback_uncertainty: float = 0.5,
) -> list[SourceCountClaim]:
    """Build count claims whose uncertainty is *measured* rather than assigned.

    Each source contributes its modal answer, weighted by how stable that answer was under
    perturbation. This replaces a hand-set reliability judgement with evidence the pipeline
    already produces.

    The source kind is still resolved from policy and still gates the claim, but it becomes
    the secondary term: a source that is demonstrably stable is not held down by a label,
    and a source that flips under preprocessing is attenuated whatever its label says.

    Args:
        evidence: Per-source answers across perturbations.
        policy: Adaptive policy, for :func:`source_kind_for`.
        fallback_uncertainty: Used when a source offers fewer than two perturbation points,
            so a single-observation source is neither trusted nor discarded outright.

    Returns:
        One claim per source, ready for :func:`speaker_count_posterior`.
    """
    claims: list[SourceCountClaim] = []
    for item in sorted(evidence, key=lambda e: e.source):
        if not item.answers:
            continue
        counts: dict[int, int] = {}
        for answer in item.answers.values():
            counts[int(answer)] = counts.get(int(answer), 0) + 1
        modal = max(counts, key=lambda c: (counts[c], -c))
        measured = perturbation_uncertainty(item)
        claims.append(
            SourceCountClaim(
                source=item.source,
                count=modal,
                uncertainty=fallback_uncertainty if measured is None else measured,
                kind=source_kind_for(item.source, policy),
            )
        )
    return claims


# ── deriving the posterior from a run's passes ──────────────────────────


def _distinct_speakers(outcome: Mapping[str, Any]) -> int | None:
    """Count distinct speaker labels in one diarization outcome."""
    if not isinstance(outcome, Mapping) or outcome.get("status") != "ok":
        return None
    result = outcome.get("result")
    if not isinstance(result, list):
        return None
    labels: set[str] = set()
    for item in result:
        for seg in item if isinstance(item, list) else [item]:
            spk = getattr(seg, "speaker", None)
            if spk is None and isinstance(seg, Mapping):
                spk = seg.get("speaker")
            if spk is not None:
                labels.add(str(spk))
    return len(labels)


def evidence_from_passes(passes: Mapping[str, Any]) -> list[PerturbationEvidence]:
    """Turn a run's per-pass diarization into per-source perturbation evidence.

    The raw and enhanced passes are the same recording under a transform, so each
    diarizer's two answers are already a stability sample — no extra inference needed. A
    diarizer that reports a different speaker count before and after enhancement is telling
    us its answer is not robust on this recording, and that is exactly the signal that
    should govern how far it moves the posterior.

    Args:
        passes: ``summary["passes"]`` — pass label → pass summary.

    Returns:
        One :class:`PerturbationEvidence` per diarization source, keyed by pass label.
    """
    by_source: dict[str, dict[str, Any]] = {}
    for label, summary in sorted(passes.items()):
        if not isinstance(summary, Mapping):
            continue
        for model, outcome in ((summary.get("diarization") or {}).get("by_model") or {}).items():
            count = _distinct_speakers(outcome)
            if count is not None:
                by_source.setdefault(str(model), {})[str(label)] = count
    return [PerturbationEvidence(source=src, answers=answers) for src, answers in sorted(by_source.items())]


def build_speaker_identity(
    passes: Mapping[str, Any],
    *,
    identity_votes: Sequence[Mapping[str, Any]] | None = None,
    policy: Mapping[str, Any] | None = None,
    gates: Mapping[str, float] | None = None,
) -> tuple[SpeakerCountPosterior, list[SpeakerHypothesis], list[SourceLabelCorrespondence]]:
    """Derive the count posterior and speaker hypotheses from a completed run.

    The count comes from the passes; the harvested per-bucket identity evidence, when
    supplied, says *which* speaker each claim is about — when they were active, whose label
    each diarizer's own naming maps to, and how confident their existence individually is.

    Args:
        passes: ``summary["passes"]``.
        identity_votes: Bucket dicts from ``harvest_identity_votes``. Optional: the
            posterior stands on the passes alone, and votes only add per-speaker detail.
        policy: Adaptive policy, for source-kind resolution.
        gates: Derivation gates.

    Returns:
        ``(posterior, hypotheses, correspondence)``. With no diarization at all the
        posterior places its mass on zero speakers rather than inventing one.
    """
    evidence = evidence_from_passes(passes)
    claims = claims_from_perturbations(evidence, policy=policy)
    posterior = speaker_count_posterior(claims, gates=gates)

    modal = posterior.modal_count
    supporters = posterior.support.get(modal, [])
    kinds = {s: source_kind_for(s, policy) for s in supporters}

    # Rank the observed clusters by how long each was active, so which cluster becomes S0 is
    # a property of the evidence rather than of iteration order. A cluster the posterior
    # does not back still gets a hypothesis: it is contested evidence, and truncating to the
    # modal count would delete the record that some source separated more speakers than the
    # posterior believes.
    ranked = list(cluster_active_time(identity_votes or []))
    speaker_ids = {cluster: f"S{i}" for i, cluster in enumerate(ranked)}
    tracks = per_speaker_tracks(identity_votes or [], speaker_ids=speaker_ids)
    spans: dict[str, tuple[float, float, float]] = {}
    for row in tracks:
        sid = str(row["speaker_id"])
        start, end = float(row["start"]), float(row["end"])
        first, last, total = spans.get(sid, (start, end, 0.0))
        spans[sid] = (min(first, start), max(last, end), total + (end - start))

    # Existence uncertainty is per speaker, not shared: the i-th speaker exists only if the
    # count is at least i+1, so its doubt is the mass on counts below that. The first
    # speaker in a run where every source heard someone is near-certain even when the
    # sources disagree about how many there are in total; the last one carries that whole
    # disagreement. A single off-modal scalar would report identical doubt for both and
    # leave a consumer no way to know which speaker to go looking for (FR-004).
    def _doubt(index: int) -> float:
        at_least = sum(p for c, p in posterior.probabilities.items() if c >= index + 1)
        return max(0.0, min(1.0, 1.0 - at_least))

    hypotheses = [
        SpeakerHypothesis(
            speaker_id=f"S{i}",
            existence_uncertainty=_doubt(i),
            supporting_sources=list(supporters),
            source_kinds=dict(kinds),
            first_seen=spans[f"S{i}"][0] if f"S{i}" in spans else None,
            last_seen=spans[f"S{i}"][1] if f"S{i}" in spans else None,
            total_active_s=spans.get(f"S{i}", (0.0, 0.0, 0.0))[2],
            converged=not posterior.is_multimodal,
        )
        for i in range(max(modal, len(ranked)))
    ]

    if identity_votes:
        correspondence = [
            SourceLabelCorrespondence(
                source=str(row["source"]),
                source_label=str(row["source_label"]),
                speaker_id=str(row["speaker_id"]),
                kind=kinds.get(str(row["source"]), source_kind_for(str(row["source"]), policy)),
                cluster_id=str(row["cluster_id"]),
            )
            for row in label_correspondence_rows(identity_votes, speaker_ids=speaker_ids)
        ]
    else:
        # No harvested labels to point at. The placeholder records which count each source
        # claimed, which is the only correspondence the passes alone can support.
        correspondence = [
            SourceLabelCorrespondence(
                source=src,
                source_label=f"<{src}:count={c}>",
                speaker_id=f"S{min(i, max(modal - 1, 0))}",
                kind=kinds.get(src, source_kind_for(src, policy)),
            )
            for i, (c, srcs) in enumerate(sorted(posterior.support.items()))
            for src in srcs
        ]
    return posterior, hypotheses, correspondence


def build_presence_tracks(
    identity_votes: Sequence[Mapping[str, Any]],
    *,
    round_index: int = 0,
    resolution_kind: str = "unresolved",
) -> list[PerSpeakerPresenceTrack]:
    """Per-speaker presence rows for ``final/per_speaker_presence.parquet``.

    Speaker ids match :func:`build_speaker_identity` — both rank clusters by active time —
    so a hypothesis and its track refer to the same person.
    """
    speaker_ids = {cluster: f"S{i}" for i, cluster in enumerate(cluster_active_time(identity_votes))}
    return [
        PerSpeakerPresenceTrack(
            speaker_id=str(row["speaker_id"]),
            start=float(row["start"]),
            end=float(row["end"]),
            presence_confidence=float(row["presence_confidence"]),
            presence_uncertainty=float(row["presence_uncertainty"]),
            overlap_with=[speaker_ids.get(c, c) for c in row["overlap_with"]],
            contributing_sources=list(row["contributing_sources"]),
            round=round_index,
            resolution_kind=resolution_kind,
        )
        for row in per_speaker_tracks(identity_votes, speaker_ids=speaker_ids)
    ]
