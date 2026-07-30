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

__all__ = [
    "PerSpeakerPresenceTrack",
    "SourceCountClaim",
    "SourceLabelCorrespondence",
    "SpeakerCountPosterior",
    "SpeakerHypothesis",
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
