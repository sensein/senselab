"""Revision attribution and the self-confirmation guard (T081, FR-011d / FR-011g).

Every state change in a mutually-influencing loop is attributable: which signal caused it,
in which round, at what weight, on what evidence. Without that, a final speaker label rests
on a chain nobody can retrace (FR-011g).

The subtler job is :func:`classify_resolution`. In a loop where signals revise one another,
uncertainty can fall for two completely different reasons:

- **New evidence arrived** — the analysis genuinely learned something.
- **The value was overwritten** — and uncertainty was then recomputed *from the overwritten
  value*, so it fell because of the edit, not because of evidence.

Both look identical in the number alone. A loop that cannot distinguish them converges on
its own edits and reports high confidence in them, which is the single largest correctness
risk in the mutual-influence design. So the distinction is structural: every revision
carries a :data:`RESOLUTION_KINDS` tag, and only ``new_evidence`` may be reported as a
confidence gain.

This generalizes a distinction the loop already made — the existing convergence machinery
separates *explained* from *improved* outcomes — rather than inventing one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "RESOLUTION_KINDS",
    "ResolutionKind",
    "RevisionRecord",
    "classify_resolution",
    "revision_log_entry",
]

ResolutionKind = Literal["new_evidence", "revision", "unresolved"]

RESOLUTION_KINDS: tuple[ResolutionKind, ...] = ("new_evidence", "revision", "unresolved")
"""How a quantity's uncertainty came to change.

``new_evidence`` — fell because independent evidence arrived. The only kind that may be
reported as improved confidence.
``revision`` — fell because the value was overwritten. **Not** a confidence gain
(FR-011d, SC-027).
``unresolved`` — did not fall, or rose.
"""

_MIN_IMPROVEMENT = 1e-9
"""Below this, a change in uncertainty is float noise rather than a resolution."""


@dataclass(frozen=True)
class RevisionRecord:
    """One attributable state change made by an influence path."""

    round: int
    quantity: str
    before: Any
    after: Any
    caused_by: str
    effective_weight: float
    resolution_kind: ResolutionKind
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject an unattributed or mis-tagged revision at construction.

        Both failures are silent if allowed through: an unattributed change breaks the
        audit chain, and an unrecognized kind would be treated as neither a confidence gain
        nor a revision by downstream checks.
        """
        if not str(self.caused_by).strip():
            raise ValueError(
                f"revision of {self.quantity!r} has no caused_by — every state change must name the signal "
                "that produced it (FR-011g)"
            )
        if self.resolution_kind not in RESOLUTION_KINDS:
            raise ValueError(
                f"unknown resolution_kind {self.resolution_kind!r}; expected one of {RESOLUTION_KINDS}"
            )

    def improves_confidence(self) -> bool:
        """Whether this change may be reported as a confidence gain (FR-011d)."""
        return self.resolution_kind == "new_evidence"

    def to_json(self) -> dict[str, Any]:
        """Serialize with a fixed key order, so output stays byte-stable (FR-011f)."""
        return {
            "round": self.round,
            "quantity": self.quantity,
            "before": self.before,
            "after": self.after,
            "caused_by": self.caused_by,
            "effective_weight": self.effective_weight,
            "resolution_kind": self.resolution_kind,
            "evidence": dict(sorted(self.evidence.items())),
        }


def classify_resolution(
    *,
    before_uncertainty: float,
    after_uncertainty: float,
    was_revised: bool,
    independent_evidence: bool,
) -> ResolutionKind:
    """Classify why a quantity's uncertainty changed.

    A drop counts as ``new_evidence`` only when independent evidence supports it. A drop
    that follows a revision with no independent support is ``revision`` — the loop
    confirming its own edit — and must not be reported as improved confidence.

    Note that a revision *with* independent corroboration is genuine improvement: the guard
    targets unsupported self-confirmation, not revision as such.

    Args:
        before_uncertainty: Uncertainty before the round.
        after_uncertainty: Uncertainty after the round.
        was_revised: Whether an influence path overwrote the value.
        independent_evidence: Whether evidence from an independent source supports it.

    Returns:
        The resolution kind.
    """
    dropped = (float(before_uncertainty) - float(after_uncertainty)) > _MIN_IMPROVEMENT
    if not dropped:
        return "unresolved"
    if independent_evidence:
        return "new_evidence"
    if was_revised:
        return "revision"
    return "new_evidence"


def revision_log_entry(record: RevisionRecord) -> dict[str, Any]:
    """Return a revision as a log entry with deterministic key order."""
    return record.to_json()
