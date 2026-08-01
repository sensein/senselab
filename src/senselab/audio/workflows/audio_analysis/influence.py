"""Uncertainty-gated influence weighting (T080, FR-011b / FR-011c).

Signals in the adaptive loop influence one another iteratively toward convergence. Two
independent gates bound how far any one signal may move another:

    effective_weight = base_weight × uncertainty_gate × derivation_gate

**The uncertainty gate** shrinks a signal's influence as its own uncertainty rises, so a
signal that does not trust itself cannot propagate its error into signals that do. It is
floored rather than taken to zero: when stability is measured over few perturbation points
the measure is coarse — with two points, normalised entropy can only be 0 or 1 — and a hard
zero would erase a dissenting claim from the posterior entirely rather than down-weighting
it. A maximally-uncertain source is left visible and unable to win.

**The derivation gate** shrinks it further for signals whose labels are a by-product of
another signal already in the system. This is the subtler of the two. A clustering-derived
pseudo-diarizer agreeing with the embeddings it was computed from is *not* corroboration —
it is one computation counted twice, and treating it as two independent votes is how a
single derived signal comes to look like consensus. The gate is required to sit strictly
below the independent gate; a configuration that equalizes them defeats the guard and is
rejected rather than honored.

Deliberately pure and dependency-free so the guards can be tested without the loop, and so
they can be sequenced *before* the influence paths they protect (spec Dependencies).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

__all__ = [
    "SOURCE_KINDS",
    "InfluenceWeight",
    "effective_weight",
    "ordered_signals",
    "resolve_influence",
]

SourceKind = Literal["independent", "derived"]

SOURCE_KINDS: tuple[SourceKind, ...] = ("independent", "derived")
"""``independent`` observes the quantity directly; ``derived`` is a by-product of another
signal already in the system (FR-007)."""


@dataclass(frozen=True)
class InfluenceWeight:
    """How much one signal may move another, with both gates recorded separately.

    Keeping the gates separate rather than only their product is what lets an audit answer
    *why* a weight was small — an unreliable observer and a derived one are different
    findings with different remedies.
    """

    signal: str
    base_weight: float
    uncertainty_gate: float
    derivation_gate: float
    effective_weight: float
    kind: SourceKind

    def to_json(self) -> dict[str, Any]:
        """Serialize per ``contracts/speaker-speaker.md``."""
        return {
            "signal": self.signal,
            "kind": self.kind,
            "base_weight": self.base_weight,
            "uncertainty_gate": self.uncertainty_gate,
            "derivation_gate": self.derivation_gate,
            "effective_weight": self.effective_weight,
        }


def effective_weight(
    base_weight: float,
    *,
    uncertainty: float,
    derivation_gate: float,
    exponent: float = 1.0,
    min_gate: float = 0.05,
) -> float:
    """Apply both gates to a base weight.

    Args:
        base_weight: Configured weight for this signal-to-target edge.
        uncertainty: The signal's own uncertainty, in ``[0, 1]``.
        derivation_gate: Multiplier for the signal's source kind.
        exponent: Sharpness of the uncertainty gate. Higher punishes uncertainty harder
            without moving the endpoints.
        min_gate: Floor on the uncertainty gate, so a maximally-uncertain source is heavily
            attenuated rather than erased. This matters when stability is measured over few
            perturbation points: with two, normalised entropy can only be 0 or 1, so any
            source that disagrees with itself would otherwise carry exactly zero weight and
            its claim would vanish from the posterior entirely. Observed on a real run,
            where a clusterer's count was silenced rather than down-weighted. The floor
            keeps a dissenting claim visible while leaving it unable to win.

    Returns:
        The weight actually applied.

    Raises:
        ValueError: If ``uncertainty`` is outside ``[0, 1]`` — an out-of-range value would
            silently produce a negative or inflated weight.
    """
    u = float(uncertainty)
    if not 0.0 <= u <= 1.0:
        raise ValueError(f"signal uncertainty must be in [0, 1]; got {u}")
    gate = max(float(min_gate), (1.0 - u) ** float(exponent))
    return float(base_weight) * gate * float(derivation_gate)


def resolve_influence(
    signal: str,
    *,
    base_weight: float,
    uncertainty: float,
    kind: str,
    gates: Mapping[str, float],
    exponent: float = 1.0,
    min_gate: float = 0.05,
) -> InfluenceWeight:
    """Resolve one signal's influence, validating the gate configuration.

    Args:
        signal: Signal name.
        base_weight: Configured weight.
        uncertainty: The signal's own uncertainty.
        kind: ``"independent"`` or ``"derived"`` (FR-007).
        gates: Per-kind gate values; ``derived`` must sit strictly below ``independent``.
        exponent: Uncertainty-gate sharpness.
        min_gate: Floor on the uncertainty gate; see :func:`effective_weight`.

    Returns:
        The resolved :class:`InfluenceWeight`.

    Raises:
        ValueError: If ``kind`` is unrecognized, or if the gate configuration does not
            place ``derived`` strictly below ``independent`` (FR-011c) — equalizing them
            would let a derived signal count as a peer, which is the failure the gate
            exists to prevent.
    """
    if kind not in SOURCE_KINDS:
        raise ValueError(f"unknown source kind {kind!r}; expected one of {SOURCE_KINDS}")
    independent = float(gates.get("independent", 1.0))
    derived = float(gates.get("derived", 0.4))
    if not derived < independent:
        raise ValueError(
            f"derivation_gate.derived ({derived}) must be strictly below independent ({independent}); "
            "equalizing them lets a derived signal count as an independent peer (FR-011c)"
        )
    gate = independent if kind == "independent" else derived
    return InfluenceWeight(
        signal=signal,
        base_weight=float(base_weight),
        uncertainty_gate=max(min_gate, (1.0 - float(uncertainty)) ** float(exponent)),
        derivation_gate=gate,
        effective_weight=effective_weight(
            base_weight, uncertainty=uncertainty, derivation_gate=gate, exponent=exponent, min_gate=min_gate
        ),
        kind=kind,  # type: ignore[arg-type]
    )


def ordered_signals(signals: Mapping[str, Any]) -> list[str]:
    """Return signal names in a fixed order.

    Iteration order must not depend on dict insertion order: mutual influence applied in a
    different sequence can reach a different fixed point, which would break the
    byte-reproducibility the convergence outputs already provide (FR-011f, SC-029).
    """
    return sorted(signals)
