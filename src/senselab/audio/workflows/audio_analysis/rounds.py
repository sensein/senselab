"""L2 iteration: regional trust and convergence.

Round 0 fuses the L1 signals as harvested, with one weight per signal for the whole recording.
Later rounds do two things round 0 cannot.

**Trust becomes regional.** A signal can be reliable in one stretch and not another, and a
global weight cannot express that. Once the mask says a region is target-free, a diarizer still
placing a speaker there has made a claim *that region* does not support — so its vote is
discounted **there**, and nowhere else. Global down-weighting for a local failure is the exact
mistake that suppressed the source which turned out to be right about the five named speakers
on a 4.9 s recording; regional trust is how the same evidence attenuates the wrong claim
without silencing the right ones.

**The mask's own confidence gates how far it may act.** A mask unsure that a region is
target-free has not earned the right to discount a signal for speaking there. Without that
gate a guess about the mask becomes a verdict about a model — and since the mask is itself
refined across rounds, that error would compound rather than settle.

An ``indeterminate`` region withdraws nothing: "I cannot tell" is not grounds to disbelieve
anyone.

Convergence is deliberately conservative about what counts as *no change*: a bucket that goes
from unmeasured to measured is progress, and treating it as stability would stop the loop
exactly when it had started working.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

__all__ = [
    "MIN_REGIONAL_TRUST",
    "regional_weights",
    "round_converged",
]

MIN_REGIONAL_TRUST = 0.05
"""Floor on a regionally-withdrawn weight, so a signal is attenuated rather than erased. Same
reasoning as the global reliability and support floors: the dissenter may be the only source
that noticed something."""

# States that constitute evidence *against* a speaker claim. ``indeterminate`` is deliberately
# absent — an unresolved region is not counter-evidence.
_CONTRADICTING_STATES = ("target_free", "nontarget_active")


def _overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return not (a[1] <= b[0] or a[0] >= b[1])


def regional_weights(
    *,
    base_weights: Mapping[str, float],
    regions: Sequence[Mapping[str, Any]],
    claims: Mapping[str, Sequence[tuple[float, float]]],
    min_trust: float = MIN_REGIONAL_TRUST,
) -> dict[tuple[float, float], dict[str, float]]:
    """Per-region signal weights, withdrawing trust only where it was violated.

    Args:
        base_weights: ``{signal → global weight}`` from round 0.
        regions: Mask regions, each with ``start``, ``end``, ``state`` and ``confidence``.
        claims: ``{signal → [(start, end), ...]}`` spans where the signal asserted a speaker.
        min_trust: Floor on the withdrawn weight.

    Returns:
        ``{(region_start, region_end) → {signal → weight}}``. A signal that made no claim in a
        region keeps its global weight there — silence about a region is not a violation in it.
    """
    out: dict[tuple[float, float], dict[str, float]] = {}
    for region in regions:
        span = (float(region.get("start", 0.0)), float(region.get("end", 0.0)))
        state = str(region.get("state") or "indeterminate")
        # A mask that is unsure has not earned the right to act. Scaling the withdrawal by the
        # mask's confidence keeps a tentative mask from producing a confident verdict about a
        # model, which matters because the mask itself is still being refined.
        mask_confidence = max(0.0, min(1.0, float(region.get("confidence", 1.0))))
        per_signal: dict[str, float] = {}
        for signal, weight in sorted(base_weights.items()):
            base = float(weight)
            violated = state in _CONTRADICTING_STATES and any(
                _overlaps(span, (float(s), float(e))) for s, e in claims.get(signal, ())
            )
            if violated:
                per_signal[signal] = max(min_trust, base * (1.0 - mask_confidence))
            else:
                per_signal[signal] = base
        out[span] = per_signal
    return out


def round_converged(
    previous: Sequence[Mapping[str, Any]],
    current: Sequence[Mapping[str, Any]],
    *,
    tolerance: float = 1e-3,
    field: str = "uncertainty",
) -> bool:
    """Whether a round left the answer unchanged.

    Args:
        previous: Rows from the prior round.
        current: Rows from this round.
        tolerance: Largest per-bucket change still counted as no change.
        field: Which quantity to compare.

    Returns:
        ``True`` only when the two rounds cover the same buckets and every shared value moved
        by less than ``tolerance``. New coverage counts as a change even where shared buckets
        agree, and a bucket going from unmeasured to measured is a change — treating that as
        stability would stop the loop exactly when it had begun to work. Two ``None`` values
        agree: both say nothing was measured.
    """

    def _index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[float, float], Any]:
        return {(round(float(r.get("start", 0.0)), 6), round(float(r.get("end", 0.0)), 6)): r.get(field) for r in rows}

    before, after = _index(previous), _index(current)
    if set(before) != set(after):
        return False
    for key, old in before.items():
        new = after[key]
        if old is None and new is None:
            continue
        if old is None or new is None:
            return False
        if abs(float(new) - float(old)) > float(tolerance):
            return False
    return True
