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

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

__all__ = [
    "RoundRecord",
    "assess_convergence",
    "DEFAULT_MAX_ROUNDS",
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


# ── Convergence: C1-C4, cycle detection, and the two guards ──────────────────

DEFAULT_MAX_ROUNDS = 10
"""Round cap (D-12). Named rather than inlined because running out of rounds and agreeing are
different outcomes, and a reader needs to see which budget produced the first."""

EPISTEMIC_TOLERANCE = 1e-3
"""Credited change below which epistemic uncertainty counts as having stopped falling."""


@dataclass(frozen=True)
class RoundRecord:
    """What one L2 round produced, in the terms convergence is judged on.

    Attributes:
        round_index: 0-based round number.
        epistemic: Mean epistemic (reducible) uncertainty after the round, or ``None`` when it was
            not measured — total uncertainty can plateau while reducible doubt remains, which is
            why C1 is stated on the reducible part.
        assignment: The ``S_k`` → activation-channel mapping this round settled on (C2), from
            ``joint.per_speaker_presence``. The numbers can settle while this still flips, so it is
            judged separately. ``None`` means it was not measured, which blocks C2 rather than
            satisfying it: two unmeasured rounds compare equal, and reading that as stability would
            let the loop declare convergence on criteria nobody checked.
        measured_buckets: How many buckets carry a measurement (C3).
        untried_actions: Actions still available and unattempted anywhere (C4). ``None`` means the
            inventory was never taken, which blocks C4 — never having looked is not the same as
            having checked and found none.
        overwrote_values: Whether any action *replaced* a signal's value this round. Feeds the
            self-confirmation guard: a fall bought by overwriting is not a confidence gain.
        signature: A hashable digest of the round's state, for cycle detection.
    """

    round_index: int
    epistemic: Optional[float]
    assignment: Optional[Mapping[str, str]]
    measured_buckets: int
    untried_actions: Optional[int]
    overwrote_values: bool
    signature: str


def assess_convergence(
    history: Sequence[RoundRecord],
    *,
    tolerance: float = EPISTEMIC_TOLERANCE,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
) -> dict[str, Any]:
    """Decide whether the round loop has converged, and if not, whether to stop anyway.

    All four criteria must hold, because each can settle while another is still moving:

    - **C1** epistemic uncertainty stopped falling. Stated on the *reducible* part: the total can
      plateau while reducible doubt remains.
    - **C2** the ``S_k`` ↔ channel assignment is stable. D-7's joint space can keep flipping while
      every number holds still.
    - **C3** no bucket went unmeasured → measured. New coverage is progress; counting it as
      stability would stop the loop exactly when it began working.
    - **C4** no region has an untried available action, or "converged" means "ran out of ideas".

    Two guards shape C1 rather than sitting beside it:

    **Self-confirmation.** A fall in uncertainty that followed an action *overwriting* a value earns
    no credit, so a round cannot buy itself more rounds by replacing a signal with its preferred
    answer. Only a fall from an independent measurement agreeing counts.

    **Divergence is legitimate.** A confirmation action that refutes a claim *should* raise
    uncertainty, so convergence is not monotone decrease — defining it that way would bias the loop
    toward ratifying round 0 and leave the confirmation half of the design unable to change
    anything. A rise means the loop has not converged; it is not an error, and it does not stop the
    loop.

    Args:
        history: Round records in order, oldest first.
        tolerance: Credited epistemic change below which C1 holds.
        max_rounds: Cap (D-12).

    Returns:
        ``{"converged", "criteria", "blocking", "credited_epistemic_change", "diverged", "stop",
        "stop_reason"}``. ``stop`` is true when the loop should end for *any* reason; only
        ``converged`` says the criteria were met, so "ran out of rounds" and "agreed" stay
        distinguishable — a bare result cannot say which happened.
    """
    records = list(history or [])
    if len(records) < 2:
        # One observation cannot show stability, the same rule reliability follows for a single
        # pass. Reporting convergence here would assert something never measured.
        return {
            "converged": False,
            "criteria": {"c1": False, "c2": False, "c3": False, "c4": False},
            "blocking": ["c1", "c2", "c3", "c4"],
            "credited_epistemic_change": None,
            "diverged": False,
            "stop": bool(records and len(records) >= int(max_rounds)),
            "stop_reason": "max_rounds" if records and len(records) >= int(max_rounds) else None,
        }

    prev, cur = records[-2], records[-1]

    raw_change: Optional[float] = None
    if prev.epistemic is not None and cur.epistemic is not None:
        raw_change = float(cur.epistemic) - float(prev.epistemic)
    diverged = bool(raw_change is not None and raw_change > float(tolerance))

    # The guard: a *fall* that followed an overwrite is not credited. A rise still counts, because
    # a refutation is real information however it was produced.
    credited = raw_change
    if credited is not None and cur.overwrote_values and credited < 0:
        credited = 0.0

    c1 = credited is not None and abs(credited) <= float(tolerance)
    # Both guard against the unmeasured case explicitly. Comparing two ``None`` assignments
    # yields equality, and a defaulted zero action count reads as an exhausted inventory — either
    # would let a criterion nobody checked report as one that passed.
    c2 = prev.assignment is not None and cur.assignment is not None and prev.assignment == cur.assignment
    c3 = int(cur.measured_buckets) <= int(prev.measured_buckets)
    c4 = cur.untried_actions is not None and int(cur.untried_actions) <= 0
    criteria = {"c1": c1, "c2": c2, "c3": c3, "c4": c4}
    converged = all(criteria.values())

    # Cycle detection is separate from slow convergence on purpose (D-12): an A→B→A→B oscillation
    # reports movement every round, so it can only be caught by state repeating.
    earlier = [r.signature for r in records[:-1]]
    cycling = cur.signature in earlier and not converged

    stop_reason: Optional[str] = None
    if converged:
        stop_reason = "converged"
    elif cycling:
        stop_reason = "cycle"
    elif len(records) >= int(max_rounds):
        stop_reason = "max_rounds"

    return {
        "converged": converged,
        "criteria": criteria,
        "blocking": [name for name, ok in criteria.items() if not ok],
        "credited_epistemic_change": credited,
        "diverged": diverged,
        "stop": stop_reason is not None,
        "stop_reason": stop_reason,
    }
