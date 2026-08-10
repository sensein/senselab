"""Turn a measured speaker-count accuracy curve into a declared ceiling.

Separated from the probe that produces the curve because this file is the only part
carrying a *judgement* rather than a measurement, and it needs no GPU, no NeMo and no
audio to exercise. `DiarizationCapabilities.max_speakers` currently reads `None` for
four of six backends, meaning **unmeasured**; this reduces a curve to the integer that
replaces it.

The reduction rule, and why it is written here rather than hidden in the probe:

    the largest k at which the backend reports exactly k speakers in >= 80% of sessions,
    and at which *every* smaller count also clears that bar.

The 80% is a judgement. The profile records it beside the curve it was applied to so a
reader who disagrees can recompute from the same numbers without re-running 160 GPU
sessions. That follows this repository's convention and its counter-example: two defects
came from literals nobody ever fitted.

The "every smaller count too" half is the part that is easy to get wrong. A curve that
dips at k=4 and recovers at k=6 does not have a ceiling of 6 — the k=6 successes are not
dependable if k=4 fails, so the honest answer is 3. A ceiling a backend intermittently
exceeds is not a ceiling.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

# A judgement, not a measurement. Exactly 16 of this probe's 20 sessions per cell, chosen
# so a single session cannot flip a verdict the way 8-of-10 would. Changing this
# invalidates every ceiling already derived under it, which is why the profile stores the
# value it used rather than reading this constant at load time.
DEFAULT_ACCURACY_THRESHOLD = 0.8


def exact_count_accuracy(predictions: Sequence[Optional[int]], true_k: int) -> float:
    """Return the fraction of sessions where the backend reported exactly ``true_k``.

    Exact match only: reporting 3 speakers when there are 4 is wrong, not partially
    right. A near-miss metric would let a backend that systematically undercounts look
    capable at high speaker counts, which is precisely the error this probe exists to
    detect.

    Args:
        predictions: One entry per completed session. ``None`` means the backend refused
            or crashed on that session — it cannot count as correct, but it is preserved
            as ``None`` upstream so a reader of the confusion can still tell a refusal
            from a wrong number.
        true_k: The generated speaker count for these sessions.

    Returns:
        The exact-match fraction in ``[0, 1]``. An empty sequence scores ``0.0`` rather
        than raising: the caller refuses on an under-filled cell, and a divide-by-zero
        crash there would obscure which cell was short.
    """
    if not predictions:
        return 0.0
    return sum(1 for p in predictions if p == true_k) / len(predictions)


def derive_ceiling(
    curve: Mapping[int, float],
    threshold: float = DEFAULT_ACCURACY_THRESHOLD,
) -> Optional[int]:
    """Reduce an accuracy curve to the largest speaker count the backend handles.

    Walks the counts in ascending order and stops at the first one that fails, so a later
    recovery cannot raise the ceiling (see the module docstring). A gap in the curve stops
    it too: a missing count cannot be assumed to have passed, and treating it as one would
    silently overstate the ceiling. The probe refuses to emit an incomplete profile, so a
    gap arriving here means something upstream let it through.

    Args:
        curve: Maps true speaker count to exact-count accuracy in ``[0, 1]``.
        threshold: Minimum accuracy to count as handled, inclusive. Defaults to
            :data:`DEFAULT_ACCURACY_THRESHOLD`.

    Returns:
        The largest ``k`` such that every count from the smallest present up to and
        including ``k`` meets ``threshold``; or ``None`` if even the smallest fails, which
        carries the same meaning as ``DiarizationCapabilities.max_speakers = None`` — the
        probe established nothing.
    """
    if not curve:
        return None

    ceiling: Optional[int] = None
    expected = min(curve)
    for k in sorted(curve):
        # A hole in the sweep: stop rather than skip. `expected` tracks the next contiguous
        # count, so k=4 following k=2 ends the walk at 2.
        if k != expected:
            break
        if curve[k] < threshold:
            break
        ceiling = k
        expected = k + 1
    return ceiling
