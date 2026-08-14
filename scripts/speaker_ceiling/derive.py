"""Turn measured speaker-count results into the two declared facts they support.

Separated from the probe that produces the numbers because this file is the only part
carrying a *judgement* rather than a measurement, and it needs no GPU, no NeMo and no
audio to exercise.

Two independent rules live here, not one:

- :func:`derive_ceiling` reduces an *accuracy curve* to a performance verdict: the
  largest count the backend counts *correctly*, often enough to trust.
- :func:`derive_structural_bound` reduces a *confusion* (the distribution of predicted
  counts at one true k) to a structural verdict: the largest count the backend can
  *emit at all*, regardless of whether the number it emits is right.

The seed-17 speaker-ceiling sweep is what made the distinction unavoidable:
`DiarizationCapabilities.max_speakers` used to read one integer meant to answer both
questions, and Sortformer and the child-adult classifier showed why that was wrong —
both plateau at a fixed output (4 and 2) long before their counting accuracy would
have suggested a limit, while Pyannote, VibeVoice, MOSS and DiariZen never plateau at
all across k=1..8 despite `derive_ceiling` giving each of them a real (if modest)
accuracy-based ceiling. Folding both into one field hid which claim a given number was
making.

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


def derive_structural_bound(confusion_at_max_k: Mapping[str, int], true_k: int) -> Optional[int]:
    """Reduce one (backend, true *k*) confusion to a structural ceiling, or ``None``.

    Answers a different question than :func:`derive_ceiling`: not "how large a count can this
    backend count *correctly*" but "how large a count can it *emit at all*". The signal is a
    backend collapsing to **one** predicted count across every completed session at the largest
    true ``k`` it was swept over — a spread (``{"5": 3, "6": 8, "7": 6, "8": 3}``) means it is
    still tracking the true count, however badly; a single accumulation point (``{"4": 20}`` at
    ``true_k=8``) means it structurally cannot say more than 4, no matter how many speakers are
    actually present. This is exactly the pattern the seed-17 probe found for Sortformer (20/20
    "4" at k=8) and the child-adult classifier (20/20 "2" at k=8), and did not find for the other
    four backends, whose k=8 confusions stayed spread across multiple values.

    Only the confusion at the *largest* tested ``k`` is trustworthy evidence: a plateau seen at a
    smaller k could still be a transient dip a higher k recovers from (the same non-monotonicity
    :func:`derive_ceiling` guards against), so a caller must pass the top of its sweep, not an
    arbitrary cell.

    A perfect (or uniformly wrong-but-consistent) score at the *true* count is explicitly
    excluded: if every session predicted exactly ``true_k``, or some value at or above it, that is
    an accuracy result at this ``k``, not evidence of a ceiling below it — a backend that nails
    k=8 could plausibly handle k=9 too, and nothing here has a basis to say otherwise. Only a
    plateau strictly below ``true_k`` counts as a real structural bound.

    Args:
        confusion_at_max_k: One (backend, k) cell's confusion, shaped like
            ``evaluate.confusion_from_outcomes``'s return value: predicted count as a string (or
            the literal ``"refused"``) mapped to the number of sessions that produced it. Must be
            the cell at the largest ``k`` the backend was swept over.
        true_k: The true speaker count of the sessions in ``confusion_at_max_k``.

    Returns:
        The plateaued count if every completed (non-refused) session in the cell reported the
        same value and that value is strictly less than ``true_k``; ``None`` otherwise. ``None``
        means "no structural ceiling observed in this sweep", the same meaning
        ``DiarizationCapabilities.max_speakers = None`` carries elsewhere in this repo — it does
        not mean unlimited. A cell where every session refused (no completed predictions at all)
        also returns ``None``: a universal refusal says nothing about what count the backend
        *can* emit.
    """
    completed = {predicted: n for predicted, n in confusion_at_max_k.items() if predicted != "refused" and n > 0}
    if len(completed) != 1:
        return None
    (bound_str,) = completed
    bound = int(bound_str)
    if bound >= true_k:
        return None
    return bound


def format_structural_bound_evidence(confusion_at_max_k: Mapping[str, int], true_k: int, probe_label: str) -> str:
    """Render :func:`derive_structural_bound`'s verdict as a ``max_speakers_evidence`` string.

    Kept beside the derivation rather than left to each caller to phrase independently, so a
    written evidence string and the number it describes cannot drift apart — a hand-typed
    "saturates at 4" next to a differently-derived bound would be worse than the ambiguity
    ``DiarizationCapabilities.max_speakers_evidence`` exists to remove.

    Args:
        confusion_at_max_k: See :func:`derive_structural_bound`.
        true_k: See :func:`derive_structural_bound`.
        probe_label: Identifies which run produced this, e.g. ``"probe seed-17"`` — embedded
            verbatim so a reader can trace the number back to a specific corpus and seed.

    Returns:
        ``"measured: saturates at {bound} on {n}/{total} k={true_k} sessions ({probe_label})"``
        when :func:`derive_structural_bound` finds a plateau; ``"measured: no saturation, emits
        up to {max} ({probe_label})"`` when it does not, where ``{max}`` is the largest predicted
        count actually observed (refusals excluded); or ``"measured: no completed sessions at
        k={true_k} ({probe_label})"`` for a cell where every session refused, since there is then
        no observed count to report a maximum of.
    """
    completed = {predicted: n for predicted, n in confusion_at_max_k.items() if predicted != "refused" and n > 0}
    total = sum(confusion_at_max_k.values())
    bound = derive_structural_bound(confusion_at_max_k, true_k)
    if bound is not None:
        return f"measured: saturates at {bound} on {completed[str(bound)]}/{total} k={true_k} sessions ({probe_label})"
    if not completed:
        return f"measured: no completed sessions at k={true_k} ({probe_label})"
    highest_observed = max(int(predicted) for predicted in completed)
    return f"measured: no saturation, emits up to {highest_observed} ({probe_label})"
