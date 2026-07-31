"""Confidence, variability, and uncertainty — three quantities, three estimators.

The codebase had been calling all of them "uncertainty", which is why a max-doubt fold, a
Shannon entropy, and a max-minus-min spread all ended up in a column of that name. They answer
different questions and only one of them is a probability:

**Confidence** — ``P(proposition)``, in ``[0, 1]``. Estimated as the weighted share of signals
asserting the proposition. Because it is a probability it can be calibrated against ground
truth, and ``0.0`` is the confident claim *"definitely not"* rather than "we did not look".

**Variability** — dispersion of repeated measurements of one quantity: the sample standard
deviation, in the units of the quantity. Deliberately *not* squeezed into ``[0, 1]``; rescaling
it would make it a different statistic and invite reading it as a probability. Needs at least
two measurements — with one, zero would assert perfect agreement that was never observed.

**Uncertainty** — how undetermined the answer is, estimated as Shannon entropy over the
distribution of outcomes, normalised by ``log k`` so an even split reads 1.0 whether there are
two outcomes or five. Raw entropy in nats is unbounded above and so cannot be compared across
axes with different outcome counts.

Uncertainty further decomposes, which is what makes it actionable:

    total = H(mean of the signals' distributions)
    aleatoric = mean of H(each signal's distribution)
    epistemic = total - aleatoric

Epistemic uncertainty is disagreement *between* signals, and it is the reducible part: another
measurement can resolve it. Aleatoric uncertainty is doubt every signal shares, which more
measurements of the same kind cannot remove. Reporting shared internal doubt as reducible would
send the adaptive loop off to gather evidence that cannot help — the decomposition exists so it
can tell the difference.

This is the standard mutual-information decomposition used for ensemble and MC-dropout
uncertainty; the speaker ``0 <= epistemic <= total`` holds by Jensen's inequality, and a
violation means a sign error rather than an interesting finding.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

__all__ = [
    "confidence",
    "entropy_uncertainty",
    "epistemic_uncertainty",
    "variability",
]


def confidence(
    votes: Sequence[bool],
    *,
    weights: Sequence[float] | None = None,
) -> float | None:
    """``P(proposition)`` as the weighted share of signals asserting it.

    Args:
        votes: One boolean per signal.
        weights: Optional per-signal weight, positionally aligned. Weights renormalise, so the
            result stays a probability.

    Returns:
        The probability in ``[0, 1]``, or ``None`` when no signal voted — distinct from ``0.0``,
        which is the confident claim that the proposition is false.

    Raises:
        ValueError: If ``weights`` has a different length than ``votes``.
    """
    if weights is not None and len(weights) != len(votes):
        raise ValueError(f"weights has length {len(weights)} but there are {len(votes)} votes")
    if not votes:
        return None
    w = [1.0] * len(votes) if weights is None else [max(0.0, float(x)) for x in weights]
    total = sum(w)
    if total <= 0.0:
        return None
    return sum(wi for vote, wi in zip(votes, w) if vote) / total


def variability(measurements: Sequence[float]) -> float | None:
    """Sample standard deviation of repeated measurements of one quantity.

    Population rather than Bessel-corrected: these are the measurements actually taken, not a
    sample drawn from a larger pool of possible measurements, so there is no inference to a
    wider population to correct for.

    Returns:
        The dispersion in the units of the quantity, or ``None`` with fewer than two
        measurements — zero would assert an agreement that was never observed.
    """
    values = [float(v) for v in measurements]
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def _normalise(distribution: Mapping[str, float]) -> dict[str, float] | None:
    """Vote masses to probabilities, or ``None`` when there is no mass to normalise."""
    masses = {str(k): max(0.0, float(v)) for k, v in distribution.items()}
    total = sum(masses.values())
    if not masses or total <= 0.0:
        return None
    return {k: v / total for k, v in masses.items()}


def _entropy_nats(probabilities: Iterable[float]) -> float:
    return -sum(p * math.log(p) for p in probabilities if p > 0.0)


def entropy_uncertainty(distribution: Mapping[str, float]) -> float | None:
    """Normalised Shannon entropy of a distribution over outcomes, in ``[0, 1]``.

    Args:
        distribution: Outcome to mass. Masses need not sum to 1; they are normalised first,
            because vote masses are not probabilities until divided by their total.

    Returns:
        ``0.0`` when one outcome is certain, ``1.0`` on an even split, or ``None`` when nothing
        was observed. A single-outcome space gives ``0.0``: there is nothing to be uncertain
        between, which is different from having observed nothing.
    """
    probabilities = _normalise(distribution)
    if probabilities is None:
        return None
    if len(probabilities) < 2:
        return 0.0
    return _entropy_nats(probabilities.values()) / math.log(len(probabilities))


def epistemic_uncertainty(
    per_signal: Sequence[Mapping[str, float]],
) -> tuple[float | None, float | None]:
    """Split total uncertainty into its total and its reducible (epistemic) part.

    Args:
        per_signal: One distribution over outcomes per signal.

    Returns:
        ``(total, epistemic)``, both normalised to ``[0, 1]``, or ``(None, None)`` when no
        signal reported a distribution. ``epistemic`` is ``0.0`` for a single signal —
        disagreement needs at least two parties — and never exceeds ``total``.
    """
    distributions = [d for d in (_normalise(p) for p in per_signal) if d is not None]
    if not distributions:
        return None, None

    outcomes = sorted({k for d in distributions for k in d})
    if len(outcomes) < 2:
        return 0.0, 0.0
    scale = math.log(len(outcomes))

    mean_distribution = {
        outcome: sum(d.get(outcome, 0.0) for d in distributions) / len(distributions) for outcome in outcomes
    }
    total = _entropy_nats(mean_distribution.values()) / scale
    aleatoric = sum(_entropy_nats(d.values()) for d in distributions) / (len(distributions) * scale)
    # Clamped at zero: the speaker holds by Jensen's inequality, so a negative value is float
    # error rather than a finding, and letting it through would report reducible doubt as
    # negative.
    return min(1.0, total), max(0.0, min(total, total - aleatoric))
