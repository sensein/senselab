"""Per-axis sub-signal aggregators (the ``--uncertainty-aggregator`` flag values).

Each aggregator collapses a list of sub-signal *uncertainties* in ``[0, 1]`` (with
``None`` indicating "this sub-signal was unavailable") into a single ``[0, 1]`` scalar or
``None`` (when all inputs were ``None``). Models lacking a native confidence signal MUST
be dropped from the aggregator's input rather than treated as zero (FR-004 / FR-007).
"""

from __future__ import annotations

import math
from typing import Literal

AGGREGATORS: tuple[str, ...] = ("min", "mean", "harmonic_mean", "disagreement_weighted")
"""Allowed values for ``--uncertainty-aggregator``. Default is ``min``."""

AggregatorName = Literal["min", "mean", "harmonic_mean", "disagreement_weighted"]


def _confidences_from_uncertainties(uncertainties: list[float]) -> list[float]:
    """Convert sub-signal uncertainties (high = bad) to confidences (high = good)."""
    return [max(0.0, min(1.0, 1.0 - u)) for u in uncertainties]


def apply_aggregator(
    sub_signals: list[float | None],
    name: AggregatorName | str,
    weights: list[float] | None = None,
) -> float | None:
    """Combine per-axis sub-signal uncertainties into a single uncertainty scalar.

    Args:
        sub_signals: Each entry is either a sub-signal uncertainty in ``[0, 1]`` or
            ``None``. ``None`` entries are dropped before aggregation.
        name: One of ``AGGREGATORS``. Must be a recognized aggregator.
        weights: Optional per-signal reliability in ``[0, 1]``, positionally aligned with
            ``sub_signals`` — typically measured by perturbation (see ``reliability.py``). A
            signal's *doubt* is scaled by its reliability before aggregation, so a signal
            that contradicts itself under perturbation cannot decide the axis on its own.
            This matters most under ``min`` (max-doubt), where an unweighted single
            saturated signal overrides every corroborating one; it was doing exactly that on
            a real recording. Weights are dropped alongside their ``None`` sub-signals, so
            positions stay aligned. Omit for unweighted aggregation.

    Returns:
        The combined uncertainty scalar in ``[0, 1]``, or ``None`` when every sub-signal
        was ``None``.

    Raises:
        ValueError: If ``name`` is not a recognized aggregator, or ``weights`` is present
            with a different length than ``sub_signals`` — recycling or truncating would
            silently attribute one signal's reliability to another.
    """
    if name not in AGGREGATORS:
        raise ValueError(f"unknown aggregator {name!r}; must be one of {AGGREGATORS}")
    if weights is not None and len(weights) != len(sub_signals):
        raise ValueError(f"weights has length {len(weights)} but there are {len(sub_signals)} sub-signals")

    kept = [(u, 1.0 if weights is None else float(weights[i])) for i, u in enumerate(sub_signals) if u is not None]
    values = [u for u, _w in kept]
    if not values:
        return None
    if weights is not None:
        # Scale doubt by reliability. An unreliable signal's objection shrinks toward the
        # corroborated answer instead of being taken at face value; the caller's floor on
        # reliability keeps it visible rather than erasing it.
        values = [float(u) * max(0.0, min(1.0, w)) for u, w in kept]

    # Clip inputs to [0, 1] defensively — sub-signals like 1 - exp(avg_logprob) can
    # nudge over 1.0 due to float rounding when avg_logprob is near 0.
    values = [max(0.0, min(1.0, float(u))) for u in values]
    confidences = _confidences_from_uncertainties(values)

    if name == "min":
        # Worst (most-doubtful) sub-signal wins — the canonical "show me where any signal
        # is unsure" default.
        return 1.0 - min(confidences)
    if name == "mean":
        return 1.0 - (sum(confidences) / len(confidences))
    if name == "harmonic_mean":
        # Penalises sub-signals close to zero confidence more aggressively than mean.
        # statistics.harmonic_mean rejects zeros, so substitute a tiny floor.
        floored = [max(c, 1e-9) for c in confidences]
        h = len(floored) / sum(1.0 / c for c in floored)
        return 1.0 - h
    if name == "disagreement_weighted":
        # Uncertainty when at least one sub-signal disagrees, scaled by mean uncertainty
        # — surfaces buckets where many signals are slightly off rather than one wildly off.
        max_u = max(values)
        mean_conf = sum(confidences) / len(confidences)
        return (1.0 - mean_conf) * max_u

    # Unreachable — name is in AGGREGATORS by the guard above.
    raise ValueError(f"unhandled aggregator {name!r}")
