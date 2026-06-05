"""Evaluate a metric definition over a signal table → per-item scores + statuses.

Scoring semantics follow ``contracts/metric-definition.schema.md``:

1. Validate the definition (≥1 term; every referenced signal must exist — FR-019).
2. For each term apply its ``transform`` (fit on the observed, non-missing values
   of that signal), multiply by ``weight``.
3. Apply the per-signal ``missing`` policy: ``unscorable`` (default) marks the
   item unscorable; ``neutral`` contributes 0; ``fill:<v>`` substitutes ``v``.
4. The combined score is the weighted sum. ``direction`` is applied later in
   ``rank.py`` (it does not change the stored score).
"""

from __future__ import annotations

import numpy as np

from senselab.audio.workflows.ranking.types import MetricDefinition, RankingItem, SignalTable


class MetricError(ValueError):
    """Raised when a metric definition is invalid for the target signal table."""


def _parse_missing(policy: str) -> tuple[str, float | None]:
    """Parse a term ``missing`` policy into ``(kind, fill_value)``."""
    if policy in ("unscorable", "neutral"):
        return policy, None
    if policy.startswith("fill:"):
        try:
            return "fill", float(policy.split(":", 1)[1])
        except ValueError as exc:
            raise MetricError(f"invalid fill policy {policy!r}") from exc
    raise MetricError(f"unknown missing policy {policy!r}")


def _apply_transform(values: np.ndarray, transform: str, params: dict) -> np.ndarray:
    """Apply a per-signal transform over a column (NaNs preserved as NaN)."""
    observed = values[~np.isnan(values)]
    if transform == "identity":
        return values
    if transform == "zscore":
        if observed.size == 0:
            return values
        mean = float(observed.mean())
        std = float(observed.std())
        return (values - mean) / std if std > 0 else values - mean
    if transform == "minmax":
        lo = float(params.get("min", observed.min())) if observed.size else 0.0
        hi = float(params.get("max", observed.max())) if observed.size else 1.0
        span = hi - lo
        return (values - lo) / span if span > 0 else np.zeros_like(values)
    if transform == "rank":
        out = np.full_like(values, np.nan)
        if observed.size:
            order = np.argsort(np.argsort(observed))
            ranks = order / max(observed.size - 1, 1)
            out[~np.isnan(values)] = ranks
        return out
    if transform == "clip":
        lo = float(params.get("min", -np.inf))
        hi = float(params.get("max", np.inf))
        return np.clip(values, lo, hi)
    if transform == "threshold":
        at = float(params.get("at", 0.5))
        out = np.where(values >= at, 1.0, 0.0)
        out[np.isnan(values)] = np.nan
        return out
    raise MetricError(f"unknown transform {transform!r}")


def validate_definition(defn: MetricDefinition, signal_columns: list[str]) -> None:
    """Validate a metric definition against the available signal columns (FR-019)."""
    if not defn.terms:
        raise MetricError(f"metric {defn.name!r} has no terms")
    available = set(signal_columns)
    for term in defn.terms:
        if term.signal not in available:
            raise MetricError(
                f"metric {defn.name!r} references unknown signal {term.signal!r}; "
                f"available signals: {sorted(available)}"
            )
        _parse_missing(term.missing)  # validates policy syntax


def score_items(table: SignalTable, defn: MetricDefinition) -> list[RankingItem]:
    """Score every item in ``table`` under ``defn``; never drops items (FR-002/006).

    Returns one :class:`RankingItem` per input item with ``score``/``status``/
    ``reason`` set; ``rank``/``percentile``/``band`` are assigned later by
    ``rank.py``.
    """
    validate_definition(defn, table.signal_columns)
    n = len(table.item_ids)

    totals = np.zeros(n, dtype=float)
    unscorable_reason: list[str | None] = [None] * n

    for term in defn.terms:
        kind, fill = _parse_missing(term.missing)
        raw = np.array(table.columns[term.signal], dtype=float)
        missing_mask = np.isnan(raw)

        if kind == "fill" and fill is not None:
            raw = np.where(missing_mask, fill, raw)

        transformed = _apply_transform(raw, term.transform, term.transform_params)
        contribution = term.weight * transformed

        for i in range(n):
            if unscorable_reason[i] is not None:
                continue
            if missing_mask[i]:
                if kind == "unscorable":
                    unscorable_reason[i] = f"missing:{term.signal}"
                    continue
                if kind == "neutral":
                    continue  # contributes 0
                # kind == "fill": value already substituted above
            totals[i] += float(contribution[i])

    items: list[RankingItem] = []
    for i, item_id in enumerate(table.item_ids):
        reason = unscorable_reason[i]
        if reason is not None:
            items.append(
                RankingItem(
                    item_id=item_id,
                    score=None,
                    rank=None,
                    percentile=None,
                    band=None,
                    status="unscorable",
                    reason=reason,
                )
            )
        else:
            items.append(
                RankingItem(
                    item_id=item_id, score=float(totals[i]), rank=None, percentile=None, band=None, status="scored"
                )
            )
    return items
