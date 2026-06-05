"""Turn per-item scores into a deterministic ranking, and orchestrate versioned runs.

Determinism (SC-003): items are ordered by score (honoring ``direction``) with
ties broken by ``item_id`` ascending — a stable two-key sort. Bands are assigned
by position (a coarse lens, default 20%), kept disjoint at small N (research D5).
"""

from __future__ import annotations

import math
from pathlib import Path

from senselab.audio.workflows.ranking import io, metric
from senselab.audio.workflows.ranking.constants import DEFAULT_BAND_FRACTION, TIE_BREAK
from senselab.audio.workflows.ranking.store import RankingStore, metric_definition_hash
from senselab.audio.workflows.ranking.types import (
    Band,
    Direction,
    MetricDefinition,
    MetricOrigin,
    Ranking,
    RankingItem,
    RecalibrationResult,
)


def assign_bands(n_scored: int, band_fraction: float) -> list[Band]:
    """Return the band label for each ranked position (index 0 = top of ranking).

    Top and bottom bands are kept disjoint; the middle may be empty at small N.
    """
    if n_scored <= 0:
        return []
    if n_scored == 1:
        return ["middle"]
    k = max(1, math.ceil(band_fraction * n_scored))
    bands: list[Band] = ["middle"] * n_scored
    if 2 * k <= n_scored:
        for i in range(k):
            bands[i] = "top"
            bands[n_scored - 1 - i] = "bottom"
    else:  # bands would overlap — split at the midpoint, middle empty
        top_count = n_scored // 2
        for i in range(n_scored):
            bands[i] = "top" if i < top_count else "bottom"
    return bands


def build_ranking(
    items: list[RankingItem],
    *,
    version_id: str,
    unit: str,
    direction: Direction,
    band_fraction: float,
    provenance: dict,
) -> Ranking:
    """Assemble a :class:`Ranking` from scored/unscorable items (FR-002/004/005)."""
    scored = [it for it in items if it.status == "scored"]
    unscorable = [it for it in items if it.status == "unscorable"]

    # Stable two-key sort: item_id ascending first, then by score (so ties keep id-asc).
    scored.sort(key=lambda it: it.item_id)
    scored.sort(key=lambda it: it.score, reverse=(direction == "higher_is_better"))  # type: ignore[arg-type,return-value]

    n = len(scored)
    bands = assign_bands(n, band_fraction)
    for idx, it in enumerate(scored):
        it.rank = idx + 1
        it.percentile = (idx / (n - 1)) if n > 1 else 0.0
        it.band = bands[idx]

    return Ranking(
        version_id=version_id,
        unit=unit,  # type: ignore[arg-type]
        band_fraction=band_fraction,
        items=scored + unscorable,
        n_scored=n,
        n_unscorable=len(unscorable),
        provenance=provenance,
    )


def rank_corpus(
    store: RankingStore,
    signal_table_path: Path | str,
    definition: MetricDefinition,
    *,
    created_at: str,
    band_fraction: float = DEFAULT_BAND_FRACTION,
    origin: MetricOrigin | None = None,
    parent_version_id: str | None = None,
    as_version: str | None = None,
    recal: RecalibrationResult | None = None,
) -> Ranking:
    """Score → rank a corpus, persisting an immutable metric version + ranking.

    ``origin`` defaults to ``initial`` for the store's first version, else ``manual``.
    """
    from senselab.audio.workflows.ranking.types import MetricVersion  # local: avoid cycle at import

    table = io.load_signal_table(Path(signal_table_path))
    metric.validate_definition(definition, table.signal_columns)
    items = metric.score_items(table, definition)

    existing = store.list_versions()
    version_id = as_version or store.next_version_id()
    if origin is None:
        origin = "initial" if not existing else "manual"

    provenance = {
        "metric_definition_hash": metric_definition_hash(definition),
        "tie_break": TIE_BREAK,
        "created_at": created_at,
        "signal_columns": ",".join(table.signal_columns),
    }
    ranking = build_ranking(
        items,
        version_id=version_id,
        unit=table.unit,
        direction=definition.direction,
        band_fraction=band_fraction,
        provenance=provenance,
    )

    version = MetricVersion(
        version_id=version_id,
        definition=definition,
        origin=origin,
        parent_version_id=parent_version_id,
        created_at=created_at,
        recal=recal,
    )
    store.write_metric_version(version, unit=table.unit)
    io.write_ranking(store.ranking_path(version_id), ranking)
    return ranking


def update_metric_manual(
    store: RankingStore,
    signal_table_path: Path | str,
    definition: MetricDefinition,
    *,
    created_at: str,
    band_fraction: float = DEFAULT_BAND_FRACTION,
) -> Ranking:
    """Create a new ``manual``-origin version + ranking from a revised definition (FR-015)."""
    versions = store.list_versions()
    parent = versions[-1] if versions else None
    return rank_corpus(
        store,
        signal_table_path,
        definition,
        created_at=created_at,
        band_fraction=band_fraction,
        origin="manual",
        parent_version_id=parent,
    )


def recalibrate_and_propose(
    store: RankingStore,
    signal_table_path: Path | str,
    *,
    base_version_id: str | None = None,
) -> RecalibrationResult:
    """Propose recalibrated weights from active annotations (advisory — FR-016/017).

    Returns a :class:`RecalibrationResult`; nothing is written until the caller
    accepts it (e.g. by passing the proposed definition to :func:`rank_corpus`
    with ``origin='recalibrated'``).
    """
    from senselab.audio.workflows.ranking import annotate, recalibrate

    versions = store.list_versions()
    if not versions:
        raise ValueError("cannot recalibrate: store has no metric version yet")
    base = base_version_id or versions[-1]
    table = io.load_signal_table(Path(signal_table_path))
    base_definition = store.read_metric_version(base).definition
    annotations = annotate.load_active_annotations(store)
    return recalibrate.propose_recalibration(table, base_definition, annotations)
