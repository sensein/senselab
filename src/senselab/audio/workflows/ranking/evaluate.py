"""Ranking-quality check: rank agreement (primary) + band separation (secondary).

Primary indicator (FR-010a): Spearman ρ between the ranking goodness (rank 1 =
best, so ``goodness = -rank``) and the annotated quality, with Kendall τ-b
alongside. Secondary (FR-008): top-vs-bottom-band pairwise agreement (AUC-style)
plus the mean-quality margin. Reports ``evaluable=false`` with a reason when
there is insufficient annotated data (FR-010).
"""

from __future__ import annotations

import numpy as np

from senselab.audio.workflows.ranking.constants import (
    DEFAULT_SEPARATION_TARGET,
    MIN_ANNOTATED_PER_BAND,
    ORDINAL_QUALITY_MAP,
)
from senselab.audio.workflows.ranking.types import Annotation, Ranking, SeparationResult


def _quality(annotation: Annotation) -> float | None:
    """Numeric quality from an annotation (ordinal label preferred, else score)."""
    if annotation.label is not None:
        return ORDINAL_QUALITY_MAP[annotation.label]
    if annotation.score is not None:
        return float(annotation.score)
    return None


def _spearman_kendall(goodness: np.ndarray, quality: np.ndarray) -> tuple[float | None, float | None]:
    """Spearman ρ and Kendall τ-b; numpy Spearman fallback if scipy is unavailable."""
    if np.unique(goodness).size < 2 or np.unique(quality).size < 2:
        return None, None
    try:
        from scipy.stats import kendalltau, spearmanr

        rho = float(spearmanr(goodness, quality)[0])  # [0] is the statistic across all SciPy versions
        tau = float(kendalltau(goodness, quality)[0])
        return (rho if not np.isnan(rho) else None, tau if not np.isnan(tau) else None)
    except ImportError:
        gr = np.argsort(np.argsort(goodness)).astype(float)
        qr = np.argsort(np.argsort(quality)).astype(float)
        rho = float(np.corrcoef(gr, qr)[0, 1])
        return (rho if not np.isnan(rho) else None, None)


def _pairwise_agreement(top_q: np.ndarray, bottom_q: np.ndarray) -> float:
    """P(top quality > bottom quality) + 0.5·P(equal) over all top×bottom pairs."""
    wins = 0.0
    for t in top_q:
        wins += float(np.sum(t > bottom_q)) + 0.5 * float(np.sum(t == bottom_q))
    return wins / (top_q.size * bottom_q.size)


def evaluate_ranking(
    ranking: Ranking,
    annotations: list[Annotation],
    *,
    separation_target: float = DEFAULT_SEPARATION_TARGET,
) -> SeparationResult:
    """Compute the ranking-quality diagnostic for ``ranking`` against ``annotations``."""
    active = {a.item_id: a for a in annotations if a.resolution == "active"}

    goodness: list[float] = []
    quality: list[float] = []
    top_q: list[float] = []
    bottom_q: list[float] = []
    for it in ranking.items:
        if it.status != "scored" or it.item_id not in active:
            continue
        q = _quality(active[it.item_id])
        if q is None:
            continue
        goodness.append(-float(it.rank))  # type: ignore[arg-type]
        quality.append(q)
        if it.band == "top":
            top_q.append(q)
        elif it.band == "bottom":
            bottom_q.append(q)

    n_annotated = len(quality)
    result = SeparationResult(
        version_id=ranking.version_id,
        rank_agreement_spearman=None,
        rank_agreement_kendall_tau_b=None,
        band_pairwise_agreement=None,
        band_quality_margin=None,
        n_annotated=n_annotated,
        n_annotated_top=len(top_q),
        n_annotated_bottom=len(bottom_q),
        evaluable=False,
        reason=None,
        meets_separation_target=None,
    )

    if n_annotated < 2:
        result.reason = f"only {n_annotated} annotated scored item(s); need ≥2"
        return result

    rho, tau = _spearman_kendall(np.asarray(goodness), np.asarray(quality))
    result.rank_agreement_spearman = rho
    result.rank_agreement_kendall_tau_b = tau
    if rho is None:
        result.reason = "no variation in ranking positions or quality among annotated items"
        return result
    result.evaluable = True

    if len(top_q) >= MIN_ANNOTATED_PER_BAND and len(bottom_q) >= MIN_ANNOTATED_PER_BAND:
        ta, ba = np.asarray(top_q), np.asarray(bottom_q)
        agreement = _pairwise_agreement(ta, ba)
        result.band_pairwise_agreement = agreement
        result.band_quality_margin = float(ta.mean() - ba.mean())
        result.meets_separation_target = agreement >= separation_target
    else:
        result.reason = (
            f"band separation not evaluable: annotated top={len(top_q)} "
            f"bottom={len(bottom_q)} (need ≥{MIN_ANNOTATED_PER_BAND} each); rank-agreement reported"
        )
    return result
