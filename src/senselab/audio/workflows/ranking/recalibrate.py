"""Assisted recalibration — propose new metric weights from annotations (FR-016/017).

Framed as learning-to-rank-lite via pairwise logistic regression (research D4):
for every pair of annotated items with *distinct* quality, the feature is the
difference of their transformed signal vectors and the label is which is better.
``LogisticRegression`` (no intercept) yields per-signal weights that maximize
rank agreement; the proposal is advisory and never auto-adopted.
"""

from __future__ import annotations

import numpy as np

from senselab.audio.workflows.ranking import metric
from senselab.audio.workflows.ranking.constants import (
    LOW_PAIR_WARN,
    MIN_ANNOTATIONS_RECAL,
    MIN_QUALITY_LEVELS_RECAL,
    ORDINAL_QUALITY_MAP,
)
from senselab.audio.workflows.ranking.types import (
    Annotation,
    MetricDefinition,
    RecalibrationResult,
    RecalStatus,
    SignalTable,
    SignalTerm,
)


def _quality(annotation: Annotation) -> float | None:
    if annotation.label is not None:
        return ORDINAL_QUALITY_MAP[annotation.label]
    if annotation.score is not None:
        return float(annotation.score)
    return None


def _feature_matrix(table: SignalTable, defn: MetricDefinition) -> dict[str, np.ndarray]:
    """Transformed feature column per term signal (NaN where the signal is missing)."""
    feats: dict[str, np.ndarray] = {}
    for term in defn.terms:
        raw = np.asarray(table.columns[term.signal], dtype=float)
        feats[term.signal] = metric._apply_transform(raw, term.transform, term.transform_params)
    return feats


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if np.unique(a).size < 2 or np.unique(b).size < 2:
        return None
    try:
        from scipy.stats import spearmanr

        rho = float(spearmanr(a, b).statistic)
    except ImportError:
        rho = float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])
    return None if np.isnan(rho) else rho


def propose_recalibration(
    table: SignalTable,
    base_definition: MetricDefinition,
    annotations: list[Annotation],
) -> RecalibrationResult:
    """Propose recalibrated weights; refuse/warn per guards (FR-017)."""
    metric.validate_definition(base_definition, table.signal_columns)
    index = {iid: i for i, iid in enumerate(table.item_ids)}
    active = [a for a in annotations if a.resolution == "active" and a.item_id in index]

    feats = _feature_matrix(table, base_definition)
    signals = [t.signal for t in base_definition.terms]

    rows: list[list[float]] = []
    quals: list[float] = []
    for a in active:
        q = _quality(a)
        if q is None:
            continue
        i = index[a.item_id]
        vec = [float(feats[s][i]) for s in signals]
        if any(np.isnan(v) for v in vec):
            continue  # missing signal → not usable as a clean training point
        rows.append(vec)
        quals.append(q)

    n_used = len(rows)
    distinct_levels = len(set(quals))

    def _refuse(message: str) -> RecalibrationResult:
        return RecalibrationResult(
            status="refused",
            proposed_definition=None,
            n_annotations_used=n_used,
            n_pairs=0,
            n_distinct_levels=distinct_levels,
            agreement_before=None,
            agreement_after=None,
            message=message,
        )

    if n_used < MIN_ANNOTATIONS_RECAL:
        return _refuse(f"need ≥{MIN_ANNOTATIONS_RECAL} usable annotations, have {n_used}")
    if distinct_levels < MIN_QUALITY_LEVELS_RECAL:
        return _refuse(f"need ≥{MIN_QUALITY_LEVELS_RECAL} distinct quality levels, have {distinct_levels}")

    feature_x = np.asarray(rows, dtype=float)
    quality_y = np.asarray(quals, dtype=float)

    # Build ordered pairwise differences for distinct-quality pairs (balanced).
    diffs: list[np.ndarray] = []
    labels: list[int] = []
    n_unordered = 0
    for i in range(n_used):
        for j in range(n_used):
            if i == j or quality_y[i] == quality_y[j]:
                continue
            if i < j:
                n_unordered += 1
            diffs.append(feature_x[i] - feature_x[j])
            labels.append(1 if quality_y[i] > quality_y[j] else 0)

    if not diffs:
        return _refuse("no orderable annotation pairs")

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(fit_intercept=False, C=1.0)
    model.fit(np.asarray(diffs), np.asarray(labels))
    weights = [float(w) for w in model.coef_[0]]

    proposed = MetricDefinition(
        name=f"{base_definition.name}+recal",
        terms=[
            SignalTerm(
                signal=t.signal,
                weight=weights[k],
                transform=t.transform,
                transform_params=dict(t.transform_params),
                missing=t.missing,
            )
            for k, t in enumerate(base_definition.terms)
        ],
        direction="higher_is_better",  # recalibration normalizes so higher = better quality
        combine="weighted_sum",
        notes=f"recalibrated from {n_used} annotations / {n_unordered} pairs",
    )

    # Agreement before/after on the annotated set (Spearman vs quality).
    sign = 1.0 if base_definition.direction == "higher_is_better" else -1.0
    base_weights = np.asarray([t.weight for t in base_definition.terms], dtype=float)
    goodness_before = sign * (feature_x @ base_weights)
    goodness_after = feature_x @ np.asarray(weights, dtype=float)
    before = _spearman(goodness_before, quality_y)
    after = _spearman(goodness_after, quality_y)

    status: RecalStatus = "warned" if n_unordered <= LOW_PAIR_WARN else "proposed"
    message = "" if status == "proposed" else f"low pair count ({n_unordered}); overfit risk"
    return RecalibrationResult(
        status=status,
        proposed_definition=proposed,
        n_annotations_used=n_used,
        n_pairs=n_unordered,
        n_distinct_levels=distinct_levels,
        agreement_before=before,
        agreement_after=after,
        message=message,
    )
