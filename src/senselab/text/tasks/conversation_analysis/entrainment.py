"""Turn-level entrainment: proximity, convergence and synchrony with a shuffled-surrogate baseline."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from senselab.text.tasks.conversation_analysis.data_structures import EntrainmentResult


def _pearson(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if len(x) < 3 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _p_value(observed: Optional[float], surrogates: List[float], two_sided: bool) -> Optional[float]:
    if observed is None or not surrogates:
        return None
    s = np.asarray(surrogates)
    hits = np.abs(s) >= abs(observed) - 1e-12 if two_sided else s >= observed - 1e-12
    return float((1 + int(hits.sum())) / (len(s) + 1))


def _mean(values: List[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def compute_entrainment(
    turn_features: pd.DataFrame,
    features: Optional[Sequence[str]] = None,
    *,
    speaker_col: str = "speaker",
    time_col: str = "start",
    n_surrogates: int = 1000,
    seed: int = 0,
) -> List[EntrainmentResult]:
    """Entrainment of per-turn feature series between speakers (Levitan & Hirschberg 2011 style).

    Rows are turns, ordered by ``time_col``. An adjacent pair is two consecutive rows by different
    speakers: the earlier is the partner's turn ``y_{t-1}``, the later the target's turn ``x_t``.

    * proximity: mean over pairs of (mean ``|x_t - y_j|`` over the partner's other turns) minus
      ``|x_t - y_{t-1}|``; positive when a turn is closer to the turn it answers than to the partner's
      turns in general.
    * convergence: Pearson r between ``|x_t - y_{t-1}|`` and the target turn's time; negative when the
      speakers grow more alike.
    * synchrony: Pearson r between ``x_t`` and ``y_{t-1}``.

    Surrogates re-pair targets with randomly permuted partner values (proximity, synchrony) or permute
    the differences against time (convergence). Rows with a missing value are dropped per feature
    before pairs are formed.

    Args:
        turn_features: One row per turn with ``speaker_col``, ``time_col`` and numeric feature columns.
        features: Feature columns; every numeric column except ``time_col`` when omitted.
        speaker_col: Speaker column.
        time_col: Time column used for ordering and convergence.
        n_surrogates: Surrogate shuffles per statistic (0 disables surrogates).
        seed: Seed of the shuffles.

    Returns:
        For each feature, one pooled result (``target_speaker="all"``) and one per responding speaker.
    """
    if features is None:
        features = [
            c
            for c in turn_features.select_dtypes("number").columns
            if c not in (time_col, speaker_col, "turn_id", "id")
        ]
    rng = np.random.default_rng(seed)
    results: List[EntrainmentResult] = []
    for feature in features:
        frame = (
            turn_features[[speaker_col, time_col, feature]]
            .dropna()
            .sort_values(time_col, kind="stable")
            .reset_index(drop=True)
        )
        speakers = frame[speaker_col].astype(str).to_numpy()
        values = frame[feature].astype(float).to_numpy()
        times = frame[time_col].astype(float).to_numpy()
        pair_rows = [i for i in range(1, len(frame)) if speakers[i] != speakers[i - 1]]
        targets = ["all"] + sorted({speakers[i] for i in pair_rows})
        for target in targets:
            rows = [i for i in pair_rows if target == "all" or speakers[i] == target]
            results.append(_entrain(feature, target, rows, speakers, values, times, n_surrogates, rng))
    return results


def _entrain(
    feature: str,
    target: str,
    rows: List[int],
    speakers: np.ndarray,
    values: np.ndarray,
    times: np.ndarray,
    n_surrogates: int,
    rng: np.random.Generator,
) -> EntrainmentResult:
    if not rows:
        return EntrainmentResult(feature=feature, target_speaker=target, n_pairs=0)
    x = values[rows]
    y = np.array([values[i - 1] for i in rows])
    t = times[rows]
    d_adj = np.abs(x - y)
    nonadj: List[Optional[float]] = []
    for i in rows:
        partner = speakers[i - 1]
        others = [values[j] for j in range(len(values)) if speakers[j] == partner and j != i - 1]
        nonadj.append(float(np.mean(np.abs(values[i] - np.asarray(others)))) if others else None)
    has_base = np.array([v is not None for v in nonadj])
    base = np.array([v for v in nonadj if v is not None], dtype=float)

    def proximity_of(adjacent: np.ndarray) -> Optional[float]:
        if not has_base.any():
            return None
        return float(np.mean(base - adjacent[has_base]))

    proximity = proximity_of(d_adj)
    convergence = _pearson(d_adj, t)
    synchrony = _pearson(y, x)
    prox_s: List[float] = []
    conv_s: List[float] = []
    sync_s: List[float] = []
    for _ in range(n_surrogates if len(rows) > 1 else 0):
        y_perm = rng.permutation(y)
        p = proximity_of(np.abs(x - y_perm))
        if p is not None:
            prox_s.append(p)
        c = _pearson(rng.permutation(d_adj), t)
        if c is not None:
            conv_s.append(c)
        s = _pearson(y_perm, x)
        if s is not None:
            sync_s.append(s)
    return EntrainmentResult(
        feature=feature,
        target_speaker=target,
        n_pairs=len(rows),
        adjacent_diff_mean=float(d_adj.mean()),
        nonadjacent_diff_mean=float(base.mean()) if base.size else None,
        proximity=proximity,
        proximity_surrogate_mean=_mean(prox_s),
        proximity_p=_p_value(proximity, prox_s, two_sided=False),
        convergence_r=convergence,
        convergence_surrogate_mean=_mean(conv_s),
        convergence_p=_p_value(convergence, conv_s, two_sided=True),
        synchrony_r=synchrony,
        synchrony_surrogate_mean=_mean(sync_s),
        synchrony_p=_p_value(synchrony, sync_s, two_sided=True),
    )
