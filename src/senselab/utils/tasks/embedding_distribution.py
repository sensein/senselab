"""Describe one set of embedding vectors: a centroid, and statistics about the distribution.

This module **describes and never decides**. There is no verdict field, no boolean, no
probability, and no thresholded label anywhere in its output. A consumer applies its own
threshold, so every statistic here is either bounded on an interpretable scale or paired with an
analytic null it can be read against.

Every reference scale is closed-form, which is what keeps the module free of literals nobody
fitted:

- sd of pairwise cosines between independent directions: ``1/sqrt(d)`` (0.0722 at d=192)
- mean resultant length of ``n`` independent directions: ``E[Rbar^2] = 1/n``, so ``1/sqrt(n)``
- participation ratio under Marchenko-Pastur: ``d*n/(d+n)``
- Mann-Whitney AUC under exchangeability: exactly ``0.5``

**The counter-intuitive one.** A *small* sd of cosines is not evidence of a coherent speaker. At
d=192 independent random directions give sd ~= 0.072, so an observed 0.05 is *below* the
random-vector null. sd is therefore never reported as a headline dispersion figure -- only beside
``nulls.cos_sd_null``.

Geometry: vectors are L2-normalised on entry, unconditionally. SpeechBrain speaker embeddings are
not unit norm, and the norm covaries with window energy and how much speech fills the window -- a
cough, or 0.4 s of speech in a 2.0 s window, gets a systematically different norm. Any
unnormalised statistic would mix that loudness/occupancy nuisance into what a reader takes for
speaker dispersion. ECAPA is trained with an angular-margin objective and scored by cosine, so
the norm is not part of the discriminative geometry to begin with.

After normalisation cosine and Euclidean are the *same* geometry --
``||x-y||^2 = 2(1-cos t)``, a strictly monotone reparametrisation -- so the common "Euclidean is
unusable at high d" objection does not apply to any rank- or neighbour-based quantity. Where a
true metric is needed (medoid, linkage) the geodesic ``arccos(clip(cos,-1,1))`` is used, because
``cos`` is not a metric and neither is ``1-cos``.

**Deliberately absent**, each for a mechanical reason rather than taste:

- *Silhouette.* ``silhouette(metric="cosine")`` and ``silhouette(metric="euclidean")`` return
  different numbers for identical geometry on unit vectors (Jensen: the square root compresses
  large distances more than small ones), so any threshold on silhouette is a threshold on a
  parameterisation choice. It is also a property of a chosen partition, not of the data.
- *k-NN purity.* Hubness is severe at this dimension -- k-occurrence skew ~3.3 at d=192, with a
  measurable fraction of points appearing in no neighbour list -- so neighbour counts carry a bias
  unrelated to speaker identity. Worse, at 50% window overlap a window's nearest neighbour is
  almost always the temporally adjacent window, so any same-file purity statistic would read ~1.0
  for every input: it would measure the hop size.
- *Intrinsic-dimensionality estimators.* Two-NN's ``r1`` becomes the distance to a near-duplicate
  window, driving the estimate down by the hop size. Same artefact.
- *von Mises-Fisher concentration.* ``kappa = Rbar(d - Rbar^2)/(1 - Rbar^2)`` is a deterministic
  function of ``Rbar`` and ``d``, both reported, so it stores nothing new; it is unbounded as
  ``Rbar -> 1``, so it is not interpretable on its own scale; and vMF assumes isotropic
  concentration, which embedding spaces violate. Recover it in one line if you want it.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import numpy as np
from pydantic import BaseModel, Field

AGGREGATOR_SPHERICAL_MEAN = "spherical_mean"
AGGREGATOR_TRIMMED_MEAN = "trimmed_mean"
AGGREGATOR_MEDOID = "medoid"
_AGGREGATORS = (AGGREGATOR_SPHERICAL_MEAN, AGGREGATOR_TRIMMED_MEAN, AGGREGATOR_MEDOID)

# The fraction trimmed by the trimmed-mean aggregator and by the robustness diagnostic. Not a
# decision threshold: nothing is accepted or rejected on it, and both the trimmed result and its
# cosine to the untrimmed mean are reported, so a reader sees what trimming did rather than
# inheriting its verdict.
_TRIM_FRACTION = 0.10


class SimilarityStats(BaseModel):
    """Quantiles of a set of cosine values, plus mean and sd.

    Quantiles rather than mean-and-sd alone because contamination makes this distribution a
    *mixture*: mean +/- sd of a bimodal mixture describes neither lobe, and the sd is inflated by
    exactly the thing worth exposing -- so sd alone cannot separate "one loose speaker" from "two
    tight speakers". ``q05`` and ``min`` are where an intruder shows up.
    """

    min: float
    q05: float
    q25: float
    q50: float
    q75: float
    q95: float
    max: float
    mean: float
    sd: float


class GeometryInfo(BaseModel):
    """What was done to the vectors, so a stored block can be interpreted later."""

    metric: str
    l2_normalised: bool
    dim: int
    distance: str
    centroid_rule: str


class CountsInfo(BaseModel):
    """Sizes, and the effective sample size after accounting for window overlap.

    Attributes:
        n_effective: ``total_windowed_duration / window_s``, which is about ``n/2`` at a 2.0 s
            window on a 1.0 s hop. ``None`` when window information was not supplied -- it cannot
            be derived from vectors alone, and a guess would be worse than an absence. Any null
            whose width scales as ``n^-1/2`` is about sqrt(2) overconfident without this discount.
    """

    n_vectors_total: int
    n_scored: int
    n_zero_norm_dropped: int
    n_files: int
    vectors_per_file: dict[str, int] = Field(default_factory=dict)
    window_s: Optional[float] = None
    hop_s: Optional[float] = None
    n_effective: Optional[float] = None


class NullsInfo(BaseModel):
    """Closed-form reference scales, so no statistic needs a fitted threshold.

    ``dim`` and ``n_scored`` are in ``CountsInfo`` and ``GeometryInfo`` specifically so a consumer
    can recompute every one of these and check ours.
    """

    cos_sd_null: float
    rbar_null: float
    participation_ratio_null: float
    auc_null: float


class SpectrumStats(BaseModel):
    """How many directions the set actually occupies.

    Attributes:
        participation_ratio: ``(sum lambda)^2 / sum lambda^2`` of the centred covariance, in
            ``[1, min(n,d)]``. Read against ``nulls.participation_ratio_null``: well below it means
            a genuinely low-dimensional set, near it means indistinguishable from white noise at
            this sample size.
        pc1_share_centred: ``lambda_1 / sum lambda`` on **centred** data. Uncentred, PC1 is the
            mean direction and explains almost everything for any coherent set, which would make
            this a field that always reads the same. Centred, a high share is the signature of
            bimodality.
        eigenvalue_shares_top5: The five largest ``lambda_i / sum lambda``, zero-padded when
            ``min(n,d) < 5``.
    """

    participation_ratio: float
    pc1_share_centred: float
    eigenvalue_shares_top5: list[float]


class WithinFileStats(BaseModel):
    """Coherence inside one file.

    Attributes:
        n_vectors: Rows from this file that survived normalisation.
        rbar: Mean resultant length of this file's rows. ``1.0`` for a single row, by definition.
        cos_to_own_centroid_q05: 5th percentile of cosine to *this file's* centroid.
        cos_to_own_centroid_q50: Median cosine to this file's centroid.
    """

    n_vectors: int
    rbar: float
    cos_to_own_centroid_q05: float
    cos_to_own_centroid_q50: float


class CrossFileStats(BaseModel):
    """How the files sit relative to each other and to the pooled centroid.

    Attributes:
        cos_file_centroid_to_pooled: Per file, the cosine of its own centroid to the pooled
            centroid. A contaminated file shows up here directly, which is what lets a caller
            curate its input without any clustering.
        file_centroid_pairwise_cos: Quantiles of the pairwise cosines between file centroids.
            ``None`` with fewer than two files, because no pair exists and a reported number would
            be invented.
    """

    cos_file_centroid_to_pooled: dict[str, float] = Field(default_factory=dict)
    file_centroid_pairwise_cos: Optional[SimilarityStats] = None


class EmbeddingDistribution(BaseModel):
    """Statistics describing one set of embedding vectors. Contains no verdict."""

    geometry: GeometryInfo
    counts: CountsInfo
    nulls: NullsInfo
    cos_to_centroid_loo: SimilarityStats
    rbar: float
    spectrum: SpectrumStats
    # Kept as two separate blocks rather than one pooled dispersion figure: prior measurement on
    # this pipeline puts essentially the whole error budget cross-file (within-file cosine
    # stability 0.984 against cross-file 0.891), so averaging the two would destroy the most
    # informative split known about this data. See `_per_file_stats`.
    within_file: dict[str, WithinFileStats] = Field(default_factory=dict)
    cross_file: CrossFileStats = Field(default_factory=CrossFileStats)


def _as_array(
    vectors: Union[Sequence[Sequence[float]], np.ndarray, Any],  # noqa: ANN401 — torch.Tensor duck-typed
) -> np.ndarray:
    """Coerce input to a 2-D float64 array without importing torch at module scope.

    Args:
        vectors: A nested sequence, a numpy array, or anything exposing ``detach``/``cpu``/``numpy``
            (a torch tensor).

    Returns:
        A 2-D ``float64`` array, one row per vector.

    Raises:
        ValueError: If the result is not 2-D.
    """
    if hasattr(vectors, "detach"):
        vectors = vectors.detach().cpu().numpy()
    arr = np.asarray(vectors, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"vectors must be 2-D (n, d); got shape {arr.shape}")
    return arr


def _l2_normalise(x: np.ndarray) -> tuple[np.ndarray, int]:
    """Return unit-norm rows and the count of zero-norm rows removed.

    Args:
        x: ``(n, d)`` array.

    Returns:
        ``(normalised, n_dropped)``. A zero-norm row has no direction, so it cannot contribute to
        any angular statistic; it is dropped rather than producing ``nan``, and counted so
        ``n_scored`` stays explainable.
    """
    norms = np.linalg.norm(x, axis=1)
    keep = norms > 0
    n_dropped = int((~keep).sum())
    return x[keep] / norms[keep][:, None], n_dropped


def _spherical_mean(x: np.ndarray) -> np.ndarray:
    """Unit-norm mean direction of unit-norm rows.

    This is the von Mises-Fisher MLE direction, and its error shrinks as ``O(n^-1/2)``. An
    arithmetic mean of *unnormalised* vectors would weight each row by its norm -- i.e. by
    loudness and speech occupancy -- so a loud cough would outvote a quiet target utterance.

    Args:
        x: ``(n, d)`` unit-norm array.

    Returns:
        A unit-norm ``(d,)`` direction.

    Raises:
        ValueError: If the rows sum to the zero vector, which has no direction.
    """
    s = x.sum(axis=0)
    norm = float(np.linalg.norm(s))
    if norm == 0.0:
        raise ValueError("vectors sum to zero; no mean direction exists")
    return s / norm


def _similarity_stats(values: np.ndarray) -> SimilarityStats:
    """Summarise a 1-D array of cosine values.

    Args:
        values: Any 1-D array of cosines.

    Returns:
        Quantiles plus mean and sd. ``sd`` is included for completeness but is meaningless without
        ``nulls.cos_sd_null`` beside it; see the module docstring.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    q = np.quantile(v, [0.05, 0.25, 0.50, 0.75, 0.95])
    return SimilarityStats(
        min=float(v.min()),
        q05=float(q[0]),
        q25=float(q[1]),
        q50=float(q[2]),
        q75=float(q[3]),
        q95=float(q[4]),
        max=float(v.max()),
        mean=float(v.mean()),
        sd=float(v.std(ddof=1)) if v.size > 1 else 0.0,
    )


def _loo_cos_to_centroid(x: np.ndarray) -> np.ndarray:
    """Cosine of each row to the centroid computed *without* that row.

    Scoring a vector against a centroid it helped define is optimistically biased -- each row pulls
    the centroid toward itself. Closed form, one pass, no loop: with ``S = sum x_j`` over unit
    rows, the leave-one-out mean direction is proportional to ``S - x_i``, so
    ``cos_loo(i) = x_i . (S - x_i) / ||S - x_i||``.

    Args:
        x: ``(n, d)`` unit-norm array with ``n >= 2``.

    Returns:
        ``(n,)`` array of leave-one-out cosines.
    """
    s = x.sum(axis=0)
    diff = s[None, :] - x
    denom = np.linalg.norm(diff, axis=1)
    numer = np.einsum("ij,ij->i", x, diff)
    out = np.zeros_like(denom)
    nz = denom > 0
    out[nz] = numer[nz] / denom[nz]
    return out


def _spectrum(x: np.ndarray) -> SpectrumStats:
    """Participation ratio, centred PC1 share, and the top-5 eigenvalue shares.

    Uses singular values of the centred matrix rather than forming the ``d x d`` covariance: both
    quantities are ratios, so the ``1/(n-1)`` factor cancels and the SVD is cheaper and better
    conditioned.

    Args:
        x: ``(n, d)`` unit-norm array.

    Returns:
        The spectrum summary.
    """
    centred = x - x.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(centred, compute_uv=False)
    lam = sv**2
    total = float(lam.sum())
    if total == 0.0:
        # Every row identical: no variation at all. One occupied direction, by definition.
        return SpectrumStats(participation_ratio=1.0, pc1_share_centred=0.0, eigenvalue_shares_top5=[0.0] * 5)
    pr = float(total**2 / float((lam**2).sum()))
    shares = (lam / total).tolist()
    top5 = [float(v) for v in shares[:5]] + [0.0] * max(0, 5 - len(shares))
    return SpectrumStats(participation_ratio=pr, pc1_share_centred=float(shares[0]), eigenvalue_shares_top5=top5)


def _per_file_stats(
    x: np.ndarray, file_ids: Optional[list[str]], pooled_centroid: np.ndarray
) -> tuple[dict[str, WithinFileStats], CrossFileStats]:
    """Within-file coherence and cross-file agreement.

    Kept strictly apart because the measured error budget on this pipeline is almost entirely
    cross-file (within-file cosine stability 0.984 against cross-file 0.891). Pooling them would
    average away the most informative split there is.

    Args:
        x: ``(n, d)`` unit-norm array.
        file_ids: One id per row, or ``None``.
        pooled_centroid: The centroid over all rows.

    Returns:
        ``(within_file, cross_file)``. Both are empty when ``file_ids`` is ``None``.
    """
    if file_ids is None:
        return {}, CrossFileStats()

    order: list[str] = []
    for f in file_ids:
        if f not in order:
            order.append(f)

    within: dict[str, WithinFileStats] = {}
    centroids: dict[str, np.ndarray] = {}
    ids = np.asarray(file_ids)
    for f in order:
        rows = x[ids == f]
        c = _spherical_mean(rows)
        centroids[f] = c
        cos_own = rows @ c
        within[f] = WithinFileStats(
            n_vectors=int(rows.shape[0]),
            rbar=float(np.linalg.norm(rows.sum(axis=0)) / rows.shape[0]),
            cos_to_own_centroid_q05=float(np.quantile(cos_own, 0.05)),
            cos_to_own_centroid_q50=float(np.quantile(cos_own, 0.50)),
        )

    to_pooled = {f: float(centroids[f] @ pooled_centroid) for f in order}

    pairwise: Optional[SimilarityStats] = None
    if len(order) >= 2:
        stacked = np.stack([centroids[f] for f in order])
        gram = stacked @ stacked.T
        iu = np.triu_indices(len(order), k=1)
        pairwise = _similarity_stats(gram[iu])

    return within, CrossFileStats(cos_file_centroid_to_pooled=to_pooled, file_centroid_pairwise_cos=pairwise)


def describe_embedding_distribution(
    vectors: Union[Sequence[Sequence[float]], np.ndarray, Any],  # noqa: ANN401 — torch.Tensor duck-typed
    file_ids: Optional[Sequence[str]] = None,
    *,
    aggregator: str = AGGREGATOR_SPHERICAL_MEAN,
    window_s: Optional[float] = None,
    hop_s: Optional[float] = None,
    window_starts_s: Optional[Sequence[float]] = None,
    n_permutations: int = 1000,
    seed: int = 0,
) -> tuple[list[float], EmbeddingDistribution]:
    """Describe one set of embedding vectors: a centroid and statistics about the distribution.

    Decides nothing. See the module docstring for the geometry, the analytic nulls, and what is
    deliberately absent.

    Args:
        vectors: ``(n, d)`` embeddings. Normalised on entry regardless of input scale.
        file_ids: One id per row, when the vectors come from several files. Enables the per-file
            statistics; ``None`` treats the set as one group.
        aggregator: ``"spherical_mean"`` (default), ``"trimmed_mean"``, or ``"medoid"``. A tool
            parameter, not a decision: the returned block always reports the cosine between the
            mean and both alternatives, so a reader sees whether the choice mattered.
        window_s: Window length in seconds, used for ``n_effective`` and the permutation block
            length. ``None`` leaves both unreported rather than guessed.
        hop_s: Hop between windows in seconds.
        window_starts_s: Start time of each window, same order as ``vectors``. Required for the
            same-file guard band; without it that guard is skipped and reported as ``None``.
        n_permutations: Permutations for the file-effect reference.
        seed: Seed for the permutation. Fixed by default so the reported quantile is reproducible.

    Returns:
        ``(centroid, distribution)``. The centroid is a unit-norm ``list[float]``.

    Raises:
        ValueError: If fewer than 2 vectors survive normalisation, if ``vectors`` is not 2-D, if
            ``aggregator`` is unknown, or if ``file_ids``/``window_starts_s`` lengths disagree with
            ``vectors``.
    """
    if aggregator not in _AGGREGATORS:
        raise ValueError(f"aggregator must be one of {_AGGREGATORS}; got {aggregator!r}")

    raw = _as_array(vectors)
    n_total = int(raw.shape[0])
    if file_ids is not None and len(file_ids) != n_total:
        raise ValueError(f"file_ids has {len(file_ids)} entries for {n_total} vectors")
    if window_starts_s is not None and len(window_starts_s) != n_total:
        raise ValueError(f"window_starts_s has {len(window_starts_s)} entries for {n_total} vectors")

    norms = np.linalg.norm(raw, axis=1)
    keep_mask = norms > 0
    x, n_dropped = _l2_normalise(raw)
    n = int(x.shape[0])
    if n < 2:
        raise ValueError(f"need at least 2 non-zero vectors to describe a distribution; got {n}")
    d = int(x.shape[1])

    kept_files = [str(f) for f, k in zip(file_ids, keep_mask) if k] if file_ids is not None else None

    centroid = _spherical_mean(x)  # Tasks 4-5 replace this with the aggregator dispatch.

    within_file, cross_file = _per_file_stats(x, kept_files, centroid)

    per_file: dict[str, int] = {}
    if kept_files is not None:
        for f in kept_files:
            per_file[f] = per_file.get(f, 0) + 1

    n_effective: Optional[float] = None
    if window_s is not None and hop_s is not None and window_s > 0:
        # total covered duration / window_s: overlapping windows do not carry independent
        # information, and pretending they do makes every n^-1/2 null overconfident.
        n_effective = float((hop_s * (n - 1) + window_s) / window_s)

    return centroid.tolist(), EmbeddingDistribution(
        geometry=GeometryInfo(
            metric="cosine",
            l2_normalised=True,
            dim=d,
            distance="angular",
            centroid_rule=aggregator,
        ),
        counts=CountsInfo(
            n_vectors_total=n_total,
            n_scored=n,
            n_zero_norm_dropped=n_dropped,
            n_files=len(per_file) if per_file else (1 if kept_files is None else 0),
            vectors_per_file=per_file,
            window_s=window_s,
            hop_s=hop_s,
            n_effective=n_effective,
        ),
        nulls=NullsInfo(
            cos_sd_null=float(1.0 / np.sqrt(d)),
            rbar_null=float(1.0 / np.sqrt(n)),
            participation_ratio_null=float(d * n / (d + n)),
            auc_null=0.5,
        ),
        cos_to_centroid_loo=_similarity_stats(_loo_cos_to_centroid(x)),
        rbar=float(np.linalg.norm(x.sum(axis=0)) / n),
        spectrum=_spectrum(x),
        within_file=within_file,
        cross_file=cross_file,
    )
