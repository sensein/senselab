"""Core statistics of one set of embedding vectors.

Every reference scale here is analytic, so these tests check the statistics against closed forms
rather than against recorded numbers. That is the point: a fitted literal would have to be
measured and maintained, while 1/sqrt(d) is true by construction.
"""

import numpy as np
import pytest

from senselab.utils.tasks.embedding_distribution import describe_embedding_distribution


def _tight_cone(n: int, d: int, spread: float, seed: int = 0, rank: int = 5) -> np.ndarray:
    """N vectors clustered around one random direction, with angular spread confined to `rank` axes.

    Noise isotropic across all d dimensions would leave the centred participation ratio pinned at
    its Marchenko-Pastur null no matter how small ``spread`` is -- PR is scale-invariant, so an
    isotropic cone can be made arbitrarily tight (high rbar) while its residual still spans a
    full-rank, noise-like shape. Confirmed numerically down to spread=1e-4 at d=192, n=400: PR
    never moved off ~129.7, the same value uniform random vectors give. A cluster that genuinely
    "occupies few directions" needs the noise itself to live in a low-rank subspace, not just be
    small. ``rank`` is parameterised (not fixed at 5) so a caller can also use this fixture to pin
    the participation-ratio formula against a *known* value: with exactly ``rank`` roughly-equal
    residual eigenvalues, PR should read close to ``rank`` itself, not merely "below something".
    """
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=d)
    axis /= np.linalg.norm(axis)
    rank = min(rank, d - 1)
    basis = rng.normal(size=(rank, d))
    basis /= np.linalg.norm(basis, axis=1, keepdims=True)
    coeffs = rng.normal(size=(n, rank))
    x = axis[None, :] + spread * (coeffs @ basis)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_uniform_random_vectors_land_on_the_analytic_nulls() -> None:
    """Random directions must reproduce the closed forms, or the nulls are wrong.

    This is the test that keeps the block free of fitted numbers: sd of pairwise cosines -> 1/sqrt(d),
    mean resultant length -> 1/sqrt(n), participation ratio -> d*n/(d+n).
    """
    n, d = 800, 192
    rng = np.random.default_rng(17)
    x = rng.normal(size=(n, d))
    x /= np.linalg.norm(x, axis=1, keepdims=True)

    _, dist = describe_embedding_distribution(x)

    assert dist.nulls.cos_sd_null == pytest.approx(1.0 / np.sqrt(d))
    assert dist.nulls.rbar_null == pytest.approx(1.0 / np.sqrt(n))
    assert dist.nulls.participation_ratio_null == pytest.approx(d * n / (d + n))
    assert dist.nulls.auc_null == 0.5

    # And the data actually sits near them.
    assert dist.rbar == pytest.approx(1.0 / np.sqrt(n), abs=0.02)
    assert dist.spectrum.participation_ratio == pytest.approx(d * n / (d + n), rel=0.1)


def test_a_tight_cone_has_high_rbar_and_low_effective_rank() -> None:
    """One coherent speaker points one way and occupies few directions."""
    x = _tight_cone(n=400, d=192, spread=0.15)
    _, dist = describe_embedding_distribution(x)

    assert dist.rbar > 0.9
    assert dist.rbar > 10 * dist.nulls.rbar_null
    # Rank-5 noise forces rank(centred) <= 5 by construction, so this bound only checks that PR
    # counts nonzero singular values roughly right, not that it combines them correctly -- an
    # inverted or mis-squared formula would likely still clear it. See
    # test_participation_ratio_matches_known_closed_form_rank for the assertion that actually
    # pins the formula.
    assert dist.spectrum.participation_ratio < 0.5 * dist.nulls.participation_ratio_null


@pytest.mark.parametrize("k", [3, 8])
def test_participation_ratio_matches_known_closed_form_rank(k: int) -> None:
    """A residual with k roughly-equal non-zero eigenvalues must give PR close to k, exactly.

    ``test_a_tight_cone_has_high_rbar_and_low_effective_rank`` only bounds participation_ratio
    from above (``< 0.5 * null``), and a rank-5 residual satisfies that bound almost regardless of
    whether the formula is right -- the bound constrains the *count* of nonzero singular values,
    not their combination. This pins the value instead: for a centred covariance with exactly k
    equal eigenvalues, ``PR = (sum lambda)^2 / sum(lambda^2) = k`` exactly, so confining the
    residual to k roughly-equal-variance directions gives a number the formula has to *hit*, which
    catches an inverted, mis-squared, or accidentally-uncentred computation that the looser bound
    would miss.
    """
    x = _tight_cone(n=400, d=192, spread=0.15, rank=k, seed=k)
    _, dist = describe_embedding_distribution(x)
    assert dist.spectrum.participation_ratio == pytest.approx(k, rel=0.15)


def test_the_centroid_is_unit_norm() -> None:
    """The returned vector is a direction. Callers compare it by cosine."""
    centroid, _ = describe_embedding_distribution(_tight_cone(n=50, d=64, spread=0.2))
    assert np.linalg.norm(np.asarray(centroid)) == pytest.approx(1.0)


def test_zero_norm_rows_are_dropped_and_counted() -> None:
    """A zero vector has no direction, so it cannot contribute.

    But silence about it would make n_scored unexplainable.
    """
    x = _tight_cone(n=20, d=32, spread=0.1)
    x = np.vstack([x, np.zeros((3, 32))])
    _, dist = describe_embedding_distribution(x)

    assert dist.counts.n_vectors_total == 23
    assert dist.counts.n_scored == 20
    assert dist.counts.n_zero_norm_dropped == 3


def test_input_is_normalised_even_when_the_caller_did_not() -> None:
    """ECAPA embeddings are not unit norm, and the norm covaries with window occupancy.

    Left alone it would inject a loudness nuisance into every statistic, so normalisation is
    unconditional and the block says so.
    """
    x = _tight_cone(n=40, d=64, spread=0.1)
    scaled = x * np.linspace(0.1, 50.0, x.shape[0])[:, None]

    c_plain, d_plain = describe_embedding_distribution(x)
    c_scaled, d_scaled = describe_embedding_distribution(scaled)

    assert d_scaled.geometry.l2_normalised is True
    assert np.allclose(c_plain, c_scaled, atol=1e-6)
    assert d_scaled.rbar == pytest.approx(d_plain.rbar, abs=1e-6)


def test_loo_cosines_match_a_naive_recomputation() -> None:
    """Scoring a vector against a centroid it helped define is optimistically biased.

    The closed form x_i . (S - x_i) / ||S - x_i|| must equal recomputing the centroid without i,
    which this checks directly on a small input.
    """
    x = _tight_cone(n=12, d=16, spread=0.3, seed=3)
    _, dist = describe_embedding_distribution(x)

    naive = []
    for i in range(x.shape[0]):
        others = np.delete(x, i, axis=0)
        c = others.sum(axis=0)
        c /= np.linalg.norm(c)
        naive.append(float(x[i] @ c))
    naive_arr = np.sort(np.asarray(naive))

    assert dist.cos_to_centroid_loo.min == pytest.approx(naive_arr.min(), abs=1e-9)
    assert dist.cos_to_centroid_loo.max == pytest.approx(naive_arr.max(), abs=1e-9)
    assert dist.cos_to_centroid_loo.q50 == pytest.approx(float(np.quantile(naive_arr, 0.5)), abs=1e-9)


def test_n_effective_discounts_overlapping_windows() -> None:
    """At a 2.0 s window on a 1.0 s hop, adjacent windows share half their audio.

    So independent information is about n/2. Reporting it lets a consumer discount nulls that
    scale as n^-1/2 instead of us pretending independence.
    """
    x = _tight_cone(n=100, d=32, spread=0.2)
    _, dist = describe_embedding_distribution(x, window_s=2.0, hop_s=1.0)
    assert dist.counts.n_effective == pytest.approx(50.0, rel=0.05)


def test_n_effective_is_none_without_window_information() -> None:
    """It cannot be derived from vectors alone, and a guessed value would be worse than none."""
    _, dist = describe_embedding_distribution(_tight_cone(n=10, d=8, spread=0.1))
    assert dist.counts.n_effective is None


def test_pc1_share_is_computed_on_centred_data() -> None:
    """Centring must actually run, or PC1 share reads near-1.0 for any coherent cluster regardless of shape.

    Uncentred, PC1 of a tight, unimodal cluster *is* the mean direction and swallows nearly all the
    variance -- a field that would read the same whether the residual is isotropic or structured,
    so on its own it carries no information. Centred, that dominant direction is subtracted out and
    what is left is the residual: an isotropic residual spreads its variance over ~d-1 directions
    roughly equally, so a correctly centred PC1 share is small. A dropped centring step would
    report the *uncentred* number here instead of the centred one -- both cleared a bare ``> 0.8``
    bound on an earlier (bimodal) fixture for this test, which is why this version computes the
    uncentred share explicitly with numpy and asserts the reported centred share sits far below it,
    rather than asserting either number in isolation.
    """
    n, d = 400, 192
    rng = np.random.default_rng(5)
    axis = rng.normal(size=d)
    axis /= np.linalg.norm(axis)
    x = axis[None, :] + 0.02 * rng.normal(size=(n, d))
    x /= np.linalg.norm(x, axis=1, keepdims=True)

    sv_uncentred = np.linalg.svd(x, compute_uv=False)
    lam_uncentred = sv_uncentred**2
    uncentred_share = float(lam_uncentred[0] / lam_uncentred.sum())

    _, dist = describe_embedding_distribution(x)

    assert uncentred_share > 0.8  # the mean direction dominates, as it does for any tight cluster
    assert dist.spectrum.pc1_share_centred < 0.1 * uncentred_share
    assert len(dist.spectrum.eigenvalue_shares_top5) == 5


def test_geometry_records_what_was_done() -> None:
    """A consumer reading a stored block has to know the geometry to interpret any number in it."""
    _, dist = describe_embedding_distribution(_tight_cone(n=10, d=8, spread=0.1))
    assert dist.geometry.metric == "cosine"
    assert dist.geometry.distance == "angular"
    assert dist.geometry.dim == 8
    assert dist.geometry.centroid_rule == "spherical_mean"


def test_too_few_vectors_raises_rather_than_returning_a_meaningless_block() -> None:
    """One vector has no distribution. Returning a block of zeros would look like a measurement."""
    with pytest.raises(ValueError, match="at least 2"):
        describe_embedding_distribution(np.ones((1, 8)))
