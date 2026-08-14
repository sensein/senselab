"""Per-file structure of an embedding set.

Within-file and cross-file dispersion are kept strictly separate because prior measurement on
this pipeline puts essentially the whole error budget cross-file: within-file cosine stability
0.984 against cross-file 0.891. A single pooled dispersion number would average those into
something uninterpretable and destroy the most informative split known about this data.
"""

import numpy as np
import pytest

from senselab.utils.tasks.embedding_distribution import describe_embedding_distribution


def _file_of_vectors(axis: np.ndarray, n: int, spread: float, seed: int) -> np.ndarray:
    """N unit vectors tightly around `axis`."""
    rng = np.random.default_rng(seed)
    x = axis[None, :] + spread * rng.normal(size=(n, axis.size))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _two_files_same_speaker(d: int = 64) -> tuple[np.ndarray, list[str]]:
    """Two files whose directions differ slightly -- one speaker, two sessions.

    The nudge has to clear the per-row noise, not just be nonzero: each row's noise vector has
    ``d`` independent components of scale ``spread``, so its norm grows as ``spread * sqrt(d)``
    (~0.16 at d=64, spread=0.02) while the nudge -- added as a single ``d``-dim unit vector scaled
    by a scalar -- does not get that boost. A nudge weight of 0.05 (this fixture's first draft)
    put the file-centroid separation *below* the per-row noise, so cross-file cosine came out
    higher than within-file and the intended inequality failed for a correct implementation.
    Verified numerically (see task-3-report.md): 0.2 is the smallest weight in
    {0.05, 0.1, 0.2, 0.3, 0.4, 0.5} that flips it; 0.3 keeps a comfortable margin without
    threatening the ``rbar > 0.95`` bound the other test on this fixture relies on.
    """
    rng = np.random.default_rng(11)
    axis = rng.normal(size=d)
    axis /= np.linalg.norm(axis)
    nudge = rng.normal(size=d)
    nudge -= (nudge @ axis) * axis
    nudge /= np.linalg.norm(nudge)

    a = _file_of_vectors(axis + 0.3 * nudge, 40, 0.02, seed=1)
    b = _file_of_vectors(axis - 0.3 * nudge, 40, 0.02, seed=2)
    return np.vstack([a, b]), ["fileA"] * 40 + ["fileB"] * 40


def test_within_file_is_reported_per_file() -> None:
    """Each file gets its own coherence figure, not a pooled one."""
    x, ids = _two_files_same_speaker()
    _, dist = describe_embedding_distribution(x, ids)

    assert set(dist.within_file) == {"fileA", "fileB"}
    assert dist.within_file["fileA"].n_vectors == 40
    assert dist.within_file["fileA"].rbar > 0.95


def test_within_file_is_tighter_than_cross_file() -> None:
    """The measured error budget on this pipeline is cross-file.

    So the two must be separable and must actually differ on data built that way.
    """
    x, ids = _two_files_same_speaker()
    _, dist = describe_embedding_distribution(x, ids)

    within = min(dist.within_file[f].cos_to_own_centroid_q50 for f in dist.within_file)
    cross = min(dist.cross_file.cos_file_centroid_to_pooled.values())
    assert within > cross


def test_cross_file_reports_each_file_centroid_against_the_pooled_one() -> None:
    """A contaminated file shows up here as a low cosine, which is what lets a caller curate."""
    d = 64
    rng = np.random.default_rng(7)
    target = rng.normal(size=d)
    target /= np.linalg.norm(target)
    intruder = rng.normal(size=d)
    intruder -= (intruder @ target) * target
    intruder /= np.linalg.norm(intruder)

    x = np.vstack(
        [
            _file_of_vectors(target, 40, 0.02, seed=1),
            _file_of_vectors(target, 40, 0.02, seed=2),
            _file_of_vectors(intruder, 40, 0.02, seed=3),
        ]
    )
    ids = ["t1"] * 40 + ["t2"] * 40 + ["bad"] * 40
    _, dist = describe_embedding_distribution(x, ids)

    assert dist.cross_file.cos_file_centroid_to_pooled["bad"] < 0.5
    assert dist.cross_file.cos_file_centroid_to_pooled["t1"] > 0.8


def test_pairwise_file_centroid_cosines_are_summarised() -> None:
    """With three or more files the pairwise spread is itself informative.

    ``-1 <= q50 <= 1`` alone holds for *any* number claiming to be a cosine, including a
    mutated implementation that ignores the real geometry and reports a constant -- mutation-
    checked directly against this fixture. fileC is built from a single realised row of fileA
    (see the line below), so its centroid over 30 fresh draws must land close to fileA's and far
    from fileB's: the min pairwise cosine (fileA/fileB or fileB/fileC) has to be visibly lower
    than the max (fileA/fileC), or the statistic is not reading the actual pair structure.
    """
    x, ids = _two_files_same_speaker()
    third = _file_of_vectors(np.asarray(x[0]), 30, 0.02, seed=4)
    x = np.vstack([x, third])
    ids = ids + ["fileC"] * 30

    _, dist = describe_embedding_distribution(x, ids)
    pairwise = dist.cross_file.file_centroid_pairwise_cos
    assert pairwise is not None
    assert -1.0 <= pairwise.q50 <= 1.0
    assert pairwise.max > 0.95  # fileA/fileC: near-duplicate centroids
    assert pairwise.min < 0.9  # fileA/fileB or fileB/fileC: the real, constructed separation


def test_a_single_file_has_no_pairwise_spread() -> None:
    """One file means no pair exists. Reporting a number would be inventing one."""
    x, _ = _two_files_same_speaker()
    _, dist = describe_embedding_distribution(x, ["only"] * x.shape[0])
    assert dist.cross_file.file_centroid_pairwise_cos is None
    assert set(dist.cross_file.cos_file_centroid_to_pooled) == {"only"}


def test_no_file_ids_leaves_the_per_file_blocks_empty() -> None:
    """Without ids there is no per-file structure to report, and inventing one would be a lie."""
    x, _ = _two_files_same_speaker()
    _, dist = describe_embedding_distribution(x)
    assert dist.within_file == {}
    assert dist.cross_file.cos_file_centroid_to_pooled == {}


def test_within_file_q05_reflects_a_tail_the_median_hides() -> None:
    """A handful of off-axis rows inside one file must show up in q05 even though the median hides them.

    Added beyond the brief: no test here exercised ``cos_to_own_centroid_q05`` at all, and
    mutation-checking confirmed the gap -- swapping the q05/q50 quantile arguments in
    ``_per_file_stats`` left every other test in this file green, because none of them read the
    q05 field. A file that is mostly tight (95 rows) with a small off-axis tail (5 rows) is built
    so the median stays high (the tail is a minority) while q05 -- the 5th percentile over 100
    rows -- lands measurably lower, since it is dominated by the tail.
    """
    d = 64
    rng = np.random.default_rng(3)
    axis = rng.normal(size=d)
    axis /= np.linalg.norm(axis)
    core = axis[None, :] + 0.02 * rng.normal(size=(95, d))
    core /= np.linalg.norm(core, axis=1, keepdims=True)

    off_axis = rng.normal(size=d)
    off_axis -= (off_axis @ axis) * axis
    off_axis /= np.linalg.norm(off_axis)
    tail = (axis[None, :] + 1.5 * off_axis[None, :]) + 0.02 * rng.normal(size=(5, d))
    tail /= np.linalg.norm(tail, axis=1, keepdims=True)

    x = np.vstack([core, tail])
    _, dist = describe_embedding_distribution(x, ["f"] * 100)
    w = dist.within_file["f"]

    assert w.cos_to_own_centroid_q05 < w.cos_to_own_centroid_q50
    assert w.cos_to_own_centroid_q05 < 0.97  # the tail pulls the 5th percentile down
    assert w.cos_to_own_centroid_q50 > 0.98  # the tight majority keeps the median high


def test_a_file_with_one_vector_still_reports_without_crashing() -> None:
    """Singleton files are ordinary in real corpora; rbar of one vector is 1.0 by definition."""
    x, ids = _two_files_same_speaker()
    x = np.vstack([x, x[:1]])
    ids = ids + ["singleton"]
    _, dist = describe_embedding_distribution(x, ids)
    assert dist.within_file["singleton"].n_vectors == 1
    assert dist.within_file["singleton"].rbar == pytest.approx(1.0)
