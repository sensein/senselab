"""Optional contamination rejection.

This is the one component in the feature that makes a decision, so it is opt-in and everything it
did is recorded. The cut has no numeric default: it is either supplied by the caller or derived
from the data by a stated rule, and whichever was used is reported.
"""

import numpy as np
import pytest

from senselab.utils.tasks.embedding_distribution import select_dominant_vectors


def _cone(axis: np.ndarray, n: int, spread: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = axis[None, :] + spread * rng.normal(size=(n, axis.size))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _two_speakers(n_target: int = 60, n_intruder: int = 20, d: int = 48) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(13)
    a = rng.normal(size=d)
    a /= np.linalg.norm(a)
    b = rng.normal(size=d)
    b -= (b @ a) * a
    b /= np.linalg.norm(b)
    x = np.vstack([_cone(a, n_target, 0.03, 1), _cone(b, n_intruder, 0.03, 2)])
    ids = ["target"] * n_target + ["intruder"] * n_intruder
    return x, ids


def test_the_intruder_group_is_dropped() -> None:
    """The measured property this exists for: a contaminating recording leaves the estimate."""
    x, ids = _two_speakers()
    sel = select_dominant_vectors(x, ids)

    assert len(sel.kept_indices) == 60
    assert len(sel.dropped_indices) == 20
    assert sel.dropped_per_file == {"intruder": 20}


def test_no_numeric_cut_default_exists() -> None:
    """A fitted literal here is exactly what this repository forbids.

    The signature must not carry one: the cut is caller-supplied or derived by a stated rule.
    """
    import inspect

    default = inspect.signature(select_dominant_vectors).parameters["cut_theta"].default
    assert default is None


def test_the_derived_cut_is_recorded_as_derived() -> None:
    """Auditable means a reader can tell where the number came from."""
    x, ids = _two_speakers()
    sel = select_dominant_vectors(x, ids)
    assert sel.rule_used.cut_source == "largest_merge_gap"
    assert sel.rule_used.cut_theta > 0
    assert len(sel.rule_used.merge_heights) == x.shape[0] - 1


def test_an_explicit_cut_overrides_and_is_recorded_verbatim() -> None:
    """A caller who disagrees with the rule must be able to say so, and see that it took."""
    x, ids = _two_speakers()
    sel = select_dominant_vectors(x, ids, cut_theta=3.0)  # larger than pi: one cluster
    assert sel.rule_used.cut_source == "caller"
    assert sel.rule_used.cut_theta == 3.0
    assert len(sel.clusters) == 1
    assert sel.dropped_indices == []


def test_selection_is_deterministic() -> None:
    """Shares are a reported field, so they must be reproducible.

    AHC takes no seed; spectral clustering with k-means assignment would, and pinning it only
    hides the variance.
    """
    x, ids = _two_speakers()
    a = select_dominant_vectors(x, ids)
    b = select_dominant_vectors(x, ids)
    assert a.kept_indices == b.kept_indices
    assert a.rule_used.merge_heights == b.rule_used.merge_heights


def test_file_balanced_share_beats_raw_duration() -> None:
    """The target is the speaker present in most files, not the one occupying most seconds.

    One long off-target recording must not outvote several short on-target ones, so selection is
    by file-balanced share -- and both shares are reported so the disagreement stays visible.
    """
    rng = np.random.default_rng(23)
    d = 48
    target = rng.normal(size=d)
    target /= np.linalg.norm(target)
    other = rng.normal(size=d)
    other -= (other @ target) * target
    other /= np.linalg.norm(other)

    # Three short target files (10 each) against one long off-target file (200).
    x = np.vstack(
        [_cone(target, 10, 0.02, 1), _cone(target, 10, 0.02, 2), _cone(target, 10, 0.02, 3), _cone(other, 200, 0.02, 4)]
    )
    ids = ["t1"] * 10 + ["t2"] * 10 + ["t3"] * 10 + ["long"] * 200

    sel = select_dominant_vectors(x, ids)
    dominant = next(c for c in sel.clusters if c.cluster_id == sel.dominant_cluster_id)

    assert dominant.n_files_contributing == 3
    assert dominant.file_balanced_share > 0.5
    assert dominant.window_share < 0.5  # raw duration disagrees, and both are reported


def test_the_runner_up_is_reported_with_its_distance() -> None:
    """'0.52/0.46 at cos 0.31' and '0.94/0.05 at cos 0.88' are different situations.

    Both must stay legible without this function deciding which matters.
    """
    x, ids = _two_speakers()
    sel = select_dominant_vectors(x, ids)
    assert sel.runner_up_cluster_id is not None
    assert sel.cos_dominant_to_runner_up is not None
    assert sel.cos_dominant_to_runner_up < 0.5


def test_the_cut_scales_with_this_datas_own_geometry() -> None:
    """A derived cut must track the data's own scale, not read as a fitted constant in disguise.

    ``_two_speakers`` happens to place its two cones almost orthogonal, so a hardcoded cut anywhere
    below that gap silently reproduces the right answer without being derived from anything -- a
    mutation to ``resolved_cut = 1.0`` passes every other test in this file untouched. Here the two
    cones sit only 0.35 rad apart (all merge heights measured well under 1.0), so a fixed literal
    cut that happens to work at orthogonal separation instead merges the intruder straight back in.
    """
    rng = np.random.default_rng(7)
    d = 48
    a = rng.normal(size=d)
    a /= np.linalg.norm(a)
    e2 = rng.normal(size=d)
    e2 -= (e2 @ a) * a
    e2 /= np.linalg.norm(e2)
    b = np.cos(0.35) * a + np.sin(0.35) * e2  # 0.35 rad from `a`, not orthogonal
    x = np.vstack([_cone(a, 40, 0.02, 11), _cone(b, 15, 0.02, 12)])
    ids = ["target"] * 40 + ["intruder"] * 15

    sel = select_dominant_vectors(x, ids)
    assert sel.rule_used.cut_theta < 1.0  # the whole merge tree sits well under this
    assert len(sel.kept_indices) == 40
    assert sel.dropped_indices != []


def test_a_single_coherent_group_keeps_everything() -> None:
    """Nothing to reject is a perfectly ordinary outcome and must not drop rows."""
    rng = np.random.default_rng(29)
    axis = rng.normal(size=32)
    axis /= np.linalg.norm(axis)
    x = _cone(axis, 50, 0.02, 5)
    sel = select_dominant_vectors(x, ["one"] * 50)
    assert len(sel.kept_indices) == 50
    assert sel.dropped_indices == []
