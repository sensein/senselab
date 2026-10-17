"""Tests for the sparse phonetic posteriorgram."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from senselab.audio.tasks.features_extraction.ppg import PHONEME_LABELS
from senselab.audio.tasks.features_extraction.sparse_ppg import (
    SparsePPG,
    as_frame_major_array,
    sparse_ppg_fidelity,
    sparsify_ppg,
)


def _dirichlet_ppg(frames: int = 500, concentration: float = 0.1, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.full(len(PHONEME_LABELS), concentration), size=frames).astype(np.float32)


def test_sparsify_keeps_argmax_and_bounds_entries() -> None:
    """Every frame keeps its most probable phone and at most top_k entries."""
    dense = _dirichlet_ppg()
    sparse = sparsify_ppg(dense, top_k=3, min_prob=0.05)
    counts = np.bincount(sparse.frame, minlength=sparse.num_frames)
    assert counts.min() >= 1 and counts.max() <= 3
    assert (sparse.to_dense().argmax(axis=1) == dense.argmax(axis=1)).all()
    kept_below = (sparse.prob.astype(np.float32) < 0.05) & (np.diff(np.r_[-1, sparse.frame.astype(np.int64)]) == 0)
    assert not kept_below.any()


def test_sparsify_is_exact_when_nothing_is_dropped() -> None:
    """With every phone kept at float32 the round trip is the identity."""
    dense = _dirichlet_ppg(frames=50)
    sparse = sparsify_ppg(dense, top_k=len(PHONEME_LABELS), min_prob=0.0, dtype=np.float32)
    np.testing.assert_array_equal(sparse.to_dense(), dense)
    report = sparse_ppg_fidelity(dense, sparse)
    assert report["max_abs_error"] == 0.0
    assert report["argmax_agreement"] == 1.0


def test_fidelity_reports_mass_and_compression() -> None:
    """A peaked posteriorgram keeps most of its mass, and its argmax, in a fraction of the bytes."""
    dense = _dirichlet_ppg(frames=2000, concentration=0.05)
    sparse = sparsify_ppg(dense)
    report = sparse_ppg_fidelity(dense, sparse)
    assert report["argmax_agreement"] == 1.0
    assert 0.75 < report["mass_retained_mean"] <= 1.0 + 1e-3
    assert report["compression_ratio"] > 5
    assert report["entries_per_frame"] <= 3


def test_renormalized_dense_sums_to_one() -> None:
    """Renormalizing restores unit mass per frame."""
    sparse = sparsify_ppg(_dirichlet_ppg(frames=100, concentration=1.0))
    np.testing.assert_allclose(sparse.to_dense(renormalize=True).sum(axis=1), 1.0, rtol=1e-5)


def test_save_load_round_trip(tmp_path: Path) -> None:
    """A saved sparse posteriorgram loads back identical."""
    sparse = sparsify_ppg(_dirichlet_ppg(frames=300), top_k=2, min_prob=0.1)
    path = sparse.save(tmp_path / "ppg")
    assert path.suffix == ".npz"
    loaded = SparsePPG.load(path)
    assert loaded.phones == sparse.phones and loaded.num_frames == sparse.num_frames
    assert loaded.frame_rate == sparse.frame_rate
    for name in ("frame", "phone", "prob"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(sparse, name))
        assert getattr(loaded, name).dtype == getattr(sparse, name).dtype


def test_accepts_ppgs_tensor_layout() -> None:
    """A ``(1, phones, frames)`` tensor reads the same as its frame-major array."""
    dense = _dirichlet_ppg(frames=64)
    tensor = torch.from_numpy(dense.T.copy()).unsqueeze(0)
    np.testing.assert_array_equal(as_frame_major_array(tensor), dense)
    np.testing.assert_array_equal(sparsify_ppg(tensor).to_dense(), sparsify_ppg(dense).to_dense())


def test_rejects_bad_arguments() -> None:
    """Out-of-range inputs are refused."""
    dense = _dirichlet_ppg(frames=10)
    with pytest.raises(ValueError, match="top_k"):
        sparsify_ppg(dense, top_k=0)
    with pytest.raises(ValueError, match="vocabulary"):
        sparsify_ppg(dense[:, :39], phones=PHONEME_LABELS)
    with pytest.raises(ValueError, match="out of range"):
        SparsePPG(
            frame_rate=100.0,
            phones=PHONEME_LABELS,
            num_frames=2,
            frame=np.array([5], dtype=np.uint32),
            phone=np.array([0], dtype=np.uint8),
            prob=np.array([1.0], dtype=np.float16),
        )
