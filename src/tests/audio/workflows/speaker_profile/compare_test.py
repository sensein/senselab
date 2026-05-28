"""Tests for profile comparison / other-voice flagging.

Covers the scoring module with injected synthetic embedding vectors (no model
downloads):

- Target windows flag ``target``; windows that look like another voice flag
  ``other_voice``.
- Other-voice detection rate is well above the target-only false-positive rate,
  and target-only false flags stay a small fraction of duration.
- Low-presence windows score ``unavailable`` rather than ``other_voice``.
- Leave-one-file-out recomputes the centroid without a contributing file.
- Consensus fusion combines per-model uncertainties.
"""

from __future__ import annotations

import numpy as np

from senselab.audio.workflows.audio_analysis.embeddings import WindowEmbedding
from senselab.audio.workflows.speaker_profile import constants as C
from senselab.audio.workflows.speaker_profile.build import TaggedWindowEmbedding, aggregate_dominant_cluster
from senselab.audio.workflows.speaker_profile.compare import (
    compare_recording_to_profile,
    leave_one_file_out_profile,
    score_window,
    within_file_holdout_profile,
)

_MODEL = C.ECAPA_MODEL_ID
_MODEL2 = C.RESNET_MODEL_ID
_DIM = 16
_WINDOW_S = 1.0
_HOP_S = 0.5
_BAND = (0.3, 0.7)


def _basis(idx: int, dim: int = _DIM) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float64)
    v[idx % dim] = 1.0
    return v


def _grid(
    rng: np.random.Generator,
    centers: list[np.ndarray],
    *,
    model: str = _MODEL,
    noise: float = 0.03,
) -> list[WindowEmbedding]:
    """Build a short-grid window list whose i-th window sits near ``centers[i]``."""
    out: list[WindowEmbedding] = []
    t = 0.0
    for c in centers:
        vec = (c + rng.normal(0.0, noise, c.shape)).astype(np.float32)
        out.append(WindowEmbedding(start_s=t, end_s=t + _WINDOW_S, vector=vec))
        t += _HOP_S
    return out


# ──────────────────────────────────────────────────────────────────────────
# score_window — consensus fusion


def test_score_window_target_is_low_uncertainty() -> None:
    """A window matching the centroid has near-zero other-voice uncertainty."""
    rng = np.random.default_rng(0)
    t = _basis(0)
    vec = (t + rng.normal(0.0, 0.02, t.shape)).astype(np.float32)
    sim, unc, per_model = score_window({_MODEL: vec}, {_MODEL: list(t)}, {_MODEL: _BAND})
    assert unc is not None and unc < 0.2
    assert sim is not None and abs(sim - (1.0 - unc)) < 1e-9


def test_score_window_consensus_combines_per_model() -> None:
    """Two models → both appear in per_model and the consensus is their mean."""
    t = _basis(0)
    centroids = {_MODEL: list(t), _MODEL2: list(t)}
    band = {_MODEL: _BAND, _MODEL2: _BAND}
    # One model sees the target, the other sees an orthogonal (other) voice.
    window_vectors = {_MODEL: t.copy(), _MODEL2: _basis(1)}
    _, unc, per_model = score_window(window_vectors, centroids, band)
    assert set(per_model) == {_MODEL, _MODEL2}
    assert unc is not None
    assert abs(unc - float(np.mean(list(per_model.values())))) < 1e-9
    # Disagreeing models → consensus lands between the two extremes.
    assert min(per_model.values()) < unc < max(per_model.values())


def test_score_window_no_overlap_returns_none() -> None:
    """No shared model between window and profile → unscorable."""
    sim, unc, per_model = score_window({"other/model": _basis(0)}, {_MODEL: list(_basis(0))}, {_MODEL: _BAND})
    assert sim is None and unc is None and per_model == {}


# ──────────────────────────────────────────────────────────────────────────
# compare_recording_to_profile — flagging


def test_target_and_other_voice_windows_are_flagged() -> None:
    """Windows near the centroid flag target; orthogonal windows flag other_voice."""
    rng = np.random.default_rng(1)
    t = _basis(0)
    intruder = _basis(1)
    centers = [t] * 10 + [intruder] * 5
    detection = {_MODEL: _grid(rng, centers)}

    results = compare_recording_to_profile(detection, {_MODEL: list(t)}, {_MODEL: _BAND})
    flags = [r.flag for r in results]
    assert flags[:10] == ["target"] * 10
    assert flags[10:] == ["other_voice"] * 5


def test_detection_rate_exceeds_false_positive_rate() -> None:
    """Overlay: other-voice detection rate is well above target-only false flags."""
    rng = np.random.default_rng(2)
    t = _basis(0)
    intruder = _basis(1)
    # 30 target windows; intruder present in windows [12, 18).
    centers = [t] * 30
    intruder_idx = set(range(12, 18))
    for i in intruder_idx:
        centers[i] = intruder
    detection = {_MODEL: _grid(rng, centers)}

    results = compare_recording_to_profile(detection, {_MODEL: list(t)}, {_MODEL: _BAND})
    flagged = [i for i, r in enumerate(results) if r.flag == "other_voice"]
    detected = len(intruder_idx & set(flagged))
    false_pos = len(set(flagged) - intruder_idx)

    detection_rate = detected / len(intruder_idx)
    target_only = len(centers) - len(intruder_idx)
    false_pos_rate = false_pos / target_only

    assert detection_rate >= 2 * false_pos_rate
    assert detection_rate >= 0.8
    # Target-only false flags stay a small fraction of duration.
    assert false_pos_rate < 0.10


def test_pure_target_recording_has_few_false_flags() -> None:
    """A target-only recording flags other_voice on < 10% of windows."""
    rng = np.random.default_rng(3)
    t = _basis(0)
    detection = {_MODEL: _grid(rng, [t] * 40)}
    results = compare_recording_to_profile(detection, {_MODEL: list(t)}, {_MODEL: _BAND})
    frac_other = sum(r.flag == "other_voice" for r in results) / len(results)
    assert frac_other < 0.10


def test_low_presence_windows_are_unavailable_not_other_voice() -> None:
    """A low-p_voice window is scored unavailable even if it looks like another voice."""
    rng = np.random.default_rng(4)
    t = _basis(0)
    intruder = _basis(1)
    centers = [t] * 4 + [intruder] * 2
    detection = {_MODEL: _grid(rng, centers)}
    # Intruder windows are non-speech (cough/breath) → low presence.
    p_voice = [0.9, 0.9, 0.9, 0.9, 0.1, 0.1]

    results = compare_recording_to_profile(detection, {_MODEL: list(t)}, {_MODEL: _BAND}, p_voice_by_window=p_voice)
    assert [r.flag for r in results[4:]] == ["unavailable", "unavailable"]
    assert all(r.other_voice_uncertainty is None for r in results[4:])
    assert all(r.flag == "target" for r in results[:4])


def test_fixed_threshold_override_changes_flags() -> None:
    """A stricter fixed cutoff flags more windows; a lax one flags fewer."""
    rng = np.random.default_rng(5)
    t = _basis(0)
    # A vector whose cosine distance to the centroid (~0.5) lands mid-band, so
    # the calibrated uncertainty (~0.5) straddles a strict vs. lax cutoff.
    grey = _basis(0) + np.sqrt(3.0) * _basis(1)
    detection = {_MODEL: _grid(rng, [grey] * 12)}
    strict = compare_recording_to_profile(detection, {_MODEL: list(t)}, {_MODEL: _BAND}, other_voice_threshold=0.1)
    lax = compare_recording_to_profile(detection, {_MODEL: list(t)}, {_MODEL: _BAND}, other_voice_threshold=0.95)
    n_strict = sum(r.flag == "other_voice" for r in strict)
    n_lax = sum(r.flag == "other_voice" for r in lax)
    assert n_strict > n_lax


# ──────────────────────────────────────────────────────────────────────────
# leave-one-file-out / within-file holdout


def _tagged(
    rng: np.random.Generator, center: np.ndarray, n: int, file_id: str, start0: float = 0.0
) -> list[TaggedWindowEmbedding]:
    out: list[TaggedWindowEmbedding] = []
    t = start0
    for _ in range(n):
        vec = (center + rng.normal(0.0, 0.03, center.shape)).astype(np.float32)
        out.append(
            TaggedWindowEmbedding(
                file_id=file_id, model_id=_MODEL, window=WindowEmbedding(start_s=t, end_s=t + 2.0, vector=vec)
            )
        )
        t += 2.0
    return out


def test_leave_one_file_out_excludes_contributing_file() -> None:
    """Excluding a file recomputes the centroid from the remaining files only."""
    rng = np.random.default_rng(6)
    t = _basis(0)
    pooled = _tagged(rng, t, 10, "fileA") + _tagged(rng, t, 10, "fileB", start0=100.0)

    res = leave_one_file_out_profile(pooled, "fileA", embedding_models=[_MODEL])
    assert res is not None
    assert "fileA" not in res.per_file_dominant
    assert "fileB" in res.per_file_dominant
    # Recomputed centroid is still the (shared) target voice.
    centroid = np.asarray(res.centroids[_MODEL])
    assert float(centroid @ t) > 0.9


def test_leave_one_file_out_single_file_returns_none() -> None:
    """A single-file subject has nothing left after exclusion → None (use holdout)."""
    rng = np.random.default_rng(7)
    pooled = _tagged(rng, _basis(0), 10, "only")
    assert leave_one_file_out_profile(pooled, "only", embedding_models=[_MODEL]) is None


def test_within_file_holdout_excludes_guard_band() -> None:
    """Within-file holdout drops windows near the test interval and re-aggregates."""
    rng = np.random.default_rng(8)
    t = _basis(0)
    file_windows = _tagged(rng, t, 16, "solo")  # 2 s windows spanning 0..32 s
    res = within_file_holdout_profile(file_windows, 10.0, 12.0, embedding_models=[_MODEL], guard_s=2.0)
    assert res is not None
    full = aggregate_dominant_cluster(file_windows, embedding_models=[_MODEL])
    assert full is not None
    # Holdout used fewer windows than the full set (the guard band was removed).
    assert res.dominant_cluster.n_windows < full.dominant_cluster.n_windows
    centroid = np.asarray(res.centroids[_MODEL])
    assert float(centroid @ t) > 0.9
