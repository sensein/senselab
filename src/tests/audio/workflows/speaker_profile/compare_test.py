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
import pytest

from senselab.audio.workflows.audio_analysis.embeddings import WindowEmbedding
from senselab.audio.workflows.speaker_profile import constants as C
from senselab.audio.workflows.speaker_profile.build import TaggedWindowEmbedding, aggregate_dominant_cluster
from senselab.audio.workflows.speaker_profile.compare import (
    GridMismatchError,
    check_grid_compatibility,
    compare_recording_to_profile,
    compute_target_quality,
    derive_window_grid,
    leave_one_file_out_profile,
    profile_votes_by_bucket,
    score_voice_groups,
    score_window,
    summarize_other_voice,
    within_file_holdout_profile,
)
from senselab.audio.workflows.speaker_profile.types import ProfileComparisonResult

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


def test_diar_overlap_corroborator_raises_other_voice() -> None:
    """A diar-overlap window flips target→other_voice even when the profile reads it as target."""
    rng = np.random.default_rng(7)
    t = _basis(0)
    # All windows sit on the centroid → all 'target' absent any overlap signal.
    detection = {_MODEL: _grid(rng, [t] * 6)}
    overlap = [False, False, True, True, False, False]
    results = compare_recording_to_profile(
        detection, {_MODEL: list(t)}, {_MODEL: _BAND}, diar_overlap_by_window=overlap
    )
    assert [r.flag for r in results] == ["target", "target", "other_voice", "other_voice", "target", "target"]
    # Overlap windows are raised to the default floor (1.0); non-overlap untouched.
    assert all(r.other_voice_uncertainty == 1.0 for r in results[2:4])
    assert all(r.flag == "target" for r in (results[0], results[1], results[4], results[5]))


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
# recording-level rollup + bucket vote mapping


def _result(
    start: float, flag: str, unc: float | None, per_model: dict[str, float] | None = None
) -> ProfileComparisonResult:
    return ProfileComparisonResult(
        start=start,
        end=start + 1.0,
        similarity=(None if unc is None else 1.0 - unc),
        other_voice_uncertainty=unc,
        flag=flag,  # type: ignore[arg-type]
        p_voice=None,
        per_model=per_model or {},
    )


def test_summarize_other_voice_fraction_and_seconds() -> None:
    """Rollup counts only speech-present windows and uses non-overlapping steps."""
    # 0.5 s hop grid: 4 target, 2 other_voice, 1 unavailable.
    results = [
        _result(0.0, "target", 0.1),
        _result(0.5, "target", 0.15),
        _result(1.0, "other_voice", 0.8),
        _result(1.5, "other_voice", 0.9),
        _result(2.0, "target", 0.2),
        _result(2.5, "target", 0.1),
        _result(3.0, "unavailable", None),
    ]
    summary = summarize_other_voice(results, "ok")
    # speech-present windows = 6 steps of 0.5 s = 3.0 s; other_voice = 2 * 0.5 = 1.0 s.
    assert abs(summary.profile_speech_present_seconds - 3.0) < 1e-6
    assert abs(summary.profile_other_voice_seconds - 1.0) < 1e-6
    assert abs(summary.profile_other_voice_fraction - (1.0 / 3.0)) < 1e-6
    assert summary.profile_peak_other_voice_uncertainty == 0.9
    assert summary.profile_confidence == "ok"


def test_summarize_other_voice_all_unavailable() -> None:
    """No speech-present windows → zeroed rollup, no division error."""
    results = [_result(0.0, "unavailable", None), _result(0.5, "unavailable", None)]
    summary = summarize_other_voice(results, "low")
    assert summary.profile_speech_present_seconds == 0.0
    assert summary.profile_other_voice_fraction == 0.0
    assert summary.profile_confidence == "low"


def test_profile_votes_by_bucket_maps_and_keys() -> None:
    """Each bucket gets a consensus vote plus one per-model vote from its window."""
    results = [
        _result(0.0, "target", 0.1, {_MODEL: 0.1, _MODEL2: 0.12}),
        _result(0.5, "other_voice", 0.8, {_MODEL: 0.82, _MODEL2: 0.78}),
    ]
    # Buckets centered on each result's window center (0.5, 1.0) plus a far one.
    buckets = [(0.0, 1.0), (1.0, 2.0), (5.0, 5.5)]
    votes = profile_votes_by_bucket(results, buckets)
    assert len(votes) == 3
    assert votes[0]["speaker_profile/consensus"]["flag"] == "target"
    assert f"speaker_profile/{_MODEL}" in votes[0]
    assert votes[1]["speaker_profile/consensus"]["flag"] == "other_voice"
    # A bucket with no overlapping window stays empty (no borrowed flag).
    assert votes[2] == {}


# ──────────────────────────────────────────────────────────────────────────
# compute_target_quality (US3)


def test_clean_recording_outranks_contaminated() -> None:
    """SC-005: a clean target-dominant recording scores higher than a contaminated one."""
    clean = [_result(i * 0.5, "target", 0.1) for i in range(10)]
    contaminated = [_result(i * 0.5, "target", 0.1) for i in range(5)] + [
        _result((5 + i) * 0.5, "other_voice", 0.85) for i in range(5)
    ]
    q_clean = compute_target_quality(clean, "ok")
    q_contam = compute_target_quality(contaminated, "ok")
    assert q_clean.profile_target_quality is not None and q_contam.profile_target_quality is not None
    assert q_clean.profile_target_quality > q_contam.profile_target_quality
    # Clean: all speech-present windows match the target.
    assert q_clean.profile_target_match_fraction == 1.0
    assert abs(q_contam.profile_target_match_fraction - 0.5) < 0.1


def test_target_quality_echoes_confidence_and_handles_empty() -> None:
    """Confidence is echoed; an all-unavailable recording yields a zeroed indicator."""
    q = compute_target_quality([_result(0.0, "unavailable", None)], "low")
    assert q.profile_confidence == "low"
    # No scorable windows → target_quality is None ("unavailable"), not 0.0,
    # so a consumer doesn't read "couldn't assess" as "confidently poor".
    assert q.profile_target_quality is None
    assert q.profile_target_match_fraction == 0.0
    assert q.profile_squim is None


def test_target_quality_reports_raw_squim_on_matched_windows() -> None:
    """SQUIM is reported as raw means over matched windows (not folded into the scalar)."""
    results = [_result(0.0, "target", 0.1), _result(0.5, "other_voice", 0.9), _result(1.0, "target", 0.1)]
    squim = [
        {"stoi": 0.9, "pesq": 3.0, "si_sdr": 12.0},
        {"stoi": 0.4, "pesq": 1.5, "si_sdr": 2.0},  # other_voice window — excluded
        {"stoi": 0.8, "pesq": 2.6, "si_sdr": 10.0},
    ]
    q = compute_target_quality(results, "ok", squim_by_window=squim)
    assert q.profile_squim is not None
    # Mean over the two target windows (indices 0 and 2), not the other_voice one.
    assert abs(q.profile_squim["stoi"] - 0.85) < 1e-9
    assert abs(q.profile_squim["pesq"] - 2.8) < 1e-9
    assert abs(q.profile_squim["si_sdr"] - 11.0) < 1e-9
    # The headline scalar is profile-only — independent of SQUIM.
    q_no_squim = compute_target_quality(results, "ok")
    assert q.profile_target_quality == q_no_squim.profile_target_quality
    assert q_no_squim.profile_squim is None


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


# ── voice-group scoring (the interface the step-2 identity rule consumes) ──
#     Naming pooled voice groups is preferred over per-window scoring wherever the
#     recording has 2+ groups: the group centroid pools many windows (far less noisy
#     than one short window), and picking the *closer* group is a relative decision
#     that needs no absolute threshold. With one group there is nothing to compare
#     against, so the result is explicitly absolute-basis.


def test_voice_group_scoring_picks_the_group_matching_the_profile() -> None:
    """With two groups, the one aligned with the centroid wins on a relative basis."""
    centroids = {_MODEL: _basis(0).tolist()}
    groups = {"R0": {_MODEL: _basis(3)}, "R1": {_MODEL: _basis(0)}}

    assignment = score_voice_groups(groups, centroids, {_MODEL: _BAND})

    assert assignment.subject_group_id == "R1"
    assert assignment.basis == "relative"
    assert assignment.margin is not None and assignment.margin > 0.0
    best = next(m for m in assignment.matches if m.group_id == "R1")
    other = next(m for m in assignment.matches if m.group_id == "R0")
    assert best.similarity is not None and other.similarity is not None
    assert best.similarity > other.similarity


def test_voice_group_scoring_is_absolute_with_a_single_group() -> None:
    """One group → no margin available, so the basis is absolute and margin is None."""
    centroids = {_MODEL: _basis(0).tolist()}
    assignment = score_voice_groups({"R0": {_MODEL: _basis(0)}}, centroids, {_MODEL: _BAND})

    assert assignment.subject_group_id == "R0"
    assert assignment.basis == "absolute"
    assert assignment.margin is None
    assert assignment.matches[0].similarity is not None


def test_voice_group_scoring_reports_no_subject_when_nothing_is_scorable() -> None:
    """No model overlap between groups and profile → no assignment, not a false pick."""
    assignment = score_voice_groups({"R0": {_MODEL2: _basis(0)}}, {_MODEL: _basis(0).tolist()}, {_MODEL: _BAND})

    assert assignment.subject_group_id is None
    assert assignment.basis == "unavailable"
    assert assignment.matches[0].similarity is None


def test_voice_group_scoring_handles_no_groups() -> None:
    """An empty group set is reported as unavailable rather than raising."""
    assignment = score_voice_groups({}, {_MODEL: _basis(0).tolist()}, {_MODEL: _BAND})
    assert assignment.matches == [] and assignment.subject_group_id is None
    assert assignment.basis == "unavailable"


# ── grid-match guard ──────────────────────────────────────────────────
#     The calibration band maps cosine distance to a calibrated uncertainty, and that
#     mapping is grid-specific. Measurement showed the band does NOT adapt (it falls back
#     to fixed literature values), so a cross-grid comparison silently misapplies it.


def _win(start: float, win: float) -> WindowEmbedding:
    return WindowEmbedding(start_s=start, end_s=start + win, vector=_basis(0).astype(np.float32))


def test_derive_window_grid_reads_length_and_hop() -> None:
    """Grid is inferred from the timestamps, not passed alongside them."""
    ws = [_win(0.0, 2.0), _win(1.0, 2.0), _win(2.0, 2.0)]
    assert derive_window_grid(ws) == (2.0, 1.0)


def test_derive_window_grid_hop_unobservable_with_one_window() -> None:
    """A single window cannot reveal a hop; say so rather than guessing."""
    assert derive_window_grid([_win(0.0, 0.5)]) == (0.5, None)
    assert derive_window_grid([]) == (0.0, None)


def test_check_grid_compatibility_accepts_a_matching_grid() -> None:
    """The exact grid the profile was built at passes."""
    check_grid_compatibility([_win(0.0, 2.0), _win(1.0, 2.0)], 2.0)


def test_check_grid_compatibility_rejects_a_different_window_length() -> None:
    """0.5 s windows against a 2.0 s profile is the miscalibration this exists to stop."""
    with pytest.raises(GridMismatchError, match="0.5s but the profile was enrolled at 2s"):
        check_grid_compatibility([_win(0.0, 0.5), _win(0.25, 0.5)], 2.0)


def test_check_grid_compatibility_ignores_hop_differences() -> None:
    """Hop is deliberately not checked: duration rollups derive coverage from timestamps.

    ``_window_step_seconds`` computes each window's step from the results themselves, so a
    different hop does not skew the reported seconds — only the window length affects how
    the calibration band applies.
    """
    check_grid_compatibility([_win(0.0, 2.0), _win(0.5, 2.0)], 2.0)


def test_check_grid_compatibility_is_a_no_op_on_empty_windows() -> None:
    """Nothing extracted is a separate condition from a mismatched grid."""
    check_grid_compatibility([], 2.0)
