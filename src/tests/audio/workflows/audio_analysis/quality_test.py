"""Tests for the per-bucket scene-quality harvester (feature 20260722-175022).

Model-free: Brouhaha frames are constructed directly, so no gated model download
is needed. Covers SC-001 (clean → low degradation), SC-002 (noised region → SNR
degradation up ≥0.3), clipping, effective bandwidth, quality-uncertainty spread,
and the null-safe / silence edge cases (FR-023).
"""

from __future__ import annotations

import numpy as np
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.scene_quality.brouhaha import BrouhahaFrames
from senselab.audio.workflows.audio_analysis.grid import BucketGrid
from senselab.audio.workflows.audio_analysis.quality import harvest_quality_scores

SR = 16000


def _audio(y: np.ndarray) -> Audio:
    """Wrap a 1-D numpy signal as a mono 16 kHz ``Audio``."""
    wf = torch.tensor(np.asarray(y, dtype=np.float32)).reshape(1, -1)
    return Audio(waveform=wf, sampling_rate=SR)


def _const_brouhaha(duration_s: float, snr_db: float, c50_db: float, hop: float = 0.02) -> BrouhahaFrames:
    """Build constant-valued Brouhaha frames spanning ``duration_s``."""
    n = max(1, int(duration_s / hop))
    return BrouhahaFrames(
        vad=np.ones(n),
        snr_db=np.full(n, snr_db),
        c50_db=np.full(n, c50_db),
        frame_hop_s=hop,
    )


def _white_noise(duration_s: float, amp: float = 0.1, seed: int = 0) -> np.ndarray:
    """Return broadband white noise (full-band, deterministic)."""
    rng = np.random.default_rng(seed)
    return amp * rng.standard_normal(int(duration_s * SR))


def _lowpass_tones(duration_s: float) -> np.ndarray:
    """Return a band-limited signal (content only ≤ ~1 kHz)."""
    t = np.arange(int(duration_s * SR)) / SR
    return 0.1 * (np.sin(2 * np.pi * 300 * t) + np.sin(2 * np.pi * 800 * t))


def test_all_scores_in_unit_range() -> None:
    """Every non-null quality score stays within [0, 1]."""
    audio = _audio(_white_noise(2.0))
    grid = BucketGrid(win_length=0.5, hop_length=0.5)
    rows = harvest_quality_scores(audio=audio, brouhaha=_const_brouhaha(2.0, 20.0, 25.0), grid=grid)
    assert rows
    for r in rows:
        for key in ("quality_snr", "quality_clip", "quality_reverb", "quality_bandwidth", "quality_uncertainty"):
            v = r[key]
            assert v is None or (0.0 <= v <= 1.0), f"{key}={v} out of range"


def test_clean_high_snr_low_degradation() -> None:
    """SC-001: high Brouhaha SNR + high C50 → low snr/reverb degradation."""
    audio = _audio(_white_noise(2.0))
    rows = harvest_quality_scores(
        audio=audio,
        brouhaha=_const_brouhaha(2.0, snr_db=30.0, c50_db=30.0),
        grid=BucketGrid(win_length=0.5, hop_length=0.5),
    )
    assert all(r["quality_snr"] is not None and r["quality_snr"] < 0.1 for r in rows)
    assert all(r["quality_reverb"] is not None and r["quality_reverb"] < 0.1 for r in rows)


def test_noised_region_snr_degradation_rises() -> None:
    """SC-002: a low-SNR region shows quality_snr degradation ≥0.3 higher than a clean region."""
    duration = 2.0
    hop = 0.02
    n = int(duration / hop)
    snr = np.full(n, 30.0)
    snr[: n // 2] = 3.0  # first half is noisy
    brouhaha = BrouhahaFrames(vad=np.ones(n), snr_db=snr, c50_db=np.full(n, 25.0), frame_hop_s=hop)
    rows = harvest_quality_scores(
        audio=_audio(_white_noise(duration)),
        brouhaha=brouhaha,
        grid=BucketGrid(win_length=0.5, hop_length=0.5),
    )
    noisy = [r["quality_snr"] for r in rows if r["end"] <= 1.0]
    clean = [r["quality_snr"] for r in rows if r["start"] >= 1.0]
    assert min(noisy) - max(clean) >= 0.3


def test_clipping_detected() -> None:
    """Hard-clipped audio yields higher quality_clip than clean audio."""
    clean = _white_noise(1.0, amp=0.2)
    clipped = np.clip(_white_noise(1.0, amp=2.0), -1.0, 1.0)  # hard clipping plateaus at ±1
    grid = BucketGrid(win_length=0.5, hop_length=0.5)
    r_clean = harvest_quality_scores(audio=_audio(clean), brouhaha=None, grid=grid)
    r_clip = harvest_quality_scores(audio=_audio(clipped), brouhaha=None, grid=grid)
    assert max(r["quality_clip"] for r in r_clip) > max(r["quality_clip"] for r in r_clean)


def test_bandwidth_full_vs_bandlimited() -> None:
    """A band-limited signal reports higher quality_bandwidth than full-band noise."""
    grid = BucketGrid(win_length=0.5, hop_length=0.5)
    full = harvest_quality_scores(audio=_audio(_white_noise(1.5)), brouhaha=None, grid=grid)
    band = harvest_quality_scores(audio=_audio(_lowpass_tones(1.5)), brouhaha=None, grid=grid)
    full_bw = np.nanmean([r["quality_bandwidth"] for r in full if r["quality_bandwidth"] is not None])
    band_bw = np.nanmean([r["quality_bandwidth"] for r in band if r["quality_bandwidth"] is not None])
    assert band_bw > full_bw
    assert band_bw > 0.5  # strongly band-limited


def test_quality_uncertainty_rises_on_estimator_disagreement() -> None:
    """FR-005: divergent SNR estimators → higher quality_uncertainty."""
    audio = _audio(_white_noise(1.0, amp=0.15))
    grid = BucketGrid(win_length=0.5, hop_length=0.5)
    agree = harvest_quality_scores(audio=audio, brouhaha=_const_brouhaha(1.0, snr_db=6.0, c50_db=20.0), grid=grid)
    disagree = harvest_quality_scores(audio=audio, brouhaha=_const_brouhaha(1.0, snr_db=60.0, c50_db=20.0), grid=grid)
    au = np.nanmean([r["quality_uncertainty"] for r in agree if r["quality_uncertainty"] is not None])
    du = np.nanmean([r["quality_uncertainty"] for r in disagree if r["quality_uncertainty"] is not None])
    assert du > au


def test_null_safe_without_brouhaha() -> None:
    """FR-023 / T010: no Brouhaha → reverb null, but snr (DSP) and bandwidth still computed."""
    rows = harvest_quality_scores(
        audio=_audio(_white_noise(1.0)),
        brouhaha=None,
        grid=BucketGrid(win_length=0.5, hop_length=0.5),
    )
    assert rows
    assert all(r["quality_reverb"] is None for r in rows)
    assert any(r["quality_snr"] is not None for r in rows)
    assert any(r["quality_bandwidth"] is not None for r in rows)


def test_silence_bucket_yields_null_quality() -> None:
    """Silent buckets report null signal-dependent quality (spec edge case)."""
    silence = np.zeros(int(1.0 * SR))
    rows = harvest_quality_scores(
        audio=_audio(silence),
        brouhaha=_const_brouhaha(1.0, 20.0, 20.0),
        grid=BucketGrid(win_length=0.5, hop_length=0.5),
    )
    assert rows
    for r in rows:
        assert r["quality_snr"] is None
        assert r["quality_reverb"] is None
        assert r["quality_bandwidth"] is None
