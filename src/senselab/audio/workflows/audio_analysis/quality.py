"""Per-bucket audio-quality degradation scores for the presence axis.

Emits four ``[0, 1]`` degradation scores (0 = clean, 1 = fully degraded) plus a
quality-uncertainty score, per presence bucket:

- ``quality_snr``     — from Brouhaha frame SNR (primary), cross-checked against
                        senselab's existing spectral-gating and peak SNR metrics;
- ``quality_clip``    — from ``proportion_clipped_metric`` (existing DSP);
- ``quality_reverb``  — from Brouhaha C50;
- ``quality_bandwidth``— from a torch.stft spectral-rolloff-vs-Nyquist measure;
- ``quality_uncertainty`` — normalized spread among the independent SNR estimators.

**Analysis resolution ≠ reporting grid.** The STFT/model estimators are unreliable
below ~0.5 s (and SQUIM/Brouhaha are trained on longer chunks), so quality is
computed on a fixed 0.5 s / 0.25 s *analysis* window and each presence *reporting*
bucket takes the value of its nearest analysis window. The analysis-window params
are recorded in provenance; the true quality resolution is 0.5 s even when presence
buckets are finer. See ``contracts/quality.md``.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.quality_control.metrics import (
    peak_snr_from_spectral_metric,
    proportion_clipped_metric,
    root_mean_square_energy_metric,
    spectral_gating_snr_metric,
)
from senselab.audio.tasks.scene_quality.brouhaha import BrouhahaFrames
from senselab.audio.workflows.audio_analysis.embeddings import _slice_audio, _window_starts
from senselab.audio.workflows.audio_analysis.grid import BucketGrid

QUALITY_ANALYSIS_WIN_S = 0.5
QUALITY_ANALYSIS_HOP_S = 0.25

# Default normalization when no fitted calibration profile is supplied (US5 replaces
# these via the calibration profile). Documented, bounded, uncalibrated.
_DEFAULT_SNR_CLEAN_DB = 30.0
_DEFAULT_SNR_FLOOR_DB = 0.0
_DEFAULT_C50_CLEAN_DB = 30.0
_DEFAULT_C50_FLOOR_DB = -5.0
_DEFAULT_BANDWIDTH_ROLLOFF_PCT = 0.85
# SNR-estimator spread (dB) that maps to full quality-uncertainty (=1.0).
_SNR_SPREAD_REF_DB = 15.0
# Below this RMS the slice is treated as silence → quality is undefined (null).
_SILENCE_RMS = 1e-4


def _linear_db_to_degradation(value_db: float, clean_db: float, floor_db: float) -> Optional[float]:
    """Map a dB quality value to a ``[0, 1]`` degradation score (0 = clean)."""
    if value_db is None or not np.isfinite(value_db):
        return None
    span = clean_db - floor_db
    if span <= 0:
        return None
    return float(np.clip((clean_db - value_db) / span, 0.0, 1.0))


def _bandwidth_degradation(slice_audio: Audio) -> Optional[float]:
    """Effective-bandwidth degradation from the 85% spectral rolloff vs Nyquist.

    Uses ``torch.stft`` (no librosa) — a band-limited signal (e.g. telephone-band
    ≤ 4 kHz) concentrates energy in the low bins, so its rolloff frequency sits
    well below Nyquist and the degradation approaches 1; a full-band signal rolls
    off near Nyquist and scores near 0.
    """
    wf = slice_audio.waveform
    if wf is None or wf.numel() == 0:
        return None
    y = wf.mean(dim=0) if wf.shape[0] > 1 else wf[0]
    y = y.detach().to(torch.float32).reshape(-1)
    n = int(y.shape[-1])
    if n < 256:
        return None
    n_fft = min(2048, 1 << int(math.floor(math.log2(n))))
    if n_fft < 256:
        return None
    spec = torch.stft(
        y,
        n_fft=n_fft,
        hop_length=n_fft // 4,
        window=torch.hann_window(n_fft, device=y.device),
        center=True,
        return_complex=True,
    )
    power = (spec.abs() ** 2).mean(dim=1)  # avg over frames → per-frequency-bin energy
    total = float(power.sum().item())
    if total <= 0:
        return None
    cumulative = torch.cumsum(power, dim=0) / total
    idx = int(torch.searchsorted(cumulative, torch.tensor(_DEFAULT_BANDWIDTH_ROLLOFF_PCT)).item())
    idx = min(idx, power.shape[0] - 1)
    sr = slice_audio.sampling_rate
    nyquist = sr / 2.0
    rolloff_hz = idx * sr / n_fft  # bin index → Hz
    if nyquist <= 0:
        return None
    return float(np.clip(1.0 - rolloff_hz / nyquist, 0.0, 1.0))


def _analysis_window_quality(
    slice_audio: Audio,
    brouhaha: Optional[BrouhahaFrames],
    start_s: float,
    end_s: float,
    calibration: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Compute the quality vector for one analysis window."""
    cal = calibration or {}
    snr_clean = float(cal.get("snr_clean_db", _DEFAULT_SNR_CLEAN_DB))
    snr_floor = float(cal.get("snr_floor_db", _DEFAULT_SNR_FLOOR_DB))
    c50_clean = float(cal.get("c50_clean_db", _DEFAULT_C50_CLEAN_DB))
    c50_floor = float(cal.get("c50_floor_db", _DEFAULT_C50_FLOOR_DB))

    rms = root_mean_square_energy_metric(slice_audio)
    is_silence = not np.isfinite(rms) or rms < _SILENCE_RMS

    # Clipping is always computable and cheap.
    try:
        clip_deg: Optional[float] = float(np.clip(proportion_clipped_metric(slice_audio), 0.0, 1.0))
    except (ValueError, TypeError):
        clip_deg = None

    if is_silence:
        # Signal-dependent quality is undefined on silence (edge case in spec).
        return {
            "quality_snr": None,
            "quality_clip": clip_deg,
            "quality_reverb": None,
            "quality_bandwidth": None,
            "quality_uncertainty": None,
            "_raw": {"silence": True, "rms": rms},
        }

    # SNR estimators (dB): Brouhaha (primary) + two existing DSP metrics.
    snr_estimates_db: list[float] = []
    brouhaha_snr_db: Optional[float] = None
    brouhaha_c50_db: Optional[float] = None
    if brouhaha is not None:
        _vad, b_snr, b_c50 = brouhaha.mean_in_window(start_s, end_s)
        if np.isfinite(b_snr):
            brouhaha_snr_db = float(b_snr)
            snr_estimates_db.append(brouhaha_snr_db)
        if np.isfinite(b_c50):
            brouhaha_c50_db = float(b_c50)
    for metric in (spectral_gating_snr_metric, peak_snr_from_spectral_metric):
        try:
            val = float(metric(slice_audio))
            if np.isfinite(val):
                snr_estimates_db.append(val)
        except (ValueError, TypeError, RuntimeError):
            continue

    # Primary SNR: Brouhaha when present, else mean of the DSP estimators.
    if brouhaha_snr_db is not None:
        primary_snr_db: Optional[float] = brouhaha_snr_db
    elif snr_estimates_db:
        primary_snr_db = float(np.mean(snr_estimates_db))
    else:
        primary_snr_db = None

    quality_snr = (
        _linear_db_to_degradation(primary_snr_db, snr_clean, snr_floor) if primary_snr_db is not None else None
    )
    quality_reverb = (
        _linear_db_to_degradation(brouhaha_c50_db, c50_clean, c50_floor) if brouhaha_c50_db is not None else None
    )
    quality_bandwidth = _bandwidth_degradation(slice_audio)

    # Quality-uncertainty: normalized spread among the independent SNR estimators.
    quality_uncertainty: Optional[float] = None
    if len(snr_estimates_db) >= 2:
        spread = float(np.std(snr_estimates_db))
        quality_uncertainty = float(np.clip(spread / _SNR_SPREAD_REF_DB, 0.0, 1.0))

    return {
        "quality_snr": quality_snr,
        "quality_clip": clip_deg,
        "quality_reverb": quality_reverb,
        "quality_bandwidth": quality_bandwidth,
        "quality_uncertainty": quality_uncertainty,
        "_raw": {
            "snr_estimates_db": snr_estimates_db,
            "brouhaha_snr_db": brouhaha_snr_db,
            "brouhaha_c50_db": brouhaha_c50_db,
            "primary_snr_db": primary_snr_db,
        },
    }


def harvest_quality_scores(
    *,
    audio: Audio,
    brouhaha: Optional[BrouhahaFrames],
    grid: BucketGrid,
    calibration: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Return one quality dict per presence bucket on ``grid``.

    Quality is computed on the fixed 0.5 s / 0.25 s analysis window and broadcast
    to each reporting bucket via nearest-analysis-window-center. Each returned dict
    carries ``start``, ``end``, the five ``quality_*`` values (any of which may be
    ``None`` when its source is unavailable — FR-023), and a ``_raw`` block for the
    parquet ``model_votes``.
    """
    duration_s = float(audio.waveform.shape[-1]) / float(audio.sampling_rate)
    if duration_s <= 0:
        return []

    starts = _window_starts(duration_s, QUALITY_ANALYSIS_WIN_S, QUALITY_ANALYSIS_HOP_S)
    analysis: list[dict[str, Any]] = []
    for t in starts:
        end = min(duration_s, t + QUALITY_ANALYSIS_WIN_S)
        sl = _slice_audio(audio, t, end)
        q = _analysis_window_quality(sl, brouhaha, t, end, calibration)
        q["_center"] = 0.5 * (t + end)
        analysis.append(q)

    if not analysis:
        return []
    centers = np.asarray([a["_center"] for a in analysis], dtype=np.float64)

    out: list[dict[str, Any]] = []
    for b_start, b_end, _idx in grid.iter_buckets(duration_s):
        b_center = 0.5 * (b_start + b_end)
        nearest = int(np.argmin(np.abs(centers - b_center)))
        src = analysis[nearest]
        out.append(
            {
                "start": b_start,
                "end": b_end,
                "quality_snr": src["quality_snr"],
                "quality_clip": src["quality_clip"],
                "quality_reverb": src["quality_reverb"],
                "quality_bandwidth": src["quality_bandwidth"],
                "quality_uncertainty": src["quality_uncertainty"],
                "_raw": src["_raw"],
            }
        )
    return out
