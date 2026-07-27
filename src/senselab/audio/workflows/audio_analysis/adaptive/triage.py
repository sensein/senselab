"""Triage round 0: cheap-signal gating decisions (spec US1, FR-002/003/004).

Design follows ``SPEECH_PRESENCE_CERTAINTY_ANALYSIS.md``: the speech gate is
driven by **continuous frame posteriors** (pyannote ``segmentation-3.0`` raw
scores — never segmentized VAD, whose hysteresis erases brief events),
aggregated on a ~100 ms window (the shortest span where "is this speech?" is
well-posed). Coarse signals (scene taggers, sentence-level ASR) do not vote
here at all. SNR comes from Brouhaha when available, with a percentile DSP
estimator as the ungated fallback.

This module is pure (numpy only): the decision function consumes arrays and a
threshold set, so it is unit-testable and reusable by both
``scripts/analyze_audio.py`` (production round 0) and ad-hoc analyses.
"""

from __future__ import annotations

from typing import Any


def triage_decision(
    *,
    p_speech: list[float],
    frame_hop_s: float,
    snr_db: list[float] | None = None,
    snr_hop_s: float | None = None,
    speech_threshold: float = 0.5,
    min_speech_s: float = 0.3,
    snr_floor_db: float = 10.0,
    low_snr_fraction_threshold: float = 0.4,
    aggregate_win_s: float = 0.1,
) -> dict[str, Any]:
    """Gate decisions from frame-level P(speech) (+ optional per-frame SNR).

    Returns a dict with:

    - ``speech_present`` — total speech time ≥ ``min_speech_s`` (FR-004 gate);
    - ``needs_enhancement`` — among speech windows, the fraction whose SNR is
      below ``snr_floor_db`` exceeds ``low_snr_fraction_threshold`` (FR-003
      gate). ``None`` when no SNR series was provided (caller decides the
      conservative default);
    - ``stats`` — speech seconds/fraction, per-window counts, SNR summary;
    - the thresholds used (provenance).

    Frame probabilities are mean-aggregated into ``aggregate_win_s`` windows
    first — reporting at the phone scale while keeping onset localization in
    the underlying posterior (analysis note §3).
    """
    import numpy as np

    probs = np.asarray(p_speech, dtype=float)
    thresholds = {
        "speech_threshold": speech_threshold,
        "min_speech_s": min_speech_s,
        "snr_floor_db": snr_floor_db,
        "low_snr_fraction_threshold": low_snr_fraction_threshold,
        "aggregate_win_s": aggregate_win_s,
    }
    if probs.size == 0 or frame_hop_s <= 0:
        return {
            "speech_present": True,  # conservative: never silently drop heavy tasks on empty evidence
            "needs_enhancement": None,
            "inconclusive": True,
            "reason": "no_frame_posteriors",
            "stats": {},
            "thresholds": thresholds,
        }

    frames_per_win = max(1, int(round(aggregate_win_s / frame_hop_s)))
    n_windows = int(np.ceil(probs.size / frames_per_win))
    win_means = np.array(
        [float(np.nanmean(probs[i * frames_per_win : (i + 1) * frames_per_win])) for i in range(n_windows)]
    )
    win_s = frames_per_win * frame_hop_s
    speech_mask = win_means >= speech_threshold
    speech_s = float(speech_mask.sum() * win_s)
    speech_present = speech_s >= min_speech_s

    needs_enhancement: bool | None = None
    snr_stats: dict[str, Any] = {}
    if snr_db is not None and snr_hop_s and len(snr_db) > 0:
        snr = np.asarray(snr_db, dtype=float)
        # SNR value per aggregation window (mean of overlapping SNR frames).
        win_snr = np.full(n_windows, np.nan)
        for i in range(n_windows):
            lo = int(np.floor(i * win_s / snr_hop_s))
            hi = max(lo + 1, int(np.ceil((i + 1) * win_s / snr_hop_s)))
            seg = snr[lo : min(hi, snr.size)]
            if seg.size:
                win_snr[i] = float(np.nanmean(seg))
        speech_snr = win_snr[speech_mask & ~np.isnan(win_snr)]
        if speech_snr.size:
            low_fraction = float((speech_snr < snr_floor_db).mean())
            needs_enhancement = bool(speech_present and low_fraction >= low_snr_fraction_threshold)
            snr_stats = {
                "median_snr_db_in_speech": round(float(np.median(speech_snr)), 2),
                "low_snr_fraction_in_speech": round(low_fraction, 4),
                "n_speech_windows_with_snr": int(speech_snr.size),
            }

    return {
        "speech_present": bool(speech_present),
        "needs_enhancement": needs_enhancement,
        "inconclusive": False,
        "stats": {
            "duration_s": round(float(probs.size * frame_hop_s), 3),
            "speech_s": round(speech_s, 3),
            "speech_fraction": round(float(speech_mask.mean()), 4),
            "n_windows": n_windows,
            "aggregate_win_s_effective": round(win_s, 4),
            "mean_p_speech": round(float(np.nanmean(probs)), 4),
            **snr_stats,
        },
        "thresholds": thresholds,
    }


def dsp_snr_series(
    wav: Any,  # noqa: ANN401
    sr: int,
    *,
    frame_s: float = 0.05,
    p_speech: list[float] | None = None,
    p_hop_s: float | None = None,
    nonspeech_threshold: float = 0.35,
) -> tuple[list[float], float]:
    """Ungated DSP SNR fallback: frame RMS dB minus an estimated noise floor.

    When a frame posterior is provided, the noise floor is the median RMS dB of
    frames the posterior calls non-speech (< ``nonspeech_threshold``) — the
    textbook noise-in-silence estimator, and robust on high-speech-fraction
    audio where a plain global percentile lands on speech frames and compresses
    the SNR. Without a posterior (or with < 3 non-speech frames) it falls back
    to the global 10th percentile. Crude relative to Brouhaha, but
    dependency-free and monotone with true SNR — sufficient for the coarse
    enhancement gate. Returns ``(snr_db_per_frame, frame_hop_s)``.
    """
    import numpy as np

    x = np.asarray(wav, dtype=float).squeeze()
    hop = max(1, int(round(frame_s * sr)))
    n = max(1, x.size // hop)
    rms = np.array([float(np.sqrt(np.mean(np.square(x[i * hop : (i + 1) * hop])) + 1e-12)) for i in range(n)])
    db = 20.0 * np.log10(rms + 1e-12)
    noise_floor: float | None = None
    if p_speech is not None and p_hop_s and len(p_speech) > 0:
        probs = np.asarray(p_speech, dtype=float)
        frame_times = (np.arange(n) + 0.5) * (hop / sr)
        idx = np.clip((frame_times / p_hop_s).astype(int), 0, probs.size - 1)
        nonspeech = db[probs[idx] < nonspeech_threshold]
        if nonspeech.size >= 3:
            noise_floor = float(np.median(nonspeech))
    if noise_floor is None:
        noise_floor = float(np.percentile(db, 10))
    return [round(float(d - noise_floor), 3) for d in db], hop / sr
