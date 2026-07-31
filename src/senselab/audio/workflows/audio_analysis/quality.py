"""L1 scene-quality measurements: what the estimators measured, in their own units.

Seven measurements per analysis window, each in native units and none rescaled:

- ``snr_brouhaha_db`` — Brouhaha's SNR head, dB;
- ``c50_brouhaha_db`` — Brouhaha's C50 (clarity) head, dB;
- ``snr_spectral_gating_db`` / ``snr_peak_db`` — senselab's two DSP SNR metrics, dB;
- ``rolloff_95_hz`` — the frequency below which 95% of spectral energy sits, Hz;
- ``proportion_clipped`` — fraction of samples at full scale;
- ``rms`` — root-mean-square energy, uncalibrated.

**Why no degradation scores here.** This module used to emit ``quality_snr`` and
``quality_reverb`` as ``[0, 1]`` scores via ``clip((clean_db - value) / span, 0, 1)`` against 25 dB
and 30 dB anchors. Both returned **0.0 in every bucket of every recording measured**, because
clean speech sits at 60-70 dB SNR and 59.8 dB C50 — far above anchors chosen for conversational
audio. Probing the model directly showed the heads were never the problem: across digital silence,
white noise and clean speech they span −5 to 70 dB SNR and discriminate speech from silence by
+0.98 on the VAD head. A working measurement was destroyed by a clamp sitting on top of it. The
anchors are calibration, so they belong in :mod:`degradation` at L2, where a fitted profile can
replace them and where a saturating choice is visible as a fusion decision rather than baked into
the recorded data.

Two related reductions were removed rather than moved. ``primary_snr_db`` picked Brouhaha and
otherwise averaged the DSP metrics — estimator selection is fusion. ``quality_uncertainty`` took
the standard deviation of all three; because they use different noise-floor definitions, that
spread measured definitional disagreement rather than measurement uncertainty and pinned at 1.0
structurally, even on perfect audio. See
``specs/20260728-221507-per-speaker-identity-scene/l1-post-processing-register.md`` items 17-24.

**Analysis resolution ≠ reporting grid.** The STFT and model estimators are unreliable below
~0.5 s (Brouhaha is trained at 6 s), so measurement happens on a fixed 0.5 s / 0.25 s analysis
window. Each reporting bucket takes its nearest analysis window's value; the true resolution is
recorded in provenance so a consumer cannot mistake a repeated value for an independent one.
"""

from __future__ import annotations

import math
from typing import Any, Optional, SupportsFloat

import numpy as np
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.quality_control.metrics import (
    peak_snr_from_spectral_metric,
    proportion_clipped_metric,
    root_mean_square_energy_metric,
    spectral_gating_snr_metric,
)
from senselab.audio.tasks.scene_quality.brouhaha import BROUHAHA_MODEL_ID, BROUHAHA_REVISION, BrouhahaFrames
from senselab.audio.workflows.audio_analysis.embeddings import _slice_audio, _window_starts
from senselab.audio.workflows.audio_analysis.grid import BucketGrid
from senselab.audio.workflows.audio_analysis.signal import SignalProvenance

__all__ = [
    "QUALITY_ANALYSIS_HOP_S",
    "QUALITY_ANALYSIS_WIN_S",
    "QUALITY_SIGNALS",
    "harvest_quality_measurements",
]

QUALITY_ANALYSIS_WIN_S = 0.5
QUALITY_ANALYSIS_HOP_S = 0.25

# Upper spectral edge, not voiced-energy tilt: an 85% roll-off sits at the voiced-energy
# concentration (~1-2 kHz) even for full-band speech and would flag every recording as
# band-limited. A high percentile tracks the actual top of the spectrum.
_ROLLOFF_PCT = 0.95

QUALITY_SIGNALS: tuple[str, ...] = (
    "snr_brouhaha_db",
    "c50_brouhaha_db",
    "snr_spectral_gating_db",
    "snr_peak_db",
    "rolloff_95_hz",
    "proportion_clipped",
    "rms",
)
"""The measurements this module emits, in provenance order."""

# One provenance record per signal. ``resolution_s`` is the analysis hop and ``window_s`` the
# analysis window: they differ, and a consumer assuming hop equals window would treat overlapping
# windows as independent samples.
_UNITS: dict[str, str] = {
    "snr_brouhaha_db": "dB",
    "c50_brouhaha_db": "dB",
    "snr_spectral_gating_db": "dB",
    "snr_peak_db": "dB",
    "rolloff_95_hz": "hertz",
    "proportion_clipped": "proportion",
    # RMS of a float waveform has no absolute reference — it scales with input gain. Saying so is
    # the point of the ``arbitrary`` unit: a consumer must not compare it across recordings.
    "rms": "arbitrary",
}
_MODELS: dict[str, str] = {
    "snr_brouhaha_db": BROUHAHA_MODEL_ID,
    "c50_brouhaha_db": BROUHAHA_MODEL_ID,
    "snr_spectral_gating_db": "senselab.spectral_gating_snr_metric",
    "snr_peak_db": "senselab.peak_snr_from_spectral_metric",
    "rolloff_95_hz": "torch.stft",
    "proportion_clipped": "senselab.proportion_clipped_metric",
    "rms": "senselab.root_mean_square_energy_metric",
}
_REDUCTIONS: dict[str, str | None] = {
    "snr_brouhaha_db": "mean over frames in the analysis window",
    "c50_brouhaha_db": "mean over frames in the analysis window",
    "snr_spectral_gating_db": None,
    "snr_peak_db": None,
    "rolloff_95_hz": f"cumulative-energy quantile at {_ROLLOFF_PCT}, mean over STFT frames",
    "proportion_clipped": None,
    "rms": None,
}


def _provenance(status_by_signal: dict[str, str]) -> dict[str, Any]:
    """Build the provenance block for one analysis window."""
    out: dict[str, Any] = {}
    for name in QUALITY_SIGNALS:
        backend = "brouhaha venv" if _MODELS[name] == BROUHAHA_MODEL_ID else "main env"
        out[name] = SignalProvenance(
            signal=name,
            model=_MODELS[name],
            units=_UNITS[name],
            revision=BROUHAHA_REVISION if _MODELS[name] == BROUHAHA_MODEL_ID else None,
            resolution_s=QUALITY_ANALYSIS_HOP_S,
            window_s=QUALITY_ANALYSIS_WIN_S,
            reduction=_REDUCTIONS[name],
            backend=backend,
            status=status_by_signal.get(name, "ok"),
        ).to_json()
    return out


def _rolloff_hz(slice_audio: Audio) -> Optional[float]:
    """Frequency below which ``_ROLLOFF_PCT`` of spectral energy sits, in Hz.

    Reported as a frequency rather than as ``1 - rolloff / nyquist``: the inversion turns a
    measurement into a badness score, and it hard-codes "band-limited is bad", which is a
    task-dependent judgement. L2 compares it against Nyquist.
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
        # No energy at all: the spectrum has no upper edge to report. A missing measurement, not
        # a measured zero.
        return None
    cumulative = torch.cumsum(power, dim=0) / total
    idx = int(torch.searchsorted(cumulative, torch.tensor(_ROLLOFF_PCT)).item())
    idx = min(idx, power.shape[0] - 1)
    return float(idx * slice_audio.sampling_rate / n_fft)


def _finite_or_none(value: SupportsFloat | None) -> Optional[float]:
    """Coerce to float, or ``None`` when absent or non-finite."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _analysis_window(
    slice_audio: Audio,
    brouhaha: Optional[BrouhahaFrames],
    start_s: float,
    end_s: float,
) -> dict[str, Any]:
    """Measure one analysis window. No thresholds, no anchors, no reductions across estimators."""
    values: dict[str, Optional[float]] = dict.fromkeys(QUALITY_SIGNALS)

    values["rms"] = _finite_or_none(root_mean_square_energy_metric(slice_audio))
    try:
        values["proportion_clipped"] = _finite_or_none(proportion_clipped_metric(slice_audio))
    except (ValueError, TypeError):
        values["proportion_clipped"] = None
    values["rolloff_95_hz"] = _rolloff_hz(slice_audio)

    if brouhaha is not None:
        _vad, b_snr, b_c50 = brouhaha.mean_in_window(start_s, end_s)
        values["snr_brouhaha_db"] = _finite_or_none(b_snr)
        values["c50_brouhaha_db"] = _finite_or_none(b_c50)

    # Both DSP estimators run unconditionally. Their disagreement with Brouhaha is information for
    # L2, so no estimator is skipped on the grounds that another one answered.
    for name, metric in (
        ("snr_spectral_gating_db", spectral_gating_snr_metric),
        ("snr_peak_db", peak_snr_from_spectral_metric),
    ):
        try:
            values[name] = _finite_or_none(metric(slice_audio))
        except (ValueError, TypeError, RuntimeError):
            values[name] = None

    statuses = {name: ("ok" if values[name] is not None else "unavailable") for name in QUALITY_SIGNALS}
    row: dict[str, Any] = dict(values)
    row["provenance"] = _provenance(statuses)
    return row


def harvest_quality_measurements(
    *,
    audio: Audio,
    brouhaha: Optional[BrouhahaFrames],
    grid: BucketGrid,
) -> list[dict[str, Any]]:
    """Return one measurement dict per reporting bucket on ``grid``.

    Args:
        audio: The pass audio.
        brouhaha: Per-frame Brouhaha outputs, or ``None`` when the model was unavailable — its
            two columns are then null (FR-023) while the DSP measurements still land.
        grid: The reporting grid.

    Returns:
        One dict per bucket carrying ``start``, ``end``, the seven values in
        :data:`QUALITY_SIGNALS` (any of which may be ``None`` when its estimator produced
        nothing), and a ``provenance`` block keyed by signal name.

        Values are measurements in the units declared in provenance. Nothing here is a
        ``[0, 1]`` score; use :func:`degradation.scene_degradation` to obtain those.
    """
    duration_s = float(audio.waveform.shape[-1]) / float(audio.sampling_rate)
    if duration_s <= 0:
        return []

    starts = _window_starts(duration_s, QUALITY_ANALYSIS_WIN_S, QUALITY_ANALYSIS_HOP_S)
    analysis: list[dict[str, Any]] = []
    for t in starts:
        end = min(duration_s, t + QUALITY_ANALYSIS_WIN_S)
        window = _analysis_window(_slice_audio(audio, t, end), brouhaha, t, end)
        window["_center"] = 0.5 * (t + end)
        analysis.append(window)

    if not analysis:
        return []
    centers = np.asarray([a["_center"] for a in analysis], dtype=np.float64)

    out: list[dict[str, Any]] = []
    for b_start, b_end, _idx in grid.iter_buckets(duration_s):
        nearest = int(np.argmin(np.abs(centers - 0.5 * (b_start + b_end))))
        src = analysis[nearest]
        row: dict[str, Any] = {"start": b_start, "end": b_end}
        row.update({name: src[name] for name in QUALITY_SIGNALS})
        row["provenance"] = src["provenance"]
        out.append(row)
    return out
