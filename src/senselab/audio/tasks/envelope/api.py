"""The broadband amplitude envelope, in dBFS, and a floor that tracks the recording."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

from senselab.audio.data_structures import Audio


def hilbert_envelope_dbfs(audio: Audio, *, lowpass_hz: float, filter_order: int) -> np.ndarray:
    """The analytic-signal magnitude, lowpassed, in dBFS.

    The filter is zero-phase, so the envelope is offline-only.

    Args:
        audio: Mono audio. A multi-channel input is averaged.
        lowpass_hz: Cutoff of the zero-phase Butterworth lowpass. Read it from
            ``envelope.lowpass_hz`` in the triage config.
        filter_order: Order of the Butterworth design. Read it from ``envelope.filter_order``.

    Returns:
        One value per input sample, in dBFS, absolute and never normalised by the input's maximum.
        A sample whose filtered envelope is non-positive or below the input representation's
        resolution has no dB value and reads ``nan``.
    """
    source = np.asarray(audio.waveform)
    source_dtype = source.dtype if np.issubdtype(source.dtype, np.floating) else np.dtype(np.float32)
    resolution = float(np.finfo(source_dtype).eps)
    x = np.asarray(source, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=0)
    b, a = butter(filter_order, lowpass_hz / (audio.sampling_rate / 2), "low")
    env = np.asarray(filtfilt(b, a, np.abs(hilbert(x))), dtype=np.float64)
    out = np.full(env.shape, np.nan)
    # A zero-phase lowpass can ring through zero after a transient. Tiny positive values on that
    # crossing are numerical residue, not a measurable acoustic level; admitting them into a local
    # percentile creates implausible downward floor spikes and inflated span contrast.
    measurable = env >= resolution
    out[measurable] = 20.0 * np.log10(env[measurable])
    return out


def rolling_floor_dbfs(
    envelope_db: np.ndarray,
    sampling_rate: int,
    *,
    window_s: float,
    percentile: float,
    eval_grid_s: float,
) -> np.ndarray:
    """A low percentile of the envelope over a sliding window.

    Args:
        envelope_db: Output of :func:`hilbert_envelope_dbfs`, ``nan`` where it had no dB value.
        sampling_rate: Samples per second of ``envelope_db``.
        window_s: Width of the sliding window. Read it from ``floor.window_s``.
        percentile: Which percentile within the window is the floor. Config `floor.percentile`.
        eval_grid_s: How often the percentile is evaluated before interpolation. Read it from
            ``floor.eval_grid_s``.

    Returns:
        One floor value per sample of ``envelope_db``, taken over that window's measured samples.
        A window holding none reads ``nan``, and so does every sample interpolating from it.
    """
    n = len(envelope_db)
    half = int(window_s * sampling_rate) // 2
    step = max(1, int(eval_grid_s * sampling_rate))
    centres = range(0, n, step)
    vals: list[float] = []
    for c in centres:
        window = envelope_db[max(0, c - half) : min(n, c + half)]
        measured = window[np.isfinite(window)]
        vals.append(float(np.percentile(measured, percentile)) if measured.size else float("nan"))
    return np.interp(np.arange(n), centres, vals)
