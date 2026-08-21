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
        One dBFS value per input sample. Absolute, never normalised by the input's maximum.
    """
    x = np.asarray(audio.waveform, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=0)
    b, a = butter(filter_order, lowpass_hz / (audio.sampling_rate / 2), "low")
    env = np.maximum(filtfilt(b, a, np.abs(hilbert(x))), 1e-12)
    return 20.0 * np.log10(env)


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
        envelope_db: Output of :func:`hilbert_envelope_dbfs`.
        sampling_rate: Samples per second of ``envelope_db``.
        window_s: Width of the sliding window. Read it from ``floor.window_s``.
        percentile: Which percentile within the window is the floor. Config `floor.percentile`.
        eval_grid_s: How often the percentile is evaluated before interpolation. Read it from
            ``floor.eval_grid_s``.

    Returns:
        One floor value per sample of ``envelope_db``.
    """
    n = len(envelope_db)
    half = int(window_s * sampling_rate) // 2
    step = max(1, int(eval_grid_s * sampling_rate))
    centres = range(0, n, step)
    vals = [float(np.percentile(envelope_db[max(0, c - half) : min(n, c + half)], percentile)) for c in centres]
    return np.interp(np.arange(n), centres, vals)
