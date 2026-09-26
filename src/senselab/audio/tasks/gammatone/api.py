"""An ERB-spaced gammatone filterbank."""

from __future__ import annotations

import numpy as np
from scipy.signal import gammatone, lfilter

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.envelope.api import analytic_magnitude


def erb_space(low_hz: float, high_hz: float, n_channels: int) -> np.ndarray:
    """Centre frequencies equally spaced on the ERB-rate scale.

    Args:
        low_hz: Lowest centre frequency.
        high_hz: Highest centre frequency.
        n_channels: How many channels.

    Returns:
        Centre frequencies in Hz, ascending.
    """
    to_erb = lambda f: 21.4 * np.log10(4.37e-3 * f + 1.0)  # noqa: E731
    from_erb = lambda e: (10.0 ** (e / 21.4) - 1.0) / 4.37e-3  # noqa: E731
    return from_erb(np.linspace(to_erb(low_hz), to_erb(high_hz), n_channels))


def gammatone_filterbank(
    audio: Audio,
    *,
    n_channels: int,
    low_hz: float,
    high_hz: float,
    hop_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Energy per auditory channel over time.

    Args:
        audio: Mono audio. A multi-channel input is averaged.
        n_channels: Number of ERB-spaced channels. Read it from ``gammatone.n_channels``.
        low_hz: Lowest centre frequency. Read it from ``gammatone.low_hz``.
        high_hz: Highest centre frequency. Read it from ``gammatone.high_hz``.
        hop_s: Frame hop for the energy summary. Read it from ``gammatone.hop_s``.

    Returns:
        ``(centre_frequencies, energy_db)`` with shapes ``(n_channels,)`` and ``(n_channels, n_frames)``.
        ``energy_db`` is absolute dBFS, not normalised to the bank's maximum.
    """
    x = np.asarray(audio.waveform, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=0)
    sr = audio.sampling_rate
    cf = erb_space(low_hz, high_hz, n_channels)
    hop = max(1, int(hop_s * sr))
    n_frames = len(x) // hop
    out = np.zeros((n_channels, n_frames))
    for k, centre in enumerate(cf):
        b, a = gammatone(centre, "iir", fs=sr)
        magnitude = analytic_magnitude(lfilter(b, a, x))
        out[k] = magnitude[: n_frames * hop].reshape(n_frames, hop).mean(axis=1)
    return cf, 20.0 * np.log10(out + 1e-10)
