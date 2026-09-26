"""Content band and long-term average spectrum, over one STFT."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from senselab.audio.data_structures import Audio


@dataclass(frozen=True)
class BandProfile:
    """One recording's spectral band, at the rate it was supplied at.

    Attributes:
        rolloff_hz: Frequency below which ``quantile`` of the energy sits.
        quantile: The cumulative-energy fraction ``rolloff_hz`` was taken at.
        nyquist_hz: Half the sampling rate the profile was measured at.
        sampling_rate: The rate the profile was measured at.
        band_edges_hz: The ``n_bands + 1`` log-spaced band edges, ascending.
        band_centre_hz: The geometric centre of each band, ``n_bands`` of them.
        level_db: Mean power per band in dB, ``n_bands`` of them; ``-inf`` for an empty band.
    """

    rolloff_hz: float
    quantile: float
    nyquist_hz: float
    sampling_rate: int
    band_edges_hz: np.ndarray
    band_centre_hz: np.ndarray
    level_db: np.ndarray


def _mono_frames(audio: Audio) -> torch.Tensor:
    """The waveform as one float32 row.

    Args:
        audio: The audio. A multi-channel input is averaged.

    Returns:
        A one-dimensional float32 tensor.

    Raises:
        ValueError: If the waveform is absent or holds no samples.
    """
    waveform = audio.waveform
    if waveform is None or waveform.numel() == 0:
        raise ValueError("the waveform holds no samples")
    return waveform.detach().to(torch.float32).mean(dim=0).reshape(-1)


def bin_power(audio: Audio, *, n_fft: int, hop_length: int) -> tuple[torch.Tensor, float]:
    """Mean power per frequency bin over the whole signal.

    Args:
        audio: The audio, at whatever rate it was supplied at.
        n_fft: Transform size, also the window length.
        hop_length: Frame hop in samples.

    Returns:
        The per-bin mean power, ``n_fft // 2 + 1`` long, and the bin spacing in Hz.

    Raises:
        ValueError: If the waveform holds no samples, or fewer than ``n_fft`` of them.
    """
    y = _mono_frames(audio)
    if int(y.shape[-1]) < n_fft:
        raise ValueError(f"{int(y.shape[-1])} samples is shorter than the {n_fft}-sample transform")
    spec = torch.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        window=torch.hann_window(n_fft, device=y.device),
        center=True,
        return_complex=True,
    )
    return (spec.abs() ** 2).mean(dim=1), float(audio.sampling_rate) / n_fft


def rolloff_hz(audio: Audio, *, quantile: float, n_fft: int, hop_length: int) -> float | None:
    """Frequency below which ``quantile`` of the spectral energy sits, in Hz.

    Args:
        audio: The audio, at whatever rate it was supplied at.
        quantile: The cumulative-energy fraction, in ``(0, 1]``.
        n_fft: Transform size, also the window length.
        hop_length: Frame hop in samples.

    Returns:
        The frequency in Hz, or None when the spectrum carries no energy at all.

    Raises:
        ValueError: If the waveform holds no samples, or fewer than ``n_fft`` of them.
    """
    power, bin_hz = bin_power(audio, n_fft=n_fft, hop_length=hop_length)
    return _rolloff_from_power(power, bin_hz=bin_hz, quantile=quantile)


def _rolloff_from_power(power: torch.Tensor, *, bin_hz: float, quantile: float) -> float | None:
    """The cumulative-energy quantile of one per-bin power vector.

    Args:
        power: Mean power per frequency bin, ascending in frequency.
        bin_hz: The bin spacing in Hz.
        quantile: The cumulative-energy fraction, in ``(0, 1]``.

    Returns:
        The frequency in Hz, or None when the total energy is not positive.
    """
    total = float(power.sum().item())
    if total <= 0:
        return None
    cumulative = torch.cumsum(power, dim=0) / total
    index = int(torch.searchsorted(cumulative, torch.tensor(quantile)).item())
    return float(min(index, int(power.shape[0]) - 1) * bin_hz)


def long_term_average_spectrum(
    power: torch.Tensor,
    *,
    bin_hz: float,
    low_hz: float,
    high_hz: float,
    n_bands: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean power per log-spaced band, in dB.

    Args:
        power: Mean power per frequency bin, ascending in frequency.
        bin_hz: The bin spacing in Hz.
        low_hz: The bottom edge of the lowest band.
        high_hz: The top edge of the highest band.
        n_bands: How many bands to span ``low_hz`` to ``high_hz`` with.

    Returns:
        ``(band_edges_hz, band_centre_hz, level_db)``, of lengths ``n_bands + 1``, ``n_bands`` and
        ``n_bands``. A band no bin falls in reads ``-inf``.

    Raises:
        ValueError: If ``n_bands`` is not positive, or the band range is not ascending and positive.
    """
    if n_bands < 1:
        raise ValueError(f"n_bands must be positive, not {n_bands}")
    if not 0.0 < low_hz < high_hz:
        raise ValueError(f"the band range must be ascending and positive, not {low_hz}-{high_hz}")
    edges = np.geomspace(low_hz, high_hz, n_bands + 1)
    centres = np.sqrt(edges[:-1] * edges[1:])
    values = power.detach().to(torch.float64).numpy()
    frequencies = np.arange(values.shape[0]) * bin_hz
    levels = np.full(n_bands, -np.inf)
    for band in range(n_bands):
        inside = (frequencies >= edges[band]) & (frequencies < edges[band + 1])
        if bool(inside.any()):
            mean = float(values[inside].mean())
            levels[band] = 10.0 * np.log10(mean) if mean > 0.0 else -np.inf
    return edges, centres, levels


def band_profile(
    audio: Audio,
    *,
    quantile: float,
    n_fft: int,
    hop_length: int,
    ltas_low_hz: float,
    ltas_bands: int,
) -> BandProfile:
    """The content band and the long-term average spectrum, from one transform.

    Args:
        audio: The audio, at whatever rate it was supplied at. Read it un-resampled: a profile
            taken after a resample reports the resampler's ceiling, not the content's.
        quantile: The cumulative-energy fraction the roll-off is taken at.
        n_fft: Transform size, also the window length.
        hop_length: Frame hop in samples.
        ltas_low_hz: The bottom edge of the lowest band. The top edge is Nyquist.
        ltas_bands: How many log-spaced bands span ``ltas_low_hz`` to Nyquist.

    Returns:
        The profile.

    Raises:
        ValueError: If the waveform holds no samples, holds fewer than ``n_fft`` of them, carries no
            energy at all, or the band layout is not realisable at this rate.
    """
    power, bin_hz = bin_power(audio, n_fft=n_fft, hop_length=hop_length)
    rolloff = _rolloff_from_power(power, bin_hz=bin_hz, quantile=quantile)
    if rolloff is None:
        raise ValueError("the spectrum carries no energy; there is no band edge to report")
    nyquist = float(audio.sampling_rate) / 2.0
    edges, centres, levels = long_term_average_spectrum(
        power, bin_hz=bin_hz, low_hz=ltas_low_hz, high_hz=nyquist, n_bands=ltas_bands
    )
    return BandProfile(
        rolloff_hz=rolloff,
        quantile=quantile,
        nyquist_hz=nyquist,
        sampling_rate=int(audio.sampling_rate),
        band_edges_hz=edges,
        band_centre_hz=centres,
        level_db=levels,
    )
