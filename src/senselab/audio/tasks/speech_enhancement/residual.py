"""What remains of a reference signal after a lag-aligned, gain-fitted stream is subtracted from it.

Lag search, gain fit, subtraction, band energies and the energy/correlation fractions two kinds of
caller both need from one such subtraction: the triage PREPROCESS residual block
(``senselab.audio.workflows.triage.nodes.preprocess``), which relates ``plain`` to FRCRN's own
enhancement of it, and the standalone comparison scripts under
``specs/20260817-triage-workflow-dag/benchmarks/scripts/`` and a working ``subtract.py`` tool that
predates this module. One implementation here is what keeps those from silently drifting apart in
what they compute.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import correlate, correlation_lags


def find_lag(reference: np.ndarray, signal: np.ndarray, sampling_rate: int, max_lag_ms: float) -> int:
    """The integer-sample lag of ``signal`` relative to ``reference``, by cross-correlation.

    Args:
        reference: The reference signal.
        signal: The signal being searched for a lag against ``reference``.
        sampling_rate: Both signals' sampling rate, in Hz.
        max_lag_ms: The search half-window, in milliseconds.

    Returns:
        The lag in samples. Positive means ``signal`` arrives later than ``reference``; aligning
        drops the first ``lag`` samples of ``signal`` (or the last ``-lag`` samples of
        ``reference``, when negative).
    """
    max_lag = max(1, int(round(max_lag_ms * sampling_rate / 1000.0)))
    n = min(len(reference), len(signal))
    r, s = reference[:n], signal[:n]
    corr = correlate(r, s, mode="full", method="fft")
    lags = correlation_lags(len(r), len(s), mode="full")
    mask = np.abs(lags) <= max_lag
    idx = np.argmax(corr[mask])
    return -int(lags[mask][idx])


def align(reference: np.ndarray, signal: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Trim ``reference``/``signal`` to their overlapping region after shifting ``signal`` by ``lag``.

    Args:
        reference: The reference signal.
        signal: The signal to align, as returned by :func:`find_lag`.
        lag: The lag in samples, in :func:`find_lag`'s sign convention.

    Returns:
        ``(reference, signal)``, trimmed to their common length.
    """
    if lag > 0:
        signal = signal[lag:]
    elif lag < 0:
        reference = reference[-lag:]
    n = min(len(reference), len(signal))
    return reference[:n], signal[:n]


def fit_gain(reference: np.ndarray, signal: np.ndarray) -> float:
    """The least-squares gain minimising ``||reference - g * signal||``: ``g = <ref, sig> / <sig, sig>``.

    Args:
        reference: The reference signal, aligned with ``signal``.
        signal: The signal being scaled onto ``reference``.

    Returns:
        The fitted gain, or 0.0 when ``signal`` carries no energy.
    """
    denom = float(np.dot(signal, signal))
    if denom == 0.0:
        return 0.0
    return float(np.dot(reference, signal) / denom)


def band_energy_fractions(x: np.ndarray, sampling_rate: int, bands_hz: list[tuple[float, float]]) -> dict[str, float]:
    """The fraction of ``x``'s spectral energy landing in each ``(lo, hi)`` Hz band.

    Args:
        x: The signal.
        sampling_rate: ``x``'s sampling rate, in Hz.
        bands_hz: The band edges to report over.

    Returns:
        ``{"lo_hi": fraction, ...}`` keyed by each band's edges. Every fraction is 0.0 when ``x``
        carries no energy.
    """
    n = len(x)
    if n == 0:
        return {f"{lo:g}_{hi:g}": 0.0 for lo, hi in bands_hz}
    spectrum = np.fft.rfft(x)
    power = np.square(np.abs(spectrum))
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
    total = float(power.sum())
    out: dict[str, float] = {}
    for lo, hi in bands_hz:
        mask = (freqs >= lo) & (freqs < hi)
        out[f"{lo:g}_{hi:g}"] = float(power[mask].sum() / total) if total > 0 else 0.0
    return out


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    """The Pearson correlation coefficient of two signals, trimmed to their common length.

    Gain-scaling one signal by a positive factor does not change this value; scaling by a negative
    one flips its sign. This is the instrument that told the two `MossFormer2_SS_16K` separation
    streams apart in the buzz-separation comparison (0.984 vs 0.077 against the input) and is used
    here for the same purpose: telling a stream that is essentially the input from one that carries
    almost none of it.

    Args:
        a: The first signal.
        b: The second signal.

    Returns:
        The correlation coefficient, or ``nan`` when fewer than two samples remain after trimming to
        the common length, or either signal carries zero variance.
    """
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if n < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


@dataclass(frozen=True)
class ResidualComputation:
    """One lag-aligned, gain-fitted subtraction and every quantity derived from it.

    Attributes:
        sampling_rate: The rate both aligned signals share, in Hz.
        lag_samples: ``signal``'s lag relative to ``reference``, in :func:`find_lag`'s convention.
        reference_aligned: ``reference``, trimmed to the overlapping region.
        signal_aligned: ``signal``, lag-aligned to ``reference`` -- NOT gain-scaled.
        gain: The least-squares gain fit on the aligned pair.
        residual: ``reference_aligned - gain * signal_aligned``.
        input_energy: ``reference_aligned``'s own energy (sum of squares).
        signal_energy: ``signal_aligned``'s own energy, before any gain scaling.
        residual_energy: ``residual``'s own energy.
        signal_energy_fraction: ``signal_energy / input_energy``; ``nan`` when ``input_energy`` is
            zero.
        residual_energy_fraction: ``residual_energy / input_energy``; ``nan`` when ``input_energy``
            is zero.
        correlation_signal: :func:`correlation` of ``reference_aligned`` and ``signal_aligned``.
        correlation_residual: :func:`correlation` of ``reference_aligned`` and ``residual``.
    """

    sampling_rate: int
    lag_samples: int
    reference_aligned: np.ndarray
    signal_aligned: np.ndarray
    gain: float
    residual: np.ndarray
    input_energy: float
    signal_energy: float
    residual_energy: float
    signal_energy_fraction: float
    residual_energy_fraction: float
    correlation_signal: float
    correlation_residual: float

    @property
    def gain_db(self) -> float:
        """``gain`` in dB; ``-inf`` when ``gain`` is exactly zero."""
        return float(20.0 * np.log10(abs(self.gain))) if self.gain != 0.0 else float("-inf")

    @property
    def lag_ms(self) -> float:
        """``lag_samples`` converted to milliseconds at ``sampling_rate``."""
        return self.lag_samples / self.sampling_rate * 1000.0


def compute_residual(
    reference: np.ndarray,
    signal: np.ndarray,
    sampling_rate: int,
    *,
    max_lag_ms: float,
) -> ResidualComputation:
    """Lag-align ``signal`` to ``reference``, fit gain, subtract, and report every derived quantity.

    Args:
        reference: The reference signal (e.g. the pipeline's ``plain`` stream).
        signal: The enhancement or separation output being related to ``reference``.
        sampling_rate: Both signals' sampling rate, in Hz.
        max_lag_ms: :func:`find_lag`'s search half-window, in milliseconds.

    Returns:
        The full computation. Neither energy fraction is gated here: whether a fraction means
        anything is a question for a downstream consumer (one that can also look at what each
        stream was classified as), not a precondition on producing the arrays.
    """
    lag = find_lag(reference, signal, sampling_rate, max_lag_ms)
    ref_aligned, sig_aligned = align(reference, signal, lag)
    input_energy = float(np.sum(np.square(ref_aligned)))
    signal_energy = float(np.sum(np.square(sig_aligned)))
    gain = fit_gain(ref_aligned, sig_aligned)
    residual = ref_aligned - gain * sig_aligned
    residual_energy = float(np.sum(np.square(residual)))
    signal_energy_fraction = signal_energy / input_energy if input_energy > 0.0 else float("nan")
    residual_energy_fraction = residual_energy / input_energy if input_energy > 0.0 else float("nan")
    return ResidualComputation(
        sampling_rate=sampling_rate,
        lag_samples=lag,
        reference_aligned=ref_aligned,
        signal_aligned=sig_aligned,
        gain=gain,
        residual=residual,
        input_energy=input_energy,
        signal_energy=signal_energy,
        residual_energy=residual_energy,
        signal_energy_fraction=signal_energy_fraction,
        residual_energy_fraction=residual_energy_fraction,
        correlation_signal=correlation(ref_aligned, sig_aligned),
        correlation_residual=correlation(ref_aligned, residual),
    )
