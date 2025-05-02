"""Audio Quality Control Checks.

Each check function processes a list of Audio objects and returns a dictionary
indicating which files should be excluded or reviewed, along with a list of
audio files that passed the check.

All checks take a list of Audio objects as input and return:
    - A dictionary with the structure:
        {
            "exclude": [Audio],  # List of Audio objects that failed the check
            "review": [Audio]    # List of Audio objects that need manual review
        }
    - A list of Audio objects that passed the check.
"""

import os
from typing import Callable, Dict, List, Optional

import pandas as pd
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.bioacoustic_qc.metrics import (
    amplitude_headroom_metric,
    amplitude_interquartile_range_metric,
    amplitude_kurtosis_metric,
    amplitude_modulation_depth_metric,
    amplitude_skew_metric,
    crest_factor_metric,
    dynamic_range_metric,
    mean_absolute_deviation_metric,
    peak_snr_from_spectral_metric,
    phase_correlation_metric,
    proportion_clipped_metric,
    proportion_silence_at_beginning_metric,
    proportion_silence_at_end_metric,
    proportion_silent_metric,
    root_mean_square_energy_metric,
    shannon_entropy_amplitude_metric,
    signal_variance_metric,
    spectral_gating_snr_metric,
    zero_crossing_rate_metric,
)


def get_metric(
    audio: Audio,
    metric_func: Callable[[Audio], float],
    df: Optional[pd.DataFrame] = None,
) -> float:
    """Return a metric, using a cached DataFrame when possible.

    Args:
        audio: The Audio instance for which the metric is required.
        metric_func: The metric function, e.g. ``zero_crossing_rate_metric``.
        df: Optional DataFrame that already contains pre-computed metrics.
            The DataFrame must have:
              * a column ``'audio_path_or_id'`` holding file names, and
              * a column named exactly ``metric_func.__name__``.

    Returns:
        The metric value for this ``audio`` item. If ``df`` is provided and
        contains the value, that cached value is returned; otherwise the metric
        is computed directly.
    """
    metric_name = metric_func.__name__

    filepath = audio.filepath()
    if df is not None and filepath:
        audio_file_name = os.path.basename(filepath)
        row = df[df["audio_path_or_id"] == audio_file_name]
        if not row.empty and metric_name in row.columns:
            return row[metric_name].iloc[0]

    return metric_func(audio)


def audio_length_positive_check(audio: Audio) -> bool:
    """Check that the waveform is not empty.

    Args:
        audio: Audio object to evaluate.

    Returns:
        True if the waveform contains one or more samples, else False.
    """
    return audio.waveform.numel() != 0


def very_low_headroom_check(
    audio: Audio,
    headroom_threshold: float = 0.005,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect signals within ``headroom_threshold`` of clipping.

    Args:
        audio: Audio object to evaluate.
        headroom_threshold: Maximum acceptable positive/negative headroom.
        df: Optional DataFrame containing a pre-computed
            ``amplitude_headroom_metric`` column.

    Returns:
        True when headroom < ``headroom_threshold``.
    """
    return get_metric(audio, amplitude_headroom_metric, df) < headroom_threshold


def very_high_headroom_check(
    audio: Audio,
    headroom_threshold: float = 0.95,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect overly quiet signals with excessive headroom.

    Args:
        audio: Audio object to evaluate.
        headroom_threshold: Minimum headroom fraction considered excessive.
        df: Optional DataFrame with ``amplitude_headroom_metric`` cached.

    Returns:
        True when headroom > ``headroom_threshold``.
    """
    return get_metric(audio, amplitude_headroom_metric, df) > headroom_threshold


def very_low_amplitude_interquartile_range_check(
    audio: Audio,
    threshold: float = 0.01,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect nearly flat audio via an extremely low amplitude IQR.

    Args:
        audio: Audio object to evaluate.
        threshold: Maximum IQR considered *very low*.
        df: Optional DataFrame containing ``amplitude_interquartile_range_metric``.

    Returns:
        True when IQR < ``threshold``.
    """
    return get_metric(audio, amplitude_interquartile_range_metric, df) < threshold


def very_high_amplitude_interquartile_range_check(
    audio: Audio,
    threshold: float = 1.5,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect very wide amplitude spread (noisy or clipped).

    Args:
        audio: Audio object to evaluate.
        threshold: Minimum IQR considered *very high*.
        df: Optional DataFrame containing ``amplitude_interquartile_range_metric``.

    Returns:
        True when IQR > ``threshold``.
    """
    return get_metric(audio, amplitude_interquartile_range_metric, df) > threshold


def very_low_amplitude_kurtosis_check(
    audio: Audio,
    threshold: float = -100,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Flag extremely flat/noisy distributions (low kurtosis).

    Args:
        audio: Audio object to evaluate.
        threshold: Upper bound regarded as *very low* kurtosis.
        df: Optional DataFrame with ``amplitude_kurtosis_metric``.

    Returns:
        True when kurtosis < ``threshold``.
    """
    return get_metric(audio, amplitude_kurtosis_metric, df) < threshold


def very_high_amplitude_kurtosis_check(
    audio: Audio,
    threshold: float = 100,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Flag spiky or clipped distributions (high kurtosis).

    Args:
        audio: Audio object to evaluate.
        threshold: Lower bound regarded as *very high* kurtosis.
        df: Optional DataFrame with ``amplitude_kurtosis_metric``.

    Returns:
        True when kurtosis > ``threshold``.
    """
    return get_metric(audio, amplitude_kurtosis_metric, df) > threshold


def very_low_amplitude_modulation_depth_check(
    audio: Audio,
    threshold: float = 0.1,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect audio with almost no loudness variation.

    Args:
        audio: Audio object to evaluate.
        threshold: Maximum modulation depth regarded as *very low*.
        df: Optional DataFrame with ``amplitude_modulation_depth_metric``.

    Returns:
        True when modulation depth < ``threshold``.
    """
    return get_metric(audio, amplitude_modulation_depth_metric, df) < threshold


def low_amplitude_modulation_depth_check(
    audio: Audio,
    min: float = 0.1,
    max: float = 0.3,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect audio with modestly low loudness variation.

    Args:
        audio: Audio object to evaluate.
        min: Inclusive lower bound for *low* modulation depth.
        max: Exclusive upper bound for *low* modulation depth.
        df: Optional DataFrame with ``amplitude_modulation_depth_metric``.

    Returns:
        True when ``min ≤ depth < max``.
    """
    depth = get_metric(audio, amplitude_modulation_depth_metric, df)
    return min <= depth < max


def proportion_clipped_check(
    audio: Audio,
    threshold: float = 0.0001,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Pass audio with at most ``threshold`` clipped samples.

    Args:
        audio: Audio object to evaluate.
        threshold: Maximum fraction of clipped samples allowed.
        df: Optional DataFrame with ``proportion_clipped_metric``.

    Returns:
        True when clipped proportion < ``threshold``.
    """
    return get_metric(audio, proportion_clipped_metric, df) < threshold


def clipping_present_check(
    audio: Audio,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect whether *any* clipping is present.

    Args:
        audio: Audio object to evaluate.
        df: Optional DataFrame with ``proportion_clipped_metric``.

    Returns:
        True when clipped proportion > 0.
    """
    return get_metric(audio, proportion_clipped_metric, df) > 0


def completely_silent_check(
    audio: Audio,
    silent_proportion: float = 1.0,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect recordings that are entirely silent.

    Args:
        audio: Audio object to evaluate.
        silent_proportion: Proportion that defines *complete* silence
            (default 1.0 means 100 %).
        df: Optional DataFrame with ``proportion_silent_metric``.

    Returns:
        True when silent proportion ≥ ``silent_proportion``.
    """
    return get_metric(audio, proportion_silent_metric, df) >= silent_proportion


def mostly_silent_check(
    audio: Audio,
    silent_proportion: float = 0.95,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect recordings that are mostly silent.

    Args:
        audio: Audio object to evaluate.
        silent_proportion: Threshold above which audio is considered *mostly*
            silent.
        df: Optional DataFrame with ``proportion_silent_metric``.

    Returns:
        True when silent proportion > ``silent_proportion``.
    """
    return get_metric(audio, proportion_silent_metric, df) > silent_proportion


def high_amplitude_skew_magnitude_check(
    audio: Audio,
    magnitude: float = 5.0,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Pass audio whose amplitude skew magnitude is ≤ ``magnitude``.

    Args:
        audio: Audio object to evaluate.
        magnitude: Maximum acceptable |skew|.
        df: Optional DataFrame with ``amplitude_skew_metric``.

    Returns:
        True when |skew| ≤ ``magnitude``.
    """
    return abs(get_metric(audio, amplitude_skew_metric, df)) <= magnitude


def high_crest_factor_check(
    audio: Audio,
    threshold: float = 20.0,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect spiky signals with crest factor ≥ ``threshold``.

    Args:
        audio: Audio object to evaluate.
        threshold: Minimum crest factor regarded as too high.
        df: Optional DataFrame with ``crest_factor_metric``.

    Returns:
        True when crest factor ≥ ``threshold``.
    """
    return get_metric(audio, crest_factor_metric, df) >= threshold


def low_crest_factor_check(
    audio: Audio,
    threshold: float = 1.5,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect over-compressed signals with crest factor ≤ ``threshold``.

    Args:
        audio: Audio object to evaluate.
        threshold: Maximum crest factor regarded as too low.
        df: Optional DataFrame with ``crest_factor_metric``.

    Returns:
        True when crest factor ≤ ``threshold``.
    """
    return get_metric(audio, crest_factor_metric, df) <= threshold


def very_low_dynamic_range_check(
    audio: Audio,
    threshold: float = 0.1,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect signals with insufficient dynamic range.

    Args:
        audio: Audio object to evaluate.
        threshold: Maximum dynamic range considered too low.
        df: Optional DataFrame with ``dynamic_range_metric``.

    Returns:
        True when dynamic range < ``threshold``.
    """
    return get_metric(audio, dynamic_range_metric, df) < threshold


def very_high_dynamic_range_check(
    audio: Audio,
    threshold: float = 1.9,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect signals with suspiciously high dynamic range.

    Args:
        audio: Audio object to evaluate.
        threshold: Minimum dynamic range considered too high.
        df: Optional DataFrame with ``dynamic_range_metric``.

    Returns:
        True when dynamic range > ``threshold``.
    """
    return get_metric(audio, dynamic_range_metric, df) > threshold


def very_low_mean_absolute_deviation_check(
    audio: Audio,
    threshold: float = 0.001,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect nearly flat signals via very low MAD.

    Args:
        audio: Audio object to evaluate.
        threshold: Maximum MAD regarded as too low.
        df: Optional DataFrame with ``mean_absolute_deviation_metric``.

    Returns:
        True when MAD < ``threshold``.
    """
    return get_metric(audio, mean_absolute_deviation_metric, df) < threshold


def very_high_mean_absolute_deviation_check(
    audio: Audio,
    threshold: float = 0.5,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect overly volatile signals via very high MAD.

    Args:
        audio: Audio object to evaluate.
        threshold: Minimum MAD regarded as too high.
        df: Optional DataFrame with ``mean_absolute_deviation_metric``.

    Returns:
        True when MAD > ``threshold``.
    """
    return get_metric(audio, mean_absolute_deviation_metric, df) > threshold


def very_low_peak_snr_from_spectral_check(
    audio: Audio,
    threshold: float = 10.0,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect audio with very low peak-SNR (< ``threshold``).

    Args:
        audio: Audio object to evaluate.
        threshold: Minimum acceptable peak-SNR (dB).
        df: Optional DataFrame with ``peak_snr_from_spectral_metric``.

    Returns:
        True when peak-SNR < ``threshold``.
    """
    return get_metric(audio, peak_snr_from_spectral_metric, df) < threshold


def low_peak_snr_from_spectral_check(
    audio: Audio,
    lower: float = 10.0,
    upper: float = 20.0,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Review audio whose peak-SNR falls in a *low* band.

    Args:
        audio: Audio object to evaluate.
        lower: Inclusive lower bound for the low SNR band.
        upper: Exclusive upper bound.
        df: Optional DataFrame with ``peak_snr_from_spectral_metric``.

    Returns:
        True when ``lower ≤ SNR < upper``.
    """
    snr = get_metric(audio, peak_snr_from_spectral_metric, df)
    return lower <= snr < upper


def very_high_peak_snr_from_spectral_check(
    audio: Audio,
    threshold: float = 60.0,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Flag unrealistically high peak-SNR (> ``threshold``).

    Args:
        audio: Audio object to evaluate.
        threshold: Peak-SNR regarded as suspiciously high (dB).
        df: Optional DataFrame with ``peak_snr_from_spectral_metric``.

    Returns:
        True when peak-SNR > ``threshold``.
    """
    return get_metric(audio, peak_snr_from_spectral_metric, df) > threshold


def low_phase_correlation_check(
    audio: Audio,
    threshold: float = 0.99,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect stereo signals with weak channel correlation.

    Args:
        audio: Audio object to evaluate.
        threshold: Minimum acceptable correlation coefficient.
        df: Optional DataFrame with ``phase_correlation_metric``.

    Returns:
        True when correlation < ``threshold``.
    """
    return get_metric(audio, phase_correlation_metric, df) < threshold


def high_proportion_silence_at_beginning_check(
    audio: Audio,
    threshold: float = 0.2,
) -> bool:
    """Flag recordings with > ``threshold`` leading silence.

    Note: This metric cannot use ``get_metric`` because the metric
    requires no DataFrame but **does** use its default internal threshold.

    Args:
        audio: Audio object to evaluate.
        threshold: Maximum acceptable leading silence proportion.

    Returns:
        True when leading silence proportion > ``threshold``.
    """
    return proportion_silence_at_beginning_metric(audio) > threshold


def high_proportion_silence_at_end_check(
    audio: Audio,
    threshold: float = 0.2,
) -> bool:
    """Flag recordings with > ``threshold`` trailing silence.

    Args:
        audio: Audio object to evaluate.
        threshold: Maximum acceptable trailing silence proportion.

    Returns:
        True when trailing silence proportion > ``threshold``.
    """
    return proportion_silence_at_end_metric(audio) > threshold


def very_low_root_mean_square_energy_check(
    audio: Audio,
    threshold: float = 0.005,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect audio that is too quiet (very low RMS).

    Args:
        audio: Audio object to evaluate.
        threshold: Maximum RMS energy regarded as very low.
        df: Optional DataFrame with ``root_mean_square_energy_metric``.

    Returns:
        True when RMS < ``threshold``.
    """
    return get_metric(audio, root_mean_square_energy_metric, df) < threshold


def very_high_root_mean_square_energy_check(
    audio: Audio,
    threshold: float = 0.5,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect audio that is too loud (very high RMS).

    Args:
        audio: Audio object to evaluate.
        threshold: Minimum RMS energy regarded as very high.
        df: Optional DataFrame with ``root_mean_square_energy_metric``.

    Returns:
        True when RMS > ``threshold``.
    """
    return get_metric(audio, root_mean_square_energy_metric, df) > threshold


def low_shannon_entropy_amplitude_check(
    audio: Audio,
    threshold: float = 2.0,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect overly predictable audio via low entropy.

    Args:
        audio: Audio object to evaluate.
        threshold: Maximum entropy regarded as too low (bits).
        df: Optional DataFrame with ``shannon_entropy_amplitude_metric``.

    Returns:
        True when entropy < ``threshold``.
    """
    return get_metric(audio, shannon_entropy_amplitude_metric, df) < threshold


def high_shannon_entropy_amplitude_check(
    audio: Audio,
    threshold: float = 7.5,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect overly noisy audio via high entropy.

    Args:
        audio: Audio object to evaluate.
        threshold: Minimum entropy regarded as too high (bits).
        df: Optional DataFrame with ``shannon_entropy_amplitude_metric``.

    Returns:
        True when entropy > ``threshold``.
    """
    return get_metric(audio, shannon_entropy_amplitude_metric, df) > threshold


def low_signal_variance_check(
    audio: Audio,
    threshold: float = 1e-4,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect signals with extremely small amplitude variance.

    Args:
        audio: Audio object to evaluate.
        threshold: Maximum variance regarded as too low.
        df: Optional DataFrame with ``signal_variance_metric``.

    Returns:
        True when variance < ``threshold``.
    """
    return get_metric(audio, signal_variance_metric, df) < threshold


def high_signal_variance_check(
    audio: Audio,
    threshold: float = 0.3,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect signals with excessively large amplitude variance.

    Args:
        audio: Audio object to evaluate.
        threshold: Minimum variance regarded as too high.
        df: Optional DataFrame with ``signal_variance_metric``.

    Returns:
        True when variance > ``threshold``.
    """
    return get_metric(audio, signal_variance_metric, df) > threshold


def low_spectral_gating_snr_check(
    audio: Audio,
    threshold: float = 10.0,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect recordings whose segmental SNR is < ``threshold`` dB.

    Args:
        audio: Audio object to evaluate.
        threshold: Minimum acceptable segmental SNR (dB).
        df: Optional DataFrame with ``spectral_gating_snr_metric``.

    Returns:
        True when SNR < ``threshold``.
    """
    return get_metric(audio, spectral_gating_snr_metric, df) < threshold


def high_spectral_gating_snr_check(
    audio: Audio,
    threshold: float = 60.0,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Flag recordings whose segmental SNR is > ``threshold`` dB.

    Args:
        audio: Audio object to evaluate.
        threshold: Segmental SNR considered suspiciously high (dB).
        df: Optional DataFrame with ``spectral_gating_snr_metric``.

    Returns:
        True when SNR > ``threshold``.
    """
    return get_metric(audio, spectral_gating_snr_metric, df) > threshold


def low_zero_crossing_rate_metric_check(
    audio: Audio,
    threshold: float = 0.01,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect signals with ZCR below ``threshold`` (likely silent/DC).

    Args:
        audio: Audio object to evaluate.
        threshold: Minimum acceptable zero-crossing rate.
        df: Optional DataFrame with ``zero_crossing_rate_metric``.

    Returns:
        True when ZCR < ``threshold``.
    """
    return get_metric(audio, zero_crossing_rate_metric, df) < threshold


def high_zero_crossing_rate_metric_check(
    audio: Audio,
    lower: float = 0.15,
    upper: float = 0.3,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Review signals with ZCR in a high (but not extreme) range.

    Args:
        audio: Audio object to evaluate.
        lower: Inclusive lower bound for the *high* ZCR range.
        upper: Exclusive upper bound for the *high* ZCR range.
        df: Optional DataFrame with ``zero_crossing_rate_metric``.

    Returns:
        True when ``lower ≤ ZCR < upper``.
    """
    zcr = get_metric(audio, zero_crossing_rate_metric, df)
    return lower <= zcr < upper


def very_high_zero_crossing_rate_metric_check(
    audio: Audio,
    threshold: float = 0.3,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect signals with extremely high ZCR (noise / corruption).

    Args:
        audio: Audio object to evaluate.
        threshold: Zero-crossing rate regarded as *very high*.
        df: Optional DataFrame with ``zero_crossing_rate_metric``.

    Returns:
        True when ZCR > ``threshold``.
    """
    return get_metric(audio, zero_crossing_rate_metric, df) > threshold
