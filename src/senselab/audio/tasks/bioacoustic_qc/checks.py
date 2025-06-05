"""Contains tests for bioacoustic quality control checks.

This module defines a set of boolean functions that evaluate individual `Audio` objects
based on specific quality metrics (e.g., clipping, dynamic range, SNR, entropy).

Each check is designed to return `True` when a defined failure condition is met,
enabling easy filtering of problematic recordings.

All checks accept an `Audio` object and optionally a DataFrame of cached metric values.
"""

import os
from typing import Callable, Optional, Union

import pandas as pd

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
    audio_or_path: Union[Audio, str],
    metric_function: Callable[[Audio], float],
    df: Optional[pd.DataFrame] = None,
) -> float:
    """Return a metric, using a cached DataFrame when possible.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        metric_function: The metric function, e.g. ``zero_crossing_rate_metric``.
        df: Optional DataFrame that already contains pre-computed metrics.
            The DataFrame must have:
              * a column ``'audio_path_or_id'`` holding file names, and
              * a column named exactly ``metric_function.__name__``.

    Returns:
        The metric value for this ``audio`` item. If ``df`` is provided and
        contains the value, that cached value is returned; otherwise the metric
        is computed directly and optionally added to ``df``.
    """
    metric_name = metric_function.__name__

    filepath = None
    if isinstance(audio_or_path, str):
        filepath = audio_or_path
    else:
        filepath = audio_or_path.filepath()

    metric = None
    if df is not None and filepath:
        audio_file_name = os.path.basename(filepath)
        row = df[df["audio_path_or_id"] == audio_file_name]
        if not row.empty and metric_name in row.columns:
            metric = row[metric_name].iloc[0]

    if metric is None and isinstance(audio_or_path, Audio):
        metric = metric_function(audio_or_path)
        if df is not None and filepath:
            audio_file_name = os.path.basename(filepath)
            if metric_name not in df.columns:
                df[metric_name] = pd.NA
            df.loc[df["audio_path_or_id"] == audio_file_name, metric_name] = metric

    if metric is None:
        raise ValueError("Expected metric to be non-None.")

    return metric


def audio_length_positive_check(audio: Audio) -> bool:
    """Check that the waveform is not empty.

    Args:
        audio: Audio object to evaluate.

    Returns:
        True if the waveform contains one or more samples, else False.
    """
    return audio.waveform.numel() != 0


def very_low_headroom_check(
    audio_or_path: Union[Audio, str],
    headroom_threshold: float = 0.005,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect signals within ``headroom_threshold`` of clipping.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        headroom_threshold: Maximum acceptable positive/negative headroom.
        df: Optional DataFrame containing a pre-computed
            ``amplitude_headroom_metric`` column.

    Returns:
        True when headroom < ``headroom_threshold``.
    """
    return get_metric(audio_or_path, amplitude_headroom_metric, df) < headroom_threshold


def very_high_headroom_check(
    audio_or_path: Union[Audio, str],
    headroom_threshold: float = 0.95,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect overly quiet signals with excessive headroom.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        headroom_threshold: Minimum headroom fraction considered excessive.
        df: Optional DataFrame with ``amplitude_headroom_metric`` cached.

    Returns:
        True when headroom > ``headroom_threshold``.
    """
    return get_metric(audio_or_path, amplitude_headroom_metric, df) > headroom_threshold


def very_high_amplitude_interquartile_range_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 1.5,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect very wide amplitude spread (noisy or clipped).

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Minimum IQR considered *very high*.
        df: Optional DataFrame containing ``amplitude_interquartile_range_metric``.

    Returns:
        True when IQR > ``threshold``.
    """
    return get_metric(audio_or_path, amplitude_interquartile_range_metric, df) > threshold


def very_low_amplitude_kurtosis_check(
    audio_or_path: Union[Audio, str],
    threshold: float = -100,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Flag extremely flat/noisy distributions (low kurtosis).

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Upper bound regarded as *very low* kurtosis.
        df: Optional DataFrame with ``amplitude_kurtosis_metric``.

    Returns:
        True when kurtosis < ``threshold``.
    """
    return get_metric(audio_or_path, amplitude_kurtosis_metric, df) < threshold


def very_high_amplitude_kurtosis_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 100,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Flag spiky or clipped distributions (high kurtosis).

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Lower bound regarded as *very high* kurtosis.
        df: Optional DataFrame with ``amplitude_kurtosis_metric``.

    Returns:
        True when kurtosis > ``threshold``.
    """
    return get_metric(audio_or_path, amplitude_kurtosis_metric, df) > threshold


def very_low_amplitude_modulation_depth_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 0.1,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect audio with almost no loudness variation.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Maximum modulation depth regarded as *very low*.
        df: Optional DataFrame with ``amplitude_modulation_depth_metric``.

    Returns:
        True when modulation depth < ``threshold``.
    """
    return get_metric(audio_or_path, amplitude_modulation_depth_metric, df) < threshold


def low_amplitude_modulation_depth_check(
    audio_or_path: Union[Audio, str],
    min: float = 0.1,
    max: float = 0.3,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect audio with modestly low loudness variation.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        min: Inclusive lower bound for *low* modulation depth.
        max: Exclusive upper bound for *low* modulation depth.
        df: Optional DataFrame with ``amplitude_modulation_depth_metric``.

    Returns:
        True when ``min ≤ depth < max``.
    """
    depth = get_metric(audio_or_path, amplitude_modulation_depth_metric, df)
    return min <= depth < max


def high_proportion_clipped_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 0.0001,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect audio with a high proportion of clipped samples.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Maximum fraction of clipped samples allowed.
        df: Optional DataFrame with ``proportion_clipped_metric``.

    Returns:
        True when clipped proportion < ``threshold``.
    """
    return get_metric(audio_or_path, proportion_clipped_metric, df) > threshold


def clipping_present_check(
    audio_or_path: Union[Audio, str],
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect whether *any* clipping is present.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        df: Optional DataFrame with ``proportion_clipped_metric``.

    Returns:
        True when clipped proportion > 0.
    """
    return get_metric(audio_or_path, proportion_clipped_metric, df) > 0


def completely_silent_check(
    audio_or_path: Union[Audio, str],
    silent_proportion: float = 1.0,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect recordings that are entirely silent.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        silent_proportion: Proportion that defines *complete* silence
            (default 1.0 means 100 %).
        df: Optional DataFrame with ``proportion_silent_metric``.

    Returns:
        True when silent proportion ≥ ``silent_proportion``.
    """
    return get_metric(audio_or_path, proportion_silent_metric, df) >= silent_proportion


def mostly_silent_check(
    audio_or_path: Union[Audio, str],
    silent_proportion: float = 0.95,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect recordings that are mostly silent.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        silent_proportion: Threshold above which audio is considered *mostly*
            silent.
        df: Optional DataFrame with ``proportion_silent_metric``.

    Returns:
        True when silent proportion > ``silent_proportion``.
    """
    return get_metric(audio_or_path, proportion_silent_metric, df) > silent_proportion


def high_amplitude_skew_magnitude_check(
    audio_or_path: Union[Audio, str],
    magnitude: float = 5.0,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Pass audio whose amplitude skew magnitude is ≤ ``magnitude``.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        magnitude: Maximum acceptable |skew|.
        df: Optional DataFrame with ``amplitude_skew_metric``.

    Returns:
        True when |skew| ≤ ``magnitude``.
    """
    return abs(get_metric(audio_or_path, amplitude_skew_metric, df)) > magnitude


def high_crest_factor_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 20.0,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect spiky signals with crest factor ≥ ``threshold``.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Minimum crest factor regarded as too high.
        df: Optional DataFrame with ``crest_factor_metric``.

    Returns:
        True when crest factor ≥ ``threshold``.
    """
    return get_metric(audio_or_path, crest_factor_metric, df) >= threshold


def low_crest_factor_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 1.5,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect over-compressed signals with crest factor ≤ ``threshold``.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Maximum crest factor regarded as too low.
        df: Optional DataFrame with ``crest_factor_metric``.

    Returns:
        True when crest factor ≤ ``threshold``.
    """
    return get_metric(audio_or_path, crest_factor_metric, df) <= threshold


def very_low_dynamic_range_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 0.1,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect signals with insufficient dynamic range.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Maximum dynamic range considered too low.
        df: Optional DataFrame with ``dynamic_range_metric``.

    Returns:
        True when dynamic range < ``threshold``.
    """
    return get_metric(audio_or_path, dynamic_range_metric, df) < threshold


def very_high_dynamic_range_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 1.9,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect signals with suspiciously high dynamic range.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Minimum dynamic range considered too high.
        df: Optional DataFrame with ``dynamic_range_metric``.

    Returns:
        True when dynamic range > ``threshold``.
    """
    return get_metric(audio_or_path, dynamic_range_metric, df) > threshold


def very_low_mean_absolute_deviation_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 0.001,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect nearly flat signals via very low MAD.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Maximum MAD regarded as too low.
        df: Optional DataFrame with ``mean_absolute_deviation_metric``.

    Returns:
        True when MAD < ``threshold``.
    """
    return get_metric(audio_or_path, mean_absolute_deviation_metric, df) < threshold


def very_high_mean_absolute_deviation_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 0.5,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect overly volatile signals via very high MAD.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Minimum MAD regarded as too high.
        df: Optional DataFrame with ``mean_absolute_deviation_metric``.

    Returns:
        True when MAD > ``threshold``.
    """
    return get_metric(audio_or_path, mean_absolute_deviation_metric, df) > threshold


def very_low_peak_snr_from_spectral_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 10.0,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect audio with very low peak-SNR (< ``threshold``).

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Minimum acceptable peak-SNR (dB).
        df: Optional DataFrame with ``peak_snr_from_spectral_metric``.

    Returns:
        True when peak-SNR < ``threshold``.
    """
    return get_metric(audio_or_path, peak_snr_from_spectral_metric, df) < threshold


def low_peak_snr_from_spectral_check(
    audio_or_path: Union[Audio, str],
    lower: float = 10.0,
    upper: float = 20.0,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Review audio whose peak-SNR falls in a *low* band.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        lower: Inclusive lower bound for the low SNR band.
        upper: Exclusive upper bound.
        df: Optional DataFrame with ``peak_snr_from_spectral_metric``.

    Returns:
        True when ``lower ≤ SNR < upper``.
    """
    snr = get_metric(audio_or_path, peak_snr_from_spectral_metric, df)
    return lower <= snr < upper


def very_high_peak_snr_from_spectral_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 60.0,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Flag unrealistically high peak-SNR (> ``threshold``).

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Peak-SNR regarded as suspiciously high (dB).
        df: Optional DataFrame with ``peak_snr_from_spectral_metric``.

    Returns:
        True when peak-SNR > ``threshold``.
    """
    return get_metric(audio_or_path, peak_snr_from_spectral_metric, df) > threshold


def low_phase_correlation_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 0.99,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect stereo signals with weak channel correlation.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Minimum acceptable correlation coefficient.
        df: Optional DataFrame with ``phase_correlation_metric``.

    Returns:
        True when correlation < ``threshold``.
    """
    return get_metric(audio_or_path, phase_correlation_metric, df) < threshold


def high_proportion_silence_at_beginning_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 0.2,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Flag recordings with > ``threshold`` leading silence.

    Note: This metric cannot use ``get_metric`` because the metric
    requires no DataFrame but **does** use its default internal threshold.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Maximum acceptable leading silence proportion.
        df: Optional DataFrame with stored metric.

    Returns:
        True when leading silence proportion > ``threshold``.
    """
    return get_metric(audio_or_path, proportion_silence_at_beginning_metric, df) > threshold


def high_proportion_silence_at_end_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 0.2,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Flag recordings with > ``threshold`` trailing silence.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Maximum acceptable trailing silence proportion.
        df: Optional DataFrame with stored metric.

    Returns:
        True when trailing silence proportion > ``threshold``.
    """
    return get_metric(audio_or_path, proportion_silence_at_end_metric, df) > threshold


def very_low_root_mean_square_energy_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 0.005,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect audio that is too quiet (very low RMS).

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Maximum RMS energy regarded as very low.
        df: Optional DataFrame with ``root_mean_square_energy_metric``.

    Returns:
        True when RMS < ``threshold``.
    """
    return get_metric(audio_or_path, root_mean_square_energy_metric, df) < threshold


def very_high_root_mean_square_energy_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 0.5,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect audio that is too loud (very high RMS).

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Minimum RMS energy regarded as very high.
        df: Optional DataFrame with ``root_mean_square_energy_metric``.

    Returns:
        True when RMS > ``threshold``.
    """
    return get_metric(audio_or_path, root_mean_square_energy_metric, df) > threshold


def low_shannon_entropy_amplitude_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 2.0,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect overly predictable audio via low entropy.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Maximum entropy regarded as too low (bits).
        df: Optional DataFrame with ``shannon_entropy_amplitude_metric``.

    Returns:
        True when entropy < ``threshold``.
    """
    return get_metric(audio_or_path, shannon_entropy_amplitude_metric, df) < threshold


def high_shannon_entropy_amplitude_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 7.5,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect overly noisy audio via high entropy.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Minimum entropy regarded as too high (bits).
        df: Optional DataFrame with ``shannon_entropy_amplitude_metric``.

    Returns:
        True when entropy > ``threshold``.
    """
    return get_metric(audio_or_path, shannon_entropy_amplitude_metric, df) > threshold


def low_signal_variance_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 1e-4,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect signals with extremely small amplitude variance.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Maximum variance regarded as too low.
        df: Optional DataFrame with ``signal_variance_metric``.

    Returns:
        True when variance < ``threshold``.
    """
    return get_metric(audio_or_path, signal_variance_metric, df) < threshold


def high_signal_variance_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 0.3,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect signals with excessively large amplitude variance.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Minimum variance regarded as too high.
        df: Optional DataFrame with ``signal_variance_metric``.

    Returns:
        True when variance > ``threshold``.
    """
    return get_metric(audio_or_path, signal_variance_metric, df) > threshold


def low_spectral_gating_snr_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 10.0,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect recordings whose segmental SNR is < ``threshold`` dB.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Minimum acceptable segmental SNR (dB).
        df: Optional DataFrame with ``spectral_gating_snr_metric``.

    Returns:
        True when SNR < ``threshold``.
    """
    return get_metric(audio_or_path, spectral_gating_snr_metric, df) < threshold


def high_spectral_gating_snr_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 60.0,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Flag recordings whose segmental SNR is > ``threshold`` dB.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Segmental SNR considered suspiciously high (dB).
        df: Optional DataFrame with ``spectral_gating_snr_metric``.

    Returns:
        True when SNR > ``threshold``.
    """
    return get_metric(audio_or_path, spectral_gating_snr_metric, df) > threshold


def low_zero_crossing_rate_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 0.01,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect signals with ZCR below ``threshold`` (likely silent/DC).

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Minimum acceptable zero-crossing rate.
        df: Optional DataFrame with ``zero_crossing_rate_metric``.

    Returns:
        True when ZCR < ``threshold``.
    """
    return get_metric(audio_or_path, zero_crossing_rate_metric, df) < threshold


def high_zero_crossing_rate_check(
    audio_or_path: Union[Audio, str],
    lower: float = 0.15,
    upper: float = 0.3,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Review signals with ZCR in a high (but not extreme) range.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        lower: Inclusive lower bound for the *high* ZCR range.
        upper: Exclusive upper bound for the *high* ZCR range.
        df: Optional DataFrame with ``zero_crossing_rate_metric``.

    Returns:
        True when ``lower ≤ ZCR < upper``.
    """
    zcr = get_metric(audio_or_path, zero_crossing_rate_metric, df)
    return lower <= zcr < upper


def very_high_zero_crossing_rate_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 0.3,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Detect signals with extremely high ZCR (noise / corruption).

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Zero-crossing rate regarded as *very high*.
        df: Optional DataFrame with ``zero_crossing_rate_metric``.

    Returns:
        True when ZCR > ``threshold``.
    """
    return get_metric(audio_or_path, zero_crossing_rate_metric, df) > threshold


def audio_intensity_positive_check(
    audio_or_path: Union[Audio, str],
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """Check that the audio has non-zero intensity.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        df: Optional DataFrame with ``dynamic_range_metric``.

    Returns:
        True if the audio has non-zero dynamic range.
    """
    return get_metric(audio_or_path, dynamic_range_metric, df) > 0
