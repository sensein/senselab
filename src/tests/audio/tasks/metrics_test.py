"""Tests audio quality metric functions."""

import inspect
import math
from typing import Pattern, Type, Union

import numpy as np
import pytest
import torch
from pytest import approx

import senselab.audio.tasks.bioacoustic_qc.metrics as metrics
from senselab.audio.data_structures import Audio
from senselab.audio.tasks.bioacoustic_qc.metrics import (
    amplitude_headroom_metric,
    amplitude_interquartile_range_metric,
    amplitude_kurtosis_metric,
    amplitude_modulation_depth_metric,
    amplitude_skew_metric,
    crest_factor_metric,
    dynamic_range_metric,
    mean_absolute_amplitude_metric,
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
    zero_crossing_rate_metric,
)


def test_metric_function_names_end_with_metric() -> None:
    """Ensure all public functions in metrics module end with 'metric'."""
    funcs = inspect.getmembers(metrics, inspect.isfunction)
    for name, _ in funcs:
        assert name.endswith("metric"), f"Function '{name}' does not end with 'metric'"


@pytest.mark.parametrize(
    "waveform, expected_silence_proportion",
    [
        (torch.tensor([[0.0, 0.0, 0.0, 0.0]]), 1.0),  # Completely silent audio
        (torch.tensor([[0.0, 0.1, 0.2, 0.3]]), 0.25),  # 25% silent
        (torch.tensor([[0.1, 0.2, 0.3, 0.4]]), 0.0),  # No silence
    ],
)
def test_proportion_silent_metric(waveform: torch.Tensor, expected_silence_proportion: float) -> None:
    """Tests proportion_silent_metric function."""
    audio = Audio(waveform=waveform, sampling_rate=16000)
    silence_proportion = proportion_silent_metric(audio, silence_threshold=0.05)
    assert (
        silence_proportion == expected_silence_proportion
    ), f"Expected {expected_silence_proportion}, got {silence_proportion}"


@pytest.mark.parametrize(
    "waveform, expected_silence_start_proportion",
    [
        (torch.tensor([[0.0, 0.0, 0.1, 0.2]]), 0.5),  # 50% silence at the beginning
        (torch.tensor([[0.0, 0.0, 0.0, 0.0]]), 1.0),  # Entirely silent audio
        (torch.tensor([[0.1, 0.2, 0.3, 0.4]]), 0.0),  # No leading silence
    ],
)
def test_proportion_silence_at_beginning(waveform: torch.Tensor, expected_silence_start_proportion: float) -> None:
    """Tests proportion_silence_at_beginning function."""
    audio = Audio(waveform=waveform, sampling_rate=16000)
    silence_start_proportion = proportion_silence_at_beginning_metric(audio, silence_threshold=0.05)
    assert (
        silence_start_proportion == expected_silence_start_proportion
    ), f"Expected {expected_silence_start_proportion}, got {silence_start_proportion}"


@pytest.mark.parametrize(
    "waveform, expected_silence_end_proportion",
    [
        (torch.tensor([[0.1, 0.2, 0.0, 0.0]]), 0.5),  # 50% silence at the end
        (torch.tensor([[0.0, 0.0, 0.0, 0.0]]), 1.0),  # Entirely silent audio
        (torch.tensor([[0.1, 0.2, 0.3, 0.4]]), 0.0),  # No trailing silence
    ],
)
def test_proportion_silence_at_end(waveform: torch.Tensor, expected_silence_end_proportion: float) -> None:
    """Tests proportion_silence_at_end_metric function."""
    audio = Audio(waveform=waveform, sampling_rate=16000)
    silence_end_proportion = proportion_silence_at_end_metric(audio, silence_threshold=0.05)
    assert (
        silence_end_proportion == expected_silence_end_proportion
    ), f"Expected {expected_silence_end_proportion}, got {silence_end_proportion}"


@pytest.mark.parametrize(
    "waveform, expected_headroom",
    [
        (torch.tensor([[0.5, -0.5, 0.8, -0.8]]), 0.2),
        (torch.tensor([[0.1, -0.1, 0.3, -0.3]]), 0.7),
        (torch.tensor([[1.0, -1.0, 0.5, -0.5]]), 0.0),
    ],
)
def test_amplitude_headroom_metric(waveform: torch.Tensor, expected_headroom: float) -> None:
    """Tests amplitude_headroom_metric with valid inputs."""
    audio = Audio(waveform=waveform, sampling_rate=16000)
    headroom = amplitude_headroom_metric(audio)
    assert headroom == approx(expected_headroom, rel=1e-6), f"Expected {expected_headroom}, got {headroom}"


@pytest.mark.parametrize(
    "waveform, expected_error, match",
    [
        (torch.tensor([[1.1, -0.5]]), ValueError, "over 1.0"),
        (torch.tensor([[0.2, -1.2]]), ValueError, "under -1.0"),
    ],
)
def test_amplitude_headroom_metric_errors(
    waveform: torch.Tensor, expected_error: Type[BaseException], match: Union[str, Pattern[str]]
) -> None:
    """Tests amplitude_headroom_metric with invalid inputs."""
    audio = Audio(waveform=waveform, sampling_rate=16000)
    with pytest.raises(expected_error, match=match):
        amplitude_headroom_metric(audio)


@pytest.mark.parametrize(
    "waveform, expected_proportion",
    [
        (torch.tensor([[0.0, 0.5, 1.0, -1.0]]), 0.5),  # 2/4 clipped
        (torch.tensor([[0.0, 0.4, 0.6, -0.9]]), 0.25),  # a large enough proportion == max value
        (torch.tensor([[1.0, 1.0, -1.0, -1.0]]), 1.0),  # All clipped
    ],
)
def test_proportion_clipped_metric(waveform: torch.Tensor, expected_proportion: float) -> None:
    """Tests proportion_clipped_metric function."""
    audio = Audio(waveform=waveform, sampling_rate=16000)
    proportion = proportion_clipped_metric(audio, clip_threshold=1.0)
    assert proportion == approx(expected_proportion, rel=1e-6), f"Expected {expected_proportion}, got {proportion}"


@pytest.mark.parametrize(
    "waveform, expected_depth",
    [
        (torch.tensor([[0.5, 0.5, 0.5, 0.5]]), 0.0),
        (torch.tensor([[0.1, 0.9, 0.1, 0.9]]), 0.8),
        (torch.tensor([[0.0, 0.0, 0.0, 0.0]]), 0.0),
        (torch.tensor([[-0.1, 0.9, -0.1, 0.9]]), 0.8),
        (torch.tensor([[-0.5, -0.5, -0.5, -0.5]]), 0.0),
    ],
)
def test_amplitude_modulation_depth_metric(waveform: torch.Tensor, expected_depth: float) -> None:
    """Tests amplitude_modulation_depth_metric function including negative values."""
    audio = Audio(waveform=waveform, sampling_rate=16000)
    depth = amplitude_modulation_depth_metric(audio)
    assert depth == approx(expected_depth, rel=1e-6), f"Expected {expected_depth}, got {depth}"


@pytest.mark.parametrize(
    "waveform, expected",
    [
        (torch.tensor([[1.0, 1.0, 1.0, 1.0]]), 1.0),
        (torch.tensor([[0.0, 0.0, 0.0, 0.0]]), 0.0),
        (torch.tensor([[1.0, -1.0, 1.0, -1.0]]), 1.0),
        (torch.tensor([[1.0, 0.0, 1.0, 0.0]]), (0.5) ** 0.5),
        (torch.tensor([[1.0, 1.0], [0.0, 0.0]]), 0.5),
    ],
)
def test_root_mean_square_energy_metric(waveform: torch.Tensor, expected: float) -> None:
    """Test RMS energy metric against expected values."""
    audio = Audio(waveform=waveform, sampling_rate=16000)
    result = root_mean_square_energy_metric(audio)
    assert result == approx(expected, rel=1e-6), f"Expected {expected}, got {result}"


@pytest.mark.parametrize(
    "waveform, expected_zcr",
    [
        (torch.tensor([[1.0, 1.0, 1.0, 1.0]]), 0.0),
        (torch.tensor([[1.0, 1.0, -1.0, -1.0]]), 1 / 3),
        (torch.tensor([[1.0, -1.0, 1.0, -1.0]]), 1.0),
        (torch.tensor([[1.0, 0.0, -1.0, 0.0]]), 0),  # intermediate 0 doesn't count
        # Multi-channel: one has 1 crossing, other 0 → avg = 0.5
        (torch.tensor([[1.0, -1.0, -1.0], [0.5, 0.5, 0.5]]), 0.25),
    ],
)
def test_zero_crossing_rate_metric(waveform: torch.Tensor, expected_zcr: float) -> None:
    """Tests zero_crossing_rate_metric function."""
    audio = Audio(waveform=waveform, sampling_rate=16000)
    result = zero_crossing_rate_metric(audio)
    assert result == approx(expected_zcr, rel=1e-6), f"Expected {expected_zcr}, got {result}"


@pytest.mark.parametrize(
    "waveform, expected",
    [
        (torch.tensor([[0.0, 0.0, 0.0, 0.0]]), 0.0),  # Constant signal → zero variance
        (torch.tensor([[1.0, -1.0, 1.0, -1.0]]), 4 / 3),  # Mean=0, variance=1
        (torch.tensor([[0.5, 0.5, -0.5, -0.5]]), 1 / 3),  # Variance of smaller range
        (torch.tensor([[1.0, 2.0, 3.0, 4.0]]), 5 / 3),  # Increasing sequence
    ],
)
def test_signal_variance_metric(waveform: torch.Tensor, expected: float) -> None:
    """Tests signal_variance_metric function."""
    audio = Audio(waveform=waveform, sampling_rate=16000)
    result = signal_variance_metric(audio)
    assert result == approx(expected, rel=1e-6), f"Expected {expected}, got {result}"


@pytest.mark.parametrize(
    "waveform, expected_range",
    [
        # Constant signal: dynamic range = 1.0 - 1.0 = 0.0
        (torch.tensor([[1.0, 1.0, 1.0, 1.0]]), 0.0),
        # Increasing signal: dynamic range = 4.0 - 1.0 = 3.0
        (torch.tensor([[1.0, 2.0, 3.0, 4.0]]), 3.0),
        # Mixed signal: e.g., min = -0.5, max = 1.0 → dynamic range = 1.0 - (-0.5) = 1.5
        (torch.tensor([[0.0, 0.5, 1.0, -0.5]]), 1.5),
        # Multi-channel signal: overall min = -1.0, overall max = 0.8 → dynamic range = 0.8 - (-1.0) = 1.8
        (torch.tensor([[-0.5, 0.3, 0.8], [-1.0, 0.0, 0.5]]), 1.8),
    ],
)
def test_dynamic_range_metric(waveform: torch.Tensor, expected_range: float) -> None:
    """Tests dynamic_range_metric function."""
    audio = Audio(waveform=waveform, sampling_rate=16000)
    result = dynamic_range_metric(audio)
    assert result == approx(expected_range, rel=1e-6), f"Expected {expected_range}, got {result}"


@pytest.mark.parametrize(
    "waveform, expected_mean_abs",
    [
        # Constant positive signal: mean abs = 1.0
        (torch.tensor([[1.0, 1.0, 1.0, 1.0]]), 1.0),
        # Alternating positive and negative: mean abs = 1.0
        (torch.tensor([[1.0, -1.0, 1.0, -1.0]]), 1.0),
        # Zero signal: mean abs = 0.0
        (torch.tensor([[0.0, 0.0, 0.0, 0.0]]), 0.0),
        # Multi-channel signal:
        # Channel 1: [1.0, 0.0, -1.0, 0.0] -> mean abs = (1+0+1+0)/4 = 0.5
        # Channel 2: [2.0, 2.0, 2.0, 2.0] -> mean abs = 2.0
        # Overall average = (0.5 + 2.0)/2 = 1.25
        (torch.tensor([[1.0, 0.0, -1.0, 0.0], [2.0, 2.0, 2.0, 2.0]]), 1.25),
    ],
)
def test_mean_absolute_amplitude_metric(waveform: torch.Tensor, expected_mean_abs: float) -> None:
    """Tests the mean_absolute_amplitude_metric function."""
    audio = Audio(waveform=waveform, sampling_rate=16000)
    result = mean_absolute_amplitude_metric(audio)
    assert result == approx(expected_mean_abs, rel=1e-6), f"Expected {expected_mean_abs}, got {result}"


@pytest.mark.parametrize(
    "waveform, expected_mad",
    [
        # Constant signal: MAD should be 0
        (torch.tensor([[1.0, 1.0, 1.0, 1.0]]), 0.0),
        # Single channel with two distinct values:
        # [1, -1] -> mean = 0, deviations = [1, 1] -> MAD = 1.0
        (torch.tensor([[1.0, -1.0]]), 1.0),
        # Single channel: [1, 0, -1, 0] -> mean = 0, deviations = [1, 0, 1, 0] -> MAD = 0.5
        (torch.tensor([[1.0, 0.0, -1.0, 0.0]]), 0.5),
        # Multi-channel: two channels with symmetric values.
        # Channel 1: [1, 2, 3, 4] -> mean = 2.5, deviations = [1.5, 0.5, 0.5, 1.5] -> MAD = 1.0
        # Channel 2: [-1, -2, -3, -4] -> mean = -2.5, deviations = [1.5, 0.5, 0.5, 1.5] -> MAD = 1.0
        # Overall MAD = (1.0 + 1.0) / 2 = 1.0
        (torch.tensor([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0]]), 1.0),
    ],
)
def test_mean_absolute_deviation_metric(waveform: torch.Tensor, expected_mad: float) -> None:
    """Tests the mean_absolute_deviation_metric function."""
    audio = Audio(waveform=waveform, sampling_rate=16000)
    result = mean_absolute_deviation_metric(audio)
    assert result == approx(expected_mad, rel=1e-6), f"Expected {expected_mad}, got {result}"


@pytest.mark.parametrize(
    "audio_fixture",
    ["mono_audio_sample", "stereo_audio_sample"],
)
def test_shannon_entropy_amplitude_metric(audio_fixture: str, request: pytest.FixtureRequest) -> None:
    """Tests Shannon entropy metric on mono and stereo audio samples."""
    audio_sample = request.getfixturevalue(audio_fixture)

    entropy = shannon_entropy_amplitude_metric(audio_sample)

    # Shannon entropy should be >= 0 and not NaN
    assert isinstance(entropy, float), "Entropy output is not a float."
    assert not np.isnan(entropy), "Entropy is NaN."
    assert entropy >= 0.0, "Entropy should be non-negative."


@pytest.mark.parametrize(
    "audio_fixture",
    ["mono_audio_sample", "stereo_audio_sample"],
)
def test_crest_factor_metric(audio_fixture: str, request: pytest.FixtureRequest) -> None:
    """Tests crest_factor_metric returns a finite value ≥1 for real audio fixtures."""
    audio = request.getfixturevalue(audio_fixture)
    cf = crest_factor_metric(audio)

    assert isinstance(cf, float), "Crest factor must be a float"
    assert not math.isnan(cf), "Crest factor should not be NaN"
    assert cf >= 1.0, f"Expected crest factor ≥1, got {cf}"
    assert not math.isinf(cf), "Crest factor should be finite for non‑silent audio"


@pytest.mark.parametrize(
    "audio_fixture",
    ["mono_audio_sample", "stereo_audio_sample"],
)
def test_peak_snr_from_spectral_metric(audio_fixture: str, request: pytest.FixtureRequest) -> None:
    """Tests peak_snr_from_spectral_metric returns a valid float ≥ 0 for real audio fixtures."""
    audio: Audio = request.getfixturevalue(audio_fixture)
    snr = peak_snr_from_spectral_metric(audio)

    assert isinstance(snr, float), "Peak SNR must be a float"
    assert not math.isnan(snr), "Peak SNR should not be NaN"
    assert snr >= 0.0, f"Expected peak SNR ≥ 0, got {snr}"
    assert not math.isinf(snr), "Peak SNR should be finite for non-silent audio"


@pytest.mark.parametrize(
    "audio_fixture",
    ["mono_audio_sample", "stereo_audio_sample"],
)
def test_amplitude_skew_metric(audio_fixture: str, request: pytest.FixtureRequest) -> None:
    """Tests amplitude_skew_metric returns a finite float for real audio fixtures.

    Args:
        audio_fixture (str): Name of the audio fixture.
        request (pytest.FixtureRequest): Pytest request object for accessing fixtures.
    """
    audio: Audio = request.getfixturevalue(audio_fixture)
    skewness = amplitude_skew_metric(audio)

    assert isinstance(skewness, float), "Skewness must be a float"
    assert not math.isnan(skewness), "Skewness should not be NaN"
    assert not math.isinf(skewness), "Skewness should be finite"


@pytest.mark.parametrize(
    "audio_fixture",
    ["mono_audio_sample", "stereo_audio_sample"],
)
def test_amplitude_kurtosis_metric(audio_fixture: str, request: pytest.FixtureRequest) -> None:
    """Tests kurtosis_amplitude_metric returns a finite float for real audio fixtures.

    Args:
        audio_fixture (str): Name of the audio fixture.
        request (pytest.FixtureRequest): Pytest request object for accessing fixtures.
    """
    audio: Audio = request.getfixturevalue(audio_fixture)
    kurt = amplitude_kurtosis_metric(audio)

    assert isinstance(kurt, float), "Kurtosis must be a float"
    assert not math.isnan(kurt), "Kurtosis should not be NaN"
    assert not math.isinf(kurt), "Kurtosis should be finite"


@pytest.mark.parametrize(
    "audio_fixture",
    ["mono_audio_sample", "stereo_audio_sample"],
)
def test_amplitude_interquartile_range_metric(audio_fixture: str, request: pytest.FixtureRequest) -> None:
    """Tests amplitude_interquartile_range_metric returns a valid finite float value."""
    audio = request.getfixturevalue(audio_fixture)
    iqr_value = amplitude_interquartile_range_metric(audio)

    assert isinstance(iqr_value, float), "IQR must be a float"
    assert not torch.isnan(torch.tensor(iqr_value)), "IQR should not be NaN"
    assert not torch.isinf(torch.tensor(iqr_value)), "IQR should be finite"
    assert iqr_value >= 0, f"Expected IQR to be non-negative, got {iqr_value}"


@pytest.mark.parametrize(
    "waveform, expected_correlation",
    [
        (torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]]), 1.0),  # Identical channels -> perfect correlation
        (
            torch.tensor([[0.1, 0.2, 0.3, 0.4], [-0.1, -0.2, -0.3, -0.4]]),
            -1.0,
        ),  # Inverted channels -> negative correlation
        (torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]), -1.0),  # Reversed channels -> negative correlation
        (torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.0, 0.0, 0.0, 0.0]]), 0.0),  # One silent channel -> no correlation
        (torch.tensor([[1.0, 1.0, 1.0, 1.0], [0.5, 0.6, 0.7, 0.8]]), 0.0),  # Constant channel -> no correlation
    ],
)
def test_phase_correlation_metric(waveform: torch.Tensor, expected_correlation: float) -> None:
    """Tests phase_correlation_metric with different stereo channel relationships."""
    audio = Audio(waveform=waveform, sampling_rate=16000)
    # Using a small frame_length to match our test waveforms
    correlation = phase_correlation_metric(audio, frame_length=4, hop_length=4)
    assert correlation == pytest.approx(
        expected_correlation, rel=1e-6
    ), f"Expected {expected_correlation}, got {correlation}"


@pytest.mark.parametrize(
    "waveform, expected_error, match",
    [
        (torch.tensor([[0.1, 0.2, 0.3, 0.4]]), ValueError, "Expected stereo audio"),  # Mono audio (1 channel)
        (torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]), ValueError, "Expected stereo audio"),  # 3 channels
    ],
)
def test_phase_correlation_metric_errors(
    waveform: torch.Tensor, expected_error: Type[BaseException], match: Union[str, Pattern[str]]
) -> None:
    """Tests phase_correlation_metric with invalid inputs (non-stereo audio)."""
    audio = Audio(waveform=waveform, sampling_rate=16000)
    with pytest.raises(expected_error, match=match):
        phase_correlation_metric(audio)


@pytest.mark.parametrize(
    "audio_fixture",
    ["stereo_audio_sample"],
)
def test_phase_correlation_metric_real_audio(audio_fixture: str, request: pytest.FixtureRequest) -> None:
    """Tests phase_correlation_metric returns a valid float between -1.0 and 1.0 for real audio fixtures."""
    audio: Audio = request.getfixturevalue(audio_fixture)
    correlation = phase_correlation_metric(audio)

    assert isinstance(correlation, float), "Correlation must be a float"
    assert not math.isnan(correlation), "Correlation should not be NaN"
    assert not math.isinf(correlation), "Correlation should be finite"
    assert -1.0 <= correlation <= 1.0, f"Expected correlation between -1.0 and 1.0, got {correlation}"
