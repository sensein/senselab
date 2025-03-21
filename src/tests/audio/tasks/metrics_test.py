"""Tests audio quality metric functions."""

import inspect
from typing import Pattern, Type, Union

import pytest
import torch
from pytest import approx

import senselab.audio.tasks.bioacoustic_qc.metrics as metrics
from senselab.audio.data_structures import Audio
from senselab.audio.tasks.bioacoustic_qc.metrics import (
    amplitude_headroom_metric,
    amplitude_modulation_depth_metric,
    clipping_present_metric,
    proportion_clipped_metric,
    proportion_silence_at_beginning_metric,
    proportion_silence_at_end_metric,
    proportion_silent_metric,
    root_mean_square_energy_metric,
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
        (torch.tensor([[0.0, 0.4, 0.6, -0.9]]), 0.0),  # No clipped samples
        (torch.tensor([[1.0, 1.0, -1.0, -1.0]]), 1.0),  # All clipped
    ],
)
def test_proportion_clipped_metric(waveform: torch.Tensor, expected_proportion: float) -> None:
    """Tests proportion_clipped_metric function."""
    audio = Audio(waveform=waveform, sampling_rate=16000)
    proportion = proportion_clipped_metric(audio, clip_threshold=1.0)
    assert proportion == approx(expected_proportion, rel=1e-6), f"Expected {expected_proportion}, got {proportion}"


@pytest.mark.parametrize(
    "waveform, expected_clipping",
    [
        (torch.tensor([[0.0, 0.5, 0.9]]), False),  # No clipping
        (torch.tensor([[0.0, 1.0, -0.5]]), True),  # One sample clipped
        (torch.tensor([[1.01, -1.0]]), True),  # Sample above threshold
        (torch.tensor([[1.0, -1.0, 1.0]]), True),  # All clipped
    ],
)
def test_clipping_present_metric(waveform: torch.Tensor, expected_clipping: bool) -> None:
    """Tests clipping_present_metric function."""
    audio = Audio(waveform=waveform, sampling_rate=16000)
    result = clipping_present_metric(audio, clip_threshold=1.0)
    assert result == expected_clipping, f"Expected {expected_clipping}, got {result}"


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
