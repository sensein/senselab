"""Tests audio quality metric functions."""

import pytest
import torch
from pytest import approx

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.bioacoustic_qc.metrics import (
    amplitude_headroom_metric,
    amplitude_toeroom_metric,
    proportion_silence_at_beginning_metric,
    proportion_silence_at_end_metric,
    proportion_silent_metric,
)


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
    """Tests amplitude_headroom_metric function."""
    audio = Audio(waveform=waveform, sampling_rate=16000)
    headroom = amplitude_headroom_metric(audio)
    assert headroom == approx(expected_headroom, rel=1e-6), f"Expected {expected_headroom}, got {headroom}"


@pytest.mark.parametrize(
    "waveform, expected_toeroom",
    [
        (torch.tensor([[0.5, -0.5, -0.8, 0.8]]), -0.2),  # Min amplitude -0.8 → Toeroom = -1.0 - (-0.8) = 0.2
        (torch.tensor([[0.1, -0.1, -0.3, 0.3]]), -0.7),  # Min amplitude -0.3 → Toeroom = 0.7
        (torch.tensor([[1.0, -1.0, 0.5, -0.5]]), 0.0),  # Min amplitude -1.0 → Toeroom = 0.0
    ],
)
def test_amplitude_toeroom_metric(waveform: torch.Tensor, expected_toeroom: float) -> None:
    """Tests amplitude_toeroom_metric function."""
    audio = Audio(waveform=waveform, sampling_rate=16000)
    toeroom = amplitude_toeroom_metric(audio)
    assert toeroom == approx(expected_toeroom, rel=1e-6), f"Expected {expected_toeroom}, got {toeroom}"


def test_amplitude_toeroom_clipping_error() -> None:
    """Tests that amplitude_toeroom_metric raises an error when samples are below -1.0."""
    waveform = torch.tensor([[0.5, -1.1, -0.9, 0.8]])  # Min amplitude -1.1
    audio = Audio(waveform=waveform, sampling_rate=16000)

    with pytest.raises(ValueError, match="Audio contains samples under -1.0"):
        amplitude_toeroom_metric(audio)
