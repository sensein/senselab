"""Module for testing the compute_normalized_cross_correlation function."""

import pytest
import torch

from senselab.utils.tasks.cross_correlation import compute_normalized_cross_correlation


def test_normalized_cross_correlation_basic() -> None:
    """Test normalized cross-correlation for basic identical signals."""
    signal1 = torch.tensor([1.0, 1.0])
    signal2 = torch.tensor([1.0, 1.0])
    expected_result = torch.tensor([0.5, 1.0, 0.5], dtype=torch.float32)
    result = compute_normalized_cross_correlation(signal1, signal2)
    assert torch.allclose(result, expected_result, atol=1e-4), f"Expected {expected_result}, but got {result}"


def test_normalized_cross_correlation_different_lengths() -> None:
    """Test normalized cross-correlation for signals of different lengths."""
    signal1 = torch.tensor([1.0, 2.0, 1.0])
    signal2 = torch.tensor([1.0, 2.0])
    expected_result = torch.tensor([0.3651, 0.9129, 0.7303, 0.1826], dtype=torch.float32)
    result = compute_normalized_cross_correlation(signal1, signal2)
    assert torch.allclose(result, expected_result, atol=1e-4), f"Expected {expected_result}, but got {result}"


def test_normalized_cross_correlation_zero_signal() -> None:
    """Test normalized cross-correlation with a zero signal."""
    signal1 = torch.tensor([0.0, 0.0, 0.0, 0.0])
    signal2 = torch.tensor([1.0, 2.0, 3.0])
    with pytest.raises(ZeroDivisionError):
        compute_normalized_cross_correlation(signal1, signal2)


def test_normalized_cross_correlation_empty_signal() -> None:
    """Test normalized cross-correlation with an empty signal."""
    signal1 = torch.tensor([])
    signal2 = torch.tensor([1.0, 2.0, 3.0])
    with pytest.raises(ZeroDivisionError):
        compute_normalized_cross_correlation(signal1, signal2)


def test_normalized_cross_correlation_non_1d_signal() -> None:
    """Test normalized cross-correlation with non-1D signals."""
    signal1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    signal2 = torch.tensor([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        compute_normalized_cross_correlation(signal1, signal2)

    signal1 = torch.tensor([1.0, 2.0, 3.0])
    signal2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError):
        compute_normalized_cross_correlation(signal1, signal2)
