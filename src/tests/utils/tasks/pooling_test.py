"""Module for testing the pooling function."""

import pytest
import torch

from senselab.utils.tasks.pooling import pooling


@pytest.fixture
def sample_data() -> torch.Tensor:
    """Sample data for testing."""
    return torch.randn(10, 20, 30)  # 10 samples, 20 features, 30 channels


def test_mean_pooling(sample_data: torch.Tensor) -> None:
    """Test mean pooling."""
    result = pooling(sample_data, pool_type="mean", dimension=0, keep_dimension=True)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (1, 20, 30)


def test_max_pooling(sample_data: torch.Tensor) -> None:
    """Test max pooling."""
    result = pooling(sample_data, pool_type="max", dimension=1, keep_dimension=False)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (10, 30)


def test_min_pooling(sample_data: torch.Tensor) -> None:
    """Test min pooling."""
    result = pooling(sample_data, pool_type="min", dimension=2)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (10, 20)


def test_invalid_pool_type(sample_data: torch.Tensor) -> None:
    """Test invalid pool type."""
    with pytest.raises(ValueError):
        pooling(sample_data, pool_type="invalid_pool_type")  # type: ignore


def test_invalid_dimension(sample_data: torch.Tensor) -> None:
    """Test invalid dimension."""
    with pytest.raises(AssertionError):
        pooling(sample_data, dimension=3)


def test_empty_input() -> None:
    """Test empty input data."""
    with pytest.raises(ValueError):
        pooling(torch.empty(0, 10))
