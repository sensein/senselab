"""Module for testing the compute_dimensionality_reduction function."""

import pytest
import torch

from senselab.utils.tasks.dimensionality_reduction import compute_dimensionality_reduction


@pytest.fixture
def sample_data() -> torch.Tensor:
    """Sample data for testing."""
    return torch.randn(100, 10)  # 100 samples, 10 features


def test_pca_reduction(sample_data: torch.Tensor) -> None:
    """Test PCA dimensionality reduction."""
    result = compute_dimensionality_reduction(sample_data, model="pca", n_components=2)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (100, 2)


def test_tsne_reduction(sample_data: torch.Tensor) -> None:
    """Test t-SNE dimensionality reduction."""
    result = compute_dimensionality_reduction(sample_data, model="tsne", n_components=3)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (100, 3)


def test_umap_reduction(sample_data: torch.Tensor) -> None:
    """Test UMAP dimensionality reduction."""
    result = compute_dimensionality_reduction(sample_data, model="umap", n_components=4)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (100, 4)


def test_invalid_model(sample_data: torch.Tensor) -> None:
    """Test invalid model choice."""
    with pytest.raises(ValueError):
        compute_dimensionality_reduction(sample_data, model="invalid_model")  # type: ignore


def test_invalid_n_components(sample_data: torch.Tensor) -> None:
    """Test invalid n_components."""
    with pytest.raises(ValueError):
        compute_dimensionality_reduction(sample_data, model="pca", n_components=-1)


def test_n_components_larger_than_features(sample_data: torch.Tensor) -> None:
    """Test n_components larger than the number of features."""
    with pytest.raises(ValueError):
        compute_dimensionality_reduction(sample_data, model="pca", n_components=15)


def test_additional_kwargs(sample_data: torch.Tensor) -> None:
    """Test additional keyword arguments."""
    result = compute_dimensionality_reduction(sample_data, model="pca", n_components=2, random_state=42)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (100, 2)


def test_empty_input() -> None:
    """Test empty input data."""
    with pytest.raises(ValueError):
        compute_dimensionality_reduction(torch.empty(0, 10), model="pca")
