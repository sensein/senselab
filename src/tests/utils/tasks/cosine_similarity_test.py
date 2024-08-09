"""This module contains unit tests for the cosine similarity function."""

import pytest
import torch

from senselab.utils.tasks.cosine_similarity import compute_cosine_similarity


def test_cosine_similarity_identical_vectors() -> None:
    """Test cosine similarity for identical vectors."""
    tensor1 = torch.tensor([1.0, 2.0, 3.0])
    tensor2 = torch.tensor([1.0, 2.0, 3.0])
    similarity = compute_cosine_similarity(tensor1, tensor2)
    assert torch.isclose(torch.tensor(similarity), torch.tensor(1.0), atol=1e-6)


def test_cosine_similarity_opposite_vectors() -> None:
    """Test cosine similarity for opposite vectors."""
    tensor1 = torch.tensor([1.0, 0.0, -1.0])
    tensor2 = torch.tensor([-1.0, 0.0, 1.0])
    similarity = compute_cosine_similarity(tensor1, tensor2)
    assert torch.isclose(torch.tensor(similarity), torch.tensor(-1.0), atol=1e-6)


def test_cosine_similarity_orthogonal_vectors() -> None:
    """Test cosine similarity for orthogonal vectors."""
    tensor1 = torch.tensor([1.0, 0.0])
    tensor2 = torch.tensor([0.0, 1.0])
    similarity = compute_cosine_similarity(tensor1, tensor2)
    assert torch.isclose(torch.tensor(similarity), torch.tensor(0.0), atol=1e-6)


def test_cosine_similarity_non_identical_vectors() -> None:
    """Test cosine similarity for non-identical but non-orthogonal vectors."""
    tensor1 = torch.tensor([1.0, 2.0, 3.0])
    tensor2 = torch.tensor([4.0, 5.0, 6.0])
    expected_value = 0.9746318461970762
    similarity = compute_cosine_similarity(tensor1, tensor2)
    assert torch.isclose(torch.tensor(similarity), torch.tensor(expected_value), atol=1e-6)


def test_cosine_similarity_different_shapes() -> None:
    """Test cosine similarity for tensors of different shapes, expecting a ValueError."""
    tensor1 = torch.tensor([1.0, 2.0])
    tensor2 = torch.tensor([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        compute_cosine_similarity(tensor1, tensor2)
