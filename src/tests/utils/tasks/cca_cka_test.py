"""Module for testing the CCA and CKA functions."""

import torch

from senselab.utils.tasks.cca_cka import CKAKernelType, compute_cca, compute_cka


def test_compute_cca() -> None:
    """Test compute_cca function with random input tensors."""
    # Create input tensors
    features_x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    features_y = torch.tensor([[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]])
    expected = 1.0  # Since features_y is a linear transformation of features_x, CCA should be perfect.

    # Call the compute_cca function
    cca_value = compute_cca(features_x, features_y)

    # Assert that the result is a float
    assert isinstance(cca_value, float), "Output should be a float."

    assert torch.isclose(torch.tensor(cca_value), torch.tensor(expected), atol=1e-6)


def test_compute_cka_linear() -> None:
    """Test compute_cka function with linear kernel and random input tensors."""
    # Create input tensors
    features_x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    features_y = torch.tensor([[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]])
    expected = 1.0  # Since features_y is a linear transformation of features_x, linear CKA should be perfect.

    # Call the compute_cka function with linear kernel
    cka_value = compute_cka(features_x, features_y, kernel=CKAKernelType.LINEAR)

    # Assert that the result is a float
    assert isinstance(cka_value, float), "Output should be a float."

    assert torch.isclose(torch.tensor(cka_value), torch.tensor(expected), atol=1e-6)


def test_compute_cka_rbf() -> None:
    """Test compute_cka function with RBF kernel and random input tensors."""
    # Create input tensors
    features_x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    features_y = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    expected = 1.0  # Since features_y is the same as features_x, RBF CKA should be perfect.

    # Call the compute_cka function with rbf kernel
    cka_value = compute_cka(features_x, features_y, kernel=CKAKernelType.RBF)

    # Assert that the result is a float
    assert isinstance(cka_value, float), "Output should be a float."

    assert torch.isclose(torch.tensor(cka_value), torch.tensor(expected), atol=1e-6)
