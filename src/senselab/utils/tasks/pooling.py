"""This module implements pooling methods for torch tensors."""

from typing import Literal

import torch


def pooling(
    data: torch.Tensor,
    pool_type: Literal["mean", "min", "max"] = "mean",
    dimension: int = 0,
    keep_dimension: bool = False,
) -> torch.Tensor:
    """Apply pooling to the input tensor along the specified dimension.

    This function provides three pooling methods: mean, min, or max. The pooled tensor is returned as a PyTorch tensor.

    Args:
        data (torch.Tensor): Input tensor of shape (n_samples, n_features, n_channels).
        pool_type (str, optional): The pooling method to use.
            Choices are:
                * "mean" for mean pooling
                * "min" for min pooling
                * "max" for max pooling
            Defaults to "mean".
        dimension (int, optional): The dimension along which to apply the pooling.
            Defaults to 0.
        keep_dimension (bool, optional): Whether to keep the original tensor dimension after pooling.

    Returns:
        torch.Tensor: The pooled tensor of shape (n_samples, n_features, n_channels).

    Raises:
        ValueError: If an invalid pooling method choice is provided.

    Examples:
        >>> data = torch.randn(10, 20, 30)  # 10 samples, 20 features, 30 channels
        >>> data.shape
        torch.Size([10, 20, 30])

        >>> pooled_data = pooling(data, pool_type="mean", dimension=0, keep_dimension=True)
        >>> print(pooled_data.shape)
        torch.Size([1, 20, 30])

        >>> pooled_data = pooling(data, pool_type="max", dimension=1, keep_dimension=False)
        >>> print(pooled_data.shape)
        torch.Size([10, 30])

        >>> pooled_data = pooling(data, pool_type="min", dimension=2)
        >>> print(pooled_data.shape)
        torch.Size([10, 20])
    """
    # Ensure the dimension is valid
    assert dimension < len(data.shape), "Invalid dimension provided."

    if data.numel() == 0:
        raise ValueError("Input data must not be empty")

    # Applying the appropriate pooling based on the pool type
    if pool_type == "max":
        return torch.max(data, dim=dimension, keepdim=keep_dimension).values
    elif pool_type == "mean":
        return torch.mean(data, dim=dimension, keepdim=keep_dimension)
    elif pool_type == "min":
        return torch.min(data, dim=dimension, keepdim=keep_dimension).values
    else:
        raise ValueError("Unsupported pooling type. Choose 'max', 'avg', or 'min'.")
