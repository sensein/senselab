"""This module provides the implementation of cosine similarity."""

import torch


def compute_cosine_similarity(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """Compute the cosine similarity between two torch tensors.

    Args:
        tensor1 (Tensor): The first input tensor.
        tensor2 (Tensor): The second input tensor.

    Returns:
        float: The cosine similarity between the two input tensors.

    Raises:
        ValueError: If the input tensors are not of the same shape.

    Examples:
        >>> tensor1 = torch.tensor([1.0, 2.0, 3.0])
        >>> tensor2 = torch.tensor([4.0, 5.0, 6.0])
        >>> cosine_similarity(tensor1, tensor2)
        0.9746318461970762

        >>> tensor1 = torch.tensor([1.0, 0.0, -1.0])
        >>> tensor2 = torch.tensor([-1.0, 0.0, 1.0])
        >>> cosine_similarity(tensor1, tensor2)
        -1.0

    Note:
        This function assumes the input tensors are 1-dimensional and have the same shape.
    """
    if tensor1.dim() != 1 or tensor2.dim() != 1:
        raise ValueError("Input tensors must be 1-dimensional")
    if tensor1.shape != tensor2.shape:
        raise ValueError("Input tensors must have the same shape")

    dot_product = torch.dot(tensor1, tensor2)
    norm_tensor1 = torch.norm(tensor1)
    norm_tensor2 = torch.norm(tensor2)

    cosine_sim = dot_product / (norm_tensor1 * norm_tensor2)
    return cosine_sim.item()
