"""This module computes CCA and CKA."""

from enum import Enum

import torch


def compute_cca(features_x: torch.Tensor, features_y: torch.Tensor, standardize: bool = True) -> float:
    """Compute the mean squared CCA correlation (R^2_{CCA}).

    Args:
        features_x (torch.Tensor): A num_examples x num_features matrix of features.
        features_y (torch.Tensor): A num_examples x num_features matrix of features.
        standardize (bool): Whether to perform z-score standardization on input features.
                            Defaults to True.

    Returns:
        float: The mean squared CCA correlations between X and Y.
    """
    if standardize:
        # Standardize inputs to unit variance (improves numerical stability)
        features_x = (features_x.clone() - features_x.mean(0)) / (features_x.std(0) + 1e-8)
        features_y = (features_y.clone() - features_y.mean(0)) / (features_y.std(0) + 1e-8)

    # Reduced QR decomposition for efficiency and stability
    qx, _ = torch.linalg.qr(features_x, mode="reduced")
    qy, _ = torch.linalg.qr(features_y, mode="reduced")
    result = torch.norm(qx.t() @ qy) ** 2 / min(features_x.shape[1], features_y.shape[1])
    return float(result)


class CKAKernelType(Enum):
    """CKA kernel types."""

    LINEAR = "linear"
    RBF = "rbf"


def compute_cka(
    features_x: torch.Tensor,
    features_y: torch.Tensor,
    kernel: CKAKernelType = CKAKernelType.LINEAR,
    threshold: float = 1.0,
    reg: float = 1e-6,
    standardize: bool = True,  # Use 'standardize' to clarify the operation
) -> float:
    """Compute Centered Kernel Alignment (CKA) with enhanced numerical stability.

    Guarantees output ∈ [0,1] through:
      1. Input standardization (prevents scale dominance)
      2. Gram matrix regularization (avoids ill-conditioning)
      3. Epsilon-protected division (prevents NaN/∞)
      4. Value clipping (enforces theoretical bounds)

    Args:
        features_x (torch.Tensor): A num_examples x num_features matrix.
        features_y (torch.Tensor): A num_examples x num_features matrix.
        kernel (CKAKernelType): Kernel type (LINEAR or RBF).
        threshold (float): Fraction of median Euclidean distance to use as RBF kernel bandwidth.
                           Ignored for LINEAR kernel.
        reg (float): Regularization term added to the Gram matrix diagonal for stability.
        standardize (bool): Whether to perform z-score standardization on input features.
                            Defaults to True.

    Returns:
        float: CKA similarity ∈ [0,1] where 1 = identical representations.

    Raises:
        ValueError: If an unsupported kernel type is provided.
    """
    if standardize:
        # Standardize inputs to unit variance (improves numerical stability)
        features_x = (features_x.clone() - features_x.mean(0)) / (features_x.std(0) + 1e-8)
        features_y = (features_y.clone() - features_y.mean(0)) / (features_y.std(0) + 1e-8)

    def _gram_linear(x: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix for a linear kernel."""
        return x @ x.t()

    def _gram_rbf(x: torch.Tensor, threshold: float) -> torch.Tensor:
        """Compute Gram matrix for an RBF kernel with robust bandwidth estimation."""
        dot_products = x @ x.t()
        sq_norms = torch.diag(dot_products)
        sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
        sq_median = torch.median(sq_distances).clamp_min(1e-10)
        return torch.exp(-sq_distances / (2 * (threshold**2) * sq_median))

    def _center_gram(gram: torch.Tensor) -> torch.Tensor:
        """Center the Gram matrix using Tikhonov regularization."""
        n = gram.size(0)
        # Add regularization to the diagonal for numerical stability.
        gram = gram + reg * torch.eye(n, device=gram.device)
        # Enforce symmetry.
        gram = (gram + gram.t()) / 2
        H = torch.eye(n, device=gram.device) - torch.ones(n, n, device=gram.device) / n
        return H @ gram @ H

    def _cka(gram_x: torch.Tensor, gram_y: torch.Tensor) -> torch.Tensor:
        """Core CKA computation with epsilon protection and value clipping."""
        gram_x = _center_gram(gram_x)
        gram_y = _center_gram(gram_y)
        numerator = torch.sum(gram_x * gram_y)
        denominator = torch.norm(gram_x) * torch.norm(gram_y) + 1e-8  # epsilon guard
        return (numerator / denominator).clamp(max=1.0)  # enforce upper bound

    # Select kernel type and compute the corresponding Gram matrices.
    if kernel == CKAKernelType.LINEAR:
        gram_x = _gram_linear(features_x)
        gram_y = _gram_linear(features_y)
    elif kernel == CKAKernelType.RBF:
        gram_x = _gram_rbf(features_x, threshold)
        gram_y = _gram_rbf(features_y, threshold)
        # Enforce symmetry to avoid potential issues during centering.
        gram_x = (gram_x + gram_x.t()) / 2
        gram_y = (gram_y + gram_y.t()) / 2
    else:
        raise ValueError("Unsupported kernel type. Use CKAKernelType.LINEAR or CKAKernelType.RBF.")

    return float(_cka(gram_x, gram_y))
