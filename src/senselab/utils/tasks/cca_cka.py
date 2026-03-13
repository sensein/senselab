"""CCA and CKA similarity utilities.

This module provides two representation-similarity metrics:

- **CCA (Canonical Correlation Analysis)** — returns the mean of squared canonical
  correlations (often denoted `R^2_{CCA}`) between two sets of features.
- **CKA (Centered Kernel Alignment)** — a kernel-based similarity with strong
  invariance properties; we provide linear and RBF variants with numerically
  stable centering and regularization.

All inputs are expected as PyTorch tensors shaped **[n_samples, n_features]**.
"""

from enum import Enum

import torch


def compute_cca(features_x: torch.Tensor, features_y: torch.Tensor, standardize: bool = True) -> float:
    r"""Compute mean squared CCA correlation (`R^2_{CCA}`) between two feature sets.

    This implementation:
      * Optionally **z-score standardizes** features per dimension (recommended),
      * Uses **reduced QR** to obtain orthonormal bases for numerical stability,
      * Computes the average of squared canonical correlations in a compact form:
        `\\|Q_x^\\top Q_y\\|_F^2 / \\min(d_x, d_y)`.

    Args:
        features_x (torch.Tensor):
            Feature matrix with shape ``[n_samples, d_x]``.
        features_y (torch.Tensor):
            Feature matrix with shape ``[n_samples, d_y]``.
        standardize (bool, optional):
            If ``True``, z-score each feature dimension across samples.
            Improves conditioning and makes scales comparable. Defaults to ``True``.

    Returns:
        float: Mean of squared canonical correlations in ``[0, 1]`` (1 means identical
        subspaces modulo linear transforms).

    Notes:
        - Both inputs must have the **same number of samples** (rows).
        - For very high-dimensional features, standardization typically improves stability.

    Example:
        >>> import torch
        >>> torch.manual_seed(0)
        >>> X = torch.randn(100, 64)
        >>> Y = X @ torch.randn(64, 64)  # linear transform
        >>> r2 = compute_cca(X, Y, standardize=True)
        >>> 0.8 <= r2 <= 1.0
        True
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
    """Kernel choices for CKA.

    Attributes:
        LINEAR: Linear kernel (dot-product).
        RBF: Radial Basis Function (Gaussian) kernel.
    """

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
    """Compute Centered Kernel Alignment (CKA) with stability guards.

    The implementation follows the standard CKA definition with additional
    numerical-stability steps:
      1) **Optional z-score** standardization per feature,
      2) **Gram-matrix regularization** (Tikhonov) added to the diagonal,
      3) **Epsilon-guarded** division to avoid NaN/Inf,
      4) **Value clipping** to ensure the result lies in ``[0, 1]``.

    For the **RBF** kernel, the bandwidth is set via a robust median heuristic:
    ``sigma^2 = (threshold^2) * median(||x_i - x_j||^2)``. Set ``threshold`` to
    tune kernel width around the median distance (``1.0`` is a common default).

    Args:
        features_x (torch.Tensor):
            Feature matrix with shape ``[n_samples, d_x]``.
        features_y (torch.Tensor):
            Feature matrix with shape ``[n_samples, d_y]``.
        kernel (CKAKernelType, optional):
            Kernel type; ``CKAKernelType.LINEAR`` or ``CKAKernelType.RBF``.
            Defaults to ``LINEAR``.
        threshold (float, optional):
            Scale for the RBF bandwidth relative to the median pairwise squared
            distance. Ignored for the linear kernel. Defaults to ``1.0``.
        reg (float, optional):
            Tikhonov regularization added to Gram diagonals during centering.
            Defaults to ``1e-6``.
        standardize (bool, optional):
            If ``True``, z-score features across samples before kernelization.
            Defaults to ``True``.

    Returns:
        float: CKA similarity in ``[0, 1]`` (1 indicates identical representations).

    Raises:
        ValueError: If an unsupported kernel type is requested.

    Notes:
        - Inputs must share the same **n_samples**.
        - Complexity is dominated by Gram computations and centering
          (``O(n^2)`` memory and time), where ``n = n_samples``.

    Example (linear CKA, identical up to linear transform):
        >>> import torch
        >>> torch.manual_seed(0)
        >>> X = torch.randn(128, 256)
        >>> A = torch.randn(256, 256)
        >>> Y = X @ A
        >>> sim = compute_cka(X, Y, kernel=CKAKernelType.LINEAR, standardize=True)
        >>> 0.8 <= sim <= 1.0
        True

    Example (RBF CKA with median heuristic):
        >>> import torch
        >>> X = torch.randn(200, 64)
        >>> Y = X + 0.1 * torch.randn_like(X)
        >>> sim = compute_cka(X, Y, kernel=CKAKernelType.RBF, threshold=1.0, standardize=True)
        >>> 0.7 <= sim <= 1.0
        True
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
