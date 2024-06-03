"""This module is for computing CCA and CKA."""

import torch


def compute_cca(features_x: torch.Tensor, features_y: torch.Tensor) -> float:
    """Compute the mean squared CCA correlation (R^2_{CCA}).

    Args:
        features_x (torch.Tensor): A num_examples x num_features matrix of features.
        features_y (torch.Tensor): A num_examples x num_features matrix of features.

    Returns:
        float: The mean squared CCA correlations between X and Y.
    """
    qx, _ = torch.qr(features_x)
    qy, _ = torch.qr(features_y)
    result = torch.norm(qx.t() @ qy) ** 2 / min(features_x.shape[1], features_y.shape[1])
    return result.item() if isinstance(result, torch.Tensor) else float(result)


def compute_cka(
    features_x: torch.Tensor, features_y: torch.Tensor, kernel: str = "linear", threshold: float = 1.0
) -> float:
    """Compute CKA between feature matrices.

    Args:
        features_x (torch.Tensor): A num_examples x num_features matrix of features.
        features_y (torch.Tensor): A num_examples x num_features matrix of features.
        kernel (str): Type of kernel to use ('linear' or 'rbf'). Default is 'linear'.
        threshold (float): Fraction of median Euclidean distance to use as RBF kernel bandwidth
            (used only if kernel is 'rbf').

    Returns:
        float: The value of CKA between X and Y.
    """

    def _gram_linear(x: torch.Tensor) -> torch.Tensor:
        """Compute Gram (kernel) matrix for a linear kernel.

        Args:
            x (torch.Tensor): A num_examples x num_features matrix of features.

        Returns:
            torch.Tensor: A num_examples x num_examples Gram matrix of examples.
        """
        return x @ x.t()

    def _gram_rbf(x: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
        """Compute Gram (kernel) matrix for an RBF kernel.

        Args:
            x (torch.Tensor): A num_examples x num_features matrix of features.
            threshold (float): Fraction of median Euclidean distance to use as RBF kernel bandwidth.

        Returns:
            torch.Tensor: A num_examples x num_examples Gram matrix of examples.
        """
        dot_products = x @ x.t()
        sq_norms = torch.diag(dot_products)
        sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
        sq_median_distance = torch.median(sq_distances)
        return torch.exp(-sq_distances / (2 * threshold**2 * sq_median_distance))

    def _center_gram(gram: torch.Tensor) -> torch.Tensor:
        """Center a symmetric Gram matrix.

        This is equivalent to centering the (possibly infinite-dimensional) features
        induced by the kernel before computing the Gram matrix.

        Args:
            gram (torch.Tensor): A num_examples x num_examples symmetric matrix.

        Returns:
            torch.Tensor: A symmetric matrix with centered columns and rows.

        Raises:
            ValueError: If the input is not a symmetric matrix.
        """
        if not torch.allclose(gram, gram.t()):
            raise ValueError("Input must be a symmetric matrix.")

        n = gram.size(0)
        unit = torch.ones(n, n, device=gram.device)
        eye = torch.eye(n, device=gram.device)
        unit = unit / n
        haitch = eye - unit
        centered_gram = haitch.mm(gram).mm(haitch)
        return centered_gram

    def _cka(gram_x: torch.Tensor, gram_y: torch.Tensor) -> float:
        """Compute CKA.

        Args:
            gram_x (torch.Tensor): A num_examples x num_examples Gram matrix.
            gram_y (torch.Tensor): A num_examples x num_examples Gram matrix.

        Returns:
            float: The value of CKA between X and Y.
        """
        gram_x = _center_gram(gram_x)
        gram_y = _center_gram(gram_y)

        scaled_hsic = torch.sum(gram_x * gram_y)

        normalization_x = torch.norm(gram_x)
        normalization_y = torch.norm(gram_y)
        return scaled_hsic / (normalization_x * normalization_y)

    if kernel == "linear":
        gram_x = _gram_linear(features_x)
        gram_y = _gram_linear(features_y)
    elif kernel == "rbf":
        gram_x = _gram_rbf(features_x, threshold)
        gram_y = _gram_rbf(features_y, threshold)
    else:
        raise ValueError("Unsupported kernel type. Use 'linear' or 'rbf'.")

    result = _cka(gram_x, gram_y)
    return result.item() if isinstance(result, torch.Tensor) else float(result)
