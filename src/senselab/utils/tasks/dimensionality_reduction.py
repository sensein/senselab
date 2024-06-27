"""This module provides the implementation of dimensionality reduction."""

from typing import Literal

import pydra
import torch
import umap
from sklearn.decomposition import PCA as sklearn_PCA
from sklearn.manifold import TSNE as sklearn_TSNE


def compute_dimensionality_reduction(
    data: torch.Tensor, model: Literal["pca", "tsne", "umap"] = "pca", n_components: int = 2, **kwargs: object
) -> torch.Tensor:
    """Reduce the dimensionality of the given data using the specified model.

    This function provides three methods: PCA, t-SNE, or UMAP. The reduced data is returned as a PyTorch tensor.

    Args:
        data (torch.Tensor): Input data tensor of shape (n_samples, n_features).
        model (str, optional): The dimensionality reduction model to use.
            Choices are:
                * "pca" for Principal Component Analysis
                * "tsne" for t-Distributed Stochastic Neighbor Embedding
                * "umap" for Uniform Manifold Approximation and Projection
        n_components (int, optional): Number of dimensions in the output.
            Must be less than or equal to the number of features in the input data.
            Defaults to 2.
        **kwargs: Additional keyword arguments to pass to the chosen model.
            (See below for links to each model's documentation.)

    Returns:
        torch.Tensor: The reduced data tensor of shape (n_samples, n_components).

    Raises:
        ValueError: If an invalid model choice is provided or if n_components is
            greater than the number of features in the input data or less than 0.

    Examples:
        >>> data = torch.randn(100, 10)  # 100 samples, 10 features
        >>> data.shape
        torch.Size([100, 10])

        >>> reduced_data = compute_dimensionality_reduction(data)
        >>> print(reduced_data.shape)
        torch.Size([100, 2])

        >>> reduced_data = compute_dimensionality_reduction(data, model="pca", n_components=2, svd_solver="full")
        >>> print(reduced_data.shape)
        torch.Size([100, 2])

        >>> reduced_data = compute_dimensionality_reduction(data, model="tsne", n_components=3, perplexity=30)
        >>> print(reduced_data.shape)
        torch.Size([100, 3])

        >>> reduced_data = compute_dimensionality_reduction(data, model="umap", n_components=4,
            force_approximation_algorithm=False,
            init='spectral', learning_rate=1.0, n_neighbors=5)
        >>> print(reduced_data.shape)
        torch.Size([100, 4])

    Notes:
        This function uses implementations from scikit-learn for PCA and t-SNE, and the umap-learn
        library for UMAP. For detailed information about each method and its parameters, please
        refer to the following documentation:

        - PCA: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
        - t-SNE: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
        - UMAP: https://umap-learn.readthedocs.io/en/latest/api.html
    """
    if n_components > data.shape[1]:
        raise ValueError("n_components must be less than or equal to the number of features in the input data")

    if n_components < 1:
        raise ValueError("n_components must be greater than 0")

    if data.numel() == 0:
        raise ValueError("Input data must not be empty")

    if model == "pca":
        # Perform PCA
        reducer = sklearn_PCA(n_components=n_components, **kwargs)
        reduced_data = reducer.fit_transform(data)

    elif model == "tsne":
        # Perform t-SNE
        reducer = sklearn_TSNE(n_components=n_components, **kwargs)
        reduced_data = reducer.fit_transform(data)

    elif model == "umap":
        # Perform UMAP
        reducer = umap.UMAP(n_components=n_components, **kwargs)
        reduced_data = reducer.fit_transform(data)

    else:
        raise ValueError(f"Invalid model choice: {model}")

    return torch.from_numpy(reduced_data).float()


compute_dimensionality_reduction_pt = pydra.mark.task(compute_dimensionality_reduction)
